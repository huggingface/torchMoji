# -*- coding: utf-8 -*-
""" Model definition functions and weight loading.
"""

from __future__ import print_function, division, unicode_literals

from os.path import exists

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence

from torchmoji.lstm import LSTMHardSigmoid
from torchmoji.attlayer import Attention
from torchmoji.global_variables import NB_TOKENS, NB_EMOJI_CLASSES


def torchmoji_feature_encoding(weight_path, return_attention=False):
    """ Loads the pretrained torchMoji model for extracting features
        from the penultimate feature layer. In this way, it transforms
        the text into its emotional encoding.

    # Arguments:
        weight_path: Path to model weights to be loaded.
        return_attention: If true, output will include weight of each input token
            used for the prediction

    # Returns:
        Pretrained model for encoding text into feature vectors.
    """

    model = TorchMoji(nb_classes=None,
                     nb_tokens=NB_TOKENS,
                     feature_output=True,
                     return_attention=return_attention)
    load_specific_weights(model, weight_path, exclude_names=['output_layer'])
    return model


def torchmoji_emojis(weight_path, return_attention=False):
    """ Loads the pretrained torchMoji model for extracting features
        from the penultimate feature layer. In this way, it transforms
        the text into its emotional encoding.

    # Arguments:
        weight_path: Path to model weights to be loaded.
        return_attention: If true, output will include weight of each input token
            used for the prediction

    # Returns:
        Pretrained model for encoding text into feature vectors.
    """

    model = TorchMoji(nb_classes=NB_EMOJI_CLASSES,
                     nb_tokens=NB_TOKENS,
                     return_attention=return_attention)
    model.load_state_dict(torch.load(weight_path))
    return model


def torchmoji_transfer(nb_classes, weight_path=None, extend_embedding=0,
                      embed_dropout_rate=0.1, final_dropout_rate=0.5):
    """ Loads the pretrained torchMoji model for finetuning/transfer learning.
        Does not load weights for the softmax layer.

        Note that if you are planning to use class average F1 for evaluation,
        nb_classes should be set to 2 instead of the actual number of classes
        in the dataset, since binary classification will be performed on each
        class individually.

        Note that for the 'new' method, weight_path should be left as None.

    # Arguments:
        nb_classes: Number of classes in the dataset.
        weight_path: Path to model weights to be loaded.
        extend_embedding: Number of tokens that have been added to the
            vocabulary on top of NB_TOKENS. If this number is larger than 0,
            the embedding layer's dimensions are adjusted accordingly, with the
            additional weights being set to random values.
        embed_dropout_rate: Dropout rate for the embedding layer.
        final_dropout_rate: Dropout rate for the final Softmax layer.

    # Returns:
        Model with the given parameters.
    """

    model = TorchMoji(nb_classes=nb_classes,
                     nb_tokens=NB_TOKENS + extend_embedding,
                     embed_dropout_rate=embed_dropout_rate,
                     final_dropout_rate=final_dropout_rate,
                     output_logits=True)
    if weight_path is not None:
        load_specific_weights(model, weight_path,
                              exclude_names=['output_layer'],
                              extend_embedding=extend_embedding)
    return model


class TorchMoji(nn.Module):
    def __init__(self, nb_classes, nb_tokens, feature_output=False, output_logits=False,
                 embed_dropout_rate=0, final_dropout_rate=0, return_attention=False):
        """
        torchMoji model.
        IMPORTANT: The model is loaded in evaluation mode by default (self.eval())

        # Arguments:
            nb_classes: Number of classes in the dataset.
            nb_tokens: Number of tokens in the dataset (i.e. vocabulary size).
            feature_output: If True the model returns the penultimate
                            feature vector rather than Softmax probabilities
                            (defaults to False).
            output_logits:  If True the model returns logits rather than probabilities
                            (defaults to False).
            embed_dropout_rate: Dropout rate for the embedding layer.
            final_dropout_rate: Dropout rate for the final Softmax layer.
            return_attention: If True the model also returns attention weights over the sentence
                              (defaults to False).
        """
        super(TorchMoji, self).__init__()

        embedding_dim = 256
        hidden_size = 512
        attention_size = 4 * hidden_size + embedding_dim

        self.feature_output = feature_output
        self.embed_dropout_rate = embed_dropout_rate
        self.final_dropout_rate = final_dropout_rate
        self.return_attention = return_attention
        self.hidden_size = hidden_size
        self.output_logits = output_logits
        self.nb_classes = nb_classes

        self.add_module('embed', nn.Embedding(nb_tokens, embedding_dim))
        # dropout2D: embedding channels are dropped out instead of words
        # many exampels in the datasets contain few words that losing one or more words can alter the emotions completely
        self.add_module('embed_dropout', nn.Dropout2d(embed_dropout_rate))
        self.add_module('lstm_0', LSTMHardSigmoid(embedding_dim, hidden_size, batch_first=True, bidirectional=True))
        self.add_module('lstm_1', LSTMHardSigmoid(hidden_size*2, hidden_size, batch_first=True, bidirectional=True))
        self.add_module('attention_layer', Attention(attention_size=attention_size, return_attention=return_attention))
        if not feature_output:
            self.add_module('final_dropout', nn.Dropout(final_dropout_rate))
            if output_logits:
                self.add_module('output_layer', nn.Sequential(nn.Linear(attention_size, nb_classes if self.nb_classes > 2 else 1)))
            else:
                self.add_module('output_layer', nn.Sequential(nn.Linear(attention_size, nb_classes if self.nb_classes > 2 else 1),
                                                              nn.Softmax() if self.nb_classes > 2 else nn.Sigmoid()))
        self.init_weights()
        # Put model in evaluation mode by default
        self.eval()

    def init_weights(self):
        """
        Here we reproduce Keras default initialization weights for consistency with Keras version
        """
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)
        nn.init.uniform(self.embed.weight.data, a=-0.5, b=0.5)
        for t in ih:
            nn.init.xavier_uniform(t)
        for t in hh:
            nn.init.orthogonal(t)
        for t in b:
            nn.init.constant(t, 0)
        if not self.feature_output:
            nn.init.xavier_uniform(self.output_layer[0].weight.data)

    def forward(self, input_seqs):
        """ Forward pass.

        # Arguments:
            input_seqs: Can be one of Numpy array, Torch.LongTensor, Torch.Variable, Torch.PackedSequence.

        # Return:
            Same format as input format (except for PackedSequence returned as Variable).
        """
        # Check if we have Torch.LongTensor inputs or not Torch.Variable (assume Numpy array in this case), take note to return same format
        return_numpy = False
        return_tensor = False
        if isinstance(input_seqs, (torch.LongTensor, torch.cuda.LongTensor)):
            input_seqs = Variable(input_seqs)
            return_tensor = True
        elif not isinstance(input_seqs, Variable):
            input_seqs = Variable(torch.from_numpy(input_seqs.astype('int64')).long())
            return_numpy = True

        # If we don't have a packed inputs, let's pack it
        reorder_output = False
        if not isinstance(input_seqs, PackedSequence):
            ho = self.lstm_0.weight_hh_l0.data.new(2, input_seqs.size()[0], self.hidden_size).zero_()
            co = self.lstm_0.weight_hh_l0.data.new(2, input_seqs.size()[0], self.hidden_size).zero_()

            # Reorder batch by sequence length
            input_lengths = torch.LongTensor([torch.max(input_seqs[i, :].data.nonzero()) + 1 for i in range(input_seqs.size()[0])])
            input_lengths, perm_idx = input_lengths.sort(0, descending=True)
            input_seqs = input_seqs[perm_idx][:, :input_lengths.max()]

            # Pack sequence and work on data tensor to reduce embeddings/dropout computations
            packed_input = pack_padded_sequence(input_seqs, input_lengths.cpu().numpy(), batch_first=True)
            reorder_output = True
        else:
            ho = self.lstm_0.weight_hh_l0.data.data.new(2, input_seqs.size()[0], self.hidden_size).zero_()
            co = self.lstm_0.weight_hh_l0.data.data.new(2, input_seqs.size()[0], self.hidden_size).zero_()
            input_lengths = input_seqs.batch_sizes
            packed_input = input_seqs

        hidden = (Variable(ho, requires_grad=False), Variable(co, requires_grad=False))

        # Embed with an activation function to bound the values of the embeddings
        x = self.embed(packed_input.data)
        x = nn.Tanh()(x)

        # pyTorch 2D dropout2d operate on axis 1 which is fine for us
        x = self.embed_dropout(x)

        # Update packed sequence data for RNN
        packed_input = PackedSequence(x, packed_input.batch_sizes)

        # skip-connection from embedding to output eases gradient-flow and allows access to lower-level features
        # ordering of the way the merge is done is important for consistency with the pretrained model
        lstm_0_output, _ = self.lstm_0(packed_input, hidden)
        lstm_1_output, _ = self.lstm_1(lstm_0_output, hidden)

        # Update packed sequence data for attention layer
        packed_input = PackedSequence(torch.cat((lstm_1_output.data,
                                                 lstm_0_output.data,
                                                 packed_input.data), dim=1),
                                      packed_input.batch_sizes)

        input_seqs, _ = pad_packed_sequence(packed_input, batch_first=True)

        x, att_weights = self.attention_layer(input_seqs, input_lengths)

        # output class probabilities or penultimate feature vector
        if not self.feature_output:
            x = self.final_dropout(x)
            outputs = self.output_layer(x)
        else:
            outputs = x

        # Reorder output if needed
        if reorder_output:
            reorered = Variable(outputs.data.new(outputs.size()))
            reorered[perm_idx] = outputs
            outputs = reorered

        # Adapt return format if needed
        if return_tensor:
            outputs = outputs.data
        if return_numpy:
            outputs = outputs.data.numpy()

        if self.return_attention:
            return outputs, att_weights
        else:
            return outputs


def load_specific_weights(model, weight_path, exclude_names=[], extend_embedding=0, verbose=True):
    """ Loads model weights from the given file path, excluding any
        given layers.

    # Arguments:
        model: Model whose weights should be loaded.
        weight_path: Path to file containing model weights.
        exclude_names: List of layer names whose weights should not be loaded.
        extend_embedding: Number of new words being added to vocabulary.
        verbose: Verbosity flag.

    # Raises:
        ValueError if the file at weight_path does not exist.
    """
    if not exists(weight_path):
        raise ValueError('ERROR (load_weights): The weights file at {} does '
                         'not exist. Refer to the README for instructions.'
                         .format(weight_path))

    if extend_embedding and 'embed' in exclude_names:
        raise ValueError('ERROR (load_weights): Cannot extend a vocabulary '
                         'without loading the embedding weights.')

    # Copy only weights from the temporary model that are wanted
    # for the specific task (e.g. the Softmax is often ignored)
    weights = torch.load(weight_path)
    for key, weight in weights.items():
        if any(excluded in key for excluded in exclude_names):
            if verbose:
                print('Ignoring weights for {}'.format(key))
            continue

        try:
            model_w = model.state_dict()[key]
        except KeyError:
            raise KeyError("Weights had parameters {},".format(key)
                           + " but could not find this parameters in model.")

        if verbose:
            print('Loading weights for {}'.format(key))

        # extend embedding layer to allow new randomly initialized words
        # if requested. Otherwise, just load the weights for the layer.
        if 'embed' in key and extend_embedding > 0:
            weight = torch.cat((weight, model_w[NB_TOKENS:, :]), dim=0)
            if verbose:
                print('Extended vocabulary for embedding layer ' +
                      'from {} to {} tokens.'.format(
                        NB_TOKENS, NB_TOKENS + extend_embedding))
        try:
            model_w.copy_(weight)
        except:
            print('While copying the weigths named {}, whose dimensions in the model are'
                  ' {} and whose dimensions in the saved file are {}, ...'.format(
                        key, model_w.size(), weight.size()))
            raise
