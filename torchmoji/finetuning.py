# -*- coding: utf-8 -*-
""" Finetuning functions for doing transfer learning to new datasets.
"""
from __future__ import print_function

import uuid
from time import sleep
from io import open

import math
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import BatchSampler, SequentialSampler
from torch.nn.utils import clip_grad_norm

from sklearn.metrics import f1_score

from torchmoji.global_variables import (FINETUNING_METHODS,
                                               FINETUNING_METRICS,
                                               WEIGHTS_DIR)
from torchmoji.tokenizer import tokenize
from torchmoji.sentence_tokenizer import SentenceTokenizer

try:
    unicode
    IS_PYTHON2 = True
except NameError:
    unicode = str
    IS_PYTHON2 = False


def load_benchmark(path, vocab, extend_with=0):
    """ Loads the given benchmark dataset.

        Tokenizes the texts using the provided vocabulary, extending it with
        words from the training dataset if extend_with > 0. Splits them into
        three lists: training, validation and testing (in that order).

        Also calculates the maximum length of the texts and the
        suggested batch_size.

    # Arguments:
        path: Path to the dataset to be loaded.
        vocab: Vocabulary to be used for tokenizing texts.
        extend_with: If > 0, the vocabulary will be extended with up to
            extend_with tokens from the training set before tokenizing.

    # Returns:
        A dictionary with the following fields:
            texts: List of three lists, containing tokenized inputs for
                training, validation and testing (in that order).
            labels: List of three lists, containing labels for training,
                validation and testing (in that order).
            added: Number of tokens added to the vocabulary.
            batch_size: Batch size.
            maxlen: Maximum length of an input.
    """
    # Pre-processing dataset
    with open(path, 'rb') as dataset:
        if IS_PYTHON2:
            data = pickle.load(dataset)
        else:
            data = pickle.load(dataset, fix_imports=True)

    # Decode data
    try:
        texts = [unicode(x) for x in data['texts']]
    except UnicodeDecodeError:
        texts = [x.decode('utf-8') for x in data['texts']]

    # Extract labels
    labels = [x['label'] for x in data['info']]

    batch_size, maxlen = calculate_batchsize_maxlen(texts)

    st = SentenceTokenizer(vocab, maxlen)

    # Split up dataset. Extend the existing vocabulary with up to extend_with
    # tokens from the training dataset.
    texts, labels, added = st.split_train_val_test(texts,
                                                   labels,
                                                   [data['train_ind'],
                                                    data['val_ind'],
                                                    data['test_ind']],
                                                   extend_with=extend_with)
    return {'texts': texts,
            'labels': labels,
            'added': added,
            'batch_size': batch_size,
            'maxlen': maxlen}


def calculate_batchsize_maxlen(texts):
    """ Calculates the maximum length in the provided texts and a suitable
        batch size. Rounds up maxlen to the nearest multiple of ten.

    # Arguments:
        texts: List of inputs.

    # Returns:
        Batch size,
        max length
    """
    def roundup(x):
        return int(math.ceil(x / 10.0)) * 10

    # Calculate max length of sequences considered
    # Adjust batch_size accordingly to prevent GPU overflow
    lengths = [len(tokenize(t)) for t in texts]
    maxlen = roundup(np.percentile(lengths, 80.0))
    batch_size = 250 if maxlen <= 100 else 50
    return batch_size, maxlen



def freeze_layers(model, unfrozen_types=[], unfrozen_keyword=None):
    """ Freezes all layers in the given model, except for ones that are
        explicitly specified to not be frozen.

    # Arguments:
        model: Model whose layers should be modified.
        unfrozen_types: List of layer types which shouldn't be frozen.
        unfrozen_keyword: Name keywords of layers that shouldn't be frozen.

    # Returns:
        Model with the selected layers frozen.
    """
    # Get trainable modules
    trainable_modules = [(n, m) for n, m in model.named_children() if len([id(p) for p in m.parameters()]) !=  0]
    for name, module in trainable_modules:
        trainable = (any(typ in str(module) for typ in unfrozen_types) or
                     (unfrozen_keyword is not None and unfrozen_keyword.lower() in name.lower()))
        change_trainable(module, trainable, verbose=False)
    return model


def change_trainable(module, trainable, verbose=False):
    """ Helper method that freezes or unfreezes a given layer.

    # Arguments:
        module: Module to be modified.
        trainable: Whether the layer should be frozen or unfrozen.
        verbose: Verbosity flag.
    """
    
    if verbose: print('Changing MODULE', module, 'to trainable =', trainable)
    for name, param in module.named_parameters():
        if verbose: print('Setting weight', name, 'to trainable =', trainable)
        param.requires_grad = trainable

    if verbose:
        action = 'Unfroze' if trainable else 'Froze'
        if verbose: print("{} {}".format(action, module))


def find_f1_threshold(model, val_gen, test_gen, average='binary'):
    """ Choose a threshold for F1 based on the validation dataset
        (see https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4442797/
        for details on why to find another threshold than simply 0.5)

    # Arguments:
        model: pyTorch model
        val_gen: Validation set dataloader.
        test_gen: Testing set dataloader.

    # Returns:
        F1 score for the given data and
        the corresponding F1 threshold
    """
    thresholds = np.arange(0.01, 0.5, step=0.01)
    f1_scores = []

    model.eval()
    val_out = [(y, model(X)) for X, y in val_gen]
    y_val, y_pred_val = (list(t) for t in zip(*val_out))

    test_out = [(y, model(X)) for X, y in test_gen]
    y_test, y_pred_test = (list(t) for t in zip(*val_out))

    for t in thresholds:
        y_pred_val_ind = (y_pred_val > t)
        f1_val = f1_score(y_val, y_pred_val_ind, average=average)
        f1_scores.append(f1_val)

    best_t = thresholds[np.argmax(f1_scores)]
    y_pred_ind = (y_pred_test > best_t)
    f1_test = f1_score(y_test, y_pred_ind, average=average)
    return f1_test, best_t


def finetune(model, texts, labels, nb_classes, batch_size, method,
             metric='acc', epoch_size=5000, nb_epochs=1000, embed_l2=1E-6,
             verbose=1):
    """ Compiles and finetunes the given pytorch model.

    # Arguments:
        model: Model to be finetuned
        texts: List of three lists, containing tokenized inputs for training,
            validation and testing (in that order).
        labels: List of three lists, containing labels for training,
            validation and testing (in that order).
        nb_classes: Number of classes in the dataset.
        batch_size: Batch size.
        method: Finetuning method to be used. For available methods, see
            FINETUNING_METHODS in global_variables.py.
        metric: Evaluation metric to be used. For available metrics, see
            FINETUNING_METRICS in global_variables.py.
        epoch_size: Number of samples in an epoch.
        nb_epochs: Number of epochs. Doesn't matter much as early stopping is used.
        embed_l2: L2 regularization for the embedding layer.
        verbose: Verbosity flag.

    # Returns:
        Model after finetuning,
        score after finetuning using the provided metric.
    """

    if method not in FINETUNING_METHODS:
        raise ValueError('ERROR (finetune): Invalid method parameter. '
                         'Available options: {}'.format(FINETUNING_METHODS))
    if metric not in FINETUNING_METRICS:
        raise ValueError('ERROR (finetune): Invalid metric parameter. '
                         'Available options: {}'.format(FINETUNING_METRICS))

    train_gen = get_data_loader(texts[0], labels[0], batch_size,
                                extended_batch_sampler=True, epoch_size=epoch_size)
    val_gen = get_data_loader(texts[1], labels[1], batch_size,
                              extended_batch_sampler=False)
    test_gen = get_data_loader(texts[2], labels[2], batch_size,
                              extended_batch_sampler=False)

    checkpoint_path = '{}/torchmoji-checkpoint-{}.bin' \
                      .format(WEIGHTS_DIR, str(uuid.uuid4()))

    if method in ['last', 'new']:
        lr = 0.001
    elif method in ['full', 'chain-thaw']:
        lr = 0.0001

    loss_op = nn.BCEWithLogitsLoss() if nb_classes <= 2 \
         else nn.CrossEntropyLoss()

    # Freeze layers if using last
    if method == 'last':
        model = freeze_layers(model, unfrozen_keyword='output_layer')

    # Define optimizer, for chain-thaw we define it later (after freezing)
    if method == 'last':
        adam = optim.Adam((p for p in model.parameters() if p.requires_grad), lr=lr)
    elif method in ['full', 'new']:
        # Add L2 regulation on embeddings only
        embed_params_id = [id(p) for p in model.embed.parameters()]
        output_layer_params_id = [id(p) for p in model.output_layer.parameters()]
        base_params = [p for p in model.parameters()
                       if id(p) not in embed_params_id and id(p) not in output_layer_params_id and p.requires_grad]
        embed_params = [p for p in model.parameters() if id(p) in embed_params_id and p.requires_grad]
        output_layer_params = [p for p in model.parameters() if id(p) in output_layer_params_id and p.requires_grad]
        adam = optim.Adam([
            {'params': base_params},
            {'params': embed_params, 'weight_decay': embed_l2},
            {'params': output_layer_params, 'lr': 0.001},
            ], lr=lr)

    # Training
    if verbose:
        print('Method:  {}'.format(method))
        print('Metric:  {}'.format(metric))
        print('Classes: {}'.format(nb_classes))

    if method == 'chain-thaw':
        result = chain_thaw(model, train_gen, val_gen, test_gen, nb_epochs, checkpoint_path, loss_op, embed_l2=embed_l2,
                            evaluate=metric, verbose=verbose)
    else:
        result = tune_trainable(model, loss_op, adam, train_gen, val_gen, test_gen, nb_epochs, checkpoint_path,
                                evaluate=metric, verbose=verbose)
    return model, result


def tune_trainable(model, loss_op, optim_op, train_gen, val_gen, test_gen,
                   nb_epochs, checkpoint_path, patience=5, evaluate='acc',
                   verbose=2):
    """ Finetunes the given model using the accuracy measure.

    # Arguments:
        model: Model to be finetuned.
        nb_classes: Number of classes in the given dataset.
        train: Training data, given as a tuple of (inputs, outputs)
        val: Validation data, given as a tuple of (inputs, outputs)
        test: Testing data, given as a tuple of (inputs, outputs)
        epoch_size: Number of samples in an epoch.
        nb_epochs: Number of epochs.
        batch_size: Batch size.
        checkpoint_weight_path: Filepath where weights will be checkpointed to
            during training. This file will be rewritten by the function.
        patience: Patience for callback methods.
        evaluate: Evaluation method to use. Can be 'acc' or 'weighted_f1'.
        verbose: Verbosity flag.

    # Returns:
        Accuracy of the trained model, ONLY if 'evaluate' is set.
    """
    if verbose:
        print("Trainable weights: {}".format([n for n, p in model.named_parameters() if p.requires_grad]))
        print("Training...")
        if evaluate == 'acc':
            print("Evaluation on test set prior training:", evaluate_using_acc(model, test_gen))
        elif evaluate == 'weighted_f1':
            print("Evaluation on test set prior training:", evaluate_using_weighted_f1(model, test_gen, val_gen))

    fit_model(model, loss_op, optim_op, train_gen, val_gen, nb_epochs, checkpoint_path, patience)

    # Reload the best weights found to avoid overfitting
    # Wait a bit to allow proper closing of weights file
    sleep(1)
    model.load_state_dict(torch.load(checkpoint_path))
    if verbose >= 2:
        print("Loaded weights from {}".format(checkpoint_path))

    if evaluate == 'acc':
        return evaluate_using_acc(model, test_gen)
    elif evaluate == 'weighted_f1':
        return evaluate_using_weighted_f1(model, test_gen, val_gen)


def evaluate_using_weighted_f1(model, test_gen, val_gen):
    """ Evaluation function using macro weighted F1 score.

    # Arguments:
        model: Model to be evaluated.
        X_test: Inputs of the testing set.
        y_test: Outputs of the testing set.
        X_val: Inputs of the validation set.
        y_val: Outputs of the validation set.
        batch_size: Batch size.

    # Returns:
        Weighted F1 score of the given model.
    """
    # Evaluate on test and val data
    f1_test, _ = find_f1_threshold(model, test_gen, val_gen, average='weighted_f1')
    return f1_test


def evaluate_using_acc(model, test_gen):
    """ Evaluation function using accuracy.

    # Arguments:
        model: Model to be evaluated.
        test_gen: Testing data iterator (DataLoader)

    # Returns:
        Accuracy of the given model.
    """

    # Validate on test_data
    model.eval()
    accs = []
    for i, data in enumerate(test_gen):
        x, y = data
        outs = model(x)
        if model.nb_classes > 2:
            pred = torch.max(outs, 1)[1]
            acc = accuracy_score(y.squeeze().numpy(), pred.squeeze().numpy())
        else:
            pred = (outs >= 0).long()
            acc = (pred == y).double().sum() / len(pred)
        accs.append(acc)
    return np.mean(accs)


def chain_thaw(model, train_gen, val_gen, test_gen, nb_epochs, checkpoint_path, loss_op,
               patience=5, initial_lr=0.001, next_lr=0.0001, embed_l2=1E-6, evaluate='acc', verbose=1):
    """ Finetunes given model using chain-thaw and evaluates using accuracy.

    # Arguments:
        model: Model to be finetuned.
        train: Training data, given as a tuple of (inputs, outputs)
        val: Validation data, given as a tuple of (inputs, outputs)
        test: Testing data, given as a tuple of (inputs, outputs)
        batch_size: Batch size.
        loss: Loss function to be used during training.
        epoch_size: Number of samples in an epoch.
        nb_epochs: Number of epochs.
        checkpoint_weight_path: Filepath where weights will be checkpointed to
            during training. This file will be rewritten by the function.
        initial_lr: Initial learning rate. Will only be used for the first
            training step (i.e. the output_layer layer)
        next_lr: Learning rate for every subsequent step.
        seed: Random number generator seed.
        verbose: Verbosity flag.
        evaluate: Evaluation method to use. Can be 'acc' or 'weighted_f1'.

    # Returns:
        Accuracy of the finetuned model.
    """
    if verbose:
        print('Training..')

    # Train using chain-thaw
    train_by_chain_thaw(model, train_gen, val_gen, loss_op, patience, nb_epochs, checkpoint_path,
                        initial_lr, next_lr, embed_l2, verbose)

    if evaluate == 'acc':
        return evaluate_using_acc(model, test_gen)
    elif evaluate == 'weighted_f1':
        return evaluate_using_weighted_f1(model, test_gen, val_gen)


def train_by_chain_thaw(model, train_gen, val_gen, loss_op, patience, nb_epochs, checkpoint_path,
                        initial_lr=0.001, next_lr=0.0001, embed_l2=1E-6, verbose=1):
    """ Finetunes model using the chain-thaw method.

    This is done as follows:
    1) Freeze every layer except the last (output_layer) layer and train it.
    2) Freeze every layer except the first layer and train it.
    3) Freeze every layer except the second etc., until the second last layer.
    4) Unfreeze all layers and train entire model.

    # Arguments:
        model: Model to be trained.
        train_gen: Training sample generator.
        val_data: Validation data.
        loss: Loss function to be used.
        finetuning_args: Training early stopping and checkpoint saving parameters
        epoch_size: Number of samples in an epoch.
        nb_epochs: Number of epochs.
        checkpoint_weight_path: Where weight checkpoints should be saved.
        batch_size: Batch size.
        initial_lr: Initial learning rate. Will only be used for the first
            training step (i.e. the output_layer layer)
        next_lr: Learning rate for every subsequent step.
        verbose: Verbosity flag.
    """
    # Get trainable layers
    layers = [m for m in model.children() if len([id(p) for p in m.parameters()]) !=  0]

    # Bring last layer to front
    layers.insert(0, layers.pop(len(layers) - 1))

    # Add None to the end to signify finetuning all layers
    layers.append(None)

    lr = None
    # Finetune each layer one by one and finetune all of them at once
    # at the end
    for layer in layers:
        if lr is None:
            lr = initial_lr
        elif lr == initial_lr:
            lr = next_lr

        # Freeze all except current layer
        for _layer in layers:
            if _layer is not None:
                trainable = _layer == layer or layer is None
                change_trainable(_layer, trainable=trainable, verbose=False)

        # Verify we froze the right layers
        for _layer in model.children():
            assert all(p.requires_grad == (_layer == layer) for p in _layer.parameters()) or layer is None

        if verbose:
            if layer is None:
                print('Finetuning all layers')
            else:
                print('Finetuning {}'.format(layer))

        special_params = [id(p) for p in model.embed.parameters()]
        base_params = [p for p in model.parameters() if id(p) not in special_params and p.requires_grad]
        embed_parameters = [p for p in model.parameters() if id(p) in special_params and p.requires_grad]
        adam = optim.Adam([
            {'params': base_params},
            {'params': embed_parameters, 'weight_decay': embed_l2},
            ], lr=lr)

        fit_model(model, loss_op, adam, train_gen, val_gen, nb_epochs,
                  checkpoint_path, patience)

        # Reload the best weights found to avoid overfitting
        # Wait a bit to allow proper closing of weights file
        sleep(1)
        model.load_state_dict(torch.load(checkpoint_path))
        if verbose >= 2:
            print("Loaded weights from {}".format(checkpoint_path))


def calc_loss(loss_op, pred, yv):
    if type(loss_op) is nn.CrossEntropyLoss:
        return loss_op(pred.squeeze(), yv.squeeze())
    else:
        return loss_op(pred.squeeze(), yv.squeeze().float())


def fit_model(model, loss_op, optim_op, train_gen, val_gen, epochs,
              checkpoint_path, patience):
    """ Analog to Keras fit_generator function.

    # Arguments:
        model: Model to be finetuned.
        loss_op: loss operation (BCEWithLogitsLoss or CrossEntropy for e.g.)
        optim_op: optimization operation (Adam e.g.)
        train_gen: Training data iterator (DataLoader)
        val_gen: Validation data iterator (DataLoader)
        epochs: Number of epochs.
        checkpoint_path: Filepath where weights will be checkpointed to
            during training. This file will be rewritten by the function.
        patience: Patience for callback methods.
        verbose: Verbosity flag.

    # Returns:
        Accuracy of the trained model, ONLY if 'evaluate' is set.
    """
    # Save original checkpoint
    torch.save(model.state_dict(), checkpoint_path)

    model.eval()
    best_loss = np.mean([calc_loss(loss_op, model(Variable(xv)), Variable(yv)).data.cpu().numpy()[0] for xv, yv in val_gen])
    print("original val loss", best_loss)

    epoch_without_impr = 0
    for epoch in range(epochs):
        for i, data in enumerate(train_gen):
            X_train, y_train = data
            X_train = Variable(X_train, requires_grad=False)
            y_train = Variable(y_train, requires_grad=False)
            model.train()
            optim_op.zero_grad()
            output = model(X_train)
            loss = calc_loss(loss_op, output, y_train)
            loss.backward()
            clip_grad_norm(model.parameters(), 1)
            optim_op.step()

            acc = evaluate_using_acc(model, [(X_train.data, y_train.data)])
            print("== Epoch", epoch, "step", i, "train loss", loss.data.cpu().numpy()[0], "train acc", acc)

        model.eval()
        acc = evaluate_using_acc(model, val_gen)
        print("val acc", acc)

        val_loss = np.mean([calc_loss(loss_op, model(Variable(xv)), Variable(yv)).data.cpu().numpy()[0] for xv, yv in val_gen])
        print("val loss", val_loss)
        if best_loss is not None and val_loss >= best_loss:
            epoch_without_impr += 1
            print('No improvement over previous best loss: ', best_loss)

        # Save checkpoint
        if best_loss is None or val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print('Saving model at', checkpoint_path)

        # Early stopping
        if epoch_without_impr >= patience:
            break

def get_data_loader(X_in, y_in, batch_size, extended_batch_sampler=True, epoch_size=25000, upsample=False, seed=42):
    """ Returns a dataloader that enables larger epochs on small datasets and
        has upsampling functionality.

    # Arguments:
        X_in: Inputs of the given dataset.
        y_in: Outputs of the given dataset.
        batch_size: Batch size.
        epoch_size: Number of samples in an epoch.
        upsample: Whether upsampling should be done. This flag should only be
            set on binary class problems.

    # Returns:
        DataLoader.
    """
    dataset = DeepMojiDataset(X_in, y_in)

    if extended_batch_sampler:
        batch_sampler = DeepMojiBatchSampler(y_in, batch_size, epoch_size=epoch_size, upsample=upsample, seed=seed)
    else:
        batch_sampler = BatchSampler(SequentialSampler(y_in), batch_size, drop_last=False)

    return DataLoader(dataset, batch_sampler=batch_sampler, num_workers=0)

class DeepMojiDataset(Dataset):
    """ A simple Dataset class.

    # Arguments:
        X_in: Inputs of the given dataset.
        y_in: Outputs of the given dataset.
    
    # __getitem__ output:
        (torch.LongTensor, torch.LongTensor)
    """
    def __init__(self, X_in, y_in):
        # Check if we have Torch.LongTensor inputs (assume Numpy array otherwise)
        if not isinstance(X_in, torch.LongTensor):
            X_in = torch.from_numpy(X_in.astype('int64')).long()
        if not isinstance(y_in, torch.LongTensor):
            y_in = torch.from_numpy(y_in.astype('int64')).long()

        self.X_in = torch.split(X_in, 1, dim=0)
        self.y_in = torch.split(y_in, 1, dim=0)

    def __len__(self):
        return len(self.X_in)

    def __getitem__(self, idx):
        return self.X_in[idx].squeeze(), self.y_in[idx].squeeze()

class DeepMojiBatchSampler(object):
    """A Batch sampler that enables larger epochs on small datasets and
        has upsampling functionality.

    # Arguments:
        y_in: Labels of the dataset.
        batch_size: Batch size.
        epoch_size: Number of samples in an epoch.
        upsample: Whether upsampling should be done. This flag should only be
            set on binary class problems.
        seed: Random number generator seed.

    # __iter__ output:
        iterator of lists (batches) of indices in the dataset
    """

    def __init__(self, y_in, batch_size, epoch_size, upsample, seed):
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.upsample = upsample

        np.random.seed(seed)

        if upsample:
            # Should only be used on binary class problems
            assert len(y_in.shape) == 1
            neg = np.where(y_in.numpy() == 0)[0]
            pos = np.where(y_in.numpy() == 1)[0]
            assert epoch_size % 2 == 0
            samples_pr_class = int(epoch_size / 2)
        else:
            ind = range(len(y_in))

        if not upsample:
            # Randomly sample observations in a balanced way
            self.sample_ind = np.random.choice(ind, epoch_size, replace=True)
        else:
            # Randomly sample observations in a balanced way
            sample_neg = np.random.choice(neg, samples_pr_class, replace=True)
            sample_pos = np.random.choice(pos, samples_pr_class, replace=True)
            concat_ind = np.concatenate((sample_neg, sample_pos), axis=0)

            # Shuffle to avoid labels being in specific order
            # (all negative then positive)
            p = np.random.permutation(len(concat_ind))
            self.sample_ind = concat_ind[p]

            label_dist = np.mean(y_in.numpy()[self.sample_ind])
            assert(label_dist > 0.45)
            assert(label_dist < 0.55)

    def __iter__(self):
        # Hand-off data using batch_size
        for i in range(int(self.epoch_size/self.batch_size)):
            start = i * self.batch_size
            end = min(start + self.batch_size, self.epoch_size)
            yield self.sample_ind[start:end]

    def __len__(self):
        # Take care of the last (maybe incomplete) batch
        return (self.epoch_size + self.batch_size - 1) // self.batch_size
