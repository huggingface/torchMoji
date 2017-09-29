from __future__ import absolute_import, print_function, division, unicode_literals

import test_helper

from nose.plugins.attrib import attr
import json
import numpy as np

from torchmoji.class_avg_finetuning import relabel
from torchmoji.sentence_tokenizer import SentenceTokenizer

from torchmoji.finetuning import (
    calculate_batchsize_maxlen,
    freeze_layers,
    change_trainable,
    finetune,
    load_benchmark
    )
from torchmoji.model_def import (
    torchmoji_transfer,
    torchmoji_feature_encoding,
    torchmoji_emojis
    )
from torchmoji.global_variables import (
    PRETRAINED_PATH,
    NB_TOKENS,
    VOCAB_PATH,
    ROOT_PATH
    )


def test_calculate_batchsize_maxlen():
    """ Batch size and max length are calculated properly.
    """
    texts = ['a b c d',
             'e f g h i']
    batch_size, maxlen = calculate_batchsize_maxlen(texts)

    assert batch_size == 250
    assert maxlen == 10, maxlen


def test_freeze_layers():
    """ Correct layers are frozen.
    """
    model = torchmoji_transfer(5)
    keyword = 'output_layer'

    model = freeze_layers(model, unfrozen_keyword=keyword)

    for name, module in model.named_children():
        trainable = keyword.lower() in name.lower()
        assert all(p.requires_grad == trainable for p in module.parameters())


def test_change_trainable():
    """ change_trainable() changes trainability of layers.
    """
    model = torchmoji_transfer(5)
    change_trainable(model.embed, False)
    assert not any(p.requires_grad for p in model.embed.parameters())
    change_trainable(model.embed, True)
    assert all(p.requires_grad for p in model.embed.parameters())


def test_torchmoji_transfer_extend_embedding():
    """ Defining torchmoji with extension.
    """
    extend_with = 50
    model = torchmoji_transfer(5, weight_path=PRETRAINED_PATH,
                              extend_embedding=extend_with)
    embedding_layer = model.embed
    assert embedding_layer.weight.size()[0] == NB_TOKENS + extend_with


def test_torchmoji_return_attention():
    seq_tensor = np.array([[1]])
    # test the output of the normal model
    model = torchmoji_emojis(weight_path=PRETRAINED_PATH)
    # check correct number of outputs
    assert len(model(seq_tensor)) == 1
    # repeat above described tests when returning attention weights
    model = torchmoji_emojis(weight_path=PRETRAINED_PATH, return_attention=True)
    assert len(model(seq_tensor)) == 2


def test_relabel():
    """ relabel() works with multi-class labels.
    """
    nb_classes = 3
    inputs = np.array([
        [True, False, False],
        [False, True, False],
        [True, False, True],
    ])
    expected_0 = np.array([True, False, True])
    expected_1 = np.array([False, True, False])
    expected_2 = np.array([False, False, True])

    assert np.array_equal(relabel(inputs, 0, nb_classes), expected_0)
    assert np.array_equal(relabel(inputs, 1, nb_classes), expected_1)
    assert np.array_equal(relabel(inputs, 2, nb_classes), expected_2)


def test_relabel_binary():
    """ relabel() works with binary classification (no changes to labels)
    """
    nb_classes = 2
    inputs = np.array([True, False, False])

    assert np.array_equal(relabel(inputs, 0, nb_classes), inputs)


@attr('slow')
def test_finetune_full():
    """ finetuning using 'full'.
    """
    DATASET_PATH = ROOT_PATH+'/data/SS-Youtube/raw.pickle'
    nb_classes = 2
    # Keras and pyTorch implementation of the Adam optimizer are slightly different and change a bit the results
    # We reduce the min accuracy needed here to pass the test
    # See e.g. https://discuss.pytorch.org/t/suboptimal-convergence-when-compared-with-tensorflow-model/5099/11
    min_acc = 0.68

    with open(VOCAB_PATH, 'r') as f:
        vocab = json.load(f)

    data = load_benchmark(DATASET_PATH, vocab, extend_with=10000)
    print('Loading pyTorch model from {}.'.format(PRETRAINED_PATH))
    model = torchmoji_transfer(nb_classes, PRETRAINED_PATH, extend_embedding=data['added'])
    print(model)
    model, acc = finetune(model, data['texts'], data['labels'], nb_classes,
                          data['batch_size'], method='full', nb_epochs=1)

    print("Finetune full SS-Youtube 1 epoch acc: {}".format(acc))
    assert acc >= min_acc


@attr('slow')
def test_finetune_last():
    """ finetuning using 'last'.
    """
    dataset_path = ROOT_PATH + '/data/SS-Youtube/raw.pickle'
    nb_classes = 2
    min_acc = 0.68

    with open(VOCAB_PATH, 'r') as f:
        vocab = json.load(f)

    data = load_benchmark(dataset_path, vocab)
    print('Loading model from {}.'.format(PRETRAINED_PATH))
    model = torchmoji_transfer(nb_classes, PRETRAINED_PATH)
    print(model)
    model, acc = finetune(model, data['texts'], data['labels'], nb_classes,
                          data['batch_size'], method='last', nb_epochs=1)

    print("Finetune last SS-Youtube 1 epoch acc: {}".format(acc))

    assert acc >= min_acc


def test_score_emoji():
    """ Emoji predictions make sense.
    """
    test_sentences = [
        'I love mom\'s cooking',
        'I love how you never reply back..',
        'I love cruising with my homies',
        'I love messing with yo mind!!',
        'I love you and now you\'re just gone..',
        'This is shit',
        'This is the shit'
    ]

    expected = [
        np.array([36,  4,  8, 16, 47]),
        np.array([1, 19, 55, 25, 46]),
        np.array([31,  6, 30, 15, 13]),
        np.array([54, 44,  9, 50, 49]),
        np.array([46,  5, 27, 35, 34]),
        np.array([55, 32, 27,  1, 37]),
        np.array([48, 11,  6, 31,  9])
    ]

    def top_elements(array, k):
        ind = np.argpartition(array, -k)[-k:]
        return ind[np.argsort(array[ind])][::-1]

    # Initialize by loading dictionary and tokenize texts
    with open(VOCAB_PATH, 'r') as f:
        vocabulary = json.load(f)

    st = SentenceTokenizer(vocabulary, 30)
    tokens, _, _ = st.tokenize_sentences(test_sentences)

    # Load model and run
    model = torchmoji_emojis(weight_path=PRETRAINED_PATH)
    prob = model(tokens)

    # Find top emojis for each sentence
    for i, t_prob in enumerate(list(prob)):
        assert np.array_equal(top_elements(t_prob, 5), expected[i])


def test_encode_texts():
    """ Text encoding is stable.
    """

    TEST_SENTENCES = ['I love mom\'s cooking',
                      'I love how you never reply back..',
                      'I love cruising with my homies',
                      'I love messing with yo mind!!',
                      'I love you and now you\'re just gone..',
                      'This is shit',
                      'This is the shit']


    maxlen = 30
    batch_size = 32

    with open(VOCAB_PATH, 'r') as f:
        vocabulary = json.load(f)

    st = SentenceTokenizer(vocabulary, maxlen)

    print('Loading model from {}.'.format(PRETRAINED_PATH))
    model = torchmoji_feature_encoding(PRETRAINED_PATH)
    print(model)
    tokenized, _, _ = st.tokenize_sentences(TEST_SENTENCES)
    encoding = model(tokenized)

    avg_across_sentences = np.around(np.mean(encoding, axis=0)[:5], 3)
    assert np.allclose(avg_across_sentences, np.array([-0.023, 0.021, -0.037, -0.001, -0.005]))

test_encode_texts()