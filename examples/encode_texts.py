# -*- coding: utf-8 -*-

""" Use torchMoji to encode texts into emotional feature vectors.
"""
from __future__ import print_function, division, unicode_literals
import json

from torchmoji.sentence_tokenizer import SentenceTokenizer
from torchmoji.model_def import torchmoji_feature_encoding
from torchmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH

TEST_SENTENCES = ['I love mom\'s cooking',
                  'I love how you never reply back..',
                  'I love cruising with my homies',
                  'I love messing with yo mind!!',
                  'I love you and now you\'re just gone..',
                  'This is shit',
                  'This is the shit']

maxlen = 30
batch_size = 32

print('Tokenizing using dictionary from {}'.format(VOCAB_PATH))
with open(VOCAB_PATH, 'r') as f:
    vocabulary = json.load(f)
st = SentenceTokenizer(vocabulary, maxlen)
tokenized, _, _ = st.tokenize_sentences(TEST_SENTENCES)

print('Loading model from {}.'.format(PRETRAINED_PATH))
model = torchmoji_feature_encoding(PRETRAINED_PATH)
print(model)

print('Encoding texts..')
encoding = model(tokenized)

print('First 5 dimensions for sentence: {}'.format(TEST_SENTENCES[0]))
print(encoding[0,:5])

# Now you could visualize the encodings to see differences,
# run a logistic regression classifier on top,
# or basically anything you'd like to do.