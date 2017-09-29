"""Finetuning example.

Trains the torchMoji model on the SS-Youtube dataset, using the 'last'
finetuning method and the accuracy metric.

The 'last' method does the following:
0) Load all weights except for the softmax layer. Do not add tokens to the
   vocabulary and do not extend the embedding layer.
1) Freeze all layers except for the softmax layer.
2) Train.
"""

from __future__ import print_function
import example_helper
import json
from torchmoji.model_def import torchmoji_transfer
from torchmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH, ROOT_PATH
from torchmoji.finetuning import (
     load_benchmark,
     finetune)

DATASET_PATH = '{}/data/SS-Youtube/raw.pickle'.format(ROOT_PATH)
nb_classes = 2

with open(VOCAB_PATH, 'r') as f:
    vocab = json.load(f)

# Load dataset.
data = load_benchmark(DATASET_PATH, vocab)

# Set up model and finetune
model = torchmoji_transfer(nb_classes, PRETRAINED_PATH)
print(model)
model, acc = finetune(model, data['texts'], data['labels'], nb_classes, data['batch_size'], method='last')
print('Acc: {}'.format(acc))
