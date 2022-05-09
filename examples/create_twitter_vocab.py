""" Creates a vocabulary from a tsv file.
"""

import codecs
import example_helper
from torchmoji.create_vocab import VocabBuilder
from torchmoji.word_generator import TweetWordGenerator

with codecs.open('../datasets/SingleLabel.csv', 'rU', 'utf-8') as stream:
    wg = TweetWordGenerator(stream)
    print(wg.english_words)
    vb = VocabBuilder(wg)
    vb.count_all_words()
    # vb.save_vocab()
