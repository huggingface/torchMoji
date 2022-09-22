# -*- coding: utf-8 -*-

""" Use torchMoji to predict emojis from a single text input
"""

from __future__ import print_function, division, unicode_literals
from cgitb import text
from http.client import MOVED_PERMANENTLY
import example_helper
import json
import csv
import argparse

import numpy as np
import emoji

from torchmoji.sentence_tokenizer import SentenceTokenizer
from torchmoji.model_def import torchmoji_emojis
from torchmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH

# Emoji map in emoji_overview.png
EMOJIS = ":joy: :unamused: :weary: :sob: :heart_eyes: \
:pensive: :ok_hand: :blush: :heart: :smirk: \
:grin: :notes: :flushed: :100: :sleeping: \
:relieved: :relaxed: :raised_hands: :two_hearts: :expressionless: \
:sweat_smile: :pray: :confused: :kissing_heart: :heartbeat: \
:neutral_face: :information_desk_person: :disappointed: :see_no_evil: :tired_face: \
:v: :sunglasses: :rage: :thumbsup: :cry: \
:sleepy: :yum: :triumph: :hand: :mask: \
:clap: :eyes: :gun: :persevere: :smiling_imp: \
:sweat: :broken_heart: :yellow_heart: :musical_note: :speak_no_evil: \
:wink: :skull: :confounded: :smile: :stuck_out_tongue_winking_eye: \
:angry: :no_good: :muscle: :facepunch: :purple_heart: \
:sparkling_heart: :blue_heart: :grimacing: :sparkles:".split(' ')


def top_elements(array, k):
    ind = np.argpartition(array, -k)[-k:]
    return ind[np.argsort(array[ind])][::-1]

# creating sample function for usage.


def predict_emoji_from_text(text, single_label=True, max_length=30):

    # print("Printing text = ",text)
    if len(text) == 0 or text == 'eos':
        return -1
    # Tokenizing using dictionary
    with open(VOCAB_PATH, 'r') as f:
        vocabulary = json.load(f)

    # st = SentenceTokenizer(vocabulary, args.maxlen)
    st = SentenceTokenizer(vocabulary, max_length)

    # Loading model
    # Toggle to get attention layer outputs
    model = torchmoji_emojis(PRETRAINED_PATH, False)
    # Running predictions
    # tokenized, _, _ = st.tokenize_sentences([args.text])
    tokenized, _, _ = st.tokenize_sentences([text])

    # print("text = ", text)

    model_returns = model(tokenized)

    # If attention layer outputs are not requested.
    if len(model_returns) == 1:
        # Get sentence probability
        # represents probability of each emoji wrt to tokenized text
        prob = model_returns[0]
    else:
        prob = model_returns[0][0]
        attention_layer_output = model_returns[1]
        print("Attention weights = ", attention_layer_output.tolist()
              [0], "Sum = ", sum(attention_layer_output.tolist()[0]))

    # Top emoji id
    emoji_ids = top_elements(prob, 27) # change to 27

    # print(emoji_ids,type(emoji_ids))

    # Removing music emoji's from predictions
    process_list = list(emoji_ids)
    if 48 in process_list and 11 in process_list:
        idx1 = process_list.index(48)
        idx2 = process_list.index(11)
        emoji_ids = np.delete(emoji_ids, [idx1,idx2])

    elif 48 in process_list:
        idx1 = process_list.index(48)
        emoji_ids = np.delete(emoji_ids, idx1)
    
    elif 11 in process_list:
        idx2 = process_list.index(11)
        emoji_ids = np.delete(emoji_ids, idx2)
        

    # print(emoji_ids, type(emoji_ids))
    # map to emojis
    # emojis = map(lambda x: EMOJIS[x], emoji_ids)

    # print("Lyrics : {tex}, Emojis: {pred}".format(
    #     tex=text, pred=emoji.emojize("{}".format(' '.join(emojis)), use_aliases=True)))
    if single_label:    
        return emoji_ids[0]
    
    return emoji_ids[:25]

# if __name__ == "__main__":
#     argparser = argparse.ArgumentParser()
#     argparser.add_argument('--text', type=str, required=True, help="Input text to emojize")
#     argparser.add_argument('--maxlen', type=int, default=30, help="Max length of input text")
#     args = argparser.parse_args()

#     predict_emoji_from_text(args.text)