from sunau import _sunau_params
from pandas import read_csv
import librosa as lib
import numpy as np

from matplotlib import pyplot as plt
import librosa.display as display

dataset = read_csv('audio_samples.csv')

sample_label = []
for idx,rows in dataset.iterrows():
    target_label = rows['emoji']

    audio_buffer, sample_rate = lib.load(rows['audio_piece'], mono=True)
    D = lib.amplitude_to_db(np.abs(lib.stft(audio_buffer)), ref=np.max)
    print(D.shape)
    sample_label.append((D,target_label))