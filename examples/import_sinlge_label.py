import pandas as pd
import numpy as np
single_label_df = pd.read_csv("../datasets/SingleLabel.csv")

for emotion_labels in single_label_df.label.unique():
    single_label_df[emotion_labels] = (single_label_df.label == emotion_labels).astype(int)


lyrics_set = single_label_df['lyrics']
np.savetxt("/../datasets/ly.txt", lyrics_set)
# print(lyrics_set)