import DALI as dali_code
from text_emojize import predict_emoji_from_text
from pandas import DataFrame
import matplotlib.pyplot as plt
"""
Load the dataset
take in paragraphs of songs
create corresponding start time and end time
create corresponding emoji
save it as a csv file
"""

dali_data_path = '/Users/amanshukla/Downloads/DALI_v1.0'

# Reading DALI data [Only a subset for now]
dali_data = dali_code.get_the_DALI_dataset(dali_data_path, skip=[], keep=["fffa380a6faa410c8a3283cef044276d",
                                                                          "fbda8aaad9f94267ad861d6af96b05e8",
                                                                          "ff5fdfedf2a442bfb1016e3c2062bca1",
                                                                          "fecaf1f8ac46423c95966194eebb7d7d",
                                                                          "fd6be3601b4d4ca4accdbadf2ee31440",
                                                                          "f762aef2e3b848c4a19af7b873c8e4c3",
                                                                          "f2d0fdb5a5c6474abeceea34e38d980a",
                                                                          "f2094b1c05814a47b5145c1d9edd81c5",
                                                                          "ee222691643f40abb4f7f49d6e990766",
                                                                          "ea3414d65ba84ec1b9e8aa74ec586434"])

dataframe = DataFrame(columns=['dali_id','start_time','end_time','para_lyrics','emoji_set'])
df_counter = 0
GLOBAL_PARA_LENGTH = []
for key in dali_data.keys():

    entry = dali_data[key]

    # Filter english songs as embedding only available for english language
    song_language = entry.info["metadata"]['language']
    if song_language == 'english':

    # can be changed from lines -> words / paragraphs
        para_lyric_set = entry.annotations['annot']['paragraphs']
        para_in_song = len(para_lyric_set)

        for i in range(para_in_song):
            start_time, end_time = para_lyric_set[i]['time'][0], para_lyric_set[i]['time'][1]
            # print(f"Start Time = {start_time}, End Time =  {end_time}")
            lyrics_for_para = para_lyric_set[i]['text']
            # print(f"Para Lyrics = {lyrics_for_para}")
            emoji_set = predict_emoji_from_text(lyrics_for_para)
            # print(f"Emoji Set = {emoji_set}", type(emoji_set), list(emoji_set))
            GLOBAL_PARA_LENGTH.append(end_time-start_time)

            dataframe.loc[df_counter] = [key, start_time,end_time, lyrics_for_para, list(emoji_set)]
            df_counter += 1

dataframe.to_csv("labelled_data.csv")
plt.hist(GLOBAL_PARA_LENGTH)
plt.show()
        