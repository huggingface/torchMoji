from text_emojize import predict_emoji_from_text
import pandas as pd
import DALI as dali_code
# import DALI as dali_code
dali_data_path = '/Users/amanshukla/Downloads/DALI_v1.0'


# Reading DALI data 
dali_data = dali_code.get_the_DALI_dataset(dali_data_path, skip=[], keep=[])


# ea3414d65ba84ec1b9e8aa74ec586434
song_info = dali_data['ea3414d65ba84ec1b9e8aa74ec586434'].info
song_artist, song_title, song_url, song_lang = song_info['artist'], song_info[
    'title'], "https://www.youtube.com/watch?v=" + song_info['audio']['url'], song_info['metadata']['language']

print("Artist: {}, Song: {}, Audio: {}, Language: {}".format(
    song_artist, song_title, song_url, song_lang))

# print(dali_data.keys())
# for key in dali_data.keys(): 

#     # can be changed from lines -> words / paragraphs
#     lines_lyric_set = dali_data[key].annotations['annot']['paragraphs']
#     lines_in_song = len(lines_lyric_set)

#     song_info = dali_data[key].info
#     song_artist, song_title, song_url, song_lang = song_info['artist'], song_info[
#         'title'], "https://www.youtube.com/watch?v=" + song_info['audio']['url'], song_info['metadata']['language']

#     print("----------- Song Information -------")
#     print("Artist: {}, Song: {}, Audio: {}, Language: {}".format(song_artist, song_title, song_url, song_lang))
#     for i in range(lines_in_song):
#         lyrics_for_line = lines_lyric_set[i]['text']
#         predict_emoji_from_text(lyrics_for_line)

# create_samples = [
#     ("https://www.youtube.com/watch?v=oytOOA9sOiE", "Take your Listen to my voice my disguise")]

#Any analysis on lyric based emoji prediction
# for lyric in create_samples:
#     print("---------------------Next Sample-----------------------")
#     predict_emoji_from_text(lyric[1])