from cProfile import label
from operator import concat
import DALI as dali_code
# from examples.song import AUDIO_PATH
from text_emojize import predict_emoji_from_text
from pandas import DataFrame, get_dummies, Series,concat
import matplotlib.pyplot as plt
import numpy as np
import librosa as lib
import glob
from pydub import AudioSegment
import os
import librosa.display as display

import warnings
warnings.filterwarnings("ignore")


SPLIT_SIZE = 15
# AUDIO_PATH = "/Users/amanshukla/miniforge3/torchMoji/data"
AUDIO_PATH = "/Volumes/TOSHIBA EXT/DALI/audios"
# IMAGE_PATH = "/Volumes/TOSHIBA EXT/DALI/images"
# IMAGE_PATH = "/Users/amanshukla/miniforge3/torchMoji/data/image"

labeling = DataFrame(columns=['image_id', 'emoji'])
COUNTER=0

def get_emoji_prediction(song_entry, audio_start_time, audio_end_time, single_label):

    # can be changed from lines -> words / paragraphs
    word_lyric_set = song_entry.annotations['annot']['words']
    collect = []
    for word in word_lyric_set:
        # Only add words between the start and end times
        if word['time'][0] >= audio_start_time and word['time'][1] < audio_end_time:
            collect.append(word['text'])

    # Padding to maintain consistency
    if len(collect) < 12:
        padd = 12 - len(collect)
        collect += ["eos"]*padd

    sentence = ''.join(collect)

    # Getting target emoji label from DeepMoji
    target_label = predict_emoji_from_text(sentence, single_label)

    return target_label


def save_image_folder(matrix, sr, target, idx):

    # Create spectrogram
    display.specshow(matrix, sr=sr)

    if target == -1:
        save_path = IMAGE_PATH
    else:    
        # Check if target directory exists
        if not os.path.isdir(IMAGE_PATH + f'/{target}'):
            os.makedirs(IMAGE_PATH + f'/{target}')
        save_path = IMAGE_PATH + f'/{target}'
    plt.savefig(save_path + f'/{idx}.png')


def convert_to_spectrogram(audio_sample_piece, piece_start_time, unique_song_id):

    # Bug : Cannot load an audio segment to make spectrogram
    # Fix: Export segment to mp3 temporarily and import from there
    D, sample_rate = None, None
    if not os.path.isdir(AUDIO_PATH + '/temp_hold'):
        os.makedirs(AUDIO_PATH + '/temp_hold')

    audio_filename = AUDIO_PATH + '/temp_hold' + \
        f"/{unique_song_id}_{piece_start_time}.mp3"
    audio_sample_piece.export(audio_filename, format="mp4")

    try:    
        audio_buffer, sample_rate = lib.load(audio_filename, mono=True)
        D = lib.amplitude_to_db(np.abs(lib.stft(audio_buffer)), ref=np.max)
    except:
        # D, sample_rate = np.array(0.0).astype(float), np.array(0.0).astype(float)
        print(f"Loading issue for {audio_filename}")
    return D, sample_rate


def check_image_exists(unique_image_id,path):

    if glob.glob(path + f"/{unique_image_id}.png", recursive=True):
        return True
    
    return False


def create_splits(mp3, song_entry, unique_song_id, single_label):
    global COUNTER
    # Get length of audio
    song_length = mp3.duration_seconds

    # Number of samples to be created
    num_of_split_samples = song_length // SPLIT_SIZE

    print(
        f"Expecting {num_of_split_samples} samples for this song {unique_song_id}")
    # Initial time to start audio processing
    start_time = 0

    # While there are samples to process
    while num_of_split_samples:
        # End time for each sample piece
        ending_time = start_time+SPLIT_SIZE

        # Slicing in audio
        piece = mp3[start_time*1000:ending_time*1000]

        # If it is a valid piece
        if piece != AudioSegment.empty():

            # Unique image id with song id and sample time
            image_id = f"/{unique_song_id}_{start_time}"
            # Check if image already exists in image folder
            if not check_image_exists(image_id):

                # Get spectrogram matrix
                image_matrix, sample_rate = convert_to_spectrogram(
                    piece, start_time, unique_song_id)

                if image_matrix is None and sample_rate is None:
                    start_time = ending_time
                    num_of_split_samples -= 1
                    continue

                # Get emoji for the piece from lyric processing
                top_emoji = get_emoji_prediction(
                    song_entry, start_time, ending_time, single_label)

                # Save the spectrogram image under corresponding emoji label
                if single_label:    
                    save_image_folder(image_matrix, sample_rate,
                                    top_emoji, image_id)

                    # print(
                    #     f"Successfully Saved ! Non - Empty piece {num_of_split_samples} from {start_time} to {ending_time} with matrix {image_matrix.shape} and sample rate {sample_rate} with emoji label {top_emoji}")
                else:
                    save_image_folder(image_matrix, sample_rate, -1, image_id)
                    # labeling.loc[COUNTER] = [image_id[1:], top_emoji]
                    # COUNTER += 1
                    # print(f"{image_id} has {top_emoji} labels")
                    print(
                        f"Successfully Saved ! Non - Empty piece {num_of_split_samples} from {start_time} to {ending_time} with emoji label {top_emoji} into dataframe")

                # print(
                #         f"Successfully Saved ! Non - Empty piece {num_of_split_samples} from {start_time} to {ending_time} with matrix {image_matrix.shape} and sample rate {sample_rate} with emoji label {top_emoji}")

            else:
                print(f" Skipping! Image {image_id} already saved.")

        start_time = ending_time
        num_of_split_samples -= 1


def create_images(dali_data, single_label):

    # Collect all songs downloaded
    tracks = glob.glob(AUDIO_PATH+'/*.mp3')

    # Iterate through the songs
    for song in tracks:
        print("--------New Track--------")

        # Parse each song id
        song_id = song.split('/')[-1][:-4]

        if song_id in dali_data:
            
            # Get annotations for particular song
            entry = dali_data[song_id]

            # Extract song language from annotation
            song_language = entry.info["metadata"]['language']

            # Only process english songs
            if song_language == 'english':

                success_load = False
                # Load audio
                try:
                    mp3_file = AudioSegment.from_file(song, format='mp4')
                    success_load = True
                except:
                    print(f"Unable to load audio. Format issue with {song_id}")

                if success_load:
                    print("Yes it is an english song, succesfully loaded")

                    # Only process files which are successfully loaded
                    create_splits(mp3_file, entry, song_id, single_label)

                else:
                    print("English song but not loaded")
            else:
                print(f"Expected English song, got {song_language} song")
        else:
            print("Song lyrics not found")


def allinone(single_label=True):

    # Path to stored dataset
    dali_data_path = '/Users/amanshukla/Downloads/DALI_v1.0'

    # Format of dali_info : array(['UNIQUE_DALI_ID', 'ARTIST NAME-SONG NAME', 'YOUTUBE LINK', 'WORKING/NOT WORKING'])
    dali_info = dali_code.get_info(dali_data_path + '/info/DALI_DATA_INFO.gz')

    # Download songs from DALI and store at AUDIO_PATH
    # Use keep to filter out a subset of data

    # Commented as songs are already downloaded
    # song_audio = dali_code.get_audio(dali_info, AUDIO_PATH, keep=[
    #                                  "ea3414d65ba84ec1b9e8aa74ec586434"])

    # Reading DALI data for text processing
    dali_data = dali_code.get_the_DALI_dataset(dali_data_path, skip=[], keep=[])

    create_images(dali_data, single_label)


# def batches_of_16(timing, batch=list(range(16, 193, 16))):
#     string_set = []
#     i, j = 0, 0
#     string = ""
#     # currently set to 12 images per song, hence get only 12 string sets
#     while i < 12 and j < len(timing):
#         if batch[i] > timing[j][1]:
#             string += ' '+timing[j][0]
#             j += 1

#         else:
#             string_set.append(string)
#             string = ' '+timing[j][0]
#             i += 1
#             j += 1

#     return string_set

# dali_data_path = '/Users/amanshukla/Downloads/DALI_v1.0'

# # Reading DALI data [Only a subset for now]
# dali_data = dali_code.get_the_DALI_dataset(dali_data_path, skip=[], keep=["fffa380a6faa410c8a3283cef044276d",
#                                                                           "fbda8aaad9f94267ad861d6af96b05e8",
#                                                                           "ff5fdfedf2a442bfb1016e3c2062bca1",
#                                                                           "fecaf1f8ac46423c95966194eebb7d7d",
#                                                                           "fd6be3601b4d4ca4accdbadf2ee31440",
#                                                                           "f762aef2e3b848c4a19af7b873c8e4c3",
#                                                                           "f2d0fdb5a5c6474abeceea34e38d980a",
#                                                                           "f2094b1c05814a47b5145c1d9edd81c5",
#                                                                           "ee222691643f40abb4f7f49d6e990766",
#                                                                           "ea3414d65ba84ec1b9e8aa74ec586434"])

# dataframe = DataFrame(columns=['dali_id','start_time','end_time','para_lyrics','emoji_set'])
# df_counter = 0
# GLOBAL_PARA_LENGTH = []
# for key in dali_data.keys():

#     entry = dali_data[key]

#     # Filter english songs as embedding only available for english language
#     song_language = entry.info["metadata"]['language']
#     if song_language == 'english':

#     # can be changed from lines -> words / paragraphs
#         word_lyric_set = entry.annotations['annot']['words']
#         collect = []
#         for word in word_lyric_set:
#             end_time = word['time'][1]
#             text = word['text']
#             collect.append((text, end_time))


#         try:
#             text_set = batches_of_16(collect)
#         except:
#             print(f"Error, too less word to extract. Found only {len(collect)}, expected atleast 192")

#         if len(text_set) < 12:
#             padd = 12 - len(text_set)
#             text_set += ["eos"]*padd

#         # Can change to other confidence levels.
#         emoji_set = [predict_emoji_from_text(sentence) for sentence in text_set]
#         print(f"text set = {len(text_set)}, emoji set = {len(emoji_set)}")
#         print(emoji_set)
#         for i in range(para_in_song):
#             start_time, end_time = para_lyric_set[i]['time'][0], para_lyric_set[i]['time'][1]
#             # print(f"Start Time = {start_time}, End Time =  {end_time}")
#             lyrics_for_para = para_lyric_set[i]['text']
#             # print(f"Para Lyrics = {lyrics_for_para}")
#             emoji_set = predict_emoji_from_text(lyrics_for_para)
#             # print(f"Emoji Set = {emoji_set}", type(emoji_set), list(emoji_set))
#             GLOBAL_PARA_LENGTH.append(end_time-start_time)

#             dataframe.loc[df_counter] = [key, start_time,end_time, lyrics_for_para, list(emoji_set)]
#             df_counter += 1

# dataframe.to_csv("labelled_data.csv")
# plt.hist(GLOBAL_PARA_LENGTH)
# plt.show()

def multi_label(single_label=False):
    # Path to stored dataset
    dali_data_path = '/Users/amanshukla/Downloads/DALI_v1.0'

    # Format of dali_info : array(['UNIQUE_DALI_ID', 'ARTIST NAME-SONG NAME', 'YOUTUBE LINK', 'WORKING/NOT WORKING'])
    dali_info = dali_code.get_info(dali_data_path + '/info/DALI_DATA_INFO.gz')

    # Reading DALI data for text processing
    dali_data = dali_code.get_the_DALI_dataset(
        dali_data_path, skip=[], keep=[])

    create_images(dali_data, single_label)
    # print(labeling.T)
    # X = [x for row in labeling.emoji.values for x in row]
    # col = ["images"] + list(set(X))
    # df2 = get_dummies(labeling['emoji'].apply(Series).stack()).sum(level=0)

    # df = concat([labeling,df2], axis=1)

    # df.to_csv('real.csv', index=False)

    # print(df)
    # return df

if __name__ == '__main__':
#     # Uncomment this function to generate spectrograms for single class labels
#     # allinone()

#     # New function to genereate images with multiple labels for multi lanel class prediction
    multi_label()   
#     # pass
