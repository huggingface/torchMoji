from fileinput import filename
import DALI as dali_code
import numpy as np
import glob
from pandas import read_csv, DataFrame
from pydub import AudioSegment
from ast import literal_eval

STANDARD_SAMPLING_RATE = 22050
AUDIO_PATH = "/Volumes/TOSHIBA EXT/DALI"


def resampling(audio,sr):
    if sr != STANDARD_SAMPLING_RATE:
        audio = lib.resample(audio,orig_sr=sr,target_sr=STANDARD_SAMPLING_RATE)
    return audio

def func_name():
        
    # import DALI as dali_code
    dali_data_path = '/Users/amanshukla/Downloads/DALI_v1.0'

    # label_data = read_csv('labelled_data.csv')

    # print(type(label_data.emoji_set))

    # keep_keys = label_data.dali_id.unique()


    # Format of dali_info : array(['UNIQUE_DALI_ID', 'ARTIST NAME-SONG NAME', 'YOUTUBE LINK', 'WORKING/NOT WORKING'])
    dali_info = dali_code.get_info(dali_data_path + '/info/DALI_DATA_INFO.gz')
    # audio_path = "/Users/amanshukla/miniforge3/torchMoji/data/audio"
    song_audio = dali_code.get_audio(dali_info, AUDIO_PATH, keep=[])


    # song = AudioSegment.from_file(
    #     '/Users/amanshukla/miniforge3/torchMoji/data/audio/ea3414d65ba84ec1b9e8aa74ec586434.mp3',
    #     format='mp4')



    # tracks = glob.glob(audio_path+'/*.mp3')

    # piecewise = DataFrame(columns=['audio_piece','emoji'])
    # counter = 0
    # for song in tracks:
    #     print("--------New Track--------")
    #     song_id = song.split('/')[-1][:-4]
    #     temp_df = label_data[label_data.dali_id == song_id]
    #     print(song_id)
    #     try:    
    #         mp3_file = AudioSegment.from_file(song, format='mp4')
    #     except:
    #         print(f"Format issue with {song_id}")
    #     for idx,row in temp_df.iterrows():
    #         # print("------New Para-------")
    #         s,e = row['start_time'], row['end_time']
    #         # print(f"sample_{song_id}_{idx}.mp3 | Start time = {s} | End Time = {e} | Lyrics = {row['para_lyrics']}")
    #         piece = mp3_file[s*1000:e*1000]
    #         # print(literal_eval(row['emoji_set'])[0])

    #         # Uncomment if want to generate samples. 
    #         # Commented as samples already generated
    #         if piece != AudioSegment.empty():
    #             audio_filename = audio_path + f'/split_samples/sample_{song_id}_{idx}.mp3'
    #             piece.export(audio_filename, format='mp4')
    #             # print(type(audio_filename))
    #             piecewise.loc[counter] = [audio_filename,
    #                                     literal_eval(row['emoji_set'])[0]]
    #             counter += 1

    # # piecewise.to_csv('audio_samples.csv')
    # print(piecewise.iloc[0].audio_piece)
    # for idx, r in piecewise.iterrows():

# sample_tracks = glob.glob(audio_path+'/split_samples/*.mp3')
# # print(len(sample_tracks))
# for s1 in sample_tracks:
#     # Using mono to have only a single channel for all audio
    # audio_buffer, sample_rate = lib.load(r['audio_piece'], mono=True)
    # D = lib.amplitude_to_db(np.abs(lib.stft(audio_buffer)), ref=np.max)
#     display.specshow(D, y_axis='hz', x_axis='time', sr=sample_rate)
#     plt.colorbar()
#     plt.show()


# Some sample keys
"""
fffa380a6faa410c8a3283cef044276d
fbda8aaad9f94267ad861d6af96b05e8
ff5fdfedf2a442bfb1016e3c2062bca1
fecaf1f8ac46423c95966194eebb7d7d
fd6be3601b4d4ca4accdbadf2ee31440
f762aef2e3b848c4a19af7b873c8e4c3
f2d0fdb5a5c6474abeceea34e38d980a
f2094b1c05814a47b5145c1d9edd81c5
ee222691643f40abb4f7f49d6e990766
ea3414d65ba84ec1b9e8aa74ec586434
"""


# plt.show()



# To-DOs

"""
Download audio tracks for records in DALI [Done, not all but some for testing]
Annotate emoji labels for each audio file in DALI [Doubt, should it be lyric based emoji?]
Create a dataloader object for the modified dataset [pending]
Do train-validation-test splits [pending]

Remix audio if different number of channels [Done, using mono for single channel]
Resample audio to same rate for all tracks (different songs may have different sampling rates) [Done, although no discrepency detected]

Duration consistency for training CNN [Doubt, each track has different shape, what does it mean?,
How is the shape different from the duration?]

ISSUE: Each of the track create a spectrogram with different dimension
    Truncate audio files of longer duration
    Pad audio files of shorter duration

Transform signal (create spectrogram) [Done]
Define CNN arrcitecture [pending]
Start training CNN [pending]
Get predictions [pending]

"""


def create_audio_samples(audio_path):
    tracks = glob.glob(audio_path+'/*.mp3')

    for song in tracks:
        print("--------New Track--------")
        song_id = song.split('/')[-1][:-4]
        try:
            mp3_file = AudioSegment.from_file(song, format='mp4')
        except:
            print(f"Format issue with {song_id}")

        # print(len(mp3_file))
        start_time = 0
        # Currently set to mean duration from tracks
        while start_time < 192:
            # print(start_time)
            piece = mp3_file[start_time*1000:(start_time+16)*1000]
            if piece != AudioSegment.empty():
                # Save only non-empty samples
                # May lead to some loss of data
                print(f"Non-empty sample at time {start_time} for {song_id}")
                audio_filename = audio_path + f'/sec_split/sample_{song_id}_{start_time}.mp3'
                piece.export(audio_filename, format='mp4')
            start_time = (start_time+16)


    



if __name__ == '__main__':
    # create_audio_samples(AUDIO_PATH)
    # print("yes")
    func_name()