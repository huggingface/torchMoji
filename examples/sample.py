from text_emojize import predict_emoji_from_text
import pandas as pd
import DALI as dali_code

# demo_dataset = pd.DataFrame(columns=["Audio", "Lyrics", "Best_emoji","Top_5_emojis"])
# import DALI as dali_code
dali_data_path = '/Users/amanshukla/Downloads/DALI_v1.0'

# Format of dali_info : array(['UNIQUE_DALI_ID', 'ARTIST NAME-SONG NAME', 'YOUTUBE LINK', 'WORKING/NOT WORKING'])
dali_info = dali_code.get_info(dali_data_path + '/info/DALI_DATA_INFO.gz')
# print(dali_info[1])

# Reading DALI data 
dali_data = dali_code.get_the_DALI_dataset(dali_data_path, skip=[], keep=[])

# Type of dali_data = dict with keys = unique dali_id and value = Annotation object
print(type(dali_data),len(dali_data))

# can be changed from lines -> words / paragraphs
lines_lyric_set = dali_data['fffdba22ae1647a2910541b4d4ec3bed'].annotations['annot']['lines']
lines_in_song = len(lines_lyric_set)

# print(lines_lyric_set)
for i in range(lines_in_song):
    lyrics_for_line = lines_lyric_set[i]['text']
    predict_emoji_from_text(lyrics_for_line)




create_samples = [
    ("https://www.youtube.com/watch?v=SrIxz9wHUX8", "Hey there, Delilah \
        What's it like in New York city? I'm a thousand miles away \
        But, girl, tonight you look so pretty Yes, you do \
        Time square can't shine as bright as you I swear, it's true \
        Hey there, Delilah Don't you worry about the distance I'm right there if you get lonely \
        Give this song another listen Close your eyes \
        Listen to my voice, it's my disguise"),
    ("https://www.youtube.com/watch?v=2vjPBrBU-TM", "Party girls don't get hurt \
     Can't feel anything, when will I learn? \
     I push it down, push it down \
     I'm the one for a good time call \
     Phone's blowin' up, ringin' my doorbell \
     I feel the love, feel the love \
     One, two, three, one, two, three, drink \
     One, two, three, one, two, three, drink \
     One, two, three, one, two, three, drink \
     Throw 'em back 'til I lose count \
     I'm gonna swing from the chandelier \
     From the chandelier \
     I'm gonna live like tomorrow doesn't exist \
     Like it doesn't exist \
     I'm gonna fly like a bird through the night \
     Feel my tears as they dry \
     I'm gonna swing from the chandelier \
     From the chandelier"),
    ("https://www.youtube.com/watch?v=BciS5krYL80", "On a dark desert highway, cool wind in my hair \
     Warm smell of colitas, rising up through the air\
     Up ahead in the distance, I saw shimmering light\
     My head grew heavy and my sight grew dim\
     I had to stop for the night\
     There she stood in the doorway\
     I heard the mission bell\
     And I was thinking to myself,\
     'This could be Heaven or this could be Hell'\
     Then she lit up a candle and she showed me the way\
     There were voices down the corridor,\
     I thought I heard them say..."),
    ("https://www.youtube.com/watch?v=nfs8NYg7yQM", "I know that dress is karma, perfume regret \
     You got me thinking 'bout when you were mine, oh \
     And now I'm all up on ya, what you expect?\
     But you're not coming home with me tonight\
     You just want attention, you don't want my heart\
     Maybe you just hate the thought of me with someone new\
     Yeah, you just want attention, I knew from the start\
     You're just making sure I'm never gettin' over you\
     you've been runnin' round, runnin' round, runnin' round throwing that dirt all on my name\
     'Cause you knew that I, knew that I, knew that I'd call you up\
     Baby, now that we're, now that we're, now that we're right here standing face-to-face\
     You already know, already know, already know that you won, oh"),
    ("https://www.youtube.com/watch?v=RgKAFK5djSk", "It's been a long day without you, my friend\
     And I'll tell you all about it when I see you again\
     We've come a long way from where we began\
     Oh, I'll tell you all about it when I see you again\
     When I see you again\
     Damn, who knew?\
     All the planes we flew, good things we been through\
     That I'd be standing right here talking to you\
     'Bout another path, I know we loved to hit the road and laugh\
     But something told me that it wouldn't last\
     Had to switch up, look at things different, see the bigger picture\
     Those were the days, hard work forever pays\
     Now I see you in a better place(see you in a better place)\
     Uh\
     How can we not talk about family when family's all that we got?\
     Everything I went through, you were standing there by my side\
     And now you gon' be with me for the last ride")
]


# for lyric in create_samples:
#     print("---------------------Next Sample-----------------------")
#     predict_emoji_from_text(lyric)