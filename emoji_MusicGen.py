# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 23:52:25 2023

@author: zhouz
"""

import openai

def get_musicgen():
    '''
    Run this function only once.
    This will take about 2-3min.
    '''
    from audiocraft.models import musicgen
    
    model = musicgen.MusicGen.get_pretrained('small', device='cpu')
    model.set_generation_params(duration=10)
    
    return model

def get_completion(prompt):
    '''
    Ask Tom for the API key or you can use your own OpenAI API key.
    Use GPT to genrate sentences with a given prompt

    Parameters
        ----------
        prompt : str
    '''
    openai.api_key = "?"
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message["content"]

def get_emotion_from_emoji(emoji):
  '''
    Use GPT to genrate emotion with a given emoji

    Parameters
        ----------
        emoji : str(emoji)
  '''

  prompt = f"""
  Task:
  '''
  Provide an emotional, figurative, and metaphorical sentence to describe the emotion of a piece of {emoji} music.
  '''

  Reqiurement:
  '''
  1. Do not mention the emoji itself in the sentence
  2. Around 50 words
  3. Focus on the emotion of {emoji}
  4. Be creative with adjectives and use as many different ones as you can
  '''

  Sentence: A piece of music that...

  Sentence:
  """

  emotion = get_completion(prompt)

  return emotion

def get_genre_from_emotion(emotion):
  '''
    Use GPT to genrate a music genre with a given emotion

    Parameters
        ----------
        emotion : str
  '''
    
  prompt = f"""
  Task:
  '''
  Provide 1 music subgenre that best match the given emotion.
  '''

  Reqiurement:
  '''
  1. If the name of any subgenre contains two or more words, connect them with hyphen.
  2. Only print the name of the subgenre in your response.
  3. Be professional and creative.
  '''

  Emotion:
  '''
  {emotion}
  '''

  Subgenre:
  """

  subgenre = get_completion(prompt)
  return subgenre

def get_prompt(emoji):
  '''
    Use GPT to genrate a prompt for music generation with a given emoji

    Parameters
        ----------
        emoji : str(emoji)
  '''
    
  emotion = get_emotion_from_emoji(emoji)
  subgenre = get_genre_from_emotion(emotion)

  prompt = f"""
  Task:
  '''
  Write 1 response with professional music terms to describe a piece of music that best matches the given genre and emotion.
  '''

  Requirements:
  '''
  1. Only include 6 parameters in your response. [Instrumentation: , Timbre: , Chord: , Melody: , Drumbeat: , Tempo:]
  2. Use around 5 professional music theory terms (do not use full sentenses) to describe each of the paremeters, including
  tempo, drumbeats, chord progression, chord types, chord extension,
  instrumentation, and timber so that it fits the emotion
  4. Around 50 words.
  5. Only print the foramtted prompt in your response and nothing else.
  '''

  Formalize the response in this template:
  '''
  Chords: , Instrument: , Timbre: , Melody: , Drumbeat: , Tempo:
  '''

  Genre:
  '''
  {"Funk-rock"}
  '''

  Emotion:
  '''
  {"A piece of music that unleashes a tempest of electrifying sensations, like a thunderstorm of sonic brilliance, where the harmonies crash and collide with a mind-bending force, leaving you in a state of awe and bewilderment. It's a kaleidoscope of euphoria and chaos, a symphony that shatters the boundaries of your imagination."}
  '''

  Prompt: Chords: Groovy and funky, major 7th chords with extended 9th and 13th, Instrument: Electric guitar, bass, drums, and brass section, Timbre: Rich and vibrant, Melody: Energetic and catchy riffs that will make you want to dance like a wild cowboy. Drumbeat: Lively and syncopated. Tempo: 100 to 120.

  Genre:
  '''
  {'Jazz-Funk'}
  '''

  Emotion:
  '''
   {"A piece of music that dances with mischievous delight, its notes playfully teasing and taunting the listener's senses. It weaves a web of sly intrigue, its melodies whispering secrets and its rhythms winking with a knowing smirk. This music is a coy charmer, seducing with its cleverness and leaving a trail of playful mischief in its wake."}
  '''

  Prompt: Chords: Complex and colorful, with extended chords and altered extensions. Instrument: Electric piano, bass, drums, and brass section. Timbre: Warm and soulful. Melody: Smooth and sultry, with intricate improvisations. Drumbeat: Funky and syncopated, with emphasis on the off-beats. Tempo: Moderate, around 90 to 110.

  Genre:
  '''
  {'Ambient-Chillout'}
  '''

  Emotion:
  '''
   {"A piece of music that transports the listener to a celestial realm, where ethereal melodies dance on a bed of shimmering notes. It evokes a sense of serenity and tranquility, like a gentle breeze caressing the soul. The harmonies soar with angelic grace, enveloping the heart in a warm embrace, leaving behind a trail of pure bliss and heavenly euphoria."}
  '''

  Genre:
  '''
  {subgenre}
  '''

  Emotion:
  '''
  {emotion}
  '''

  Prompt:
  """

  response = get_completion(prompt)
  return (subgenre, emotion, response)

def gen_and_save(emoji):
    '''
    Use music Gen to generate a piece of music from a given emoji
    '''
    import torchaudio
    
    subgenre, emotion, prompt = get_prompt(emoji)
    res = model.generate([f"""Genre: {subgenre}, {prompt} Emotion: {emotion}"""], progress=True)
    samplerate = 32000
    out_f = emoji + '_emolast.wav'
    torchaudio.save(out_f, res_emolast[0].to("cpu"), samplerate)

if __name__ == "__main__":
    emoji = input("Emoji: ")
    gen_and_save(emoji)
