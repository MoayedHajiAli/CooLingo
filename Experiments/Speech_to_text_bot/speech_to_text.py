#!/usr/bin/env python3

import speech_recognition as sr

from os import path
import os
AUDIO_FILE = path.join(path.dirname(path.realpath(__file__)), path.join("audio_output", "microphone-results.wav"))

output_file_name = "text_from_speech.txt"
output_path = os.getcwd()
output_path = os.path.join(output_path, output_file_name)

# use the audio file as the audio source
r = sr.Recognizer()
with sr.AudioFile(AUDIO_FILE) as source:
    audio = r.record(source)  # read the entire audio file

try:
    # for testing purposes, we're just using the default API key
    # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
    # instead of `r.recognize_google(audio)`
    print("Google Speech Recognition thinks you said " + r.recognize_google(audio))
    with open(output_path, 'w') as file:
        file.write(r.recognize_google(audio))
except sr.UnknownValueError:
    print("Google Speech Recognition could not understand audio")
except sr.RequestError as e:
    print("Could not request results from Google Speech Recognition service; {0}".format(e))
