#!/usr/bin/env python3

# NOTE: this example requires PyAudio because it uses the Microphone class

import speech_recognition as sr
import os

output_path = os.getcwd()
output_path = os.path.join(output_path, 'audio_output')

# obtain audio from the microphone
r = sr.Recognizer()

with sr.Microphone() as source:
    print("Say something!", flush=True)
    audio = r.listen(source)

# write audio to a WAV file
temp_path = os.path.join(output_path, "microphone-results.wav")
with open(temp_path, "wb") as f:
    f.write(audio.get_wav_data())
