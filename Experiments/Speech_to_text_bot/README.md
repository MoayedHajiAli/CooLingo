*** IMPORTANT ***

First, create an Anaconda environment with python=3.7

conda create --name speechbot python=3.7

Then, install the following packages:
1. pip install SpeechRecognition
2. pip install pyaudio

NOTE: if pyaudio did not work try the following:
pip install pipwin
pipwin install pyaudio.

Now you are all set!

For generating the audio file run get_speech.py
For generating the text from the speech run speech_to_text.py