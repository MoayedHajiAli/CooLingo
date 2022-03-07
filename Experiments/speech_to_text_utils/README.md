## Instructions:

**1. Create an Anaconda environment with python=3.7**\
	`conda create --name speechbot python=3.7`

**2. Install the following packages:**\
	a. `pip install SpeechRecognition`\
	b. `pip install pyaudio`

**NOTE:** if pyaudio did not work try the following:\
`pip install pipwin`\
`pipwin install pyaudio`

Now you are all set!

* Use the method `generate_text_from_speech` to get the speech from the mic and convert it into text (The method returns a string which is the text obtained from the input speech).
* For generating the audio file run the `get_speech.py` file
* For generating the text from the speech run the `speech_to_text.py` file

**NOTE:** see `example.py` file to check how this class works.
