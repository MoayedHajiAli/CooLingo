import os
import re

OUTPUT_FILE_NAME = "output_speach.wav"


def convertTextToSpeach(text_to_convert, output_path):
    output_path = os.path.join(output_path, OUTPUT_FILE_NAME)
    os.system(
        f"tts --text \"{text_to_convert}\" --out_path {output_path}"
    )
    return output_path
