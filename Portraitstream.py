import numpy as np
import cv2
from src.liveportrait import LiveSpeechPortraits
from PIL import Image 


prev_frame_time = 0
new_frame_time = 0

cv2.startWindowThread()
cv2.namedWindow("frame")
# inti model 
portrait = LiveSpeechPortraits()

frames = portrait.generate_protrait('sample_data/audio/sample.wav', fps=30, batch_size=8)
# Reading the video file until finished
# frames = [np.array(Image.open(f'sample_results/sample_data/audio/sample/input_{i}.jpg')) for i in range(1, 100)]
for cur_frames in frames:
    for frame in cur_frames:
        # displaying the frame with fps
        cv2.imshow('frame', frame)

        # press 'Q' if you want to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
 
# Destroy the all windows now
cv2.destroyAllWindows()