import cv2
import numpy as np

from typing import List, Tuple

class BackgroundRemover:
    def __init__(self, show_window: bool = True):
        self.show_window = show_window

    def train(self, video_file: str):
        cap = cv2.VideoCapture(video_file)
        ret, frame = cap.read()

        # convert frame to 32bit image
        self.bg = frame.astype(np.float32)

        if self.show_window:
            cv2.namedWindow("background-training")
            cv2.imshow("background-training", self.bg)

        # get video length
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        wait_interval = int(1000 / frame_count)
        if wait_interval == 0:
            wait_interval = 1
        current_frame = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            current_frame += 1
            cv2.accumulate(frame, self.bg)
            if self.show_window:
                cv2.imshow("background-training", (self.bg / current_frame).astype(np.uint8))
                cv2.setWindowTitle("background-training", f"Training background model: {current_frame} / {frame_count}")
                cv2.waitKey(wait_interval)


        cap.release()
        self.bg = (self.bg / frame_count).astype(np.uint8)
        if self.show_window:
            cv2.destroyWindow("background-training")
        
    
    def remove_background(self, frame: cv2.UMat):
        return cv2.absdiff(frame, self.bg)