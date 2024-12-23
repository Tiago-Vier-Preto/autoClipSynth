import cv2
import numpy as np

class Analysis:
    def __init__(self, music_path, video_paths):
        self.videos_frames = map(self.load_videos, video_paths)
        self.music = self.load_music(music_path)

    def run(self):
        pass
    
    def load_videos(self, video_paths):
        cap = cv2.VideoCapture(video_paths)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)  # Convert to grayscale
        cap.release()
        return frames
    
    def load_music(self, music_path):
        pass


    