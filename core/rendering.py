import math
import cv2
import numpy as np

class Rendering:
    def __init__(self, videos, optimal_sequence, musical_segments, output_name):
        self.videos = videos
        self.optimal_sequence = optimal_sequence
        self.musical_segments = musical_segments
        self.output_name = output_name

    def run(self):
        sorted_sequence = sorted(self.optimal_sequence, key=lambda x: x[4])
        new_video = []
        for clip in sorted_sequence:
            video_segment = []
            segment_start_time = self.musical_segments[clip[4]][0][0][0]  # Time of the first note in the first bar
            segment_end_time = self.musical_segments[clip[4]][-1][-1][0]  # Time of the last note in the last bar
            segment_duration = segment_end_time - segment_start_time

            video_index = clip[0]
            start_time = clip[1]
            end_time = clip[2]
            temporal_scale = clip[3]
            video = self.videos[video_index]
            frame = start_time
            while frame < end_time:
                frame_index = math.floor(frame)
                if frame_index < len(video):
                    video_segment.append(video[frame_index])
                else :
                    frame_index = start_time
                    video_segment.append(video[frame_index])
                frame += temporal_scale

            new_video.extend(video_segment)

        # Assuming all frames are of the same size and type
        height, width, layers = new_video[0].shape
        size = (width, height)

        # Create a video writer object
        out = cv2.VideoWriter(self.output_name, cv2.VideoWriter_fourcc(*'mp4v'), 30, size)

        for frame in new_video:
            out.write(frame)

        out.release()