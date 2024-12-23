import cv2
import numpy as np
from scipy.signal import welch

class Analysis:
    def __init__(self, music_path, video_paths):
        self.videos, self.grayscale_videos = load_videos(video_paths)
        self.music = load_music(music_path)

    def run(self):
        optical_flows = list(map(compute_optical_flow, self.grayscale_videos))
        saliency_maps = list(map(compute_saliency_maps, self.grayscale_videos))
        temporal_differences = list(map(compute_temporal_difference_with_advection, optical_flows))
        mcr_values = list(map(lambda args: compute_mcr(*args), zip(temporal_differences, saliency_maps)))
        flow_peak_values = list(map(lambda args: compute_flow_peak(*args), zip(optical_flows, saliency_maps)))
        dynamism_values = list(map(compute_dynamism, optical_flows))
        peak_frequency_values = list(map(compute_peak_frequency, mcr_values))
        return [mcr_values, flow_peak_values, dynamism_values, peak_frequency_values]

def load_videos(video_paths):
    color_frames_list = [] 
    grayscale_frames_list = []

    for video_path in video_paths:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video file: {video_path}")
        color_frames = []
        grayscale_frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            color_frames.append(frame)
            grayscale_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        cap.release()
        color_frames_list.append(color_frames)
        grayscale_frames_list.append(grayscale_frames)

    return color_frames_list, grayscale_frames_list

def compute_optical_flow(frames):
    flows = []
    for i in range(1, len(frames)):
        flow = cv2.calcOpticalFlowFarneback(
            frames[i-1], frames[i], None,
            pyr_scale=0.5, levels=3, winsize=15, iterations=3,
            poly_n=5, poly_sigma=1.2, flags=0
        )
        flows.append(flow)
    return flows

def compute_saliency_maps(frames):
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    saliency_maps = []
    for frame in frames:
        _, saliency_map = saliency.computeSaliency(frame)
        saliency_maps.append(cv2.normalize(saliency_map, None, 0, 1, cv2.NORM_MINMAX))
    return saliency_maps

def compute_temporal_difference_with_advection(flows):
    temporal_differences = []
    for i in range(1, len(flows)):
        flow_current = flows[i]
        flow_previous = flows[i - 1]
        h, w = flow_current.shape[:2]
        x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
        advected_x = (x_coords - flow_previous[..., 0]).astype(np.float32)
        advected_y = (y_coords - flow_previous[..., 1]).astype(np.float32)
        for _ in range(3):
            advected_x_new = np.clip(advected_x - flow_previous[..., 0], 0, w - 1)
            advected_y_new = np.clip(advected_y - flow_previous[..., 1], 0, h - 1)
            advected_x, advected_y = advected_x_new, advected_y_new
        advected_flow = cv2.remap(flow_current, advected_x, advected_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        temporal_difference = flow_current - advected_flow
        temporal_differences.append(temporal_difference)
    return temporal_differences

def compute_mcr(temporal_differences, saliency_maps):
    mcr_values = []
    for i, diff in enumerate(temporal_differences):
        saliency = saliency_maps[i]
        magnitude = np.linalg.norm(diff, axis=2)
        weighted_magnitude = saliency * magnitude
        mcr = np.sum(weighted_magnitude) / (weighted_magnitude.shape[0] * weighted_magnitude.shape[1])
        mcr_values.append(mcr)
    max_mcr = max(mcr_values) if mcr_values else 1e-6
    mcr_values = [mcr / max_mcr for mcr in mcr_values]
    return mcr_values

def compute_flow_peak(optical_flows, saliency_maps):
    flow_peak_values = []
    for i, flow in enumerate(optical_flows):
        saliency = saliency_maps[i]
        magnitude = np.linalg.norm(flow, axis=2)
        weighted_magnitude = saliency * magnitude
        peak_value = np.percentile(weighted_magnitude, 99.9)
        flow_peak_values.append(peak_value)
    
    max_peak = max(flow_peak_values) if flow_peak_values else 1e-6
    flow_peak_values = [peak / max_peak for peak in flow_peak_values]
    
    return flow_peak_values

def compute_dynamism(optical_flows, threshold=2):
    dynamism_values = []
    
    for flow in optical_flows:
        magnitude = np.linalg.norm(flow, axis=2)  # Compute flow magnitude
        num_pixels = magnitude.size  # Total number of pixels
        num_high_motion_pixels = np.sum(magnitude > threshold)  # Count pixels above threshold
        dynamism = num_high_motion_pixels / num_pixels  # Ratio of high-motion pixels
        dynamism_values.append(dynamism)
    
    return dynamism_values

def compute_peak_frequency(mcr_values, window_size=30, threshold=-10):
    peak_frequencies = []
    half_window = window_size // 2
    for i in range(len(mcr_values)):
        start = max(0, i - half_window)
        end = min(len(mcr_values), i + half_window + 1)
        windowed_mcr = mcr_values[start:end]
        freqs, psd = welch(windowed_mcr, nperseg=min(len(windowed_mcr), 256))
        max_power_index = np.argmax(psd)
        max_power = psd[max_power_index]
        if max_power > threshold:
            peak_frequency = freqs[max_power_index]
        else:
            peak_frequency = 0
        peak_frequencies.append(peak_frequency)

    return peak_frequencies

#WILL BE IMPLEMENTED LATER IGNORE FOR NOW
def load_music(music_path):
    return None