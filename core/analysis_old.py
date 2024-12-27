import cv2
import numpy as np
from scipy.signal import welch
from mido import MidiFile
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from scipy.stats import mode
from scipy.ndimage import gaussian_filter1d

# Constants for segment distance calculation
W0 = 10
W1 = 1
W2 = 1

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
        granularity = 10  # Desired number of segments
        music_segments = music_segmentation(self.music, granularity)
        saliency_results = calculate_saliency(self.music)
        return mcr_values, flow_peak_values, dynamism_values, peak_frequency_values, music_segments, saliency_results, self.videos

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

# Helper function to load and parse MIDI file
def load_music(music_path):
    midi = MidiFile(music_path)
    tracks = []

    for track in midi.tracks:
        notes = []
        time = 0
        for msg in track:
            time += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                notes.append((time, msg.note))
        tracks.append(notes)

    return tracks

# Function to compute pace, median pitch, and pitch std for a segment
def compute_segment_features(segment):
    times, pitches = zip(*segment)
    duration = times[-1] - times[0]
    pace = len(set(times)) / duration if duration > 0 else 0
    median_pitch = np.median(pitches)
    std_pitch = np.std(pitches)
    return pace, median_pitch, std_pitch

# Function to calculate segment distance
def segment_distance(m1, m2, mode_pace, sigma_pitch):
    pace_m1, median_pitch_m1, std_pitch_m1 = m1
    pace_m2, median_pitch_m2, std_pitch_m2 = m2

    dist = (
        W0 * abs(pace_m1 - pace_m2) / mode_pace +
        W1 * abs(median_pitch_m1 - median_pitch_m2) / sigma_pitch +
        W2 * abs(std_pitch_m1 - std_pitch_m2) / sigma_pitch
    )
    return dist

# Music segmentation function
def music_segmentation(tracks, granularity):
    all_bars = []

    # Split tracks into bars
    for notes in tracks:
        bars = []
        current_bar = []
        for i, note in enumerate(notes):
            current_bar.append(note)
            if (i + 1) % 4 == 0:  # Assume 4 beats per bar
                bars.append(current_bar)
                current_bar = []
        if current_bar:
            bars.append(current_bar)
        all_bars.extend(bars)

    # Compute features for each bar
    segment_features = [compute_segment_features(bar) for bar in all_bars]
    mode_pace = np.median([f[0] for f in segment_features])
    sigma_pitch = np.std([f[1] for f in segment_features])

    # Create condensed distance matrix manually
    num_segments = len(segment_features)
    condensed_distances = np.zeros((num_segments * (num_segments - 1)) // 2)
    k = 0
    for i in range(num_segments - 1):
        for j in range(i + 1, num_segments):
            condensed_distances[k] = segment_distance(segment_features[i], segment_features[j], mode_pace, sigma_pitch)
            k += 1

    # Perform hierarchical clustering
    Z = linkage(condensed_distances, method='average')
    clusters = fcluster(Z, t=granularity, criterion='maxclust')

    # Group bars into segments based on clusters
    segmented_music = []
    for cluster_id in np.unique(clusters):
        segment = [all_bars[i] for i in range(len(clusters)) if clusters[i] == cluster_id]
        segmented_music.append(segment)

    return segmented_music

def calculate_saliency(tracks):
    note_onsets = []  # List of tuples (time, pitch, velocity)
    tempo = 500000  # Default tempo (500000 microseconds per beat)

    for track in tracks:
        for time, note in track:
            note_onsets.append((time, note, 127))  # Assuming maximum velocity for simplicity

    note_onsets = sorted(note_onsets)  # Ensure they are sorted by time

    saliency_scores = {
        "pitch_peak": np.zeros(len(note_onsets)),
        "before_long_interval": np.zeros(len(note_onsets)),
        "after_long_interval": np.zeros(len(note_onsets)),
        "start_of_bar": np.zeros(len(note_onsets)),
        "start_of_new_bar": np.zeros(len(note_onsets)),
        "start_of_different_bar": np.zeros(len(note_onsets)),
        "pitch_shift": np.zeros(len(note_onsets)),
        "deviated_pitch": np.zeros(len(note_onsets)),
    }

    beats_per_bar = 4  # Assuming 4/4 time signature
    bar_duration = beats_per_bar * (60 / (tempo / 1e6))  # Bar duration in seconds
    one_beat = 60 / (tempo / 1e6)  # Duration of one beat

    def relative_pitches(notes):
        return [p - notes[0] for p in notes]

    for i, (time, pitch, velocity) in enumerate(note_onsets):
        if i > 0 and i < len(note_onsets) - 1:
            prev_pitch = max(n[1] for n in note_onsets if n[0] == note_onsets[i - 1][0])
            next_pitch = max(n[1] for n in note_onsets if n[0] == note_onsets[i + 1][0])

            if pitch > 2 * prev_pitch and pitch > 2 * next_pitch:
                saliency_scores["pitch_peak"][i] = 1

        if i < len(note_onsets) - 1:
            next_time = note_onsets[i + 1][0]
            if (next_time - time) >= one_beat:
                saliency_scores["before_long_interval"][i] = 1

        if i > 0:
            prev_time = note_onsets[i - 1][0]
            if (time - prev_time) >= one_beat:
                saliency_scores["after_long_interval"][i] = 1

        if time % bar_duration == 0:
            saliency_scores["start_of_bar"][i] = 1

        prev_bar_notes = [n for n in note_onsets if (time - bar_duration) <= n[0] < time]
        if len(prev_bar_notes) == 0:
            saliency_scores["start_of_new_bar"][i] = 1

        prev_bar = [n for n in note_onsets if n[0] // bar_duration == (time // bar_duration) - 1]
        curr_bar = [n for n in note_onsets if n[0] // bar_duration == time // bar_duration]

        if len(prev_bar) > 0 and len(curr_bar) > 0:
            prev_relative = relative_pitches([n[1] for n in prev_bar])
            curr_relative = relative_pitches([n[1] for n in curr_bar])
            match_onsets = len(set(n[0] for n in prev_bar).intersection(n[0] for n in curr_bar)) / len(curr_bar) > 0.8
            match_pitches = len(set(prev_relative).intersection(curr_relative)) / len(curr_relative) > 0.5

            if not (match_onsets and match_pitches):
                saliency_scores["start_of_different_bar"][i] = 1

        if len(prev_bar) > 0 and len(curr_bar) > 0:
            pitch_diff = [n2[1] - n1[1] for n1, n2 in zip(prev_bar, curr_bar) if n1[0] == n2[0]]
            if len(pitch_diff) > 0 and len(set(pitch_diff)) <= 1:
                saliency_scores["pitch_shift"][i] = mode(pitch_diff).mode[0]

        if len(prev_bar) > 0 and len(curr_bar) > 0:
            pitch_diff = [n2[1] - n1[1] for n1, n2 in zip(prev_bar, curr_bar) if n1[0] == n2[0]]
            if len(pitch_diff) > 0 and np.abs(pitch_diff - np.mean(pitch_diff)).max() > 2 * np.std(pitch_diff):
                saliency_scores["deviated_pitch"][i] = 1

    combined_saliency = np.zeros(len(note_onsets))
    for i, (time, pitch, velocity) in enumerate(note_onsets):
        combined_saliency[i] = (1 + velocity / 127) * sum(saliency_scores[key][i] for key in saliency_scores)

    time_diffs = np.diff([n[0] for n in note_onsets], prepend=0)
    sigmas = np.minimum(0.1, 0.25 * time_diffs)  # Define sigma based on time differences
    continuous_saliency = gaussian_filter1d(combined_saliency, sigma=1)

    return saliency_scores, combined_saliency, continuous_saliency