import numpy as np
from scipy.stats import norm

class Synth:
    def __init__(self, data_video, data_music):
        self.data_video = data_video  # [mcr_values, flow_peak_values, dynamism_values, peak_frequency_values]
        self.data_music = data_music  # [segments, saliency_results]

    def run(self):
        total_match_cost = 0

        # Compute match costs
        for j, music_segment in enumerate(self.data_music[0]):
            for i in range(len(self.data_video[3])):
                video_features = {
                    "motion_change_rate": self.data_video[0][i],
                    "peak_frequency": self.data_video[3][i],
                    "dynamism": self.data_video[2][i],
                    "velocity": self.data_video[1][i],
                }
                saliency = self.data_music[1][j]
                cost = match_cost(music_segment, saliency, video_features)
                total_match_cost += cost

        # Compute transition costs
        total_transition_cost = aggregate_transition_cost(self.data_music[0], self.data_video)

        # Compute duplication costs
        duplication_cost = calculate_duplication_cost(self.data_video)

        return total_match_cost + total_transition_cost + duplication_cost

def co_occurrence_cost(music_saliency, motion_change_rate, sigma_co=0.05):
    if len(music_saliency) != len(motion_change_rate):
        raise ValueError("Length of music_saliency and motion_change_rate must be equal.")
    music_saliency = music_saliency - np.mean(music_saliency)
    motion_change_rate = motion_change_rate - np.mean(motion_change_rate)
    x = np.dot(music_saliency, motion_change_rate) / len(music_saliency)
    if x >= 0:
        return norm.cdf(x, scale=sigma_co)
    else:
        return 2 - norm.cdf(x, scale=sigma_co)

def pace_cost(music_pace, video_peak_freq, threshold_low=0.5, threshold_high=2):
    if music_pace > 1 and video_peak_freq < threshold_low:
        return 1
    elif music_pace < -1 and video_peak_freq > threshold_high:
        return 1
    return 0

def match_cost(music_segment, music_saliency, video_features, sigma_co=0.05):
    motion_change_rate = video_features["motion_change_rate"]
    video_peak_freq = video_features["peak_frequency"]
    music_pace = music_segment["pace"]

    co_occurrence = co_occurrence_cost(music_saliency, motion_change_rate, sigma_co)
    pace_mismatch = pace_cost(music_pace, video_peak_freq)

    return co_occurrence + pace_mismatch

def Delta(mi, mi_plus1, theta_i, theta_i_plus1):
    kappa_p = mi_plus1["pace"] / (mi["pace"] + 1e-6)
    kappa_v = theta_i_plus1["velocity"] / (theta_i["velocity"] + 1e-6)

    if kappa_p < 0.5 and kappa_v > 0.75:
        return 1
    elif kappa_p > 2 and kappa_v < 1.5:
        return 1
    return 0

def Lambda(mi, mi_plus1, theta_i, theta_i_plus1):
    delta_t = mi_plus1["num_tracks"] - mi["num_tracks"]
    delta_d = theta_i_plus1["dynamism"] - theta_i["dynamism"]

    if delta_t < 0 and delta_d > -0.3:
        return 1
    elif delta_t > 0 and delta_d < 0.3:
        return 1
    return 0

def transit(mi, mi_plus1, theta_i, theta_i_plus1):
    delta_cost = Delta(mi, mi_plus1, theta_i, theta_i_plus1)
    lambda_cost = Lambda(mi, mi_plus1, theta_i, theta_i_plus1)

    return delta_cost + lambda_cost

def aggregate_transition_cost(music_segments, video_segments):
    total_transition_cost = 0

    for i in range(len(music_segments) - 1):
        mi, mi_plus1 = music_segments[i], music_segments[i + 1]
        theta_i = {
            "dynamism": video_segments[2][i],
            "velocity": video_segments[1][i],
        }
        theta_i_plus1 = {
            "dynamism": video_segments[2][i + 1],
            "velocity": video_segments[1][i + 1],
        }

        delta_cost = Delta(mi, mi_plus1, theta_i, theta_i_plus1)
        lambda_cost = Lambda(mi, mi_plus1, theta_i, theta_i_plus1)

        total_transition_cost += delta_cost + lambda_cost

    return total_transition_cost

def calculate_duplication_cost(video_segments):
    video_counts = {}
    for segment_list in zip(*video_segments):  # Iterate through segment lists
        for segment in segment_list:
            if segment not in video_counts:
                video_counts[segment] = 0
            video_counts[segment] += 1

    duplication_cost = 0
    for count in video_counts.values():
        if count > 1:
            duplication_cost += 2**(count - 1) - 1

    return duplication_cost
