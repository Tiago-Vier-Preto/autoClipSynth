import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import itertools

def boltzmann_probability(energy, temperature=1.0):
    """Converts energy to probability using Boltzmann distribution."""
    return np.exp(-energy / temperature)

class Synth:
    def __init__(self, data_video, data_music):
        self.data_video = data_video  # [mcr_values, flow_peak_values, dynamism_values, peak_frequency_values]
        self.data_music = data_music  # [segments, saliency_results]
        self.precomputed_candidates = []

    def temporal_snapping(self, music_saliency, video_keyframes):
        """Aligns keyframes to salient notes using dynamic programming."""
        n = len(music_saliency)
        m = len(video_keyframes)

        dp = np.zeros((n + 1, m + 1))
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = abs(music_saliency[i - 1] - video_keyframes[j - 1])
                dp[i, j] = min(dp[i - 1, j - 1] + cost, dp[i, j - 1], dp[i - 1, j])

        alignment = []
        i, j = n, m
        while i > 0 and j > 0:
            if dp[i, j] == dp[i - 1, j - 1] + abs(music_saliency[i - 1] - video_keyframes[j - 1]):
                alignment.append((i - 1, j - 1))
                i, j = i - 1, j - 1
            elif dp[i, j] == dp[i, j - 1]:
                j -= 1
            else:
                i -= 1

        return alignment[::-1]

    def precompute_candidates(self):
        """Precomputes the optimal starting frame and scaling factor for each music-video pair."""
        for j, music_segment in enumerate(self.data_music[0]):
            music_saliency = self.data_music[1][j]
            candidates = []
            for i, video_features in enumerate(zip(*self.data_video)):
                motion_change_rate, flow_peak_value, dynamism, peak_frequency = video_features

                def cost_function(scale):
                    return match_cost(
                        music_segment,
                        music_saliency,
                        {
                            "motion_change_rate": motion_change_rate,
                            "peak_frequency": peak_frequency,
                            "dynamism": dynamism,
                            "velocity": flow_peak_value,
                        },
                        scale=scale
                    )

                best_cost = float('inf')
                optimal_scale = 1.0

                for init in np.linspace(0.5, 2.0, 10):
                    result = minimize(cost_function, x0=init, bounds=[(0.5, 2.0)])
                    if result.fun < best_cost:
                        best_cost = result.fun
                        optimal_scale = result.x[0]

                candidates.append((i, optimal_scale))

            self.precomputed_candidates.append(candidates)

    def metropolis_hastings(self, num_iter=1000, temperature=1.0):
        """Implements the Metropolis-Hastings algorithm for MCMC sampling."""
        theta = [min(candidates, key=lambda x: x[1])[0] for candidates in self.precomputed_candidates]

        def energy(labels):
            total_cost = 0
            for j, label in enumerate(labels):
                video_features = {
                    "motion_change_rate": self.data_video[0][label],
                    "peak_frequency": self.data_video[3][label],
                    "dynamism": self.data_video[2][label],
                    "velocity": self.data_video[1][label],
                }
                music_segment = self.data_music[0][j]
                saliency = self.data_music[1][j]
                total_cost += match_cost(music_segment, saliency, video_features)

            total_cost += aggregate_transition_cost(self.data_music[0], [
                [self.data_video[k][label] for label in labels] for k in range(4)
            ])

            total_cost += calculate_duplication_cost([
                [self.data_video[k][label] for label in labels] for k in range(4)
            ])

            return total_cost

        current_energy = energy(theta)

        for _ in range(num_iter):
            if np.random.rand() < 0.7:
                idx = np.random.randint(len(theta))
                theta_prime = theta[:]
                theta_prime[idx] = np.random.randint(len(self.data_video[0]))
            else:
                idx1, idx2 = np.random.choice(len(theta), 2, replace=False)
                theta_prime = theta[:]
                theta_prime[idx1], theta_prime[idx2] = theta_prime[idx2], theta_prime[idx1]

            new_energy = energy(theta_prime)
            acceptance_ratio = boltzmann_probability(new_energy, temperature) / boltzmann_probability(current_energy, temperature)

            if np.random.rand() < min(1, acceptance_ratio):
                theta = theta_prime
                current_energy = new_energy

        return theta

    def run(self):
        """Executes the two-stage optimization."""
        self.precompute_candidates()
        optimized_labels = self.metropolis_hastings()
        return optimized_labels

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

def match_cost(music_segment, music_saliency, video_features, scale=1.0, sigma_co=0.05):
    motion_change_rate = video_features["motion_change_rate"]
    video_peak_freq = video_features["peak_frequency"]
    music_pace = music_segment["pace"] * scale

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
    for segment_list in zip(*video_segments):
        for segment in segment_list:
            if segment not in video_counts:
                video_counts[segment] = 0
            video_counts[segment] += 1

    duplication_cost = 0
    for count in video_counts.values():
        if count > 1:
            duplication_cost += 2**(count - 1) - 1

    return duplication_cost
