import numpy as np
import random
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import linear_sum_assignment

class Synth:
    def __init__(self, mcr_values, flow_peak_values, dynamism_values, peak_frequency_values, music_segments, saliency_results, videos, fps_list):
        self.mcr_values = mcr_values
        self.flow_peak_values = flow_peak_values
        self.dynamism_values = dynamism_values
        self.peak_frequency_values = peak_frequency_values
        self.music_segments = music_segments
        self.music_saliency = saliency_results
        self.mean_pace, self.std_pace = calculate_mean_and_std(music_segments)
        self.videos = videos
        self.fps_list = fps_list    

    def run(self):
        # Perform global alignment
        best_theta, best_cost = global_alignment(self.videos, self.music_segments, self.music_saliency, self.mcr_values, self.peak_frequency_values, self.mean_pace, self.std_pace, self.fps_list)
        num_iter = 10
        # Perform MCMC sampling
        print(best_theta)
        print("\nfim do gordo e inicio do magrao\n")
        optimal_sequence = self.metropolis_hastings(best_theta, num_iter)
        print(optimal_sequence)
        return optimal_sequence
    
    def metropolis_hastings(self, results, num_iter):
        # Inicialização de θ a partir dos resultados fornecidos
        theta = results
        q = len(theta)
        P = lambda E: np.exp(-E / 1.0)  # Distribuição de Boltzmann com T=1.0
        
        for iter in range(num_iter):
            # Calcula o custo inicial
            matching_costs = [
                matching_cost(
                    self.music_segments[theta[i][4]],
                    theta[i],
                    self.music_saliency,
                    self.mcr_values[i % len(self.mcr_values)],
                    len(theta),
                    self.peak_frequency_values[i % len(self.peak_frequency_values)],
                    self.mean_pace,
                    self.std_pace
                ) for i in range(len(theta))
            ]
            E0 = totalMatchingCost(matching_costs) + globalConstraint([t[0] for t in theta])
            P0 = P(E0)
            
            # Realiza uma mutação
            if random.uniform(0, 1) < 0.7:
                # Tipo 1: Atualiza um índice aleatório
                i = random.randint(0, q - 1)
                j = random.randint(0, len(self.videos) - 1)
                old_value = theta[i]
                theta[i] = (j, *theta[i][1:])  # Atualiza o índice do vídeo
            else:
                # Tipo 2: Troca dois índices de segmento
                i, j = random.sample(range(q), 2)
                theta[i], theta[j] = theta[j], theta[i]
            
            # Calcula o custo após a mutação
            matching_costs = [
                matching_cost(
                    self.music_segments[theta[i][4]],
                    theta[i],
                    self.music_saliency,
                    self.mcr_values[i % len(self.mcr_values)],
                    len(theta),
                    self.peak_frequency_values[i % len(self.peak_frequency_values)],
                    self.mean_pace,
                    self.std_pace
                ) for i in range(len(theta))
            ]
            E1 = totalMatchingCost(matching_costs) + globalConstraint([t[0] for t in theta])
            
            # Adiciona custo de transição entre pares de segmentos
            """for i in range(q - 1):
                E1 += transit(
                    theta[i],
                    theta[i + 1],
                    self.music_segments[i],
                    self.music_segments[i + 1],
                    self.mean_pace,
                    self.std_pace,
                    self.flow_peak_values[i],
                    self.dynamism_values
                )"""
            
            P1 = P(E1)
            
            # Critério de aceitação/rejeição
            if random.uniform(0, 1) > min(P1 / P0, 1):
                # Rejeita a mutação, desfazendo-a
                if random.uniform(0, 1) < 0.7:
                    theta[i] = old_value
                else:
                    theta[i], theta[j] = theta[j], theta[i]

        return theta

        
def co_occurance_cost(saliency, mcr, N):
    saliency_new = np.array(saliency) - np.mean(saliency)
    mcr_new = np.array(mcr) - np.mean(mcr)
    mcr_new /= N

    # Align shapes of saliency_new and mcr_new
    if len(saliency_new) != len(mcr_new):
        min_length = min(len(saliency_new), len(mcr_new))
        saliency_new = saliency_new[:min_length]
        mcr_new = mcr_new[:min_length]

    x = np.dot(saliency_new.T, mcr_new)  # Ensure x is a scalar
    x = np.array([x])  # Wrap x in a numpy array to make it 1D

    sigma = 0.05

    if np.all(x >= 0):
        return gaussian_filter1d(x, sigma=sigma)[0]  # Return the scalar result
    else:
        return (2 - gaussian_filter1d(x, sigma=sigma))[0]  # Return the scalar result

def pace_peak_mismatch(music_segment, theta, peak_freq_values, mean_pace, std_pace):
    pace = calculate_pace(music_segment, mean_pace, std_pace)
    sum_peak_freq_values = 0
    count = 0
    for i in range(theta[1], theta[2]):
        if i < len(peak_freq_values):  # Ensure index is within bounds
            sum_peak_freq_values += peak_freq_values[i]
            count += 1

    if count > 0:
        mean_peak_freq_values = sum_peak_freq_values / count
    else:
        mean_peak_freq_values = 0

    if pace > 1 and mean_peak_freq_values < 0.5:
        return 1
    if pace < -1 and mean_peak_freq_values > 2:
        return 1
    else:
        return 0
    
    
def matching_cost(music_segment, theta, saliency, mcr, N, peak_freq_values, mean_pace, std_pace):
    return co_occurance_cost(saliency, mcr, N) + pace_peak_mismatch(music_segment, theta, peak_freq_values, mean_pace, std_pace)

def totalMatchingCost(matching_costs):
    return np.sum(matching_costs)

def calculate_pace(music_segment, mean_pace, std_pace):
    pace = 0
    count = 0
    for bar in music_segment:
        times, pitches = zip(*bar)
        duration = times[-1] - times[0]
        pace += len(set(times)) / duration if duration > 0 else 0
        count += 1
    
    pace /= count if count > 0 else 1

    # Padroniza o valor do pace
    standardized_pace = (pace - mean_pace) / std_pace if std_pace > 0 else 0
    return standardized_pace

def calculate_mean_and_std(segments):
    paces = [calculate_pace(segment, 0, 1) for segment in segments]  # Calcula os paces sem padronização
    mean = np.mean(paces)
    std = np.std(paces)
    return mean, std

def globalConstraint(video_segments):
    segments = set(video_segments)
    sum_duplication_cost = 0

    for segment in segments:
        sum_duplication_cost += pow(2, video_segments.count(segment) - 1) - 1

    return sum_duplication_cost


def transit(theta, theta_next, music_segment, music_segment_next, mean_pace, std_pace, flow_peak_values, dynamism_values):
    return deltaTransit(music_segment, music_segment_next, theta, theta_next, mean_pace, std_pace, flow_peak_values) + lambdaTransit(music_segment, music_segment_next, theta, theta_next, dynamism_values)

def deltaTransit(music_segment, music_segment_next, theta, theta_next, mean_pace, std_pace, flow_peak_values):
    vel = np.mean(flow_peak_values[theta[1]:theta[2] + 1])
    vel_next = np.mean(flow_peak_values[theta_next[1]:theta_next[2] + 1])
    k_v = vel_next / vel
    k_p = calculate_pace(music_segment_next, mean_pace, std_pace) / calculate_pace(music_segment, mean_pace, std_pace)

    if k_p < 0.5 and k_v > 0.75:
        return 1
    if k_p > 2 and k_v < 1.5:
        return 1
    else:
        return 0
    

def lambdaTransit(music_segment, music_segment_next, theta, theta_next, dynamism_values):
    delta_t = len(music_segment_next) - len(music_segment)
    delta_d = dynamism_values[theta_next[1]] - dynamism_values[theta[2]]

    if delta_t < 0 and delta_d > -0.3:
        return 1
    if delta_t > 0 and delta_d < 0.3:
        return 1
    else:   
        return 0
    
def global_alignment(videos, music_segments, saliency_results, mcr_values, peak_freq_values, mean_pace, std_pace, fps_list):
    best_cost = float('inf')
    best_theta = {}
    start = 0

    for video_index, video in enumerate(videos):
        video_length = len(video)  # Número total de quadros no vídeo

        for index_segment, music_segment in enumerate(music_segments):
            end = start + len(music_segment)
            for start_frame in range(video_length):

                # Inicializações regulares do fator de escala temporal
                initial_scales = np.linspace(0.5, 2.0, num=10)

                for initial_scale in initial_scales:
                    # Otimização contínua via gradiente descendente
                    scale_factor = initial_scale
                    learning_rate = 0.01
                    tolerance = 1e-5
                    max_iterations = 3

                    end_frame = calculate_video_end_frame(music_segment, start_frame, scale_factor, fps_list[video_index])

                    for _ in range(max_iterations):
                        # Define o theta atual
                        theta = (video_index, start_frame, end_frame, scale_factor)

                        # Calcula o custo de correspondência para o par música-vídeo
                        cost = matching_cost(music_segment, theta, saliency_results[start:end], mcr_values[video_index], theta[2] - theta[1] + 1, peak_freq_values[video_index], mean_pace, std_pace)

                        # Gradiente (derivado aproximado numericamente)
                        scale_factor_delta = scale_factor + 1e-4
                        theta_delta = (video_index, start_frame, end_frame, scale_factor_delta)
                        cost_delta = matching_cost(music_segment, theta_delta, saliency_results[start:end], mcr_values[video_index], theta_delta[2] - theta_delta[1] + 1, peak_freq_values[video_index], mean_pace, std_pace)

                        gradient = (cost_delta - cost) / 1e-4

                        # Atualiza o fator de escala
                        scale_factor -= learning_rate * gradient

                        # Verifica a convergência
                        if np.all(abs(gradient) < tolerance):
                            break

                    # Avalia o custo final para este ponto de partida e fator de escala
                    final_cost = matching_cost(
                        music_segment,
                        (video_index, start_frame, end_frame, scale_factor),
                        saliency_results[start:end],
                        mcr_values[video_index],
                        theta_delta[2] - theta_delta[1] + 1,
                        peak_freq_values[video_index],
                        mean_pace,
                        std_pace,
                    )

                    # Store the best configuration for this (video_index, index_segment) pair
                    if (video_index, index_segment) not in best_theta or final_cost < best_theta[(video_index, index_segment)][1]:
                        best_theta[(video_index, index_segment)] = ((video_index, start_frame, end_frame, scale_factor, index_segment), final_cost)
        start = end + 1

    # Extract only the theta values
    optimal_theta = [theta[0] for theta in best_theta.values()]
    return optimal_theta, best_cost


def calculate_video_end_frame(music_segment, video_start_frame, scaling_factor, video_fps):
    """
    Calculate the end frame for a video segment based on the music segment duration.

    Parameters:
        music_segment (list): A music segment, a list of bars with note tuples (time, pitch).
        video_start_frame (int): The start frame of the video.
        scaling_factor (float): The scaling factor to convert MIDI time to video time.
        video_fps (int): Frames per second of the video.

    Returns:
        int: The end frame for the video segment.
    """
    # Calculate segment duration in MIDI time
    segment_start_time = music_segment[0][0][0]  # Time of the first note in the first bar
    segment_end_time = music_segment[-1][-1][0]  # Time of the last note in the last bar
    segment_duration = segment_end_time - segment_start_time

    # Scale MIDI time to video frames
    segment_duration_seconds = segment_duration * scaling_factor
    segment_duration_frames = int(segment_duration_seconds * video_fps)

    # Calculate end frame for the segment
    end_frame = video_start_frame + segment_duration_frames

    return end_frame
            

def temporal_snapping(global_alignment_result, videos, music_segments, flow_peak_values, music_saliency):
        video_index, start_frame, end_frame, scale_factor = global_alignment_result
        
        video = videos[video_index][start_frame:end_frame]
        music_segment = music_segments[video_index]

        # Identify keyframes and salient notes
        keyframes = identify_keyframes(video, flow_peak_values)
        salient_notes = identify_salient_notes(music_segment, music_saliency[video_index])

        # Compute matching via dynamic programming
        matched_frames, cost = temporal_matching(keyframes, salient_notes)

        # Apply snapping and scaling
        snapped_video = apply_snapping(video, keyframes, salient_notes, matched_frames)
        return snapped_video

def identify_keyframes(video, flow_peak_values):
    keyframes = [0, len(video) - 1]  # Start and end frames

    # Identify local peaks above the 90th percentile
    motion_change_rate = np.diff(flow_peak_values)
    threshold = np.percentile(motion_change_rate, 90)

    for i in range(1, len(motion_change_rate) - 1):
        if motion_change_rate[i] > threshold and motion_change_rate[i] > motion_change_rate[i - 1] and motion_change_rate[i] > motion_change_rate[i + 1]:
            keyframes.append(i)

    keyframes.sort()
    return [(t, flow_peak_values[t]) for t in keyframes if t < len(flow_peak_values)]

def identify_salient_notes(music_segment, saliency):
    if not isinstance(saliency, (list, np.ndarray)):
        saliency = np.array([saliency])  # Convert to array if it's a scalar

    if len(saliency) < 2:
        raise ValueError("Saliency must have at least two elements.")

    salient_notes = [
        (0, saliency[0]),  # First note
        (len(music_segment) - 1, saliency[-1])  # Last note
    ]

    for i, (onset, score) in enumerate(zip(music_segment, saliency)):
        if score >= 0.5:
            salient_notes.append((i, score))

    salient_notes.sort()
    return salient_notes

def temporal_matching(keyframes, salient_notes):
    n, m = len(keyframes), len(salient_notes)
    cost_matrix = np.zeros((n, m))

    for i in range(n):
        for j in range(m):
            t1, w1 = keyframes[i]
            t2, w2 = salient_notes[j]
            cost_matrix[i, j] = (t1 - t2) ** 2 - min(w1 * w2, 0.25)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched_frames = [(keyframes[i], salient_notes[j]) for i, j in zip(row_ind, col_ind)]
    total_cost = cost_matrix[row_ind, col_ind].sum()
    return matched_frames, total_cost

def apply_snapping(video, keyframes, salient_notes, matched_frames):
    snapped_video = []
    
    for ((t1, _), (t2, _)) in matched_frames:
        segment_length = abs(t2 - t1)
        snapped_video.extend(video[t1:t1 + segment_length])

    return snapped_video

