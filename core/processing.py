from core.analysis import Analysis  
from core.synth import Synth
from core.rendering import Rendering

def run_processing(music_path, video_paths, output_name):
    analysis = Analysis(music_path, video_paths)
    mcr_values, flow_peak_values, dynamism_values, peak_frequency_values, music_segments, saliency_results, videos, fps_list = analysis.run()
    synth = Synth(mcr_values, flow_peak_values, dynamism_values, peak_frequency_values, music_segments, saliency_results, videos, fps_list)
    optimal_sequence = synth.run()
    rendering = Rendering(videos, optimal_sequence, music_segments, output_name)
    rendering.run()

