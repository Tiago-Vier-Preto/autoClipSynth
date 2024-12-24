from core.analysis import Analysis  
from core.synth import Synth
from core.rendering import Rendering

def run_processing(music_path, video_paths):
    analysis = Analysis(music_path, video_paths)
    data_video, data_music = analysis.run()
    synth = Synth(data_video, data_music)
    return synth.run()
    #rendering.run(synth_data)

