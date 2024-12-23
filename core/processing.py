from core.analysis import Analysis  
from core.synth import Synth
from core.rendering import Rendering

class Processing:
    def __init__(self, music_path, video_paths):
        self.analysis = Analysis(music_path, video_paths)
        self.synth = Synth()
        self.rendering = Rendering()
            
    def run_processing(self):
        data = self.analysis.run()
        synth_data = self.synth.run(data)
        self.rendering.run(synth_data)

