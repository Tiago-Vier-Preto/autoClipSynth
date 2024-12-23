from analysis import Analysis  
from synth import Synth
from rendering import Rendering

class Processing:
    def __init__(self, music_path, video_paths):
        self.analysis = Analysis(music_path, video_paths)
        # self.synth = Synth()
        # self.rendering = Rendering()
            
    def run_processing(self):
        data = self.analysis.run()
        return data
        synth_data = self.synth.run(data)
        self.rendering.run(synth_data)

