import customtkinter as ctk
from tkinter import filedialog
import os
from core.processing import run_processing

class Window (ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("autoClipSynth")
        self.geometry("700x500")

        # Widgets
        self.music_button = ctk.CTkButton(self, text="Select Music Input", command=self.select_music)
        self.music_button.pack(pady=(50, 5))
        self.music_path = None

        self.music_label = ctk.CTkLabel(self, text="")
        self.music_label.pack(pady=10)

        # Frame to hold video buttons
        self.video_button_frame = ctk.CTkFrame(self)
        self.video_button_frame.pack(pady=10)

        self.clip_button = ctk.CTkButton(self.video_button_frame, text="Select Video Clips", command=self.select_clip)
        self.clip_button.pack(side="left", padx=5)

        self.reset_video_button = ctk.CTkButton(self.video_button_frame, text="Reset Video Clips", command=self.reset_video)
        self.reset_video_button.pack(side="left", padx=5)

        self.video_paths = []
        
        self.clip_label = ctk.CTkLabel(self, text="")
        self.clip_label.pack(pady=10)

        self.output_name = ctk.CTkEntry(self, placeholder_text="Output Name")
        self.output_name.pack(pady=10)

        self.generate_button = ctk.CTkButton(self, text="Generate Synth", command=self.generate)
        self.generate_button.pack(pady=10)

    def run(self):
        self.mainloop()
            
    def select_music(self):
        file_path = filedialog.askopenfilename(
            initialdir=os.getcwd(),
            title="Selecionar Música",
            filetypes=[("Audio Files", "*.midi *.mid")]
        )
        if file_path:
            self.music_path = file_path
            self.music_label.configure(text=f"Música selecionada: {file_path}")
    
    def select_clip(self):
        file_path = filedialog.askopenfilename(
            initialdir=os.getcwd(),
            title="Selecionar Vídeo",
            filetypes=[("Video Files", "*.mp4")]
        )
        if file_path:
            if file_path not in self.video_paths:
                self.video_paths.append(file_path)
                self.clip_label.configure(text="\n".join(self.video_paths))

    def reset_video(self):
        self.video_paths = []
        self.clip_label.configure(text="")

    def generate(self):
        run_processing(self.music_path, self.video_paths, self.output_name.get())
        