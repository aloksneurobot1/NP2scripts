# -*- coding: utf-8 -*-
"""
This script creates a GUI for generating and playing different types of noise.

Install in root conda:
pip install numpy sounddevice

- tkinter: Used for creating the graphical user interface (GUI).
- numpy: Used for numerical operations and generating the noise.
- sounddevice: Used to play the audio in a non-blocking way.
- threading: Used to run audio generation in the background to keep the GUI responsive.
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
import sounddevice as sd
import threading
import time

class NoiseApp:
    def __init__(self, master):
        self.master = master
        master.title("Sleep Noise Generator")
        master.geometry("350x220") # Increased height for the timer
        master.resizable(False, False)

        # --- State Variables ---
        self.stream = None
        self.is_playing = False
        self.is_paused = False
        self.noise_data = None
        self.current_frame = 0
        self.start_time = 0
        self.elapsed_time = 0
        self.timer_running = False


        # --- Style ---
        style = ttk.Style()
        style.configure("TButton", padding=6, relief="flat", font=('Helvetica', 10))
        style.configure("TRadiobutton", font=('Helvetica', 10))
        style.configure("TLabel", font=('Helvetica', 10))
        style.configure("TFrame", background='#f0f0f0')

        # --- Main Frame ---
        main_frame = ttk.Frame(master, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Noise Selection ---
        noise_frame = ttk.Frame(main_frame)
        noise_frame.pack(pady=5, fill=tk.X)
        
        ttk.Label(noise_frame, text="Noise Type:").pack(side=tk.LEFT, padx=5)
        self.noise_type = tk.StringVar(value="pink")
        
        ttk.Radiobutton(noise_frame, text="White", variable=self.noise_type, value="white").pack(side=tk.LEFT)
        ttk.Radiobutton(noise_frame, text="Pink", variable=self.noise_type, value="pink").pack(side=tk.LEFT)
        ttk.Radiobutton(noise_frame, text="Brown", variable=self.noise_type, value="brown").pack(side=tk.LEFT)

        # --- Controls ---
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(pady=10, fill=tk.X, expand=True)

        self.play_pause_button = ttk.Button(control_frame, text="Play", command=self.toggle_play_pause)
        self.play_pause_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)

        self.stop_button = ttk.Button(control_frame, text="Stop", command=self.stop_noise)
        self.stop_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)

        # --- Status Bar ---
        status_frame = ttk.Frame(master, relief=tk.SUNKEN)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var, anchor=tk.W, padding=(5,2))
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.timer_var = tk.StringVar(value="00:00")
        self.timer_label = ttk.Label(status_frame, textvariable=self.timer_var, anchor=tk.E, padding=(5,2))
        self.timer_label.pack(side=tk.RIGHT)
        
        master.protocol("WM_DELETE_WINDOW", self.on_closing)

    def generate_noise_data(self, noise_type, duration=3600, samplerate=44100):
        """Generates a long audio track to avoid gaps during looping."""
        self.status_var.set(f"Generating {noise_type} noise...")
        self.master.update_idletasks()
        
        num_samples = int(duration * samplerate)
        
        if noise_type == "white":
            noise = np.random.randn(num_samples)
        elif noise_type == "pink":
            white = np.random.randn(num_samples)
            fft_white = np.fft.rfft(white)
            frequencies = np.fft.rfftfreq(num_samples, 1/samplerate)
            with np.errstate(divide='ignore', invalid='ignore'):
                pink_spectrum = fft_white / np.sqrt(np.maximum(frequencies, 1e-9))
            pink_spectrum[0] = 0
            noise = np.fft.irfft(pink_spectrum)
        elif noise_type == "brown":
            white = np.random.randn(num_samples)
            noise = np.cumsum(white)
        else:
            noise = np.zeros(num_samples)

        noise /= np.max(np.abs(noise))
        self.noise_data = noise.astype(np.float32)
        self.status_var.set("Ready to play")

    def audio_callback(self, outdata, frames, time, status):
        """
        This function is called by the sounddevice stream to provide audio data.
        It's designed to be thread-safe to prevent race conditions.
        """
        if status:
            print(f"Stream status: {status}")

        # Create a local reference to the data to prevent race conditions from the GUI thread.
        # This is the key change to fix the TypeError.
        noise_data_local = self.noise_data

        if noise_data_local is None:
            # If there's no data (e.g., after 'Stop' is pressed), fill the buffer with silence.
            outdata.fill(0)
            return

        chunk_end = self.current_frame + frames

        # Loop the audio using the local reference
        if chunk_end > len(noise_data_local):
            remaining_frames = chunk_end - len(noise_data_local)
            # Part 1: from current frame to the end of the array
            outdata[:frames-remaining_frames] = noise_data_local[self.current_frame:].reshape(-1, 1)
            # Part 2: from the beginning of the array to fill the rest
            outdata[frames-remaining_frames:] = noise_data_local[:remaining_frames].reshape(-1, 1)
            self.current_frame = remaining_frames
        else:
            outdata[:] = noise_data_local[self.current_frame:chunk_end].reshape(-1, 1)
            self.current_frame += frames


    def start_playback(self):
        """Starts the audio stream."""
        self.stream = sd.OutputStream(
            samplerate=44100,
            channels=1,
            callback=self.audio_callback,
            blocksize=1024
        )
        self.stream.start()
        self.is_playing = True
        self.is_paused = False
        self.play_pause_button.config(text="Pause")
        self.status_var.set(f"Playing {self.noise_type.get()} noise")

    def toggle_play_pause(self):
        """Handles the logic for the Play/Pause/Resume button."""
        if not self.is_playing: # If stopped, start playing
            threading.Thread(target=self.generate_and_play_thread, daemon=True).start()
        elif self.is_paused: # If paused, resume
            self.stream.start()
            self.is_paused = False
            self.timer_running = True
            self.start_time = time.time() # Reset start time for accurate elapsed calculation
            self.update_timer()
            self.play_pause_button.config(text="Pause")
            self.status_var.set(f"Playing {self.noise_type.get()} noise")
        else: # If playing, pause
            self.stream.stop()
            self.is_paused = True
            self.timer_running = False
            self.elapsed_time += time.time() - self.start_time # Add played time to total
            self.play_pause_button.config(text="Resume")
            self.status_var.set("Paused")
            
    def generate_and_play_thread(self):
        """Target for the generation thread."""
        self.generate_noise_data(self.noise_type.get())
        # Schedule the GUI-related start on the main thread
        self.master.after(0, self.start_playback_and_timer)

    def start_playback_and_timer(self):
        """Starts playback and the timer from the main thread."""
        self.current_frame = 0
        self.elapsed_time = 0
        self.start_time = time.time()
        self.timer_running = True
        self.update_timer()
        self.start_playback()

    def stop_noise(self):
        """Stops playback and resets the state."""
        if self.stream:
            self.stream.stop()
            self.stream.close()
        self.stream = None
        self.is_playing = False
        self.is_paused = False
        self.timer_running = False
        self.current_frame = 0
        self.noise_data = None
        self.play_pause_button.config(text="Play")
        self.status_var.set("Stopped. Ready.")
        self.timer_var.set("00:00")

    def update_timer(self):
        """Recursively updates the timer label."""
        if self.timer_running:
            current_elapsed = self.elapsed_time + (time.time() - self.start_time)
            minutes, seconds = divmod(int(current_elapsed), 60)
            self.timer_var.set(f"{minutes:02d}:{seconds:02d}")
            self.master.after(1000, self.update_timer)

    def on_closing(self):
        """Handles window close event to ensure the audio stream is stopped."""
        self.stop_noise()
        self.master.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = NoiseApp(root)
    root.mainloop()
