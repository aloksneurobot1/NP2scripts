# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 13:58:50 2025
Modified on Fri Apr 11 2025
Processes the *entire* NIDAQ binary file to extract frame timestamps
based on camera TTL pulses. Uses original sampling rate for peak detection.
Calculates FPS based on total frames / total duration.
Includes plotting for the first 10 seconds.
Saves output as [original_base_name]_timestamps.npy

@author: HT_bo 
"""
import numpy as np
from pathlib import Path
from tkinter import Tk
from tkinter import filedialog
from DemoReadSGLXData.readSGLX import readMeta, SampRate, makeMemMapRaw, ExtractDigital
from scipy.signal import find_peaks
import datetime
import matplotlib.pyplot as plt

def AnalyzeNIdaqRecordings_Full_NoDownsample():

    # Get file from user
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)  # For Windows
    binFullPath = Path(filedialog.askopenfilename(title="Select the NIDAQ Binary file (.nidq.bin)"))
    root.destroy()

    if not binFullPath or not binFullPath.is_file(): # Added check if file exists
        print("No file selected or file does not exist. Exiting.")
        return

    print(f"Selected file: {binFullPath}")

    # Parse the corresponding metafile
    meta = readMeta(binFullPath)
    sRate = SampRate(meta) # Original sampling rate
    print(f"Sampling rate: {sRate} Hz")

    # --- Parameters ---
    digital_channel_line_camera = 1 # Line index for Digital Channel 1 (Camera TTL) - Check your wiring!
    digital_channel_line_clock = 0  # Line index for Digital Channel 0 (Clock) - Assuming not critical for frame times
    
    # -----------------------------------

    # --- Calculate total samples and duration to read the entire file ---
    try:
        bytes_per_sample = 2 # Usually int16 = 2 bytes
        num_channels = int(meta['nSavedChans'])
        file_bytes = int(meta['fileSizeBytes'])
        totSamp = file_bytes // (num_channels * bytes_per_sample)
        # Get duration directly from metadata for accurate FPS calculation
        file_duration_sec = float(meta['fileTimeSecs'])
        print(f"Total samples in file: {totSamp} (Duration from meta: {file_duration_sec:.4f} seconds)")
    except KeyError as e:
        print(f"Error: Metadata file {meta['metaFile']} missing key needed to calculate total samples/duration: {e}")
        return
    except Exception as e:
        print(f"Error calculating total samples/duration: {e}")
        return

    if file_duration_sec <= 0:
        print("Error: File duration reported in metadata is zero or negative.")
        return
    plot_duration_sec = file_duration_sec # Duration to plot at the beginning
    # Make memory map of the raw data file
    rawData = makeMemMapRaw(binFullPath, meta)

    print(f"Extracting digital channels for the full duration ({file_duration_sec:.2f}s)...")
    # --- Extract digital channel for camera (entire duration) ---
    try:
        digital_channel_camera = ExtractDigital(rawData, 0, totSamp - 1, 0, [digital_channel_line_camera], meta)
        print(f"  Extracted Camera TTL data (Channel {digital_channel_line_camera}), shape: {digital_channel_camera.shape}")
    except Exception as e:
        print(f"Error extracting Camera TTL digital channel: {e}")
        return

    # --- Extract the clock digital channel (optional, entire duration) ---
    digital_channel_clock = None # Initialize as None
    try:
        digital_channel_clock = ExtractDigital(rawData, 0, totSamp - 1, 0, [digital_channel_line_clock], meta)
        print(f"  Extracted Clock data (Channel {digital_channel_line_clock}), shape: {digital_channel_clock.shape}")
    except Exception as e:
        print(f"Error extracting Clock digital channel: {e}")
        print("  Proceeding without Clock channel data (will not be plotted or saved).")


    # --- Process Camera TTL Data (Full Duration, Original Sampling Rate) ---
    print("Finding frame timestamps from Camera TTL channel (full duration, original rate)...")
    # WARNING: Peak finding on full-resolution data for long recordings can be memory/CPU intensive!
    ttl_data = digital_channel_camera[0, :] # Use original data

    # Find peaks (rising edges)
    min_frame_interval_sec = 0.5 / 35 # Min interval assuming max ~35Hz FPS - adjust if needed
    min_dist_samples = int(min_frame_interval_sec * sRate)
    peaks_output, _ = find_peaks(ttl_data, height=0.5, distance=max(1, min_dist_samples)) # Ensure distance is at least 1
    print(f"  Found {len(peaks_output)} potential frame triggers (peaks).")

    # Calculate First Frame Time
    if len(peaks_output) > 0:
        FirstFrameTimeInSec = peaks_output[0] / sRate
        print(f"  First Frame Time: {FirstFrameTimeInSec:.4f} seconds")
    else:
        FirstFrameTimeInSec = None
        print("  Warning: No frame triggers found for FirstFrameTimeInSec.")

    # Calculate All Frame Times
    if len(peaks_output) > 0:
        FramesTimesInSec = peaks_output / sRate
    else:
        FramesTimesInSec = np.array([]) # Use empty array instead of None
        print("  Warning: No frame triggers found for FramesTimesInSec.")

    # --- Calculate FPS (Method 2: Total Frames / Total Duration) ---
    if file_duration_sec > 0 and len(peaks_output) > 0:
        fps_ttl_channel = len(peaks_output) / file_duration_sec
        print(f"  Calculated FPS (total frames / duration): {fps_ttl_channel:.4f} Hz")
    else:
        fps_ttl_channel = None
        print(f"  Could not calculate FPS (duration is zero or no frames detected).")

    # --- Get stimuli insertion and removal frames (user input) ---
    try:
        StimulusInsertionFrame = int(input("Stimuli insertion frame (enter 0 if none/unknown): "))
        StimulusRemovalFrame = int(input("Stimuli removal frame (enter 0 if none/unknown): "))
    except ValueError:
        print("Invalid input for frame numbers, setting to 0.")
        StimulusInsertionFrame = 0
        StimulusRemovalFrame = 0
    # ------------------------------------------------------------


    # --- Visualization (First N Seconds Only) ---
    # (Visualization code remains the same)
    print(f"\nPlotting first {plot_duration_sec} seconds of digital signals...")
    plot_samples = int(plot_duration_sec * sRate)
    plot_samples = min(plot_samples, totSamp) # Ensure we don't exceed total samples

    if plot_samples > 0:
        time_vector_plot = np.arange(plot_samples) / sRate

        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(12, 6))
        fig.suptitle(f"Digital Signals (First {plot_duration_sec} seconds)", fontsize=14)

        # Plot Clock Channel (if extracted)
        if digital_channel_clock is not None:
            clock_plot_data = digital_channel_clock[0, :plot_samples]
            ax[0].plot(time_vector_plot, clock_plot_data, label=f'Digital Channel {digital_channel_line_clock} (Clock)')
            ax[0].set_ylabel('Signal Value')
            ax[0].set_title('Clock Channel')
            ax[0].legend(loc='upper right')
            ax[0].grid(True, linestyle=':', alpha=0.7)
        else:
            ax[0].set_title('Clock Channel (Not Extracted or Error)')
            ax[0].text(0.5, 0.5, "Clock data not available", horizontalalignment='center', verticalalignment='center', transform=ax[0].transAxes)

        # Plot Camera TTL Channel
        camera_plot_data = digital_channel_camera[0, :plot_samples]
        ax[1].plot(time_vector_plot, camera_plot_data, label=f'Camera TTL (Digital Line {digital_channel_line_camera})', color='tab:orange')

        # Plot detected peaks in this window
        if len(peaks_output) > 0:
             peaks_in_window_mask = peaks_output < plot_samples
             peaks_in_window_indices = peaks_output[peaks_in_window_mask]
             peaks_in_window_times = peaks_in_window_indices / sRate
             if len(peaks_in_window_indices) > 0:
                 ax[1].plot(peaks_in_window_times, camera_plot_data[peaks_in_window_indices], "x", color='red', markersize=8, label='Detected Frames')

        ax[1].set_xlabel('Time (seconds)')
        ax[1].set_ylabel('Signal Value')
        ax[1].set_title('Camera TTL Channel')
        ax[1].legend(loc='upper right')
        ax[1].grid(True, linestyle=':', alpha=0.7)

        plt.xlim(0, plot_duration_sec) # Set x-axis limit
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for suptitle
        plt.show()
    else:
        print("  Skipping plot: Recording duration is zero or negative.")
    # --- End Visualization ---

    # --- Save Timestamps ---
    timestamp_data = {
        'FirstFrameTimeInSec': FirstFrameTimeInSec,
        'FramesTimesInSec': FramesTimesInSec,
        'StimulusInsertionFrame': StimulusInsertionFrame,
        'StimulusRemovalFrame': StimulusRemovalFrame,
        'DigitalChannelCamera': digital_channel_camera,
        'DigitalChannelClock': digital_channel_clock if digital_channel_clock is not None else np.array([]),
        'OriginalFileName': binFullPath.name,
        'FPS_CameraTTLChannel': fps_ttl_channel
    }

    # --- Create output filename and path (MODIFIED) ---
    original_name = binFullPath.name # e.g., M2_tdTom_Baseline_g0_t0.nidq.bin
    if original_name.endswith('.bin'):
        base_name = original_name[:-4] # Remove .bin
    else:
        base_name = original_name # Keep original name if it doesn't end with .bin

    output_filename = f"{base_name}_timestamps.npy" # e.g., M2_tdTom_Baseline_g0_t0.nidq_timestamps.npy
    output_path = binFullPath.parent / output_filename
    # --- End Modification ---

    try:
        np.save(output_path, timestamp_data)
        print(f"\nTimestamps for full recording saved to: {output_path}") # Updated print message
        print(f"  Number of frames saved: {len(FramesTimesInSec)}")
    except Exception as e:
        print(f"\nError saving timestamp data to {output_path}: {e}")

    # Optional: Load back and print summary to verify saving
    try:
        print("\nVerifying saved file contents (summary):")
        loaded_data = np.load(output_path, allow_pickle=True).item()
        for key, value in loaded_data.items():
            if isinstance(value, np.ndarray):
                print(f"  {key}: Shape={value.shape}, Type={value.dtype}, Size={value.size}")
            elif value is None:
                 print(f"  {key}: None")
            else:
                print(f"  {key}: {value}")
    except Exception as e:
        print(f"Error loading or verifying saved file: {e}")


if __name__ == "__main__":
    AnalyzeNIdaqRecordings_Full_NoDownsample()