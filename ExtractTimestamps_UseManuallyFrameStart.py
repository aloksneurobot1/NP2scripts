# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 2025
Processes the *entire* NIDAQ binary file to extract frame timestamps
based on camera TTL pulses. Uses original sampling rate for peak detection.
Allows optional manual input for the first frame time, SAVING the manual time if provided.
Calculates FPS based on total frames / total duration.
Includes plotting for the first 10 seconds.
Saves output as [original_base_name]_timestamps.npy

@author: Alok
"""
import sys
import numpy as np
from pathlib import Path
from tkinter import Tk
from tkinter import filedialog
# IMPORTANT: Ensure 'DemoReadSGLXData' package (or the readSGLX.py file)
# is accessible in your Python environment/path.
try:
    from DemoReadSGLXData.readSGLX import readMeta, SampRate, makeMemMapRaw, ExtractDigital
except ImportError:
    print("Error: Could not import readSGLX functions.")
    print("Please ensure 'DemoReadSGLXData' is installed or readSGLX.py is in the Python path.")
    sys.exit(1) # Added sys import needed for exit
import sys # Added import
from scipy.signal import find_peaks
import datetime
import matplotlib.pyplot as plt

def AnalyzeNIdaqRecordings_Full_ManualStart_SaveManual(): # Renamed function

    # Get file from user
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)  # For Windows
    binFullPath = Path(filedialog.askopenfilename(title="Select the NIDAQ Binary file (.nidq.bin)"))
    root.destroy()

    if not binFullPath or not binFullPath.is_file():
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
    # plot_duration_sec = 16103.561874 # Duration to plot at the beginning
    # -----------------------------------

    # --- Calculate total samples and duration to read the entire file ---
    try:
        bytes_per_sample = 2 # Usually int16 = 2 bytes
        num_channels = int(meta['nSavedChans'])
        file_bytes = int(meta['fileSizeBytes'])
        totSamp = file_bytes // (num_channels * bytes_per_sample)
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
    print("\nFinding frame timestamps from Camera TTL channel (full duration, original rate)...")
    # WARNING: Peak finding on full-resolution data for long recordings can be memory/CPU intensive!
    ttl_data = digital_channel_camera[0, :]

    # Find *all* peaks (rising edges) first
    min_frame_interval_sec = 0.5 / 35 # Min interval assuming max ~35Hz FPS - adjust if needed
    min_dist_samples = int(min_frame_interval_sec * sRate)
    peaks_output, _ = find_peaks(ttl_data, height=0.5, distance=max(1, min_dist_samples))
    print(f"  Found {len(peaks_output)} potential frame triggers (peaks) in total.")

    auto_first_frame_time_sec = None
    if len(peaks_output) > 0:
        auto_first_frame_time_sec = peaks_output[0] / sRate
        print(f"  Automatically detected first frame time: {auto_first_frame_time_sec:.4f} seconds")
    else:
        print("  Warning: No frame triggers detected automatically.")

    # --- Manual First Frame Time Input ---
    manual_first_frame_time_sec = None
    start_time_threshold = None
    use_manual_time = False
    while True: # Loop until valid input or decision
        if auto_first_frame_time_sec is not None:
            prompt = (f"Use detected first frame time ({auto_first_frame_time_sec:.4f}s)? \n"
                      f"Enter 'y' to use, 'n' to enter manually, or 'q' to quit: ")
        else:
             prompt = ("No frames automatically detected. \n"
                       "Enter 'n' to specify a start time manually, or 'q' to quit: ")

        choice = input(prompt).lower().strip()

        if choice == 'y':
            if auto_first_frame_time_sec is None:
                print("Error: Cannot use auto-detected time as none was found.")
                continue # Re-prompt
            start_time_threshold = auto_first_frame_time_sec
            use_manual_time = False # Explicitly set flag
            print(f"Using auto-detected start time threshold: {start_time_threshold:.4f}s")
            break
        elif choice == 'n':
            try:
                manual_time_str = input("Enter the desired first frame time in seconds (e.g., 10.5): ")
                manual_first_frame_time_sec = float(manual_time_str)
                if manual_first_frame_time_sec < 0:
                    print("Error: Time cannot be negative.")
                    continue # Re-prompt
                start_time_threshold = manual_first_frame_time_sec
                use_manual_time = True # Set flag
                print(f"Using manual start time threshold: {start_time_threshold:.4f}s")
                break
            except ValueError:
                print("Invalid input. Please enter a number.")
            except Exception as e:
                 print(f"An error occurred: {e}")
        elif choice == 'q':
             print("Quitting.")
             return # Exit the function
        else:
             print("Invalid choice. Please enter 'y', 'n', or 'q'.")
    # ------------------------------------

    # --- Filter peaks based on the chosen start time threshold ---
    if start_time_threshold is None:
        print("Error: No start time threshold determined. Exiting.")
        return

    if len(peaks_output) > 0:
        all_absolute_frame_times = peaks_output / sRate
        valid_peaks_mask = all_absolute_frame_times >= start_time_threshold
        filtered_peaks_indices = peaks_output[valid_peaks_mask]
        FramesTimesInSec = all_absolute_frame_times[valid_peaks_mask] # Final timestamps array
        print(f"  Filtered peaks: Kept {len(FramesTimesInSec)} frames occurring at or after {start_time_threshold:.4f}s.")
    else:
        filtered_peaks_indices = np.array([])
        FramesTimesInSec = np.array([])
        print("  No peaks detected, resulting timestamp array will be empty.")


    # --- Determine the FirstFrameTimeInSec to SAVE ---
    # MODIFIED: Save the manual time if provided, otherwise save time of first detected peak after threshold
    if use_manual_time:
        Final_FirstFrameTimeInSec_ToSave = manual_first_frame_time_sec
        print(f"  Saving manually provided start time as FirstFrameTimeInSec: {Final_FirstFrameTimeInSec_ToSave:.4f}s")
    elif len(FramesTimesInSec) > 0:
        Final_FirstFrameTimeInSec_ToSave = FramesTimesInSec[0]
        print(f"  Saving time of first detected frame after threshold as FirstFrameTimeInSec: {Final_FirstFrameTimeInSec_ToSave:.4f} seconds")
    else:
        Final_FirstFrameTimeInSec_ToSave = None # No frames left after threshold
        print("  Warning: No frames remain after applying the start time threshold. FirstFrameTimeInSec will be None.")


    # --- Calculate FPS (Total *Filtered* Frames / Total *File* Duration) ---
    if file_duration_sec > 0 and len(FramesTimesInSec) > 0:
        fps_ttl_channel = len(FramesTimesInSec) / file_duration_sec
        print(f"  Calculated Avg FPS (filtered frames / total duration): {fps_ttl_channel:.4f} Hz")
    else:
        fps_ttl_channel = None
        print(f"  Could not calculate FPS (duration zero, no frames detected, or no frames after threshold).")


    # --- Get stimuli insertion and removal frames (user input) ---
    try:
        StimulusInsertionFrame = int(input("Stimuli insertion frame number (relative to original video, 0 if none): "))
        StimulusRemovalFrame = int(input("Stimuli removal frame number (relative to original video, 0 if none): "))
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

        # Plot detected peaks *after filtering* within this window
        if len(filtered_peaks_indices) > 0: # Use filtered peaks indices
             peaks_in_window_mask = filtered_peaks_indices < plot_samples
             peaks_in_window_indices = filtered_peaks_indices[peaks_in_window_mask]
             peaks_in_window_times = peaks_in_window_indices / sRate
             if len(peaks_in_window_indices) > 0:
                 ax[1].plot(peaks_in_window_times, camera_plot_data[peaks_in_window_indices], "x", color='red', markersize=8, label='Kept Frames')

        # Plot the threshold time
        if start_time_threshold is not None and start_time_threshold < plot_duration_sec:
             ax[1].axvline(start_time_threshold, color='green', linestyle='--', lw=2, label=f'Start Threshold ({start_time_threshold:.2f}s)')


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
        'FirstFrameTimeInSec': Final_FirstFrameTimeInSec_ToSave, # MODIFIED: Use the determined value
        'FramesTimesInSec': FramesTimesInSec,     # Frame times *after* threshold
        'StartTimeThresholdSec': start_time_threshold, # Added for reference which threshold was used
        'StimulusInsertionFrame': StimulusInsertionFrame,
        'StimulusRemovalFrame': StimulusRemovalFrame,
        # WARNING: Saving full digital channels can make the file VERY large!
        'DigitalChannelCamera': digital_channel_camera,
        'DigitalChannelClock': digital_channel_clock if digital_channel_clock is not None else np.array([]),
        'OriginalFileName': binFullPath.name,
        'FPS_CameraTTLChannel': fps_ttl_channel
    }

    # Create output filename
    original_name = binFullPath.name
    if original_name.endswith('.bin'): base_name = original_name[:-4]
    else: base_name = original_name
    output_filename = f"{base_name}_timestamps.npy"
    output_path = binFullPath.parent / output_filename

    try:
        np.save(output_path, timestamp_data)
        print(f"\nTimestamps (potentially filtered by start time) saved to: {output_path}")
        print(f"  Number of frame timestamps saved: {len(FramesTimesInSec)}")
        if use_manual_time:
            print(f"  Saved FirstFrameTimeInSec based on manual input: {Final_FirstFrameTimeInSec_ToSave:.4f}s")
        elif Final_FirstFrameTimeInSec_ToSave is not None:
             print(f"  Saved FirstFrameTimeInSec based on first detected frame >= threshold: {Final_FirstFrameTimeInSec_ToSave:.4f}s")
        else:
             print(f"  Saved FirstFrameTimeInSec is None (no frames found after threshold).")

    except Exception as e:
        print(f"\nError saving timestamp data to {output_path}: {e}")

    # Verification printout
    try:
        print("\nVerifying saved file contents (summary):")
        loaded_data = np.load(output_path, allow_pickle=True).item()
        for key, value in loaded_data.items():
            if isinstance(value, np.ndarray): print(f"  {key}: Shape={value.shape}, Type={value.dtype}, Size={value.size}")
            elif value is None: print(f"  {key}: None")
            else: print(f"  {key}: {value}")
    except Exception as e:
        print(f"Error loading or verifying saved file: {e}")


if __name__ == "__main__":
    # Make sure to import sys at the top if using sys.exit
    AnalyzeNIdaqRecordings_Full_ManualStart_SaveManual()