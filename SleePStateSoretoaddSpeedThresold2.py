# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 17:00:07 2025

@author: HT_bo
"""
import numpy as np
from scipy import signal
from sklearn.decomposition import PCA, FastICA
from scipy.stats import zscore
from tkinter import Tk
from tkinter import filedialog
from pathlib import Path
from DemoReadSGLXData.readSGLX import readMeta
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd

# Redirect stdout and stderr to a file
output_file_path = "script_output.txt"  # You can change the filename if you like
original_stdout = sys.stdout
original_stderr = sys.stderr
sys.stdout = open(output_file_path, 'w')
sys.stderr = sys.stdout

def read_bin_data(file_path, meta_path, data_type='int16'):
    """
    Reads binary data from a file using memory-mapping.

    Args:
        file_path (str): Path to the binary file.
        meta_path (str): Path to the corresponding metadata file.
        data_type (str): Data type of the samples.

    Returns:
        numpy.ndarray: Data read from the file.
    """
    # Read metadata to get the number of channels
    meta = readMeta(meta_path)
    num_channels = int(meta['nSavedChans'])

    # Calculate the first dimension of the shape
    file_size = os.path.getsize(file_path)
    num_samples = file_size // (num_channels * np.dtype(data_type).itemsize)
    shape = (num_samples, num_channels)

    # Memory-map the binary file
    with open(file_path, 'rb') as f:
        data = np.memmap(f, dtype=data_type, mode='r', shape=shape)

    return data

def downsample_data(data, original_fs, target_fs):
    """
    Downsamples data to a target sampling rate.

    Args:
        data (numpy.ndarray): Data to downsample.
        original_fs (int): Original sampling rate.
        target_fs (int): Target sampling rate.

    Returns:
        numpy.ndarray: Downsampled data.
    """
    downsampling_factor = original_fs // target_fs
    downsampled_data = data[::downsampling_factor,:] #This is the fixed line
    return downsampled_data

def filter_data(data, fs, cutoff_freq):
    """
    Filters data using a sinc filter.

    Args:
        data (numpy.ndarray): Data to filter.
        fs (int): Sampling rate.
        cutoff_freq (int): Cutoff frequency for the filter.

    Returns:
        numpy.ndarray: Filtered data.
    """
    print("Shape of data entering filter_data:", data.shape)  # Debug print
    print("Data type of data entering filter_data:", data.dtype) # Debug print
    nyquist_freq = fs / 2
    normalized_cutoff_freq = cutoff_freq / nyquist_freq
    numtaps = 31  # Reduced filter order significantly
    taps = signal.firwin(numtaps=numtaps, cutoff=normalized_cutoff_freq)
    print("Shape of filter taps:", taps.shape) # Debug print
    filtered_data = signal.lfilter(taps, 1.0, data)
    print("Shape of filtered data exiting filter_data:", filtered_data.shape) # Debug print
    print("Data type of filtered data exiting filter_data:", filtered_data.dtype) # Debug print
    return filtered_data

def compute_spectrogram(data, fs, window_size, step_size):
    """
    Computes spectrogram of the data using scipy.signal.spectrogram.

    Args:
        data (numpy.ndarray): Data to compute spectrogram from.
        fs (int): Sampling rate.
        window_size (int): Size of the window in seconds.
        step_size (int): Step size in seconds.

    Returns:
        numpy.ndarray, numpy.ndarray, numpy.ndarray: Spectrogram, frequencies, and times.
    """
    window_samples = int(window_size * fs)
    step_samples = int(step_size * fs)
    win = signal.windows.hann(window_samples) # Use Hann window
    noverlap = window_samples - step_samples

    # Compute spectrogram using scipy.signal.spectrogram
    frequencies, times, spectrogram = signal.spectrogram(
        data,
        fs=fs,
        window=win,
        nperseg=window_samples,
        noverlap=noverlap,
        scaling='density',
        mode='psd' # Power Spectral Density
    )
    return spectrogram, frequencies, times

def compute_pca(data):
    """
    Performs Principal Component Analysis (PCA) on the z-scored data.

    Args:
        data (numpy.ndarray): Data to perform PCA on.

    Returns:
        numpy.ndarray: First principal component (1D array).
    """
    zscored_data = zscore(data, axis=0) # z-score along frequency axis (axis=0)
    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(zscored_data.T) # Transpose for PCA on frequency axis (rows), shape will be (n_times, 1)
    return pc1.squeeze() # Remove the trailing dimension to get (n_times,) shape

def compute_theta_dominance(spectrogram, frequencies):
    """
    Computes theta dominance from spectrogram.

    Args:
        spectrogram (numpy.ndarray): Spectrogram data.
        frequencies (numpy.ndarray): Frequencies corresponding to the spectrogram.

    Returns:
        numpy.ndarray: Theta dominance.
    """
    print("Shape of spectrogram inside compute_theta_dominance:", spectrogram.shape) # Debug
    theta_band_indices = np.where((frequencies >= 5) & (frequencies <= 10))[0] # Get indices
    theta_power = np.mean(spectrogram[theta_band_indices,:], axis=0)
    print("Shape of theta_power:", theta_power.shape) # Debug
    total_band_indices = np.where((frequencies >= 2) & (frequencies <= 16))[0] # Get indices
    total_power = np.mean(spectrogram[total_band_indices,:], axis=0)
    print("Shape of total_power:", total_power.shape) # Debug
    theta_dominance = theta_power / total_power
    print("Shape of theta_dominance:", theta_dominance.shape) # Debug
    return theta_dominance

def compute_emg(data, fs, window_size, step_size, n_ica_components=10, ica_epochs_duration=240, num_channels_for_ica=96): # Added num_channels_for_ica parameter
    """
    Estimates EMG from LFP data using Independent Component Analysis (ICA) with scikit-learn FastICA.
    Now allows specifying the number of channels to use for ICA.

    Args:
        data (numpy.ndarray): LFP data, shape (n_samples, n_channels).
        fs (float): Sampling frequency.
        window_size (float): Spectrogram window size in seconds.
        step_size (float): Spectrogram step size in seconds.
        n_ica_components (int): Number of ICA components to compute.
        ica_epochs_duration (int): Duration in seconds to use for ICA fitting.
        num_channels_for_ica (int): Number of channels to use for ICA. Defaults to 96.

    Returns:
        numpy.ndarray: Estimated EMG signal, segmented for spectrogram time windows.
    """
    print("\n---ICA-based EMG Computation (scikit-learn FastICA)---")

    # Determine channels to use for ICA based on num_channels_for_ica parameter
    actual_channels_available = data.shape[1]
    channels_to_use = min(num_channels_for_ica, actual_channels_available) # Use up to num_channels_for_ica, or fewer if data has less
    ica_channels_indices = range(channels_to_use)
    ica_data = data[:, ica_channels_indices]
    print(f"Applying ICA to channels: {ica_channels_indices}") # Corrected print statement for channel range

    # --- DEBUGGING CHANNEL SELECTION - ADD THESE LINES ---
    print(f"Shape of data going into ICA channel selection: {data.shape}")
    print(f"Shape of ica_data after channel selection: {ica_data.shape}")
    if ica_data.shape[1] > num_channels_for_ica: # Check against num_channels_for_ica parameter
            print(f"**WARNING: ICA data still has more channels than requested ({ica_data.shape[1]} > {num_channels_for_ica}). Channel selection may be failing!**")
    # --- DEBUGGING CHANNEL SELECTION - END ---

    # Use a shorter epoch for ICA fitting for speed
    ica_n_samples = int(ica_epochs_duration * fs)
    ica_data_subset = ica_data[:min(ica_n_samples, len(ica_data))]
    if len(ica_data_subset) < ica_n_samples:
        print(f"Warning: Data length shorter than requested ICA epoch duration ({ica_epochs_duration}s). Using available data for ICA.")

    # Apply bandpass filter before ICA (optional, but might help - adjust filter if needed)
    emg_cutoff_low_ica = 1 # High-pass
    emg_cutoff_high_ica = 150 # Low-pass
    nyquist_freq_ica = fs / 2
    normalized_cutoff_low_ica = emg_cutoff_low_ica / nyquist_freq_ica
    normalized_cutoff_high_ica = emg_cutoff_high_ica / nyquist_freq_ica
    numtaps_ica = 31
    ica_filter_taps = signal.firwin(numtaps_ica, [normalized_cutoff_low_ica, normalized_cutoff_high_ica],
                                    pass_zero='bandpass')
    ica_filtered_data_subset = signal.lfilter(ica_filter_taps, 1.0, ica_data_subset, axis=0)
    print(f"Bandpass filtering for ICA: {emg_cutoff_low_ica}-{emg_cutoff_high_ica} Hz") # Corrected print statement for filter frequencies

    # Run FastICA
    ica = FastICA(n_components=n_ica_components, random_state=42)
    print(f"Running FastICA with {n_ica_components} components on {channels_to_use} channels...") # Updated print statement
    ica_source_signals = ica.fit_transform(ica_filtered_data_subset)
    print("FastICA fitted and transformed.")
    print("Shape of ICA source signals:", ica_source_signals.shape)

    # --- IC Component Selection (VERY BASIC - NEEDS REFINEMENT) ---
    emg_ic_index = 0 # <--- Placeholder: Select the first IC
    print(f"Using IC {emg_ic_index} as estimated EMG.")
    emg_ic = ica_source_signals[:, emg_ic_index]

    # Reconstruct "EMG" signal from the selected IC
    reconstruction_matrix = np.zeros_like(ica_source_signals)
    reconstruction_matrix[:, emg_ic_index] = emg_ic
    emg_reconstructed_ica_data = ica.inverse_transform(reconstruction_matrix)
    emg_reconstructed = emg_reconstructed_ica_data[:, 0]
    print("Shape of reconstructed EMG (from ICA):", emg_reconstructed.shape)
    print("Range of reconstructed EMG (from ICA): Min=", np.min(emg_reconstructed), "Max=", np.max(emg_reconstructed))

    # --- Segment and Average Reconstructed EMG ---
    window_samples = int(window_size * fs)
    step_samples = int(step_size * fs)
    num_segments = (len(data) - window_samples) // step_samples + 1
    emg_segmented = np.zeros(num_segments)

    for i in range(num_segments):
        start_sample = i * step_samples
        end_sample = start_sample + window_samples
        if end_sample <= len(emg_reconstructed):
            emg_segmented[i] = np.mean(np.abs(emg_reconstructed[start_sample:end_sample]))

    print("Shape of segmented ICA-EMG:", emg_segmented.shape)
    print("Range of segmented ICA-EMG: Min=", np.min(emg_segmented), "Max=", np.max(emg_segmented))

    return emg_segmented, ica, ica_source_signals # Modified return to include ica and ica_source_signals

def score_sleep_states(pc1, theta_dominance, emg, buffer_size=10, fixed_emg_threshold=None, speed_csv_path=None, video_fps=30, video_start_offset_seconds=0, speed_threshold=1.5, use_emg=True):
    """
    Scores sleep states based on the provided metrics with soft sticky
    thresholds, now with optional fixed EMG threshold and speed threshold, and optional EMG usage.

    Args:
        pc1 (numpy.ndarray): First principal component.
        theta_dominance (numpy.ndarray): Theta dominance.
        emg (numpy.ndarray): EMG estimate.
        buffer_size (int): Size of the buffer zone in samples.
        fixed_emg_threshold (float, optional): Fixed EMG threshold value. Defaults to None (uses percentile-based threshold).
        speed_csv_path (str, optional): Path to CSV file containing speed data. Defaults to None (speed thresholding disabled).
        video_fps (int, optional): Frames per second of the video. Defaults to 30.
        video_start_offset_seconds (float, optional): Offset in seconds between video start and neural recording start. Defaults to 0.
        speed_threshold (float, optional): Threshold for speed in cm/s. Defaults to 1.5.
        use_emg (bool, optional): Flag to indicate whether to use EMG in sleep scoring. Defaults to True.

    Returns:
        numpy.ndarray: Sleep state labels.
    """
    nrem_threshold = np.percentile(pc1, 75)
    rem_threshold = np.percentile(theta_dominance, 75)
    emg_threshold = 0 # Initialize emg_threshold to 0, will be calculated or used only if use_emg is True

    if use_emg: # Only calculate or use EMG threshold if use_emg is True
        # Determine EMG threshold - use fixed if provided, otherwise percentile
        if fixed_emg_threshold is not None:
            emg_threshold = fixed_emg_threshold
            print(f"Using fixed EMG threshold: {emg_threshold:.4f}")
        else:
            emg_threshold = np.percentile(emg, 25)
            print(f"Using percentile-based EMG threshold: {emg_threshold:.4f}")
    else:
        print("EMG usage in sleep scoring is disabled.")

    print("---Threshold Values---") # Debug print for thresholds
    print(f"NREM Threshold: {nrem_threshold:.4f}")
    print(f"REM Threshold: {rem_threshold:.4f}")
    if use_emg:
        print(f"EMG Threshold: {emg_threshold:.4f}")
    if speed_csv_path:
        print(f"Speed Threshold: {speed_threshold:.4f} cm/s")
    print("----------------------")

    sleep_states = np.zeros_like(pc1, dtype=int)
    current_state = 0  # Initialize current state as Awake
    nrem_count = 0
    rem_count = 0
    awake_count = 0 # Initialize counters

    # Ensure theta_dominance and emg are at least as long as pc1 to prevent IndexError
    min_len = min(len(pc1), len(theta_dominance), len(emg)) # Find the minimum length
    theta_dominance = theta_dominance[:min_len] # Truncate arrays to the minimum length
    emg = emg[:min_len]
    pc1 = pc1[:min_len] # Truncate pc1 as well to ensure consistent length

    # Load and process speed data if speed_csv_path is provided
    speed_data_interp = None # Initialize to None
    if speed_csv_path:
        try:
            speed_df = pd.read_csv(speed_csv_path) # Assuming CSV has a 'speed' column
            speed_values = speed_df['speed'].values
            num_speed_frames = len(speed_values)
            video_sampling_rate = video_fps # FPS of video

            # Calculate timestamps for speed data and spectrogram data
            speed_time = np.arange(num_speed_frames) / video_sampling_rate + video_start_offset_seconds
            spectrogram_time = np.arange(len(pc1)) * step_size # step_size is in seconds

            # Interpolate speed data to match spectrogram time points
            speed_data_interp = np.interp(spectrogram_time, speed_time, speed_values, left=np.nan, right=np.nan)
            print(f"Speed data loaded and interpolated. Shape of interpolated speed data: {speed_data_interp.shape}")

        except Exception as e:
            print(f"Error loading or processing speed CSV: {e}. Speed thresholding will be disabled.")
            speed_data_interp = None # Ensure speed thresholding is disabled if error occurs

    for i in range(len(pc1)):
        print(f"\n---Time Point: {i}---") # Debug print for each time point
        print(f"Current State: {current_state}")
        print(f"PC1: {pc1[i]:.4f}, Theta Dominance: {theta_dominance[i]:.4f}")
        if use_emg:
            print(f"EMG: {emg[i]:.4f}")
        speed_val = speed_data_interp[i] if speed_data_interp is not None and i < len(speed_data_interp) else np.nan # Get speed if available, otherwise NaN
        if not np.isnan(speed_val):
            print(f"Speed: {speed_val:.4f} cm/s")

        is_speed_high = False # Initialize speed condition as false
        if speed_data_interp is not None and not np.isnan(speed_val): # Check if speed data is available and not NaN for this time point
            is_speed_high = speed_val > speed_threshold

        if current_state == 0:  # Awake
            print("State: Awake")
            awake_by_speed = False # Initialize speed condition for awake state

            if is_speed_high:
                awake_by_speed = True
                print(f"  Condition Awake by Speed: Speed > Speed Thresh ({speed_val:.4f} > {speed_threshold:.4f})")

            nrem_condition = (pc1[i] > nrem_threshold) and not awake_by_speed # Initialize NREM condition
            rem_condition = (theta_dominance[i] > rem_threshold) and not awake_by_speed # Initialize REM condition

            if use_emg: # If using EMG, add EMG condition to NREM/REM checks
                nrem_condition = nrem_condition and (emg[i] < emg_threshold)
                rem_condition = rem_condition and (emg[i] < emg_threshold)

            if nrem_condition:
                nrem_count += 1
                condition_desc = f"PC1 > NREM Thresh ({pc1[i]:.4f} > {nrem_threshold:.4f})"
                if use_emg: condition_desc += f" and EMG < EMG Thresh ({emg[i]:.4f} < {emg_threshold:.4f})"
                print(f"  Condition NREM met: {condition_desc}")
            else:
                nrem_count = 0
                print("  Condition NREM not met")

            if rem_condition:
                rem_count += 1
                condition_desc = f"Theta > REM Thresh ({theta_dominance[i]:.4f} > {rem_threshold:.4f})"
                if use_emg: condition_desc += f" and EMG < EMG Thresh ({emg[i]:.4f} < {emg_threshold:.4f})"
                print(f"  Condition REM met: {condition_desc}")
            else:
                rem_count = 0
                print("  Condition REM not met")

            print(f"  NREM Count: {nrem_count}, REM Count: {rem_count}")

            if nrem_count >= buffer_size:
                current_state = 1  # Switch to NREM
                print("  Transition to NREM (State 1)")
            elif rem_count >= buffer_size:
                current_state = 2  # Switch to REM
                print("  Transition to REM (State 2)")

        elif current_state == 1:  # NREM
            print("State: NREM")
            awake_condition = (pc1[i] < nrem_threshold) or is_speed_high # Initialize awake condition for NREM exit
            if use_emg: # If using EMG, add EMG condition to awake check for NREM exit
                awake_condition = awake_condition or (emg[i] > emg_threshold)

            if awake_condition:
                awake_count += 1
                condition_str = "  Condition Awake met (NREM exit): "
                conditions_met = []
                if (pc1[i] < nrem_threshold): conditions_met.append(f"PC1 < NREM Thresh ({pc1[i]:.4f} < {nrem_threshold:.4f})")
                if use_emg and (emg[i] > emg_threshold): conditions_met.append(f"EMG > EMG Thresh ({emg[i]:.4f} > {emg_threshold:.4f})")
                if is_speed_high: conditions_met.append(f"Speed > Speed Thresh ({speed_val:.4f} > {speed_threshold:.4f})")
                print(condition_str + " or ".join(conditions_met))

            else:
                awake_count = 0
                print("  Condition Awake not met (NREM sustain)")

            print(f"  Awake Count (NREM exit): {awake_count}")
            if awake_count >= buffer_size:
                current_state = 0  # Switch to Awake
                print("  Transition to Awake (State 0) from NREM")

        elif current_state == 2:  # REM
            print("State: REM")
            awake_condition = (theta_dominance[i] < rem_threshold) or is_speed_high # Initialize awake condition for REM exit
            if use_emg: # If using EMG, add EMG condition to awake check for REM exit
                awake_condition = awake_condition or (emg[i] > emg_threshold)

            if awake_condition:
                awake_count += 1
                condition_str = "  Condition Awake met (REM exit): "
                conditions_met = []
                if (theta_dominance[i] < rem_threshold): conditions_met.append(f"Theta < REM Thresh ({theta_dominance[i]:.4f} < {rem_threshold:.4f})")
                if use_emg and (emg[i] > emg_threshold): conditions_met.append(f"EMG > EMG Thresh ({emg[i]:.4f} > {emg_threshold:.4f})")
                if is_speed_high: conditions_met.append(f"Speed > Speed Thresh ({speed_val:.4f} > {speed_threshold:.4f})")
                print(condition_str + " or ".join(conditions_met))
            else:
                awake_count = 0
                print("  Condition Awake not met (REM sustain)")
            print(f"  Awake Count (REM exit): {awake_count}")
            if awake_count >= buffer_size:
                current_state = 0  # Switch to Awake
                print("  Transition to Awake (State 0) from REM")
        sleep_states[i] = current_state

    return sleep_states

# Parameters
original_fs = 2500
target_fs = 1250
cutoff_freq = 450
window_size = 10 # Changed to 10s window as per description
step_size = 1 # Changed to 1s step as per description
num_channels = 48
n_ica_components = 10
ica_epochs_duration = 240
num_channels_for_ica = 48

# --- Optional Threshold Parameters ---
calculate_emg = False # Set to False to disable EMG calculation and usage in scoring
fixed_emg_threshold_val = None # Set to a float value to use fixed threshold, or None for percentile-based
speed_csv_file_path = None # Set to path to speed CSV file to enable speed thresholding, or None to disable
video_frame_rate = 30 # FPS of your video recording
video_start_delay_seconds = 0 # Offset in seconds between video and neural recording start
mouse_speed_threshold = 1.5 # cm/s, threshold for high speed

# Get file path from user using a browser
# Get file path from user using a browser window
root = Tk()
root.withdraw()
root.attributes("-topmost", True)
lfp_file_path_str = filedialog.askopenfilename(title="Select LFP binary file") # Renamed to avoid conflict with 'file_path' inside loop
lfp_file_path = Path(lfp_file_path_str) if lfp_file_path_str else None # Handle case where user cancels file selection

speed_csv_file_path_str = filedialog.askopenfilename(title="Select Speed CSV file (optional)", filetypes=[("CSV files", "*.csv")]) if speed_csv_file_path is None else speed_csv_file_path # Only open dialog if speed_csv_file_path is None from parameters
speed_csv_file_path = Path(speed_csv_file_path_str) if speed_csv_file_path_str else None # Handle cancel and parameter path

root.destroy()

if lfp_file_path is None:
    print("No LFP binary file selected. Exiting.")
    sys.exit()

# Load data
data = read_bin_data(lfp_file_path, lfp_file_path.with_suffix('.meta'))

# Select last num_channels channels
data = data[:, -num_channels:]

print("Shape of data before filtering:", data.shape) # Debug print
print("Data type of data before filtering:", data.dtype) # Debug print

# Filter and downsample data
filtered_data = filter_data(data, original_fs, cutoff_freq)
downsampled_data = downsample_data(filtered_data, original_fs, target_fs)

# Average across channels to get a 1D signal (time series)
averaged_lfp_data = np.mean(downsampled_data, axis=1)

print("Shape of averaged_lfp_data before spectrogram:", averaged_lfp_data.shape) # Debug print

# Compute spectrogram
spectrogram, frequencies, times = compute_spectrogram(
    averaged_lfp_data, target_fs, window_size, step_size
)
if spectrogram is None:
    print("Spectrogram computation failed...")
    sys.exit()
# Compute PCA after z-scoring
pc1 = compute_pca(spectrogram)

# Compute theta dominance
theta_dominance = compute_theta_dominance(spectrogram, frequencies)

# Compute EMG conditionally
if calculate_emg:
    emg, ica_object, ica_components = compute_emg(
        downsampled_data, target_fs, window_size, step_size, n_ica_components=n_ica_components,
        num_channels_for_ica=num_channels_for_ica, ica_epochs_duration=ica_epochs_duration) #Added window/step size and num_channels_for_ica
else:
    emg_shape = len(pc1) # EMG needs to have the same length as other metrics for scoring
    emg = np.zeros(emg_shape) # Create a zero-filled EMG if not calculated
    ica_object, ica_components = None, None # Set ICA related variables to None as they are not computed

print("Shape of pc1 before score_sleep_states:", pc1.shape)
print("Shape of theta_dominance before score_sleep_states:", theta_dominance.shape)
print("Shape of emg before score_sleep_states:", emg.shape)

print("\n---Ranges of Metrics---") # Add this section
print(f"PC1 Range:  Min={np.min(pc1):.4f}, Max={np.max(pc1):.4f}")
print(f"Theta Range: Min={np.min(theta_dominance):.4f}, Max={np.max(theta_dominance):.4f}")
print(f"EMG Range:   Min={np.min(emg):.4f}, Max={np.max(emg):.4f}")
print("-----------------------")

# Score sleep states, passing optional thresholds and EMG usage flag
sleep_states = score_sleep_states(pc1, theta_dominance, emg,
                                    fixed_emg_threshold=fixed_emg_threshold_val, speed_csv_path=speed_csv_file_path,
                                    video_fps=video_frame_rate, video_start_offset_seconds=video_start_delay_seconds,
                                    speed_threshold=mouse_speed_threshold, use_emg=calculate_emg)

# Save sleep states in same directory as LFP bin file
output_directory = lfp_file_path.parent
sleep_states_save_path = output_directory / 'sleep_states.npy'
np.save(sleep_states_save_path, sleep_states)

# --- Visualization of Independent Components - Conditionally Plot if EMG was calculated ---
if calculate_emg and ica_components is not None: # Only plot if ICA was actually run and components exist
    num_ics_to_plot = 10  # Or less if you used fewer components

    plt.figure(figsize=(12, 8))
    for i in range(num_ics_to_plot):
        plt.subplot(num_ics_to_plot, 1, i + 1)
        plt.plot(ica_components[:, i]) # Time series of IC i
        plt.title(f"IC {i}")
        plt.ylabel("Amplitude")
        if i == num_ics_to_plot - 1:
            plt.xlabel("Samples")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 8))
    for i in range(num_ics_to_plot):
        plt.subplot(num_ics_to_plot, 1, i + 1)
        frequencies_psd, power_spectrum = signal.welch(ica_components[:, i], fs=target_fs*2, nperseg=256) # Example Welch PSD, using target_fs*2 as ICA is done before downsampling.
        plt.plot(frequencies_psd, power_spectrum)
        plt.title(f"IC {i} - Power Spectrum")
        plt.ylabel("Power Spectral Density")
        plt.xlim([0, 200]) # Focus on lower frequencies, adjust as needed
        if i == num_ics_to_plot - 1:
            plt.xlabel("Frequency (Hz)")
    plt.tight_layout()
    plt.show()
else:
    print("ICA-EMG calculation and IC plots were skipped because 'calculate_emg' was set to False.")

sys.stdout.close()
sys.stderr.close()
sys.stdout = original_stdout
sys.stderr = original_stderr
print(f"Script output redirected to: {output_file_path}")
print(f"Sleep states saved to: {sleep_states_save_path}")
print("Script execution completed. Please check script_output.txt and sleep_states.npy (and IC plots if EMG was calculated).")