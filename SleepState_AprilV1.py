# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 2025
State scoring based on Supplementary Materials 
Wannan Yang Science 383,1478(2024)_Buzsaki Lab sleep state score
@author: Alok

Applies downsampling( to 1250hz) BEFORE filtering to handle large files and avoid MemoryErrors.
Uses scipy.signal.decimate for proper anti-aliased downsampling.
"""
import numpy as np
from scipy import signal # Make sure signal is imported
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
import math 

# Redirect stdout and stderr to a file
output_file_path = "script_output.txt"  # You can change the filename if you like
original_stdout = sys.stdout
original_stderr = sys.stderr
try:
    # Ensure the directory for the output file exists if it's not the current directory
    # os.makedirs(os.path.dirname(output_file_path), exist_ok=True) # Uncomment if needed
    sys.stdout = open(output_file_path, 'w', encoding='utf-8') 
    sys.stderr = sys.stdout
except Exception as e:
    print(f"Error redirecting output: {e}")
    # Fallback to original stdout/stderr if redirection fails
    sys.stdout = original_stdout
    sys.stderr = original_stderr

def read_bin_data(file_path, meta_path, data_type='int16'):
    """
    Reads binary data from a file using memory-mapping.
    Args:
        file_path (str): Path to the binary file.
        meta_path (str): Path to the corresponding metadata file.
        data_type (str): Data type of the samples.
    Returns:
        numpy.ndarray: Memory-mapped data read from the file.
    """
    try:
        # Read metadata to get the number of channels
        meta = readMeta(meta_path)
        num_channels = int(meta['nSavedChans'])
        print(f"Reading metadata: {num_channels} saved channels found.")

        # Calculate the first dimension of the shape
        file_size = os.path.getsize(file_path)
        item_size = np.dtype(data_type).itemsize
        if file_size % (num_channels * item_size) != 0:
             print(f"Warning: File size {file_size} is not an exact multiple of {num_channels} channels * {item_size} bytes/sample.")
             print("This might indicate a corrupted file or incorrect metadata.")
             # Adjust num_samples calculation to handle potential truncation
             num_samples = file_size // (num_channels * item_size)
             print(f"Calculated samples based on integer division: {num_samples}")
        else:
            num_samples = file_size // (num_channels * item_size)
            print(f"Calculated samples: {num_samples}")

        shape = (num_samples, num_channels)
        print(f"Expected data shape: {shape}")

        # Memory-map the binary file
        # Use 'r' mode for read-only access, prevents accidental modification
        data = np.memmap(file_path, dtype=data_type, mode='r', shape=shape)
        print(f"Successfully memory-mapped file: {file_path}")
        return data
    except FileNotFoundError:
        print(f"Error: File not found - {file_path} or {meta_path}")
        return None
    except KeyError as e:
        print(f"Error: Metadata key missing in {meta_path} - {e}")
        return None
    except ValueError as e:
         print(f"Error: Problem with shape or dtype during memmap for {file_path} - {e}")
         return None
    except Exception as e:
        print(f"An unexpected error occurred in read_bin_data for {file_path}: {e}")
        return None

def filter_data(data, fs, cutoff_freq):
    """
    Filters data using a sinc FIR filter (using scipy.signal.firwin and lfilter).
    Args:
        data (numpy.ndarray): Data to filter (should be float type).
        fs (int): Sampling rate OF THE INPUT DATA.
        cutoff_freq (int): Cutoff frequency for the low-pass filter.
    Returns:
        numpy.ndarray: Filtered data (typically float64).
    """
    print(f"Shape of data entering filter_data: {data.shape}")
    print(f"Data type entering filter_data: {data.dtype}")
    print(f"Filtering with fs={fs} Hz, cutoff={cutoff_freq} Hz")

    if not np.issubdtype(data.dtype, np.floating):
        print("Warning: Data type is not float, converting to float32 for filtering.")
        data = data.astype(np.float32)

    nyquist_freq = fs / 2.0
    if cutoff_freq >= nyquist_freq:
        print(f"Warning: Cutoff frequency ({cutoff_freq} Hz) is >= Nyquist frequency ({nyquist_freq} Hz). Filtering will likely have no effect or be unstable.")
        # Option 1: Skip filtering
        # return data
        # Option 2: Adjust cutoff (e.g., slightly below Nyquist)
        cutoff_freq = nyquist_freq * 0.99
        print(f"Adjusted cutoff frequency to {cutoff_freq:.2f} Hz")

    normalized_cutoff_freq = cutoff_freq / nyquist_freq
    numtaps = 31  # Reduced filter order significantly
    # Ensure numtaps is odd for Type I FIR filter (zero freq gain = 1)
    if numtaps % 2 == 0:
        numtaps += 1

    try:
        taps = signal.firwin(numtaps=numtaps, cutoff=normalized_cutoff_freq, window='hamming', pass_zero='lowpass')
        print(f"Shape of filter taps: {taps.shape}")

        # Apply filter channel by channel (axis=0 because time is the first axis)
        # lfilter is generally faster for FIR than filtfilt but introduces phase delay
        # Using float32 for taps might save some memory during convolution step
        filtered_data = signal.lfilter(taps.astype(np.float32), 1.0, data, axis=0)

        print(f"Shape of filtered data exiting filter_data: {filtered_data.shape}")
        print(f"Data type of filtered data exiting filter_data: {filtered_data.dtype}") # Often float64
        return filtered_data
    except Exception as e:
        print(f"Error during filtering: {e}")
        return None # Return None or raise error to indicate failure

def compute_spectrogram(data, fs, window_size, step_size):
    """
    Computes spectrogram of the data using scipy.signal.spectrogram.
    Args:
        data (numpy.ndarray): 1D data array to compute spectrogram from.
        fs (int): Sampling rate.
        window_size (int): Size of the window in seconds.
        step_size (int): Step size in seconds.
    Returns:
        numpy.ndarray, numpy.ndarray, numpy.ndarray: Spectrogram, frequencies, and times or (None, None, None) on error.
    """
    print(f"Computing spectrogram: fs={fs}, window={window_size}s, step={step_size}s")
    print(f"Input data shape for spectrogram: {data.shape}")
    if data.ndim != 1:
        print("Error: Spectrogram input data must be 1D.")
        return None, None, None
    if len(data) == 0:
         print("Error: Spectrogram input data is empty.")
         return None, None, None

    try:
        window_samples = int(window_size * fs)
        step_samples = int(step_size * fs)

        if window_samples > len(data):
             print(f"Warning: Window size ({window_samples} samples) > data length ({len(data)} samples). Adjusting window size.")
             window_samples = len(data) // 2 # Adjust to something smaller, e.g., half the data length
             if window_samples == 0:
                  print("Error: Data too short for any window.")
                  return None, None, None

        # Ensure overlap is valid
        noverlap = window_samples - step_samples
        if noverlap < 0:
            print(f"Warning: Step size ({step_samples}) > window size ({window_samples}). Setting overlap to 0.")
            noverlap = 0
        elif noverlap >= window_samples:
             print(f"Warning: Overlap ({noverlap}) >= window size ({window_samples}). This implies step_size <= 0. Setting overlap to window_size - 1.")
             noverlap = window_samples - 1 # Ensure at least 1 sample step

        print(f"Spectrogram params: nperseg={window_samples}, noverlap={noverlap}")

        win = signal.windows.hann(window_samples) # Use Hann window

        # Compute spectrogram using scipy.signal.spectrogram
        frequencies, times, spectrogram = signal.spectrogram(
            data.astype(np.float64), # Spectrogram often works better with float64
            fs=fs,
            window=win,
            nperseg=window_samples,
            noverlap=noverlap,
            scaling='density',
            mode='psd' # Power Spectral Density
        )
        print(f"Spectrogram computed. Shape: {spectrogram.shape}, Freq shape: {frequencies.shape}, Time shape: {times.shape}")
        return spectrogram, frequencies, times
    except Exception as e:
        print(f"Error computing spectrogram: {e}")
        return None, None, None

def compute_pca(data):
    """
    Performs Principal Component Analysis (PCA) on the z-scored data (spectrogram).
    Args:
        data (numpy.ndarray): Spectrogram data (frequencies x times).
    Returns:
        numpy.ndarray: First principal component (1D array, length = n_times) or None on error.
    """
    print(f"Computing PCA on data of shape: {data.shape}")
    if data is None or data.size == 0 or data.shape[0] < 2 or data.shape[1] < 2:
        print("Error: Invalid data for PCA (empty, too small, or None).")
        return None
    try:
        # Z-score along frequency axis (axis=0). Transpose needed if frequencies are rows.
        # Check for NaNs/Infs before z-scoring
        if not np.all(np.isfinite(data)):
             print("Warning: Non-finite values (NaN/Inf) found in data before PCA. Replacing with zeros.")
             data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0) # Or use imputation

        # Avoid z-scoring if variance is zero along the axis
        if np.any(np.std(data, axis=0) == 0):
             print("Warning: Zero variance found in some frequency bins. PCA might be unstable.")
             # Handle appropriately, e.g., add small noise or skip those features?
             # For now, proceed, but be aware. Z-score will produce NaNs here if not handled.
             zscored_data = zscore(data, axis=0, nan_policy='omit') # 'omit' might not be ideal
             zscored_data = np.nan_to_num(zscored_data) # Replace NaNs resulting from zero std dev
        else:
             zscored_data = zscore(data, axis=0) # z-score along frequency axis (axis=0)

        pca = PCA(n_components=1)
        # PCA expects samples as rows, features as columns.
        # Spectrogram is (freq, time). We want PCA across frequencies for each time point.
        # So, transpose it to (time, freq) before fitting.
        pc1 = pca.fit_transform(zscored_data.T) # Transpose for PCA, shape will be (n_times, 1)
        print(f"PCA computed. PC1 shape: {pc1.shape}")
        return pc1.squeeze() # Remove the trailing dimension to get (n_times,) shape
    except Exception as e:
        print(f"Error computing PCA: {e}")
        return None

def compute_theta_dominance(spectrogram, frequencies):
    """
    Computes theta dominance from spectrogram.
    Args:
        spectrogram (numpy.ndarray): Spectrogram data (frequencies x times).
        frequencies (numpy.ndarray): Frequencies corresponding to the spectrogram.
    Returns:
        numpy.ndarray: Theta dominance (1D array, length = n_times) or None on error.
    """
    print(f"Computing Theta Dominance. Spectrogram shape: {spectrogram.shape}, Freq shape: {frequencies.shape}")
    if spectrogram is None or frequencies is None or spectrogram.size == 0 or frequencies.size == 0:
        print("Error: Invalid input for theta dominance calculation.")
        return None
    try:
        theta_band_indices = np.where((frequencies >= 5) & (frequencies <= 10))[0]
        total_band_indices = np.where((frequencies >= 2) & (frequencies <= 16))[0]

        if len(theta_band_indices) == 0 or len(total_band_indices) == 0:
             print("Warning: Could not find frequency bins for Theta (5-10Hz) or Total (2-16Hz). Check frequencies.")
             return np.zeros(spectrogram.shape[1]) # Return zeros or handle differently

        # Calculate mean power, ensure axis=0 (mean across frequency rows)
        # Add a small epsilon to avoid division by zero if total power is zero
        epsilon = 1e-9
        theta_power = np.mean(spectrogram[theta_band_indices,:], axis=0)
        total_power = np.mean(spectrogram[total_band_indices,:], axis=0) + epsilon

        print(f"Shape of theta_power: {theta_power.shape}")
        print(f"Shape of total_power: {total_power.shape}")

        theta_dominance = theta_power / total_power
        print(f"Shape of theta_dominance: {theta_dominance.shape}")

        # Check for NaNs/Infs resulting from calculation
        if not np.all(np.isfinite(theta_dominance)):
             print("Warning: Non-finite values found in theta dominance. Clamping or replacing.")
             theta_dominance = np.nan_to_num(theta_dominance, nan=0.0, posinf=1.0, neginf=0.0) # Example handling

        return theta_dominance
    except Exception as e:
        print(f"Error computing Theta Dominance: {e}")
        return None

def compute_emg(data, fs, window_size, step_size, n_ica_components=10, ica_epochs_duration=240, num_channels_for_ica=48):
    """
    Estimates EMG from LFP data using Independent Component Analysis (ICA) with scikit-learn FastICA.
    Now allows specifying the number of channels to use for ICA.
    Args:
        data (numpy.ndarray): LFP data, shape (n_samples, n_channels). Should be downsampled & filtered.
        fs (float): Sampling frequency of the input data.
        window_size (float): Spectrogram window size in seconds (used for segmenting EMG).
        step_size (float): Spectrogram step size in seconds (used for segmenting EMG).
        n_ica_components (int): Number of ICA components to compute.
        ica_epochs_duration (int): Duration in seconds of data subset to use for ICA fitting.
        num_channels_for_ica (int): Number of channels to use for ICA.
    Returns:
        tuple: (Estimated EMG signal (segmented), ICA object, ICA source signals) or (None, None, None) on error.
    """
    print("\n---ICA-based EMG Computation (scikit-learn FastICA)---")
    print(f"Input data shape for EMG: {data.shape}, fs={fs}")

    if data is None or data.size == 0:
        print("Error: Input data for EMG computation is invalid.")
        return None, None, None

    try:
        actual_channels_available = data.shape[1]
        if actual_channels_available == 0:
             print("Error: Input data for EMG has 0 channels.")
             return None, None, None

        channels_to_use = min(num_channels_for_ica, actual_channels_available)
        # Ensure n_components is not more than the number of channels used
        n_ica_components = min(n_ica_components, channels_to_use)

        if n_ica_components <= 0:
             print("Error: Number of ICA components must be positive.")
             return None, None, None

        ica_channels_indices = np.arange(channels_to_use) # Use first N channels
        # Or select channels differently, e.g., based on variance or specific indices
        # ica_channels_indices = range(channels_to_use) # Original way
        ica_data = data[:, ica_channels_indices]
        print(f"Applying ICA to first {channels_to_use} channels.")
        print(f"Shape of data selected for ICA: {ica_data.shape}")

        # Use a shorter epoch for ICA fitting for speed
        ica_n_samples = int(ica_epochs_duration * fs)
        ica_n_samples = min(ica_n_samples, ica_data.shape[0]) # Don't exceed available data length

        if ica_n_samples <= 0:
             print("Error: Not enough data samples for ICA fitting based on duration and fs.")
             return None, None, None

        ica_data_subset = ica_data[:ica_n_samples, :]
        print(f"Using {ica_n_samples} samples ({ica_data_subset.shape[0] / fs:.2f} seconds) for ICA fitting.")

        # Optional: Apply bandpass filter *before* ICA (on the subset)
        # This filter is *different* from the main LFP filter
        emg_cutoff_low_ica = 30  # Adjust these based on expected EMG frequency range
        emg_cutoff_high_ica = 300
        nyquist_freq_ica = fs / 2.0
        normalized_cutoff_low_ica = emg_cutoff_low_ica / nyquist_freq_ica
        normalized_cutoff_high_ica = emg_cutoff_high_ica / nyquist_freq_ica

        # Ensure frequency range is valid
        if normalized_cutoff_low_ica >= normalized_cutoff_high_ica or normalized_cutoff_high_ica >= 1.0:
            print("Warning: Invalid EMG frequency band for ICA pre-filter. Skipping filter.")
            ica_filtered_data_subset = ica_data_subset.astype(np.float32) # Ensure float
        else:
            numtaps_ica = 31 # Keep filter order reasonable
            if numtaps_ica % 2 == 0: numtaps_ica += 1
            print(f"Bandpass filtering ICA subset: {emg_cutoff_low_ica}-{emg_cutoff_high_ica} Hz")
            ica_filter_taps = signal.firwin(numtaps_ica, [normalized_cutoff_low_ica, normalized_cutoff_high_ica],
                                            pass_zero='bandpass', window='hamming')
            # Apply filter, ensure float32
            ica_filtered_data_subset = signal.lfilter(ica_filter_taps.astype(np.float32), 1.0, ica_data_subset.astype(np.float32), axis=0)

        # Run FastICA
        # Whiten the data before ICA (FastICA does this by default if whiten=True)
        ica = FastICA(n_components=n_ica_components, random_state=42, whiten='unit-variance', max_iter=500, tol=1e-3) # Adjusted params
        print(f"Running FastICA with {n_ica_components} components on {channels_to_use} channels...")
        # FastICA expects samples x features (time x channels), which is correct here
        ica_source_signals = ica.fit_transform(ica_filtered_data_subset) # Fit on the filtered subset
        print("FastICA fitted and transformed subset.")
        print("Shape of ICA source signals (subset):", ica_source_signals.shape)

        # --- IC Component Selection (VERY BASIC - NEEDS IMPROVEMENT) ---
        # This part is crucial and often requires manual inspection or a more sophisticated metric.
        # Example: Select component with highest power in the EMG band (e.g., > 100 Hz)
        # For now, we stick to the placeholder:
        emg_ic_index = 0 # <--- Placeholder: Select the first IC. **REPLACE WITH BETTER LOGIC**
        print(f"**Placeholder Warning**: Using IC {emg_ic_index} as estimated EMG. Selection logic needed!")

        # --- Apply the learned ICA transformation to the *entire* dataset ---
        # First, filter the *entire* dataset with the same bandpass filter used for ICA fitting
        if normalized_cutoff_low_ica < normalized_cutoff_high_ica and normalized_cutoff_high_ica < 1.0:
             print("Applying learned ICA transform to full dataset (after bandpass)...")
             ica_filtered_data_full = signal.lfilter(ica_filter_taps.astype(np.float32), 1.0, ica_data.astype(np.float32), axis=0)
             full_source_signals = ica.transform(ica_filtered_data_full) # Use transform, not fit_transform
        else:
             print("Applying learned ICA transform to full dataset (unfiltered)...")
             full_source_signals = ica.transform(ica_data.astype(np.float32)) # Apply to unfiltered if pre-filter was skipped


        print("Shape of full source signals:", full_source_signals.shape)
        emg_ic_full = full_source_signals[:, emg_ic_index]

        # The "EMG signal" is often taken as the absolute value or power of the selected component
        emg_signal_raw = np.abs(emg_ic_full)
        # Alternatively, smooth the power:
        # emg_signal_raw = signal.convolve(emg_ic_full**2, np.ones(int(fs*0.1))/int(fs*0.1), mode='same') # Example smoothing

        print("Shape of raw EMG signal (from selected IC):", emg_signal_raw.shape)
        print(f"Range of raw EMG: Min={np.min(emg_signal_raw):.4f}, Max={np.max(emg_signal_raw):.4f}")


        # --- Segment and Average EMG to match spectrogram time bins ---
        window_samples = int(window_size * fs)
        step_samples = int(step_size * fs)
        # Calculate number of segments based on the length of the raw EMG signal
        num_segments = (len(emg_signal_raw) - window_samples) // step_samples + 1

        if num_segments <= 0:
             print("Error: Not enough EMG data to create segments based on window/step size.")
             return None, ica, ica_source_signals # Return fitted objects maybe?

        emg_segmented = np.zeros(num_segments)
        print(f"Segmenting EMG into {num_segments} segments...")

        for i in range(num_segments):
            start_sample = i * step_samples
            end_sample = start_sample + window_samples
            # Ensure indices are within bounds of emg_signal_raw
            if end_sample > len(emg_signal_raw):
                # This shouldn't happen with the calculation above, but check just in case
                print(f"Warning: Segment {i} exceeds data length. Adjusting end sample.")
                end_sample = len(emg_signal_raw)
                if start_sample >= end_sample: continue # Skip if segment is empty

            segment = emg_signal_raw[start_sample:end_sample]
            if segment.size > 0:
                 emg_segmented[i] = np.mean(segment) # Use mean of absolute value

        print("Shape of segmented ICA-EMG:", emg_segmented.shape)
        print(f"Range of segmented ICA-EMG: Min={np.min(emg_segmented):.4f}, Max={np.max(emg_segmented):.4f}")

        return emg_segmented, ica, full_source_signals # Return segmented EMG and full source signals

    except Exception as e:
        print(f"An error occurred during EMG computation: {e}")
        # Print traceback for debugging
        import traceback
        traceback.print_exc()
        return None, None, None

def score_sleep_states(pc1, theta_dominance, emg, times, # Added times array
                       step_size, # Need step size to map times to indices
                       buffer_size=10, fixed_emg_threshold=None,
                       speed_csv_path=None, video_fps=30, video_start_offset_seconds=0,
                       speed_threshold=1.5, use_emg=False):
    """
    Scores sleep states based on the provided metrics with sticky thresholds.
    Includes optional fixed EMG threshold, speed thresholding, and optional EMG usage.
    Args:
        pc1 (numpy.ndarray): First principal component (len = n_times).
        theta_dominance (numpy.ndarray): Theta dominance (len = n_times).
        emg (numpy.ndarray): EMG estimate (segmented, len = n_times).
        times (numpy.ndarray): Timestamps corresponding to pc1, theta, emg (from spectrogram).
        step_size (float): Step size used for spectrogram/segmentation (in seconds).
        buffer_size (int): Number of consecutive samples needed to switch state.
        fixed_emg_threshold (float, optional): Fixed EMG threshold. Defaults to None (percentile-based).
        speed_csv_path (str, optional): Path to speed CSV file. Defaults to None.
        video_fps (int, optional): Video FPS. Defaults to 30.
        video_start_offset_seconds (float, optional): Video start offset vs neural data. Defaults to 0.
        speed_threshold (float, optional): Speed threshold (cm/s). Defaults to 1.5.
        use_emg (bool, optional): Whether to use EMG in scoring. Defaults to True.
    Returns:
        numpy.ndarray: Sleep state labels (0=Awake, 1=NREM, 2=REM) or None on error.
    """
    print("\n--- Scoring Sleep States ---")
    # Validate inputs
    if pc1 is None or theta_dominance is None or emg is None or times is None:
        print("Error: One or more input metrics (PC1, Theta, EMG, Times) are None. Cannot score.")
        return None
    if not (len(pc1) == len(theta_dominance) == len(emg) == len(times)):
        print(f"Error: Input metrics have inconsistent lengths: PC1={len(pc1)}, Theta={len(theta_dominance)}, EMG={len(emg)}, Times={len(times)}")
        return None
    if len(pc1) == 0:
         print("Error: Input metrics are empty.")
         return None

    num_time_points = len(pc1)
    print(f"Scoring based on {num_time_points} time points.")

    # Calculate thresholds
    try:
        # Use nanpercentile to handle potential NaNs if PCA/Theta failed partially
        nrem_threshold = np.nanpercentile(pc1, 75)
        rem_threshold = np.nanpercentile(theta_dominance, 75)
    except Exception as e:
        print(f"Error calculating NREM/REM thresholds: {e}. Check PC1/Theta data.")
        return None

    emg_threshold = 0 # Initialize
    if use_emg:
        try:
            if fixed_emg_threshold is not None:
                emg_threshold = fixed_emg_threshold
                print(f"Using fixed EMG threshold: {emg_threshold:.4f}")
            elif len(emg) > 0: # Check if EMG array is not empty
                 # Check for non-finite values before percentile calculation
                 if not np.all(np.isfinite(emg)):
                      print("Warning: Non-finite values found in EMG. Using nanpercentile.")
                      emg_threshold = np.nanpercentile(emg, 25)
                 else:
                      emg_threshold = np.percentile(emg, 25)
                 print(f"Using percentile-based EMG threshold (25th percentile): {emg_threshold:.4f}")
            else:
                 print("Warning: EMG array is empty, cannot calculate percentile threshold. EMG threshold set to 0.")
                 emg_threshold = 0 # Fallback EMG threshold if array is empty
        except Exception as e:
            print(f"Error calculating EMG threshold: {e}. EMG threshold set to 0.")
            emg_threshold = 0 # Fallback
    else:
        print("EMG usage in sleep scoring is disabled.")

    print("---Threshold Values---")
    print(f"NREM Threshold (PC1 >): {nrem_threshold:.4f}")
    print(f"REM Threshold (Theta >): {rem_threshold:.4f}")
    if use_emg: print(f"EMG Threshold (< for NREM/REM): {emg_threshold:.4f}")


    # Load and process speed data
    speed_data_interp = None
    if speed_csv_path:
        print(f"Loading speed data from: {speed_csv_path}")
        try:
            speed_df = pd.read_csv(speed_csv_path)
            if 'speed' not in speed_df.columns:
                print(f"Error: 'speed' column not found in {speed_csv_path}. Disabling speed thresholding.")
            elif speed_df['speed'].isnull().all():
                 print(f"Error: 'speed' column contains only NaN values in {speed_csv_path}. Disabling speed thresholding.")
            else:
                speed_values = speed_df['speed'].values
                num_speed_frames = len(speed_values)
                video_sampling_rate = video_fps

                # Timestamps for speed data (relative to video start + offset)
                speed_time = np.arange(num_speed_frames) / video_sampling_rate + video_start_offset_seconds
                # Timestamps for spectrogram data (already provided as 'times' argument)
                spectrogram_time = times

                # Interpolate speed data to match spectrogram time points
                # Handle cases where spectrogram times are outside speed times
                speed_data_interp = np.interp(spectrogram_time, speed_time, speed_values, left=np.nan, right=np.nan)
                print(f"Speed data loaded and interpolated. Shape: {speed_data_interp.shape}")
                print(f"Speed Threshold (> for Awake): {speed_threshold:.4f} cm/s")

                # Check how many points were interpolated vs NaN
                num_nan = np.sum(np.isnan(speed_data_interp))
                if num_nan > 0:
                     print(f"Warning: {num_nan}/{len(speed_data_interp)} speed data points are NaN after interpolation (likely due to time range mismatch).")

        except FileNotFoundError:
            print(f"Error: Speed CSV file not found: {speed_csv_path}. Speed thresholding disabled.")
        except Exception as e:
            print(f"Error loading or processing speed CSV ({speed_csv_path}): {e}. Speed thresholding disabled.")
            speed_data_interp = None
    else:
         print("No speed file provided. Speed thresholding disabled.")

    print("----------------------")

    # Scoring logic
    sleep_states = np.zeros(num_time_points, dtype=int) # 0=Awake, 1=NREM, 2=REM
    current_state = 0  # Start in Awake
    nrem_count = 0
    rem_count = 0
    awake_count = 0

    # Verbose logging flag (set to False to reduce output)
    verbose_scoring = False

    for i in range(num_time_points):
        if verbose_scoring: print(f"\n--- Time Point: {i} (Time: {times[i]:.2f}s) ---")

        # Get metrics for current time point, handle potential NaNs
        pc1_val = pc1[i] if np.isfinite(pc1[i]) else np.nan
        theta_val = theta_dominance[i] if np.isfinite(theta_dominance[i]) else np.nan
        emg_val = emg[i] if (use_emg and np.isfinite(emg[i])) else np.nan

        speed_val = np.nan
        is_speed_high = False
        if speed_data_interp is not None and i < len(speed_data_interp) and np.isfinite(speed_data_interp[i]):
            speed_val = speed_data_interp[i]
            is_speed_high = speed_val > speed_threshold

        if verbose_scoring:
            print(f"Current State: {['Awake', 'NREM', 'REM'][current_state]}")
            print(f"PC1: {pc1_val:.4f}, Theta: {theta_val:.4f}")
            if use_emg: print(f"EMG: {emg_val:.4f}")
            if not np.isnan(speed_val): print(f"Speed: {speed_val:.4f} cm/s")
            else: print("Speed: NaN or N/A")


        # --- State Transition Logic ---
        # Handle NaN values in comparisons: a NaN compared to anything is False
        pc1_above_nrem = (not np.isnan(pc1_val)) and (pc1_val > nrem_threshold)
        theta_above_rem = (not np.isnan(theta_val)) and (theta_val > rem_threshold)
        emg_below_thresh = (not use_emg) or ((not np.isnan(emg_val)) and (emg_val < emg_threshold)) # True if not using EMG or if EMG is low
        emg_above_thresh = use_emg and (not np.isnan(emg_val)) and (emg_val > emg_threshold) # True only if using EMG and EMG is high

        if current_state == 0:  # Awake
            # Conditions to potentially switch TO NREM or REM
            nrem_condition = pc1_above_nrem and emg_below_thresh and (not is_speed_high)
            rem_condition = theta_above_rem and emg_below_thresh and (not is_speed_high)

            if nrem_condition: nrem_count += 1
            else: nrem_count = 0

            if rem_condition: rem_count += 1
            else: rem_count = 0

            if verbose_scoring: print(f"  NREM Cond: {nrem_condition}, REM Cond: {rem_condition} | Counts: NREM={nrem_count}, REM={rem_count}")

            # Check for state switch (REM takes precedence if both met simultaneously?)
            # Current logic: If NREM buffer met, switch to NREM. If REM buffer met after that check, switch to REM.
            # Consider if REM should override NREM if both counts exceed buffer.
            if nrem_count >= buffer_size:
                current_state = 1  # Switch to NREM
                nrem_count, rem_count, awake_count = 0, 0, 0 # Reset counters
                if verbose_scoring: print("  Transition -> NREM")
            elif rem_count >= buffer_size: # Check REM only if NREM condition not met for buffer duration
                current_state = 2 # Switch to REM
                nrem_count, rem_count, awake_count = 0, 0, 0 # Reset counters
                if verbose_scoring: print("  Transition -> REM")

        elif current_state == 1:  # NREM
            # Conditions to potentially switch TO Awake
            awake_by_pc1 = (not np.isnan(pc1_val)) and (pc1_val < nrem_threshold)
            awake_by_emg = emg_above_thresh
            awake_by_speed = is_speed_high

            awake_condition = awake_by_pc1 or awake_by_emg or awake_by_speed

            if awake_condition:
                 awake_count += 1
                 if verbose_scoring:
                      reasons = []
                      if awake_by_pc1: reasons.append("PC1 Low")
                      if awake_by_emg: reasons.append("EMG High")
                      if awake_by_speed: reasons.append("Speed High")
                      print(f"  Awake Cond Met (from NREM): {' or '.join(reasons)} | Count: {awake_count}")
            else:
                awake_count = 0
                if verbose_scoring: print("  Awake Cond Not Met (NREM sustain)")


            if awake_count >= buffer_size:
                current_state = 0 # Switch to Awake
                nrem_count, rem_count, awake_count = 0, 0, 0 # Reset counters
                if verbose_scoring: print("  Transition -> Awake")
            # Add potential direct NREM -> REM transition? (Optional, depends on model)
            # Example: if theta_above_rem and emg_below_thresh: rem_count += 1 else rem_count = 0 ...


        elif current_state == 2:  # REM
            # Conditions to potentially switch TO Awake
            awake_by_theta = (not np.isnan(theta_val)) and (theta_val < rem_threshold)
            awake_by_emg = emg_above_thresh
            awake_by_speed = is_speed_high

            awake_condition = awake_by_theta or awake_by_emg or awake_by_speed

            if awake_condition:
                awake_count += 1
                if verbose_scoring:
                     reasons = []
                     if awake_by_theta: reasons.append("Theta Low")
                     if awake_by_emg: reasons.append("EMG High")
                     if awake_by_speed: reasons.append("Speed High")
                     print(f"  Awake Cond Met (from REM): {' or '.join(reasons)} | Count: {awake_count}")
            else:
                awake_count = 0
                if verbose_scoring: print("  Awake Cond Not Met (REM sustain)")


            if awake_count >= buffer_size:
                current_state = 0 # Switch to Awake
                nrem_count, rem_count, awake_count = 0, 0, 0 # Reset counters
                if verbose_scoring: print("  Transition -> Awake")
            # Add potential direct REM -> NREM transition? (Optional)

        sleep_states[i] = current_state

    print("--- Sleep Scoring Complete ---")
    return sleep_states

# === Main Script Execution ===

# --- Parameters ---
# Data Loading & Preprocessing
original_fs = 2500        # Original sampling rate from SGLX meta
target_fs = 1250          # Target sampling rate after downsampling
cutoff_freq = 450         # Low-pass filter cutoff frequency (applied AFTER downsampling)
num_channels_to_use = 5   # Number of channels to select from the END of the recording

# Spectrogram & Derived Metrics
window_size = 10          # Spectrogram window size in seconds
step_size = 1             # Spectrogram step size in seconds

# EMG Calculation (via ICA)
calculate_emg = True     # <<< SET TO True TO ENABLE EMG CALCULATION >>>
n_ica_components = 10     # Number of ICA components to extract
ica_epochs_duration = 240 # Duration (seconds) of data subset for ICA fitting
num_channels_for_ica = 5 # Number of channels (from selected LFP channels) to feed into ICA

# Sleep Scoring Thresholds & Options
scoring_buffer_size = 5   # Number of consecutive steps (e.g., 5*step_size seconds) to confirm state change
fixed_emg_threshold_val = None # Set to a float value (e.g., 0.5) to use fixed threshold, or None for percentile
speed_csv_file_path = None # Set to path to speed CSV file to enable speed thresholding, or None to disable
video_frame_rate = 30      # FPS of your video recording (if using speed)
video_start_delay_seconds = 0 # Offset (seconds) between video start and neural recording start
mouse_speed_threshold = 1.5 # cm/s, threshold for high speed classifying as Awake

# --- User Input: Select Directories ---
root = Tk()
root.withdraw()
root.attributes("-topmost", True)

print("Please select the directory containing LFP binary (.bin) and meta (.meta) files.")
lfp_bin_meta_dir_str = filedialog.askdirectory(title="Select directory containing LFP binary and meta files")
lfp_bin_meta_dir = Path(lfp_bin_meta_dir_str) if lfp_bin_meta_dir_str else None

# Optional: Timestamp directory (currently not used in core logic but kept for structure)
# print("Please select the directory containing timestamp files (optional).")
# timestamps_dir_str = filedialog.askdirectory(title="Select directory containing timestamp files (optional)")
# timestamps_dir = Path(timestamps_dir_str) if timestamps_dir_str else None

print("Please select the directory containing speed CSV files (optional, needed if speed thresholding is enabled).")
speed_files_dir_str = filedialog.askdirectory(title="Select directory containing speed files (optional)")
speed_files_dir = Path(speed_files_dir_str) if speed_files_dir_str else None

print("Please select the output directory for saving sleep state files.")
output_dir_str = filedialog.askdirectory(title="Select output directory for sleep state files")
output_dir = Path(output_dir_str) if output_dir_str else None

root.destroy()

# --- Validation of Inputs ---
if lfp_bin_meta_dir is None or not lfp_bin_meta_dir.is_dir():
    print("Error: LFP binary/meta directory not selected or invalid. Exiting.")
    sys.exit(1)
if output_dir is None or not output_dir.is_dir():
    # Try creating the output directory if it doesn't exist
    try:
         output_dir.mkdir(parents=True, exist_ok=True)
         print(f"Created output directory: {output_dir}")
    except Exception as e:
        print(f"Error: Output directory not selected or could not be created: {e}. Exiting.")
        sys.exit(1)

# --- Find LFP Files ---
lfp_files = sorted([p for p in lfp_bin_meta_dir.glob('*.lf.bin') if p.is_file()])

if not lfp_files:
    print(f"Error: No LFP binary (.bin) files found in: {lfp_bin_meta_dir}. Exiting.")
    sys.exit(1)

print(f"\nFound {len(lfp_files)} LFP binary files to process:")
for f in lfp_files: print(f" - {f.name}")

# --- Main Processing Loop ---
for lfp_file_path in lfp_files:
    print(f"\n{'='*20} Processing recording: {lfp_file_path.name} {'='*20}")

    # Find corresponding meta file
    meta_file_path = lfp_file_path.with_suffix('.meta')
    if not meta_file_path.exists():
        print(f"Warning: Corresponding meta file (.meta) not found for {lfp_file_path.name}. Skipping this recording.")
        continue

    # Find corresponding speed file (if directory provided)
    current_speed_file_path = None
    if speed_files_dir and speed_files_dir.is_dir():
        # Match based on the base name (without extensions)
        base_name = lfp_file_path.stem.split('.')[0] # Handle potential '.imec0' etc.
        possible_speed_files = list(speed_files_dir.glob(f'{base_name}*.csv')) # Allow flexibility in suffix
        if possible_speed_files:
            current_speed_file_path = possible_speed_files[0] # Take the first match
            print(f"Found corresponding speed file: {current_speed_file_path.name}")
        else:
            print(f"No speed file matching base name '{base_name}' found in {speed_files_dir}.")
            if speed_csv_file_path: # Check if the global variable was set (user intended to use speed)
                 print("Warning: Speed thresholding was requested but no matching file found for this recording.")
    elif speed_csv_file_path: # Global variable set, but directory not selected/invalid
         print("Warning: Speed thresholding was requested but speed file directory is missing or invalid.")


    # 1. Load Data (Memory Mapped)
    print("\n--- 1. Loading Data ---")
    data_memmap = read_bin_data(lfp_file_path, meta_file_path)
    if data_memmap is None:
        print(f"Failed to load data for {lfp_file_path.name}. Skipping.")
        continue

    # 2. Select Channels
    print("\n--- 2. Selecting Channels ---")
    if data_memmap.shape[1] < num_channels_to_use:
        print(f"Warning: Recording has {data_memmap.shape[1]} channels, less than requested {num_channels_to_use}. Using all available channels.")
        num_chans_actual = data_memmap.shape[1]
        if num_chans_actual == 0:
             print("Error: Recording has 0 channels. Skipping.")
             continue
    else:
        num_chans_actual = num_channels_to_use

    # Select LAST N channels
    data_selected_channels = data_memmap[:, -num_chans_actual:]
    print(f"Selected last {num_chans_actual} channels. Shape: {data_selected_channels.shape}")
    print(f"Data type: {data_selected_channels.dtype}")

    # 3. Downsample (using Decimate with Anti-Aliasing)
    print("\n--- 3. Downsampling ---")
    downsampling_factor = original_fs // target_fs
    print(f"Downsampling factor: {downsampling_factor} (from {original_fs} Hz to {target_fs} Hz)")

    # Ensure data is float before decimate (use float32 to save memory)
    if not np.issubdtype(data_selected_channels.dtype, np.floating):
         print("Converting data to float32 for downsampling...")
         # IMPORTANT: This step loads the selected channels into memory!
         # If even this causes MemoryError, chunking the downsampling is needed.#####
         try:
              data_selected_float = data_selected_channels.astype(np.float32)
         except MemoryError:
              print(f"MemoryError: Could not even load selected {num_chans_actual} channels into memory as float32.")
              print("The dataset is too large even after channel selection. Chunking is required for both downsampling and filtering.")
              print(f"Skipping recording: {lfp_file_path.name}")
              # Clean up memmap explicitly before skipping?
              del data_memmap
              del data_selected_channels
              continue # Skip to next file
         except Exception as e:
              print(f"Error converting selected channels to float32: {e}")
              del data_memmap
              del data_selected_channels
              continue
         finally:
              # Explicitly delete the memmap object now that we have the float version (or failed)
              # This helps release the file handle, though Python's GC should handle it
              del data_memmap

    else:
         # If already float, still need to potentially load from memmap for decimate
         print("Data is already float. Loading selected channels into memory for downsampling...")
         try:
              data_selected_float = np.array(data_selected_channels, dtype=np.float32) # Load as float32
         except MemoryError:
              print(f"MemoryError: Could not load selected {num_chans_actual} float channels into memory.")
              print(f"Skipping recording: {lfp_file_path.name}")
              del data_memmap
              del data_selected_channels
              continue
         except Exception as e:
              print(f"Error loading selected float channels: {e}")
              del data_memmap
              del data_selected_channels
              continue
         finally:
              del data_memmap
    try:
        # Decimate works along an axis (axis=0 for time)
        # Applies an anti-aliasing FIR filter by default, zero_phase=True recommended
        print(f"Applying signal.decimate along axis 0... Input shape: {data_selected_float.shape}")
        downsampled_data = signal.decimate(data_selected_float, downsampling_factor, axis=0, ftype='fir', zero_phase=True)
        print(f"Shape of data AFTER downsampling: {downsampled_data.shape}")
        print(f"Data type after downsampling: {downsampled_data.dtype}") # Usually float64
        # We can free memory from the pre-downsampled array now if decimate succeeded
        del data_selected_float

    except MemoryError:
         print(f"MemoryError during signal.decimate. The downsampled array might still be too large or decimate needs temporary memory.")
         print(f"Skipping recording: {lfp_file_path.name}")
         # Check if variable exists before deleting in except block
         if 'data_selected_float' in locals():
             del data_selected_float
         continue # Skip to next file
    except ValueError as e:
         # Example: Factor too large, data too short etc.
         print(f"ValueError during decimation: {e}. Check data length vs factor.")
         # Check if variable exists before deleting in except block
         if 'data_selected_float' in locals():
             del data_selected_float
         continue # Skip to next file
    except Exception as e:
         print(f"Error during decimation: {e}")
         # Check if variable exists before deleting in except block
         if 'data_selected_float' in locals():
             del data_selected_float
         continue 

    # 4. Filter (Low-pass) the Downsampled Data
    print("\n--- 4. Filtering ---")
    # Pass the TARGET sampling rate now
    filtered_data = filter_data(downsampled_data.astype(np.float32), target_fs, cutoff_freq) # Use float32 input

    if filtered_data is None:
        print(f"Filtering failed for {lfp_file_path.name}. Skipping.")
        del downsampled_data # Clean up
        continue

    # 5. Average Channels
    print("\n--- 5. Averaging Channels ---")
    # Average across channels (axis=1) to get a 1D signal
    averaged_lfp_data = np.mean(filtered_data, axis=1)
    print(f"Shape of averaged LFP data: {averaged_lfp_data.shape}")

    # 6. Compute Spectrogram
    print("\n--- 6. Computing Spectrogram ---")
    spectrogram, frequencies, times = compute_spectrogram(
        averaged_lfp_data, target_fs, window_size, step_size
    )

    if spectrogram is None:
        print(f"Spectrogram computation failed for {lfp_file_path.name}. Skipping.")
        del downsampled_data, filtered_data, averaged_lfp_data # Clean up
        continue

    # 7. Compute PCA on Spectrogram
    print("\n--- 7. Computing PCA ---")
    pc1 = compute_pca(spectrogram)
    if pc1 is None:
         print(f"PCA computation failed for {lfp_file_path.name}. Skipping.")
         del downsampled_data, filtered_data, averaged_lfp_data, spectrogram # Clean up
         continue

    # 8. Compute Theta Dominance
    print("\n--- 8. Computing Theta Dominance ---")
    theta_dominance = compute_theta_dominance(spectrogram, frequencies)
    if theta_dominance is None:
         print(f"Theta dominance computation failed for {lfp_file_path.name}. Skipping.")
         del downsampled_data, filtered_data, averaged_lfp_data, spectrogram, pc1 # Clean up
         continue

    # 9. Compute EMG (Conditional)
    print("\n--- 9. Computing EMG (Conditional) ---")
    emg = None # Initialize
    ica_object = None
    ica_components = None
    if calculate_emg:
        # Ensure the number of ICA channels doesn't exceed available channels in filtered_data
        num_channels_available_post_filter = filtered_data.shape[1]
        actual_num_channels_for_ica = min(num_channels_for_ica, num_channels_available_post_filter)
        actual_n_ica_components = min(n_ica_components, actual_num_channels_for_ica)

        if actual_num_channels_for_ica <= 0 or actual_n_ica_components <= 0:
             print("Warning: Not enough channels available after filtering/downsampling for requested ICA. Skipping EMG.")
             # Create dummy EMG of correct length if needed for scoring later
             emg_shape = len(pc1)
             emg = np.zeros(emg_shape)
             calculate_emg = False # Ensure scoring logic knows EMG wasn't calculated
        else:
             print(f"Proceeding with EMG calculation using {actual_num_channels_for_ica} channels and {actual_n_ica_components} components.")
             emg, ica_object, ica_components = compute_emg(
                 filtered_data, # Use filtered (and downsampled) data
                 target_fs,    # Pass the correct sampling rate
                 window_size,  # Pass window/step for segmentation
                 step_size,
                 n_ica_components=actual_n_ica_components,
                 num_channels_for_ica=actual_num_channels_for_ica, # Use adjusted number
                 ica_epochs_duration=ica_epochs_duration
             )
             if emg is None:
                 print("EMG computation failed. Scoring will proceed without EMG.")
                 emg_shape = len(pc1)
                 emg = np.zeros(emg_shape) # Create dummy EMG
                 calculate_emg = False # Ensure scoring logic knows EMG wasn't calculated
             elif len(emg) != len(pc1):
                  print(f"Warning: Length mismatch between segmented EMG ({len(emg)}) and PC1/Theta ({len(pc1)}). This indicates an issue in segmentation or spectrogram calculation.")
                  print("Attempting to truncate/pad EMG, but results may be unreliable.")
                  # Example: Truncate or pad EMG to match PC1 length
                  target_len = len(pc1)
                  if len(emg) > target_len:
                      emg = emg[:target_len]
                  else:
                      emg_padded = np.zeros(target_len)
                      emg_padded[:len(emg)] = emg
                      emg = emg_padded
                  print(f"Adjusted EMG length to {len(emg)}")


    else:
        print("EMG calculation is disabled by user setting ('calculate_emg = False').")
        # Create a dummy EMG array of zeros with the correct length for the scoring function
        emg_shape = len(pc1)
        emg = np.zeros(emg_shape)

    # 10. Score Sleep States
    print("\n--- 10. Scoring Sleep States ---")
    # Ensure all inputs to scoring are valid and have the same length
    min_len = min(len(pc1), len(theta_dominance), len(emg), len(times))
    if len(pc1) != min_len or len(theta_dominance) != min_len or len(emg) != min_len or len(times) != min_len:
        print(f"Warning: Truncating metrics to minimum length ({min_len}) before scoring.")
        pc1 = pc1[:min_len]
        theta_dominance = theta_dominance[:min_len]
        emg = emg[:min_len]
        times = times[:min_len]

    print(f"Shape of pc1 before score_sleep_states: {pc1.shape}")
    print(f"Shape of theta_dominance before score_sleep_states: {theta_dominance.shape}")
    print(f"Shape of emg before score_sleep_states: {emg.shape}")
    print(f"Shape of times before score_sleep_states: {times.shape}")


    print("\n---Ranges of Metrics before Scoring---")
    print(f"PC1 Range:      Min={np.nanmin(pc1):.4f}, Max={np.nanmax(pc1):.4f}")
    print(f"ThetaDom Range: Min={np.nanmin(theta_dominance):.4f}, Max={np.nanmax(theta_dominance):.4f}")
    print(f"EMG Range:      Min={np.nanmin(emg):.4f}, Max={np.nanmax(emg):.4f}")
    print("------------------------------------")


    sleep_states = score_sleep_states(
        pc1, theta_dominance, emg, times, step_size, # Pass times and step_size
        buffer_size=scoring_buffer_size,
        fixed_emg_threshold=fixed_emg_threshold_val,
        speed_csv_path=str(current_speed_file_path) if current_speed_file_path else None,
        video_fps=video_frame_rate, video_start_offset_seconds=video_start_delay_seconds,
        speed_threshold=mouse_speed_threshold,
        use_emg=calculate_emg # Pass the flag indicating if EMG was successfully calculated and should be used
    )

    if sleep_states is None:
        print(f"Sleep scoring failed for {lfp_file_path.name}. Skipping save.")
        # Clean up intermediate data before next loop iteration
        del downsampled_data, filtered_data, averaged_lfp_data, spectrogram, pc1, theta_dominance, emg
        if ica_components is not None: del ica_components
        continue

    # 11. Save Results
    print("\n--- 11. Saving Results ---")
    output_filename_base = lfp_file_path.stem # Original filename without .bin
    output_filename = f'{output_filename_base}_sleep_states_EMG.npy'
    sleep_states_save_path = output_dir / output_filename
    try:
        np.save(sleep_states_save_path, sleep_states)
        print(f"Sleep states saved successfully to: {sleep_states_save_path}")

        # Also save the corresponding times array for alignment
        times_filename = f'{output_filename_base}_sleep_state_times_EMG.npy'
        times_save_path = output_dir / times_filename
        np.save(times_save_path, times)
        print(f"Corresponding times saved to: {times_save_path}")

    except Exception as e:
        print(f"Error saving sleep states or times to {output_dir}: {e}")


    # 12. Visualization (Optional)
    print("\n--- 12. Visualization (Optional) ---")
    if calculate_emg and ica_components is not None: # Only plot if ICA was actually run
        try:
            num_ics_to_plot = min(10, ica_components.shape[1]) # Plot up to 10 ICs

            # Plot IC Time Series
            plt.figure(figsize=(15, 10))
            plt.suptitle(f"ICA Components Time Series - {output_filename_base}", y=0.99)
            for i in range(num_ics_to_plot):
                plt.subplot(num_ics_to_plot, 1, i + 1)
                # Plot a subset for clarity if too long
                plot_len = min(len(times) * window_size * target_fs // step_size, ica_components.shape[0]) # Approx length match
                plot_len = min(plot_len, int(target_fs * 600)) # Max 10 mins plot
                plot_time_axis = np.arange(plot_len) / target_fs
                plt.plot(plot_time_axis, ica_components[:plot_len, i])
                plt.title(f"IC {i}")
                plt.ylabel("Amplitude")
                if i < num_ics_to_plot - 1: plt.xticks([]) # Remove time labels for inner plots
            plt.xlabel("Time (seconds)")
            plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout for suptitle
            plot_filename_ts = output_dir / f'{output_filename_base}_ICA_TimeSeries.png'
            plt.savefig(plot_filename_ts)
            print(f"Saved ICA time series plot: {plot_filename_ts}")
            # plt.show() # Comment out plt.show() for batch processing
            plt.close() # Close the figure

            # Plot IC Power Spectra (using Welch method)
            plt.figure(figsize=(15, 10))
            plt.suptitle(f"ICA Components Power Spectra - {output_filename_base}", y=0.99)
            for i in range(num_ics_to_plot):
                plt.subplot(num_ics_to_plot, 1, i + 1)
                # Use Welch on the component's time series
                # nperseg recommendation: power of 2, relates to frequency resolution
                welch_nperseg = min(2048, ica_components.shape[0] // 8) # Example segment length
                if welch_nperseg == 0: continue # Skip if component too short

                frequencies_psd, power_spectrum = signal.welch(ica_components[:, i], fs=target_fs, nperseg=welch_nperseg, scaling='density')
                plt.semilogy(frequencies_psd, power_spectrum) # Log scale for power often useful
                plt.title(f"IC {i} Power Spectrum")
                plt.ylabel("PSD (log)")
                plt.xlim([0, target_fs / 4]) # Focus on lower frequencies, e.g., up to target_fs/4
                if i == num_ics_to_plot - 1: plt.xlabel("Frequency (Hz)")
            plt.tight_layout(rect=[0, 0.03, 1, 0.97])
            plot_filename_psd = output_dir / f'{output_filename_base}_ICA_PSD.png'
            plt.savefig(plot_filename_psd)
            print(f"Saved ICA PSD plot: {plot_filename_psd}")
            # plt.show()
            plt.close() # Close the figure

        except Exception as e:
            print(f"Error during visualization: {e}")
            plt.close('all') # Close any potentially open figures

    else:
        print("ICA-EMG calculation was skipped or failed, no IC plots generated.")

    # --- Clean up memory before next loop ---
    print("\n--- Cleaning up memory ---")
    try:
        del downsampled_data, filtered_data, averaged_lfp_data, spectrogram, pc1, theta_dominance, emg, sleep_states, times
        if 'ica_components' in locals() and ica_components is not None: del ica_components
        if 'ica_object' in locals() and ica_object is not None: del ica_object
        import gc
        gc.collect()
        print("Memory cleanup attempted.")
    except NameError:
         print("Some variables for cleanup were not defined (likely due to earlier skip).")
    except Exception as e:
         print(f"Error during cleanup: {e}")


# --- End of Script ---
print(f"\n{'='*20} Script Execution Completed {'='*20}")
print(f"Processed {len(lfp_files)} files.")
print(f"Results saved in: {output_dir}")
print(f"Detailed logs are in: {output_file_path}")

# Restore stdout/stderr
sys.stdout.close()
sys.stderr.close()
sys.stdout = original_stdout
sys.stderr = original_stderr
print(f"\nScript execution completed. Check logs in '{output_file_path}' and results in '{output_dir}'.")