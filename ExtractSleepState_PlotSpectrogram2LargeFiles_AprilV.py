# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 2025 (Visualization added, refined V6 - No LFP, AutoShade)
State scoring based on Supplementary Materials
Wannan Yang Science 383,1478(2024)_Buzsaki Lab sleep state score

@author: Alok (modified by AI Assistant)

Applies downsampling (to 1250Hz or other target) BEFORE filtering to handle large files.
Uses scipy.signal.decimate for proper anti-aliased downsampling.
Generates detailed overview plots (excluding raw LFP) for each recording,
including split spectrograms (with auto shading) and scoring metrics aligned in time.
"""
import io
import numpy as np
from scipy import signal
from sklearn.decomposition import PCA, FastICA
from scipy.stats import zscore
from tkinter import Tk
from tkinter import filedialog
from pathlib import Path
from DemoReadSGLXData.readSGLX import readMeta # Assuming this module exists and works
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import math
import gc
import traceback

# --- Redirect stdout and stderr to a file ---
output_file_path = "script_output.txt"  # You can change the filename if you like
original_stdout = sys.stdout
original_stderr = sys.stderr
try:
    # os.makedirs(os.path.dirname(output_file_path), exist_ok=True) # Uncomment if output_file_path includes a directory
    sys.stdout = open(output_file_path, 'w', encoding='utf-8')
    sys.stderr = sys.stdout
    print(f"--- Script Started: {pd.Timestamp.now()} ---") # Log start time
except Exception as e:
    print(f"Fatal Error: Could not redirect output to {output_file_path}: {e}")
    # Fallback to original stdout/stderr if redirection fails
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    print("Warning: Output redirection failed. Logging to console.")

# --- Function Definitions ---

def read_bin_data(file_path, meta_path, data_type='int16'):
    """
    Reads binary data from a file using memory-mapping.
    Args:
        file_path (str): Path to the binary file.
        meta_path (str): Path to the corresponding metadata file.
        data_type (str): Data type of the samples.
    Returns:
        tuple: (numpy.ndarray: Memory-mapped data, float: sampling rate) or (None, None) on error.
    """
    sampling_rate = None
    data = None
    try:
        # Read metadata to get the number of channels and sampling rate
        meta = readMeta(meta_path)
        num_channels = int(meta['nSavedChans'])
        # Get precise sampling rate if available
        if 'imSampRate' in meta:
            sampling_rate = float(meta['imSampRate'])
        elif 'niSampRate' in meta:
            sampling_rate = float(meta['niSampRate'])
        else:
            print(f"Error: Sampling rate key ('imSampRate' or 'niSampRate') not found in {meta_path}.")
            return None, None

        if sampling_rate <= 0:
            print(f"Error: Invalid sampling rate ({sampling_rate}) found in {meta_path}.")
            return None, None

        print(f"Reading metadata: {num_channels} saved channels found. Sampling rate: {sampling_rate:.6f} Hz")

        # Calculate the first dimension of the shape
        file_size = os.path.getsize(file_path)
        item_size = np.dtype(data_type).itemsize
        if num_channels <= 0 or item_size <= 0:
             print(f"Error: Invalid number of channels ({num_channels}) or item size ({item_size}). Cannot calculate shape.")
             return None, sampling_rate

        expected_total_bytes = file_size - (file_size % (num_channels * item_size))
        if file_size != expected_total_bytes:
            print(f"Warning: File size {file_size} is not an exact multiple of {num_channels} channels * {item_size} bytes/sample.")
            print(f" Potential extra bytes at end of file: {file_size % (num_channels * item_size)}. Processing based on truncated size.")
        num_samples = expected_total_bytes // (num_channels * item_size)

        if num_samples <= 0:
            print("Error: Calculated number of samples is zero or negative. Check file size and metadata.")
            return None, sampling_rate

        print(f"Calculated samples: {num_samples}")
        shape = (num_samples, num_channels)
        print(f"Expected data shape: {shape}")

        # Memory-map the binary file
        data = np.memmap(file_path, dtype=data_type, mode='r', shape=shape, offset=0) # Ensure offset is 0 unless header exists
        print(f"Successfully memory-mapped file: {file_path}")
        return data, sampling_rate

    except FileNotFoundError:
        print(f"Error: File not found - {file_path} or {meta_path}")
        return None, None
    except KeyError as e:
        print(f"Error: Metadata key missing in {meta_path} - {e}")
        return None, sampling_rate
    except ValueError as e:
         print(f"Error: Problem with shape or dtype during memmap for {file_path} - {e}")
         if data is not None and hasattr(data, '_mmap'): data._mmap.close()
         return None, sampling_rate
    except Exception as e:
        print(f"An unexpected error occurred in read_bin_data for {file_path}: {e}")
        traceback.print_exc()
        if data is not None and hasattr(data, '_mmap'): data._mmap.close()
        return None, None

def filter_data(data, fs, cutoff_freq):
    """
    Filters data using a FIR filter (scipy.signal.firwin and lfilter).
    Args:
        data (numpy.ndarray): Data to filter (should be float type).
        fs (float): Sampling rate OF THE INPUT DATA.
        cutoff_freq (float): Cutoff frequency for the low-pass filter.
    Returns:
        numpy.ndarray: Filtered data (typically float64) or None on error.
    """
    print(f"Shape of data entering filter_data: {data.shape}")
    print(f"Data type entering filter_data: {data.dtype}")
    print(f"Filtering with fs={fs:.6f} Hz, cutoff={cutoff_freq} Hz")

    if data is None or data.size == 0:
         print("Error: Cannot filter empty data.")
         return None

    # Ensure data is float32 for filtering to save memory, only copy if necessary
    if not np.issubdtype(data.dtype, np.floating):
        print("Warning: Data type is not float, converting to float32 for filtering.")
        try:
            data_float = data.astype(np.float32)
        except MemoryError:
            print("MemoryError converting data to float32 inside filter_data.")
            return None
        except Exception as e:
            print(f"Error converting data to float32 inside filter_data: {e}")
            return None
    elif data.dtype != np.float32:
        print(f"Data type is {data.dtype}, converting to float32 for filtering.")
        try:
            data_float = data.astype(np.float32)
        except MemoryError:
             print("MemoryError converting data to float32 inside filter_data.")
             return None
        except Exception as e:
            print(f"Error converting data to float32 inside filter_data: {e}")
            return None
    else:
        data_float = data # No conversion needed if already float32

    nyquist_freq = fs / 2.0
    if cutoff_freq >= nyquist_freq:
        print(f"Warning: Cutoff frequency ({cutoff_freq} Hz) is >= Nyquist frequency ({nyquist_freq:.2f} Hz).")
        cutoff_freq = nyquist_freq * 0.99 # Adjust cutoff
        print(f"Adjusted cutoff frequency to {cutoff_freq:.2f} Hz")
    elif cutoff_freq <= 0:
         print("Error: Cutoff frequency must be positive.")
         return None

    normalized_cutoff_freq = cutoff_freq / nyquist_freq
    numtaps = 31  # Keep filter order manageable
    if numtaps > data_float.shape[0]:
        print(f"Warning: Filter order ({numtaps}) > data length ({data_float.shape[0]}). Reducing filter order.")
        numtaps = (data_float.shape[0] // 2) * 2 - 1 # Make it odd and < data length
        if numtaps < 3:
             print("Error: Data too short for filtering (numtaps < 3).")
             return None

    if numtaps % 2 == 0: numtaps += 1 # Ensure odd for Type I

    try:
        taps = signal.firwin(numtaps=numtaps, cutoff=normalized_cutoff_freq, window='hamming', pass_zero='lowpass')
        print(f"Shape of filter taps: {taps.shape}")

        # Apply filter channel by channel (axis=0: time axis)
        filtered_data = signal.lfilter(taps.astype(np.float32), 1.0, data_float, axis=0)

        print(f"Shape of filtered data exiting filter_data: {filtered_data.shape}")
        print(f"Data type of filtered data exiting filter_data: {filtered_data.dtype}") # Often float64
        return filtered_data
    except MemoryError:
        print("MemoryError during signal.lfilter.")
        return None
    except Exception as e:
        print(f"Error during filtering (lfilter): {e}")
        traceback.print_exc()
        return None

def compute_spectrogram(data, fs, window_size, step_size):
    """
    Computes spectrogram using scipy.signal.spectrogram.
    Args:
        data (numpy.ndarray): 1D data array.
        fs (float): Sampling rate.
        window_size (float): Window size in seconds.
        step_size (float): Step size in seconds.
    Returns:
        tuple: (Spectrogram, frequencies, times) or (None, None, None) on error.
    """
    print(f"Computing spectrogram: fs={fs:.6f}, window={window_size}s, step={step_size}s")
    print(f"Input data shape for spectrogram: {data.shape}")
    if data is None or data.ndim != 1:
        print("Error: Spectrogram input data must be 1D and not None.")
        return None, None, None
    if len(data) == 0:
         print("Error: Spectrogram input data is empty.")
         return None, None, None

    try:
        window_samples = int(round(window_size * fs))
        step_samples = int(round(step_size * fs))

        if window_samples <= 0 or step_samples <= 0:
             print("Error: Window or step size results in non-positive samples.")
             return None, None, None

        if window_samples > len(data):
             print(f"Error: Window size ({window_samples} samples) > data length ({len(data)}). Cannot compute spectrogram.")
             return None, None, None

        noverlap = window_samples - step_samples
        if noverlap < 0:
            print(f"Warning: Step size ({step_samples}) > window size ({window_samples}). Setting overlap to 0.")
            noverlap = 0

        print(f"Spectrogram params: nperseg={window_samples}, noverlap={noverlap}, step_samples={step_samples}")

        win = signal.windows.hann(window_samples)

        frequencies, times, spectrogram = signal.spectrogram(
            data.astype(np.float64), # Use float64 for precision in FFT
            fs=fs,
            window=win,
            nperseg=window_samples,
            noverlap=noverlap,
            scaling='density', # Power spectral density
            mode='psd'         # Compute PSD directly
        )
        print(f"Spectrogram computed. Shape: {spectrogram.shape}, Freq shape: {frequencies.shape}, Time shape: {times.shape}")
        if spectrogram.size == 0 or times.size == 0 or frequencies.size == 0:
             print("Error: Spectrogram computation resulted in one or more empty arrays.")
             return None, None, None
        return spectrogram, frequencies, times
    except MemoryError:
        print("MemoryError during spectrogram computation.")
        return None, None, None
    except Exception as e:
        print(f"Error computing spectrogram: {e}")
        traceback.print_exc()
        return None, None, None

def compute_pca(data):
    """
    Performs PCA on z-scored spectrogram data.
    Args:
        data (numpy.ndarray): Spectrogram data (frequencies x times).
    Returns:
        numpy.ndarray: First principal component (1D array, length = n_times) or None on error.
    """
    print(f"Computing PCA on data of shape: {data.shape}")
    if data is None or data.size == 0 or data.shape[0] < 2 or data.shape[1] < 2:
        print("Error: Invalid data for PCA (None, empty, or too small).")
        return None
    try:
        # Replace non-finite values before z-scoring
        if not np.all(np.isfinite(data)):
             print("Warning: Non-finite values (NaN/Inf) found before PCA. Replacing with zeros.")
             data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        # Z-score along frequency axis (axis=0)
        try:
            zscored_data = zscore(data, axis=0, ddof=0) # Use population std dev (ddof=0)
            if np.any(np.isnan(zscored_data)):
                print("Warning: NaNs found after z-scoring (likely zero variance bins). Replacing NaNs with 0.")
                zscored_data = np.nan_to_num(zscored_data)
        except Exception as e_zscore:
             print(f"Error during z-scoring before PCA: {e_zscore}")
             return None

        pca = PCA(n_components=1)
        # Transpose spectrogram from (freq, time) to (time, freq) for PCA fit
        pc1 = pca.fit_transform(zscored_data.T)
        print(f"PCA computed. PC1 shape: {pc1.shape}, Explained variance: {pca.explained_variance_ratio_[0]:.4f}")
        return pc1.squeeze() # Shape (n_times,)

    except MemoryError:
        print("MemoryError during PCA computation.")
        return None
    except Exception as e:
        print(f"Error computing PCA: {e}")
        traceback.print_exc()
        return None

def compute_theta_dominance(spectrogram, frequencies):
    """
    Computes theta dominance from spectrogram.
    Args:
        spectrogram (numpy.ndarray): Spectrogram data (frequencies x times).
        frequencies (numpy.ndarray): Frequencies corresponding to spectrogram.
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

        if len(theta_band_indices) == 0: print("Warning: No frequency bins found for Theta band (5-10Hz).")
        if len(total_band_indices) == 0: print("Warning: No frequency bins found for Total band (2-16Hz).")
        if len(theta_band_indices) == 0 or len(total_band_indices) == 0:
             print("Cannot calculate theta dominance if bands are missing.")
             return np.full(spectrogram.shape[1], np.nan) # Return NaNs if bands not found

        # Use nanmean to be robust to potential NaNs in spectrogram slices
        epsilon = 1e-12
        # Ensure indices are valid before slicing
        theta_power = np.nanmean(spectrogram[theta_band_indices, :], axis=0)
        total_power = np.nanmean(spectrogram[total_band_indices, :], axis=0)

        print(f"Shape of theta_power: {theta_power.shape}")
        print(f"Shape of total_power: {total_power.shape}")

        # Avoid division by zero or near-zero; output NaN where total_power is too small
        theta_dominance = np.full_like(theta_power, np.nan) # Initialize with NaNs
        valid_mask = total_power > epsilon
        theta_dominance[valid_mask] = np.divide(theta_power[valid_mask], total_power[valid_mask])

        print(f"Shape of theta_dominance: {theta_dominance.shape}")

        # Final check for non-finite values (should only be NaNs now)
        num_non_finite = np.sum(~np.isfinite(theta_dominance))
        if num_non_finite > 0:
             print(f"Warning: {num_non_finite} non-finite (NaN) values found in theta dominance (likely due to low total power).")

        return theta_dominance
    except Exception as e:
        print(f"Error computing Theta Dominance: {e}")
        traceback.print_exc()
        return None

def compute_emg(data, fs, window_size, step_size, n_ica_components=10, ica_epochs_duration=240, num_channels_for_ica=48):
    """
    Estimates EMG from LFP data using ICA (FastICA).
    Args: (See previous version)
    Returns: tuple: (Segmented EMG, ICA object, ICA source signals) or (None, None, None) on error.
    """
    print("\n---ICA-based EMG Computation (scikit-learn FastICA)---")
    print(f"Input data shape for EMG: {data.shape}, fs={fs:.6f}")

    if data is None or data.size == 0:
        print("Error: Input data for EMG computation is invalid.")
        return None, None, None

    ica_object_fitted = None # Initialize ICA object
    full_source_signals = None # Initialize source signals

    try:
        actual_channels_available = data.shape[1]
        if actual_channels_available == 0:
             print("Error: Input data for EMG has 0 channels.")
             return None, None, None

        channels_to_use = min(num_channels_for_ica, actual_channels_available)
        n_ica_components = min(n_ica_components, channels_to_use) # Ensure components <= channels used

        if n_ica_components <= 0:
             print("Error: Number of ICA components must be positive.")
             return None, None, None

        ica_channels_indices = np.arange(channels_to_use) # Use first N channels
        ica_data = data[:, ica_channels_indices]
        print(f"Applying ICA to first {channels_to_use} channels.")
        print(f"Shape of data selected for ICA: {ica_data.shape}")

        # --- ICA Fitting Subset ---
        ica_n_samples = int(round(ica_epochs_duration * fs))
        ica_n_samples = min(ica_n_samples, ica_data.shape[0]) # Don't exceed available data

        if ica_n_samples < n_ica_components: # Need at least as many samples as components
             print(f"Error: Not enough samples ({ica_n_samples}) for {n_ica_components} ICA components.")
             return None, None, None

        ica_data_subset = ica_data[:ica_n_samples, :].astype(np.float32) # Use float32 subset
        print(f"Using {ica_n_samples} samples ({ica_data_subset.shape[0] / fs:.2f} seconds) for ICA fitting.")

        # --- Optional: Pre-filter subset for ICA ---
        emg_cutoff_low_ica = 30
        emg_cutoff_high_ica = 300
        nyquist_freq_ica = fs / 2.0
        normalized_cutoff_low_ica = emg_cutoff_low_ica / nyquist_freq_ica
        normalized_cutoff_high_ica = emg_cutoff_high_ica / nyquist_freq_ica

        ica_filtered_data_subset = None
        ica_filter_taps = None
        pre_filter_applied = False
        if 0 < normalized_cutoff_low_ica < normalized_cutoff_high_ica < 1.0:
            numtaps_ica = 31
            if numtaps_ica > ica_data_subset.shape[0]: numtaps_ica = (ica_data_subset.shape[0] // 2) * 2 - 1
            if numtaps_ica % 2 == 0: numtaps_ica += 1
            if numtaps_ica >= 3:
                print(f"Bandpass filtering ICA subset: {emg_cutoff_low_ica}-{emg_cutoff_high_ica} Hz")
                ica_filter_taps = signal.firwin(numtaps_ica, [normalized_cutoff_low_ica, normalized_cutoff_high_ica],
                                                pass_zero='bandpass', window='hamming')
                ica_filtered_data_subset = signal.lfilter(ica_filter_taps.astype(np.float32), 1.0, ica_data_subset, axis=0)
                pre_filter_applied = True
            else: print("Warning: ICA subset too short for bandpass filter. Using unfiltered.")
        else: print("Warning: Invalid EMG freq band for ICA pre-filter or disabled. Using unfiltered.")

        if ica_filtered_data_subset is None: # Fallback if filtering failed/skipped
            ica_filtered_data_subset = ica_data_subset

        # --- Run FastICA ---
        ica_object_fitted = FastICA(n_components=n_ica_components, random_state=42, whiten='unit-variance', max_iter=500, tol=1e-3)
        print(f"Running FastICA with {n_ica_components} components on {channels_to_use} channels...")
        # Use try-except for fit_transform as it can fail (e.g., singular matrix)
        try:
            ica_source_signals_subset = ica_object_fitted.fit_transform(ica_filtered_data_subset)
        except Exception as ica_fit_err:
             print(f"Error during FastICA fit_transform: {ica_fit_err}")
             return None, None, None # Cannot proceed if fit fails
        print("FastICA fitted and transformed subset.")
        print("Shape of ICA source signals (subset):", ica_source_signals_subset.shape)
        del ica_filtered_data_subset, ica_data_subset # Free memory

        # --- IC Component Selection (Placeholder) ---
        emg_ic_index = 0 # <<< Placeholder: Needs better logic >>>
        print(f"**Placeholder Warning**: Using IC {emg_ic_index} as estimated EMG. Selection logic needed!")

        # --- Apply transform to full dataset ---
        print("Applying learned ICA transform to full dataset...")
        ica_data_float = ica_data.astype(np.float32)
        if pre_filter_applied and ica_filter_taps is not None:
             print(f"Applying bandpass filter ({emg_cutoff_low_ica}-{emg_cutoff_high_ica} Hz) to full data before ICA transform...")
             # Filter full data chunk by chunk if too large? For now, assume it fits.
             try:
                 ica_filtered_data_full = signal.lfilter(ica_filter_taps.astype(np.float32), 1.0, ica_data_float, axis=0)
                 full_source_signals = ica_object_fitted.transform(ica_filtered_data_full)
                 del ica_filtered_data_full
             except MemoryError:
                  print("MemoryError filtering full data for ICA transform. Cannot proceed.")
                  return None, ica_object_fitted, None # Return fitted object but no signals/EMG
        else:
             full_source_signals = ica_object_fitted.transform(ica_data_float)
        del ica_data_float, ica_data # Free memory

        print("Shape of full source signals:", full_source_signals.shape)
        emg_ic_full = full_source_signals[:, emg_ic_index]
        emg_signal_raw = np.abs(emg_ic_full) # Use absolute value of selected component

        print("Shape of raw EMG signal (selected IC):", emg_signal_raw.shape)
        print(f"Range of raw EMG: Min={np.min(emg_signal_raw):.4f}, Max={np.max(emg_signal_raw):.4f}")

        # --- Segment and Average EMG ---
        window_samples = int(round(window_size * fs))
        step_samples = int(round(step_size * fs))
        num_segments_raw = (len(emg_signal_raw) - window_samples) // step_samples + 1

        if num_segments_raw <= 0:
             print("Error: Not enough EMG data to create segments.")
             return None, ica_object_fitted, full_source_signals # Return fitted obj & sources

        emg_segmented = np.zeros(num_segments_raw)
        print(f"Segmenting EMG into {num_segments_raw} segments (based on raw length)...")
        for i in range(num_segments_raw):
            start_sample = i * step_samples
            end_sample = min(start_sample + window_samples, len(emg_signal_raw))
            start_sample = min(start_sample, end_sample) # Ensure start <= end

            segment = emg_signal_raw[start_sample:end_sample]
            if segment.size > 0:
                 emg_segmented[i] = np.mean(segment) # Use mean of absolute value

        print("Shape of segmented ICA-EMG (before alignment check):", emg_segmented.shape)
        print(f"Range of segmented ICA-EMG: Min={np.min(emg_segmented):.4f}, Max={np.max(emg_segmented):.4f}")

        # Note: Length alignment check happens in main loop before scoring

        return emg_segmented, ica_object_fitted, full_source_signals

    except MemoryError:
        print("MemoryError during EMG computation.")
        return None, ica_object_fitted, full_source_signals # Return what we have
    except Exception as e:
        print(f"An error occurred during EMG computation: {e}")
        traceback.print_exc()
        return None, ica_object_fitted, full_source_signals

def score_sleep_states(pc1, theta_dominance, emg, times,
                       step_size, buffer_size=10, fixed_emg_threshold=None,
                       speed_csv_path=None, video_fps=30, video_start_offset_seconds=0,
                       speed_threshold=1.5, use_emg=False):
    """
    Scores sleep states based on provided metrics with sticky thresholds.
    Args: (See previous definitions)
    Returns: numpy.ndarray: Sleep state labels (0=Awake, 1=NREM, 2=REM) or None on error.
    """
    print("\n--- Scoring Sleep States ---")
    # Assume inputs are already validated and aligned in length by the caller
    if pc1 is None or theta_dominance is None or emg is None or times is None: return None
    num_time_points = len(pc1)
    if num_time_points == 0: print("Error: Input metrics are empty."); return None
    print(f"Scoring based on {num_time_points} time points.")

    # --- Calculate Thresholds ---
    try:
        nrem_threshold = np.nanpercentile(pc1, 75)
        rem_threshold = np.nanpercentile(theta_dominance, 75)
        local_emg_threshold = np.nan # Use local variable for EMG threshold logic here
    except Exception as e:
        print(f"Error calculating NREM/REM percentiles: {e}. Cannot score.")
        return None

    if use_emg:
        if fixed_emg_threshold is not None:
            local_emg_threshold = fixed_emg_threshold
            print(f"Using fixed EMG threshold for scoring logic: {local_emg_threshold:.4f}")
        elif len(emg) > 0 and np.any(np.isfinite(emg)):
             local_emg_threshold = np.nanpercentile(emg[np.isfinite(emg)], 25)
             print(f"Using percentile EMG threshold for scoring logic: {local_emg_threshold:.4f}")
        else:
             local_emg_threshold = 0 # Fallback if EMG is unusable but flag was True
             print("Warning: EMG unusable but use_emg=True. Using EMG threshold=0 for logic.")
    else:
        print("EMG usage disabled for scoring logic.")
        local_emg_threshold = np.nan # Ensure it's NaN if not used

    print("---Thresholds Used for Scoring Logic---")
    print(f"NREM Threshold (PC1 >): {nrem_threshold:.4f}")
    print(f"REM Threshold (Theta >): {rem_threshold:.4f}")
    if use_emg: print(f"EMG Threshold (< for NREM/REM): {local_emg_threshold:.4f}")
    else: print(f"EMG Threshold: N/A (use_emg=False)")

    # --- Load and process speed data ---
    speed_data_interp = None
    if speed_csv_path:
        print(f"Loading speed data from: {speed_csv_path}")
        try:
            speed_df = pd.read_csv(speed_csv_path)
            if 'speed' in speed_df.columns and not speed_df['speed'].isnull().all():
                speed_values = speed_df['speed'].values
                speed_time = np.arange(len(speed_values)) / video_fps + video_start_offset_seconds
                speed_data_interp = np.interp(times, speed_time, speed_values, left=np.nan, right=np.nan)
                print(f"Speed data loaded and interpolated. Speed Threshold (> Aw): {speed_threshold:.2f} cm/s")
                num_nan = np.sum(np.isnan(speed_data_interp))
                if num_nan > 0: print(f"Warning: {num_nan}/{len(speed_data_interp)} speed points are NaN after interpolation.")
            else: print(f"Error: 'speed' column missing or empty in {speed_csv_path}. Disabling speed.")
        except Exception as e: print(f"Error loading/processing speed CSV: {e}. Disabling speed.")
    else: print("No speed file provided. Speed thresholding disabled.")
    print("----------------------")

    # --- Scoring Logic ---
    sleep_states = np.zeros(num_time_points, dtype=int)
    current_state = 0
    nrem_count, rem_count, awake_count = 0, 0, 0
    verbose_scoring = False # Set True for detailed step-by-step logging

    for i in range(num_time_points):
        pc1_val, theta_val, emg_val = pc1[i], theta_dominance[i], emg[i]

        is_speed_high = False
        if speed_data_interp is not None and i < len(speed_data_interp) and np.isfinite(speed_data_interp[i]):
            is_speed_high = speed_data_interp[i] > speed_threshold

        # Conditions: Use local_emg_threshold. Handle NaNs (NaN compared to anything is False)
        pc1_above_nrem = pc1_val > nrem_threshold
        theta_above_rem = theta_val > rem_threshold
        # If not using EMG, emg_below is True, emg_above is False
        # If using EMG, depends on comparison with local_emg_threshold (NaN threshold means False)
        emg_below_thresh = (not use_emg) or (np.isfinite(local_emg_threshold) and emg_val < local_emg_threshold)
        emg_above_thresh = use_emg and (np.isfinite(local_emg_threshold) and emg_val > local_emg_threshold)

        if verbose_scoring: # Add logging details if needed
             pass # print(...)

        # State transitions (same logic as before)
        if current_state == 0:  # Awake
            nrem_condition = pc1_above_nrem and emg_below_thresh and (not is_speed_high)
            rem_condition = theta_above_rem and emg_below_thresh and (not is_speed_high)
            nrem_count = nrem_count + 1 if nrem_condition else 0
            rem_count = rem_count + 1 if rem_condition else 0
            if nrem_count >= buffer_size: current_state = 1; nrem_count=0; rem_count=0; awake_count=0
            elif rem_count >= buffer_size: current_state = 2; nrem_count=0; rem_count=0; awake_count=0
        elif current_state == 1:  # NREM
            awake_condition = (pc1_val < nrem_threshold) or emg_above_thresh or is_speed_high
            awake_count = awake_count + 1 if awake_condition else 0
            if awake_count >= buffer_size: current_state = 0; nrem_count=0; rem_count=0; awake_count=0
        elif current_state == 2:  # REM
            awake_condition = (theta_val < rem_threshold) or emg_above_thresh or is_speed_high
            awake_count = awake_count + 1 if awake_condition else 0
            if awake_count >= buffer_size: current_state = 0; nrem_count=0; rem_count=0; awake_count=0

        sleep_states[i] = current_state

    print("--- Sleep Scoring Complete ---")
    # Calculate state percentages
    unique, counts = np.unique(sleep_states, return_counts=True)
    state_dict = dict(zip(unique, counts))
    total_counts = len(sleep_states)
    print("State Distribution:")
    for state_code, state_name in {0:"Awake", 1:"NREM", 2:"REM"}.items():
        count = state_dict.get(state_code, 0)
        percent = (count / total_counts) * 100 if total_counts > 0 else 0
        print(f"  {state_name}: {count} bins ({percent:.1f}%)")

    return sleep_states

# *** UPDATED VISUALIZATION FUNCTION V6 (No LFP Plot, Auto Shading) ***
def plot_scoring_overview(times, frequencies, spectrogram, pc1, theta_dominance, emg,
                          sleep_states, thresholds, calculate_emg, step_size,
                          output_path, base_filename):
    """
    Generates and saves a multi-panel plot visualizing the sleep scoring results.
    Includes split spectrogram views and scoring metrics aligned in time.
    Uses 'auto' shading for spectrograms to reduce memory usage.

    Args: (Removed lfp_segmented)
        ... (other args same as V5) ...
    """
    print("--- Generating Scoring Overview Plot (No Raw LFP, Auto Shading) ---")
    if times is None or frequencies is None or spectrogram is None or pc1 is None \
       or theta_dominance is None or emg is None or sleep_states is None:
        print("Error: Cannot generate plot due to missing input data.")
        return

    try:
        # Determine number of panels: 2x Spectrogram + PC1 + Theta + EMG (optional)
        num_panels = 5 if calculate_emg else 4 # <--- Adjusted number of panels
        fig, axs = plt.subplots(nrows=num_panels, ncols=1, figsize=(18, num_panels * 2.3), # Adjusted height
                                sharex=True, constrained_layout=True)
        fig.suptitle(f'Sleep Scoring Overview - {base_filename}', fontsize=16, y=1.02)

        # State colors and labels
        state_colors_bg = {0: 'whitesmoke', 1: 'blue', 2: 'coral'}
        state_labels = {0: 'Awake', 1: 'NREM', 2: 'REM'}
        shading_alpha = 0.45 # Opacity for state background

        if num_panels == 1: axs = [axs] # Ensure axs is iterable

        # --- Calculate shared color limits for spectrograms ---
        spec_db = 10 * np.log10(spectrogram + 1e-18)
        freq_mask_for_clim = (frequencies >= 0) & (frequencies <= 250)
        valid_spec_db_subset = spec_db[freq_mask_for_clim, :]
        if valid_spec_db_subset.size > 0 and np.any(np.isfinite(valid_spec_db_subset)):
             vmin, vmax = np.percentile(valid_spec_db_subset[np.isfinite(valid_spec_db_subset)], [5, 98])
        else: # Fallback if subset is empty or all non-finite
            valid_spec_db_all = spec_db[np.isfinite(spec_db)]
            if valid_spec_db_all.size > 0: vmin, vmax = np.percentile(valid_spec_db_all, [5, 98])
            else: vmin, vmax = (-60, 0)


        # --- Panel 0: Spectrogram (0-20 Hz) --- NOW axs[0]
        ax = axs[0]
        try:
            # *** CHANGED SHADING HERE ***
            img1 = ax.pcolormesh(times, frequencies, spec_db, cmap='viridis', shading='auto', vmin=vmin, vmax=vmax)
        except Exception as e_plot: print(f"Error plotting low-freq spectrogram: {e_plot}"); img1 = None
        ax.set_ylabel('Frequency (Hz)')
        ax.set_ylim(0, 20)
        ax.set_title('Spectrogram (0-20 Hz) & Sleep States') # Add context here

        # --- Panel 1: Spectrogram (100-250 Hz) --- NOW axs[1]
        ax = axs[1]
        try:
            # *** CHANGED SHADING HERE ***
            img2 = ax.pcolormesh(times, frequencies, spec_db, cmap='viridis', shading='auto', vmin=vmin, vmax=vmax)
        except Exception as e_plot: print(f"Error plotting high-freq spectrogram: {e_plot}"); img2 = None
        ax.set_ylabel('Frequency (Hz)')
        ax.set_ylim(100, 250)
        ax.set_title('Spectrogram (100-250 Hz)')
        if img1 or img2:
            mappable = img1 if img1 else img2 # Use one of the images for the mappable
            try: fig.colorbar(mappable, ax=[axs[0], axs[1]], label='Power/Freq (dB/Hz)', aspect=15, shrink=0.7) # Use new indices
            except Exception as e_cb: print(f"Error adding colorbar: {e_cb}")


        # --- Panel 2: PC1 Score --- NOW axs[2]
        ax = axs[2]
        ax.plot(times, pc1, color='black', linewidth=0.8, label='PC1')
        if 'nrem' in thresholds and np.isfinite(thresholds['nrem']):
            ax.axhline(thresholds['nrem'], color='darkblue', linestyle='--', linewidth=1, label=f'NREM Thr ({thresholds["nrem"]:.2f})')
        ax.set_ylabel('PC1 Score')
        ax.legend(loc='upper right', fontsize='small')

        # --- Panel 3: Theta Dominance --- NOW axs[3]
        ax = axs[3]
        ax.plot(times, theta_dominance, color='green', linewidth=0.8, label='Theta Dominance')
        if 'rem' in thresholds and np.isfinite(thresholds['rem']):
            ax.axhline(thresholds['rem'], color='darkred', linestyle='--', linewidth=1, label=f'REM Thr ({thresholds["rem"]:.2f})')
        ax.set_ylabel('Theta Dominance')
        ax.legend(loc='upper right', fontsize='small')

        # --- Panel 4: EMG (Conditional) --- NOW axs[4]
        last_ax_idx = 3 # Index of last plot without EMG
        if calculate_emg:
            last_ax_idx = 4 # Index of last plot with EMG
            ax = axs[4]
            ax.plot(times, emg, color='purple', linewidth=0.8, label='EMG Estimate')
            if 'emg' in thresholds and np.isfinite(thresholds['emg']):
                ax.axhline(thresholds['emg'], color='magenta', linestyle='--', linewidth=1, label=f'EMG Thr ({thresholds["emg"]:.2f})')
            ax.set_ylabel('EMG Estimate')
            ax.legend(loc='upper right', fontsize='small')

        # Add x-axis label to the bottom-most plot
        axs[last_ax_idx].set_xlabel('Time (seconds)')

        # --- Add Sleep State Background Shading to ALL panels ---
        plotted_labels = set()
        # Iterate through ALL axes now
        for i, ax in enumerate(axs):
            ymin, ymax = ax.get_ylim()
            # Find indices where state changes, include start and end virtual changes
            change_indices = np.where(np.diff(sleep_states, prepend=np.nan, append=np.nan))[0]

            for k in range(len(change_indices) - 1):
                start_idx = change_indices[k]
                end_idx = change_indices[k+1]
                if start_idx >= len(sleep_states): continue
                state = sleep_states[start_idx]
                if start_idx >= len(times): continue
                t_start = times[start_idx]
                # Ensure t_end does not exceed the last time point significantly
                t_end = times[end_idx] if end_idx < len(times) else times[-1] + step_size

                # Add state label to legend only once (use first plot, axs[0])
                label = state_labels[state] if i == 0 and state_labels[state] not in plotted_labels else None
                try:
                    ax.fill_between([t_start, t_end], ymin, ymax,
                                      color=state_colors_bg[state], alpha=shading_alpha, label=label,
                                      edgecolor=None, step='post')
                    if label: plotted_labels.add(state_labels[state])
                except Exception as e_fill: print(f"Error during fill_between: {e_fill}")

            # Restore y-limits and set consistent x-limits
            ax.set_ylim(ymin, ymax)
            if len(times) > 0:
                 ax.set_xlim(times[0], times[-1] + step_size)
            else:
                 ax.set_xlim(0, 1)

        # Add combined legend for states to the first plot (axs[0])
        if plotted_labels:
             handles, labels = axs[0].get_legend_handles_labels()
             # Filter out any non-state labels if other things were plotted on axs[0] with labels
             state_handles_labels = [(h,l) for h,l in zip(handles, labels) if l in state_labels.values()]
             axs[0].legend(handles=[h for h,l in state_handles_labels],
                           labels=[l for h,l in state_handles_labels],
                           title="Sleep State", loc='upper right', fontsize='small')


        # --- Save and Close Plot ---
        plot_filename = output_path / f'{base_filename}_ScoringOverview_V6_AutoShade.png' # V6 AutoShade
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        print(f"Saved scoring overview plot: {plot_filename}")
        plt.close(fig)

    except Exception as e:
        print(f"Error generating scoring overview plot V6 for {base_filename}: {e}")
        traceback.print_exc()
        plt.close('all')


# === Main Script Execution ================================================

# --- Parameters ---
print("--- Script Parameters ---")
TARGET_FS_USER_DEFINED = 1250.0 # Desired target sampling rate
cutoff_freq = 450.0         # Low-pass filter cutoff (Hz)
num_channels_to_use = 5     # Number of channels from END of recording
window_size = 10.0          # Spectrogram window size (seconds)
step_size = 1.0             # Spectrogram step size (seconds)
calculate_emg = True        # <<< SET True TO ENABLE EMG CALCULATION >>>
n_ica_components = 10       # Number of ICA components
ica_epochs_duration = 240   # Duration (seconds) for ICA fitting
num_channels_for_ica = 5    # Number of channels for ICA (from selected LFP)
scoring_buffer_size = 5     # Steps to confirm state change
fixed_emg_threshold_val = None # Use None for percentile, or set fixed value
speed_csv_file_path = None  # Set path to enable speed thresholding
video_frame_rate = 30       # Video FPS (if using speed)
video_start_delay_seconds = 0 # Video vs Neural offset (seconds)
mouse_speed_threshold = 1.5 # Speed threshold (cm/s)

# --- User Input: Select Directories ---
root = Tk()
root.withdraw()
root.attributes("-topmost", True)
print("\nPlease select the directory containing LFP binary (*.lf.bin) and meta (.meta) files.")
lfp_bin_meta_dir_str = filedialog.askdirectory(title="Select directory containing LFP binary and meta files")
lfp_bin_meta_dir = Path(lfp_bin_meta_dir_str) if lfp_bin_meta_dir_str else None
print("\nPlease select the directory containing speed CSV files (optional).")
speed_files_dir_str = filedialog.askdirectory(title="Select directory containing speed files (optional)")
speed_files_dir = Path(speed_files_dir_str) if speed_files_dir_str else None
print("\nPlease select the output directory for saving results.")
output_dir_str = filedialog.askdirectory(title="Select output directory for sleep state files")
output_dir = Path(output_dir_str) if output_dir_str else None
root.destroy()

# --- Validation of Inputs ---
if lfp_bin_meta_dir is None or not lfp_bin_meta_dir.is_dir(): print("Error: LFP directory invalid. Exiting."); sys.exit(1)
if output_dir is None: print("Error: Output directory not selected. Exiting."); sys.exit(1)
try: output_dir.mkdir(parents=True, exist_ok=True); print(f"Using output directory: {output_dir}")
except Exception as e: print(f"Error: Output directory invalid or could not be created: {e}. Exiting."); sys.exit(1)

# --- Find LFP Files (Specifically .lf.bin) ---
lfp_files = sorted([p for p in lfp_bin_meta_dir.glob('*.lf.bin') if p.is_file()])
if not lfp_files: print(f"Error: No *.lf.bin files found in: {lfp_bin_meta_dir}. Exiting."); sys.exit(1)
print(f"\nFound {len(lfp_files)} LFP '.lf.bin' files to process:")
for f in lfp_files: print(f" - {f.name}")

# --- Main Processing Loop ---
for lfp_file_path in lfp_files:
    print(f"\n{'='*40}\nProcessing recording: {lfp_file_path.name}\n{'='*40}")
    output_filename_base = lfp_file_path.stem # Base name for output files

    # Initialize variables for this loop iteration
    data_memmap, original_fs, target_fs, data_selected_channels, data_selected_float = None, None, None, None, None
    downsampled_data, filtered_data, averaged_lfp_data = None, None, None
    spectrogram, frequencies, times = None, None, None
    pc1, theta_dominance, emg = None, None, None
    ica_object, ica_components = None, None
    sleep_states = None
    # REMOVED lfp_segmented_for_plot initialization
    nrem_threshold, rem_threshold, emg_threshold = np.nan, np.nan, np.nan
    calculate_emg_this_file = False # Default to false

    try: # Wrap entire file processing in a try...except...finally block
        # Find corresponding meta file
        meta_file_path = lfp_file_path.with_suffix('.meta')
        if not meta_file_path.exists(): print(f"Warning: Meta file not found. Skipping."); continue

        # Find corresponding speed file
        current_speed_file_path = None
        if speed_files_dir and speed_files_dir.is_dir():
            base_name = lfp_file_path.stem.split('.')[0]
            possible_speed_files = list(speed_files_dir.glob(f'{base_name}*.csv'))
            if possible_speed_files:
                current_speed_file_path = possible_speed_files[0]
                print(f"Found corresponding speed file: {current_speed_file_path.name}")

        # 1. Load Data & Get Sampling Rate
        print("\n--- 1. Loading Data ---")
        data_memmap, original_fs = read_bin_data(lfp_file_path, meta_file_path)
        if data_memmap is None or original_fs is None: print(f"Failed load/get SR. Skipping."); continue

        # Set target FS
        target_fs = TARGET_FS_USER_DEFINED
        if abs(original_fs / 2.0 - TARGET_FS_USER_DEFINED) < 1.0:
             target_fs = original_fs / 2.0
             print(f"Using precise half target FS: {target_fs:.6f} Hz")
        else: print(f"Using user-defined target FS: {target_fs:.6f} Hz")

        # 2. Select Channels
        print("\n--- 2. Selecting Channels ---")
        if data_memmap.shape[1] < num_channels_to_use:
            print(f"Warning: Has {data_memmap.shape[1]} < requested {num_channels_to_use}. Using all.")
            num_chans_actual = data_memmap.shape[1]
            if num_chans_actual == 0: print("Error: 0 channels. Skipping."); continue
        else: num_chans_actual = num_channels_to_use
        data_selected_channels = data_memmap[:, -num_chans_actual:]
        print(f"Selected last {num_chans_actual} channels. Shape: {data_selected_channels.shape}, Type: {data_selected_channels.dtype}")

        # 3. Downsample
        print("\n--- 3. Downsampling ---")
        downsampling_factor = int(round(original_fs / target_fs))
        if downsampling_factor < 1: downsampling_factor = 1
        if downsampling_factor == 1:
            print("Downsampling factor is 1. Copying data as float32.")
            try: downsampled_data = data_selected_channels.astype(np.float32)
            except MemoryError: print("MemoryError copying data. Skipping."); continue
            finally: del data_memmap # Release memmap even if only copying
        else:
            print(f"Downsampling factor: {downsampling_factor} (from {original_fs:.6f} Hz to target ~{target_fs:.6f} Hz)")
            print("Converting selected channels to float32 for downsampling...")
            try: data_selected_float = data_selected_channels.astype(np.float32)
            except MemoryError: print(f"MemoryError loading chans as float32. Skipping."); continue
            finally: del data_memmap # Release memmap

            try:
                print(f"Applying signal.decimate (factor={downsampling_factor})...")
                downsampled_data = signal.decimate(data_selected_float, downsampling_factor, axis=0, ftype='fir', zero_phase=True)
                print(f"Shape AFTER downsampling: {downsampled_data.shape}, Type: {downsampled_data.dtype}")
            except MemoryError: print(f"MemoryError during decimate. Skipping."); continue
            except Exception as e: print(f"Error during decimation: {e}. Skipping."); continue
            finally: del data_selected_float # Free intermediate

        # Update target_fs to the actual achieved rate
        effective_target_fs = original_fs / downsampling_factor
        print(f"Effective sampling rate after decimation: {effective_target_fs:.6f} Hz")
        target_fs = effective_target_fs # Use this effective rate going forward

        # Check if downsampled data is valid before proceeding
        if downsampled_data is None or downsampled_data.size == 0:
            print("Error: Downsampled data is invalid. Skipping.")
            continue

        # 4. Filter
        print("\n--- 4. Filtering ---")
        filtered_data = filter_data(downsampled_data, target_fs, cutoff_freq)
        if filtered_data is None: print(f"Filtering failed. Skipping."); continue

        # 5. Average Channels
        print("\n--- 5. Averaging Channels ---")
        averaged_lfp_data = np.mean(filtered_data, axis=1)
        print(f"Shape of averaged LFP data: {averaged_lfp_data.shape}")

        # 6. Compute Spectrogram
        print("\n--- 6. Computing Spectrogram ---")
        spectrogram, frequencies, times = compute_spectrogram(averaged_lfp_data, target_fs, window_size, step_size)
        if spectrogram is None: print(f"Spectrogram failed. Skipping."); continue

        # 7. Compute PCA
        print("\n--- 7. Computing PCA ---")
        pc1 = compute_pca(spectrogram)
        if pc1 is None: print(f"PCA failed. Skipping."); continue

        # 8. Compute Theta Dominance
        print("\n--- 8. Computing Theta Dominance ---")
        theta_dominance = compute_theta_dominance(spectrogram, frequencies)
        if theta_dominance is None: print(f"Theta dominance failed. Skipping."); continue

        # 9. Compute EMG (Conditional)
        print("\n--- 9. Computing EMG (Conditional) ---")
        if calculate_emg:
            num_ch_avail = filtered_data.shape[1]
            actual_num_ch_ica = min(num_channels_for_ica, num_ch_avail)
            actual_n_comp = min(n_ica_components, actual_num_ch_ica)
            if actual_num_ch_ica <= 0 or actual_n_comp <= 0:
                 print("Warning: Not enough channels/components for ICA. Skipping EMG.")
                 emg = np.zeros(len(pc1)) # Dummy
            else:
                 print(f"Calculating EMG: {actual_num_ch_ica} chans, {actual_n_comp} comps.")
                 emg_seg, ica_object, ica_components = compute_emg(
                     filtered_data, target_fs, window_size, step_size,
                     n_ica_components=actual_n_comp, num_channels_for_ica=actual_num_ch_ica,
                     ica_epochs_duration=ica_epochs_duration)

                 if emg_seg is None:
                     print("EMG computation failed. Scoring without EMG.")
                     emg = np.zeros(len(pc1))
                 elif len(emg_seg) == len(pc1):
                     emg = emg_seg
                     calculate_emg_this_file = True # Success!
                 else: # Length mismatch handling
                     print(f"Warning: EMG length ({len(emg_seg)}) != PC1 length ({len(pc1)}). Truncating/padding.")
                     target_len = len(pc1); current_len = len(emg_seg)
                     if current_len > target_len: emg = emg_seg[:target_len]
                     else: emg = np.pad(emg_seg, (0, target_len - current_len))
                     calculate_emg_this_file = True # Still usable
        else:
            print("EMG calculation disabled by user setting.")
            emg = np.zeros(len(pc1)) # Dummy

        # 10. Score Sleep States
        print("\n--- 10. Scoring Sleep States ---")
        if len(pc1) == 0: print("Error: Zero length data before scoring. Skipping."); continue

        # Calculate thresholds needed for scoring and plotting
        nrem_threshold = np.nanpercentile(pc1, 75)
        rem_threshold = np.nanpercentile(theta_dominance, 75)
        if calculate_emg_this_file:
            if fixed_emg_threshold_val is not None: emg_threshold = fixed_emg_threshold_val
            elif np.any(np.isfinite(emg)): emg_threshold = np.nanpercentile(emg[np.isfinite(emg)], 25)
            else: emg_threshold = 0; print("Warning: Cannot calc percentile EMG thresh.")
        else: emg_threshold = np.nan

        sleep_states = score_sleep_states(
            pc1, theta_dominance, emg, times, step_size,
            buffer_size=scoring_buffer_size,
            fixed_emg_threshold=fixed_emg_threshold_val,
            speed_csv_path=str(current_speed_file_path) if current_speed_file_path else None,
            video_fps=video_frame_rate, video_start_offset_seconds=video_start_delay_seconds,
            speed_threshold=mouse_speed_threshold,
            use_emg=calculate_emg_this_file
        )
        if sleep_states is None: print(f"Sleep scoring failed. Skipping save/plot."); continue

        # 11. Save Results
        print("\n--- 11. Saving Results ---")
        output_suffix = "_EMG" if calculate_emg_this_file else "_NoEMG"
        state_filename = f'{output_filename_base}_sleep_states{output_suffix}.npy'
        times_filename = f'{output_filename_base}_sleep_state_times{output_suffix}.npy'
        try:
            np.save(output_dir / state_filename, sleep_states)
            print(f"Saved states: {output_dir / state_filename}")
            np.save(output_dir / times_filename, times)
            print(f"Saved times: {output_dir / times_filename}")
        except Exception as e: print(f"Error saving results: {e}")

        # 12. Visualization
        print("\n--- 12. Visualization ---")
        # REMOVED LFP Segmentation Calculation
        plot_thresholds = {'nrem': nrem_threshold, 'rem': rem_threshold, 'emg': emg_threshold}
        # Check if all data needed for plot exists
        if (times is not None and frequencies is not None and spectrogram is not None and
            pc1 is not None and theta_dominance is not None and emg is not None and
            sleep_states is not None): # REMOVED lfp_segmented check

            plot_scoring_overview( # CALL THE UPDATED PLOT FUNCTION (V6)
                times=times, frequencies=frequencies, spectrogram=spectrogram,
                pc1=pc1, theta_dominance=theta_dominance, emg=emg,
                sleep_states=sleep_states, thresholds=plot_thresholds,
                calculate_emg=calculate_emg_this_file, step_size=step_size,
                # REMOVED lfp_segmented argument
                output_path=output_dir, base_filename=output_filename_base
            )
        else:
            print("Skipping overview plot due to missing data required for plotting.")


    except Exception as e_main:
        print(f"\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"FATAL ERROR PROCESSING FILE: {lfp_file_path.name}")
        print(f"Error details: {e_main}")
        traceback.print_exc() # Print full traceback to log file
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")

    finally:
        # --- Clean up memory before next loop ---
        print("\n--- Cleaning up memory for next file ---")
        vars_to_del = [
            'data_memmap', 'data_selected_channels', 'data_selected_float',
            'downsampled_data', 'filtered_data', 'averaged_lfp_data',
            'spectrogram', 'frequencies', 'times', 'pc1', 'theta_dominance', 'emg',
            'sleep_states', 'ica_object', 'ica_components', 'plot_thresholds',
            # REMOVED 'lfp_segmented_for_plot'
            # REMOVED 'filtered_lfp_last_chan_data'
        ]
        deleted_count = 0
        local_vars = locals() # Get local symbol table
        for var_name in vars_to_del:
            if var_name in local_vars: # Check if it exists in scope
                try:
                    del local_vars[var_name] # Delete from scope
                    deleted_count += 1
                except Exception as del_e:
                     print(f"Note: Could not delete variable {var_name}: {del_e}")
        collected = gc.collect()
        print(f"Attempted delete on {deleted_count} vars. Garbage collection: {collected} objects cleared.")


# --- End of Script ---
print(f"\n{'='*40}\nScript Execution Completed\n{'='*40}")
if 'lfp_files' in locals() and lfp_files is not None: print(f"Processed {len(lfp_files)} files.")
else: print("No files were processed.")
if output_dir: print(f"Results saved in: {output_dir}")
print(f"Detailed logs are in: {output_file_path}")

# Restore stdout/stderr
if isinstance(sys.stdout, io.IOBase):
    if sys.stdout != original_stdout:
        try: sys.stdout.close()
        except Exception as e: print(f"(Info) Error closing redirected stdout: {e}", file=original_stdout)

if isinstance(sys.stderr, io.IOBase) and sys.stderr != sys.stdout:
    if sys.stderr != original_stderr:
        try: sys.stderr.close()
        except Exception as e: print(f"(Info) Error closing redirected stderr: {e}", file=original_stderr)

sys.stdout = original_stdout
sys.stderr = original_stderr
print(f"\nScript execution completed. Check logs in '{output_file_path}' and results in '{output_dir}'.")