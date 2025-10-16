# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 2025 (Visualization added, refined V10 - pcolormesh nearest)
State scoring based on Supplementary Materials
Wannan Yang Science 383,1478(2024)_Buzsaki Lab sleep state score

EMG extraction (from 300-600Hz using ICA method) https://doi.org/10.1016/j.crmeth.2023.100482
Osanai, H., Yamamoto, J., & Kitamura, T. (2023). Cell Reports Methods, 3(6).
and Liu, K., Sibille, J. & Dragoi, G. Nat Neurosci 27, 1816–1828 (2024). https://doi.org/10.1038/s41593-024-01703-6
@author: Alok 

Applies downsampling (to 1250Hz or other target) BEFORE filtering.
Uses scipy.signal.decimate for proper anti-aliased downsampling.
Generates detailed overview plots for each recording,
including split spectrograms and scoring metrics aligned in time.
Handles multiple epochs based on timestamp files and pads sleep state data.
Fixes timestamp matching, skew calculation, and plotting errors.
Addresses memory errors during plot saving by optimizing pcolormesh.
Uses pcolormesh with shading='nearest' to avoid dimension mismatch errors.
"""
import io
import numpy as np
from scipy import signal
from scipy import stats # For skewness
from sklearn.decomposition import PCA, FastICA
from scipy.stats import zscore
from tkinter import Tk
from tkinter import filedialog
from pathlib import Path
import re # For filename matching
from DemoReadSGLXData.readSGLX import readMeta

import os
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import math
import gc
import traceback

# --- Redirect stdout and stderr to a file ---
output_file_path = "script_output_epochs_EMGCont.txt" # Changed filename
original_stdout = sys.stdout
original_stderr = sys.stderr
# Clear previous log file if it exists
if os.path.exists(output_file_path):
    try:
        os.remove(output_file_path)
    except OSError as e:
        print(f"Warning: Could not remove previous log file {output_file_path}: {e}")

try:
    # os.makedirs(os.path.dirname(output_file_path), exist_ok=True) # Uncomment if output_file_path includes a directory
    sys.stdout = open(output_file_path, 'w', encoding='utf-8')
    sys.stderr = sys.stdout # Redirect stderr to the same file
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
    Reads binary data from a file using memory-mapping. Args:
        file_path (str): Path to the binary file. meta_path (str): Path to the corresponding metadata file. data_type (str): Data type of the samples. Returns:
        tuple: (numpy.ndarray: Memory-mapped data, float: sampling rate) or (None, None) on error. """
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
         if data is not None and hasattr(data, '_mmap'):
            try: data._mmap.close()
            except Exception: pass
         return None, sampling_rate
    except Exception as e:
        print(f"An unexpected error occurred in read_bin_data for {file_path}: {e}")
        traceback.print_exc()
        if data is not None and hasattr(data, '_mmap'):
            try: data._mmap.close()
            except Exception: pass
        return None, None

def filter_data(data, fs, cutoff_freq):
    """
    Filters data using a FIR filter (scipy.signal.firwin and lfilter). Args:
        data (numpy.ndarray): Data to filter (should be float type). fs (float): Sampling rate OF THE INPUT DATA. cutoff_freq (float): Cutoff frequency for the low-pass filter. Returns:
        numpy.ndarray: Filtered data (typically float64) or None on error. """
    print(f"Shape of data entering filter_data: {data.shape}")
    print(f"Data type entering filter_data: {data.dtype}")
    print(f"Filtering with fs={fs:.6f} Hz, cutoff={cutoff_freq} Hz")

    if data is None or data.size == 0:
         print("Error: Cannot filter empty data.")
         return None

    # Ensure data is float32 for filtering to save memory, only copy if necessary
    data_float = None # Initialize
    try:
        if not np.issubdtype(data.dtype, np.floating):
            print("Warning: Data type is not float, converting to float32 for filtering.")
            data_float = data.astype(np.float32)
        elif data.dtype != np.float32:
            print(f"Data type is {data.dtype}, converting to float32 for filtering.")
            data_float = data.astype(np.float32)
        else:
            data_float = data # No conversion needed if already float32
    except MemoryError:
        print("MemoryError converting data to float32 inside filter_data.")
        return None
    except Exception as e:
        print(f"Error converting data to float32 inside filter_data: {e}")
        return None

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
    data_len = data_float.shape[0] if data_float.ndim > 0 else 0

    if numtaps >= data_len: # Use >= because filter needs length > order
        print(f"Warning: Filter order ({numtaps}) >= data length ({data_len}). Reducing filter order.")
        numtaps = (data_len // 2) * 2 - 1 # Make it odd and < data length
        if numtaps < 3:
             print(f"Error: Data too short ({data_len}) for filtering (minimum numtaps=3).")
             return None

    if numtaps % 2 == 0: numtaps += 1 # Ensure odd for Type I

    try:
        taps = signal.firwin(numtaps=numtaps, cutoff=normalized_cutoff_freq, window='hamming', pass_zero='lowpass')
        print(f"Shape of filter taps: {taps.shape}")

        # Apply filter channel by channel (axis=0: time axis)
        # Use filtfilt for zero-phase filtering if desired and possible (doubles filter order effectively)
        # filtered_data = signal.filtfilt(taps.astype(np.float32), 1.0, data_float, axis=0)
        # Or stick to lfilter for causal filtering as before
        filtered_data = signal.lfilter(taps.astype(np.float32), 1.0, data_float, axis=0)

        print(f"Shape of filtered data exiting filter_data: {filtered_data.shape}")
        print(f"Data type of filtered data exiting filter_data: {filtered_data.dtype}") # Often float64 from lfilter/filtfilt
        return filtered_data
    except MemoryError:
        print("MemoryError during signal.lfilter/filtfilt.")
        return None
    except Exception as e:
        print(f"Error during filtering: {e}")
        traceback.print_exc()
        return None

def compute_spectrogram(data, fs, window_size, step_size):
    """
    Computes spectrogram using scipy.signal.spectrogram. Args:
        data (numpy.ndarray): 1D data array. fs (float): Sampling rate. window_size (float): Window size in seconds. step_size (float): Step size in seconds. Returns:
        tuple: (Spectrogram, frequencies, times) or (None, None, None) on error. """
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
             # Option: Use shorter window? Or return error. Sticking to error.
             return None, None, None

        noverlap = window_samples - step_samples
        if noverlap < 0:
            print(f"Warning: Step size ({step_samples}) > window size ({window_samples}). Setting overlap to 0.")
            noverlap = 0
        # Ensure non-negative overlap
        noverlap = max(0, noverlap)

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
        # Check if first time deviates significantly from expected window center
        if len(times)>0 and abs(times[0] - window_size/2.0) > step_size:
             print(f"Warning: First spectrogram time bin starts at {times[0]:.3f}s (expected ~{window_size/2.0:.3f}s). Check parameters.")

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
    Performs PCA on z-scored spectrogram data. Args:
        data (numpy.ndarray): Spectrogram data (frequencies x times). Returns:
        numpy.ndarray: First principal component (1D array, length = n_times) or None on error. """
    print(f"Computing PCA on data of shape: {data.shape}")
    if data is None or data.size == 0 or data.shape[0] < 2 or data.shape[1] < 2:
        print("Error: Invalid data for PCA (None, empty, or too small).")
        return None
    try:
        # Replace non-finite values before z-scoring
        if not np.all(np.isfinite(data)):
             print("Warning: Non-finite values (NaN/Inf) found before PCA. Replacing with zeros.")
             data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        # Z-score along frequency axis (axis=1, across time for each freq)
        zscored_data = None
        try:
            std_dev = np.std(data, axis=1, ddof=0, keepdims=True)
            mean_val = np.mean(data, axis=1, keepdims=True)
            valid_mask = (std_dev > 1e-9).flatten()
            zscored_data = np.zeros_like(data)

            if np.any(valid_mask):
                zscored_data[valid_mask,:] = (data[valid_mask,:] - mean_val[valid_mask,:]) / std_dev[valid_mask,:]
            else:
                print("Warning: All frequency bins have zero/near-zero variance across time. Cannot z-score.")
                return None

            num_zero_var = data.shape[0] - np.sum(valid_mask)
            if num_zero_var > 0:
                 print(f"Warning: {num_zero_var} frequency bins have zero/near-zero variance across time. Z-score set to 0 for these.")

            if np.any(np.isnan(zscored_data)):
                print("Warning: NaNs found after z-scoring. Replacing NaNs with 0.")
                zscored_data = np.nan_to_num(zscored_data)
        except Exception as e_zscore:
             print(f"Error during z-scoring before PCA: {e_zscore}")
             traceback.print_exc()
             return None

        pca = PCA(n_components=1)
        pc1 = pca.fit_transform(zscored_data.T)
        print(f"PCA computed. PC1 shape: {pc1.shape}, Explained variance: {pca.explained_variance_ratio_[0]:.4f}")
        if pc1.ndim == 2 and pc1.shape[1]==1:
             return pc1.squeeze()
        else:
             print(f"Error: Unexpected PC1 shape: {pc1.shape}")
             return None


    except MemoryError:
        print("MemoryError during PCA computation.")
        return None
    except Exception as e:
        print(f"Error computing PCA: {e}")
        traceback.print_exc()
        return None

def compute_theta_dominance(spectrogram, frequencies):
    """
    Computes theta dominance from spectrogram. Args:
        spectrogram (numpy.ndarray): Spectrogram data (frequencies x times). frequencies (numpy.ndarray): Frequencies corresponding to spectrogram.
    Returns:
        numpy.ndarray: Theta dominance (1D array, length = n_times) or None on error. """
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
             return np.full(spectrogram.shape[1], np.nan)

        epsilon = 1e-12
        theta_power = np.nanmean(spectrogram[theta_band_indices, :], axis=0)
        total_power = np.nanmean(spectrogram[total_band_indices, :], axis=0)

        print(f"Shape of theta_power: {theta_power.shape}")
        print(f"Shape of total_power: {total_power.shape}")

        theta_dominance = np.full_like(theta_power, np.nan)
        valid_mask = (np.isfinite(theta_power)) & (np.isfinite(total_power)) & (total_power > epsilon)

        theta_dominance[valid_mask] = np.divide(theta_power[valid_mask], total_power[valid_mask])

        print(f"Shape of theta_dominance: {theta_dominance.shape}")

        num_non_finite = np.sum(~np.isfinite(theta_dominance))
        if num_non_finite > 0:
            print(f"Warning: {num_non_finite} non-finite (NaN) values found in theta dominance.")

        return theta_dominance
    except Exception as e:
        print(f"Error computing Theta Dominance: {e}")
        traceback.print_exc()
        return None

def compute_emg(data, fs, window_size, step_size, n_ica_components=10, ica_epochs_duration=240, num_channels_for_ica=48):
    """
    Estimates EMG from LFP data using ICA (FastICA). Args: (See previous version)
    Returns: tuple: (Segmented EMG, ICA object, ICA source signals) or (None, None, None) on error. """
    print("\n---ICA-based EMG Computation (scikit-learn FastICA)---")
    print(f"Input data shape for EMG: {data.shape}, fs={fs:.6f}")

    if data is None or data.size == 0:
        print("Error: Input data for EMG computation is invalid.")
        return None, None, None

    ica_object_fitted = None
    full_source_signals = None

    try:
        actual_channels_available = data.shape[1] if data.ndim == 2 else 1
        if actual_channels_available == 0:
             print("Error: Input data for EMG has 0 channels.")
             return None, None, None

        channels_to_use = min(num_channels_for_ica, actual_channels_available)
        n_ica_components = min(n_ica_components, channels_to_use)

        if n_ica_components <= 0:
             print("Error: Number of ICA components must be positive.")
             return None, None, None

        if data.ndim == 1:
             print("Warning: Input data is 1D. Cannot apply ICA for EMG estimation. Using segmented absolute value.")
             emg_signal_raw = np.abs(data)
             ica_object_fitted = None
             full_source_signals = None
        else:
             ica_channels_indices = np.arange(channels_to_use)
             ica_data = data[:, ica_channels_indices]
             print(f"Applying ICA to first {channels_to_use} channels.")
             print(f"Shape of data selected for ICA: {ica_data.shape}")

             ica_n_samples = int(round(ica_epochs_duration * fs))
             ica_n_samples = min(ica_n_samples, ica_data.shape[0])

             if ica_n_samples < n_ica_components:
                  print(f"Error: Not enough samples ({ica_n_samples}) for {n_ica_components} ICA components.")
                  return None, None, None

             ica_data_subset = ica_data[:ica_n_samples, :].astype(np.float32)
             print(f"Using {ica_n_samples} samples ({ica_data_subset.shape[0] / fs:.2f} seconds) for ICA fitting.")

             emg_cutoff_low_ica = 600
             emg_cutoff_high_ica = 1200
             nyquist_freq_ica = fs / 2.0
             normalized_cutoff_low_ica = emg_cutoff_low_ica / nyquist_freq_ica if nyquist_freq_ica > 0 else 0
             normalized_cutoff_high_ica = emg_cutoff_high_ica / nyquist_freq_ica if nyquist_freq_ica > 0 else 0

             ica_filtered_data_subset = None
             ica_filter_taps = None
             pre_filter_applied = False
             if 0 < normalized_cutoff_low_ica < normalized_cutoff_high_ica < 1.0:
                 numtaps_ica = 31
                 if numtaps_ica >= ica_data_subset.shape[0]: numtaps_ica = (ica_data_subset.shape[0] // 2) * 2 - 1
                 if numtaps_ica < 3: numtaps_ica = 3

                 if numtaps_ica >= 3:
                     print(f"Bandpass filtering ICA subset: {emg_cutoff_low_ica}-{emg_cutoff_high_ica} Hz (Numtaps: {numtaps_ica})")
                     try:
                         ica_filter_taps = signal.firwin(numtaps_ica, [normalized_cutoff_low_ica, normalized_cutoff_high_ica],
                                                         pass_zero='bandpass', window='hamming')
                         ica_filtered_data_subset = signal.filtfilt(ica_filter_taps.astype(np.float32), 1.0, ica_data_subset, axis=0)
                         pre_filter_applied = True
                     except Exception as filter_err:
                         print(f"Warning: Error during ICA subset bandpass filtering: {filter_err}. Using unfiltered.")
                         ica_filtered_data_subset = None
                 else: print("Warning: ICA subset too short for bandpass filter. Using unfiltered.")
             else: print("Warning: Invalid EMG freq band for ICA pre-filter or disabled. Using unfiltered.")

             if ica_filtered_data_subset is None:
                 ica_filtered_data_subset = ica_data_subset

             ica_object_fitted = FastICA(n_components=n_ica_components, random_state=42, whiten='unit-variance', max_iter=500, tol=1e-3)
             print(f"Running FastICA with {n_ica_components} components on {channels_to_use} channels...")
             try:
                 ica_source_signals_subset = ica_object_fitted.fit_transform(ica_filtered_data_subset)
             except Exception as ica_fit_err:
                  print(f"Error during FastICA fit_transform: {ica_fit_err}")
                  return None, None, None
             print("FastICA fitted and transformed subset.")
             print("Shape of ICA source signals (subset):", ica_source_signals_subset.shape)

             emg_ic_index = 0
             if ica_source_signals_subset is not None and ica_source_signals_subset.shape[1] > 0:
                 try:
                     skewness = np.abs(stats.skew(ica_source_signals_subset, axis=0))
                     if np.any(np.isnan(skewness)):
                         print("Warning: NaN encountered in skewness calculation. Defaulting to IC 0.")
                         emg_ic_index = 0
                     else:
                         emg_ic_index = np.argmax(skewness)
                         print(f"Selected IC {emg_ic_index} based on max absolute skewness ({skewness[emg_ic_index]:.3f}) from subset.")
                 except Exception as skew_err:
                     print(f"Warning: Error calculating skewness for IC selection: {skew_err}. Defaulting to IC 0.")
                     emg_ic_index = 0
             else:
                 print("Warning: Cannot select IC based on skewness (no subset signals). Defaulting to IC 0.")
             del ica_filtered_data_subset, ica_data_subset

             print("Applying learned ICA transform to full dataset...")
             ica_data_float = ica_data.astype(np.float32)
             if not ica_data_float.flags['C_CONTIGUOUS']:
                 ica_data_float = np.ascontiguousarray(ica_data_float)

             if pre_filter_applied and ica_filter_taps is not None:
                  print(f"Applying bandpass filter ({emg_cutoff_low_ica}-{emg_cutoff_high_ica} Hz) to full data before ICA transform (using filtfilt)...")
                  try:
                      ica_filtered_data_full = signal.filtfilt(ica_filter_taps.astype(np.float32), 1.0, ica_data_float, axis=0)
                      full_source_signals = ica_object_fitted.transform(ica_filtered_data_full)
                      del ica_filtered_data_full
                  except MemoryError:
                       print("MemoryError filtering full data for ICA transform. Cannot proceed.")
                       return None, ica_object_fitted, None
                  except Exception as filter_full_err:
                       print(f"Error filtering full data for ICA: {filter_full_err}. Trying without filter.")
                       full_source_signals = ica_object_fitted.transform(ica_data_float)
             else:
                  full_source_signals = ica_object_fitted.transform(ica_data_float)
             del ica_data_float, ica_data

             print("Shape of full source signals:", full_source_signals.shape)
             emg_ic_full = full_source_signals[:, emg_ic_index]
             emg_signal_raw = np.abs(emg_ic_full)

        # --- Segment and Average EMG ---
        print("Shape of raw EMG signal (before segmentation):", emg_signal_raw.shape)
        print(f"Range of raw EMG: Min={np.min(emg_signal_raw):.4f}, Max={np.max(emg_signal_raw):.4f}")

        window_samples = int(round(window_size * fs))
        step_samples = int(round(step_size * fs))

        num_segments_raw = 0
        if len(emg_signal_raw) >= window_samples :
            num_segments_raw = (len(emg_signal_raw) - window_samples) // step_samples + 1
        else:
            print("Warning: EMG data length < window size. Cannot create segments.")
            return None, ica_object_fitted, full_source_signals

        if num_segments_raw <= 0:
             print("Error: Not enough EMG data to create segments.")
             return None, ica_object_fitted, full_source_signals

        emg_segmented = np.zeros(num_segments_raw)
        print(f"Segmenting EMG into {num_segments_raw} segments (based on raw length)...")
        for i in range(num_segments_raw):
            start_sample = i * step_samples
            end_sample = min(start_sample + window_samples, len(emg_signal_raw))
            start_sample = min(start_sample, end_sample)

            segment = emg_signal_raw[start_sample:end_sample]
            if segment.size > 0:
                emg_segmented[i] = np.mean(segment)
            else:
                emg_segmented[i] = np.nan

        print("Shape of segmented EMG (before alignment check):", emg_segmented.shape)
        print(f"Range of segmented EMG: Min={np.nanmin(emg_segmented):.4f}, Max={np.nanmax(emg_segmented):.4f}")

        return emg_segmented, ica_object_fitted, full_source_signals

    except MemoryError:
        print("MemoryError during EMG computation.")
        return None, ica_object_fitted, full_source_signals
    except Exception as e:
        print(f"An error occurred during EMG computation: {e}")
        traceback.print_exc()
        return None, ica_object_fitted, full_source_signals

def score_sleep_states(pc1, theta_dominance, emg, times,
                       step_size, buffer_size=10, fixed_emg_threshold=None,
                       speed_csv_path=None, video_fps=30, video_start_offset_seconds=0,
                       speed_threshold=1.5, use_emg=False):
    """
    Scores sleep states based on provided metrics with sticky thresholds. Args: (See previous definitions)
    Returns: numpy.ndarray: Sleep state labels (0=Awake, 1=NREM, 2=REM) or None on error. """
    print("\n--- Scoring Sleep States ---")
    if pc1 is None or theta_dominance is None or emg is None or times is None:
        print("Error: One or more input metrics for scoring are None.")
        return None
    num_time_points = len(pc1)
    if num_time_points == 0: print("Error: Input metrics are empty."); return None
    if not (len(theta_dominance) == num_time_points and len(emg) == num_time_points and len(times) == num_time_points):
         print("Error: Length mismatch between input metrics/times for scoring.")
         print(f"  PC1: {len(pc1)}, Theta: {len(theta_dominance)}, EMG: {len(emg)}, Times: {len(times)}")
         return None

    print(f"Scoring based on {num_time_points} time points.")

    # --- Calculate Thresholds ---
    try:
        pc1_finite = pc1[np.isfinite(pc1)]
        theta_finite = theta_dominance[np.isfinite(theta_dominance)]
        nrem_threshold = np.nanpercentile(pc1_finite, 75) if len(pc1_finite)>0 else np.nan
        rem_threshold = np.nanpercentile(theta_finite, 75) if len(theta_finite)>0 else np.nan
        local_emg_threshold = np.nan
    except Exception as e:
        print(f"Error calculating NREM/REM percentiles: {e}. Cannot score.")
        return None

    if use_emg:
        if fixed_emg_threshold is not None:
            local_emg_threshold = fixed_emg_threshold
            print(f"Using fixed EMG threshold for scoring logic: {local_emg_threshold:.4f}")
        else:
             emg_finite = emg[np.isfinite(emg)]
             if len(emg_finite) > 0:
                 try:
                    local_emg_threshold = np.nanpercentile(emg_finite, 25)
                    print(f"Using percentile EMG threshold for scoring logic: {local_emg_threshold:.4f}")
                 except Exception as e_emg_thresh:
                    print(f"Warning: Error calculating percentile EMG threshold: {e_emg_thresh}. Using 0.")
                    local_emg_threshold = 0
             else:
                 local_emg_threshold = 0
                 print("Warning: EMG data is all NaN. Using EMG threshold=0 for logic.")

    else:
        print("EMG usage disabled for scoring logic.")
        local_emg_threshold = np.nan

    if np.isnan(nrem_threshold) or np.isnan(rem_threshold):
        print("Error: NREM or REM threshold calculation resulted in NaN. Cannot score.")
        return None
    if use_emg and np.isnan(local_emg_threshold):
        print("Warning: EMG threshold is NaN even though use_emg=True. Treating EMG as unusable for scoring.")
        use_emg = False

    print("---Thresholds Used for Scoring Logic---")
    print(f"NREM Threshold (PC1 >): {nrem_threshold:.4f}")
    print(f"REM Threshold (Theta >): {rem_threshold:.4f}")
    if use_emg: print(f"EMG Threshold (< for NREM/REM): {local_emg_threshold:.4f}")
    else: print(f"EMG Threshold: N/A (use_emg=False or threshold NaN)")

    # --- Load and process speed data ---
    speed_data_interp = None
    if speed_csv_path:
        print(f"Loading speed data from: {speed_csv_path}")
        try:
            speed_df = pd.read_csv(speed_csv_path)
            if 'speed' in speed_df.columns and not speed_df['speed'].isnull().all():
                speed_values = speed_df['speed'].values
                speed_time_rel = np.arange(len(speed_values)) / video_fps
                speed_time_abs = speed_time_rel + video_start_offset_seconds
                speed_data_interp = np.interp(times, speed_time_abs, speed_values, left=np.nan, right=np.nan)
                print(f"Speed data loaded and interpolated. Speed Threshold (> Aw): {speed_threshold:.2f} cm/s")
                num_nan = np.sum(np.isnan(speed_data_interp))
                if num_nan > 0: print(f"Warning: {num_nan}/{len(speed_data_interp)} speed points are NaN after interpolation.")
            else: print(f"Warning: 'speed' column missing or empty in {speed_csv_path}. Disabling speed.")
        except Exception as e: print(f"Error loading/processing speed CSV: {e}. Disabling speed.")
    else: print("No speed file provided. Speed thresholding disabled.")
    print("----------------------")

    # --- Scoring Logic ---
    sleep_states = np.zeros(num_time_points, dtype=int)
    current_state = 0
    nrem_count, rem_count, awake_count = 0, 0, 0
    verbose_scoring = False

    for i in range(num_time_points):
        pc1_val, theta_val, emg_val = pc1[i], theta_dominance[i], emg[i]

        is_speed_high = False
        if speed_data_interp is not None and i < len(speed_data_interp) and np.isfinite(speed_data_interp[i]):
            is_speed_high = speed_data_interp[i] > speed_threshold

        pc1_above_nrem = np.isfinite(pc1_val) and pc1_val > nrem_threshold
        theta_above_rem = np.isfinite(theta_val) and theta_val > rem_threshold

        emg_below_thresh = (not use_emg) or (np.isfinite(emg_val) and emg_val < local_emg_threshold)
        emg_above_thresh = use_emg and (np.isfinite(emg_val) and emg_val > local_emg_threshold)


        if verbose_scoring:
             print(f"i={i}, State={current_state}, PC1={pc1_val:.2f}(>{nrem_threshold:.2f}?{pc1_above_nrem}), Theta={theta_val:.2f}(>{rem_threshold:.2f}?{theta_above_rem}), EMG={emg_val:.2f}(<{local_emg_threshold:.2f}?{emg_below_thresh}, >?{emg_above_thresh}), SpeedHigh?{is_speed_high}")

        if current_state == 0:  # Awake
            nrem_condition = pc1_above_nrem and emg_below_thresh and (not is_speed_high)
            rem_condition = theta_above_rem and emg_below_thresh and (not is_speed_high)
            nrem_count = nrem_count + 1 if nrem_condition else 0
            rem_count = rem_count + 1 if rem_condition else 0
            if nrem_count >= buffer_size: current_state = 1; nrem_count=0; rem_count=0; awake_count=0
            elif rem_count >= buffer_size: current_state = 2; nrem_count=0; rem_count=0; awake_count=0
        elif current_state == 1:  # NREM
            awake_condition = (not pc1_above_nrem) or emg_above_thresh or is_speed_high
            awake_count = awake_count + 1 if awake_condition else 0
            if awake_count >= buffer_size: current_state = 0; nrem_count=0; rem_count=0; awake_count=0
        elif current_state == 2:  # REM
            awake_condition = (not theta_above_rem) or emg_above_thresh or is_speed_high
            awake_count = awake_count + 1 if awake_condition else 0
            if awake_count >= buffer_size: current_state = 0; nrem_count=0; rem_count=0; awake_count=0

        sleep_states[i] = current_state

    print("--- Sleep Scoring Complete ---")
    unique, counts = np.unique(sleep_states, return_counts=True)
    state_dict = dict(zip(unique, counts))
    total_counts = len(sleep_states)
    print("State Distribution:")
    for state_code, state_name in {0:"Awake", 1:"NREM", 2:"REM"}.items():
        count = state_dict.get(state_code, 0)
        percent = (count / total_counts) * 100 if total_counts > 0 else 0
        print(f"  {state_name}: {count} bins ({percent:.1f}%)")

    return sleep_states

def plot_scoring_overview(original_times, frequencies, spectrogram, pc1, theta_dominance, emg,
                          padded_sleep_states, padded_times, # Use original times for metrics/spec, padded for states
                          thresholds, calculate_emg, step_size,
                          epoch_boundaries, total_duration_sec, # Added epoch info
                          output_path, base_filename):
    """
    Generates and saves a multi-panel plot visualizing the sleep scoring results.
    Includes split spectrogram views, scoring metrics, and epoch boundaries. Args: (Updated)
        original_times (np.ndarray): Time vector for spectrogram and metrics (centers).
        frequencies (np.ndarray): Frequency vector for spectrogram (centers).
        ...
        padded_sleep_states (np.ndarray): Sleep state vector potentially padded to total_duration.
        padded_times (np.ndarray): Time vector corresponding to padded_sleep_states.
        ... (other args same as V9) ...
    """
    print("--- Generating Scoring Overview Plot (V10 - pcolormesh nearest) ---")
    # Validate necessary inputs for plotting
    if original_times is None or frequencies is None or spectrogram is None or pc1 is None \
       or theta_dominance is None or emg is None or padded_sleep_states is None or padded_times is None:
        print("Error: Cannot generate plot due to missing input data for V10.")
        return
    if total_duration_sec is None or total_duration_sec <=0:
         print("Error: Invalid total_duration_sec for plotting.")
         total_duration_sec = padded_times[-1] if len(padded_times)>0 else 1.0
    # Check if spectrogram dimensions match original_times and frequencies length before proceeding
    if spectrogram.shape[0] != len(frequencies) or spectrogram.shape[1] != len(original_times):
         print(f"Error: Mismatch between spectrogram dimensions ({spectrogram.shape}) and times/freq lengths ({len(original_times)}, {len(frequencies)}). Cannot plot.")
         return

    try:
        # Determine number of panels
        num_panels = 5 if calculate_emg else 4
        fig, axs = plt.subplots(nrows=num_panels, ncols=1, figsize=(18, num_panels * 2.3),
                                sharex=True, constrained_layout=True)
        try:
             fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, hspace=0.01, wspace=0.01)
        except Exception: pass

        fig.suptitle(f'Sleep Scoring Overview - {base_filename}', fontsize=16, y=1.02)

        # State colors and labels
        state_colors_bg = {0: 'whitesmoke', 1: 'lightblue', 2: 'lightcoral'}
        state_labels = {0: 'Awake', 1: 'NREM', 2: 'REM'}
        shading_alpha = 0.45

        if num_panels == 1: axs = [axs]

        # --- Calculate shared color limits ---
        spec_db = 10 * np.log10(spectrogram + 1e-18)
        valid_spec_db = spec_db[np.isfinite(spec_db)]
        if valid_spec_db.size > 0:
             freq_mask_for_clim = (frequencies >= 0) & (frequencies <= 250)
             if np.any(freq_mask_for_clim):
                 spec_db_subset = spec_db[freq_mask_for_clim, :]
                 valid_spec_db_subset = spec_db_subset[np.isfinite(spec_db_subset)]
                 if valid_spec_db_subset.size > 50:
                     vmin, vmax = np.percentile(valid_spec_db_subset, [1, 99])
                 else:
                     vmin, vmax = np.percentile(valid_spec_db, [1, 99])
             else:
                 vmin, vmax = np.percentile(valid_spec_db, [1, 99])
        else:
             vmin, vmax = (-60, 0)

        # --- Panels 0 & 1: Spectrogram --- Use original_times/frequencies ---
        # --- FIX: Use shading='nearest' ---
        for i_spec, ax in enumerate(axs[:2]): # Loop through first two axes for spectrograms
            is_low_freq = (i_spec == 0)
            ylim_spec = (0, 20) if is_low_freq else (100, 250)
            title_spec = 'Spectrogram (0-20 Hz) & Sleep States & Epochs' if is_low_freq else 'Spectrogram (100-250 Hz)'

            try:
                 # Pass original center coordinates with shading='nearest'
                 img = ax.pcolormesh(original_times, frequencies, spec_db,
                                      cmap='viridis', shading='nearest', # Changed shading
                                      vmin=vmin, vmax=vmax, rasterized=True)
                 if is_low_freq: img1 = img
                 else: img2 = img
            except Exception as e_plot:
                 print(f"Error plotting {'low' if is_low_freq else 'high'}-freq spectrogram: {e_plot}")
                 if is_low_freq: img1 = None
                 else: img2 = None

            ax.set_ylabel('Frequency (Hz)')
            ax.set_ylim(ylim_spec)
            ax.set_title(title_spec)

        # Add colorbar shared between spectrograms
        if 'img1' in locals() or 'img2' in locals():
            mappable = img1 if 'img1' in locals() and img1 is not None else (img2 if 'img2' in locals() and img2 is not None else None)
            if mappable:
                try: fig.colorbar(mappable, ax=axs[:2], label='Power/Freq (dB/Hz)', aspect=15, shrink=0.7)
                except Exception as e_cb: print(f"Error adding colorbar: {e_cb}")


        # --- Panel 2: PC1 Score --- Use original_times ---
        ax = axs[2]
        if len(original_times) == len(pc1):
             ax.plot(original_times, pc1, color='black', linewidth=0.8, label='PC1')
        else: print(f"Plot Error: PC1 length ({len(pc1)}) != original_times length ({len(original_times)})")
        if 'nrem' in thresholds and np.isfinite(thresholds['nrem']):
            ax.axhline(thresholds['nrem'], color='darkblue', linestyle='--', linewidth=1, label=f'NREM Thr ({thresholds["nrem"]:.2f})')
        ax.set_ylabel('PC1 Score')
        ax.legend(loc='upper right', fontsize='small')

        # --- Panel 3: Theta Dominance --- Use original_times ---
        ax = axs[3]
        if len(original_times) == len(theta_dominance):
             ax.plot(original_times, theta_dominance, color='green', linewidth=0.8, label='Theta Dominance')
        else: print(f"Plot Error: Theta length ({len(theta_dominance)}) != original_times length ({len(original_times)})")
        if 'rem' in thresholds and np.isfinite(thresholds['rem']):
            ax.axhline(thresholds['rem'], color='darkred', linestyle='--', linewidth=1, label=f'REM Thr ({thresholds["rem"]:.2f})')
        ax.set_ylabel('Theta Dominance')
        ax.legend(loc='upper right', fontsize='small')

        # --- Panel 4: EMG (Conditional) --- Use original_times ---
        last_ax_idx = 3
        if calculate_emg:
            last_ax_idx = 4
            ax = axs[4]
            if len(original_times) == len(emg):
                 ax.plot(original_times, emg, color='purple', linewidth=0.8, label='EMG Estimate')
            else: print(f"Plot Error: EMG length ({len(emg)}) != original_times length ({len(original_times)})")
            if 'emg' in thresholds and np.isfinite(thresholds['emg']):
                ax.axhline(thresholds['emg'], color='magenta', linestyle='--', linewidth=1, label=f'EMG Thr ({thresholds["emg"]:.2f})')
            ax.set_ylabel('EMG Estimate')
            ax.legend(loc='upper right', fontsize='small')

        axs[last_ax_idx].set_xlabel('Time (s)')

        # --- Add Sleep State Background Shading --- Use padded_sleep_states and padded_times ---
        plotted_labels = set()
        if len(padded_sleep_states) != len(padded_times):
            print("Error: Mismatch between padded_sleep_states and padded_times lengths. Skipping state shading.")
        else:
            states_diff = np.diff(padded_sleep_states, prepend=np.nan)
            change_indices = np.where(states_diff != 0)[0]
            segment_indices = np.unique(np.concatenate(([0], change_indices, [len(padded_sleep_states)])))

            for i_ax, ax in enumerate(axs):
                ymin, ymax = ax.get_ylim()

                for k in range(len(segment_indices) - 1):
                    start_idx = segment_indices[k]
                    end_idx = segment_indices[k+1]

                    if start_idx >= len(padded_sleep_states): continue
                    state = padded_sleep_states[start_idx]

                    t_start = padded_times[start_idx]
                    t_end = padded_times[end_idx] if end_idx < len(padded_times) else total_duration_sec

                    label = state_labels[state] if i_ax == 0 and state_labels[state] not in plotted_labels else None
                    try:
                        ax.fill_between([t_start, t_end], ymin, ymax,
                                          color=state_colors_bg[state], alpha=shading_alpha, label=label,
                                          edgecolor=None)
                        if label: plotted_labels.add(state_labels[state])
                    except Exception as e_fill: print(f"Error during fill_between: {e_fill} for state {state} from {t_start} to {t_end}")


                # Add Epoch Boundaries and Labels to ALL panels
                if epoch_boundaries:
                    for i_ep, (start_sec, end_sec) in enumerate(epoch_boundaries):
                        if i_ep > 0 and start_sec > 0:
                            ax.axvline(start_sec, color='black', linestyle='--', linewidth=1.2, ymin=0.0, ymax=1.0)

                        if i_ax == 0:
                            label_x_pos = start_sec + 5
                            plot_start, plot_end = ax.get_xlim()
                            if label_x_pos < plot_end:
                                label_y_pos = ymax - (ymax - ymin) * 0.05
                                ax.text(label_x_pos, label_y_pos, f'Epoch {i_ep}',
                                         horizontalalignment='left', verticalalignment='top',
                                         fontsize=9, color='black', weight='bold',
                                         bbox=dict(facecolor='white', alpha=0.6, pad=0.2, boxstyle='round,pad=0.3'))


                # Restore y-limits and set consistent x-limits based on total duration
                ax.set_ylim(ymin, ymax)
                plot_start_time = original_times[0] if len(original_times) > 0 else 0
                ax.set_xlim(max(0, plot_start_time - step_size), total_duration_sec)


        # Add combined legend for states to the first plot (axs[0])
        if plotted_labels:
             handles, labels = axs[0].get_legend_handles_labels()
             state_handles_labels = [(h,l) for h,l in zip(handles, labels) if l in state_labels.values()]
             if state_handles_labels:
                 axs[0].legend(handles=[h for h,l in state_handles_labels],
                            labels=[l for h,l in state_handles_labels],
                            title="Sleep State", loc='upper right', fontsize='small')
             else:
                 print("Warning: No state labels found to create legend for axs[0].")


        # --- Save and Close Plot ---
        plot_filename_svg = output_path / f'{base_filename}_ScoringOverview_Epochs.svg'
        plot_filename_png = output_path / f'{base_filename}_ScoringOverview_Epochs.png'
        try:
            plt.savefig(plot_filename_svg, dpi=300, bbox_inches='tight')
            print(f"Saved scoring overview plot: {plot_filename_svg}")
            plt.savefig(plot_filename_png, dpi=150, bbox_inches='tight')
            print(f"Saved scoring overview plot: {plot_filename_png}")
        except Exception as save_err:
            print(f"Error saving plot: {save_err}")
            traceback.print_exc()
        plt.close(fig)

    except Exception as e:
        print(f"Error generating scoring overview plot V10 for {base_filename}: {e}")
        traceback.print_exc()
        plt.close('all')

# === Main Script Execution ================================================
# --- Parameters ---
print("--- Script Parameters ---")
TARGET_FS_USER_DEFINED = 1250.0 # Desired target sampling rate
cutoff_freq = 600.0         # Low-pass filter cutoff (Hz)
num_channels_to_use = 48     # Number of channels from END of recording
window_size = 10.0          # Spectrogram window size (seconds)
step_size = 1.0             # Spectrogram step size (seconds)
calculate_emg = True        # SET True TO ENABLE EMG CALCULATION
n_ica_components = 10       # Number of ICA components (if using EMG)
ica_epochs_duration = 240   # Duration (seconds) for ICA fitting (if using EMG)
num_channels_for_ica = 48    # Number of channels for ICA (from selected LFP, if using EMG)
scoring_buffer_size = 5     # Steps to confirm state change
fixed_emg_threshold_val = None # Use None for percentile, or set fixed value
# Speed related params (optional)
# speed_csv_file_path = None
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

print("\nPlease select the directory containing timestamp (*_timestamps.npy) files.")
timestamps_dir_str = filedialog.askdirectory(title="Select directory containing Timestamp NPY files")
timestamps_dir = Path(timestamps_dir_str) if timestamps_dir_str else None

print("\nPlease select the directory containing speed CSV files (optional).")
speed_files_dir_str = filedialog.askdirectory(title="Select directory containing speed files (optional)")
speed_files_dir = Path(speed_files_dir_str) if speed_files_dir_str else None

print("\nPlease select the output directory for saving results.")
output_dir_str = filedialog.askdirectory(title="Select output directory for sleep state files")
output_dir = Path(output_dir_str) if output_dir_str else None
root.destroy()

# --- Validation of Inputs ---
if lfp_bin_meta_dir is None or not lfp_bin_meta_dir.is_dir(): print("Error: LFP directory invalid. Exiting."); sys.exit(1)
if timestamps_dir is None or not timestamps_dir.is_dir(): print("Error: Timestamps directory invalid. Exiting."); sys.exit(1)
if output_dir is None: print("Error: Output directory not selected. Exiting."); sys.exit(1)
try: output_dir.mkdir(parents=True, exist_ok=True); print(f"Using output directory: {output_dir}")
except Exception as e: print(f"Error: Output directory invalid or could not be created: {e}. Exiting."); sys.exit(1)
if speed_files_dir and not speed_files_dir.is_dir(): print("Warning: Speed files directory selected but invalid. Disabling speed analysis.") ; speed_files_dir = None

# --- Find LFP Files (Specifically .lf.bin) ---
lfp_files = sorted([p for p in lfp_bin_meta_dir.glob('*.lf.bin') if p.is_file()])
if not lfp_files: print(f"Error: No *.lf.bin files found in: {lfp_bin_meta_dir}. Exiting."); sys.exit(1)
print(f"\nFound {len(lfp_files)} LFP '.lf.bin' files to process:")
for f in lfp_files: print(f" - {f.name}")

# --- Main Processing Loop ---
for lfp_file_path in lfp_files:
    print(f"\n{'='*40}\nProcessing recording: {lfp_file_path.name}\n{'='*40}")

    # --- Improved Base Filename Extraction ---
    lfp_filename = lfp_file_path.name
    match = re.match(r"^(.*?)(\.(imec|nidq)\d?)?\.lf\.bin$", lfp_filename)
    if match:
        output_filename_base = match.group(1)
        print(f"Derived base filename: {output_filename_base}")
    else:
        output_filename_base = lfp_filename.replace('.lf.bin', '')
        print(f"Warning: Using fallback base filename: {output_filename_base}")

    # Initialize variables for this loop iteration
    data_memmap = None # Initialize for Pyflakes
    original_fs, target_fs, data_selected_channels, data_selected_float = None, None, None, None
    downsampled_data, filtered_data_ref, averaged_lfp_data = None, None, None # Keep ref to filtered_data if needed
    spectrogram, frequencies, original_times, padded_times = None, None, None, None
    pc1, theta_dominance, emg = None, None, None
    ica_object, ica_components = None, None
    original_sleep_states, padded_sleep_states = None, None
    nrem_threshold, rem_threshold, emg_threshold = np.nan, np.nan, np.nan
    calculate_emg_this_file = False
    timestamp_data, epoch_frame_data, total_duration_sec, epoch_boundaries = None, None, None, None

    try:
        # --- Find corresponding meta file ---
        meta_file_path = lfp_file_path.with_suffix('.meta')
        if not meta_file_path.exists():
            print(f"Warning: Meta file '{meta_file_path.name}' not found. Skipping LFP file.")
            continue

        # --- Find and Load Timestamp File ---
        print("\n--- Loading Timestamp Data ---")
        expected_ts_name = f"{output_filename_base}.nidq_timestamps.npy"
        timestamp_file_path = timestamps_dir / expected_ts_name

        if not timestamp_file_path.exists():
            print(f"Warning: Could not find timestamp file: {timestamp_file_path.name}")
            base_name_alt = output_filename_base.replace('_tcat','')
            expected_ts_name_alt = f"{base_name_alt}.nidq_timestamps.npy"
            timestamp_file_path_alt = timestamps_dir / expected_ts_name_alt
            if timestamp_file_path_alt.exists():
                 timestamp_file_path = timestamp_file_path_alt
                 print(f"Found alternative timestamp file: {timestamp_file_path.name}")
            else:
                 print(f"  Also tried: {timestamp_file_path_alt.name}")
                 print("  Skipping epoch processing for this file.")
                 epoch_boundaries = []
                 total_duration_sec = None

        if timestamp_file_path and timestamp_file_path.exists():
             print(f"Found corresponding timestamp file: {timestamp_file_path.name}")
             try:
                timestamp_data = np.load(timestamp_file_path, allow_pickle=True).item()
                if 'EpochFrameData' not in timestamp_data or 'TotalDurationSec' not in timestamp_data:
                     print(f"Error: Timestamp file {timestamp_file_path.name} is missing required keys ('EpochFrameData', 'TotalDurationSec'). Skipping epoch processing.")
                     epoch_boundaries = []
                     total_duration_sec = None
                else:
                     epoch_frame_data = timestamp_data['EpochFrameData']
                     total_duration_sec = timestamp_data['TotalDurationSec']
                     if not isinstance(epoch_frame_data, (list, np.ndarray)) or \
                        (len(epoch_frame_data) > 0 and not all(isinstance(ep, dict) for ep in epoch_frame_data)):
                         print(f"Error: 'EpochFrameData' in {timestamp_file_path.name} is not a list/array of dictionaries. Skipping epoch processing.")
                         epoch_boundaries = []
                     else:
                         epoch_boundaries = []
                         valid_epochs = True
                         for i, ep in enumerate(epoch_frame_data):
                              if 'start_time_sec' in ep and 'end_time_sec' in ep:
                                   if ep['end_time_sec'] >= ep['start_time_sec']:
                                       epoch_boundaries.append((ep['start_time_sec'], ep['end_time_sec']))
                                   else:
                                       print(f"Warning: Epoch {i} in {timestamp_file_path.name} has end_time < start_time. Skipping this epoch.")
                                       valid_epochs = False
                              else:
                                   print(f"Warning: Epoch {i} in {timestamp_file_path.name} missing 'start_time_sec' or 'end_time_sec'.")
                                   valid_epochs = False
                         if not valid_epochs:
                              print("Skipping epoch boundary plotting due to missing/invalid time keys in some epochs.")
                              epoch_boundaries = []

                         print(f"Loaded {len(epoch_boundaries)} valid epochs. Total duration: {total_duration_sec:.2f} s")

             except Exception as e_ts:
                print(f"Error loading or parsing timestamp file {timestamp_file_path.name}: {e_ts}")
                traceback.print_exc()
                epoch_boundaries = []
                total_duration_sec = None


        # Fallback for total duration if not found/loaded from timestamp file
        if total_duration_sec is None:
            print("Attempting to get total duration from meta file...")
            try:
                meta = readMeta(meta_file_path)
                if 'fileTimeSecs' in meta:
                     total_duration_sec = float(meta['fileTimeSecs'])
                     print(f"Using total duration from meta file: {total_duration_sec:.2f} s")
                else:
                     print("Error: Could not determine total duration from timestamp or meta file. Skipping file.")
                     continue
            except Exception as e_meta_dur:
                 print(f"Error reading duration from meta file: {e_meta_dur}. Skipping file.")
                 continue


        # --- Find corresponding speed file ---
        current_speed_file_path = None
        if speed_files_dir and speed_files_dir.is_dir():
            base_name_for_speed = output_filename_base
            possible_speed_files = list(speed_files_dir.glob(f'{base_name_for_speed}*.csv'))
            if possible_speed_files:
                current_speed_file_path = possible_speed_files[0]
                print(f"Found corresponding speed file: {current_speed_file_path.name}")
            else:
                print(f"Note: No speed file found matching pattern '{base_name_for_speed}*.csv' in {speed_files_dir}")


        # 1. Load Data & Get Sampling Rate
        print("\n--- 1. Loading Data ---")
        data_memmap, original_fs = read_bin_data(lfp_file_path, meta_file_path)
        if data_memmap is None or original_fs is None: print(f"Failed load/get SR. Skipping."); continue

        # Set target FS
        target_fs = TARGET_FS_USER_DEFINED
        min_original_fs_for_target = TARGET_FS_USER_DEFINED * 2.0
        if original_fs < min_original_fs_for_target:
             print(f"Warning: Original FS ({original_fs:.2f} Hz) is less than 2x target FS ({TARGET_FS_USER_DEFINED:.2f} Hz).")
             target_fs = original_fs / 2.0 * 0.95
             target_fs = max(target_fs, 1.0)
             print(f"Adjusted target FS to {target_fs:.2f} Hz.")
        else:
             print(f"Using target FS: {target_fs:.6f} Hz")


        # 2. Select Channels
        print("\n--- 2. Selecting Channels ---")
        if data_memmap is None or data_memmap.shape[1] == 0:
             print("Error: Memory-mapped data is invalid or has 0 channels. Skipping.")
             continue
        if data_memmap.shape[1] < num_channels_to_use:
            print(f"Warning: File has {data_memmap.shape[1]} channels < requested {num_channels_to_use}. Using all available.")
            num_chans_actual = data_memmap.shape[1]
        else: num_chans_actual = num_channels_to_use
        try:
            data_selected_channels = data_memmap[:, -num_chans_actual:]
            print(f"Selected last {num_chans_actual} channels. Shape: {data_selected_channels.shape}, Type: {data_selected_channels.dtype}")
        except Exception as e_select:
            print(f"Error selecting channels: {e_select}. Skipping.")
            if hasattr(data_memmap, '_mmap'): data_memmap._mmap.close()
            del data_memmap
            continue

        # 3.Downsampling-
        print("\n--- 3. Downsampling ---")
        downsampling_factor = int(round(original_fs / target_fs))
        if downsampling_factor < 1: downsampling_factor = 1
        effective_target_fs = original_fs / downsampling_factor

        if downsampling_factor == 1:
            print("Downsampling factor is 1. Copying data as float32.")
            try:
                downsampled_data = data_selected_channels[:].astype(np.float32)
            except MemoryError:
                print("MemoryError copying data. Skipping.");
                downsampled_data = None
            except Exception as e_copy:
                 print(f"Error copying data: {e_copy}. Skipping.")
                 downsampled_data = None
        else:
            print(f"Downsampling factor: {downsampling_factor} (from {original_fs:.6f} Hz to target ~{target_fs:.6f} Hz)")
            print(f"Effective sampling rate will be approx: {effective_target_fs:.6f} Hz")
            print("Converting selected channels to float32 for downsampling...")
            data_selected_float = None
            try:
                data_selected_float = data_selected_channels[:].astype(np.float32)
            except MemoryError:
                print(f"MemoryError loading selected channels as float32. Skipping.");
                data_selected_float = None
            except Exception as e_load_float:
                print(f"Error loading selected channels as float32: {e_load_float}. Skipping.")
                data_selected_float = None

            if data_selected_float is not None:
                 try:
                    print(f"Applying signal.decimate (factor={downsampling_factor})...")
                    if not data_selected_float.flags['C_CONTIGUOUS']:
                        print("Warning: Data not C-contiguous for decimate. Making a copy.")
                        data_selected_float = np.ascontiguousarray(data_selected_float)

                    downsampled_data = signal.decimate(data_selected_float, downsampling_factor, axis=0, ftype='fir', zero_phase=True)
                    print(f"Shape AFTER downsampling: {downsampled_data.shape}, Type: {downsampled_data.dtype}")
                 except MemoryError: print(f"MemoryError during decimate. Skipping."); downsampled_data = None
                 except Exception as e: print(f"Error during decimation: {e}. Skipping."); downsampled_data = None
                 finally: del data_selected_float
            else:
                 downsampled_data = None

        # Explicitly close and delete memmap object after copying or decimating
        if 'data_memmap' in locals() and data_memmap is not None:
            if hasattr(data_memmap, '_mmap'):
                try: data_memmap._mmap.close()
                except Exception: pass
            del data_memmap

        if downsampled_data is None:
             print("Downsampling step failed or produced no data. Skipping file.")
             gc.collect()
             continue

        target_fs = effective_target_fs
        # --- End of Simplified Downsampling Block ---

        # 4. Filter
        print("\n--- 4. Filtering ---")
        nyquist_effective = target_fs / 2.0
        current_cutoff = cutoff_freq
        if current_cutoff >= nyquist_effective:
            print(f"Warning: Original cutoff ({current_cutoff} Hz) >= effective Nyquist ({nyquist_effective:.2f} Hz).")
            current_cutoff = nyquist_effective * 0.95
            print(f"Adjusted filter cutoff to {current_cutoff:.2f} Hz.")

        filtered_data_ref = filter_data(downsampled_data, target_fs, current_cutoff) # Keep reference
        del downsampled_data
        if filtered_data_ref is None: print(f"Filtering failed. Skipping."); continue

        # 5. Average Channels
        print("\n--- 5. Averaging Channels ---")
        if filtered_data_ref is None or filtered_data_ref.size == 0:
             print("Error: Filtered data is invalid. Skipping.")
             continue
        elif filtered_data_ref.ndim == 2 and filtered_data_ref.shape[1] > 0:
             if filtered_data_ref.shape[1] > 1:
                 averaged_lfp_data = np.mean(filtered_data_ref, axis=1)
                 print(f"Averaged {filtered_data_ref.shape[1]} channels.")
             else:
                 averaged_lfp_data = filtered_data_ref.squeeze()
                 print("Only one channel available, using as is.")
             print(f"Shape of averaged LFP data: {averaged_lfp_data.shape}")
        elif filtered_data_ref.ndim == 1:
             averaged_lfp_data = filtered_data_ref
             print("Data seems to be 1D already, using as is.")
        else:
             print(f"Error: Filtered data has unexpected shape {filtered_data_ref.shape}. Cannot average. Skipping.")
             del filtered_data_ref; continue

        if averaged_lfp_data.ndim != 1:
            print(f"Error: Averaged LFP data is not 1D (shape: {averaged_lfp_data.shape}). Skipping.")
            if 'filtered_data_ref' in locals(): del filtered_data_ref
            del averaged_lfp_data; continue


        # 6. Compute Spectrogram
        print("\n--- 6. Computing Spectrogram ---")
        spectrogram, frequencies, original_times = compute_spectrogram(averaged_lfp_data, target_fs, window_size, step_size)
        del averaged_lfp_data
        if spectrogram is None or original_times is None: print(f"Spectrogram failed. Skipping."); continue
        print(f"Original times vector length: {len(original_times)}")

        # 7. Compute PCA
        print("\n--- 7. Computing PCA ---")
        pc1 = compute_pca(spectrogram)
        if pc1 is None: print(f"PCA failed. Skipping."); continue
        if len(pc1) != len(original_times):
             print(f"Error: PCA length ({len(pc1)}) mismatch with times ({len(original_times)}). Skipping.")
             continue

        # 8. Compute Theta Dominance
        print("\n--- 8. Computing Theta Dominance ---")
        theta_dominance = compute_theta_dominance(spectrogram, frequencies)
        if theta_dominance is None: print(f"Theta dominance failed. Skipping."); continue
        if len(theta_dominance) != len(original_times):
             print(f"Error: Theta Dom length ({len(theta_dominance)}) mismatch with times ({len(original_times)}). Skipping.")
             continue

        # 9. Compute EMG (Conditional)
        print("\n--- 9. Computing EMG (Conditional) ---")
        emg = np.zeros(len(original_times))
        calculate_emg_this_file = False

        if calculate_emg:
            if 'filtered_data_ref' in locals() and filtered_data_ref is not None:
                if filtered_data_ref.ndim == 2 and filtered_data_ref.shape[1] > 0:
                    num_ch_avail_filt = filtered_data_ref.shape[1]
                    actual_num_ch_ica = min(num_channels_for_ica, num_ch_avail_filt)
                    actual_n_comp = min(n_ica_components, actual_num_ch_ica)

                    if actual_num_ch_ica <= 0 or actual_n_comp <= 0:
                        print("Warning: Not enough channels available in filtered data for ICA. Skipping EMG.")
                    else:
                        print(f"Calculating EMG: Using {actual_num_ch_ica} chans from filtered data, {actual_n_comp} comps.")
                        emg_seg, ica_object, ica_components = compute_emg(
                            filtered_data_ref, target_fs, window_size, step_size,
                            n_ica_components=actual_n_comp, num_channels_for_ica=actual_num_ch_ica,
                            ica_epochs_duration=ica_epochs_duration)

                        if emg_seg is None:
                            print("EMG computation failed. Scoring without EMG.")
                        elif len(emg_seg) == len(original_times):
                            emg = emg_seg
                            calculate_emg_this_file = True
                        else:
                            print(f"Warning: Calculated EMG length ({len(emg_seg)}) != Original times length ({len(original_times)}). Adjusting EMG length.")
                            target_len = len(original_times)
                            current_len = len(emg_seg)
                            if current_len > target_len:
                                emg = emg_seg[:target_len]
                            else:
                                pad_value = emg_seg[-1] if current_len > 0 else 0
                                emg_padded_temp = np.pad(emg_seg, (0, target_len - current_len), mode='constant', constant_values=pad_value) # Renamed temp var
                                if len(emg_padded_temp) == target_len:
                                     emg = emg_padded_temp
                                else:
                                     emg = np.resize(emg_seg, target_len)
                                     print("Warning: Padding/resizing EMG resulted in unexpected length. Used np.resize.")
                            calculate_emg_this_file = True
                elif filtered_data_ref.ndim == 1:
                     print("Warning: Filtered data is 1D. Cannot run ICA. Calculating 'EMG' from mean absolute value.")
                     emg_seg_1d, _, _ = compute_emg(filtered_data_ref, target_fs, window_size, step_size)
                     if emg_seg_1d is not None and len(emg_seg_1d) == len(original_times):
                          emg = emg_seg_1d
                          calculate_emg_this_file = False
                          print("Note: 'EMG' score derived from single LFP channel, not ICA.")
                     else:
                          print("Failed to calculate segmented EMG from 1D data or length mismatch.")
                else:
                    print("Warning: Filtered data not available or not suitable for EMG calculation.")
            else:
                 print("Warning: filtered_data was not available for EMG calculation (check code flow).") # Added check
        else:
            print("EMG calculation disabled by user setting.")

        # --- Delete filtered_data AFTER EMG ---
        if 'filtered_data_ref' in locals() and filtered_data_ref is not None:
             del filtered_data_ref
             gc.collect()

        # 10. Score Sleep States
        print("\n--- 10. Scoring Sleep States ---")
        if len(original_times) == 0: print("Error: Zero length data before scoring. Skipping."); continue

        nrem_threshold = np.nanpercentile(pc1[np.isfinite(pc1)], 75) if np.any(np.isfinite(pc1)) else np.nan
        rem_threshold = np.nanpercentile(theta_dominance[np.isfinite(theta_dominance)], 75) if np.any(np.isfinite(theta_dominance)) else np.nan
        if calculate_emg_this_file:
            if fixed_emg_threshold_val is not None: emg_threshold = fixed_emg_threshold_val
            else:
                 emg_finite = emg[np.isfinite(emg)]
                 if len(emg_finite) > 0 : emg_threshold = np.nanpercentile(emg_finite, 25)
                 else: emg_threshold = 0; print("Warning: Cannot calc percentile EMG thresh (all NaN?). Using 0.")
        else: emg_threshold = np.nan

        original_sleep_states = score_sleep_states(
            pc1, theta_dominance, emg, original_times, step_size,
            buffer_size=scoring_buffer_size,
            fixed_emg_threshold=emg_threshold if fixed_emg_threshold_val is None else fixed_emg_threshold_val,
            speed_csv_path=str(current_speed_file_path) if current_speed_file_path else None,
            video_fps=video_frame_rate, video_start_offset_seconds=video_start_delay_seconds,
            speed_threshold=mouse_speed_threshold,
            use_emg=calculate_emg_this_file
        )
        if original_sleep_states is None: print(f"Sleep scoring failed. Skipping save/plot."); continue
        print(f"Scored states length: {len(original_sleep_states)}")

        # --- Padding Logic ---
        print("\n--- 10b. Adjusting Length to Total Duration ---")
        padded_sleep_states = original_sleep_states
        padded_times = original_times

        if total_duration_sec is not None and len(original_times) > 0:
            calculated_end_time = original_times[-1] + step_size
            print(f"Calculated end time from states: {calculated_end_time:.4f} s")
            print(f"Total duration from source file: {total_duration_sec:.4f} s")

            missing_seconds = total_duration_sec - calculated_end_time
            if missing_seconds > (0.1 * step_size):
                num_steps_to_pad = int(math.ceil(missing_seconds / step_size))
                print(f"Padding required: {missing_seconds:.2f} s ({num_steps_to_pad} steps)")
                if num_steps_to_pad > 0 and len(original_sleep_states) > 0:
                    last_state = original_sleep_states[-1]
                    padding_states_arr = np.full(num_steps_to_pad, last_state, dtype=original_sleep_states.dtype)
                    padded_sleep_states = np.concatenate((original_sleep_states, padding_states_arr))

                    padding_times_arr = original_times[-1] + np.arange(1, num_steps_to_pad + 1) * step_size
                    padded_times = np.concatenate((original_times, padding_times_arr))

                    print(f"Padded states/times length: {len(padded_sleep_states)}")
                else:
                     print("No padding applied (missing duration too small or no states to pad).")
            elif calculated_end_time > total_duration_sec + step_size :
                print(f"Warning: Calculated end time ({calculated_end_time:.2f}) > total duration ({total_duration_sec:.2f}).")
                valid_indices = np.where(original_times < total_duration_sec)[0]
                if len(valid_indices) > 0:
                    last_valid_index = valid_indices[-1]
                    if last_valid_index + 1 < len(original_times):
                        print(f"Truncating original arrays to index {last_valid_index} (time {original_times[last_valid_index]:.2f}s)")
                        original_times = original_times[:last_valid_index+1]
                        # Important: Truncate spectrogram and metrics as well!
                        spectrogram = spectrogram[:, :last_valid_index+1]
                        pc1 = pc1[:last_valid_index+1]
                        theta_dominance = theta_dominance[:last_valid_index+1]
                        emg = emg[:last_valid_index+1]
                        original_sleep_states = original_sleep_states[:last_valid_index+1]
                        padded_sleep_states = original_sleep_states
                        padded_times = original_times
                    else:
                         print("No truncation applied.")

                else:
                     print("Warning: All time points seem beyond total duration. Check inputs.")
            else:
                 print("No padding needed.")


        # 11. Save Results
        print("\n--- 11. Saving Results ---")
        output_suffix = "_EMG" if calculate_emg_this_file else "_NoEMG"
        state_filename = f'{output_filename_base}_sleep_states{output_suffix}.npy'
        times_filename = f'{output_filename_base}_sleep_state_times{output_suffix}.npy'
        boundaries_filename = f'{output_filename_base}_epoch_boundaries{output_suffix}.npy'

        try:
            if padded_sleep_states is not None:
                np.save(output_dir / state_filename, padded_sleep_states)
                print(f"Saved states: {output_dir / state_filename}")
            else: print("Warning: padded_sleep_states is None, cannot save.")

            if padded_times is not None:
                np.save(output_dir / times_filename, padded_times)
                print(f"Saved times: {output_dir / times_filename}")
            else: print("Warning: padded_times is None, cannot save.")

            if epoch_boundaries:
                 np.save(output_dir / boundaries_filename, np.array(epoch_boundaries, dtype=object))
                 print(f"Saved epoch boundaries: {output_dir / boundaries_filename}")

        except Exception as e: print(f"Error saving results: {e}")

        # 12. Visualization
        print("\n--- 12. Visualization ---")
        plot_thresholds = {'nrem': nrem_threshold, 'rem': rem_threshold, 'emg': emg_threshold}

        plot_possible = (original_times is not None and frequencies is not None and spectrogram is not None and
                         pc1 is not None and theta_dominance is not None and emg is not None and
                         padded_sleep_states is not None and padded_times is not None and total_duration_sec is not None)

        if plot_possible:
            if spectrogram.shape[1] != len(original_times): # Check against original_times
                 print(f"Error: Final spectrogram columns ({spectrogram.shape[1]}) != original_times length ({len(original_times)}). Skipping plot.")
            else:
                 plot_scoring_overview( # CALL THE UPDATED PLOT FUNCTION (V10)
                    original_times=original_times, frequencies=frequencies, spectrogram=spectrogram,
                    pc1=pc1, theta_dominance=theta_dominance, emg=emg,
                    padded_sleep_states=padded_sleep_states, padded_times=padded_times,
                    thresholds=plot_thresholds,
                    calculate_emg=calculate_emg_this_file,
                    step_size=step_size,
                    epoch_boundaries=epoch_boundaries,
                    total_duration_sec=total_duration_sec,
                    output_path=output_dir, base_filename=output_filename_base
                 )
        else:
            print("Skipping overview plot due to missing data required for plotting.")
            missing_vars = [name for name, var in locals().items() if name in ['original_times', 'frequencies', 'spectrogram', 'pc1', 'theta_dominance', 'emg', 'padded_sleep_states', 'padded_times', 'total_duration_sec'] and var is None]
            if missing_vars: print(f"  Missing variables: {', '.join(missing_vars)}")


    except Exception as e_main:
        print(f"\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"FATAL ERROR PROCESSING FILE: {lfp_file_path.name}")
        print(f"Error details: {e_main}")
        traceback.print_exc() # Print full traceback to log file
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")

    finally:
        # --- Clean up memory before next loop ---
        print("\n--- Cleaning up memory for next file ---")
        # Close memmap if it somehow wasn't closed earlier
        if 'data_memmap' in locals() and data_memmap is not None and hasattr(data_memmap, '_mmap'):
             print("Attempting final close of data_memmap...")
             try: data_memmap._mmap.close()
             except Exception as e_close: print(f"Error closing memmap: {e_close}")

        vars_to_del = [
            'data_memmap', 'data_selected_channels', 'data_selected_float',
            'downsampled_data', 'filtered_data_ref', 'averaged_lfp_data', 
            'spectrogram', 'frequencies', 'original_times', 'padded_times',
            'pc1', 'theta_dominance', 'emg', 'emg_seg', 'emg_seg_1d', 'emg_padded_temp', 
            'original_sleep_states', 'padded_sleep_states', 'padding_states_arr', 'padding_times_arr',
            'ica_object', 'ica_components', 'plot_thresholds',
            'timestamp_data', 'epoch_frame_data', 'total_duration_sec', 'epoch_boundaries',
            'meta', 'states_diff', 'change_indices', 'segment_indices', 'valid_indices',
            'time_edges', 'freq_edges' 
        ]
        deleted_count = 0
        local_vars = locals()
        for var_name in vars_to_del:
            if var_name in local_vars and local_vars[var_name] is not None:
                try:
                    local_vars[var_name] = None
                    deleted_count += 1
                except Exception as del_e:
                     print(f"Note: Could not clear variable {var_name}: {del_e}")

        collected = gc.collect()
        print(f"Attempted clear on {deleted_count} vars. Garbage collection: {collected} objects cleared.")

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

if sys.stderr != original_stderr:
     if isinstance(sys.stderr, io.IOBase) and sys.stderr != sys.stdout:
         try: sys.stderr.close()
         except Exception as e: print(f"(Info) Error closing redirected stderr: {e}", file=original_stderr)
     sys.stderr = original_stderr

sys.stdout = original_stdout
sys.stderr = original_stderr
print(f"\nScript execution completed. Check logs in '{output_file_path}' and results in '{output_dir}'.")