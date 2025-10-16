# -*- coding: utf-8 -*-
"""
Created on Thu May  1 16:11:57 2025

@author: HT_bo
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy import stats
import pywt # Optional: for wavelet spectrograms
from pathlib import Path
import warnings
import re # For filename matching
import os
import gc
import traceback
# --- Import Tkinter ---
from tkinter import Tk
from tkinter import filedialog
# --- Import for Parallel Processing ---
from joblib import Parallel, delayed
import multiprocessing # To get CPU count
from DemoReadSGLXData.readSGLX import readMeta

# --- Constants ---
# Ripple detection parameters
RIPPLE_FILTER_LOWCUT = 80.0  # Hz
RIPPLE_FILTER_HIGHCUT = 250.0 # Hz
RIPPLE_POWER_LP_CUTOFF = 55.0 # Hz (for baseline calculation)
RIPPLE_DETECTION_SD_THRESHOLD = 4.0 # SD above mean
RIPPLE_EXPANSION_SD_THRESHOLD = 2.0 # SD above mean
RIPPLE_MIN_DURATION_MS = 15.0 # ms
RIPPLE_MERGE_GAP_MS = 15.0 # ms

# Sharp-wave (SPW) detection parameters (for CA1)
SPW_FILTER_LOWCUT = 5.0   # Hz
SPW_FILTER_HIGHCUT = 40.0  # Hz
SPW_DETECTION_SD_THRESHOLD = 2.5 # SD above mean (applied to absolute filtered LFP)
SPW_MIN_DURATION_MS = 20.0  # ms
SPW_MAX_DURATION_MS = 400.0 # ms

# Co-occurrence window
COOCCURRENCE_WINDOW_MS = 60.0 # ms (+/- around reference ripple peak)

# Spectrogram parameters
SPECTROGRAM_WINDOW_MS = 200 # ms (+/- around event timestamp)
SPECTROGRAM_FREQS = np.arange(10, 300, 2) # Example frequency range for spectrogram

# --- Downsampling Configuration ---
DOWNSAMPLING_FACTOR = 2 # Set to 1 to disable downsampling
print(f"Configuring analysis with downsampling factor: {DOWNSAMPLING_FACTOR}")

# --- Parallel Processing Configuration ---
# Use fewer cores initially to avoid memory issues, leave plenty for system.
# Adjust this based on observed performance and memory usage.
NUM_CORES = max(1, multiprocessing.cpu_count() // 2) # Use half the cores
# NUM_CORES = 4 # Or set a fixed lower number like 4
print(f"Configuring parallel processing to use {NUM_CORES} cores.")


# --- Data Loading Functions (Adapted from Example) ---

def load_lfp_data_memmap(file_path, meta_path, data_type='int16'):
    """
    Loads LFP data using memory-mapping, based on example script.
    Args:
        file_path (Path): Path to the binary LFP file (*.lf.bin).
        meta_path (Path): Path to the corresponding metadata file (*.meta).
        data_type (str): Data type of the samples.
    Returns:
        tuple: (numpy.memmap: Memory-mapped data, float: sampling rate, int: n_channels, int: n_samples) or (None, None, 0, 0) on error.
    """
    sampling_rate = None
    data = None
    n_channels = 0
    n_samples = 0
    print(f"Attempting to load LFP data from: {file_path}")
    print(f"Using metadata from: {meta_path}")
    try:
        # Read metadata to get the number of channels and sampling rate
        meta = readMeta(meta_path) # Use imported or placeholder readMeta
        n_channels = int(meta['nSavedChans'])

        # Get precise sampling rate
        if 'imSampRate' in meta:
            sampling_rate = float(meta['imSampRate'])
        elif 'niSampRate' in meta:
            sampling_rate = float(meta['niSampRate'])
        else:
            print(f"Error: Sampling rate key ('imSampRate' or 'niSampRate') not found in {meta_path}.")
            return None, None, 0, 0

        if sampling_rate <= 0:
            print(f"Error: Invalid sampling rate ({sampling_rate}) found in {meta_path}.")
            return None, None, n_channels, 0

        print(f"Metadata: {n_channels} saved channels. Sampling rate: {sampling_rate:.6f} Hz")

        # Calculate the shape for memmap
        file_size = file_path.stat().st_size
        item_size = np.dtype(data_type).itemsize
        if n_channels <= 0 or item_size <= 0:
             print(f"Error: Invalid number of channels ({n_channels}) or item size ({item_size}). Cannot calculate shape.")
             return None, sampling_rate, n_channels, 0 # Return fs/n_chan even if data fails

        # Ensure file size is a multiple of (n_channels * item_size)
        expected_total_bytes = file_size - (file_size % (n_channels * item_size))
        if file_size != expected_total_bytes:
            print(f"Warning: File size {file_size} is not an exact multiple of {n_channels} channels * {item_size} bytes/sample.")
            print(f" Potential extra bytes at end of file: {file_size % (n_channels * item_size)}. Processing based on truncated size.")

        n_samples = expected_total_bytes // (n_channels * item_size)

        if n_samples <= 0:
            print("Error: Calculated number of samples is zero or negative. Check file size and metadata.")
            return None, sampling_rate, n_channels, 0

        # Shape for memmap (samples, channels)
        shape = (n_samples, n_channels)
        print(f"Calculated samples: {n_samples}")
        print(f"Expected data shape for memmap: {shape}")

        # Memory-map the binary file (read-only)
        data = np.memmap(file_path, dtype=data_type, mode='r', shape=shape, offset=0)
        print(f"Successfully memory-mapped file: {file_path}")
        # Return data as (samples, channels), fs, n_channels, n_samples
        return data, sampling_rate, n_channels, n_samples

    except FileNotFoundError:
        print(f"Error: File not found - {file_path} or {meta_path}")
        return None, None, 0, 0
    except KeyError as e:
        print(f"Error: Metadata key missing in {meta_path} - {e}")
        return None, sampling_rate, n_channels, 0 # Return fs/n_chan if meta loaded but key missing
    except ValueError as e:
         print(f"Error: Problem with shape or dtype during memmap for {file_path} - {e}")
         if data is not None and hasattr(data, '_mmap'):
             try: data._mmap.close()
             except Exception: pass
         return None, sampling_rate, n_channels, 0
    except Exception as e:
        print(f"An unexpected error occurred in load_lfp_data_memmap for {file_path}: {e}")
        traceback.print_exc()
        if data is not None and hasattr(data, '_mmap'):
            try: data._mmap.close()
            except Exception: pass
        return None, None, 0, 0

def load_channel_info(filepath):
    """Loads channel information from the CSV file."""
    print(f"Loading channel info from {filepath}")
    try:
        channel_df = pd.read_csv(filepath)
        # Ensure required columns exist
        required_cols = ['global_channel_index', 'shank_index', 'acronym', 'name']
        if not all(col in channel_df.columns for col in required_cols):
            raise ValueError(f"Channel info CSV must contain columns: {required_cols}")
        print(f"Loaded channel info for {len(channel_df)} channels.")
        return channel_df
    except FileNotFoundError:
        print(f"Error: Channel info file not found at {filepath}")
        raise
    except Exception as e:
        print(f"Error loading channel info: {e}")
        raise

def load_sleep_and_epoch_data(sleep_state_path, epoch_boundaries_path, fs):
    """
    Loads sleep state times and epoch boundaries, identifies non-REM periods.
    Args:
        sleep_state_path (Path or None): Path to the '*_sleep_states*.npy' file.
        epoch_boundaries_path (Path or None): Path to the '*_epoch_boundaries*.npy' file.
        fs (float): Sampling frequency needed to convert times to samples.
    Returns:
        tuple: (sleep_state_data, epoch_boundaries_sec, non_rem_periods_sec)
               Returns (None, [], []) on error or if files not found/provided.
    """
    print(f"Attempting to load sleep states from: {sleep_state_path}")
    print(f"Attempting to load epoch boundaries from: {epoch_boundaries_path}")

    sleep_state_data = None
    epoch_boundaries_sec = []
    non_rem_periods_sec = []

    # Load Epoch Boundaries
    if epoch_boundaries_path and epoch_boundaries_path.exists():
        try:
            epoch_boundaries_sec = np.load(epoch_boundaries_path, allow_pickle=True)
            if not isinstance(epoch_boundaries_sec, (list, np.ndarray)):
                 print(f"Warning: Epoch boundaries file {epoch_boundaries_path.name} content is not a list/array. Ignoring.")
                 epoch_boundaries_sec = []
            else:
                 # Validate format: list/array of tuples/lists with 2 numbers
                 valid_epochs = []
                 for i, ep in enumerate(epoch_boundaries_sec):
                     if isinstance(ep, (list, tuple, np.ndarray)) and len(ep) == 2 and all(isinstance(t, (int, float)) for t in ep):
                         if ep[1] >= ep[0]:
                            valid_epochs.append(tuple(ep))
                         else: print(f"Warning: Epoch {i} in {epoch_boundaries_path.name} has end_time < start_time. Skipping.")
                     else: print(f"Warning: Invalid format for epoch {i} in {epoch_boundaries_path.name}. Skipping.")
                 epoch_boundaries_sec = valid_epochs
                 print(f"Loaded {len(epoch_boundaries_sec)} valid epoch boundaries (sec).")
        except Exception as e:
            print(f"Error loading epoch boundaries file {epoch_boundaries_path.name}: {e}")
            epoch_boundaries_sec = []
    elif epoch_boundaries_path is None:
         print("No epoch boundaries file selected by user.")
    else: # Path provided but doesn't exist
        print(f"Warning: Selected epoch boundaries file not found at {epoch_boundaries_path}. Cannot perform epoch-specific analysis.")

    # Load Sleep States and find non-REM periods
    if sleep_state_path and sleep_state_path.exists():
        try:
            # Assume sleep_state_path points to the *states* file.
            state_codes = np.load(sleep_state_path, allow_pickle=True)

            # Try to find the corresponding times file based on the states filename
            times_path = None
            # Attempt 1: simple replace
            times_path_str_1 = sleep_state_path.name.replace('_sleep_states', '_sleep_state_times')
            times_path_1 = sleep_state_path.parent / times_path_str_1
            if times_path_1.exists():
                 times_path = times_path_1
            else:
                 # Attempt 2: regex base name + generic times name
                 base_name_match = re.match(r"^(.*?)_sleep_states.*\.npy$", sleep_state_path.name)
                 if base_name_match:
                      base_name = base_name_match.group(1)
                      times_path_alt_generic = sleep_state_path.parent / f"{base_name}_sleep_state_times.npy"
                      if times_path_alt_generic.exists():
                           times_path = times_path_alt_generic
                      else: # Attempt 3: base name + common suffix times name
                           times_path_alt_emg = sleep_state_path.parent / f"{base_name}_sleep_state_times_EMG.npy"
                           if times_path_alt_emg.exists():
                                times_path = times_path_alt_emg

            if times_path and times_path.exists():
                 state_times_sec = np.load(times_path, allow_pickle=True)
                 print(f"Loaded state codes ({len(state_codes)}) from {sleep_state_path.name}")
                 print(f"Loaded state times ({len(state_times_sec)}) from {times_path.name}")

                 if len(state_codes) == len(state_times_sec):
                     sleep_state_data = np.column_stack((state_times_sec, state_codes)) # Combine into [time, state]

                     # Find non-REM (state code 1) periods
                     nrem_indices = np.where(sleep_state_data[:, 1] == 1)[0]
                     if len(nrem_indices) > 0:
                         # Find contiguous blocks
                         diff = np.diff(nrem_indices)
                         splits = np.where(diff != 1)[0] + 1
                         nrem_blocks_indices = np.split(nrem_indices, splits)

                         for block in nrem_blocks_indices:
                             if len(block) > 0:
                                 start_idx = block[0]
                                 end_idx = block[-1]
                                 # Get times corresponding to the start and end of the NREM block
                                 start_time = sleep_state_data[start_idx, 0]
                                 # End time is the start time of the *next* state bin, or end of recording if last state
                                 if end_idx + 1 < len(sleep_state_data):
                                     end_time = sleep_state_data[end_idx + 1, 0]
                                 else:
                                     # Estimate end time based on step size if possible, otherwise add small duration
                                     if len(sleep_state_data) > 1:
                                         step = np.median(np.diff(sleep_state_data[:,0]))
                                         end_time = sleep_state_data[end_idx, 0] + step
                                     else: end_time = sleep_state_data[end_idx, 0] + 1.0 # Default 1s step

                                 non_rem_periods_sec.append((start_time, end_time))
                         print(f"Identified {len(non_rem_periods_sec)} non-REM periods (state code 1).")
                     else:
                         print("No NREM (state code 1) segments found in sleep state data.")

                 else:
                     print(f"Warning: Mismatch between state codes ({len(state_codes)}) and times ({len(state_times_sec)}). Cannot determine non-REM periods.")
                     sleep_state_data = None # Invalidate combined data

            else:
                print(f"Warning: Could not find corresponding times file for states ({sleep_state_path.name}). Tried various patterns. Cannot determine non-REM periods.")

        except Exception as e:
            print(f"Error loading sleep state file {sleep_state_path.name} or associated times file: {e}")
            sleep_state_data = None
            non_rem_periods_sec = [] # Ensure it's empty on error
    elif sleep_state_path is None:
         print("No sleep state file selected by user.")
    else: # Path provided but doesn't exist
        print(f"Warning: Selected sleep state file not found at {sleep_state_path}. Baseline calculation will use the whole signal.")

    if not non_rem_periods_sec:
         warnings.warn("No non-REM periods identified. Baseline calculation will use the whole signal.")

    return sleep_state_data, epoch_boundaries_sec, non_rem_periods_sec


# --- Signal Processing Functions --- (Mostly unchanged, added checks)

def apply_fir_filter(data, lowcut, highcut, fs, numtaps=513, pass_zero=False):
    """
    Applies a zero-lag linear phase FIR bandpass filter.
    numtaps should be odd for Type I linear phase filter.
    Handles potential edge cases with data length vs filter order.
    """
    if data is None or data.size == 0:
        # print("Error in apply_fir_filter: Input data is None or empty.") # Reduced verbosity
        return None

    data_len = data.shape[-1] # Assume time is the last axis
    if numtaps >= data_len:
        # print(f"Warning in apply_fir_filter: numtaps ({numtaps}) >= data length ({data_len}). Reducing numtaps.")
        numtaps = data_len - 1 if data_len > 1 else 1
        if numtaps % 2 == 0: numtaps -= 1 # Ensure odd
        if numtaps < 3:
             # print(f"Error in apply_fir_filter: Data length ({data_len}) too short for filtering. Returning original data.")
             return data # Or return None? Returning original might be safer downstream.

    try:
        if lowcut is None and highcut is None:
             # print("Warning in apply_fir_filter: Both lowcut and highcut are None. Returning original data.")
             return data
        elif lowcut is None: # Lowpass
            b = signal.firwin(numtaps, highcut, fs=fs, pass_zero=True, window='hamming')
        elif highcut is None: # Highpass
             b = signal.firwin(numtaps, lowcut, fs=fs, pass_zero=False, window='hamming')
        else: # Bandpass
            if lowcut >= highcut:
                 print(f"Error in apply_fir_filter: lowcut ({lowcut}) >= highcut ({highcut}).")
                 return None
            # Ensure Nyquist frequency is respected for the *current* fs
            nyq = fs / 2.0
            if highcut >= nyq:
                highcut = nyq * 0.99 # Adjust highcut if needed
                print(f"Warning: Highcut adjusted to {highcut:.2f} Hz due to sampling rate {fs:.2f} Hz.")
            if lowcut >= highcut: # Check again after adjustment
                 print(f"Error: lowcut ({lowcut}) >= adjusted highcut ({highcut}).")
                 return None

            b = signal.firwin(numtaps, [lowcut, highcut], fs=fs, pass_zero=pass_zero, window='hamming')

        # Apply filter forward and backward for zero phase delay
        # Ensure data is float for filtfilt
        if not np.issubdtype(data.dtype, np.floating):
             data = data.astype(np.float64) # Use float64 for precision
        filtered_data = signal.filtfilt(b, 1, data, axis=-1)
        return filtered_data
    except Exception as e:
        print(f"Error during FIR filtering: {e}")
        # traceback.print_exc() # Can be verbose
        return None


def calculate_instantaneous_power(data):
    """Calculates instantaneous power using Hilbert transform."""
    if data is None or data.size == 0:
        # print("Error in calculate_instantaneous_power: Input data is None or empty.")
        return None
    try:
        # Ensure data is float for hilbert
        if not np.issubdtype(data.dtype, np.floating):
             data = data.astype(np.float64)
        analytic_signal = signal.hilbert(data, axis=-1)
        amplitude_envelope = np.abs(analytic_signal)
        power = amplitude_envelope**2
        return power
    except Exception as e:
        print(f"Error calculating instantaneous power: {e}")
        # traceback.print_exc()
        return None

def get_baseline_stats(power_signal, fs, non_rem_periods_samples):
    """
    Calculates baseline mean and SD from power during specified periods (non-REM).
    Uses the described method: clip@4SD, rectify, low-pass 55Hz.
    """
    if power_signal is None or power_signal.size == 0:
        raise ValueError("Cannot compute baseline statistics: Input power signal is invalid.")

    baseline_power_segments = []
    if not non_rem_periods_samples: # Use whole signal if no non-REM defined
        # warnings.warn("Calculating baseline stats over the entire signal duration (no non-REM periods provided).")
        baseline_power_segments = [power_signal]
    else:
        # print(f"Calculating baseline stats using {len(non_rem_periods_samples)} non-REM periods.")
        for start, end in non_rem_periods_samples:
             start = max(0, start) # Ensure indices are within bounds
             end = min(power_signal.shape[0], end)
             if start < end : # Ensure segment has positive length
                 baseline_power_segments.append(power_signal[start:end])

        if not baseline_power_segments:
             warnings.warn("No valid non-REM segments found within signal bounds after clipping. Calculating baseline over entire signal.")
             baseline_power_segments = [power_signal]


    if not baseline_power_segments:
         raise ValueError("Cannot compute baseline statistics: No valid data segments found.")

    # Concatenate segments carefully to avoid memory issues if possible
    try:
        total_baseline_samples = sum(seg.size for seg in baseline_power_segments)
        if total_baseline_samples == 0:
             raise ValueError("Cannot compute baseline statistics: Concatenated baseline power is empty.")

        if total_baseline_samples * power_signal.itemsize > 1e9: # Heuristic check
             warnings.warn("Large baseline data size (>1GB). Concatenation might be slow or fail.")

        full_baseline_power = np.concatenate(baseline_power_segments)
    except MemoryError:
        print("MemoryError during baseline segment concatenation. Cannot compute baseline.")
        raise ValueError("MemoryError during baseline calculation.")
    except Exception as e:
        print(f"Error concatenating baseline segments: {e}")
        raise ValueError("Error preparing baseline data.")

    # Calculate initial SD for clipping
    initial_mean = np.mean(full_baseline_power)
    initial_sd = np.std(full_baseline_power)

    if initial_sd == 0:
        # warnings.warn("Initial SD of baseline power is zero. Clipping might not be effective.")
        clipped_power = full_baseline_power
    else:
        clip_threshold = initial_mean + 4 * initial_sd
        clipped_power = np.clip(full_baseline_power, a_min=None, a_max=clip_threshold)

    rectified_power = np.abs(clipped_power) # Redundant for power

    # Low-pass filter at 55 Hz
    numtaps_lp = 101 # Adjust as needed
    if numtaps_lp >= rectified_power.size:
         # warnings.warn(f"Baseline data too short ({rectified_power.size}) for LP filter numtaps ({numtaps_lp}). Skipping LP filter.")
         processed_baseline_power = rectified_power
    else:
         try:
             # Ensure cutoff is valid for the *current* fs
             nyq_lp = fs / 2.0
             lp_cutoff = RIPPLE_POWER_LP_CUTOFF
             if lp_cutoff >= nyq_lp:
                  lp_cutoff = nyq_lp * 0.99
                  # warnings.warn(f"Baseline LP cutoff adjusted to {lp_cutoff:.2f} Hz for fs={fs:.2f} Hz.")

             b_lp = signal.firwin(numtaps_lp, lp_cutoff, fs=fs, pass_zero=True, window='hamming')
             processed_baseline_power = signal.filtfilt(b_lp, 1, rectified_power)
         except Exception as e_lp:
              warnings.warn(f"Error applying baseline LP filter: {e_lp}. Using rectified power directly.")
              processed_baseline_power = rectified_power

    # Final mean and SD for detection thresholds
    baseline_mean = np.mean(processed_baseline_power)
    baseline_sd = np.std(processed_baseline_power)

    # if baseline_sd == 0:
    #      warnings.warn("Final baseline SD is zero. Detection thresholds might be problematic.")

    # print(f"Baseline calculated: Mean={baseline_mean:.4f}, SD={baseline_sd:.4f} (based on {total_baseline_samples} samples)")
    return baseline_mean, baseline_sd


def detect_events(signal_data, fs, threshold_high, threshold_low, min_duration_ms, merge_gap_ms, event_type="Event"):
    """
    Detects events based on signal thresholds, duration, and merging gaps.
    """
    if signal_data is None or signal_data.size == 0:
        # print(f"Error in detect_events: Input signal_data for {event_type} is invalid.")
        return np.array([], dtype=int), np.array([], dtype=int)

    min_duration_samples = int(min_duration_ms * fs / 1000)
    merge_gap_samples = int(merge_gap_ms * fs / 1000)

    try:
        with np.errstate(invalid='ignore'): # Ignore potential NaNs temporarily
            above_high_threshold = signal_data > threshold_high
        if np.any(np.isnan(signal_data)):
            # warnings.warn(f"NaNs found in input signal for {event_type} detection. Treating NaNs as below threshold.")
            above_high_threshold[np.isnan(signal_data)] = False

        diff_high = np.diff(above_high_threshold.astype(np.int8))
        starts_high = np.where(diff_high == 1)[0] + 1
        ends_high = np.where(diff_high == -1)[0]

        if len(above_high_threshold) > 0:
            if above_high_threshold[0]: starts_high = np.insert(starts_high, 0, 0)
            if above_high_threshold[-1]: ends_high = np.append(ends_high, len(signal_data) - 1)
        else:
             return np.array([], dtype=int), np.array([], dtype=int)

    except Exception as e_thresh:
        print(f"Error during thresholding for {event_type}: {e_thresh}")
        return np.array([], dtype=int), np.array([], dtype=int)

    if len(starts_high) == 0 or len(ends_high) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    valid_pairs = []
    current_start_idx = 0
    current_end_idx = 0
    while current_start_idx < len(starts_high) and current_end_idx < len(ends_high):
        start_val = starts_high[current_start_idx]
        end_val = ends_high[current_end_idx]
        if start_val <= end_val:
            valid_pairs.append((start_val, end_val))
            current_start_idx += 1
            current_end_idx += 1
        else:
            current_end_idx += 1

    if not valid_pairs:
        return np.array([], dtype=int), np.array([], dtype=int)

    starts_high, ends_high = zip(*valid_pairs)
    starts_high = np.array(starts_high)
    ends_high = np.array(ends_high)

    expanded_starts = []
    expanded_ends = []
    try:
        with np.errstate(invalid='ignore'):
             below_low_mask = signal_data < threshold_low
        if np.any(np.isnan(signal_data)):
             below_low_mask[np.isnan(signal_data)] = True

        for start, end in zip(starts_high, ends_high):
            current_start = start
            while current_start > 0 and not below_low_mask[current_start - 1]: current_start -= 1
            current_end = end
            while current_end < len(signal_data) - 1 and not below_low_mask[current_end + 1]: current_end += 1
            expanded_starts.append(current_start)
            expanded_ends.append(current_end)
    except Exception as e_expand:
         print(f"Error during event expansion for {event_type}: {e_expand}")
         return np.array([], dtype=int), np.array([], dtype=int)

    if not expanded_starts:
        return np.array([], dtype=int), np.array([], dtype=int)

    merged_starts = [expanded_starts[0]]
    merged_ends = [expanded_ends[0]]
    for i in range(1, len(expanded_starts)):
        gap = expanded_starts[i] - merged_ends[-1] - 1
        if gap < merge_gap_samples:
            merged_ends[-1] = max(merged_ends[-1], expanded_ends[i])
        else:
            merged_starts.append(expanded_starts[i])
            merged_ends.append(expanded_ends[i])

    final_starts = []
    final_ends = []
    for start, end in zip(merged_starts, merged_ends):
        duration_samples = end - start + 1
        if duration_samples >= min_duration_samples:
            final_starts.append(start)
            final_ends.append(end)

    return np.array(final_starts, dtype=int), np.array(final_ends, dtype=int)

def find_event_features(lfp_segment, power_segment):
    """
    Finds peak power index and nearest trough index within an event segment.
    """
    if power_segment is None or lfp_segment is None or power_segment.size == 0 or lfp_segment.size == 0: return -1, -1
    if len(lfp_segment) != len(power_segment):
         # warnings.warn(f"LFP/Power segment length mismatch in find_event_features.")
         return -1,-1

    try:
        peak_power_idx_rel = np.nanargmax(power_segment)

        if not np.issubdtype(lfp_segment.dtype, np.floating): lfp_segment_float = lfp_segment.astype(np.float64)
        else: lfp_segment_float = lfp_segment

        troughs_rel, _ = signal.find_peaks(-lfp_segment_float)

        if len(troughs_rel) == 0:
            trough_idx_rel = np.nanargmin(lfp_segment_float)
        else:
            trough_idx_rel = troughs_rel[np.nanargmin(np.abs(troughs_rel - peak_power_idx_rel))]

        return peak_power_idx_rel, trough_idx_rel

    except Exception as e:
        # warnings.warn(f"Error finding event features: {e}")
        return -1, -1


def calculate_cwt_spectrogram(data, fs, freqs, timestamp_samples, window_samples):
    """Calculates wavelet spectrogram around a timestamp using PyWavelets."""
    if data is None or data.size == 0: return np.full((len(freqs), window_samples), np.nan)

    start_sample = int(timestamp_samples - window_samples // 2)
    end_sample = int(start_sample + window_samples)

    pad_left = 0
    pad_right = 0
    if start_sample < 0: pad_left = -start_sample; start_sample = 0
    if end_sample > len(data): pad_right = end_sample - len(data); end_sample = len(data)

    segment = data[start_sample:end_sample]
    if segment.size == 0: return np.full((len(freqs), window_samples), np.nan)

    if pad_left > 0 or pad_right > 0:
        try: segment = np.pad(segment, (pad_left, pad_right), mode='reflect')
        except ValueError: segment = np.pad(segment, (pad_left, pad_right), mode='edge')

    if len(segment) != window_samples:
         # warnings.warn(f"Segment length {len(segment)} != window_samples {window_samples} after padding.")
         # Adjust segment length if padding failed or resulted in wrong size
         if len(segment) > window_samples: segment = segment[:window_samples]
         else: segment = np.pad(segment, (0, window_samples - len(segment)), mode='edge')


    try:
        wavelet = f'cmor1.5-1.0'
        scales = pywt.frequency2scale(wavelet, freqs / fs) # Use the provided fs (should be downsampled fs if called from worker)
        coeffs, _ = pywt.cwt(segment.astype(np.float64), scales, wavelet, sampling_period=1.0/fs)
        power_spectrogram = np.abs(coeffs)**2

        if power_spectrogram.shape[1] != window_samples:
             # warnings.warn(f"Spectrogram width ({power_spectrogram.shape[1]}) != window ({window_samples}). Interpolating.")
             from scipy.interpolate import interp1d
             x_old = np.linspace(0, 1, power_spectrogram.shape[1])
             x_new = np.linspace(0, 1, window_samples)
             interp_func = interp1d(x_old, power_spectrogram, axis=1, kind='linear', fill_value='extrapolate')
             power_spectrogram = interp_func(x_new)

        return power_spectrogram

    except MemoryError:
         # warnings.warn(f"MemoryError calculating CWT spectrogram for timestamp {timestamp_samples}.")
         return np.full((len(freqs), window_samples), np.nan)
    except Exception as e:
         # warnings.warn(f"Error calculating CWT spectrogram for timestamp {timestamp_samples}: {e}")
         return np.full((len(freqs), window_samples), np.nan)


# --- Worker Functions for Parallel Processing ---

def _calculate_baseline_worker(ch_idx, lfp_filepath, meta_filepath, non_rem_periods_samples_orig, fs_orig, n_channels):
    """
    Worker function to calculate baseline stats for a single channel.
    Includes downsampling step.
    """
    lfp_data_memmap_worker = None
    fs_eff = fs_orig / DOWNSAMPLING_FACTOR if DOWNSAMPLING_FACTOR > 1 else fs_orig
    try:
        # Get n_samples for memmap shape
        meta = readMeta(meta_filepath)
        file_size = lfp_filepath.stat().st_size
        item_size = np.dtype('int16').itemsize
        expected_total_bytes = file_size - (file_size % (n_channels * item_size))
        n_samples = expected_total_bytes // (n_channels * item_size)
        shape = (n_samples, n_channels)

        lfp_data_memmap_worker = np.memmap(lfp_filepath, dtype='int16', mode='r', shape=shape)
        lfp_ch_full_orig = lfp_data_memmap_worker[:, ch_idx].astype(np.float64)

        # --- Downsample ---
        if DOWNSAMPLING_FACTOR > 1:
            lfp_ch_full_down = signal.decimate(lfp_ch_full_orig, q=DOWNSAMPLING_FACTOR, axis=0, ftype='fir', zero_phase=True)
            del lfp_ch_full_orig # Free original data
        else:
            lfp_ch_full_down = lfp_ch_full_orig # No downsampling

        # --- Process Downsampled Data ---
        lfp_ch_ripple_filtered_down = apply_fir_filter(lfp_ch_full_down, RIPPLE_FILTER_LOWCUT, RIPPLE_FILTER_HIGHCUT, fs_eff)
        del lfp_ch_full_down
        if lfp_ch_ripple_filtered_down is None: return ch_idx, None

        ripple_power_down = calculate_instantaneous_power(lfp_ch_ripple_filtered_down)
        del lfp_ch_ripple_filtered_down
        if ripple_power_down is None: return ch_idx, None

        # Adjust non-REM periods to downsampled timebase
        non_rem_periods_samples_down = [(s // DOWNSAMPLING_FACTOR, e // DOWNSAMPLING_FACTOR) for s, e in non_rem_periods_samples_orig]

        baseline_mean, baseline_sd = get_baseline_stats(ripple_power_down, fs_eff, non_rem_periods_samples_down)
        del ripple_power_down
        return ch_idx, (baseline_mean, baseline_sd)

    except Exception as e:
        print(f"Error in baseline worker for channel {ch_idx}: {e}")
        return ch_idx, None
    finally:
        if lfp_data_memmap_worker is not None and hasattr(lfp_data_memmap_worker, '_mmap'):
            try: lfp_data_memmap_worker._mmap.close()
            except Exception: pass
        gc.collect()


def _detect_events_worker(ch_idx, ch_region, epoch_start_sample_orig, epoch_end_sample_orig,
                         lfp_filepath, n_samples_orig, n_channels, baseline_stats, fs_orig):
    """
    Worker function to detect events for a single channel within an epoch.
    Includes downsampling and scales indices back to original fs.
    """
    lfp_data_memmap_worker = None
    ripple_events = []
    spw_events = []
    fs_eff = fs_orig / DOWNSAMPLING_FACTOR if DOWNSAMPLING_FACTOR > 1 else fs_orig

    try:
        # Create memmap view in worker
        shape = (n_samples_orig, n_channels)
        lfp_data_memmap_worker = np.memmap(lfp_filepath, dtype='int16', mode='r', shape=shape)

        # Load the required segment at original sampling rate
        lfp_ch_epoch_orig = lfp_data_memmap_worker[epoch_start_sample_orig:epoch_end_sample_orig, ch_idx].astype(np.float64)

        # --- Downsample ---
        if DOWNSAMPLING_FACTOR > 1:
            lfp_ch_epoch_down = signal.decimate(lfp_ch_epoch_orig, q=DOWNSAMPLING_FACTOR, axis=0, ftype='fir', zero_phase=True)
        else:
            lfp_ch_epoch_down = lfp_ch_epoch_orig # No downsampling

        del lfp_ch_epoch_orig # Free original epoch data

        # --- Ripple Detection on Downsampled Data ---
        lfp_ch_ripple_filtered_down = apply_fir_filter(lfp_ch_epoch_down, RIPPLE_FILTER_LOWCUT, RIPPLE_FILTER_HIGHCUT, fs_eff)
        if lfp_ch_ripple_filtered_down is not None:
            ripple_power_down = calculate_instantaneous_power(lfp_ch_ripple_filtered_down)
            if ripple_power_down is not None:
                baseline_mean, baseline_sd = baseline_stats
                if baseline_sd > 0:
                    detection_threshold = baseline_mean + RIPPLE_DETECTION_SD_THRESHOLD * baseline_sd
                    expansion_threshold = baseline_mean + RIPPLE_EXPANSION_SD_THRESHOLD * baseline_sd

                    # Detect events relative to start of the downsampled epoch segment
                    ripple_starts_rel_down, ripple_ends_rel_down = detect_events(
                        ripple_power_down, fs_eff, threshold_high=detection_threshold, threshold_low=expansion_threshold,
                        min_duration_ms=RIPPLE_MIN_DURATION_MS, merge_gap_ms=RIPPLE_MERGE_GAP_MS, event_type="Ripple"
                    )

                    for start_rel_down, end_rel_down in zip(ripple_starts_rel_down, ripple_ends_rel_down):
                        if start_rel_down < 0 or end_rel_down >= len(lfp_ch_ripple_filtered_down): continue

                        lfp_segment_down = lfp_ch_ripple_filtered_down[start_rel_down:end_rel_down+1]
                        power_segment_down = ripple_power_down[start_rel_down:end_rel_down+1]
                        peak_power_idx_rel_down, trough_idx_rel_down = find_event_features(lfp_segment_down, power_segment_down)

                        if peak_power_idx_rel_down != -1 and trough_idx_rel_down != -1:
                            # --- Scale relative downsampled indices back to absolute original indices ---
                            start_sample_abs = epoch_start_sample_orig + (start_rel_down * DOWNSAMPLING_FACTOR)
                            # Estimate end by scaling duration (more robust than scaling end index directly)
                            duration_samples_down = end_rel_down - start_rel_down + 1
                            end_sample_abs = start_sample_abs + (duration_samples_down * DOWNSAMPLING_FACTOR) - 1 # Inclusive end

                            peak_sample_abs = epoch_start_sample_orig + ((start_rel_down + peak_power_idx_rel_down) * DOWNSAMPLING_FACTOR)
                            trough_sample_abs = epoch_start_sample_orig + ((start_rel_down + trough_idx_rel_down) * DOWNSAMPLING_FACTOR)

                            peak_power_val = power_segment_down[peak_power_idx_rel_down]
                            duration_ms = duration_samples_down / fs_eff * 1000 # Duration from downsampled data

                            ripple_events.append({
                                'start_sample': start_sample_abs, 'end_sample': end_sample_abs,
                                'peak_sample': peak_sample_abs, 'trough_sample': trough_sample_abs,
                                'peak_power': peak_power_val, 'duration_ms': duration_ms
                            })
                del ripple_power_down
            del lfp_ch_ripple_filtered_down

        # --- SPW Detection (CA1 only) on Downsampled Data ---
        if ch_region == 'CA1':
            lfp_ch_spw_filtered_down = apply_fir_filter(lfp_ch_epoch_down, SPW_FILTER_LOWCUT, SPW_FILTER_HIGHCUT, fs_eff)
            if lfp_ch_spw_filtered_down is not None:
                spw_signal_mean_down = np.nanmean(lfp_ch_spw_filtered_down)
                spw_signal_sd_down = np.nanstd(lfp_ch_spw_filtered_down)

                if spw_signal_sd_down > 0 and np.isfinite(spw_signal_sd_down):
                    spw_detection_threshold = SPW_DETECTION_SD_THRESHOLD * spw_signal_sd_down
                    spw_expansion_threshold = 1.0 * spw_signal_sd_down

                    spw_starts_rel_down, spw_ends_rel_down = detect_events(
                         np.abs(lfp_ch_spw_filtered_down - spw_signal_mean_down), fs_eff, threshold_high=spw_detection_threshold,
                         threshold_low=spw_expansion_threshold, min_duration_ms=SPW_MIN_DURATION_MS,
                         merge_gap_ms=1, event_type="SPW"
                    )

                    for start_rel_down, end_rel_down in zip(spw_starts_rel_down, spw_ends_rel_down):
                         if start_rel_down < 0 or end_rel_down >= len(lfp_ch_spw_filtered_down): continue

                         duration_samples_down = end_rel_down - start_rel_down + 1
                         duration_ms = duration_samples_down / fs_eff * 1000

                         if duration_ms <= SPW_MAX_DURATION_MS:
                              spw_segment_down = lfp_ch_spw_filtered_down[start_rel_down:end_rel_down+1]
                              trough_idx_rel_in_segment_down = np.nanargmin(spw_segment_down)

                              # --- Scale indices back ---
                              start_sample_abs = epoch_start_sample_orig + (start_rel_down * DOWNSAMPLING_FACTOR)
                              end_sample_abs = start_sample_abs + (duration_samples_down * DOWNSAMPLING_FACTOR) - 1
                              trough_sample_abs = epoch_start_sample_orig + ((start_rel_down + trough_idx_rel_in_segment_down) * DOWNSAMPLING_FACTOR)

                              spw_events.append({
                                  'start_sample': start_sample_abs, 'end_sample': end_sample_abs,
                                  'trough_sample': trough_sample_abs, 'duration_ms': duration_ms
                              })
                del lfp_ch_spw_filtered_down

        del lfp_ch_epoch_down # Free downsampled epoch segment memory
        return ch_idx, ripple_events, spw_events

    except Exception as e:
        print(f"Error in event worker for channel {ch_idx}, epoch {epoch_start_sample_orig}-{epoch_end_sample_orig}: {e}")
        return ch_idx, [], []
    finally:
        if lfp_data_memmap_worker is not None and hasattr(lfp_data_memmap_worker, '_mmap'):
            try: lfp_data_memmap_worker._mmap.close()
            except Exception: pass
        gc.collect()


def _calculate_spectrogram_worker(ts_sample_orig, spec_ch_idx, lfp_filepath, n_samples_orig, n_channels, fs_orig):
    """
    Worker function to calculate spectrogram for a single timestamp.
    Includes downsampling. Timestamp is relative to original fs.
    """
    lfp_data_memmap_worker = None
    fs_eff = fs_orig / DOWNSAMPLING_FACTOR if DOWNSAMPLING_FACTOR > 1 else fs_orig
    # Adjust window size based on original fs for loading, but use downsampled fs for CWT
    window_samples_orig = int(SPECTROGRAM_WINDOW_MS * fs_orig / 1000)
    window_samples_eff = int(SPECTROGRAM_WINDOW_MS * fs_eff / 1000) # Window size for CWT

    try:
        # Create memmap view in worker
        shape = (n_samples_orig, n_channels)
        lfp_data_memmap_worker = np.memmap(lfp_filepath, dtype='int16', mode='r', shape=shape)

        # --- Load segment at original fs ---
        start_sample_orig = int(ts_sample_orig - window_samples_orig // 2)
        end_sample_orig = int(start_sample_orig + window_samples_orig)
        # Handle boundaries for loading
        pad_left_orig = 0
        pad_right_orig = 0
        if start_sample_orig < 0: pad_left_orig = -start_sample_orig; start_sample_orig = 0
        if end_sample_orig > n_samples_orig: pad_right_orig = end_sample_orig - n_samples_orig; end_sample_orig = n_samples_orig

        segment_orig = lfp_data_memmap_worker[start_sample_orig:end_sample_orig, spec_ch_idx].astype(np.float64)

        # Pad original segment if needed before downsampling
        if pad_left_orig > 0 or pad_right_orig > 0:
             try: segment_orig = np.pad(segment_orig, (pad_left_orig, pad_right_orig), mode='reflect')
             except ValueError: segment_orig = np.pad(segment_orig, (pad_left_orig, pad_right_orig), mode='edge')

        # --- Downsample the segment ---
        if DOWNSAMPLING_FACTOR > 1 and segment_orig.size > DOWNSAMPLING_FACTOR * 2: # Need enough samples
             segment_down = signal.decimate(segment_orig, q=DOWNSAMPLING_FACTOR, axis=0, ftype='fir', zero_phase=True)
        else:
             segment_down = segment_orig # Use original if no downsampling or segment too short

        del segment_orig # Free original segment

        # --- Calculate spectrogram on downsampled data ---
        # The center timestamp for the downsampled CWT should correspond to the original timestamp
        # The CWT function itself handles the windowing around the segment center
        spec = calculate_cwt_spectrogram(
            segment_down, fs_eff, SPECTROGRAM_FREQS,
            timestamp_samples=len(segment_down) // 2, # Use center of the downsampled segment
            window_samples=window_samples_eff
        )
        del segment_down
        return spec

    except Exception as e:
        # print(f"Error in spectrogram worker for timestamp {ts_sample_orig}, channel {spec_ch_idx}: {e}") # Verbose
        return np.full((len(SPECTROGRAM_FREQS), window_samples_eff), np.nan)
    finally:
        if lfp_data_memmap_worker is not None and hasattr(lfp_data_memmap_worker, '_mmap'):
            try: lfp_data_memmap_worker._mmap.close()
            except Exception: pass
        gc.collect()


# --- Main Analysis Function ---

def run_ripple_analysis(lfp_filepath, meta_filepath, channel_info_filepath,
                        sleep_state_filepath, epoch_boundaries_filepath, # Updated inputs
                        output_dir):
    """Main function to run the ripple and SPW detection analysis with parallel processing and downsampling."""

    # Ensure input paths are Path objects
    lfp_filepath = Path(lfp_filepath)
    meta_filepath = Path(meta_filepath)
    channel_info_filepath = Path(channel_info_filepath)
    sleep_state_filepath = Path(sleep_state_filepath) if sleep_state_filepath else None
    epoch_boundaries_filepath = Path(epoch_boundaries_filepath) if epoch_boundaries_filepath else None
    output_dir = Path(output_dir)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_path}")

    # --- Derive base filename ---
    lfp_filename = lfp_filepath.name
    match = re.match(r"^(.*?)(\.(imec|nidq)\d?)?\.lf\.bin$", lfp_filename)
    output_filename_base = match.group(1) if match else lfp_filename.replace('.lf.bin', '')
    print(f"Derived base filename: {output_filename_base}")

    # Initialize results storage
    all_ripple_events_by_epoch = {}
    all_spw_events_by_epoch = {}
    all_ripple_events_global = {}
    all_spw_events_global = {}
    ca1_spwr_events_global = {}
    reference_sites = {}
    averaged_timestamps = {}
    cooccurrence_results = {}
    averaged_spectrograms = {}

    lfp_data_memmap = None # Define outside try block

    try:
        # 1. Load LFP Data using Memmap (Get dimensions and fs)
        print("\n--- 1. Loading LFP Data ---")
        _, fs_orig, n_channels, n_samples_orig = load_lfp_data_memmap(lfp_filepath, meta_filepath)
        if fs_orig is None or n_channels == 0 or n_samples_orig == 0:
            raise ValueError("Failed to load LFP data or get valid dimensions/sampling rate.")
        duration_sec = n_samples_orig / fs_orig
        fs_eff = fs_orig / DOWNSAMPLING_FACTOR if DOWNSAMPLING_FACTOR > 1 else fs_orig
        print(f"LFP data info: {n_channels} channels, {n_samples_orig} samples ({duration_sec:.2f} seconds)")
        print(f"Original Fs: {fs_orig:.6f} Hz. Effective Fs for analysis: {fs_eff:.6f} Hz")


        # 2. Load Channel Info
        print("\n--- 2. Loading Channel Info ---")
        channel_df = load_channel_info(channel_info_filepath)
        target_regions = ['CA1', 'CA3', 'CA2']
        region_channels_df = channel_df[channel_df['acronym'].isin(target_regions)].copy()
        if region_channels_df.empty: raise ValueError(f"No channels found in target regions: {target_regions}")
        print(f"Found {len(region_channels_df)} channels in {target_regions}.")
        channels_to_process = sorted(region_channels_df['global_channel_index'].astype(int).unique())
        channel_region_map = pd.Series(region_channels_df.acronym.values, index=region_channels_df.global_channel_index).to_dict()


        # 3. Load Sleep State and Epoch Data
        print("\n--- 3. Loading Sleep State and Epoch Data ---")
        sleep_state_data, epoch_boundaries_sec, non_rem_periods_sec = load_sleep_and_epoch_data(
            sleep_state_filepath, epoch_boundaries_filepath, fs_orig
        )
        # IMPORTANT: Keep indices relative to ORIGINAL sampling rate for epoch boundaries and non-REM periods
        non_rem_periods_samples_orig = [(int(start * fs_orig), int(end * fs_orig)) for start, end in non_rem_periods_sec]
        epoch_boundaries_samples_orig = [(int(start * fs_orig), int(end * fs_orig)) for start, end in epoch_boundaries_sec]
        if not epoch_boundaries_samples_orig:
             warnings.warn("No epoch boundaries loaded. Processing entire recording as one epoch.")
             epoch_boundaries_samples_orig = [(0, n_samples_orig)]


        # 4. Calculate Global Baseline Statistics (Parallel)
        print(f"\n--- 4. Calculating Global Baseline Statistics (Parallel using {NUM_CORES} cores) ---")
        baseline_stats_per_channel = {}
        baseline_tasks = [(ch_idx, lfp_filepath, meta_filepath, non_rem_periods_samples_orig, fs_orig, n_channels)
                          for ch_idx in channels_to_process if ch_idx < n_channels]

        results = Parallel(n_jobs=NUM_CORES, backend="loky")(
            delayed(_calculate_baseline_worker)(*task) for task in baseline_tasks
        )

        valid_baselines = 0
        for ch_idx, stats_result in results:
            if stats_result is not None:
                baseline_stats_per_channel[ch_idx] = stats_result
                valid_baselines += 1
            else:
                print(f"Warning: Baseline calculation failed for channel {ch_idx}.")
        print(f"Successfully calculated baseline stats for {valid_baselines}/{len(channels_to_process)} relevant channels.")
        if valid_baselines == 0:
             raise ValueError("Baseline calculation failed for all relevant channels. Cannot proceed.")


        # 5. Per-Epoch Event Detection (Parallel over channels within each epoch)
        print(f"\n--- 5. Detecting Ripples and Sharp Waves (Parallel using {NUM_CORES} cores per epoch) ---")
        for epoch_idx, (epoch_start_sample_orig, epoch_end_sample_orig) in enumerate(epoch_boundaries_samples_orig):
            print(f"\nProcessing Epoch {epoch_idx} (Original Samples: {epoch_start_sample_orig} - {epoch_end_sample_orig})...")
            epoch_duration_samples_orig = epoch_end_sample_orig - epoch_start_sample_orig
            if epoch_duration_samples_orig <= 0: continue

            all_ripple_events_by_epoch[epoch_idx] = {}
            all_spw_events_by_epoch[epoch_idx] = {}

            epoch_tasks = []
            for ch_idx in channels_to_process:
                 if ch_idx in baseline_stats_per_channel:
                      ch_region = channel_region_map.get(ch_idx, 'Unknown')
                      epoch_tasks.append((ch_idx, ch_region, epoch_start_sample_orig, epoch_end_sample_orig,
                                          lfp_filepath, n_samples_orig, n_channels, baseline_stats_per_channel[ch_idx], fs_orig))

            if not epoch_tasks: continue

            print(f"  Detecting events for {len(epoch_tasks)} channels in parallel...")
            epoch_results = Parallel(n_jobs=NUM_CORES, backend="loky")(
                delayed(_detect_events_worker)(*task) for task in epoch_tasks
            )

            # Process results for this epoch (Indices are already scaled back to original fs in worker)
            for ch_idx, ripple_events, spw_events in epoch_results:
                 for ev in ripple_events: ev['epoch_idx'] = epoch_idx
                 for ev in spw_events: ev['epoch_idx'] = epoch_idx

                 if ripple_events:
                      all_ripple_events_by_epoch[epoch_idx][ch_idx] = ripple_events
                      all_ripple_events_global.setdefault(ch_idx, []).extend(ripple_events)
                 if spw_events:
                      all_spw_events_by_epoch[epoch_idx][ch_idx] = spw_events
                      all_spw_events_global.setdefault(ch_idx, []).extend(spw_events)

            print(f"  Finished event detection for Epoch {epoch_idx}.")
            gc.collect()


        # --- Post-Epoch Processing (using global event lists referenced to original fs) ---

        # 6. SPW-R Coincidence Check (CA1 only - using global lists)
        print("\n--- 6. Checking SPW-Ripple Coincidence for CA1 (Globally) ---")
        # ... (Logic remains the same as indices are already absolute original fs) ...
        ca1_channel_indices = region_channels_df[region_channels_df['acronym'] == 'CA1']['global_channel_index'].values
        for ch_idx in ca1_channel_indices:
             if ch_idx not in all_ripple_events_global: continue
             ripples = all_ripple_events_global[ch_idx]
             if ch_idx not in all_spw_events_global or not all_spw_events_global[ch_idx]: continue
             spws = all_spw_events_global[ch_idx]
             coincident_ripples = []
             for ripple in ripples:
                 ripple_start, ripple_end = ripple['start_sample'], ripple['end_sample']
                 is_coincident = False
                 associated_spw_trough = None
                 for spw in spws:
                     spw_start, spw_end = spw['start_sample'], spw['end_sample']
                     if ripple_start <= spw_end and ripple_end >= spw_start:
                         is_coincident = True
                         associated_spw_trough = spw['trough_sample']
                         break
                 if is_coincident:
                     ripple['associated_spw_trough_sample'] = associated_spw_trough
                     coincident_ripples.append(ripple)
             if coincident_ripples:
                  ca1_spwr_events_global[ch_idx] = coincident_ripples
                  print(f"  Channel {ch_idx} (CA1): Found {len(coincident_ripples)} SPW-R events globally.")


        # 7. Determine Reference Sites per Shank (Globally)
        print("\n--- 7. Determining Reference Sites per Shank (Globally) ---")
        # ... (Logic remains the same) ...
        shanks = region_channels_df['shank_index'].unique()
        reference_sites = {}
        for shank_idx in shanks:
            shank_channels_df = region_channels_df[region_channels_df['shank_index'] == shank_idx]
            ref_sites_for_shank = {}
            for region in target_regions:
                region_shank_channels = shank_channels_df[shank_channels_df['acronym'] == region]
                if region_shank_channels.empty: continue
                max_ripple_metric = -1
                ref_ch_idx = -1
                for _, ch_row in region_shank_channels.iterrows():
                    ch_idx = int(ch_row['global_channel_index'])
                    events_to_consider = []
                    if region == 'CA1':
                         if ch_idx in ca1_spwr_events_global: events_to_consider = ca1_spwr_events_global[ch_idx]
                    else:
                         if ch_idx in all_ripple_events_global: events_to_consider = all_ripple_events_global[ch_idx]
                    if not events_to_consider: continue
                    peak_powers = [event['peak_power'] for event in events_to_consider if 'peak_power' in event and np.isfinite(event['peak_power'])]
                    if not peak_powers: continue
                    mean_power = np.mean(peak_powers)
                    if mean_power > max_ripple_metric: max_ripple_metric = mean_power; ref_ch_idx = ch_idx
                if ref_ch_idx != -1:
                    ref_sites_for_shank[region] = ref_ch_idx
                    print(f"  Shank {shank_idx}, Region {region}: Reference site Channel {ref_ch_idx} (Avg Power: {max_ripple_metric:.2f})")
            if ref_sites_for_shank: reference_sites[shank_idx] = ref_sites_for_shank


        # 8. Generate Averaged Timestamps per Region (Globally)
        print("\n--- 8. Generating Averaged Timestamps per Region (Globally) ---")
        # ... (Logic remains the same, timestamps are absolute original fs) ...
        averaged_timestamps = {}
        for region in target_regions:
            regional_ref_sites_indices = [sites[region] for shank, sites in reference_sites.items() if region in sites]
            if not regional_ref_sites_indices: continue
            all_regional_event_times = []
            event_type = "Unknown"
            if region == 'CA1':
                event_type = "SPW Trough (SPW-R)"
                for ch_idx in regional_ref_sites_indices:
                    if ch_idx in ca1_spwr_events_global: all_regional_event_times.extend([event['associated_spw_trough_sample'] for event in ca1_spwr_events_global[ch_idx] if 'associated_spw_trough_sample' in event])
            elif region in ['CA2', 'CA3']:
                 event_type = "Ripple Peak"
                 for ch_idx in regional_ref_sites_indices:
                     if ch_idx in all_ripple_events_global: all_regional_event_times.extend([event['peak_sample'] for event in all_ripple_events_global[ch_idx]])
            if not all_regional_event_times: continue
            pooled_times = np.sort(np.unique(all_regional_event_times))
            averaged_timestamps[region] = pooled_times # These are absolute original fs indices
            print(f"  Region {region}: Generated {len(pooled_times)} pooled {event_type} timestamps.")


        # 9. Co-occurrence Detection (Globally)
        print("\n--- 9. Detecting Co-occurring Ripples (Globally) ---")
        # ... (Logic remains the same, uses absolute original fs timestamps) ...
        cooccurrence_results = {}
        window_samples = int(COOCCURRENCE_WINDOW_MS * fs_orig / 1000) # Use original fs for window size in samples
        ref_region = 'CA2'
        target_check_regions = ['CA1', 'CA3']
        if ref_region in averaged_timestamps:
            ref_timestamps = averaged_timestamps[ref_region]
            print(f"  Using {len(ref_timestamps)} {ref_region} timestamps as reference.")
            target_site_indices = {}
            for target_reg in target_check_regions:
                 sites_in_region = [sites[target_reg] for shank, sites in reference_sites.items() if target_reg in sites]
                 if sites_in_region: target_site_indices[target_reg] = sites_in_region[0]
            cooccurrence_results[ref_region] = {}
            for target_reg, target_ch_idx in target_site_indices.items():
                 target_event_times = []
                 if target_reg == 'CA1':
                     if target_ch_idx in ca1_spwr_events_global: target_event_times = np.array([ev['peak_sample'] for ev in ca1_spwr_events_global[target_ch_idx]])
                 elif target_reg == 'CA3':
                     if target_ch_idx in all_ripple_events_global: target_event_times = np.array([ev['peak_sample'] for ev in all_ripple_events_global[target_ch_idx]])
                 if len(target_event_times) == 0:
                     cooccurrence_results[ref_region][target_reg] = {'count': 0, 'details': []}; continue
                 cooccur_count = 0; cooccur_details = []
                 target_event_times.sort()
                 for ref_time in ref_timestamps:
                     lower_bound, upper_bound = ref_time - window_samples, ref_time + window_samples
                     start_idx = np.searchsorted(target_event_times, lower_bound, side='left')
                     end_idx = np.searchsorted(target_event_times, upper_bound, side='right')
                     indices_in_window = np.arange(start_idx, end_idx)
                     if len(indices_in_window) > 0:
                         cooccur_count += 1
                         cooccur_details.append({'ref_time_sample': ref_time, 'target_event_times': target_event_times[indices_in_window].tolist()})
                 print(f"    Found {cooccur_count} co-occurrences between {ref_region} (ref) and {target_reg} (target site {target_ch_idx}).")
                 cooccurrence_results[ref_region][target_reg] = {'count': cooccur_count, 'details': cooccur_details}


        # 10. Spectrogram Calculation and Averaging (Parallel)
        print(f"\n--- 10. Calculating and Averaging Spectrograms (Parallel using {NUM_CORES} cores) ---")
        averaged_spectrograms = {}
        for region, timestamps_orig_fs in averaged_timestamps.items():
            regional_ref_sites_indices = [sites[region] for shank, sites in reference_sites.items() if region in sites]
            if not regional_ref_sites_indices: continue

            spec_ch_idx = regional_ref_sites_indices[0]
            print(f"  Calculating spectrograms for region {region} using reference site {spec_ch_idx} LFP...")
            print(f"  Processing {len(timestamps_orig_fs)} timestamps for {region}...")

            # Pass original fs timestamps to worker, worker handles downsampling
            spec_tasks = [(ts, spec_ch_idx, lfp_filepath, n_samples_orig, n_channels, fs_orig)
                          for ts in timestamps_orig_fs]

            region_spectrograms_results = Parallel(n_jobs=NUM_CORES, backend="loky")(
                delayed(_calculate_spectrogram_worker)(*task) for task in spec_tasks
            )

            valid_spectrograms = [spec for spec in region_spectrograms_results if spec is not None and not np.isnan(spec).all()]

            if valid_spectrograms:
                try:
                    avg_spec = np.nanmean(np.stack(valid_spectrograms, axis=0), axis=0)
                    averaged_spectrograms[region] = avg_spec
                    print(f"  Averaged {len(valid_spectrograms)}/{len(timestamps_orig_fs)} spectrograms for region {region}.")
                except MemoryError: warnings.warn(f"MemoryError stacking spectrograms for region {region}. Cannot average.")
                except Exception as e_avg: warnings.warn(f"Error averaging spectrograms for region {region}: {e_avg}")
            else:
                print(f"  No valid spectrograms generated for region {region}.")
            del region_spectrograms_results, valid_spectrograms
            gc.collect()


        # 11. Save Results
        print("\n--- 11. Saving Results ---")
        # ... (Saving logic remains the same, data is already referenced to original fs) ...
        np.save(output_path / f'{output_filename_base}_ripple_references.npy', reference_sites, allow_pickle=True)
        np.save(output_path / f'{output_filename_base}_ripple_timestamps.npy', averaged_timestamps, allow_pickle=True)
        np.save(output_path / f'{output_filename_base}_ripple_cooccurrence.npy', cooccurrence_results, allow_pickle=True)
        np.save(output_path / f'{output_filename_base}_ripple_avg_spectrograms.npy', averaged_spectrograms, allow_pickle=True)
        np.save(output_path / f'{output_filename_base}_ripple_events_global.npy', all_ripple_events_global, allow_pickle=True)
        np.save(output_path / f'{output_filename_base}_spw_events_global.npy', all_spw_events_global, allow_pickle=True)
        np.save(output_path / f'{output_filename_base}_ca1_spwr_events_global.npy', ca1_spwr_events_global, allow_pickle=True)
        np.save(output_path / f'{output_filename_base}_ripple_events_by_epoch.npy', all_ripple_events_by_epoch, allow_pickle=True)
        np.save(output_path / f'{output_filename_base}_spw_events_by_epoch.npy', all_spw_events_by_epoch, allow_pickle=True)
        region_channels_df.to_csv(output_path / f'{output_filename_base}_ripple_analyzed_channels.csv', index=False)
        print(f"Results saved to {output_path} with prefix '{output_filename_base}'")


    except Exception as e:
        print(f"\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"ERROR processing file: {lfp_filepath.name}")
        print(f"Error details: {e}")
        traceback.print_exc()
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")

    finally:
        # --- Clean up ---
        print("\n--- Cleaning up memory ---")
        # Memmap is handled within workers or main function scope now
        # Explicitly delete large data structures
        vars_to_del = [
             'all_ripple_events_by_epoch', 'all_spw_events_by_epoch',
             'all_ripple_events_global', 'all_spw_events_global',
             'ca1_spwr_events_global', 'reference_sites', 'averaged_timestamps',
             'cooccurrence_results', 'averaged_spectrograms', 'baseline_stats_per_channel',
             'sleep_state_data', 'epoch_boundaries_sec', 'non_rem_periods_sec',
             'non_rem_periods_samples_orig', 'epoch_boundaries_samples_orig', 'channel_df',
             'region_channels_df', 'channel_region_map',
             'results', 'epoch_results', 'region_spectrograms_results', 'valid_spectrograms', 'avg_spec'
        ]
        deleted_count = 0
        local_vars = locals()
        for var_name in vars_to_del:
            if var_name in local_vars and local_vars[var_name] is not None:
                try: local_vars[var_name] = None; deleted_count += 1
                except Exception: pass
        collected = gc.collect()
        print(f"Attempted clear on {deleted_count} vars. Garbage collected: {collected} objects.")
        print("--- Analysis Complete for this file ---")


# --- Script Execution ---
if __name__ == "__main__":

    # --- Use Tkinter to select files ---
    root = Tk()
    root.withdraw() # Hide the main tkinter window
    root.attributes("-topmost", True) # Keep dialogs on top

    print("Please select the LFP binary file (*.lf.bin)...")
    lfp_file_str = filedialog.askopenfilename(
        title="Select LFP Binary File (*.lf.bin)",
        filetypes=[("LFP binary files", "*.lf.bin"), ("All files", "*.*")]
    )
    if not lfp_file_str: print("No LFP file selected. Exiting."); exit()
    LFP_FILE = Path(lfp_file_str)

    print("Please select the corresponding Meta file (*.meta)...")
    suggested_meta_name = LFP_FILE.stem + ".meta"
    meta_file_str = filedialog.askopenfilename(
        title="Select Meta File (*.meta)", initialdir=LFP_FILE.parent, initialfile=suggested_meta_name,
        filetypes=[("Meta files", "*.meta"), ("All files", "*.*")]
    )
    if not meta_file_str: print("No Meta file selected. Exiting."); exit()
    META_FILE = Path(meta_file_str)

    print("Please select the Channel Info CSV file...")
    channel_info_file_str = filedialog.askopenfilename(
        title="Select Channel Info CSV File", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    if not channel_info_file_str: print("No Channel Info file selected. Exiting."); exit()
    CHANNEL_INFO_FILE = Path(channel_info_file_str)

    print("Please select the Sleep State file (*_sleep_states*.npy) (Optional)...")
    sleep_state_file_str = filedialog.askopenfilename(
        title="Select Sleep State File (Optional - Cancel if none)", filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")]
    )
    SLEEP_STATE_FILE = Path(sleep_state_file_str) if sleep_state_file_str else None
    if SLEEP_STATE_FILE: print(f"Selected Sleep State file: {SLEEP_STATE_FILE.name}")
    else: print("No Sleep State file selected (Optional).")

    print("Please select the Epoch Boundaries file (*_epoch_boundaries*.npy) (Optional)...")
    epoch_boundaries_file_str = filedialog.askopenfilename(
        title="Select Epoch Boundaries File (Optional - Cancel if none)", filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")]
    )
    EPOCH_BOUNDARIES_FILE = Path(epoch_boundaries_file_str) if epoch_boundaries_file_str else None
    if EPOCH_BOUNDARIES_FILE: print(f"Selected Epoch Boundaries file: {EPOCH_BOUNDARIES_FILE.name}")
    else: print("No Epoch Boundaries file selected (Optional).")

    print("Please select the Output directory...")
    output_dir_str = filedialog.askdirectory(title="Select Output Directory")
    if not output_dir_str: print("No output directory selected. Exiting."); exit()
    OUTPUT_DIRECTORY = Path(output_dir_str)

    root.destroy() # Close the hidden tkinter root window

    # --- Validation of selected paths ---
    if not LFP_FILE.is_file(): print(f"Error: Selected LFP file not found: {LFP_FILE}"); exit()
    if not META_FILE.is_file(): print(f"Error: Selected Meta file not found: {META_FILE}"); exit()
    if not CHANNEL_INFO_FILE.is_file(): print(f"Error: Selected Channel Info file not found: {CHANNEL_INFO_FILE}"); exit()
    if SLEEP_STATE_FILE and not SLEEP_STATE_FILE.is_file(): print(f"Error: Selected Sleep State file not found: {SLEEP_STATE_FILE}"); exit()
    if EPOCH_BOUNDARIES_FILE and not EPOCH_BOUNDARIES_FILE.is_file(): print(f"Error: Selected Epoch Boundaries file not found: {EPOCH_BOUNDARIES_FILE}"); exit()
    try: OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True); print(f"Using output directory: {OUTPUT_DIRECTORY}")
    except Exception as e: print(f"Error creating output directory {OUTPUT_DIRECTORY}: {e}"); exit()

    # --- Run Analysis for the selected files ---
    print(f"\n{'='*50}\nStarting processing for: {LFP_FILE.name}\n{'='*50}")
    run_ripple_analysis(
        lfp_filepath=LFP_FILE,
        meta_filepath=META_FILE,
        channel_info_filepath=CHANNEL_INFO_FILE,
        sleep_state_filepath=SLEEP_STATE_FILE, # Pass Path object or None
        epoch_boundaries_filepath=EPOCH_BOUNDARIES_FILE, # Pass Path object or None
        output_dir=OUTPUT_DIRECTORY
    )

    print(f"\n{'='*50}\nProcessing complete for the selected file.\n{'='*50}")

