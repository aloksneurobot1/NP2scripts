# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 17:27:37 2025

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

# --- Data Loading Functions (Adapted from Example) ---
def load_lfp_data_memmap(file_path, meta_path, data_type='int16'):
    """
    Loads LFP data using memory-mapping, based on example script.
    Args:
        file_path (Path): Path to the binary LFP file (*.lf.bin).
        meta_path (Path): Path to the corresponding metadata file (*.meta).
        data_type (str): Data type of the samples.
    Returns:
        tuple: (numpy.memmap: Memory-mapped data, float: sampling rate) or (None, None) on error.
    """
    sampling_rate = None
    data = None
    print(f"Attempting to load LFP data from: {file_path}")
    print(f"Using metadata from: {meta_path}")
    try:
        # Read metadata to get the number of channels and sampling rate
        meta = readMeta(meta_path) # Use imported or placeholder readMeta
        num_channels = int(meta['nSavedChans'])

        # Get precise sampling rate
        if 'imSampRate' in meta:
            sampling_rate = float(meta['imSampRate'])
        # elif 'niSampRate' in meta:
        #     sampling_rate = float(meta['niSampRate'])
        else:
            print(f"Error: Sampling rate key ('imSampRate' or 'niSampRate') not found in {meta_path}.")
            return None, None

        if sampling_rate <= 0:
            print(f"Error: Invalid sampling rate ({sampling_rate}) found in {meta_path}.")
            return None, None

        print(f"Metadata: {num_channels} saved channels. Sampling rate: {sampling_rate:.6f} Hz")

        # Calculate the shape for memmap
        file_size = file_path.stat().st_size
        item_size = np.dtype(data_type).itemsize
        if num_channels <= 0 or item_size <= 0:
             print(f"Error: Invalid number of channels ({num_channels}) or item size ({item_size}). Cannot calculate shape.")
             return None, sampling_rate # Return fs even if data fails

        # Ensure file size is a multiple of (num_channels * item_size)
        expected_total_bytes = file_size - (file_size % (num_channels * item_size))
        if file_size != expected_total_bytes:
            print(f"Warning: File size {file_size} is not an exact multiple of {num_channels} channels * {item_size} bytes/sample.")
            print(f" Potential extra bytes at end of file: {file_size % (num_channels * item_size)}. Processing based on truncated size.")

        num_samples = expected_total_bytes // (num_channels * item_size)

        if num_samples <= 0:
            print("Error: Calculated number of samples is zero or negative. Check file size and metadata.")
            return None, sampling_rate

        # Note: SGLX data is channels x time, but memmap needs samples x channels
        # We will handle the transpose later if needed, memmap reads sequentially.
        # The shape should reflect how data is stored on disk (interleaved).
        # Example script uses (samples, channels) shape for memmap.
        shape = (num_samples, num_channels)
        print(f"Calculated samples: {num_samples}")
        print(f"Expected data shape for memmap: {shape}")

        # Memory-map the binary file
        data = np.memmap(file_path, dtype=data_type, mode='r', shape=shape, offset=0)
        print(f"Successfully memory-mapped file: {file_path}")
        # Return data as (samples, channels) and fs
        return data, sampling_rate

    except FileNotFoundError:
        print(f"Error: File not found - {file_path} or {meta_path}")
        return None, None
    except KeyError as e:
        print(f"Error: Metadata key missing in {meta_path} - {e}")
        return None, sampling_rate # Return fs if meta loaded but key missing
    except ValueError as e:
         print(f"Error: Problem with shape or dtype during memmap for {file_path} - {e}")
         if data is not None and hasattr(data, '_mmap'):
             try: data._mmap.close()
             except Exception: pass
         return None, sampling_rate
    except Exception as e:
        print(f"An unexpected error occurred in load_lfp_data_memmap for {file_path}: {e}")
        traceback.print_exc()
        if data is not None and hasattr(data, '_mmap'):
            try: data._mmap.close()
            except Exception: pass
        return None, None

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
        print("Error in apply_fir_filter: Input data is None or empty.")
        return None

    data_len = data.shape[-1] # Assume time is the last axis
    if numtaps >= data_len:
        print(f"Warning in apply_fir_filter: numtaps ({numtaps}) >= data length ({data_len}). Reducing numtaps.")
        numtaps = data_len - 1 if data_len > 1 else 1
        if numtaps % 2 == 0: numtaps -= 1 # Ensure odd
        if numtaps < 3:
             print(f"Error in apply_fir_filter: Data length ({data_len}) too short for filtering. Returning original data.")
             return data # Or return None? Returning original might be safer downstream.

    try:
        if lowcut is None and highcut is None:
             print("Warning in apply_fir_filter: Both lowcut and highcut are None. Returning original data.")
             return data
        elif lowcut is None: # Lowpass
            b = signal.firwin(numtaps, highcut, fs=fs, pass_zero=True, window='hamming')
        elif highcut is None: # Highpass
             b = signal.firwin(numtaps, lowcut, fs=fs, pass_zero=False, window='hamming')
        else: # Bandpass
            if lowcut >= highcut:
                 print(f"Error in apply_fir_filter: lowcut ({lowcut}) >= highcut ({highcut}).")
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
        traceback.print_exc()
        return None


def calculate_instantaneous_power(data):
    """Calculates instantaneous power using Hilbert transform."""
    if data is None or data.size == 0:
        print("Error in calculate_instantaneous_power: Input data is None or empty.")
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
        traceback.print_exc()
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
        warnings.warn("Calculating baseline stats over the entire signal duration (no non-REM periods provided).")
        baseline_power_segments = [power_signal]
    else:
        print(f"Calculating baseline stats using {len(non_rem_periods_samples)} non-REM periods.")
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
    # If total size is very large, consider calculating stats iteratively
    try:
        # Check total size before concatenating
        total_baseline_samples = sum(seg.size for seg in baseline_power_segments)
        if total_baseline_samples == 0:
             raise ValueError("Cannot compute baseline statistics: Concatenated baseline power is empty.")

        # Heuristic: If total size exceeds ~1GB, process iteratively (more complex)
        # For now, proceed with concatenation, assuming sufficient memory
        if total_baseline_samples * power_signal.itemsize > 1e9:
             warnings.warn("Large baseline data size (>1GB). Concatenation might be slow or fail. Consider iterative stats.")

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
        warnings.warn("Initial SD of baseline power is zero. Clipping might not be effective.")
        clipped_power = full_baseline_power
    else:
        # Clip at 4 SD
        clip_threshold = initial_mean + 4 * initial_sd
        clipped_power = np.clip(full_baseline_power, a_min=None, a_max=clip_threshold)

    # Rectify (already positive due to power calculation, but included for completeness)
    rectified_power = np.abs(clipped_power) # Should not change anything

    # Low-pass filter at 55 Hz
    numtaps_lp = 101 # Adjust as needed
    if numtaps_lp >= rectified_power.size:
         warnings.warn(f"Baseline data too short ({rectified_power.size}) for LP filter numtaps ({numtaps_lp}). Skipping LP filter.")
         processed_baseline_power = rectified_power
    else:
         try:
             b_lp = signal.firwin(numtaps_lp, RIPPLE_POWER_LP_CUTOFF, fs=fs, pass_zero=True, window='hamming')
             processed_baseline_power = signal.filtfilt(b_lp, 1, rectified_power)
         except Exception as e_lp:
              warnings.warn(f"Error applying baseline LP filter: {e_lp}. Using rectified power directly.")
              processed_baseline_power = rectified_power

    # Final mean and SD for detection thresholds
    baseline_mean = np.mean(processed_baseline_power)
    baseline_sd = np.std(processed_baseline_power)

    if baseline_sd == 0:
         warnings.warn("Final baseline SD is zero. Detection thresholds might be problematic.")

    print(f"Baseline calculated: Mean={baseline_mean:.4f}, SD={baseline_sd:.4f} (based on {total_baseline_samples} samples)")
    return baseline_mean, baseline_sd


def detect_events(signal_data, fs, threshold_high, threshold_low, min_duration_ms, merge_gap_ms, event_type="Event"):
    """
    Detects events based on signal thresholds, duration, and merging gaps.
    Args:
        signal_data (np.ndarray): The signal to detect events in (e.g., power, abs(LFP)).
        fs (float): Sampling frequency.
        threshold_high (float): Threshold for initial event detection.
        threshold_low (float): Threshold for expanding event boundaries.
        min_duration_ms (float): Minimum duration for an event to be kept.
        merge_gap_ms (float): Maximum gap between events to merge them.
        event_type (str): Name for logging messages (e.g., "Ripple", "SPW").
    Returns:
        tuple: (np.array: start indices, np.array: end indices) of detected events.
    """
    if signal_data is None or signal_data.size == 0:
        print(f"Error in detect_events: Input signal_data for {event_type} is invalid.")
        return np.array([], dtype=int), np.array([], dtype=int)

    min_duration_samples = int(min_duration_ms * fs / 1000)
    merge_gap_samples = int(merge_gap_ms * fs / 1000)

    # Find where signal exceeds the high threshold
    try:
        with np.errstate(invalid='ignore'): # Ignore potential NaNs temporarily
            above_high_threshold = signal_data > threshold_high
        if np.any(np.isnan(signal_data)):
            warnings.warn(f"NaNs found in input signal for {event_type} detection. Treating NaNs as below threshold.")
            above_high_threshold[np.isnan(signal_data)] = False

        # Find transitions
        diff_high = np.diff(above_high_threshold.astype(np.int8)) # Use int8 for diff
        starts_high = np.where(diff_high == 1)[0] + 1
        ends_high = np.where(diff_high == -1)[0]

        # Handle edge cases where signal starts or ends above threshold
        if len(above_high_threshold) > 0: # Add check for empty array
            if above_high_threshold[0]:
                starts_high = np.insert(starts_high, 0, 0)
            if above_high_threshold[-1]:
                ends_high = np.append(ends_high, len(signal_data) - 1)
        else: # Handle empty input case explicitly
             return np.array([], dtype=int), np.array([], dtype=int)


    except Exception as e_thresh:
        print(f"Error during thresholding for {event_type}: {e_thresh}")
        return np.array([], dtype=int), np.array([], dtype=int)


    if len(starts_high) == 0 or len(ends_high) == 0:
        # print(f"No {event_type} crossings found above high threshold.")
        return np.array([], dtype=int), np.array([], dtype=int) # No events detected

    # Ensure starts and ends align correctly and are paired
    valid_pairs = []
    current_start_idx = 0
    current_end_idx = 0
    while current_start_idx < len(starts_high) and current_end_idx < len(ends_high):
        start_val = starts_high[current_start_idx]
        end_val = ends_high[current_end_idx]

        if start_val <= end_val: # Valid pair
            valid_pairs.append((start_val, end_val))
            current_start_idx += 1
            current_end_idx += 1
        else: # End is before start, advance end index
            current_end_idx += 1

    if not valid_pairs:
        # print(f"No valid start/end pairs found for {event_type} after alignment.")
        return np.array([], dtype=int), np.array([], dtype=int)

    starts_high, ends_high = zip(*valid_pairs)
    starts_high = np.array(starts_high)
    ends_high = np.array(ends_high)

    # Expand events until signal falls below the low threshold
    expanded_starts = []
    expanded_ends = []
    try:
        with np.errstate(invalid='ignore'): # Ignore potential NaNs temporarily
             below_low_mask = signal_data < threshold_low
        if np.any(np.isnan(signal_data)):
             below_low_mask[np.isnan(signal_data)] = True # Treat NaN as below threshold for expansion stop

        for start, end in zip(starts_high, ends_high):
            # Expand start backwards
            current_start = start
            while current_start > 0 and not below_low_mask[current_start - 1]:
                current_start -= 1

            # Expand end forwards
            current_end = end
            while current_end < len(signal_data) - 1 and not below_low_mask[current_end + 1]:
                current_end += 1

            expanded_starts.append(current_start)
            expanded_ends.append(current_end)
    except Exception as e_expand:
         print(f"Error during event expansion for {event_type}: {e_expand}")
         # Fallback: use non-expanded events? Or return empty? Returning empty is safer.
         return np.array([], dtype=int), np.array([], dtype=int)


    if not expanded_starts:
        # print(f"No {event_type} events remained after expansion step.")
        return np.array([], dtype=int), np.array([], dtype=int)

    # Merge adjacent events
    merged_starts = [expanded_starts[0]]
    merged_ends = [expanded_ends[0]]

    for i in range(1, len(expanded_starts)):
        gap = expanded_starts[i] - merged_ends[-1] - 1 # Gap is between end of prev and start of next
        if gap < merge_gap_samples: # Use < as gap=0 means adjacent samples
            # Merge: Update the end time of the last merged event
            merged_ends[-1] = max(merged_ends[-1], expanded_ends[i])
        else:
            # No merge: Add the new event
            merged_starts.append(expanded_starts[i])
            merged_ends.append(expanded_ends[i])

    # Filter by minimum duration
    final_starts = []
    final_ends = []
    for start, end in zip(merged_starts, merged_ends):
        # Duration is inclusive: (end - start + 1) samples
        duration_samples = end - start + 1
        if duration_samples >= min_duration_samples:
            final_starts.append(start)
            final_ends.append(end)
        # else:
            # print(f"Discarding {event_type} event: duration {duration_samples} < min {min_duration_samples}")


    return np.array(final_starts, dtype=int), np.array(final_ends, dtype=int)

def find_event_features(lfp_segment, power_segment):
    """
    Finds peak power index and nearest trough index within an event segment.
    Args:
        lfp_segment (np.ndarray): Filtered LFP data for the event window.
        power_segment (np.ndarray): Instantaneous power for the event window.
    Returns:
        tuple: (peak_power_idx_rel, trough_idx_rel) relative to segment start, or (-1, -1) on error.
    """
    if power_segment is None or lfp_segment is None or power_segment.size == 0 or lfp_segment.size == 0:
        return -1, -1 # Indicate failure
    if len(lfp_segment) != len(power_segment):
         warnings.warn(f"LFP segment length ({len(lfp_segment)}) != Power segment length ({len(power_segment)}) in find_event_features.")
         # Try to proceed if lengths are close? Or return error? Error is safer.
         return -1,-1

    try:
        peak_power_idx_rel = np.nanargmax(power_segment) # Use nanargmax

        # Find troughs (negative peaks) in the LFP segment using find_peaks
        # Ensure LFP segment is float for find_peaks
        if not np.issubdtype(lfp_segment.dtype, np.floating):
             lfp_segment_float = lfp_segment.astype(np.float64)
        else: lfp_segment_float = lfp_segment

        troughs_rel, _ = signal.find_peaks(-lfp_segment_float)

        if len(troughs_rel) == 0:
            # Fallback: use the minimum value index if no peaks found
            trough_idx_rel = np.nanargmin(lfp_segment_float)
            # warnings.warn(f"No troughs found in event segment of length {len(lfp_segment)}. Using minimum value index {trough_idx_rel}.")
        else:
            # Find the trough closest to the peak power index
            trough_idx_rel = troughs_rel[np.nanargmin(np.abs(troughs_rel - peak_power_idx_rel))]

        return peak_power_idx_rel, trough_idx_rel

    except Exception as e:
        warnings.warn(f"Error finding event features: {e}")
        return -1, -1


def calculate_cwt_spectrogram(data, fs, freqs, timestamp_samples, window_samples):
    """Calculates wavelet spectrogram around a timestamp using PyWavelets."""
    if data is None or data.size == 0: return np.full((len(freqs), window_samples), np.nan)

    start_sample = int(timestamp_samples - window_samples // 2)
    end_sample = int(start_sample + window_samples)

    # --- Boundary checks ---
    pad_left = 0
    pad_right = 0
    if start_sample < 0:
        pad_left = -start_sample
        start_sample = 0
    if end_sample > len(data):
        pad_right = end_sample - len(data)
        end_sample = len(data)

    segment = data[start_sample:end_sample]

    if segment.size == 0:
        return np.full((len(freqs), window_samples), np.nan)

    # --- Padding ---
    if pad_left > 0 or pad_right > 0:
        try:
            # Use reflect padding for potentially better results at edges than edge padding
            segment = np.pad(segment, (pad_left, pad_right), mode='reflect')
        except ValueError: # Fallback if reflect fails (e.g., segment too small)
             segment = np.pad(segment, (pad_left, pad_right), mode='edge')


    # --- CWT Calculation ---
    if len(segment) != window_samples:
         # This shouldn't happen after padding, but check just in case
         warnings.warn(f"Segment length {len(segment)} != window_samples {window_samples} after padding. Resizing might occur.")
         # Could resize here if necessary, but pywt might handle it.

    try:
        # Choose a wavelet (e.g., 'cmorB-C' like Complex Morlet)
        wavelet = f'cmor1.5-1.0' # Example: B=1.5, C=1.0

        # Calculate scales corresponding to desired frequencies
        scales = pywt.frequency2scale(wavelet, freqs / fs)

        # Perform Continuous Wavelet Transform (CWT)
        coeffs, _ = pywt.cwt(segment.astype(np.float64), scales, wavelet, sampling_period=1.0/fs) # Ensure float64

        # Calculate power (magnitude squared)
        power_spectrogram = np.abs(coeffs)**2

        # Ensure output shape matches expected window size
        if power_spectrogram.shape[1] != window_samples:
             warnings.warn(f"Spectrogram width ({power_spectrogram.shape[1]}) doesn't match window ({window_samples}). Resizing/interpolating.")
             # Simple linear interpolation to fix size
             from scipy.interpolate import interp1d
             x_old = np.linspace(0, 1, power_spectrogram.shape[1])
             x_new = np.linspace(0, 1, window_samples)
             interp_func = interp1d(x_old, power_spectrogram, axis=1, kind='linear', fill_value='extrapolate')
             power_spectrogram = interp_func(x_new)


        return power_spectrogram # Shape: (n_freqs, n_timepoints_in_window)

    except MemoryError:
         warnings.warn(f"MemoryError calculating CWT spectrogram for timestamp {timestamp_samples}.")
         return np.full((len(freqs), window_samples), np.nan)
    except Exception as e:
         warnings.warn(f"Error calculating CWT spectrogram for timestamp {timestamp_samples}: {e}")
         return np.full((len(freqs), window_samples), np.nan)


# --- Main Analysis Function ---

def run_ripple_analysis(lfp_filepath, meta_filepath, channel_info_filepath,
                        sleep_state_filepath, epoch_boundaries_filepath, # Updated inputs
                        output_dir):
    """Main function to run the ripple and SPW detection analysis."""

    # Ensure input paths are Path objects
    lfp_filepath = Path(lfp_filepath)
    meta_filepath = Path(meta_filepath)
    channel_info_filepath = Path(channel_info_filepath)
    # Handle optional files which might be None
    sleep_state_filepath = Path(sleep_state_filepath) if sleep_state_filepath else None
    epoch_boundaries_filepath = Path(epoch_boundaries_filepath) if epoch_boundaries_filepath else None
    output_dir = Path(output_dir)


    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_path}")

    # --- Derive base filename ---
    lfp_filename = lfp_filepath.name
    match = re.match(r"^(.*?)(\.(imec|nidq)\d?)?\.lf\.bin$", lfp_filename)
    if match:
        output_filename_base = match.group(1)
        print(f"Derived base filename: {output_filename_base}")
    else:
        output_filename_base = lfp_filename.replace('.lf.bin', '')
        print(f"Warning: Using fallback base filename: {output_filename_base}")

    # Initialize results storage
    all_ripple_events_by_epoch = {} # Dict: epoch_idx -> {channel_idx -> list of ripple events}
    all_spw_events_by_epoch = {}    # Dict: epoch_idx -> {channel_idx -> list of SPW events}
    # Global storage for combined results across epochs
    all_ripple_events_global = {} # Dict: channel_idx -> list of ripple events (absolute indices)
    all_spw_events_global = {}    # Dict: channel_idx -> list of SPW events (absolute indices)
    ca1_spwr_events_global = {}   # Dict: channel_idx -> list of SPW-coincident ripples (absolute indices)
    reference_sites = {}          # Dict: shank_idx -> {region: channel_idx}
    averaged_timestamps = {}      # Dict: region -> np.array of timestamps (samples)
    cooccurrence_results = {}     # Dict: {ref_region: {target_region: counts/details}}
    averaged_spectrograms = {}    # Dict: region -> avg_spectrogram

    lfp_data_memmap = None # Define outside try block

    try:
        # 1. Load LFP Data using Memmap
        print("\n--- 1. Loading LFP Data ---")
        lfp_data_memmap, fs = load_lfp_data_memmap(lfp_filepath, meta_filepath)
        if lfp_data_memmap is None or fs is None:
            raise ValueError("Failed to load LFP data or sampling rate.")

        # LFP data shape is (n_samples, n_channels) from memmap
        n_samples, n_channels = lfp_data_memmap.shape
        duration_sec = n_samples / fs
        print(f"LFP data loaded: {n_channels} channels, {n_samples} samples ({duration_sec:.2f} seconds), Fs={fs} Hz")

        # 2. Load Channel Info
        print("\n--- 2. Loading Channel Info ---")
        channel_df = load_channel_info(channel_info_filepath)
        # Filter channel info for relevant regions
        target_regions = ['CA1', 'CA3', 'CA2']
        region_channels_df = channel_df[channel_df['acronym'].isin(target_regions)].copy()
        if region_channels_df.empty:
            raise ValueError(f"No channels found in target regions: {target_regions}")
        print(f"Found {len(region_channels_df)} channels in {target_regions}.")
        # Get list of channel indices to process
        channels_to_process = region_channels_df['global_channel_index'].astype(int).unique()


        # 3. Load Sleep State and Epoch Data
        print("\n--- 3. Loading Sleep State and Epoch Data ---")
        sleep_state_data, epoch_boundaries_sec, non_rem_periods_sec = load_sleep_and_epoch_data(
            sleep_state_filepath, epoch_boundaries_filepath, fs
        )
        # Convert times to sample indices
        non_rem_periods_samples = [(int(start * fs), int(end * fs)) for start, end in non_rem_periods_sec]
        epoch_boundaries_samples = [(int(start * fs), int(end * fs)) for start, end in epoch_boundaries_sec]

        if not epoch_boundaries_samples:
             warnings.warn("No epoch boundaries loaded or found. Processing entire recording as one epoch.")
             epoch_boundaries_samples = [(0, n_samples)] # Process full duration


        # 4. Calculate Global Baseline Statistics
        print("\n--- 4. Calculating Global Baseline Statistics ---")
        # Requires ripple power across the whole recording for one channel (or average?)
        # Let's calculate baseline per channel based on its non-REM periods across the whole recording.
        baseline_stats_per_channel = {} # Store {ch_idx: (mean, sd)}
        print("Calculating baseline stats for each relevant channel...")
        for ch_idx in channels_to_process:
             if ch_idx >= n_channels:
                 warnings.warn(f"Channel index {ch_idx} out of bounds ({n_channels} channels). Skipping baseline.")
                 continue
             print(f"  Calculating baseline for Channel {ch_idx}...")
             lfp_ch_full = lfp_data_memmap[:, ch_idx].astype(np.float64) # Load full channel data

             # Filter for ripples (whole channel)
             lfp_ch_ripple_filtered_full = apply_fir_filter(lfp_ch_full, RIPPLE_FILTER_LOWCUT, RIPPLE_FILTER_HIGHCUT, fs)
             if lfp_ch_ripple_filtered_full is None:
                  warnings.warn(f"Could not ripple-filter channel {ch_idx}. Skipping baseline.")
                  del lfp_ch_full; gc.collect()
                  continue

             # Calculate power (whole channel)
             ripple_power_full = calculate_instantaneous_power(lfp_ch_ripple_filtered_full)
             del lfp_ch_ripple_filtered_full # Free memory
             if ripple_power_full is None:
                  warnings.warn(f"Could not calculate power for channel {ch_idx}. Skipping baseline.")
                  del lfp_ch_full; gc.collect()
                  continue

             # Calculate baseline mean and SD using the specified method over non-REM periods
             try:
                 baseline_mean, baseline_sd = get_baseline_stats(ripple_power_full, fs, non_rem_periods_samples)
                 baseline_stats_per_channel[ch_idx] = (baseline_mean, baseline_sd)
                 print(f"  Channel {ch_idx}: Baseline Mean={baseline_mean:.4f}, SD={baseline_sd:.4f}")
             except ValueError as e:
                  warnings.warn(f"Could not calculate baseline for channel {ch_idx}: {e}. Skipping channel for event detection.")
             except MemoryError:
                  warnings.warn(f"MemoryError calculating baseline for channel {ch_idx}. Skipping channel.")

             del lfp_ch_full, ripple_power_full; gc.collect()


        # 5. Per-Epoch Event Detection
        print("\n--- 5. Detecting Ripples and Sharp Waves (per epoch) ---")
        for epoch_idx, (epoch_start_sample, epoch_end_sample) in enumerate(epoch_boundaries_samples):
            print(f"\nProcessing Epoch {epoch_idx} (Samples: {epoch_start_sample} - {epoch_end_sample})...")
            epoch_duration_samples = epoch_end_sample - epoch_start_sample
            if epoch_duration_samples <= 0:
                 print(f"Skipping epoch {epoch_idx} due to zero or negative duration.")
                 continue

            # Initialize storage for this epoch
            all_ripple_events_by_epoch[epoch_idx] = {}
            all_spw_events_by_epoch[epoch_idx] = {}

            for _, channel_row in region_channels_df.iterrows():
                ch_idx = int(channel_row['global_channel_index'])
                ch_region = channel_row['acronym']

                if ch_idx >= n_channels:
                    warnings.warn(f"Channel index {ch_idx} out of bounds for LFP data ({n_channels} channels). Skipping.")
                    continue
                if ch_idx not in baseline_stats_per_channel:
                     # print(f"Skipping channel {ch_idx} in epoch {epoch_idx} (no baseline stats available).")
                     continue # Skip if baseline failed

                print(f"  Processing Channel {ch_idx} (Region: {ch_region}) in Epoch {epoch_idx}...")

                # Load LFP data for this channel and epoch
                try:
                    lfp_ch_epoch = lfp_data_memmap[epoch_start_sample:epoch_end_sample, ch_idx].astype(np.float64)
                except MemoryError:
                     warnings.warn(f"MemoryError loading LFP segment for channel {ch_idx}, epoch {epoch_idx}. Skipping.")
                     continue
                except Exception as e_load:
                     warnings.warn(f"Error loading LFP segment for channel {ch_idx}, epoch {epoch_idx}: {e_load}. Skipping.")
                     continue

                # --- Ripple Detection ---
                lfp_ch_ripple_filtered = apply_fir_filter(lfp_ch_epoch, RIPPLE_FILTER_LOWCUT, RIPPLE_FILTER_HIGHCUT, fs)
                if lfp_ch_ripple_filtered is None: continue # Skip channel if filtering failed

                ripple_power = calculate_instantaneous_power(lfp_ch_ripple_filtered)
                if ripple_power is None: continue # Skip channel if power calc failed

                # Use GLOBAL baseline stats for detection thresholds
                baseline_mean, baseline_sd = baseline_stats_per_channel[ch_idx]
                if baseline_sd == 0: # Avoid division by zero or meaningless thresholds
                     warnings.warn(f"Baseline SD is zero for channel {ch_idx}. Skipping event detection.")
                     continue

                detection_threshold = baseline_mean + RIPPLE_DETECTION_SD_THRESHOLD * baseline_sd
                expansion_threshold = baseline_mean + RIPPLE_EXPANSION_SD_THRESHOLD * baseline_sd

                # Detect ripple events within the epoch segment
                ripple_starts_rel, ripple_ends_rel = detect_events(
                    ripple_power, fs,
                    threshold_high=detection_threshold,
                    threshold_low=expansion_threshold,
                    min_duration_ms=RIPPLE_MIN_DURATION_MS,
                    merge_gap_ms=RIPPLE_MERGE_GAP_MS,
                    event_type="Ripple"
                )
                # print(f"    Found {len(ripple_starts_rel)} candidate ripple events relative to epoch start.")

                # Store ripple event details with ABSOLUTE sample indices
                epoch_channel_ripples = []
                for start_rel, end_rel in zip(ripple_starts_rel, ripple_ends_rel):
                    # Ensure relative indices are within the loaded segment bounds
                    if start_rel < 0 or end_rel >= len(lfp_ch_ripple_filtered):
                         warnings.warn(f"Invalid relative ripple indices ({start_rel}, {end_rel}) vs segment length {len(lfp_ch_ripple_filtered)}. Skipping event.")
                         continue

                    lfp_segment = lfp_ch_ripple_filtered[start_rel:end_rel+1]
                    power_segment = ripple_power[start_rel:end_rel+1]
                    peak_power_idx_rel, trough_idx_rel = find_event_features(lfp_segment, power_segment)

                    if peak_power_idx_rel != -1 and trough_idx_rel != -1:
                         # Convert relative indices within the event segment to absolute indices
                         peak_sample_abs = epoch_start_sample + start_rel + peak_power_idx_rel
                         trough_sample_abs = epoch_start_sample + start_rel + trough_idx_rel
                         start_sample_abs = epoch_start_sample + start_rel
                         end_sample_abs = epoch_start_sample + end_rel
                         peak_power_val = power_segment[peak_power_idx_rel]

                         event_details = {
                             'start_sample': start_sample_abs,
                             'end_sample': end_sample_abs,
                             'peak_sample': peak_sample_abs,
                             'trough_sample': trough_sample_abs,
                             'peak_power': peak_power_val,
                             'duration_ms': (end_rel - start_rel + 1) / fs * 1000, # Use relative end/start for duration
                             'epoch_idx': epoch_idx
                         }
                         epoch_channel_ripples.append(event_details)
                         # Append to global list as well
                         all_ripple_events_global.setdefault(ch_idx, []).append(event_details)

                if epoch_channel_ripples:
                     all_ripple_events_by_epoch[epoch_idx][ch_idx] = epoch_channel_ripples
                     # print(f"    Stored {len(epoch_channel_ripples)} valid ripple events for channel {ch_idx}, epoch {epoch_idx}.")


                # --- SPW Detection (only for CA1 channels) ---
                if ch_region == 'CA1':
                    lfp_ch_spw_filtered = apply_fir_filter(lfp_ch_epoch, SPW_FILTER_LOWCUT, SPW_FILTER_HIGHCUT, fs)
                    if lfp_ch_spw_filtered is None: continue # Skip if filtering failed

                    # Use z-scoring on the epoch's filtered signal for SPW detection threshold
                    spw_signal_mean = np.nanmean(lfp_ch_spw_filtered)
                    spw_signal_sd = np.nanstd(lfp_ch_spw_filtered)

                    if spw_signal_sd == 0 or not np.isfinite(spw_signal_sd):
                        # warnings.warn(f"SD of SPW-filtered signal is zero/NaN for channel {ch_idx}, epoch {epoch_idx}. Skipping SPW detection.")
                        continue

                    # Detect based on ABSOLUTE value of filtered LFP exceeding threshold
                    spw_detection_threshold = SPW_DETECTION_SD_THRESHOLD * spw_signal_sd # Threshold relative to mean=0 after z-score idea
                    spw_expansion_threshold = 1.0 * spw_signal_sd # Lower threshold for expansion (heuristic)

                    spw_starts_rel, spw_ends_rel = detect_events(
                         np.abs(lfp_ch_spw_filtered - spw_signal_mean), fs, # Detect deviations from mean
                         threshold_high=spw_detection_threshold,
                         threshold_low=spw_expansion_threshold,
                         min_duration_ms=SPW_MIN_DURATION_MS,
                         merge_gap_ms=1, # Minimal merge gap for SPWs
                         event_type="SPW"
                    )
                    # print(f"    Found {len(spw_starts_rel)} candidate SPW events relative to epoch start.")


                    # Filter SPW by max duration and store with ABSOLUTE indices
                    epoch_channel_spws = []
                    for start_rel, end_rel in zip(spw_starts_rel, spw_ends_rel):
                         # Ensure relative indices are within the loaded segment bounds
                         if start_rel < 0 or end_rel >= len(lfp_ch_spw_filtered):
                              warnings.warn(f"Invalid relative SPW indices ({start_rel}, {end_rel}) vs segment length {len(lfp_ch_spw_filtered)}. Skipping event.")
                              continue

                         duration_ms = (end_rel - start_rel + 1) / fs * 1000
                         if duration_ms <= SPW_MAX_DURATION_MS:
                              # Find SPW trough (minimum value in the SPW-filtered signal within the event)
                              spw_segment = lfp_ch_spw_filtered[start_rel:end_rel+1]
                              trough_idx_rel_in_segment = np.nanargmin(spw_segment)
                              trough_sample_abs = epoch_start_sample + start_rel + trough_idx_rel_in_segment
                              start_sample_abs = epoch_start_sample + start_rel
                              end_sample_abs = epoch_start_sample + end_rel

                              event_details = {
                                  'start_sample': start_sample_abs,
                                  'end_sample': end_sample_abs,
                                  'trough_sample': trough_sample_abs, # Main feature for SPW
                                  'duration_ms': duration_ms,
                                  'epoch_idx': epoch_idx
                              }
                              epoch_channel_spws.append(event_details)
                              # Append to global list as well
                              all_spw_events_global.setdefault(ch_idx, []).append(event_details)

                    if epoch_channel_spws:
                        all_spw_events_by_epoch[epoch_idx][ch_idx] = epoch_channel_spws
                        # print(f"    Stored {len(epoch_channel_spws)} valid SPW events for channel {ch_idx}, epoch {epoch_idx}.")

                # Clean up epoch-specific data for this channel
                del lfp_ch_epoch, lfp_ch_ripple_filtered, ripple_power
                if ch_region == 'CA1' and 'lfp_ch_spw_filtered' in locals(): del lfp_ch_spw_filtered
                gc.collect()


        # --- Post-Epoch Processing (using global event lists) ---

        # 6. SPW-R Coincidence Check (CA1 only - using global lists)
        print("\n--- 6. Checking SPW-Ripple Coincidence for CA1 (Globally) ---")
        ca1_channel_indices = region_channels_df[region_channels_df['acronym'] == 'CA1']['global_channel_index'].values
        for ch_idx in ca1_channel_indices:
             if ch_idx not in all_ripple_events_global: continue # Skip if no ripples on this CA1 channel

             ripples = all_ripple_events_global[ch_idx]
             if ch_idx not in all_spw_events_global or not all_spw_events_global[ch_idx]:
                 # ca1_spwr_events_global[ch_idx] = [] # No SPWs detected on this channel
                 continue

             spws = all_spw_events_global[ch_idx]
             coincident_ripples = []

             # Efficient check using sorted times (optional but good for many events)
             # spw_intervals = sorted([(spw['start_sample'], spw['end_sample'], spw['trough_sample']) for spw in spws])

             for ripple in ripples:
                 ripple_start = ripple['start_sample']
                 ripple_end = ripple['end_sample']
                 is_coincident = False
                 associated_spw_trough = None

                 # Simple overlap check
                 for spw in spws:
                     spw_start = spw['start_sample']
                     spw_end = spw['end_sample']
                     # Check for overlap: (StartA <= EndB) and (EndA >= StartB)
                     if ripple_start <= spw_end and ripple_end >= spw_start:
                         is_coincident = True
                         associated_spw_trough = spw['trough_sample']
                         break # Found one overlapping SPW

                 if is_coincident:
                     ripple['associated_spw_trough_sample'] = associated_spw_trough # Add trough info
                     coincident_ripples.append(ripple)

             if coincident_ripples:
                  ca1_spwr_events_global[ch_idx] = coincident_ripples
                  print(f"  Channel {ch_idx} (CA1): Found {len(coincident_ripples)} SPW-R events globally.")


        # 7. Determine Reference Sites per Shank (Globally)
        print("\n--- 7. Determining Reference Sites per Shank (Globally) ---")
        shanks = region_channels_df['shank_index'].unique()
        reference_sites = {} # Reset here, calculate globally

        for shank_idx in shanks:
            shank_channels_df = region_channels_df[region_channels_df['shank_index'] == shank_idx]
            ref_sites_for_shank = {}
            for region in target_regions:
                region_shank_channels = shank_channels_df[shank_channels_df['acronym'] == region]
                if region_shank_channels.empty:
                    continue

                max_ripple_metric = -1 # Use mean power or count? Desc says amplitude (power)
                ref_ch_idx = -1

                for _, ch_row in region_shank_channels.iterrows():
                    ch_idx = int(ch_row['global_channel_index'])

                    # Use appropriate GLOBAL event list based on region
                    events_to_consider = []
                    if region == 'CA1':
                         if ch_idx in ca1_spwr_events_global: # Use SPW-R events for CA1 ref site
                              events_to_consider = ca1_spwr_events_global[ch_idx]
                    else: # CA2, CA3
                         if ch_idx in all_ripple_events_global: # Use all ripples for CA2/CA3 ref site
                              events_to_consider = all_ripple_events_global[ch_idx]

                    if not events_to_consider: continue

                    # Calculate mean peak power for this channel's relevant events
                    peak_powers = [event['peak_power'] for event in events_to_consider if 'peak_power' in event and np.isfinite(event['peak_power'])]
                    if not peak_powers: continue
                    mean_power = np.mean(peak_powers)

                    if mean_power > max_ripple_metric:
                        max_ripple_metric = mean_power
                        ref_ch_idx = ch_idx

                if ref_ch_idx != -1:
                    ref_sites_for_shank[region] = ref_ch_idx
                    print(f"  Shank {shank_idx}, Region {region}: Reference site set to Channel {ref_ch_idx} (Max avg ripple power: {max_ripple_metric:.2f})")
                else:
                     print(f"  Shank {shank_idx}, Region {region}: No suitable reference site found (no valid events?).")

            if ref_sites_for_shank: # Only add shank if it has at least one reference site
                 reference_sites[shank_idx] = ref_sites_for_shank


        # 8. Generate Averaged Timestamps per Region (Globally)
        print("\n--- 8. Generating Averaged Timestamps per Region (Globally) ---")
        averaged_timestamps = {} # Reset here

        for region in target_regions:
            # Find all reference site channel indices for this region across all shanks
            regional_ref_sites_indices = [sites[region] for shank, sites in reference_sites.items() if region in sites]
            if not regional_ref_sites_indices:
                print(f"  No reference sites found for region {region}. Skipping timestamp generation.")
                continue

            all_regional_event_times = []
            event_type = "Unknown"
            if region == 'CA1':
                # Use SPW trough times from coincident SPW-Rs on reference sites
                event_type = "SPW Trough (SPW-R)"
                for ch_idx in regional_ref_sites_indices:
                    if ch_idx in ca1_spwr_events_global:
                        all_regional_event_times.extend([event['associated_spw_trough_sample'] for event in ca1_spwr_events_global[ch_idx] if 'associated_spw_trough_sample' in event])
            elif region == 'CA2':
                # Use ripple peak power times from reference sites
                 event_type = "Ripple Peak"
                 for ch_idx in regional_ref_sites_indices:
                     if ch_idx in all_ripple_events_global:
                          all_regional_event_times.extend([event['peak_sample'] for event in all_ripple_events_global[ch_idx]])
            else: # CA3 (Assume ripple peak times, similar to CA2)
                event_type = "Ripple Peak"
                for ch_idx in regional_ref_sites_indices:
                    if ch_idx in all_ripple_events_global:
                        all_regional_event_times.extend([event['peak_sample'] for event in all_ripple_events_global[ch_idx]])


            if not all_regional_event_times:
                 print(f"  No {event_type} times found for region {region} across reference sites.")
                 continue

            # Pool all unique event times from the reference sites in the region
            pooled_times = np.sort(np.unique(all_regional_event_times))
            averaged_timestamps[region] = pooled_times # Store the pooled times
            print(f"  Region {region}: Generated {len(pooled_times)} pooled {event_type} timestamps.")


        # 9. Co-occurrence Detection (Globally)
        print("\n--- 9. Detecting Co-occurring Ripples (Globally) ---")
        cooccurrence_results = {} # Reset here
        window_samples = int(COOCCURRENCE_WINDOW_MS * fs / 1000)

        # Example: Use CA2 ripple peaks as reference triggers
        ref_region = 'CA2'
        target_check_regions = ['CA1', 'CA3']

        if ref_region in averaged_timestamps:
            ref_timestamps = averaged_timestamps[ref_region]
            print(f"  Using {len(ref_timestamps)} {ref_region} timestamps as reference.")

            # Select one reference site per target region (e.g., the first one found globally)
            target_site_indices = {}
            for target_reg in target_check_regions:
                 sites_in_region = [sites[target_reg] for shank, sites in reference_sites.items() if target_reg in sites]
                 if sites_in_region:
                      target_site_indices[target_reg] = sites_in_region[0] # Just pick the first one
                      print(f"    Checking target region {target_reg} using global reference site {sites_in_region[0]}")
                 else:
                      print(f"    No global reference site found for target region {target_reg}. Cannot check co-occurrence.")


            cooccurrence_results[ref_region] = {}
            for target_reg, target_ch_idx in target_site_indices.items():
                 cooccur_count = 0
                 target_event_times = [] # Get relevant times from the target channel
                 if target_reg == 'CA1':
                     # Use peak times of SPW-R events on the target CA1 ref site
                     if target_ch_idx in ca1_spwr_events_global:
                         target_event_times = np.array([ev['peak_sample'] for ev in ca1_spwr_events_global[target_ch_idx]])
                 elif target_reg == 'CA3':
                     # Use peak times of all ripple events on the target CA3 ref site
                     if target_ch_idx in all_ripple_events_global:
                         target_event_times = np.array([ev['peak_sample'] for ev in all_ripple_events_global[target_ch_idx]])

                 if len(target_event_times) == 0:
                     print(f"    No relevant events found for target site {target_ch_idx} in {target_reg}.")
                     cooccurrence_results[ref_region][target_reg] = {'count': 0, 'details': []}
                     continue

                 cooccur_details = [] # Store details of co-occurring events
                 # Efficient check using searchsorted if many events
                 target_event_times.sort()
                 for ref_time in ref_timestamps:
                     lower_bound = ref_time - window_samples
                     upper_bound = ref_time + window_samples
                     # Find indices of target events within the window
                     start_idx = np.searchsorted(target_event_times, lower_bound, side='left')
                     end_idx = np.searchsorted(target_event_times, upper_bound, side='right')
                     indices_in_window = np.arange(start_idx, end_idx)

                     if len(indices_in_window) > 0:
                         cooccur_count += 1
                         # Store which reference event triggered, and which target event(s) occurred
                         cooccur_details.append({
                             'ref_time_sample': ref_time,
                             'target_event_indices': indices_in_window.tolist(), # Indices within target_event_times
                             'target_event_times': target_event_times[indices_in_window].tolist()
                         })

                 print(f"    Found {cooccur_count} co-occurrences between {ref_region} (ref) and {target_reg} (target site {target_ch_idx}).")
                 cooccurrence_results[ref_region][target_reg] = {'count': cooccur_count, 'details': cooccur_details}

        else:
            print(f"  Reference region {ref_region} has no global timestamps. Skipping co-occurrence analysis.")


        # 10. Spectrogram Calculation and Averaging (Globally)
        print("\n--- 10. Calculating and Averaging Spectrograms (Globally) ---")
        averaged_spectrograms = {} # Reset here
        spec_window_samples = int(SPECTROGRAM_WINDOW_MS * fs / 1000)

        for region, timestamps in averaged_timestamps.items():
            regional_ref_sites_indices = [sites[region] for shank, sites in reference_sites.items() if region in sites]
            if not regional_ref_sites_indices: continue

            # Use the LFP from the first reference site found for this region for spectrograms
            spec_ch_idx = regional_ref_sites_indices[0]
            print(f"  Calculating spectrograms for region {region} using reference site {spec_ch_idx} LFP.")

            # Load LFP data for the spec channel only when needed
            try:
                 lfp_spec_ch = lfp_data_memmap[:, spec_ch_idx].astype(np.float64)
            except MemoryError:
                 warnings.warn(f"MemoryError loading LFP for spectrogram channel {spec_ch_idx}. Skipping spectrograms for region {region}.")
                 continue
            except Exception as e_load_spec:
                 warnings.warn(f"Error loading LFP for spectrogram channel {spec_ch_idx}: {e_load_spec}. Skipping spectrograms for region {region}.")
                 continue


            region_spectrograms = []
            num_timestamps_for_spec = len(timestamps)
            print(f"  Processing {num_timestamps_for_spec} timestamps for {region} spectrograms...")
            for i, ts_sample in enumerate(timestamps):
                 if i % 100 == 0: print(f"    Spectrogram progress: {i}/{num_timestamps_for_spec}") # Progress indicator

                 # Check timestamp bounds relative to the full recording length (n_samples)
                 if ts_sample < spec_window_samples // 2 or ts_sample >= n_samples - spec_window_samples // 2:
                     # warnings.warn(f"Skipping spectrogram for timestamp {ts_sample} (too close to edge).")
                     continue # Skip timestamps too close to the edge

                 spec = calculate_cwt_spectrogram(
                     lfp_spec_ch, fs, SPECTROGRAM_FREQS,
                     timestamp_samples=ts_sample,
                     window_samples=spec_window_samples
                 )
                 # Check if spec is valid before appending
                 if spec is not None and not np.isnan(spec).all():
                      # Ensure consistent shape before averaging
                      if spec.shape == (len(SPECTROGRAM_FREQS), spec_window_samples):
                            region_spectrograms.append(spec)
                      # else: # Warning handled inside calculate_cwt_spectrogram
                      #      pass

            del lfp_spec_ch # Free memory for spec channel LFP
            gc.collect()

            if region_spectrograms:
                # Average the spectrograms across all events for this region
                try:
                    avg_spec = np.nanmean(np.stack(region_spectrograms, axis=0), axis=0) # Use nanmean
                    averaged_spectrograms[region] = avg_spec
                    print(f"  Averaged {len(region_spectrograms)} spectrograms for region {region}.")
                except MemoryError:
                     warnings.warn(f"MemoryError stacking spectrograms for region {region}. Cannot average.")
                except Exception as e_avg:
                     warnings.warn(f"Error averaging spectrograms for region {region}: {e_avg}")
            else:
                print(f"  No valid spectrograms generated for region {region}.")


        # 11. Save Results
        print("\n--- 11. Saving Results ---")
        # Save results derived from global analysis
        np.save(output_path / f'{output_filename_base}_ripple_references.npy', reference_sites)
        np.save(output_path / f'{output_filename_base}_ripple_timestamps.npy', averaged_timestamps)
        np.save(output_path / f'{output_filename_base}_ripple_cooccurrence.npy', cooccurrence_results)
        np.save(output_path / f'{output_filename_base}_ripple_avg_spectrograms.npy', averaged_spectrograms)

        # Save detected events (global and per-epoch)
        # Convert lists of dicts to structured arrays or save as pickle
        # Saving as .npy (pickle enabled) is easiest for dicts of lists of dicts
        np.save(output_path / f'{output_filename_base}_ripple_events_global.npy', all_ripple_events_global, allow_pickle=True)
        np.save(output_path / f'{output_filename_base}_spw_events_global.npy', all_spw_events_global, allow_pickle=True)
        np.save(output_path / f'{output_filename_base}_ca1_spwr_events_global.npy', ca1_spwr_events_global, allow_pickle=True)
        np.save(output_path / f'{output_filename_base}_ripple_events_by_epoch.npy', all_ripple_events_by_epoch, allow_pickle=True)
        np.save(output_path / f'{output_filename_base}_spw_events_by_epoch.npy', all_spw_events_by_epoch, allow_pickle=True)

        # Save channel info used
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
        # Close memmap if it's still open
        if lfp_data_memmap is not None and hasattr(lfp_data_memmap, '_mmap'):
             try:
                 lfp_data_memmap._mmap.close()
                 print("Closed LFP data memory map.")
             except Exception as e_close:
                 print(f"Warning: Error closing memmap: {e_close}")
        del lfp_data_memmap
        # Delete large variables explicitly
        vars_to_del = [
             'all_ripple_events_by_epoch', 'all_spw_events_by_epoch',
             'all_ripple_events_global', 'all_spw_events_global',
             'ca1_spwr_events_global', 'reference_sites', 'averaged_timestamps',
             'cooccurrence_results', 'averaged_spectrograms', 'baseline_stats_per_channel',
             'sleep_state_data', 'epoch_boundaries_sec', 'non_rem_periods_sec',
             'non_rem_periods_samples', 'epoch_boundaries_samples', 'channel_df',
             'region_channels_df', 'lfp_ch_epoch', 'lfp_ch_ripple_filtered', 'ripple_power',
             'lfp_ch_spw_filtered', 'lfp_spec_ch', 'region_spectrograms', 'avg_spec'
             # Add others if they become large
        ]
        deleted_count = 0
        local_vars = locals()
        for var_name in vars_to_del:
            if var_name in local_vars and local_vars[var_name] is not None:
                try:
                    # Instead of del, assign None to help GC
                    local_vars[var_name] = None
                    deleted_count += 1
                except NameError: pass # Already deleted or out of scope
                except Exception as del_e:
                     print(f"Note: Could not clear variable {var_name}: {del_e}")

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
    if not lfp_file_str:
        print("No LFP file selected. Exiting.")
        exit()
    LFP_FILE = Path(lfp_file_str)

    print("Please select the corresponding Meta file (*.meta)...")
    # Suggest meta file based on LFP file name
    suggested_meta_name = LFP_FILE.stem + ".meta"
    meta_file_str = filedialog.askopenfilename(
        title="Select Meta File (*.meta)",
        initialdir=LFP_FILE.parent,
        initialfile=suggested_meta_name,
        filetypes=[("Meta files", "*.meta"), ("All files", "*.*")]
    )
    if not meta_file_str:
        print("No Meta file selected. Exiting.")
        exit()
    META_FILE = Path(meta_file_str)

    print("Please select the Channel Info CSV file...")
    channel_info_file_str = filedialog.askopenfilename(
        title="Select Channel Info CSV File",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    if not channel_info_file_str:
        print("No Channel Info file selected. Exiting.")
        exit()
    CHANNEL_INFO_FILE = Path(channel_info_file_str)

    print("Please select the Sleep State file (*_sleep_states*.npy) (Optional)...")
    sleep_state_file_str = filedialog.askopenfilename(
        title="Select Sleep State File (Optional - Cancel if none)",
        filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")]
    )
    # Allow sleep state file to be optional
    SLEEP_STATE_FILE = Path(sleep_state_file_str) if sleep_state_file_str else None
    if SLEEP_STATE_FILE:
        print(f"Selected Sleep State file: {SLEEP_STATE_FILE.name}")
    else:
        print("No Sleep State file selected (Optional).")


    print("Please select the Epoch Boundaries file (*_epoch_boundaries*.npy) (Optional)...")
    epoch_boundaries_file_str = filedialog.askopenfilename(
        title="Select Epoch Boundaries File (Optional - Cancel if none)",
        filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")]
    )
    # Allow epoch boundaries file to be optional
    EPOCH_BOUNDARIES_FILE = Path(epoch_boundaries_file_str) if epoch_boundaries_file_str else None
    if EPOCH_BOUNDARIES_FILE:
        print(f"Selected Epoch Boundaries file: {EPOCH_BOUNDARIES_FILE.name}")
    else:
        print("No Epoch Boundaries file selected (Optional).")


    print("Please select the Output directory...")
    output_dir_str = filedialog.askdirectory(
        title="Select Output Directory"
    )
    if not output_dir_str:
        print("No output directory selected. Exiting.")
        exit()
    OUTPUT_DIRECTORY = Path(output_dir_str)

    root.destroy() # Close the hidden tkinter root window

    # --- Validation of selected paths ---
    if not LFP_FILE.is_file(): print(f"Error: Selected LFP file not found: {LFP_FILE}"); exit()
    if not META_FILE.is_file(): print(f"Error: Selected Meta file not found: {META_FILE}"); exit()
    if not CHANNEL_INFO_FILE.is_file(): print(f"Error: Selected Channel Info file not found: {CHANNEL_INFO_FILE}"); exit()
    if SLEEP_STATE_FILE and not SLEEP_STATE_FILE.is_file(): print(f"Error: Selected Sleep State file not found: {SLEEP_STATE_FILE}"); exit()
    if EPOCH_BOUNDARIES_FILE and not EPOCH_BOUNDARIES_FILE.is_file(): print(f"Error: Selected Epoch Boundaries file not found: {EPOCH_BOUNDARIES_FILE}"); exit()
    try:
        OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
        print(f"Using output directory: {OUTPUT_DIRECTORY}")
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

