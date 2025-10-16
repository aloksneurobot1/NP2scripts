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

# --- Make sure DemoReadSGLXData is accessible or adjust import path ---
try:
    from DemoReadSGLXData.readSGLX import readMeta
    print("Successfully imported readMeta from DemoReadSGLXData.readSGLX.")
except ImportError:
    print("Warning: Could not import readMeta from DemoReadSGLXData.readSGLX.")
    print(" Using placeholder readMeta function.")
    def readMeta(meta_path):
        """Placeholder for readMeta if DemoReadSGLXData is not available."""
        print(f"Warning: Using placeholder readMeta for {meta_path}. Returns dummy metadata.")
        return {'nSavedChans': '385', 'imSampRate': '2500.0', 'fileTimeSecs': '19000'}

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
NUM_CORES = max(1, multiprocessing.cpu_count() // 2) # Use half the cores
print(f"Configuring parallel processing to use {NUM_CORES} cores.")

# --- Sleep States to Include ---
# Define which states to keep events from (0=Awake, 1=NREM, 2=REM)
# Now used to create separate outputs
STATES_TO_ANALYZE = {
    0: "Awake",
    1: "NREM"
}
print(f"Analyzing events separately for states: {STATES_TO_ANALYZE}")


# --- Data Loading Functions --- (Unchanged from previous version)
def load_lfp_data_memmap(file_path, meta_path, data_type='int16'):
    """Loads LFP data using memory-mapping."""
    sampling_rate = None; data = None; n_channels = 0; n_samples = 0
    print(f"Attempting to load LFP data from: {file_path}")
    print(f"Using metadata from: {meta_path}")
    try:
        meta = readMeta(meta_path); n_channels = int(meta['nSavedChans'])
        if 'imSampRate' in meta: sampling_rate = float(meta['imSampRate'])
        elif 'niSampRate' in meta: sampling_rate = float(meta['niSampRate'])
        else: print(f"Error: Sampling rate key not found."); return None, None, 0, 0
        if sampling_rate <= 0: print(f"Error: Invalid sampling rate."); return None, None, n_channels, 0
        print(f"Metadata: {n_channels} channels. Rate: {sampling_rate:.6f} Hz")
        file_size = file_path.stat().st_size; item_size = np.dtype(data_type).itemsize
        if n_channels <= 0 or item_size <= 0: print(f"Error: Invalid channels/itemsize."); return None, sampling_rate, n_channels, 0
        expected_total_bytes = file_size - (file_size % (n_channels * item_size))
        if file_size != expected_total_bytes: print(f"Warning: File size mismatch.")
        n_samples = expected_total_bytes // (n_channels * item_size)
        if n_samples <= 0: print("Error: Zero samples calculated."); return None, sampling_rate, n_channels, 0
        shape = (n_samples, n_channels); print(f"Calculated samples: {n_samples}. Shape: {shape}")
        data = np.memmap(file_path, dtype=data_type, mode='r', shape=shape, offset=0)
        print(f"Successfully memory-mapped file: {file_path}")
        return data, sampling_rate, n_channels, n_samples
    except FileNotFoundError: print(f"Error: File not found - {file_path} or {meta_path}"); return None, None, 0, 0
    except KeyError as e: print(f"Error: Metadata key missing - {e}"); return None, sampling_rate, n_channels, 0
    except ValueError as e: print(f"Error: Memmap shape/dtype error - {e}"); return None, sampling_rate, n_channels, 0
    except Exception as e: print(f"Unexpected error loading LFP: {e}"); traceback.print_exc(); return None, None, 0, 0

def load_channel_info(filepath):
    """Loads channel information from the CSV file."""
    print(f"Loading channel info from {filepath}")
    try:
        channel_df = pd.read_csv(filepath); required_cols = ['global_channel_index', 'shank_index', 'acronym', 'name']
        if not all(col in channel_df.columns for col in required_cols): raise ValueError(f"Missing columns in CSV: {required_cols}")
        print(f"Loaded channel info for {len(channel_df)} channels.")
        return channel_df
    except FileNotFoundError: print(f"Error: Channel info file not found: {filepath}"); raise
    except Exception as e: print(f"Error loading channel info: {e}"); raise

def load_sleep_and_epoch_data(sleep_state_path, epoch_boundaries_path, fs):
    """Loads sleep state and epoch data."""
    print(f"Attempting to load sleep states from: {sleep_state_path}")
    print(f"Attempting to load epoch boundaries from: {epoch_boundaries_path}")
    sleep_state_data = None; epoch_boundaries_sec = []; non_rem_periods_sec = []
    if epoch_boundaries_path and epoch_boundaries_path.exists():
        try:
            loaded_epochs = np.load(epoch_boundaries_path, allow_pickle=True)
            if isinstance(loaded_epochs, (list, np.ndarray)):
                 valid_epochs = [tuple(ep) for ep in loaded_epochs if isinstance(ep, (list, tuple, np.ndarray)) and len(ep) == 2 and all(isinstance(t, (int, float)) for t in ep) and ep[1] >= ep[0]]
                 epoch_boundaries_sec = valid_epochs; print(f"Loaded {len(epoch_boundaries_sec)} valid epoch boundaries (sec).")
        except Exception as e: print(f"Error loading epoch boundaries file {epoch_boundaries_path.name}: {e}")
    if sleep_state_path and sleep_state_path.exists():
        try:
            state_codes = np.load(sleep_state_path, allow_pickle=True); times_path = None
            times_path_str_1 = sleep_state_path.name.replace('_sleep_states', '_sleep_state_times'); times_path_1 = sleep_state_path.parent / times_path_str_1
            if times_path_1.exists(): times_path = times_path_1
            else:
                 base_name_match = re.match(r"^(.*?)_sleep_states.*\.npy$", sleep_state_path.name)
                 if base_name_match:
                      base_name = base_name_match.group(1); times_path_alt_generic = sleep_state_path.parent / f"{base_name}_sleep_state_times.npy"
                      if times_path_alt_generic.exists(): times_path = times_path_alt_generic
                      else: times_path_alt_emg = sleep_state_path.parent / f"{base_name}_sleep_state_times_EMG.npy";
                           if times_path_alt_emg.exists(): times_path = times_path_alt_emg
            if times_path and times_path.exists():
                 state_times_sec = np.load(times_path, allow_pickle=True); print(f"Loaded state codes ({len(state_codes)}) and times ({len(state_times_sec)}).")
                 if len(state_codes) == len(state_times_sec):
                     sleep_state_data = np.column_stack((state_times_sec, state_codes)); print(f"Sleep state data shape: {sleep_state_data.shape}")
                     nrem_indices = np.where(sleep_state_data[:, 1] == 1)[0]
                     if len(nrem_indices) > 0:
                         diff = np.diff(nrem_indices); splits = np.where(diff != 1)[0] + 1; nrem_blocks_indices = np.split(nrem_indices, splits)
                         for block in nrem_blocks_indices:
                             if len(block) > 0:
                                 start_idx, end_idx = block[0], block[-1]; start_time = sleep_state_data[start_idx, 0]
                                 end_time = sleep_state_data[end_idx + 1, 0] if end_idx + 1 < len(sleep_state_data) else sleep_state_data[end_idx, 0] + (np.median(np.diff(sleep_state_data[:,0])) if len(sleep_state_data)>1 else 1.0)
                                 non_rem_periods_sec.append((start_time, end_time))
                         print(f"Identified {len(non_rem_periods_sec)} non-REM periods for baseline.")
        except Exception as e: print(f"Error loading sleep state file/times: {e}"); sleep_state_data = None
    if sleep_state_data is None: warnings.warn("Could not load valid sleep state data. Event filtering by state is disabled.")
    if not non_rem_periods_sec: warnings.warn("No non-REM periods identified. Baseline calculation will use the whole signal.")
    return sleep_state_data, epoch_boundaries_sec, non_rem_periods_sec


# --- Signal Processing Functions --- (Unchanged)
def apply_fir_filter(data, lowcut, highcut, fs, numtaps=513, pass_zero=False):
    if data is None or data.size == 0: return None; data_len = data.shape[-1]
    if numtaps >= data_len: numtaps = data_len - 1 if data_len > 1 else 1; numtaps -= (numtaps % 2 == 0);
    if numtaps < 3: return data
    try:
        if lowcut is None and highcut is None: return data
        elif lowcut is None: b = signal.firwin(numtaps, highcut, fs=fs, pass_zero=True, window='hamming')
        elif highcut is None: b = signal.firwin(numtaps, lowcut, fs=fs, pass_zero=False, window='hamming')
        else: nyq = fs / 2.0; highcut = min(highcut, nyq * 0.99);
              if lowcut >= highcut: print(f"Error: lowcut>=highcut"); return None
              b = signal.firwin(numtaps, [lowcut, highcut], fs=fs, pass_zero=pass_zero, window='hamming')
        if not np.issubdtype(data.dtype, np.floating): data = data.astype(np.float64)
        return signal.filtfilt(b, 1, data, axis=-1)
    except Exception as e: print(f"FIR filter error: {e}"); return None

def calculate_instantaneous_power(data):
    if data is None or data.size == 0: return None
    try:
        if not np.issubdtype(data.dtype, np.floating): data = data.astype(np.float64)
        analytic_signal = signal.hilbert(data, axis=-1); return np.abs(analytic_signal)**2
    except Exception as e: print(f"Power calc error: {e}"); return None

def get_baseline_stats(power_signal, fs, non_rem_periods_samples):
    if power_signal is None or power_signal.size == 0: raise ValueError("Invalid power signal.")
    baseline_power_segments = [power_signal[max(0,s):min(power_signal.shape[0],e)] for s,e in non_rem_periods_samples if max(0,s)<min(power_signal.shape[0],e)] if non_rem_periods_samples else [power_signal]
    if not baseline_power_segments: baseline_power_segments = [power_signal]
    if not baseline_power_segments: raise ValueError("No valid baseline segments.")
    try: full_baseline_power = np.concatenate(baseline_power_segments)
    except MemoryError: raise ValueError("MemoryError concatenating baseline.")
    except Exception as e: raise ValueError(f"Error preparing baseline: {e}")
    if full_baseline_power.size == 0: raise ValueError("Baseline power empty.")
    initial_mean, initial_sd = np.mean(full_baseline_power), np.std(full_baseline_power)
    clipped_power = np.clip(full_baseline_power, None, initial_mean + 4 * initial_sd) if initial_sd > 0 else full_baseline_power
    rectified_power = np.abs(clipped_power); numtaps_lp = 101
    if numtaps_lp >= rectified_power.size: processed_baseline_power = rectified_power
    else:
         try: nyq_lp = fs / 2.0; lp_cutoff = min(RIPPLE_POWER_LP_CUTOFF, nyq_lp * 0.99)
              b_lp = signal.firwin(numtaps_lp, lp_cutoff, fs=fs, pass_zero=True, window='hamming')
              processed_baseline_power = signal.filtfilt(b_lp, 1, rectified_power)
         except Exception as e_lp: warnings.warn(f"Baseline LP filter error: {e_lp}"); processed_baseline_power = rectified_power
    return np.mean(processed_baseline_power), np.std(processed_baseline_power)

def detect_events(signal_data, fs, threshold_high, threshold_low, min_duration_ms, merge_gap_ms, event_type="Event"):
    if signal_data is None or signal_data.size == 0: return np.array([], dtype=int), np.array([], dtype=int)
    min_duration_samples = int(min_duration_ms * fs / 1000); merge_gap_samples = int(merge_gap_ms * fs / 1000)
    try:
        with np.errstate(invalid='ignore'): above_high_threshold = signal_data > threshold_high
        if np.any(np.isnan(signal_data)): above_high_threshold[np.isnan(signal_data)] = False
        diff_high = np.diff(above_high_threshold.astype(np.int8)); starts_high, ends_high = np.where(diff_high == 1)[0] + 1, np.where(diff_high == -1)[0]
        if len(above_high_threshold) > 0:
            if above_high_threshold[0]: starts_high = np.insert(starts_high, 0, 0)
            if above_high_threshold[-1]: ends_high = np.append(ends_high, len(signal_data) - 1)
        else: return np.array([], dtype=int), np.array([], dtype=int)
    except Exception as e: print(f"Thresholding error ({event_type}): {e}"); return np.array([], dtype=int), np.array([], dtype=int)
    if len(starts_high) == 0 or len(ends_high) == 0: return np.array([], dtype=int), np.array([], dtype=int)
    valid_pairs = []; s_idx, e_idx = 0, 0
    while s_idx < len(starts_high) and e_idx < len(ends_high):
        if starts_high[s_idx] <= ends_high[e_idx]: valid_pairs.append((starts_high[s_idx], ends_high[e_idx])); s_idx += 1; e_idx += 1
        else: e_idx += 1
    if not valid_pairs: return np.array([], dtype=int), np.array([], dtype=int)
    starts_high, ends_high = zip(*valid_pairs); starts_high, ends_high = np.array(starts_high), np.array(ends_high)
    expanded_starts, expanded_ends = [], []
    try:
        with np.errstate(invalid='ignore'): below_low_mask = signal_data < threshold_low
        if np.any(np.isnan(signal_data)): below_low_mask[np.isnan(signal_data)] = True
        for start, end in zip(starts_high, ends_high):
            cs = start; while cs > 0 and not below_low_mask[cs - 1]: cs -= 1
            ce = end; while ce < len(signal_data) - 1 and not below_low_mask[ce + 1]: ce += 1
            expanded_starts.append(cs); expanded_ends.append(ce)
    except Exception as e: print(f"Expansion error ({event_type}): {e}"); return np.array([], dtype=int), np.array([], dtype=int)
    if not expanded_starts: return np.array([], dtype=int), np.array([], dtype=int)
    merged_starts, merged_ends = [expanded_starts[0]], [expanded_ends[0]]
    for i in range(1, len(expanded_starts)):
        gap = expanded_starts[i] - merged_ends[-1] - 1
        if gap < merge_gap_samples: merged_ends[-1] = max(merged_ends[-1], expanded_ends[i])
        else: merged_starts.append(expanded_starts[i]); merged_ends.append(expanded_ends[i])
    final_starts, final_ends = [], []
    for start, end in zip(merged_starts, merged_ends):
        if (end - start + 1) >= min_duration_samples: final_starts.append(start); final_ends.append(end)
    return np.array(final_starts, dtype=int), np.array(final_ends, dtype=int)

def find_event_features(lfp_segment, power_segment):
    if power_segment is None or lfp_segment is None or power_segment.size==0 or lfp_segment.size==0 or len(lfp_segment)!=len(power_segment): return -1,-1
    try:
        peak_power_idx_rel = np.nanargmax(power_segment)
        if not np.issubdtype(lfp_segment.dtype, np.floating): lfp_segment_float = lfp_segment.astype(np.float64)
        else: lfp_segment_float = lfp_segment
        troughs_rel, _ = signal.find_peaks(-lfp_segment_float)
        if len(troughs_rel) == 0: trough_idx_rel = np.nanargmin(lfp_segment_float)
        else: trough_idx_rel = troughs_rel[np.nanargmin(np.abs(troughs_rel - peak_power_idx_rel))]
        return peak_power_idx_rel, trough_idx_rel
    except Exception: return -1, -1

def calculate_cwt_spectrogram(data, fs, freqs, timestamp_samples, window_samples):
    if data is None or data.size == 0: return np.full((len(freqs), window_samples), np.nan)
    start_sample = int(timestamp_samples - window_samples // 2); end_sample = int(start_sample + window_samples)
    pad_left, pad_right = 0, 0
    if start_sample < 0: pad_left = -start_sample; start_sample = 0
    if end_sample > len(data): pad_right = end_sample - len(data); end_sample = len(data)
    segment = data[start_sample:end_sample]
    if segment.size == 0: return np.full((len(freqs), window_samples), np.nan)
    if pad_left > 0 or pad_right > 0:
        try: segment = np.pad(segment, (pad_left, pad_right), mode='reflect')
        except ValueError: segment = np.pad(segment, (pad_left, pad_right), mode='edge')
    if len(segment) != window_samples:
         if len(segment) > window_samples: segment = segment[:window_samples]
         else: segment = np.pad(segment, (0, window_samples - len(segment)), mode='edge')
    try:
        wavelet = f'cmor1.5-1.0'; scales = pywt.frequency2scale(wavelet, freqs / fs)
        coeffs, _ = pywt.cwt(segment.astype(np.float64), scales, wavelet, sampling_period=1.0/fs)
        power_spectrogram = np.abs(coeffs)**2
        if power_spectrogram.shape[1] != window_samples:
             from scipy.interpolate import interp1d; x_old = np.linspace(0, 1, power_spectrogram.shape[1]); x_new = np.linspace(0, 1, window_samples)
             interp_func = interp1d(x_old, power_spectrogram, axis=1, kind='linear', fill_value='extrapolate'); power_spectrogram = interp_func(x_new)
        return power_spectrogram
    except MemoryError: return np.full((len(freqs), window_samples), np.nan)
    except Exception: return np.full((len(freqs), window_samples), np.nan)


# --- Worker Functions for Parallel Processing --- (Unchanged)
def _calculate_baseline_worker(ch_idx, lfp_filepath, meta_filepath, non_rem_periods_samples_orig, fs_orig, n_channels):
    """Worker: Calculate baseline stats for one channel, includes downsampling."""
    lfp_data_memmap_worker = None; fs_eff = fs_orig / DOWNSAMPLING_FACTOR if DOWNSAMPLING_FACTOR > 1 else fs_orig
    try:
        meta = readMeta(meta_filepath); file_size = lfp_filepath.stat().st_size; item_size = np.dtype('int16').itemsize
        expected_total_bytes = file_size - (file_size % (n_channels * item_size)); n_samples = expected_total_bytes // (n_channels * item_size)
        shape = (n_samples, n_channels); lfp_data_memmap_worker = np.memmap(lfp_filepath, dtype='int16', mode='r', shape=shape)
        lfp_ch_full_orig = lfp_data_memmap_worker[:, ch_idx].astype(np.float64)
        if DOWNSAMPLING_FACTOR > 1: lfp_ch_full_down = signal.decimate(lfp_ch_full_orig, q=DOWNSAMPLING_FACTOR, axis=0, ftype='fir', zero_phase=True); del lfp_ch_full_orig
        else: lfp_ch_full_down = lfp_ch_full_orig
        lfp_ch_ripple_filtered_down = apply_fir_filter(lfp_ch_full_down, RIPPLE_FILTER_LOWCUT, RIPPLE_FILTER_HIGHCUT, fs_eff); del lfp_ch_full_down
        if lfp_ch_ripple_filtered_down is None: return ch_idx, None
        ripple_power_down = calculate_instantaneous_power(lfp_ch_ripple_filtered_down); del lfp_ch_ripple_filtered_down
        if ripple_power_down is None: return ch_idx, None
        non_rem_periods_samples_down = [(s // DOWNSAMPLING_FACTOR, e // DOWNSAMPLING_FACTOR) for s, e in non_rem_periods_samples_orig]
        baseline_mean, baseline_sd = get_baseline_stats(ripple_power_down, fs_eff, non_rem_periods_samples_down); del ripple_power_down
        return ch_idx, (baseline_mean, baseline_sd)
    except Exception as e: print(f"Baseline worker error (Ch {ch_idx}): {e}"); return ch_idx, None
    finally:
        if lfp_data_memmap_worker is not None and hasattr(lfp_data_memmap_worker, '_mmap'):
            try: lfp_data_memmap_worker._mmap.close()
            except Exception: pass
        gc.collect()

def _detect_events_worker(ch_idx, ch_region, epoch_start_sample_orig, epoch_end_sample_orig,
                         lfp_filepath, n_samples_orig, n_channels, baseline_stats, fs_orig):
    """Worker: Detect events for one channel/epoch, includes downsampling, scales indices back."""
    lfp_data_memmap_worker = None; ripple_events = []; spw_events = []; fs_eff = fs_orig / DOWNSAMPLING_FACTOR if DOWNSAMPLING_FACTOR > 1 else fs_orig
    try:
        shape = (n_samples_orig, n_channels); lfp_data_memmap_worker = np.memmap(lfp_filepath, dtype='int16', mode='r', shape=shape)
        lfp_ch_epoch_orig = lfp_data_memmap_worker[epoch_start_sample_orig:epoch_end_sample_orig, ch_idx].astype(np.float64)
        if DOWNSAMPLING_FACTOR > 1: lfp_ch_epoch_down = signal.decimate(lfp_ch_epoch_orig, q=DOWNSAMPLING_FACTOR, axis=0, ftype='fir', zero_phase=True)
        else: lfp_ch_epoch_down = lfp_ch_epoch_orig
        del lfp_ch_epoch_orig
        lfp_ch_ripple_filtered_down = apply_fir_filter(lfp_ch_epoch_down, RIPPLE_FILTER_LOWCUT, RIPPLE_FILTER_HIGHCUT, fs_eff)
        if lfp_ch_ripple_filtered_down is not None:
            ripple_power_down = calculate_instantaneous_power(lfp_ch_ripple_filtered_down)
            if ripple_power_down is not None:
                baseline_mean, baseline_sd = baseline_stats
                if baseline_sd > 0:
                    det_thr = baseline_mean + RIPPLE_DETECTION_SD_THRESHOLD * baseline_sd; exp_thr = baseline_mean + RIPPLE_EXPANSION_SD_THRESHOLD * baseline_sd
                    r_starts_rel_d, r_ends_rel_d = detect_events(ripple_power_down, fs_eff, det_thr, exp_thr, RIPPLE_MIN_DURATION_MS, RIPPLE_MERGE_GAP_MS, "Ripple")
                    for sr_d, er_d in zip(r_starts_rel_d, r_ends_rel_d):
                        if sr_d < 0 or er_d >= len(lfp_ch_ripple_filtered_down): continue
                        lfp_seg_d = lfp_ch_ripple_filtered_down[sr_d:er_d+1]; pow_seg_d = ripple_power_down[sr_d:er_d+1]
                        pk_idx_d, tr_idx_d = find_event_features(lfp_seg_d, pow_seg_d)
                        if pk_idx_d != -1 and tr_idx_d != -1:
                            dur_samp_d = er_d - sr_d + 1; start_abs = epoch_start_sample_orig + (sr_d * DOWNSAMPLING_FACTOR); end_abs = start_abs + (dur_samp_d * DOWNSAMPLING_FACTOR) - 1
                            pk_abs = epoch_start_sample_orig + ((sr_d + pk_idx_d) * DOWNSAMPLING_FACTOR); tr_abs = epoch_start_sample_orig + ((sr_d + tr_idx_d) * DOWNSAMPLING_FACTOR)
                            ripple_events.append({'start_sample': start_abs, 'end_sample': end_abs, 'peak_sample': pk_abs, 'trough_sample': tr_abs, 'peak_power': pow_seg_d[pk_idx_d], 'duration_ms': dur_samp_d / fs_eff * 1000})
                del ripple_power_down
            del lfp_ch_ripple_filtered_down
        if ch_region == 'CA1':
            lfp_ch_spw_filtered_down = apply_fir_filter(lfp_ch_epoch_down, SPW_FILTER_LOWCUT, SPW_FILTER_HIGHCUT, fs_eff)
            if lfp_ch_spw_filtered_down is not None:
                spw_mean_d = np.nanmean(lfp_ch_spw_filtered_down); spw_sd_d = np.nanstd(lfp_ch_spw_filtered_down)
                if spw_sd_d > 0 and np.isfinite(spw_sd_d):
                    spw_det_thr = SPW_DETECTION_SD_THRESHOLD * spw_sd_d; spw_exp_thr = 1.0 * spw_sd_d
                    s_starts_rel_d, s_ends_rel_d = detect_events(np.abs(lfp_ch_spw_filtered_down - spw_mean_d), fs_eff, spw_det_thr, spw_exp_thr, SPW_MIN_DURATION_MS, 1, "SPW")
                    for sr_d, er_d in zip(s_starts_rel_d, s_ends_rel_d):
                         if sr_d < 0 or er_d >= len(lfp_ch_spw_filtered_down): continue
                         dur_samp_d = er_d - sr_d + 1; dur_ms = dur_samp_d / fs_eff * 1000
                         if dur_ms <= SPW_MAX_DURATION_MS:
                              spw_seg_d = lfp_ch_spw_filtered_down[sr_d:er_d+1]; tr_idx_seg_d = np.nanargmin(spw_seg_d)
                              start_abs = epoch_start_sample_orig + (sr_d * DOWNSAMPLING_FACTOR); end_abs = start_abs + (dur_samp_d * DOWNSAMPLING_FACTOR) - 1
                              tr_abs = epoch_start_sample_orig + ((sr_d + tr_idx_seg_d) * DOWNSAMPLING_FACTOR)
                              spw_events.append({'start_sample': start_abs, 'end_sample': end_abs, 'trough_sample': tr_abs, 'duration_ms': dur_ms})
                del lfp_ch_spw_filtered_down
        del lfp_ch_epoch_down
        return ch_idx, ripple_events, spw_events
    except Exception as e: print(f"Event worker error (Ch {ch_idx}, Ep {epoch_start_sample_orig}): {e}"); return ch_idx, [], []
    finally:
        if lfp_data_memmap_worker is not None and hasattr(lfp_data_memmap_worker, '_mmap'):
            try: lfp_data_memmap_worker._mmap.close()
            except Exception: pass
        gc.collect()

def _calculate_spectrogram_worker(ts_sample_orig, spec_ch_idx, lfp_filepath, n_samples_orig, n_channels, fs_orig):
    """Worker: Calculate spectrogram for one timestamp, includes downsampling."""
    lfp_data_memmap_worker = None; fs_eff = fs_orig / DOWNSAMPLING_FACTOR if DOWNSAMPLING_FACTOR > 1 else fs_orig
    window_samples_orig = int(SPECTROGRAM_WINDOW_MS * fs_orig / 1000); window_samples_eff = int(SPECTROGRAM_WINDOW_MS * fs_eff / 1000)
    try:
        shape = (n_samples_orig, n_channels); lfp_data_memmap_worker = np.memmap(lfp_filepath, dtype='int16', mode='r', shape=shape)
        start_sample_orig = int(ts_sample_orig - window_samples_orig // 2); end_sample_orig = int(start_sample_orig + window_samples_orig)
        pad_left_orig, pad_right_orig = 0, 0
        if start_sample_orig < 0: pad_left_orig = -start_sample_orig; start_sample_orig = 0
        if end_sample_orig > n_samples_orig: pad_right_orig = end_sample_orig - n_samples_orig; end_sample_orig = n_samples_orig
        segment_orig = lfp_data_memmap_worker[start_sample_orig:end_sample_orig, spec_ch_idx].astype(np.float64)
        if pad_left_orig > 0 or pad_right_orig > 0:
             try: segment_orig = np.pad(segment_orig, (pad_left_orig, pad_right_orig), mode='reflect')
             except ValueError: segment_orig = np.pad(segment_orig, (pad_left_orig, pad_right_orig), mode='edge')
        if DOWNSAMPLING_FACTOR > 1 and segment_orig.size > DOWNSAMPLING_FACTOR * 2: segment_down = signal.decimate(segment_orig, q=DOWNSAMPLING_FACTOR, axis=0, ftype='fir', zero_phase=True)
        else: segment_down = segment_orig
        del segment_orig
        spec = calculate_cwt_spectrogram(segment_down, fs_eff, SPECTROGRAM_FREQS, len(segment_down) // 2, window_samples_eff); del segment_down
        return spec
    except Exception as e: return np.full((len(SPECTROGRAM_FREQS), window_samples_eff), np.nan)
    finally:
        if lfp_data_memmap_worker is not None and hasattr(lfp_data_memmap_worker, '_mmap'):
            try: lfp_data_memmap_worker._mmap.close()
            except Exception: pass
        gc.collect()


# --- Main Analysis Function ---

def run_ripple_analysis(lfp_filepath, meta_filepath, channel_info_filepath,
                        sleep_state_filepath, epoch_boundaries_filepath,
                        output_dir):
    """Main function for ripple/SPW analysis with epoch-specific NREM/Awake focus and separate outputs."""

    lfp_filepath = Path(lfp_filepath); meta_filepath = Path(meta_filepath)
    channel_info_filepath = Path(channel_info_filepath)
    sleep_state_filepath = Path(sleep_state_filepath) if sleep_state_filepath else None
    epoch_boundaries_filepath = Path(epoch_boundaries_filepath) if epoch_boundaries_filepath else None
    output_dir = Path(output_dir)
    output_path = Path(output_dir); output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_path}")

    match = re.match(r"^(.*?)(\.(imec|nidq)\d?)?\.lf\.bin$", lfp_filepath.name)
    output_filename_base = match.group(1) if match else lfp_filepath.stem
    print(f"Derived base filename: {output_filename_base}")

    # Initialize results storage (more granular)
    all_ripple_events_by_epoch = {} # All detected ripples per epoch
    all_spw_events_by_epoch = {}    # All detected SPWs per epoch
    # State-specific storage
    ripple_events_by_state_epoch = {state: {} for state in STATES_TO_ANALYZE} # {state_code: {epoch_idx: {ch_idx: [events]}}}
    spw_events_by_state_epoch = {state: {} for state in STATES_TO_ANALYZE}    # {state_code: {epoch_idx: {ch_idx: [events]}}}
    ripple_events_by_state_global = {state: {} for state in STATES_TO_ANALYZE} # {state_code: {ch_idx: [events]}}
    spw_events_by_state_global = {state: {} for state in STATES_TO_ANALYZE}    # {state_code: {ch_idx: [events]}}
    ca1_spwr_events_by_state_global = {state: {} for state in STATES_TO_ANALYZE} # {state_code: {ch_idx: [events]}}
    averaged_timestamps_by_state_epoch = {state: {} for state in STATES_TO_ANALYZE} # {state_code: {epoch_idx: {region: timestamps}}}
    cooccurrence_results_by_state_epoch = {state: {} for state in STATES_TO_ANALYZE} # {state_code: {epoch_idx: {ref_region: {target_region: results}}}}
    averaged_spectrograms_by_state_epoch = {state: {} for state in STATES_TO_ANALYZE} # {state_code: {epoch_idx: {region: avg_spec}}}
    # Global reference sites (determined once)
    reference_sites = {}

    try:
        # 1. Load LFP Info & Channel Info
        print("\n--- 1 & 2. Loading LFP Info & Channel Info ---")
        _, fs_orig, n_channels, n_samples_orig = load_lfp_data_memmap(lfp_filepath, meta_filepath)
        if fs_orig is None: raise ValueError("Failed to load LFP.")
        fs_eff = fs_orig / DOWNSAMPLING_FACTOR if DOWNSAMPLING_FACTOR > 1 else fs_orig
        print(f"LFP: {n_channels} channels, {n_samples_orig} samples. Orig Fs={fs_orig:.2f} Hz, Eff Fs={fs_eff:.2f} Hz")
        channel_df = load_channel_info(channel_info_filepath)
        target_regions = ['CA1', 'CA3', 'CA2']
        region_channels_df = channel_df[channel_df['acronym'].isin(target_regions)].copy()
        if region_channels_df.empty: raise ValueError("No target region channels found.")
        channels_to_process = sorted(region_channels_df['global_channel_index'].astype(int).unique())
        channel_region_map = pd.Series(region_channels_df.acronym.values, index=region_channels_df.global_channel_index).to_dict()

        # 3. Load Sleep/Epoch Data
        print("\n--- 3. Loading Sleep State and Epoch Data ---")
        sleep_state_data, epoch_boundaries_sec, non_rem_periods_sec = load_sleep_and_epoch_data(sleep_state_filepath, epoch_boundaries_filepath, fs_orig)
        non_rem_periods_samples_orig = [(int(s*fs_orig), int(e*fs_orig)) for s,e in non_rem_periods_sec]
        epoch_boundaries_samples_orig = [(int(s*fs_orig), int(e*fs_orig)) for s,e in epoch_boundaries_sec]
        if not epoch_boundaries_samples_orig: epoch_boundaries_samples_orig = [(0, n_samples_orig)]

        # Prepare sleep state lookup
        state_lookup = None
        if sleep_state_data is not None:
             print("Preparing sleep state lookup...")
             sleep_state_data = sleep_state_data[np.argsort(sleep_state_data[:, 0]), :]
             state_times_sec = sleep_state_data[:, 0]; state_codes = sleep_state_data[:, 1]
             def get_state_at_sample(sample_idx, fs):
                 time_sec = sample_idx / fs; idx = np.searchsorted(state_times_sec, time_sec, side='right') - 1
                 return state_codes[idx] if idx >= 0 else -1
             state_lookup = get_state_at_sample; print("Sleep state lookup ready.")
        else:
             raise ValueError("Sleep state data is required for state-specific analysis.")

        # 4. Calculate Global Baseline (Parallel)
        print(f"\n--- 4. Calculating Global Baseline Statistics (Parallel using {NUM_CORES} cores) ---")
        baseline_stats_per_channel = {}
        baseline_tasks = [(ch, lfp_filepath, meta_filepath, non_rem_periods_samples_orig, fs_orig, n_channels) for ch in channels_to_process if ch < n_channels]
        results = Parallel(n_jobs=NUM_CORES, backend="loky")(delayed(_calculate_baseline_worker)(*task) for task in baseline_tasks)
        valid_baselines = 0
        for ch, stats_res in results:
            if stats_res: baseline_stats_per_channel[ch] = stats_res; valid_baselines += 1
        print(f"Calculated baseline for {valid_baselines}/{len(channels_to_process)} channels.")
        if valid_baselines == 0: raise ValueError("Baseline calculation failed.")

        # 5. Per-Epoch Event Detection (Parallel) - Detect ALL events first
        print(f"\n--- 5. Detecting ALL Events (Parallel using {NUM_CORES} cores per epoch) ---")
        for epoch_idx, (ep_start, ep_end) in enumerate(epoch_boundaries_samples_orig):
            print(f"\nProcessing Epoch {epoch_idx} (Orig Samples: {ep_start} - {ep_end})...")
            if ep_end <= ep_start: continue
            all_ripple_events_by_epoch[epoch_idx] = {}
            all_spw_events_by_epoch[epoch_idx] = {}
            epoch_tasks = [(ch, channel_region_map.get(ch), ep_start, ep_end, lfp_filepath, n_samples_orig, n_channels, baseline_stats_per_channel[ch], fs_orig) for ch in channels_to_process if ch in baseline_stats_per_channel]
            if not epoch_tasks: continue
            print(f"  Detecting events for {len(epoch_tasks)} channels...")
            epoch_results = Parallel(n_jobs=NUM_CORES, backend="loky")(delayed(_detect_events_worker)(*task) for task in epoch_tasks)
            for ch, ripple_evs, spw_evs in epoch_results:
                 if ripple_evs: all_ripple_events_by_epoch[epoch_idx][ch] = ripple_evs
                 if spw_evs: all_spw_events_by_epoch[epoch_idx][ch] = spw_evs
            print(f"  Finished ALL event detection for Epoch {epoch_idx}.")
            gc.collect()

        # 6. Filter Events by State, Aggregate, Calculate Z-scores and IRIs
        print("\n--- 6. Filtering Events, Calculating Z-Scores & IRIs ---")
        ripple_power_stats_by_state_channel = {state: {} for state in STATES_TO_ANALYZE} # {state: {ch: (mean, std)}}

        for state_code, state_name in STATES_TO_ANALYZE.items():
            print(f"  Processing state: {state_name}")
            ripple_events_by_state_epoch[state_code] = {}
            spw_events_by_state_epoch[state_code] = {}
            ripple_events_by_state_global[state_code] = {}
            spw_events_by_state_global[state_code] = {}

            # First pass: Filter and aggregate globally to get stats for Z-scoring
            print(f"    Aggregating ripple powers for Z-score calculation...")
            all_powers_state_ch = {} # {ch_idx: [powers]}
            for epoch_idx in all_ripple_events_by_epoch:
                for ch_idx, events in all_ripple_events_by_epoch[epoch_idx].items():
                    # Filter ripples for current state
                    filtered_ripples = [ev for ev in events if state_lookup(ev['peak_sample'], fs_orig) == state_code]
                    if filtered_ripples:
                        # Store filtered events temporarily per epoch/channel for this state
                        ripple_events_by_state_epoch[state_code].setdefault(epoch_idx, {})[ch_idx] = filtered_ripples
                        # Aggregate powers globally for this channel/state
                        powers = [ev['peak_power'] for ev in filtered_ripples if 'peak_power' in ev and np.isfinite(ev['peak_power'])]
                        if powers:
                             all_powers_state_ch.setdefault(ch_idx, []).extend(powers)

            # Calculate mean/std for Z-scoring per channel for this state
            print(f"    Calculating Z-score statistics...")
            for ch_idx, powers in all_powers_state_ch.items():
                if len(powers) > 1: # Need >1 point for std dev
                    mean_p = np.mean(powers)
                    std_p = np.std(powers)
                    ripple_power_stats_by_state_channel[state_code][ch_idx] = (mean_p, std_p)
                else:
                    ripple_power_stats_by_state_channel[state_code][ch_idx] = (np.mean(powers) if powers else 0, 0) # Handle single ripple or no ripples

            # Second pass: Calculate Z-scores, IRIs and aggregate globally
            print(f"    Calculating Z-scores, IRIs, and aggregating globally...")
            for epoch_idx in ripple_events_by_state_epoch[state_code]:
                for ch_idx, events in ripple_events_by_state_epoch[state_code][epoch_idx].items():
                    # Sort events within this epoch/channel/state by peak time for IRI
                    events.sort(key=lambda x: x['peak_sample'])
                    ch_mean, ch_std = ripple_power_stats_by_state_channel[state_code].get(ch_idx, (0, 0))
                    previous_peak_sample = -np.inf # Initialize for IRI calculation

                    processed_events = []
                    for ev in events:
                        # Calculate Z-score
                        if ch_std > 0:
                            ev['peak_power_zscore'] = (ev['peak_power'] - ch_mean) / ch_std
                        else:
                            ev['peak_power_zscore'] = 0.0 # Or np.nan? Let's use 0 if std is 0

                        # Calculate IRI
                        current_peak_sample = ev['peak_sample']
                        if previous_peak_sample > -np.inf:
                            iri_samples = current_peak_sample - previous_peak_sample
                            ev['iri_ms'] = (iri_samples / fs_orig) * 1000.0
                        else:
                            ev['iri_ms'] = np.nan # First event in the group
                        previous_peak_sample = current_peak_sample

                        # Add epoch index
                        ev['epoch_idx'] = epoch_idx
                        processed_events.append(ev)

                    # Update epoch dict with processed events
                    ripple_events_by_state_epoch[state_code][epoch_idx][ch_idx] = processed_events
                    # Aggregate processed events globally
                    ripple_events_by_state_global[state_code].setdefault(ch_idx, []).extend(processed_events)

            # Aggregate SPWs globally (no Z-score or IRI for SPWs currently)
            for epoch_idx in all_spw_events_by_epoch:
                 spw_events_by_state_epoch[state_code][epoch_idx] = {}
                 for ch_idx, events in all_spw_events_by_epoch[epoch_idx].items():
                      filtered_events = [ev for ev in events if state_lookup(ev['trough_sample'], fs_orig) == state_code]
                      if filtered_events:
                          spw_events_by_state_epoch[state_code][epoch_idx][ch_idx] = filtered_events
                          for ev in filtered_events: ev['epoch_idx'] = epoch_idx
                          spw_events_by_state_global[state_code].setdefault(ch_idx, []).extend(filtered_events)

            print(f"Finished processing for state: {state_name}")


        # 7. SPW-R Coincidence Check (State-Specific, using processed events)
        print("\n--- 7. Checking SPW-Ripple Coincidence (State-Specific) ---")
        # ... (Logic unchanged, but operates on state-specific global dicts) ...
        ca1_indices = region_channels_df[region_channels_df['acronym'] == 'CA1']['global_channel_index'].values
        for state_code, state_name in STATES_TO_ANALYZE.items():
            ca1_spwr_events_by_state_global[state_code] = {}
            print(f"  Checking for state: {state_name}")
            for ch in ca1_indices:
                 ripples_filt = ripple_events_by_state_global[state_code].get(ch, [])
                 spws_filt = spw_events_by_state_global[state_code].get(ch, [])
                 if not ripples_filt or not spws_filt: continue
                 coincident_ripples = []
                 for r in ripples_filt:
                     is_co, spw_trough = False, None
                     for s in spws_filt:
                         if r['start_sample'] <= s['end_sample'] and r['end_sample'] >= s['start_sample']: is_co, spw_trough = True, s['trough_sample']; break
                     if is_co: r['associated_spw_trough_sample'] = spw_trough; coincident_ripples.append(r)
                 if coincident_ripples:
                      ca1_spwr_events_by_state_global[state_code][ch] = coincident_ripples
                      print(f"    Ch {ch} (CA1): Found {len(coincident_ripples)} {state_name} SPW-R events.")


        # 8. Determine Reference Sites (Globally, based on ALL original ripples/SPW-Rs)
        print("\n--- 8. Determining Reference Sites (Globally, based on ALL events) ---")
        # ... (Logic unchanged) ...
        shanks = region_channels_df['shank_index'].unique(); reference_sites = {}
        _all_ripple_global_temp = {}; _all_spw_global_temp = {}
        for ep_idx in all_ripple_events_by_epoch:
             for ch_idx, evs in all_ripple_events_by_epoch[ep_idx].items(): _all_ripple_global_temp.setdefault(ch_idx, []).extend(evs)
        for ep_idx in all_spw_events_by_epoch:
             for ch_idx, evs in all_spw_events_by_epoch[ep_idx].items(): _all_spw_global_temp.setdefault(ch_idx, []).extend(evs)
        _all_ca1_spwr_global_temp = {}
        for ch in ca1_indices:
             if ch not in _all_ripple_global_temp or ch not in _all_spw_global_temp: continue
             _ripples = _all_ripple_global_temp[ch]; _spws = _all_spw_global_temp[ch]; _coincident = []
             for r in _ripples:
                 is_co = False;
                 for s in _spws:
                     if r['start_sample'] <= s['end_sample'] and r['end_sample'] >= s['start_sample']: is_co = True; break
                 if is_co: _coincident.append(r)
             if _coincident: _all_ca1_spwr_global_temp[ch] = _coincident
        for shank in shanks:
            shank_ch_df = region_channels_df[region_channels_df['shank_index'] == shank]; ref_sites_shank = {}
            for region in target_regions:
                reg_shank_ch = shank_ch_df[shank_ch_df['acronym'] == region];
                if reg_shank_ch.empty: continue
                max_pow, ref_ch = -1, -1
                for _, row in reg_shank_ch.iterrows():
                    ch = int(row['global_channel_index']); evs_ref = []
                    if region == 'CA1':
                         if ch in _all_ca1_spwr_global_temp: evs_ref = _all_ca1_spwr_global_temp[ch]
                    else:
                         if ch in _all_ripple_global_temp: evs_ref = _all_ripple_global_temp[ch]
                    if not evs_ref: continue
                    powers = [e['peak_power'] for e in evs_ref if 'peak_power' in e and np.isfinite(e['peak_power'])]
                    if not powers: continue
                    mean_pow = np.mean(powers)
                    if mean_pow > max_pow: max_pow, ref_ch = mean_pow, ch
                if ref_ch != -1: ref_sites_shank[region] = ref_ch; print(f"  Shank {shank}, Region {region}: Ref Ch {ref_ch} (Avg Power: {max_pow:.2f})")
            if ref_sites_shank: reference_sites[shank] = ref_sites_shank
        del _all_ripple_global_temp, _all_spw_global_temp, _all_ca1_spwr_global_temp

        # --- Epoch-Specific & State-Specific Analyses ---

        # 9. Generate Timestamps per State per Epoch
        print("\n--- 9. Generating Timestamps per State per Epoch ---")
        # ... (Logic unchanged, uses state-specific global events filtered by epoch_idx) ...
        averaged_timestamps_by_state_epoch = {state: {} for state in STATES_TO_ANALYZE}
        for state_code, state_name in STATES_TO_ANALYZE.items():
            print(f"  Processing state: {state_name}")
            epochs_with_state_events = set()
            if state_code in ripple_events_by_state_epoch: epochs_with_state_events.update(ripple_events_by_state_epoch[state_code].keys())
            if state_code in spw_events_by_state_epoch: epochs_with_state_events.update(spw_events_by_state_epoch[state_code].keys())

            for epoch_idx in sorted(list(epochs_with_state_events)):
                 averaged_timestamps_by_state_epoch[state_code][epoch_idx] = {}
                 for region in target_regions:
                     regional_ref_indices = [sites[region] for shank, sites in reference_sites.items() if region in sites]
                     if not regional_ref_indices: continue
                     epoch_state_region_times = []
                     event_type = "Unknown"
                     if region == 'CA1':
                         event_type = "SPW Trough (SPW-R)"
                         for ch in regional_ref_indices:
                             ch_spwr_events = ca1_spwr_events_by_state_global[state_code].get(ch, [])
                             epoch_ch_events = [ev for ev in ch_spwr_events if ev.get('epoch_idx') == epoch_idx and 'associated_spw_trough_sample' in ev]
                             epoch_state_region_times.extend([ev['associated_spw_trough_sample'] for ev in epoch_ch_events])
                     elif region in ['CA2', 'CA3']:
                          event_type = "Ripple Peak"
                          for ch in regional_ref_indices:
                              ch_ripple_events = ripple_events_by_state_global[state_code].get(ch, [])
                              epoch_ch_events = [ev for ev in ch_ripple_events if ev.get('epoch_idx') == epoch_idx]
                              epoch_state_region_times.extend([ev['peak_sample'] for ev in epoch_ch_events])

                     if epoch_state_region_times:
                          pooled_times = np.sort(np.unique(epoch_state_region_times))
                          averaged_timestamps_by_state_epoch[state_code][epoch_idx][region] = pooled_times
                          print(f"    Epoch {epoch_idx}, State {state_name}, Region {region}: Generated {len(pooled_times)} pooled {event_type} timestamps.")


        # 10. Co-occurrence Detection per State per Epoch (CA2 and CA3 ref)
        print("\n--- 10. Detecting Co-occurring Ripples per State per Epoch ---")
        # ... (Logic expanded to loop through ref_regions CA2 and CA3) ...
        cooccurrence_results_by_state_epoch = {state: {} for state in STATES_TO_ANALYZE}
        window_samples = int(COOCCURRENCE_WINDOW_MS * fs_orig / 1000)
        cooccurrence_pairs = {'CA2': ['CA1', 'CA3'], 'CA3': ['CA1', 'CA2']}

        for state_code, state_name in STATES_TO_ANALYZE.items():
             print(f"  Processing state: {state_name}")
             cooccurrence_results_by_state_epoch[state_code] = {}
             for epoch_idx in averaged_timestamps_by_state_epoch.get(state_code, {}):
                 print(f"    Processing co-occurrence for Epoch {epoch_idx}...")
                 cooccurrence_results_by_state_epoch[state_code][epoch_idx] = {}

                 for ref_region, target_check_regions in cooccurrence_pairs.items(): # Loop CA2/CA3 ref
                     if ref_region not in averaged_timestamps_by_state_epoch[state_code][epoch_idx]: continue
                     ref_timestamps_epoch_state = averaged_timestamps_by_state_epoch[state_code][epoch_idx][ref_region]
                     if len(ref_timestamps_epoch_state) == 0: continue
                     print(f"      Ref: {ref_region} ({len(ref_timestamps_epoch_state)} events)")
                     cooccurrence_results_by_state_epoch[state_code][epoch_idx][ref_region] = {}
                     target_site_indices = {tr: [sites[tr] for s, sites in reference_sites.items() if tr in sites][0] for tr in target_check_regions if any(tr in sites for s, sites in reference_sites.items())}

                     for target_reg, target_ch in target_site_indices.items():
                         target_event_times_epoch_state = []
                         if target_reg == 'CA1': ch_spwr_events = ca1_spwr_events_by_state_global[state_code].get(target_ch, []) ; target_event_times_epoch_state = np.array([ev['peak_sample'] for ev in ch_spwr_events if ev.get('epoch_idx') == epoch_idx])
                         elif target_reg in ['CA2', 'CA3']: ch_ripple_events = ripple_events_by_state_global[state_code].get(target_ch, []); target_event_times_epoch_state = np.array([ev['peak_sample'] for ev in ch_ripple_events if ev.get('epoch_idx') == epoch_idx])
                         if len(target_event_times_epoch_state) == 0: cooccurrence_results_by_state_epoch[state_code][epoch_idx][ref_region][target_reg] = {'count': 0, 'details': []}; continue
                         co_count, co_details = 0, []
                         target_event_times_epoch_state.sort()
                         for ref_t in ref_timestamps_epoch_state:
                             lb, ub = ref_t - window_samples, ref_t + window_samples; start_i = np.searchsorted(target_event_times_epoch_state, lb); end_i = np.searchsorted(target_event_times_epoch_state, ub, side='right')
                             indices = np.arange(start_i, end_i)
                             if len(indices) > 0: co_count += 1; co_details.append({'ref_time_sample': ref_t, 'target_event_times': target_event_times_epoch_state[indices].tolist()})
                         print(f"        Epoch {epoch_idx}, State {state_name}: Found {co_count} co-occurrences {ref_region} -> {target_reg}.")
                         cooccurrence_results_by_state_epoch[state_code][epoch_idx][ref_region][target_reg] = {'count': co_count, 'details': co_details}


        # 11. Spectrogram Calculation per State per Epoch (Parallel)
        print(f"\n--- 11. Calculating Spectrograms per State per Epoch (Parallel using {NUM_CORES} cores) ---")
        # ... (Logic unchanged, uses state/epoch specific timestamps) ...
        averaged_spectrograms_by_state_epoch = {state: {} for state in STATES_TO_ANALYZE}
        for state_code, state_name in STATES_TO_ANALYZE.items():
             print(f"\n  Processing spectrograms for state: {state_name}")
             averaged_spectrograms_by_state_epoch[state_code] = {}
             for epoch_idx in averaged_timestamps_by_state_epoch.get(state_code, {}):
                 print(f"    Processing Epoch {epoch_idx}...")
                 averaged_spectrograms_by_state_epoch[state_code][epoch_idx] = {}
                 for region, timestamps_epoch_state in averaged_timestamps_by_state_epoch[state_code][epoch_idx].items():
                     regional_ref_indices = [sites[region] for shank, sites in reference_sites.items() if region in sites]
                     if not regional_ref_indices or len(timestamps_epoch_state) == 0: continue
                     spec_ch = regional_ref_indices[0]
                     print(f"      Calculating {len(timestamps_epoch_state)} spectrograms for Region {region} (Ref Ch {spec_ch})...")
                     spec_tasks = [(ts, spec_ch, lfp_filepath, n_samples_orig, n_channels, fs_orig) for ts in timestamps_epoch_state]
                     region_specs_results = Parallel(n_jobs=NUM_CORES, backend="loky")(delayed(_calculate_spectrogram_worker)(*task) for task in spec_tasks)
                     valid_specs = [s for s in region_specs_results if s is not None and not np.isnan(s).all()]
                     if valid_specs:
                         try:
                             avg_spec = np.nanmean(np.stack(valid_specs, axis=0), axis=0)
                             averaged_spectrograms_by_state_epoch[state_code][epoch_idx][region] = avg_spec
                             print(f"      Epoch {epoch_idx}, State {state_name}, Region {region}: Averaged {len(valid_specs)}/{len(timestamps_epoch_state)} spectrograms.")
                         except MemoryError: warnings.warn(f"MemoryError stacking spectrograms.")
                         except Exception as e: warnings.warn(f"Error averaging spectrograms: {e}")
                     else: print(f"      Epoch {epoch_idx}, State {state_name}, Region {region}: No valid spectrograms generated.")
                     del region_specs_results, valid_specs; gc.collect()


        # 11.5 Save Detailed Ripple Events to CSV
        print("\n--- 11.5 Saving Detailed Ripple Events to CSV ---")
        csv_columns = ['epoch_idx', 'channel_idx', 'region', 'state_code', 'state_name',
                       'start_sample', 'peak_sample', 'end_sample', 'trough_sample',
                       'duration_ms', 'peak_power', 'peak_power_zscore', 'iri_ms',
                       'is_spwr', 'associated_spw_trough_sample']

        for state_code, state_name in STATES_TO_ANALYZE.items():
            all_state_epoch_ripple_data = []
            state_suffix = f"_{state_name}"
            print(f"  Preparing CSV data for state: {state_name}")

            # Iterate through epochs that have ripple data for this state
            for epoch_idx in ripple_events_by_state_epoch.get(state_code, {}):
                for ch_idx, events in ripple_events_by_state_epoch[state_code][epoch_idx].items():
                    ch_region = channel_region_map.get(ch_idx, 'Unknown')
                    # Check if these ripples are SPW-Rs for this state
                    spwr_peaks_state_ch = set()
                    if ch_idx in ca1_spwr_events_by_state_global.get(state_code, {}):
                        spwr_peaks_state_ch = {ev['peak_sample'] for ev in ca1_spwr_events_by_state_global[state_code][ch_idx]}

                    for ev in events:
                        row = {
                            'epoch_idx': epoch_idx,
                            'channel_idx': ch_idx,
                            'region': ch_region,
                            'state_code': state_code,
                            'state_name': state_name,
                            'start_sample': ev.get('start_sample', np.nan),
                            'peak_sample': ev.get('peak_sample', np.nan),
                            'end_sample': ev.get('end_sample', np.nan),
                            'trough_sample': ev.get('trough_sample', np.nan),
                            'duration_ms': ev.get('duration_ms', np.nan),
                            'peak_power': ev.get('peak_power', np.nan),
                            'peak_power_zscore': ev.get('peak_power_zscore', np.nan),
                            'iri_ms': ev.get('iri_ms', np.nan),
                            'is_spwr': ev.get('peak_sample', -1) in spwr_peaks_state_ch if ch_region == 'CA1' else False,
                            'associated_spw_trough_sample': ev.get('associated_spw_trough_sample', np.nan) if ch_region == 'CA1' else np.nan
                        }
                        all_state_epoch_ripple_data.append(row)

            if all_state_epoch_ripple_data:
                df_state_epoch = pd.DataFrame(all_state_epoch_ripple_data)
                csv_filename = output_path / f'{output_filename_base}_ripple_details{state_suffix}.csv'
                try:
                    df_state_epoch.to_csv(csv_filename, index=False, columns=csv_columns, na_rep='NaN')
                    print(f"  Saved detailed ripple CSV: {csv_filename}")
                except Exception as e:
                    print(f"  Error saving CSV for state {state_name}: {e}")
            else:
                print(f"  No ripple events found for state {state_name} to save to CSV.")


        # 12. Save NPY Results
        print("\n--- 12. Saving NPY Results ---")
        np.save(output_path / f'{output_filename_base}_ripple_references_global.npy', reference_sites, allow_pickle=True)
        region_channels_df.to_csv(output_path / f'{output_filename_base}_ripple_analyzed_channels.csv', index=False)
        # Save state-specific results
        for state_code, state_name in STATES_TO_ANALYZE.items():
             state_suffix = f"_{state_name}"
             np.save(output_path / f'{output_filename_base}_ripple_events{state_suffix}_global.npy', ripple_events_by_state_global[state_code], allow_pickle=True)
             np.save(output_path / f'{output_filename_base}_spw_events{state_suffix}_global.npy', spw_events_by_state_global[state_code], allow_pickle=True)
             np.save(output_path / f'{output_filename_base}_ca1_spwr_events{state_suffix}_global.npy', ca1_spwr_events_by_state_global[state_code], allow_pickle=True)
             np.save(output_path / f'{output_filename_base}_ripple_events{state_suffix}_by_epoch.npy', ripple_events_by_state_epoch[state_code], allow_pickle=True)
             np.save(output_path / f'{output_filename_base}_spw_events{state_suffix}_by_epoch.npy', spw_events_by_state_epoch[state_code], allow_pickle=True)
             np.save(output_path / f'{output_filename_base}_ripple_timestamps{state_suffix}_by_epoch.npy', averaged_timestamps_by_state_epoch[state_code], allow_pickle=True)
             np.save(output_path / f'{output_filename_base}_ripple_cooccurrence{state_suffix}_by_epoch.npy', cooccurrence_results_by_state_epoch[state_code], allow_pickle=True)
             np.save(output_path / f'{output_filename_base}_ripple_avg_spectrograms{state_suffix}_by_epoch.npy', averaged_spectrograms_by_state_epoch[state_code], allow_pickle=True)
        print(f"NPY results saved to {output_path} with prefix '{output_filename_base}' (State-Specific)")
        # Optionally save ALL detected events
        # np.save(output_path / f'{output_filename_base}_ripple_events_ALL_by_epoch.npy', all_ripple_events_by_epoch, allow_pickle=True)
        # np.save(output_path / f'{output_filename_base}_spw_events_ALL_by_epoch.npy', all_spw_events_by_epoch, allow_pickle=True)

    except Exception as e:
        print(f"\n!!!!!!!!!!!!!!!! ERROR PROCESSING FILE: {lfp_filepath.name} !!!!!!!!!!!!!!")
        print(f"Error details: {e}"); traceback.print_exc()
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
    finally:
        print("\n--- Cleaning up ---"); gc.collect()
        print("--- Analysis Complete ---")


# --- Script Execution ---
if __name__ == "__main__":
    # --- Use Tkinter to select files ---
    root = Tk(); root.withdraw(); root.attributes("-topmost", True)
    print("Select LFP binary file (*.lf.bin)...")
    lfp_file_str = filedialog.askopenfilename(title="Select LFP Binary File", filetypes=[("LFP binary files", "*.lf.bin"), ("All files", "*.*")])
    if not lfp_file_str: print("Cancelled."); exit()
    LFP_FILE = Path(lfp_file_str)
    print("Select corresponding Meta file (*.meta)...")
    meta_file_str = filedialog.askopenfilename(title="Select Meta File", initialdir=LFP_FILE.parent, initialfile=LFP_FILE.stem+".meta", filetypes=[("Meta files", "*.meta"), ("All files", "*.*")])
    if not meta_file_str: print("Cancelled."); exit()
    META_FILE = Path(meta_file_str)
    print("Select Channel Info CSV file...")
    channel_info_file_str = filedialog.askopenfilename(title="Select Channel Info CSV", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
    if not channel_info_file_str: print("Cancelled."); exit()
    CHANNEL_INFO_FILE = Path(channel_info_file_str)
    print("Select Sleep State file (*_sleep_states*.npy) (Required for NREM/Awake filtering)...")
    sleep_state_file_str = filedialog.askopenfilename(title="Select Sleep State File (Required)", filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")])
    if not sleep_state_file_str: print("Sleep state file required for this analysis. Exiting."); exit() # Make mandatory
    SLEEP_STATE_FILE = Path(sleep_state_file_str); print(f"Selected Sleep State file: {SLEEP_STATE_FILE.name}")
    print("Select Epoch Boundaries file (*_epoch_boundaries*.npy) (Optional)...")
    epoch_boundaries_file_str = filedialog.askopenfilename(title="Select Epoch Boundaries File (Optional)", filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")])
    EPOCH_BOUNDARIES_FILE = Path(epoch_boundaries_file_str) if epoch_boundaries_file_str else None; print(f"Selected Epoch Boundaries file: {EPOCH_BOUNDARIES_FILE.name if EPOCH_BOUNDARIES_FILE else 'None'}")
    print("Select Output directory...")
    output_dir_str = filedialog.askdirectory(title="Select Output Directory")
    if not output_dir_str: print("Cancelled."); exit()
    OUTPUT_DIRECTORY = Path(output_dir_str)
    root.destroy()

    # --- Validation ---
    if not LFP_FILE.is_file(): print(f"Error: LFP file not found: {LFP_FILE}"); exit()
    if not META_FILE.is_file(): print(f"Error: Meta file not found: {META_FILE}"); exit()
    if not CHANNEL_INFO_FILE.is_file(): print(f"Error: Channel Info file not found: {CHANNEL_INFO_FILE}"); exit()
    if not SLEEP_STATE_FILE.is_file(): print(f"Error: Sleep State file not found: {SLEEP_STATE_FILE}"); exit() # Check mandatory file
    if EPOCH_BOUNDARIES_FILE and not EPOCH_BOUNDARIES_FILE.is_file(): print(f"Error: Epoch Boundaries file not found: {EPOCH_BOUNDARIES_FILE}"); exit()
    try: OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True); print(f"Output directory: {OUTPUT_DIRECTORY}")
    except Exception as e: print(f"Error creating output directory: {e}"); exit()

    # --- Run Analysis ---
    print(f"\n{'='*50}\nStarting processing for: {LFP_FILE.name}\n{'='*50}")
    run_ripple_analysis(lfp_filepath=LFP_FILE, meta_filepath=META_FILE, channel_info_filepath=CHANNEL_INFO_FILE,
                        sleep_state_filepath=SLEEP_STATE_FILE, epoch_boundaries_filepath=EPOCH_BOUNDARIES_FILE,
                        output_dir=OUTPUT_DIRECTORY)
    print(f"\n{'='*50}\nProcessing complete.\n{'='*50}")

