# -*- coding: utf-8 -*-
"""
Created on Fri May 30 11:50:46 2025

@author: HT_bo
"""

import numpy as np
import pandas as pd
from scipy import signal # For signal.decimate, signal.firwin, signal.filtfilt, signal.hilbert, signal.find_peaks
from scipy.signal import butter, filtfilt as scipy_filtfilt, welch, decimate as scipy_decimate # Explicit imports for PSD function
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
from DemoReadSGLXData.readSGLX import readMeta # Assuming this is in the PYTHONPATH or same directory

# --- Import for Plotting ---
import matplotlib
matplotlib.use('Agg') # Use Agg backend for non-interactive plotting
import matplotlib.pyplot as plt

# --- Configuration: Reference Channel Selection Mode ---
REFERENCE_SELECTION_MODE = "manual" # Options: "manual" or "automatic"
print(f"Configuring reference channel selection mode: {REFERENCE_SELECTION_MODE}")


# --- Constants ---

RIPPLE_FILTER_LOWCUT = 100.0
RIPPLE_FILTER_HIGHCUT = 250.0
RIPPLE_POWER_LP_CUTOFF = 55.0
RIPPLE_DETECTION_SD_THRESHOLD = 4.0
RIPPLE_EXPANSION_SD_THRESHOLD = 2.0
RIPPLE_MIN_DURATION_MS = 20.0
RIPPLE_MAX_DURATION_MS = 400.0
RIPPLE_MERGE_GAP_MS = 15.0

SPW_FILTER_LOWCUT = 5.0
SPW_FILTER_HIGHCUT = 40.0
SPW_DETECTION_SD_THRESHOLD = 2.5
SPW_MIN_DURATION_MS = 20.0
SPW_MAX_DURATION_MS = 400.0

COOCCURRENCE_WINDOW_MS = 60.0

SPECTROGRAM_WINDOW_MS = 200
SPECTROGRAM_FREQS = np.arange(10, 300, 2)

PLOT_LFP_EXAMPLES = True
LFP_PLOT_WINDOW_MS = 200
MAX_PLOTS_PER_REF_SITE_STATE = 3

DOWNSAMPLING_FACTOR = 2
print(f"Configuring analysis with downsampling factor: {DOWNSAMPLING_FACTOR}")

NUM_CORES = max(1, multiprocessing.cpu_count() // 2)
print(f"Configuring parallel processing to use {NUM_CORES} cores.")

# STATES_TO_ANALYZE = {0: "Awake", 1: "NREM"}
STATES_TO_ANALYZE = { 1: "NREM"}
print(f"Analyzing events separately for states: {STATES_TO_ANALYZE}")

# --- Helper Function for Voltage Scaling ---
def get_voltage_scaling_factor(meta):
    # ... (unchanged from previous version) ...
    try:
        v_max = float(meta['imAiRangeMax'])
        i_max = int(meta['imMaxInt'])
        probe_type = int(meta.get('imDatPrb_type', 0))
        lfp_gain = 250.0 
        if probe_type in [21, 24, 2013]: 
            lfp_gain = 80.0
        if i_max == 0 or lfp_gain == 0: raise ValueError("Imax or LFP gain is zero.")
        scaling_factor_uv = (v_max * 1e6) / (i_max * lfp_gain)
        return scaling_factor_uv
    except KeyError as e: warnings.warn(f"Warning: Missing key in metadata for voltage scaling: {e}. Returning None."); return None
    except ValueError as e: warnings.warn(f"Warning: Invalid value in metadata for voltage scaling: {e}. Returning None."); return None


# --- Helper Function for MANUAL Reference Channel Selection (All Regions) - OVERLAID PLOTTING ---
def select_ref_channel_by_manual_psd_visualization(
    lfp_filepath,
    meta_filepath,
    n_total_channels,
    n_total_samples,
    region_channel_indices_on_shank, 
    current_shank_idx, 
    current_region_name, 
    output_dir_psd_plots, 
    readMeta_func, 
    get_voltage_scaling_factor_func, 
    target_psd_fs=1250.0,
    plot_freq_range=(1.0, 200.0) 
):
    print(f"\n--- Manual {current_region_name} Reference Selection for Shank {current_shank_idx} ---")
    # ... (Data loading and PSD calculation logic remains the same as previous version) ...
    selected_channel_idx = None
    all_channel_psds = {}
    all_channel_freqs = {}
    valid_channels_for_shank_region = [] # Store channels for which PSD was successfully computed

    meta = readMeta_func(meta_filepath)
    if meta is None:
        warnings.warn(f"PSD Sel (Manual {current_region_name}): Could not read meta file {meta_filepath} for shank {current_shank_idx}. Cannot select ref channel.")
        return None

    fs_orig_str = meta.get('imSampRate') or meta.get('niSampRate')
    if fs_orig_str is None:
        warnings.warn(f"PSD Sel (Manual {current_region_name}): Sampling rate not found for shank {current_shank_idx}. Cannot select ref channel.")
        return None
    fs_orig = float(fs_orig_str)
    uv_scale_factor = get_voltage_scaling_factor_func(meta)

    if not region_channel_indices_on_shank:
        warnings.warn(f"PSD Sel (Manual {current_region_name}): No channel indices provided for PSD selection on shank {current_shank_idx}.")
        return None
    if fs_orig <= 0:
        warnings.warn(f"PSD Sel (Manual {current_region_name}): Invalid original sampling rate ({fs_orig} Hz) for shank {current_shank_idx}. Cannot select ref channel.")
        return None

    q = 1
    current_fs_for_psd = fs_orig
    if fs_orig > target_psd_fs:
        q_float = fs_orig / target_psd_fs
        q = int(round(q_float))
        if q <= 0: q = 1
    if q > 1: current_fs_for_psd = fs_orig / q
    else: current_fs_for_psd = fs_orig

    duration_to_load_sec = 60 
    samples_to_load_orig_fs = int(duration_to_load_sec * fs_orig)
    if n_total_samples <= 0:
        warnings.warn(f"PSD Sel (Manual {current_region_name}): Total samples {n_total_samples} for shank {current_shank_idx}. Cannot load data.")
        return None
    start_sample_offset = max(0, (n_total_samples // 2) - (samples_to_load_orig_fs // 2))
    samples_to_load_orig_fs = min(samples_to_load_orig_fs, n_total_samples - start_sample_offset)
    if samples_to_load_orig_fs < current_fs_for_psd * 5 * q : 
        warnings.warn(f"PSD Sel (Manual {current_region_name}): Not enough data for shank {current_shank_idx}. Required at orig_fs: {int(current_fs_for_psd * 5 * q)}, available: {samples_to_load_orig_fs}")
        return None
            
    lfp_memmap_psd = None
    try:
        lfp_memmap_psd = np.memmap(lfp_filepath, dtype='int16', mode='r', shape=(n_total_samples, n_total_channels))
        for ch_idx in region_channel_indices_on_shank:
            if not (0 <= ch_idx < n_total_channels):
                warnings.warn(f"PSD Sel (Manual {current_region_name}): Ch {ch_idx} out of bounds for shank {current_shank_idx}. Skipping.")
                continue
            lfp_segment_raw = lfp_memmap_psd[start_sample_offset : start_sample_offset + samples_to_load_orig_fs, ch_idx].astype(np.float64)
            lfp_segment_scaled = (lfp_segment_raw * uv_scale_factor) if uv_scale_factor is not None else lfp_segment_raw
            del lfp_segment_raw

            lfp_processed_for_psd = lfp_segment_scaled
            if q > 1: 
                if len(lfp_segment_scaled) > q * 20: 
                    lfp_processed_for_psd = scipy_decimate(lfp_segment_scaled, q=q, ftype='fir', zero_phase=True)
                else:
                    warnings.warn(f"PSD Sel (Manual {current_region_name}): Ch {ch_idx} on shank {current_shank_idx}: Not enough data for decimation (q={q}). Using data at {fs_orig} Hz for PSD.")
            del lfp_segment_scaled
            
            nperseg_welch = int(2 * current_fs_for_psd) 
            if len(lfp_processed_for_psd) < nperseg_welch:
                warnings.warn(f"PSD Sel (Manual {current_region_name}): Ch {ch_idx} on shank {current_shank_idx}: Not enough data for Welch PSD. Skipping.")
                continue
            try:
                frequencies, psd_values = welch(lfp_processed_for_psd, fs=current_fs_for_psd, nperseg=nperseg_welch)
            except ValueError as ve_welch:
                warnings.warn(f"PSD Sel (Manual {current_region_name}): Ch {ch_idx} on shank {current_shank_idx}: Error Welch: {ve_welch}. Skipping.")
                continue
            del lfp_processed_for_psd
            psd_db = 10 * np.log10(psd_values + np.finfo(float).eps) 
            zscored_psd_db = stats.zscore(psd_db) if len(psd_db) >= 2 else psd_db # Z-score entire PSD for this channel
            all_channel_psds[ch_idx] = zscored_psd_db
            all_channel_freqs[ch_idx] = frequencies
            valid_channels_for_shank_region.append(ch_idx) # Add to list of successfully processed channels
            gc.collect()
    except Exception as e:
        warnings.warn(f"PSD Sel (Manual {current_region_name}): Error processing data for shank {current_shank_idx}: {e}")
        return None
    finally:
        if lfp_memmap_psd is not None and hasattr(lfp_memmap_psd, '_mmap') and lfp_memmap_psd._mmap is not None:
            try: lfp_memmap_psd._mmap.close()
            except Exception: pass
        gc.collect()

    if not valid_channels_for_shank_region:
        warnings.warn(f"PSD Sel (Manual {current_region_name}): No valid channels processed to plot for shank {current_shank_idx}.")
        return None
    
    valid_channels_for_shank_region.sort() # Ensure consistent plotting order

    # --- OVERLAID PLOTTING (Matplotlib) ---
    plt.figure(figsize=(12, 8)) 
    plt.suptitle(f"Manual {current_region_name} Ref Channel Selection: Shank {current_shank_idx}", fontsize=16, y=0.98)
    plt.title(f"Target Fs for PSD: {current_fs_for_psd:.2f} Hz. Z-scored PSDs ({plot_freq_range[0]}-{plot_freq_range[1]} Hz) Overlaid.", fontsize=10)
    
    plot_low_freq, plot_high_freq = plot_freq_range
    ax = plt.gca()

    # Define a list of colors to cycle through for better distinction if many channels
    # colors = plt.cm.get_cmap('tab20', len(valid_channels_for_shank_region)) # 'tab20' has 20 distinct colors
    # Or for more channels, plt.cm.viridis or other continuous maps with np.linspace

    for i, ch_idx_plot in enumerate(valid_channels_for_shank_region):
        if ch_idx_plot in all_channel_psds:
            freqs_plot = all_channel_freqs[ch_idx_plot]
            psd_data_plot = all_channel_psds[ch_idx_plot] # This is Z-scored
            
            actual_plot_high_freq = min(plot_high_freq, (current_fs_for_psd / 2.0) * 0.99)
            actual_plot_low_freq = plot_low_freq if plot_low_freq < actual_plot_high_freq else actual_plot_high_freq * 0.1
            freq_mask_for_plot = (freqs_plot >= actual_plot_low_freq) & (freqs_plot <= actual_plot_high_freq)
            
            if np.any(freq_mask_for_plot):
                ax.plot(freqs_plot[freq_mask_for_plot], 
                        psd_data_plot[freq_mask_for_plot], 
                        label=f'Ch {ch_idx_plot}', 
                        linewidth=1, 
                        alpha=0.6) # Removed color=colors(i) for simplicity, default cycling will occur
            else:
                warnings.warn(f"PSD Sel (Manual {current_region_name}): Ch {ch_idx_plot} on shank {current_shank_idx} had no PSD data in the plotting range [{actual_plot_low_freq:.2f}-{actual_plot_high_freq:.2f}] Hz.")
    
    ax.set_xlabel("Frequency (Hz)", fontsize=12)
    ax.set_ylabel("Z-scored Power Spectral Density (dB)", fontsize=12)
    
    if valid_channels_for_shank_region:
        # Adjust legend: more columns if many channels, or place outside
        num_ch = len(valid_channels_for_shank_region)
        ncol_legend = 1
        if num_ch > 10: ncol_legend = 2
        if num_ch > 20: ncol_legend = 3
        if num_ch > 30: # Legend might become too large, consider placing outside
            # ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small', title="Channels", ncol=1)
            ax.legend(loc='best', fontsize='x-small', title="Channels", ncol=ncol_legend)
        else:
            ax.legend(loc='upper right', fontsize='small', title="Channels", ncol=ncol_legend)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.grid(True, linestyle=':', alpha=0.5) 
    plt.xlim(actual_plot_low_freq, actual_plot_high_freq)

    plt.tight_layout(rect=[0, 0.03, 1, 0.93]) 
    plot_savename = output_dir_psd_plots / f"MANUAL_SELECTION_shank{current_shank_idx}_region{current_region_name}_PSDs_OVERLAID.png" # Added _OVERLAID
    try:
        plt.savefig(plot_savename)
        print(f"\nPLOT SAVED: {plot_savename}")
        print(f"Please open this image to inspect {current_region_name} channel PSDs for shank {current_shank_idx}.")
    except Exception as e_save: print(f"Error saving PSD plot: {e_save}")
    plt.close() 
    # --- END OVERLAID PLOTTING ---

    print(f"Available {current_region_name} channels processed on shank {current_shank_idx}: {sorted(valid_channels_for_shank_region)}")
    while True: # Prompt for manual input
        try:
            user_input = input(f"Enter the selected {current_region_name} reference channel index for shank {current_shank_idx} (from list above, or 'skip'): ")
            if user_input.strip().lower() == 'skip':
                warnings.warn(f"PSD Sel (Manual {current_region_name}): User skipped selection for shank {current_shank_idx}.")
                selected_channel_idx = None; break
            potential_idx = int(user_input)
            if potential_idx in valid_channels_for_shank_region:
                selected_channel_idx = potential_idx
                print(f"  PSD Sel (Manual {current_region_name}): Selected ref channel for shank {current_shank_idx}: {selected_channel_idx}")
                break
            else: print(f"Error: Channel {potential_idx} is not valid for {current_region_name} on shank {current_shank_idx}.")
        except ValueError: print("Error: Invalid input. Please enter a number or 'skip'.")
        except EOFError: warnings.warn(f"PSD Sel (Manual {current_region_name}): EOFError for shank {current_shank_idx}. No selection."); return None
    return selected_channel_idx


# --- Helper Function for AUTOMATED CA1 Reference Channel Selection (PSD Metric) ---
def select_ca1_ref_by_automated_psd_metric(
    # ... (unchanged from previous version) ...
    lfp_filepath, meta_filepath, n_total_channels, n_total_samples,
    shank_ca1_channel_indices, current_shank_idx, 
    readMeta_func, get_voltage_scaling_factor_func, target_psd_fs=1250.0
):
    print(f"--- Automated CA1 PSD Metric Selection for Shank {current_shank_idx} ---")
    selected_channel_idx = None
    max_psd_metric = -np.inf
    meta = readMeta_func(meta_filepath)
    if meta is None: warnings.warn(f"Auto PSD Sel CA1: No meta for shank {current_shank_idx}."); return None
    fs_orig_str = meta.get('imSampRate') or meta.get('niSampRate')
    if fs_orig_str is None: warnings.warn(f"Auto PSD Sel CA1: No Fs for shank {current_shank_idx}."); return None
    fs_orig = float(fs_orig_str)
    uv_scale_factor = get_voltage_scaling_factor_func(meta)
    if not shank_ca1_channel_indices: warnings.warn(f"Auto PSD Sel CA1: No CA1 channels for shank {current_shank_idx}."); return None
    if fs_orig <= 0: warnings.warn(f"Auto PSD Sel CA1: Invalid Fs for shank {current_shank_idx}."); return None

    q = 1; current_fs_for_psd = fs_orig
    if fs_orig > target_psd_fs:
        q = int(round(fs_orig / target_psd_fs)); q = max(1, q)
    if q > 1: current_fs_for_psd = fs_orig / q
    else: current_fs_for_psd = fs_orig

    duration_to_load_sec = 60; samples_to_load_orig_fs = int(duration_to_load_sec * fs_orig)
    if n_total_samples <= 0: warnings.warn(f"Auto PSD Sel CA1: Total samples {n_total_samples} for shank {current_shank_idx}."); return None
    start_sample_offset = max(0, (n_total_samples // 2) - (samples_to_load_orig_fs // 2))
    samples_to_load_orig_fs = min(samples_to_load_orig_fs, n_total_samples - start_sample_offset)
    if samples_to_load_orig_fs < current_fs_for_psd * 5 * q : warnings.warn(f"Auto PSD Sel CA1: Not enough data for shank {current_shank_idx}."); return None
            
    lfp_memmap_psd = None
    try:
        lfp_memmap_psd = np.memmap(lfp_filepath, dtype='int16', mode='r', shape=(n_total_samples, n_total_channels))
        for ch_idx in shank_ca1_channel_indices:
            if not (0 <= ch_idx < n_total_channels): continue
            lfp_segment_raw = lfp_memmap_psd[start_sample_offset : start_sample_offset + samples_to_load_orig_fs, ch_idx].astype(np.float64)
            lfp_segment_scaled = (lfp_segment_raw * uv_scale_factor) if uv_scale_factor is not None else lfp_segment_raw
            del lfp_segment_raw

            lfp_processed = lfp_segment_scaled
            if q > 1: 
                if len(lfp_segment_scaled) > q * 20: lfp_processed = scipy_decimate(lfp_segment_scaled, q=q, ftype='fir', zero_phase=True)
                else: warnings.warn(f"Auto PSD Sel CA1: Ch {ch_idx} shank {current_shank_idx}: Not enough for decimation."); 
            del lfp_segment_scaled

            lfp_ripple_filtered = apply_fir_filter(lfp_processed, RIPPLE_FILTER_LOWCUT, RIPPLE_FILTER_HIGHCUT, current_fs_for_psd)
            del lfp_processed
            if lfp_ripple_filtered is None: continue
            
            nperseg_welch = int(2 * current_fs_for_psd) 
            if len(lfp_ripple_filtered) < nperseg_welch: continue
            try:
                frequencies, psd_values = welch(lfp_ripple_filtered, fs=current_fs_for_psd, nperseg=nperseg_welch)
            except ValueError: continue
            del lfp_ripple_filtered
            psd_db = 10 * np.log10(psd_values + np.finfo(float).eps) 
            
            metric_freq_mask = (frequencies >= RIPPLE_FILTER_LOWCUT) & (frequencies <= RIPPLE_FILTER_HIGHCUT)
            if not np.any(metric_freq_mask): continue
            psd_db_in_band = psd_db[metric_freq_mask]
            zscored_psd_db_in_band = stats.zscore(psd_db_in_band) if len(psd_db_in_band) >= 2 else psd_db_in_band
            metric = np.mean(zscored_psd_db_in_band)
            if not np.isnan(metric) and metric > max_psd_metric:
                max_psd_metric = metric; selected_channel_idx = ch_idx
            gc.collect()
    except Exception as e: warnings.warn(f"Auto PSD Sel CA1: Error processing shank {current_shank_idx}: {e}"); return None
    finally:
        if lfp_memmap_psd is not None and hasattr(lfp_memmap_psd, '_mmap') and lfp_memmap_psd._mmap is not None:
            try: lfp_memmap_psd._mmap.close()
            except Exception: pass
        gc.collect()
    if selected_channel_idx is not None: print(f"  Auto PSD Sel CA1: Selected ref ch for shank {current_shank_idx}: {selected_channel_idx} (Metric: {max_psd_metric:.2f})")
    else: print(f"  Auto PSD Sel CA1: Could not select ref ch for shank {current_shank_idx} (Max metric: {max_psd_metric}).")
    return selected_channel_idx

# ... (The rest of the script: Data Loading, Signal Processing, Worker functions, Main run_ripple_analysis structure, and __main__ block remain UNCHANGED from the previous version where manual/auto modes were introduced.) ...
# ... I am omitting them here for brevity as they are extensive and their internal logic does not change with this specific plotting style update. ...
# ... The `run_ripple_analysis` function will correctly call the updated 
# `select_ref_channel_by_manual_psd_visualization` if REFERENCE_SELECTION_MODE is "manual".

# --- Data Loading Functions (from original script) ---
def load_lfp_data_memmap(file_path, meta_path, data_type='int16'):
    """Loads LFP data using memory-mapping and extracts scaling factor."""
    sampling_rate = None; data = None; n_channels = 0; n_samples = 0; uv_scale_factor_val = None 
    try:
        meta = readMeta(meta_path)  
        n_channels = int(meta['nSavedChans'])
        if 'imSampRate' in meta: sampling_rate = float(meta['imSampRate'])
        elif 'niSampRate' in meta: sampling_rate = float(meta['niSampRate'])
        else: print(f"Error: Sampling rate key not found."); return None, None, 0, 0, None
        if sampling_rate <= 0: print(f"Error: Invalid sampling rate."); return None, None, n_channels, 0, None
        uv_scale_factor_val = get_voltage_scaling_factor(meta) 
        if uv_scale_factor_val is None: warnings.warn("Could not calculate voltage scaling factor.")
        file_size = file_path.stat().st_size; item_size = np.dtype(data_type).itemsize
        if n_channels <= 0 or item_size <= 0: print(f"Error: Invalid channels/itemsize."); return None, sampling_rate, n_channels, 0, uv_scale_factor_val
        expected_total_bytes = file_size - (file_size % (n_channels * item_size))
        if file_size != expected_total_bytes: print(f"Warning: File size mismatch.")
        n_samples = expected_total_bytes // (n_channels * item_size)
        if n_samples <= 0: print("Error: Zero samples calculated."); return None, sampling_rate, n_channels, 0, uv_scale_factor_val
        shape = (n_samples, n_channels); 
        data = np.memmap(file_path, dtype=data_type, mode='r', shape=shape, offset=0)
        return data, sampling_rate, n_channels, n_samples, uv_scale_factor_val
    except FileNotFoundError: print(f"Error: File not found - {file_path} or {meta_path}"); return None, None, 0, 0, None
    except KeyError as e: print(f"Error: Metadata key missing - {e}"); return None, sampling_rate, n_channels, 0, None
    except ValueError as e: print(f"Error: Memmap shape/dtype error - {e}"); return None, sampling_rate, n_channels, 0, None
    except Exception as e: print(f"Unexpected error loading LFP: {e}"); traceback.print_exc(); return None, None, 0, 0, None

def load_channel_info(filepath):
    print(f"Loading channel info from {filepath}")
    try:
        channel_df = pd.read_csv(filepath); required_cols = ['global_channel_index', 'shank_index', 'acronym', 'name']
        if not all(col in channel_df.columns for col in required_cols): raise ValueError(f"Missing columns in CSV: {required_cols}")
        print(f"Loaded channel info for {len(channel_df)} channels.")
        return channel_df
    except FileNotFoundError: print(f"Error: Channel info file not found: {filepath}"); raise
    except Exception as e: print(f"Error loading channel info: {e}"); raise

def load_sleep_and_epoch_data(sleep_state_path, epoch_boundaries_path, fs):
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
                    else: 
                        times_path_alt_emg = sleep_state_path.parent / f"{base_name}_sleep_state_times_EMG.npy"
                        if times_path_alt_emg.exists():
                            times_path = times_path_alt_emg
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

def apply_fir_filter(data, lowcut, highcut, fs, numtaps=513, pass_zero=False):
    if data is None or data.size == 0: return None
    data_len = data.shape[-1] 
    if numtaps >= data_len:
        numtaps = data_len - 1 if data_len > 1 else 1
        numtaps -= (numtaps % 2 == 0) 
        if numtaps < 3: warnings.warn(f"Warning: Data length ({data_len}) too short for filter order ({numtaps}). Returning original data."); return data
    try:
        if lowcut is None and highcut is None: return data
        elif lowcut is None: b = signal.firwin(numtaps, highcut, fs=fs, pass_zero=True, window='hamming')
        elif highcut is None: b = signal.firwin(numtaps, lowcut, fs=fs, pass_zero=False, window='hamming')
        else: 
            nyq = fs / 2.0
            if highcut >= nyq: highcut = nyq * 0.99
            if lowcut >= highcut: warnings.warn(f"Error: lowcut ({lowcut}) >= highcut ({highcut}) after adjustment."); return None 
            b = signal.firwin(numtaps, [lowcut, highcut], fs=fs, pass_zero=pass_zero, window='hamming')
        if not np.issubdtype(data.dtype, np.floating): data = data.astype(np.float64)
        return signal.filtfilt(b, 1, data, axis=-1)
    except Exception as e: print(f"FIR filter error: {e}"); return None

def calculate_instantaneous_power_and_phase(data):
    if data is None or data.size == 0: return None, None
    try:
        if not np.issubdtype(data.dtype, np.floating): data = data.astype(np.float64)
        analytic_signal = signal.hilbert(data, axis=-1)
        amplitude_envelope = np.abs(analytic_signal)
        power = amplitude_envelope**2 
        phase = np.angle(analytic_signal) 
        return power, phase
    except Exception as e: print(f"Power/Phase calc error: {e}"); return None, None

def get_baseline_stats(power_signal, fs, non_rem_periods_samples):
    if power_signal is None or power_signal.size == 0: raise ValueError("Invalid power signal.")
    baseline_power_segments = [power_signal[max(0,s):min(power_signal.shape[0],e)] for s,e in non_rem_periods_samples if max(0,s)<min(power_signal.shape[0],e)] if non_rem_periods_samples else [power_signal]
    if not baseline_power_segments: baseline_power_segments = [power_signal] 
    if not any(seg.size > 0 for seg in baseline_power_segments): raise ValueError("No valid baseline segments or segments are empty.")
    try: full_baseline_power = np.concatenate([seg for seg in baseline_power_segments if seg.size > 0])
    except MemoryError: raise ValueError("MemoryError concatenating baseline.")
    except Exception as e: raise ValueError(f"Error preparing baseline: {e}")
    if full_baseline_power.size == 0: raise ValueError("Baseline power empty after concatenation.")
    initial_mean, initial_sd = np.mean(full_baseline_power), np.std(full_baseline_power)
    clipped_power = np.clip(full_baseline_power, None, initial_mean + 4 * initial_sd) if initial_sd > 0 else full_baseline_power
    rectified_power = np.abs(clipped_power); numtaps_lp = 101
    if numtaps_lp >= rectified_power.size:
        processed_baseline_power = rectified_power
    else:
        try:
            nyq_lp = fs / 2.0; lp_cutoff = min(RIPPLE_POWER_LP_CUTOFF, nyq_lp * 0.99)
            b_lp = signal.firwin(numtaps_lp, lp_cutoff, fs=fs, pass_zero=True, window='hamming')
            processed_baseline_power = signal.filtfilt(b_lp, 1, rectified_power)
        except Exception as e_lp: warnings.warn(f"Baseline LP filter error: {e_lp}"); processed_baseline_power = rectified_power
    return np.mean(processed_baseline_power), np.std(processed_baseline_power)

def detect_events(signal_data, fs, threshold_high, threshold_low, min_duration_ms, merge_gap_ms, event_type="Event", max_duration_ms=None):
    if signal_data is None or signal_data.size == 0: return np.array([], dtype=int), np.array([], dtype=int)
    min_duration_samples = int(min_duration_ms * fs / 1000); merge_gap_samples = int(merge_gap_ms * fs / 1000)
    max_duration_samples = int(max_duration_ms * fs / 1000) if max_duration_ms is not None else None
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
            cs = start;
            while cs > 0 and not below_low_mask[cs - 1]: cs -= 1
            ce = end;
            while ce < len(signal_data) - 1 and not below_low_mask[ce + 1]: ce += 1
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
        duration_samples = end - start + 1
        valid_duration = duration_samples >= min_duration_samples
        if max_duration_samples is not None: valid_duration = valid_duration and (duration_samples <= max_duration_samples)
        if valid_duration: final_starts.append(start); final_ends.append(end)
    return np.array(final_starts, dtype=int), np.array(final_ends, dtype=int)

def find_event_features(lfp_segment, power_segment, phase_segment):
    if power_segment is None or lfp_segment is None or phase_segment is None or \
       power_segment.size==0 or lfp_segment.size==0 or phase_segment.size==0 or \
       len(lfp_segment)!=len(power_segment) or len(lfp_segment)!=len(phase_segment):
        return -1, -1, np.nan
    try:
        peak_power_idx_rel = np.nanargmax(power_segment)
        phase_at_peak = phase_segment[peak_power_idx_rel]
        if not np.issubdtype(lfp_segment.dtype, np.floating): lfp_segment_float = lfp_segment.astype(np.float64)
        else: lfp_segment_float = lfp_segment
        troughs_rel, _ = signal.find_peaks(-lfp_segment_float) 
        if len(troughs_rel) == 0: trough_idx_rel = np.nanargmin(lfp_segment_float) 
        else: trough_idx_rel = troughs_rel[np.nanargmin(np.abs(troughs_rel - peak_power_idx_rel))] 
        return peak_power_idx_rel, trough_idx_rel, phase_at_peak
    except Exception: return -1, -1, np.nan

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
        if power_spectrogram.shape[0] == len(freqs):
            power_spectrogram = power_spectrogram * freqs[:, np.newaxis] 
        else:
            warnings.warn(f"Spectrogram frequency dim ({power_spectrogram.shape[0]}) != freqs len ({len(freqs)}). Skipping 1/f scaling.")
        if power_spectrogram.shape[1] != window_samples: 
            from scipy.interpolate import interp1d; x_old = np.linspace(0, 1, power_spectrogram.shape[1]); x_new = np.linspace(0, 1, window_samples)
            interp_func = interp1d(x_old, power_spectrogram, axis=1, kind='linear', fill_value='extrapolate'); power_spectrogram = interp_func(x_new)
        return power_spectrogram
    except MemoryError: warnings.warn("MemoryError in CWT calculation."); return np.full((len(freqs), window_samples), np.nan)
    except Exception as e: warnings.warn(f"CWT spectrogram error: {e}"); return np.full((len(freqs), window_samples), np.nan)

def plot_lfp_trace_with_event(lfp_filepath, n_total_channels, n_total_samples,
                               channel_idx_to_plot, event_peak_sample_orig,
                               event_type_str, plot_window_ms, fs_orig, uv_scale_factor,
                               output_dir_plots, plot_filename_base,
                               ripple_band_filtered_lfp=None, fs_ripple_filtered=None, 
                               event_start_sample_orig=None, event_end_sample_orig=None):
    lfp_memmap_plot = None
    try:
        output_dir_plots.mkdir(parents=True, exist_ok=True)
        window_samples_half_orig = int((plot_window_ms / 1000) * fs_orig / 2)
        plot_start_sample_orig = event_peak_sample_orig - window_samples_half_orig
        plot_end_sample_orig = event_peak_sample_orig + window_samples_half_orig
        pad_left = 0; pad_right = 0
        if plot_start_sample_orig < 0: pad_left = abs(plot_start_sample_orig); plot_start_sample_orig = 0
        if plot_end_sample_orig > n_total_samples: pad_right = plot_end_sample_orig - n_total_samples; plot_end_sample_orig = n_total_samples
        if plot_start_sample_orig >= plot_end_sample_orig: warnings.warn(f"Plotting: Invalid sample window. Skipping."); return
        lfp_memmap_plot = np.memmap(lfp_filepath, dtype='int16', mode='r', shape=(n_total_samples, n_total_channels))
        lfp_segment_orig = lfp_memmap_plot[plot_start_sample_orig:plot_end_sample_orig, channel_idx_to_plot].astype(np.float64)
        if pad_left > 0 or pad_right > 0: lfp_segment_orig = np.pad(lfp_segment_orig, (pad_left, pad_right), mode='reflect')
        if uv_scale_factor is not None: lfp_segment_orig *= uv_scale_factor; y_label = "LFP (\u00b5V)"
        else: y_label = "LFP (ADC units)"
        time_axis_ms = (np.arange(len(lfp_segment_orig)) - window_samples_half_orig - pad_left) * (1000 / fs_orig)
        plt.figure(figsize=(10, 6))
        plt.plot(time_axis_ms, lfp_segment_orig, label=f'Raw LFP Ch {channel_idx_to_plot}', color='black', linewidth=0.75)
        ripple_filtered_overlay = apply_fir_filter(lfp_segment_orig.copy(), RIPPLE_FILTER_LOWCUT, RIPPLE_FILTER_HIGHCUT, fs_orig)
        if ripple_filtered_overlay is not None:
             plt.plot(time_axis_ms, ripple_filtered_overlay, label='Ripple-band (100-250Hz)', color='red', alpha=0.7, linewidth=1)
        plt.axvline(0, color='magenta', linestyle='--', linewidth=1.5, label=f'{event_type_str} Peak')
        if event_start_sample_orig is not None:
            start_time_ms = (event_start_sample_orig - event_peak_sample_orig) * (1000 / fs_orig)
            plt.axvline(start_time_ms, color='green', linestyle=':', linewidth=1, label='Event Start')
        if event_end_sample_orig is not None:
            end_time_ms = (event_end_sample_orig - event_peak_sample_orig) * (1000 / fs_orig)
            plt.axvline(end_time_ms, color='blue', linestyle=':', linewidth=1, label='Event End')
        plt.xlabel("Time relative to event peak (ms)"); plt.ylabel(y_label)
        plt.title(f"LFP Trace: {event_type_str} on Ch {channel_idx_to_plot}\nPeak at orig sample: {event_peak_sample_orig}")
        plt.legend(loc='upper right'); plt.grid(True, linestyle=':', alpha=0.7); plt.tight_layout()
        plot_filepath = output_dir_plots / f"{plot_filename_base}.png"
        plt.savefig(plot_filepath); plt.close()
    except Exception as e: warnings.warn(f"Plot LFP trace error: {e}")
    finally:
        if lfp_memmap_plot is not None and hasattr(lfp_memmap_plot, '_mmap') and lfp_memmap_plot._mmap is not None:
            try: lfp_memmap_plot._mmap.close()
            except Exception: pass
        gc.collect()

def _calculate_baseline_worker(ch_idx, lfp_filepath, meta_filepath, non_rem_periods_samples_orig, fs_orig, n_channels, uv_scale_factor_val):
    lfp_data_memmap_worker = None; fs_eff = fs_orig / DOWNSAMPLING_FACTOR if DOWNSAMPLING_FACTOR > 1 else fs_orig
    try:
        meta = readMeta(meta_filepath); file_size = lfp_filepath.stat().st_size; item_size = np.dtype('int16').itemsize
        expected_total_bytes = file_size - (file_size % (n_channels * item_size)); n_samples = expected_total_bytes // (n_channels * item_size)
        shape = (n_samples, n_channels); lfp_data_memmap_worker = np.memmap(lfp_filepath, dtype='int16', mode='r', shape=shape)
        lfp_ch_full_orig = lfp_data_memmap_worker[:, ch_idx].astype(np.float64)
        if uv_scale_factor_val is not None: lfp_ch_full_orig *= uv_scale_factor_val
        else: warnings.warn(f"Ch {ch_idx}: No scaling factor, baseline power in AU^2")
        if DOWNSAMPLING_FACTOR > 1: lfp_ch_full_down = signal.decimate(lfp_ch_full_orig, q=DOWNSAMPLING_FACTOR, axis=0, ftype='fir', zero_phase=True); del lfp_ch_full_orig
        else: lfp_ch_full_down = lfp_ch_full_orig 
        lfp_ch_ripple_filtered_down = apply_fir_filter(lfp_ch_full_down, RIPPLE_FILTER_LOWCUT, RIPPLE_FILTER_HIGHCUT, fs_eff); del lfp_ch_full_down
        if lfp_ch_ripple_filtered_down is None: return ch_idx, None
        ripple_power_down, _ = calculate_instantaneous_power_and_phase(lfp_ch_ripple_filtered_down); del lfp_ch_ripple_filtered_down
        if ripple_power_down is None: return ch_idx, None
        non_rem_periods_samples_down = [(s // DOWNSAMPLING_FACTOR, e // DOWNSAMPLING_FACTOR) for s, e in non_rem_periods_samples_orig]
        baseline_mean, baseline_sd = get_baseline_stats(ripple_power_down, fs_eff, non_rem_periods_samples_down); del ripple_power_down
        return ch_idx, (baseline_mean, baseline_sd)
    except Exception as e: print(f"ERROR in baseline worker for channel {ch_idx}: {e}"); traceback.print_exc(); return ch_idx, None
    finally:
        if lfp_data_memmap_worker is not None and hasattr(lfp_data_memmap_worker, '_mmap') and lfp_data_memmap_worker._mmap is not None:
            try: lfp_data_memmap_worker._mmap.close()
            except Exception: pass
        gc.collect()

def _detect_events_worker(ch_idx, ch_region, epoch_start_sample_orig, epoch_end_sample_orig,
                           lfp_filepath, n_samples_orig, n_channels, baseline_stats, fs_orig, uv_scale_factor_val):
    lfp_data_memmap_worker = None; ripple_events = []; spw_events = []; fs_eff = fs_orig / DOWNSAMPLING_FACTOR if DOWNSAMPLING_FACTOR > 1 else fs_orig
    try:
        shape = (n_samples_orig, n_channels); lfp_data_memmap_worker = np.memmap(lfp_filepath, dtype='int16', mode='r', shape=shape)
        lfp_ch_epoch_orig = lfp_data_memmap_worker[epoch_start_sample_orig:epoch_end_sample_orig, ch_idx].astype(np.float64)
        if uv_scale_factor_val is not None: lfp_ch_epoch_orig *= uv_scale_factor_val
        else: warnings.warn(f"Ch {ch_idx}, Ep {epoch_start_sample_orig}: No scaling factor, power in AU^2")
        if DOWNSAMPLING_FACTOR > 1: lfp_ch_epoch_down = signal.decimate(lfp_ch_epoch_orig, q=DOWNSAMPLING_FACTOR, axis=0, ftype='fir', zero_phase=True)
        else: lfp_ch_epoch_down = lfp_ch_epoch_orig
        del lfp_ch_epoch_orig 
        lfp_ch_ripple_filtered_down = apply_fir_filter(lfp_ch_epoch_down, RIPPLE_FILTER_LOWCUT, RIPPLE_FILTER_HIGHCUT, fs_eff)
        if lfp_ch_ripple_filtered_down is not None:
            ripple_power_down, ripple_phase_down = calculate_instantaneous_power_and_phase(lfp_ch_ripple_filtered_down)
            if ripple_power_down is not None:
                baseline_mean, baseline_sd = baseline_stats
                if baseline_sd > 0: 
                    det_thr = baseline_mean + RIPPLE_DETECTION_SD_THRESHOLD * baseline_sd; exp_thr = baseline_mean + RIPPLE_EXPANSION_SD_THRESHOLD * baseline_sd
                    r_starts_rel_d, r_ends_rel_d = detect_events(ripple_power_down, fs_eff, det_thr, exp_thr, RIPPLE_MIN_DURATION_MS, RIPPLE_MERGE_GAP_MS, "Ripple", max_duration_ms=RIPPLE_MAX_DURATION_MS)
                    for sr_d, er_d in zip(r_starts_rel_d, r_ends_rel_d):
                        if sr_d < 0 or er_d >= len(lfp_ch_ripple_filtered_down): continue 
                        lfp_ripple_seg_d = lfp_ch_ripple_filtered_down[sr_d:er_d+1] 
                        pow_seg_d = ripple_power_down[sr_d:er_d+1]
                        phase_seg_d = ripple_phase_down[sr_d:er_d+1] if ripple_phase_down is not None else np.full_like(pow_seg_d, np.nan)
                        pk_idx_d, tr_idx_d, phase_at_pk = find_event_features(lfp_ripple_seg_d, pow_seg_d, phase_seg_d)
                        if pk_idx_d != -1:
                            dur_samp_d = er_d - sr_d + 1
                            start_abs = epoch_start_sample_orig + (sr_d * DOWNSAMPLING_FACTOR)
                            end_abs = start_abs + (dur_samp_d * DOWNSAMPLING_FACTOR) - DOWNSAMPLING_FACTOR 
                            pk_abs = epoch_start_sample_orig + ((sr_d + pk_idx_d) * DOWNSAMPLING_FACTOR)
                            trough_abs_val = epoch_start_sample_orig + ((sr_d + tr_idx_d) * DOWNSAMPLING_FACTOR) if tr_idx_d != -1 else np.nan
                            ripple_events.append({'start_sample': start_abs, 'end_sample': end_abs,
                                                  'peak_sample': pk_abs, 'trough_sample': trough_abs_val,
                                                  'peak_power': pow_seg_d[pk_idx_d],
                                                  'duration_ms': dur_samp_d / fs_eff * 1000,
                                                  'peak_phase': phase_at_pk})
                if ripple_power_down is not None: del ripple_power_down
                if ripple_phase_down is not None: del ripple_phase_down
            del lfp_ch_ripple_filtered_down
        if ch_region == 'CA1':
            lfp_ch_spw_filtered_down = apply_fir_filter(lfp_ch_epoch_down, SPW_FILTER_LOWCUT, SPW_FILTER_HIGHCUT, fs_eff)
            if lfp_ch_spw_filtered_down is not None:
                spw_mean_d = np.nanmean(lfp_ch_spw_filtered_down); spw_sd_d = np.nanstd(lfp_ch_spw_filtered_down)
                if spw_sd_d > 0 and np.isfinite(spw_sd_d):
                    spw_det_thr = SPW_DETECTION_SD_THRESHOLD * spw_sd_d; spw_exp_thr = 1.0 * spw_sd_d 
                    s_starts_rel_d, s_ends_rel_d = detect_events(np.abs(lfp_ch_spw_filtered_down - spw_mean_d), fs_eff, spw_det_thr, spw_exp_thr, SPW_MIN_DURATION_MS, 1, "SPW", max_duration_ms=SPW_MAX_DURATION_MS)
                    for sr_d, er_d in zip(s_starts_rel_d, s_ends_rel_d):
                        if sr_d < 0 or er_d >= len(lfp_ch_spw_filtered_down): continue
                        dur_samp_d = er_d - sr_d + 1; dur_ms = dur_samp_d / fs_eff * 1000
                        spw_seg_d = lfp_ch_spw_filtered_down[sr_d:er_d+1]; tr_idx_seg_d = np.nanargmin(spw_seg_d) 
                        start_abs = epoch_start_sample_orig + (sr_d * DOWNSAMPLING_FACTOR); end_abs = start_abs + (dur_samp_d * DOWNSAMPLING_FACTOR) - DOWNSAMPLING_FACTOR
                        tr_abs = epoch_start_sample_orig + ((sr_d + tr_idx_seg_d) * DOWNSAMPLING_FACTOR)
                        spw_events.append({'start_sample': start_abs, 'end_sample': end_abs, 'trough_sample': tr_abs, 'duration_ms': dur_ms})
                del lfp_ch_spw_filtered_down
        del lfp_ch_epoch_down 
        return ch_idx, ripple_events, spw_events
    except Exception as e: print(f"ERROR in event worker for channel {ch_idx}, epoch {epoch_start_sample_orig}-{epoch_end_sample_orig}: {e}"); traceback.print_exc(); return ch_idx, [], []
    finally:
        if lfp_data_memmap_worker is not None and hasattr(lfp_data_memmap_worker, '_mmap') and lfp_data_memmap_worker._mmap is not None:
            try: lfp_data_memmap_worker._mmap.close()
            except Exception: pass
        gc.collect()

def _calculate_spectrogram_worker(ts_sample_orig, spec_ch_idx, lfp_filepath, n_samples_orig, n_channels, fs_orig, uv_scale_factor_val):
    lfp_data_memmap_worker = None; fs_eff = fs_orig / DOWNSAMPLING_FACTOR if DOWNSAMPLING_FACTOR > 1 else fs_orig
    window_samples_orig = int(SPECTROGRAM_WINDOW_MS * fs_orig / 1000); window_samples_eff = int(SPECTROGRAM_WINDOW_MS * fs_eff / 1000)
    try:
        shape = (n_samples_orig, n_channels); lfp_data_memmap_worker = np.memmap(lfp_filepath, dtype='int16', mode='r', shape=shape)
        start_sample_orig = int(ts_sample_orig - window_samples_orig // 2); end_sample_orig = int(start_sample_orig + window_samples_orig)
        pad_left_orig, pad_right_orig = 0, 0
        if start_sample_orig < 0: pad_left_orig = -start_sample_orig; start_sample_orig = 0
        if end_sample_orig > n_samples_orig: pad_right_orig = end_sample_orig - n_samples_orig; end_sample_orig = n_samples_orig
        segment_orig = lfp_data_memmap_worker[start_sample_orig:end_sample_orig, spec_ch_idx].astype(np.float64)
        if uv_scale_factor_val is not None: segment_orig *= uv_scale_factor_val
        if pad_left_orig > 0 or pad_right_orig > 0:
            try: segment_orig = np.pad(segment_orig, (pad_left_orig, pad_right_orig), mode='reflect')
            except ValueError: segment_orig = np.pad(segment_orig, (pad_left_orig, pad_right_orig), mode='edge')
        if DOWNSAMPLING_FACTOR > 1 and segment_orig.size > DOWNSAMPLING_FACTOR * 20: 
            segment_down = signal.decimate(segment_orig, q=DOWNSAMPLING_FACTOR, axis=0, ftype='fir', zero_phase=True)
        else: segment_down = segment_orig
        del segment_orig
        spec = calculate_cwt_spectrogram(segment_down, fs_eff, SPECTROGRAM_FREQS, len(segment_down) // 2, window_samples_eff)
        del segment_down
        return spec
    except Exception as e: return np.full((len(SPECTROGRAM_FREQS), window_samples_eff), np.nan) 
    finally:
        if lfp_data_memmap_worker is not None and hasattr(lfp_data_memmap_worker, '_mmap') and lfp_data_memmap_worker._mmap is not None:
            try: lfp_data_memmap_worker._mmap.close()
            except Exception: pass
        gc.collect()

# --- Main Analysis Function ---
def run_ripple_analysis(lfp_filepath, meta_filepath, channel_info_filepath,
                         sleep_state_filepath, epoch_boundaries_filepath,
                         output_dir):
    lfp_filepath = Path(lfp_filepath); meta_filepath = Path(meta_filepath)
    channel_info_filepath = Path(channel_info_filepath)
    sleep_state_filepath = Path(sleep_state_filepath) if sleep_state_filepath else None
    epoch_boundaries_filepath = Path(epoch_boundaries_filepath) if epoch_boundaries_filepath else None
    output_dir = Path(output_dir)
    output_path = Path(output_dir); output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_path}")

    output_dir_lfp_plots = output_path / "lfp_event_plots"
    if PLOT_LFP_EXAMPLES: output_dir_lfp_plots.mkdir(parents=True, exist_ok=True)
    output_dir_psd_plots = output_path / "psd_selection_plots" 
    output_dir_psd_plots.mkdir(parents=True, exist_ok=True)

    match = re.match(r"^(.*?)(\.(imec|nidq)\d?)?\.lf\.bin$", lfp_filepath.name)
    output_filename_base = match.group(1) if match else lfp_filepath.stem
    print(f"Derived base filename: {output_filename_base}")

    all_ripple_events_by_epoch = {}; all_spw_events_by_epoch = {}
    ripple_events_by_state_epoch = {state: {} for state in STATES_TO_ANALYZE}
    spw_events_by_state_epoch = {state: {} for state in STATES_TO_ANALYZE}
    ripple_events_by_state_global = {state: {} for state in STATES_TO_ANALYZE}
    spw_events_by_state_global = {state: {} for state in STATES_TO_ANALYZE}
    ca1_spwr_events_by_state_global = {state: {} for state in STATES_TO_ANALYZE}
    averaged_timestamps_by_state_epoch = {state: {} for state in STATES_TO_ANALYZE}
    cooccurrence_results_by_state_epoch = {state: {} for state in STATES_TO_ANALYZE}
    averaged_spectrograms_by_state_epoch = {state: {} for state in STATES_TO_ANALYZE}
    reference_sites = {} 
    reference_channel_set = set() 
    noise_ripple_power_down = None 
    ripple_rates_by_state_epoch = {state: {} for state in STATES_TO_ANALYZE}
    state_duration_in_epoch = {state: {} for state in STATES_TO_ANALYZE}
    ripple_counts_by_state_epoch = {state: {} for state in STATES_TO_ANALYZE}
    consolidated_event_timestamps = []
    dynamically_selected_noise_channel_idx = None
    found_potential_noise_channel_flag = False 

### Steps 1 & 2: Loading LFP Info, Channel Info & Voltage Scaling ###
    try:
        print("\n--- Steps 1 & 2: Loading LFP Info, Channel Info & Voltage Scaling ---")
        _, fs_orig, n_channels, n_samples_orig, uv_scale_factor = load_lfp_data_memmap(lfp_filepath, meta_filepath)
        if fs_orig is None: raise ValueError("Failed to load LFP.")
        fs_eff = fs_orig / DOWNSAMPLING_FACTOR if DOWNSAMPLING_FACTOR > 1 else fs_orig
        print(f"LFP: {n_channels} channels, {n_samples_orig} samples. Orig Fs={fs_orig:.2f} Hz, Eff Fs={fs_eff:.2f} Hz")
        if uv_scale_factor is None: warnings.warn("Proceeding without voltage scaling.")

        channel_df = load_channel_info(channel_info_filepath)
        
        noise_acronyms_priority = ['DG-mo', 'DG', 'DG-sg'] 
        for acronym_search in noise_acronyms_priority:
            if dynamically_selected_noise_channel_idx is not None: break
            noise_ch_candidates = channel_df[channel_df['acronym'] == acronym_search]
            if not noise_ch_candidates.empty:
                dynamically_selected_noise_channel_idx = int(noise_ch_candidates['global_channel_index'].iloc[0])
                print(f"Dynamically identified potential noise channel: Index {dynamically_selected_noise_channel_idx} (Acronym: {acronym_search})")
                found_potential_noise_channel_flag = True
        if not found_potential_noise_channel_flag:
            print("Noise exclusion: No DG-mo, DG, or DG-sg channel found. Noise exclusion will be disabled.")

        target_regions = ['CA1', 'CA3', 'CA2'] 
        region_channels_df = channel_df[channel_df['acronym'].isin(target_regions)].copy()
        if region_channels_df.empty: raise ValueError("No target region channels found in channel_info file.")
        channels_to_process = sorted(region_channels_df['global_channel_index'].astype(int).unique())
        channel_region_map = pd.Series(region_channels_df.acronym.values, index=region_channels_df.global_channel_index).to_dict()
        shanks = sorted(region_channels_df['shank_index'].unique())

# --- Step 3: Loading Sleep State and Epoch Data ---
        print("\n--- Step 3: Loading Sleep State and Epoch Data ---")
        sleep_state_data, epoch_boundaries_sec, non_rem_periods_sec = load_sleep_and_epoch_data(sleep_state_filepath, epoch_boundaries_filepath, fs_orig)
        non_rem_periods_samples_orig = [(int(s*fs_orig), int(e*fs_orig)) for s,e in non_rem_periods_sec]
        epoch_boundaries_samples_orig = [(int(s*fs_orig), int(e*fs_orig)) for s,e in epoch_boundaries_sec]
        if not epoch_boundaries_samples_orig: epoch_boundaries_samples_orig = [(0, n_samples_orig)]
        state_lookup = None
        if sleep_state_data is not None:
            print("Preparing sleep state lookup..."); sleep_state_data = sleep_state_data[np.argsort(sleep_state_data[:, 0]), :] 
            state_times_sec_lookup = sleep_state_data[:, 0]; state_codes_lookup = sleep_state_data[:, 1]
            def get_state_at_sample(sample_idx, fs_val): 
                time_sec = sample_idx / fs_val; idx = np.searchsorted(state_times_sec_lookup, time_sec, side='right') - 1
                return state_codes_lookup[idx] if idx >= 0 and idx < len(state_codes_lookup) else -1 
            state_lookup = get_state_at_sample; print("Sleep state lookup ready.")
        else: raise ValueError("Sleep state data is required for state-specific analysis.")

        if REFERENCE_SELECTION_MODE == "manual":
            print(f"\n--- Step M.0: Determining Reference Sites MANUALLY (All Regions: {', '.join(target_regions)}) ---")
            for shank_idx_iter in shanks:
                ref_sites_shank = {}
                print(f"  Processing Shank {shank_idx_iter} for MANUAL reference site selection...")
                for region_name_iter in target_regions: 
                    shank_region_ch_df = region_channels_df[
                        (region_channels_df['shank_index'] == shank_idx_iter) & 
                        (region_channels_df['acronym'] == region_name_iter)
                    ]
                    if shank_region_ch_df.empty:
                        print(f"    No channels found for {region_name_iter} on Shank {shank_idx_iter}. Skipping manual selection for this region/shank.")
                        continue
                    region_ch_indices_on_shank = shank_region_ch_df['global_channel_index'].astype(int).tolist()
                    print(f"    Prompting for {region_name_iter} on Shank {shank_idx_iter} (Candidate Channels: {region_ch_indices_on_shank})...")
                    selected_ref_ch = select_ref_channel_by_manual_psd_visualization(
                        lfp_filepath=lfp_filepath, meta_filepath=meta_filepath,
                        n_total_channels=n_channels, n_total_samples=n_samples_orig,
                        region_channel_indices_on_shank=region_ch_indices_on_shank, 
                        current_shank_idx=shank_idx_iter, current_region_name=region_name_iter,
                        output_dir_psd_plots=output_dir_psd_plots, 
                        readMeta_func=readMeta, get_voltage_scaling_factor_func=get_voltage_scaling_factor,
                        target_psd_fs=1250.0, plot_freq_range=(1.0, 200.0)
                    )
                    if selected_ref_ch is not None: 
                        ref_sites_shank[region_name_iter] = selected_ref_ch
                if ref_sites_shank: 
                    reference_sites[shank_idx_iter] = ref_sites_shank
            reference_channel_set = set(ch_idx for shank_data in reference_sites.values() for ch_idx in shank_data.values() if ch_idx is not None)
            print(f"MANUALLY selected {len(reference_channel_set)} unique reference channels: {sorted(list(reference_channel_set))}")
            if not reference_channel_set: 
                warnings.warn("MANUAL reference selection resulted in NO channels. Subsequent analysis might be empty or fail.")
        
##### --- Step 4: Calculating Global Baseline Statistics (Parallel using {NUM_CORES} cores) ---#####
        print(f"\n--- Step 4: Calculating Global Baseline Statistics (Parallel using {NUM_CORES} cores) ---")
        baseline_stats_per_channel = {}
        baseline_tasks = [(ch, lfp_filepath, meta_filepath, non_rem_periods_samples_orig, fs_orig, n_channels, uv_scale_factor) for ch in channels_to_process if ch < n_channels]
        results = Parallel(n_jobs=NUM_CORES, backend="loky")(delayed(_calculate_baseline_worker)(*task) for task in baseline_tasks)
        valid_baselines = 0
        for ch_res, stats_res in results: 
            if stats_res: baseline_stats_per_channel[ch_res] = stats_res; valid_baselines += 1
        print(f"Calculated baseline for {valid_baselines}/{len(channels_to_process)} channels.")
        if valid_baselines == 0 and len(channels_to_process) > 0 : raise ValueError("Baseline calculation failed for all target region channels.")
        elif valid_baselines == 0 and len(channels_to_process) == 0: warnings.warn("No target region channels found to calculate baselines for.")

        use_noise_channel = False 
        if dynamically_selected_noise_channel_idx is not None:
            print(f"\n--- Step 4.5: Precomputing Noise Channel ({dynamically_selected_noise_channel_idx}) Ripple Power ---")
            if dynamically_selected_noise_channel_idx >= n_channels or dynamically_selected_noise_channel_idx < 0:
                warnings.warn(f"Dynamically selected noise channel index {dynamically_selected_noise_channel_idx} is out of bounds ({n_channels} total channels). Disabling noise exclusion.")
            else: 
                lfp_noise_memmap = None
                try:
                    shape = (n_samples_orig, n_channels)
                    lfp_noise_memmap = np.memmap(lfp_filepath, dtype='int16', mode='r', shape=shape)
                    lfp_ch_noise_orig = lfp_noise_memmap[:, dynamically_selected_noise_channel_idx].astype(np.float64)
                    if uv_scale_factor is not None: lfp_ch_noise_orig *= uv_scale_factor
                    if DOWNSAMPLING_FACTOR > 1: lfp_ch_noise_down = signal.decimate(lfp_ch_noise_orig, q=DOWNSAMPLING_FACTOR, axis=0, ftype='fir', zero_phase=True)
                    else: lfp_ch_noise_down = lfp_ch_noise_orig
                    del lfp_ch_noise_orig
                    lfp_ch_noise_filt = apply_fir_filter(lfp_ch_noise_down, RIPPLE_FILTER_LOWCUT, RIPPLE_FILTER_HIGHCUT, fs_eff); del lfp_ch_noise_down
                    if lfp_ch_noise_filt is not None:
                        noise_ripple_power_down, _ = calculate_instantaneous_power_and_phase(lfp_ch_noise_filt); del lfp_ch_noise_filt
                        if noise_ripple_power_down is None: 
                            warnings.warn(f"Failed to calculate power for noise channel {dynamically_selected_noise_channel_idx}. Disabling noise exclusion.")
                        else: 
                            print(f"  Successfully precomputed ripple power for noise channel {dynamically_selected_noise_channel_idx}.")
                            use_noise_channel = True 
                    else: 
                        warnings.warn(f"Failed to filter noise channel {dynamically_selected_noise_channel_idx}. Disabling noise exclusion.")
                except Exception as e: 
                    warnings.warn(f"Error processing noise channel {dynamically_selected_noise_channel_idx}: {e}. Disabling noise exclusion.")
                    noise_ripple_power_down = None 
                finally:
                    if lfp_noise_memmap is not None and hasattr(lfp_noise_memmap, '_mmap') and lfp_noise_memmap._mmap is not None:
                        try: lfp_noise_memmap._mmap.close()
                        except Exception: pass
                    gc.collect()
        if not use_noise_channel and found_potential_noise_channel_flag:
             print("Noise exclusion was configured with a DG channel, but it was later found to be invalid or failed processing. Noise exclusion disabled.")
        elif not found_potential_noise_channel_flag:
             print("Noise exclusion is disabled (no suitable DG channel was identified initially).")

# --- Step 5: Detecting ALL Events (Parallel using {NUM_CORES} cores per epoch) ---###########
        print(f"\n--- Step 5: Detecting ALL Events (Parallel using {NUM_CORES} cores per epoch) ---")
        for epoch_idx, (ep_start, ep_end) in enumerate(epoch_boundaries_samples_orig):
            print(f"\nProcessing Epoch {epoch_idx} (Orig Samples: {ep_start} - {ep_end})...")
            if ep_end <= ep_start: print(f"  Skipping empty epoch {epoch_idx}."); continue
            all_ripple_events_by_epoch[epoch_idx] = {}
            all_spw_events_by_epoch[epoch_idx] = {}
            epoch_tasks = [(ch, channel_region_map.get(ch), ep_start, ep_end, lfp_filepath, n_samples_orig, n_channels, baseline_stats_per_channel[ch], fs_orig, uv_scale_factor) for ch in channels_to_process if ch in baseline_stats_per_channel]
            if not epoch_tasks: print(f"  No channels with baseline stats for epoch {epoch_idx}."); continue
            print(f"  Detecting events for {len(epoch_tasks)} channels...")
            epoch_results = Parallel(n_jobs=NUM_CORES, backend="loky")(delayed(_detect_events_worker)(*task) for task in epoch_tasks)
            for ch_res, ripple_evs, spw_evs in epoch_results: 
                if ripple_evs: all_ripple_events_by_epoch[epoch_idx][ch_res] = ripple_evs
                if spw_evs: all_spw_events_by_epoch[epoch_idx][ch_res] = spw_evs
            print(f"  Finished ALL event detection for Epoch {epoch_idx}.")
            gc.collect()

# --- Step 6: Filtering Events by State & Noise, Calculating Z-Scores & IRIs ---
        print("\n--- Step 6: Filtering Events by State & Noise, Calculating Z-Scores & IRIs ---")
        ripple_power_stats_by_state_channel = {state: {} for state in STATES_TO_ANALYZE}
        noise_events_excluded = 0
        for state_code, state_name in STATES_TO_ANALYZE.items():
            print(f"  Processing state: {state_name}")
            ripple_events_by_state_epoch[state_code] = {} 
            spw_events_by_state_epoch[state_code] = {}   
            ripple_events_by_state_global[state_code] = {}
            spw_events_by_state_global[state_code] = {}   
            all_powers_state_ch = {} 
            print(f"    Filtering by state & noise, aggregating ripple powers...")
            for epoch_idx in all_ripple_events_by_epoch:
                for ch_idx, events in all_ripple_events_by_epoch[epoch_idx].items():
                    state_filtered_ripples = [ev for ev in events if state_lookup(ev['peak_sample'], fs_orig) == state_code]
                    if not state_filtered_ripples: continue
                    final_filtered_ripples = []
                    if use_noise_channel and noise_ripple_power_down is not None and ch_idx != dynamically_selected_noise_channel_idx:
                        if ch_idx not in baseline_stats_per_channel: 
                            warnings.warn(f"Ch {ch_idx} has no baseline stats, skipping noise exclusion for it.")
                            final_filtered_ripples = state_filtered_ripples 
                        else:
                            baseline_mean, baseline_sd = baseline_stats_per_channel[ch_idx]
                            if baseline_sd > 0:
                                noise_check_threshold = baseline_mean + RIPPLE_DETECTION_SD_THRESHOLD * baseline_sd
                                for ev in state_filtered_ripples:
                                    start_samp_d = ev['start_sample'] // DOWNSAMPLING_FACTOR
                                    end_samp_d = ev['end_sample'] // DOWNSAMPLING_FACTOR
                                    start_samp_d = max(0, start_samp_d); end_samp_d = min(len(noise_ripple_power_down) - 1, end_samp_d)
                                    if start_samp_d <= end_samp_d:
                                        noise_segment = noise_ripple_power_down[start_samp_d : end_samp_d + 1]
                                        if not np.any(noise_segment > noise_check_threshold): final_filtered_ripples.append(ev)
                                        else: noise_events_excluded += 1
                                    else: final_filtered_ripples.append(ev) 
                            else: final_filtered_ripples = state_filtered_ripples 
                    else: final_filtered_ripples = state_filtered_ripples
                    if final_filtered_ripples:
                        ripple_events_by_state_epoch[state_code].setdefault(epoch_idx, {})[ch_idx] = final_filtered_ripples
                        powers = [ev['peak_power'] for ev in final_filtered_ripples if 'peak_power' in ev and np.isfinite(ev['peak_power'])]
                        if powers: all_powers_state_ch.setdefault(ch_idx, []).extend(powers)
            print(f"    Calculating Z-score statistics...")
            for ch_idx, powers in all_powers_state_ch.items():
                mean_p = np.mean(powers) if powers else 0; std_p = np.std(powers) if len(powers) > 1 else 0
                ripple_power_stats_by_state_channel[state_code][ch_idx] = (mean_p, std_p)
            print(f"    Calculating Z-scores, IRIs, and aggregating globally...")
            for epoch_idx in ripple_events_by_state_epoch[state_code]:
                for ch_idx, events in ripple_events_by_state_epoch[state_code][epoch_idx].items():
                    events.sort(key=lambda x: x['peak_sample'])
                    ch_mean, ch_std = ripple_power_stats_by_state_channel[state_code].get(ch_idx, (0, 0))
                    previous_peak_sample = -np.inf; processed_events = []
                    for ev in events:
                        ev['peak_power_zscore'] = (ev['peak_power'] - ch_mean) / ch_std if ch_std > 1e-9 else 0.0
                        iri_samples = ev['peak_sample'] - previous_peak_sample
                        ev['iri_ms'] = (iri_samples / fs_orig) * 1000.0 if previous_peak_sample != -np.inf and iri_samples > 0 else np.nan
                        previous_peak_sample = ev['peak_sample']; ev['epoch_idx'] = epoch_idx
                        processed_events.append(ev)
                    ripple_events_by_state_epoch[state_code][epoch_idx][ch_idx] = processed_events
                    ripple_events_by_state_global[state_code].setdefault(ch_idx, []).extend(processed_events)
            for epoch_idx in all_spw_events_by_epoch: 
                spw_events_by_state_epoch[state_code].setdefault(epoch_idx, {}) 
                for ch_idx, events in all_spw_events_by_epoch[epoch_idx].items():
                    filtered_events = [ev for ev in events if state_lookup(ev['trough_sample'], fs_orig) == state_code]
                    if filtered_events:
                        spw_events_by_state_epoch[state_code][epoch_idx][ch_idx] = filtered_events
                        for ev in filtered_events: ev['epoch_idx'] = epoch_idx
                        spw_events_by_state_global[state_code].setdefault(ch_idx, []).extend(filtered_events)
            print(f"Finished processing for state: {state_name}")

# --- Step 6.5: Calculating Ripple Rates per State per Epoch ---
        print("\n--- Step 6.5: Calculating Ripple Rates per State per Epoch ---")
        ripple_rates_by_state_epoch = {state: {} for state in STATES_TO_ANALYZE}
        state_duration_in_epoch = {state: {} for state in STATES_TO_ANALYZE} 
        ripple_counts_by_state_epoch = {state: {} for state in STATES_TO_ANALYZE} 
        if state_lookup: 
            print("  Calculating state durations within epochs...")
            for epoch_idx, (ep_start_samp, ep_end_samp) in enumerate(epoch_boundaries_samples_orig):
                epoch_start_sec = ep_start_samp / fs_orig; epoch_end_sec = ep_end_samp / fs_orig
                start_interval_idx = np.searchsorted(state_times_sec_lookup, epoch_start_sec, side='right') - 1
                start_interval_idx = max(0, start_interval_idx)
                end_interval_idx = np.searchsorted(state_times_sec_lookup, epoch_end_sec, side='left')
                for state_code_target, state_name_target in STATES_TO_ANALYZE.items():
                    duration_sec = 0
                    for i in range(start_interval_idx, min(end_interval_idx + 1, len(state_codes_lookup))): 
                        current_interval_state_code = state_codes_lookup[i]
                        if current_interval_state_code == state_code_target:
                            interval_start_sec = state_times_sec_lookup[i]
                            interval_end_sec = state_times_sec_lookup[i+1] if i + 1 < len(state_times_sec_lookup) else epoch_end_sec 
                            clipped_start = max(interval_start_sec, epoch_start_sec)
                            clipped_end = min(interval_end_sec, epoch_end_sec)
                            if clipped_end > clipped_start: duration_sec += (clipped_end - clipped_start)
                    state_duration_in_epoch[state_code_target][epoch_idx] = duration_sec
            print("  Calculating ripple rates...")
            for state_code, state_name in STATES_TO_ANALYZE.items():
                ripple_rates_by_state_epoch[state_code] = {} 
                ripple_counts_by_state_epoch[state_code] = {} 
                for epoch_idx in ripple_events_by_state_epoch.get(state_code, {}):
                    total_ripples_in_epoch_state = sum(len(ch_events) for ch_events in ripple_events_by_state_epoch[state_code][epoch_idx].values())
                    duration_sec = state_duration_in_epoch[state_code].get(epoch_idx, 0)
                    rate = total_ripples_in_epoch_state / duration_sec if duration_sec > 1e-6 else 0.0
                    if duration_sec <= 1e-6 and total_ripples_in_epoch_state > 0: warnings.warn(f"Epoch {epoch_idx}, State {state_name}: Found {total_ripples_in_epoch_state} ripples but duration is {duration_sec:.4f}s. Rate set to 0.")
                    ripple_rates_by_state_epoch[state_code][epoch_idx] = rate
                    ripple_counts_by_state_epoch[state_code][epoch_idx] = total_ripples_in_epoch_state
                    print(f"    Epoch {epoch_idx}, State {state_name}: Duration={duration_sec:.2f}s, Ripples={total_ripples_in_epoch_state}, Rate={rate:.3f} Hz")
        else: print("  Skipping ripple rate calculation (no sleep state data).")
        
        if found_potential_noise_channel_flag and use_noise_channel:
            print(f"\nINFO (Noise Filter Summary): Total ripple events excluded due to noise: {noise_events_excluded}")
        elif found_potential_noise_channel_flag and not use_noise_channel:
            print(f"\nINFO (Noise Filter Summary): A potential noise channel (DG type) was identified, but noise exclusion was ultimately disabled. Events excluded: {noise_events_excluded}")
        else: 
            print(f"\nINFO (Noise Filter Summary): No suitable DG-type channel found for noise exclusion. Events excluded: {noise_events_excluded}")

# --- Step 7: Checking SPW-Ripple Coincidence (State-Specific) ---
        print("\n--- Step 7: Checking SPW-Ripple Coincidence (State-Specific) ---")
        ca1_indices = region_channels_df[region_channels_df['acronym'] == 'CA1']['global_channel_index'].values
        for state_code, state_name in STATES_TO_ANALYZE.items():
            ca1_spwr_events_by_state_global[state_code] = {} 
            print(f"  Checking for state: {state_name}")
            for ch in ca1_indices:
                ripples_filt = ripple_events_by_state_global[state_code].get(ch, [])
                spws_filt = spw_events_by_state_global[state_code].get(ch, []) 
                if not ripples_filt or not spws_filt: continue
                coincident_ripples = []
                for r_event in ripples_filt:
                    is_co, spw_trough_sample = False, None
                    for s_event in spws_filt: 
                        if r_event['start_sample'] <= s_event['end_sample'] and r_event['end_sample'] >= s_event['start_sample']:
                            is_co, spw_trough_sample = True, s_event['trough_sample']; break 
                    if is_co: 
                        r_event_copy = r_event.copy() 
                        r_event_copy['associated_spw_trough_sample'] = spw_trough_sample
                        coincident_ripples.append(r_event_copy)
                if coincident_ripples:
                    ca1_spwr_events_by_state_global[state_code][ch] = coincident_ripples
                    print(f"    Ch {ch} (CA1): Found {len(coincident_ripples)} {state_name} SPW-R events.")

        if REFERENCE_SELECTION_MODE == "automatic":
            print(f"\n--- Step A.8: Determining Reference Sites AUTOMATICALLY ---")
            _all_ripple_global_temp_for_ref_auto = {} 
            for ep_idx_auto in all_ripple_events_by_epoch:
                for ch_idx_auto, evs_auto in all_ripple_events_by_epoch[ep_idx_auto].items():
                    _all_ripple_global_temp_for_ref_auto.setdefault(ch_idx_auto, []).extend(evs_auto)
            for shank_idx_iter_auto in shanks:
                shank_ch_df_auto = region_channels_df[region_channels_df['shank_index'] == shank_idx_iter_auto]
                ref_sites_shank_auto = {}
                print(f"  Processing Shank {shank_idx_iter_auto} for AUTOMATIC reference site selection...")
                for region_auto in target_regions:
                    reg_shank_ch_df_auto = shank_ch_df_auto[shank_ch_df_auto['acronym'] == region_auto]
                    if reg_shank_ch_df_auto.empty: continue
                    selected_ref_ch_auto = None
                    if region_auto == 'CA1':
                        shank_ca1_indices_auto = reg_shank_ch_df_auto['global_channel_index'].astype(int).tolist()
                        if shank_ca1_indices_auto:
                            selected_ref_ch_auto = select_ca1_ref_by_automated_psd_metric(
                                lfp_filepath, meta_filepath, n_channels, n_samples_orig,
                                shank_ca1_indices_auto, shank_idx_iter_auto, 
                                readMeta, get_voltage_scaling_factor
                            )
                    else: 
                        max_pow_auto, current_ref_ch_auto = -1.0, -1
                        for _, row_auto in reg_shank_ch_df_auto.iterrows():
                            ch_auto = int(row_auto['global_channel_index'])
                            evs_ref_auto = _all_ripple_global_temp_for_ref_auto.get(ch_auto, [])
                            if not evs_ref_auto: continue
                            powers_auto = [e['peak_power'] for e in evs_ref_auto if 'peak_power' in e and np.isfinite(e['peak_power'])]
                            if not powers_auto: continue
                            mean_pow_auto = np.mean(powers_auto)
                            if mean_pow_auto > max_pow_auto: max_pow_auto, current_ref_ch_auto = mean_pow_auto, ch_auto
                        selected_ref_ch_auto = current_ref_ch_auto
                        if selected_ref_ch_auto != -1:
                             print(f"    Auto Power Sel {region_auto}: Selected ref ch for shank {shank_idx_iter_auto}: {selected_ref_ch_auto} (Avg Ripple Power: {max_pow_auto:.2f})")
                    if selected_ref_ch_auto is not None and selected_ref_ch_auto != -1:
                        ref_sites_shank_auto[region_auto] = selected_ref_ch_auto
                    else:
                        print(f"    Could not automatically determine reference for {region_auto} on Shank {shank_idx_iter_auto}.")
                if ref_sites_shank_auto:
                    reference_sites[shank_idx_iter_auto] = ref_sites_shank_auto
            reference_channel_set = set(ch_idx for shank_data in reference_sites.values() for ch_idx in shank_data.values() if ch_idx is not None)
            print(f"AUTOMATICALLY selected {len(reference_channel_set)} unique reference channels: {sorted(list(reference_channel_set))}")
            if not reference_channel_set: warnings.warn("AUTOMATIC reference selection resulted in NO channels.")
            del _all_ripple_global_temp_for_ref_auto

        if not reference_sites:
            warnings.warn("CRITICAL: Reference sites were not determined. Subsequent analysis dependent on reference sites will likely fail or be empty.")
        if not reference_channel_set : 
            warnings.warn("No reference channels were selected by any method. Downstream analysis will be affected.")

        print("\n--- Step 9: Generating Timestamps per State per Epoch (for co-occurrence/spectrograms) ---")
        averaged_timestamps_by_state_epoch = {state: {} for state in STATES_TO_ANALYZE} 
        for state_code, state_name in STATES_TO_ANALYZE.items():
            print(f"  Processing state: {state_name}")
            relevant_epochs_for_state = set()
            for ch_events_dict in [ripple_events_by_state_global[state_code], ca1_spwr_events_by_state_global[state_code]]:
                for ch_idx_event_list, events_list in ch_events_dict.items(): 
                    for event in events_list:
                        if 'epoch_idx' in event: relevant_epochs_for_state.add(event['epoch_idx'])
            for epoch_idx in sorted(list(relevant_epochs_for_state)):
                averaged_timestamps_by_state_epoch[state_code][epoch_idx] = {}
                for region in target_regions:
                    regional_ref_indices_for_current_region = [
                        sites[region] for shank, sites in reference_sites.items() if region in sites and sites.get(region) is not None 
                    ]
                    if not regional_ref_indices_for_current_region: continue 
                    epoch_state_region_times = []
                    event_type_desc = "Unknown" 
                    if region == 'CA1':
                        event_type_desc = "SPW Trough (SPW-R)"
                        for ch_ref in regional_ref_indices_for_current_region: 
                            ch_spwr_events = ca1_spwr_events_by_state_global[state_code].get(ch_ref, [])
                            epoch_ch_events = [ev for ev in ch_spwr_events if ev.get('epoch_idx') == epoch_idx and 'associated_spw_trough_sample' in ev]
                            epoch_state_region_times.extend([ev['associated_spw_trough_sample'] for ev in epoch_ch_events])
                    elif region in ['CA2', 'CA3']:
                        event_type_desc = "Ripple Peak"
                        for ch_ref in regional_ref_indices_for_current_region: 
                            ch_ripple_events = ripple_events_by_state_global[state_code].get(ch_ref, [])
                            epoch_ch_events = [ev for ev in ch_ripple_events if ev.get('epoch_idx') == epoch_idx]
                            epoch_state_region_times.extend([ev['peak_sample'] for ev in epoch_ch_events])
                    if epoch_state_region_times:
                        pooled_times = np.sort(np.unique(np.array(epoch_state_region_times, dtype=np.int64))).tolist()
                        averaged_timestamps_by_state_epoch[state_code][epoch_idx][region] = pooled_times
                        print(f"    Epoch {epoch_idx}, State {state_name}, Region {region}: Generated {len(pooled_times)} pooled {event_type_desc} timestamps for downstream analysis.")
        
        print("\n--- Step 10: Detecting Co-occurring Ripples per State per Epoch ---")
        cooccurrence_results_by_state_epoch = {state: {} for state in STATES_TO_ANALYZE}
        window_samples = int(COOCCURRENCE_WINDOW_MS * fs_orig / 1000)
        cooccurrence_pairs = {'CA2': ['CA1', 'CA3'], 'CA3': ['CA1', 'CA2']} 
        for state_code, state_name in STATES_TO_ANALYZE.items():
            print(f"  Processing state: {state_name}")
            cooccurrence_results_by_state_epoch[state_code] = {}
            for epoch_idx in averaged_timestamps_by_state_epoch.get(state_code, {}):
                print(f"    Processing co-occurrence for Epoch {epoch_idx}...")
                cooccurrence_results_by_state_epoch[state_code][epoch_idx] = {}
                for ref_region, target_check_regions in cooccurrence_pairs.items(): 
                    if ref_region not in averaged_timestamps_by_state_epoch[state_code][epoch_idx]: continue
                    ref_timestamps_epoch_state = averaged_timestamps_by_state_epoch[state_code][epoch_idx][ref_region]
                    if len(ref_timestamps_epoch_state) == 0: continue
                    print(f"      Ref: {ref_region} ({len(ref_timestamps_epoch_state)} events)")
                    cooccurrence_results_by_state_epoch[state_code][epoch_idx][ref_region] = {}
                    target_site_indices = {} 
                    for tr_reg in target_check_regions: 
                        target_ch_for_tr_reg_list = [sites[tr_reg] for shank, sites in reference_sites.items() if tr_reg in sites and sites.get(tr_reg) is not None]
                        if target_ch_for_tr_reg_list: target_site_indices[tr_reg] = target_ch_for_tr_reg_list[0] 
                    for target_reg_str, target_ch_idx_cooc in target_site_indices.items(): 
                        target_event_times_epoch_state = []
                        if target_reg_str == 'CA1': 
                            ch_spwr_evs = ca1_spwr_events_by_state_global[state_code].get(target_ch_idx_cooc, [])
                            target_event_times_epoch_state = np.array([ev['peak_sample'] for ev in ch_spwr_evs if ev.get('epoch_idx') == epoch_idx and 'peak_sample' in ev])
                        elif target_reg_str in ['CA2', 'CA3']: 
                            ch_rip_evs = ripple_events_by_state_global[state_code].get(target_ch_idx_cooc, [])
                            target_event_times_epoch_state = np.array([ev['peak_sample'] for ev in ch_rip_evs if ev.get('epoch_idx') == epoch_idx and 'peak_sample' in ev])
                        if len(target_event_times_epoch_state) == 0:
                            cooccurrence_results_by_state_epoch[state_code][epoch_idx][ref_region][target_reg_str] = {'count': 0, 'details': []}; continue
                        co_count, co_details = 0, []
                        target_event_times_epoch_state.sort() 
                        for ref_t in ref_timestamps_epoch_state: 
                            lb, ub = ref_t - window_samples, ref_t + window_samples
                            start_i = np.searchsorted(target_event_times_epoch_state, lb, side='left')
                            end_i = np.searchsorted(target_event_times_epoch_state, ub, side='right')
                            indices = np.arange(start_i, end_i)
                            if len(indices) > 0: 
                                co_count += 1
                                co_details.append({'ref_time_sample': ref_t, 'target_event_times_samples': target_event_times_epoch_state[indices].tolist()})
                        print(f"        Epoch {epoch_idx}, State {state_name}: Found {co_count} co-occurrences {ref_region} -> {target_reg_str} (Ch {target_ch_idx_cooc}).")
                        cooccurrence_results_by_state_epoch[state_code][epoch_idx][ref_region][target_reg_str] = {'count': co_count, 'details': co_details}

        print(f"\n--- Step 11: Calculating Spectrograms per State per Epoch (Parallel using {NUM_CORES} cores) ---")
        averaged_spectrograms_by_state_epoch = {state: {} for state in STATES_TO_ANALYZE}
        for state_code, state_name in STATES_TO_ANALYZE.items():
            print(f"\n  Processing spectrograms for state: {state_name}")
            averaged_spectrograms_by_state_epoch[state_code] = {}
            for epoch_idx in averaged_timestamps_by_state_epoch.get(state_code, {}):
                print(f"    Processing Epoch {epoch_idx}...")
                averaged_spectrograms_by_state_epoch[state_code][epoch_idx] = {}
                for region, timestamps_epoch_state_list in averaged_timestamps_by_state_epoch[state_code][epoch_idx].items():
                    regional_ref_indices_for_spec = [sites[region] for shank, sites in reference_sites.items() if region in sites and sites.get(region) is not None] 
                    if not regional_ref_indices_for_spec or len(timestamps_epoch_state_list) == 0: continue
                    spec_ch_for_region = regional_ref_indices_for_spec[0] 
                    print(f"      Calculating {len(timestamps_epoch_state_list)} spectrograms for Region {region} (Ref Ch {spec_ch_for_region})...")
                    spec_tasks = [(ts, spec_ch_for_region, lfp_filepath, n_samples_orig, n_channels, fs_orig, uv_scale_factor) for ts in timestamps_epoch_state_list] 
                    region_specs_results = Parallel(n_jobs=NUM_CORES, backend="loky")(delayed(_calculate_spectrogram_worker)(*task) for task in spec_tasks)
                    valid_specs = [s for s in region_specs_results if s is not None and not np.isnan(s).all()]
                    if valid_specs:
                        try:
                            avg_spec = np.nanmean(np.stack(valid_specs, axis=0), axis=0)
                            averaged_spectrograms_by_state_epoch[state_code][epoch_idx][region] = avg_spec
                            print(f"      Epoch {epoch_idx}, State {state_name}, Region {region}: Averaged {len(valid_specs)}/{len(timestamps_epoch_state_list)} spectrograms.")
                        except MemoryError: warnings.warn(f"MemoryError stacking spectrograms for Epoch {epoch_idx}, State {state_name}, Region {region}.")
                        except Exception as e_spec_avg: warnings.warn(f"Error averaging spectrograms for Epoch {epoch_idx}, State {state_name}, Region {region}: {e_spec_avg}")
                    else: print(f"      Epoch {epoch_idx}, State {state_name}, Region {region}: No valid spectrograms generated.")
                    del region_specs_results, valid_specs; gc.collect()

        print("\n--- Filtering events to keep only reference channels for detailed CSV saving ---")
        ripple_events_by_state_global_ref = {state: {} for state in STATES_TO_ANALYZE}
        ca1_spwr_events_by_state_global_ref = {state: {} for state in STATES_TO_ANALYZE}
        for state_code in STATES_TO_ANALYZE:
            ripple_events_by_state_global_ref[state_code] = {ch_ref_filter: evs for ch_ref_filter, evs in ripple_events_by_state_global[state_code].items() if ch_ref_filter in reference_channel_set} 
            ca1_spwr_events_by_state_global_ref[state_code] = {ch_ref_filter: evs for ch_ref_filter, evs in ca1_spwr_events_by_state_global[state_code].items() if ch_ref_filter in reference_channel_set} 

        if PLOT_LFP_EXAMPLES:
            print(f"\n--- Step 11.V: Plotting LFP Traces for Example Events (Max {MAX_PLOTS_PER_REF_SITE_STATE} per ref site/state) ---")
            for state_code, state_name in STATES_TO_ANALYZE.items():
                print(f"  Plotting for state: {state_name}")
                events_to_plot_source = {} 
                for ch_idx_ref in reference_channel_set:
                    if channel_region_map.get(ch_idx_ref) == 'CA1' and ch_idx_ref in ca1_spwr_events_by_state_global_ref[state_code]:
                        events_to_plot_source[ch_idx_ref] = (ca1_spwr_events_by_state_global_ref[state_code][ch_idx_ref], "SPWR")
                    elif ch_idx_ref in ripple_events_by_state_global_ref[state_code]: 
                        events_to_plot_source[ch_idx_ref] = (ripple_events_by_state_global_ref[state_code][ch_idx_ref], "Ripple")
                for ch_idx_plot, (events_list_plot, event_label_prefix_plot) in events_to_plot_source.items():
                    plotted_count_for_this_ch_state = 0
                    try: events_list_plot.sort(key=lambda x: x.get('peak_power_zscore', -np.inf), reverse=True)
                    except: pass 
                    for an_event_plot in events_list_plot:
                        if plotted_count_for_this_ch_state >= MAX_PLOTS_PER_REF_SITE_STATE: break
                        event_peak_samp_plot = an_event_plot['peak_sample']
                        event_start_samp_plot = an_event_plot.get('start_sample')
                        event_end_samp_plot = an_event_plot.get('end_sample')
                        plot_file_base_name = (f"{output_filename_base}_epoch{an_event_plot.get('epoch_idx','All')}_"
                                             f"{state_name}_ch{ch_idx_plot}_{event_label_prefix_plot}_peak{event_peak_samp_plot}")
                        plot_lfp_trace_with_event(lfp_filepath, n_channels, n_samples_orig, ch_idx_plot, event_peak_samp_plot,
                                                  event_label_prefix_plot, LFP_PLOT_WINDOW_MS, fs_orig, uv_scale_factor,
                                                  output_dir_lfp_plots, plot_file_base_name,
                                                  event_start_sample_orig=event_start_samp_plot, event_end_sample_orig=event_end_samp_plot)
                        plotted_count_for_this_ch_state += 1
                    if plotted_count_for_this_ch_state > 0:
                        print(f"    Ch {ch_idx_plot} ({channel_region_map.get(ch_idx_plot, 'N/A')}): Plotted {plotted_count_for_this_ch_state} {event_label_prefix_plot} traces.")

        print("\n--- Step 11.5: Saving Detailed Ripple Events to CSV (Reference Channels Only) ---")
        csv_columns = ['epoch_idx', 'channel_idx', 'region', 'state_code', 'state_name',
                       'start_sample', 'peak_sample', 'end_sample', 'trough_sample',
                       'duration_ms', 'peak_power_uV2', 'peak_power_zscore', 'peak_phase_rad', 'iri_ms', 
                       'is_spwr', 'associated_spw_trough_sample']
        for state_code, state_name in STATES_TO_ANALYZE.items():
            all_state_ripple_data_for_csv = []
            state_suffix_csv = f"_{state_name}_RefCh" 
            print(f"  Preparing CSV data for state: {state_name} (Reference Channels)")
            for ch_idx_csv, events_csv in ripple_events_by_state_global_ref[state_code].items(): 
                ch_region_csv = channel_region_map.get(ch_idx_csv, 'Unknown')
                spwr_peaks_state_ch_csv = set()
                if ch_idx_csv in ca1_spwr_events_by_state_global_ref.get(state_code, {}):
                    spwr_peaks_state_ch_csv = {ev_csv['peak_sample'] for ev_csv in ca1_spwr_events_by_state_global_ref[state_code][ch_idx_csv]}
                for ev_csv_item in events_csv:
                    row = { 'epoch_idx': ev_csv_item.get('epoch_idx', np.nan), 'channel_idx': ch_idx_csv, 'region': ch_region_csv,
                            'state_code': state_code, 'state_name': state_name, 'start_sample': ev_csv_item.get('start_sample', np.nan),
                            'peak_sample': ev_csv_item.get('peak_sample', np.nan), 'end_sample': ev_csv_item.get('end_sample', np.nan),
                            'trough_sample': ev_csv_item.get('trough_sample', np.nan), 'duration_ms': ev_csv_item.get('duration_ms', np.nan),
                            'peak_power_uV2': ev_csv_item.get('peak_power', np.nan), 'peak_power_zscore': ev_csv_item.get('peak_power_zscore', np.nan),
                            'peak_phase_rad': ev_csv_item.get('peak_phase', np.nan), 'iri_ms': ev_csv_item.get('iri_ms', np.nan),
                            'is_spwr': (ev_csv_item.get('peak_sample', -1) in spwr_peaks_state_ch_csv) if ch_region_csv == 'CA1' else False,
                            'associated_spw_trough_sample': ev_csv_item.get('associated_spw_trough_sample', np.nan) if ch_region_csv == 'CA1' else np.nan
                    }
                    all_state_ripple_data_for_csv.append(row)
            if all_state_ripple_data_for_csv:
                df_state_csv = pd.DataFrame(all_state_ripple_data_for_csv)
                if 'epoch_idx' in df_state_csv.columns and 'peak_sample' in df_state_csv.columns:
                    df_state_csv = df_state_csv.sort_values(by=['epoch_idx', 'peak_sample']).reset_index(drop=True)
                csv_filename_path = output_path / f'{output_filename_base}_ripple_details{state_suffix_csv}.csv'
                try: df_state_csv.to_csv(csv_filename_path, index=False, columns=csv_columns, na_rep='NaN')
                except Exception as e_csv: print(f"  Error saving CSV for state {state_name}: {e_csv}")
            else: print(f"  No ripple events found for state {state_name} on reference channels to save to CSV.")

        print("\n--- Step 11.6: Saving Ripple Rates to CSV ---")
        rate_data_for_csv = []
        rate_csv_columns = ['epoch_idx', 'state_code', 'state_name', 'state_duration_sec', 'ripple_count', 'ripple_rate_hz']
        for state_code, state_name in STATES_TO_ANALYZE.items():
            for epoch_idx_rate, rate_val in ripple_rates_by_state_epoch.get(state_code, {}).items():
                duration_rate = state_duration_in_epoch.get(state_code, {}).get(epoch_idx_rate, np.nan)
                count_rate = ripple_counts_by_state_epoch.get(state_code, {}).get(epoch_idx_rate, 0) 
                rate_data_for_csv.append({ 'epoch_idx': epoch_idx_rate, 'state_code': state_code, 'state_name': state_name,
                    'state_duration_sec': duration_rate, 'ripple_count': count_rate, 'ripple_rate_hz': rate_val })
        if rate_data_for_csv:
            df_rates_csv = pd.DataFrame(rate_data_for_csv)
            df_rates_csv = df_rates_csv.sort_values(by=['epoch_idx', 'state_code']).reset_index(drop=True)
            rate_csv_filename_path = output_path / f'{output_filename_base}_ripple_rates_by_state_epoch.csv'
            try: df_rates_csv.to_csv(rate_csv_filename_path, index=False, columns=rate_csv_columns, na_rep='NaN')
            except Exception as e_rate_csv: print(f"  Error saving ripple rates CSV: {e_rate_csv}")
        else: print("  No ripple rate data to save to CSV.")

        print("\n--- Step 11.7: Consolidating All Event Timestamps for Final Output ---")
        all_final_event_timestamps_list = []
        for state_code, state_name in STATES_TO_ANALYZE.items():
            for epoch_idx, regions_data in averaged_timestamps_by_state_epoch.get(state_code, {}).items():
                for region, timestamps_list in regions_data.items():
                    all_final_event_timestamps_list.extend(timestamps_list)
        if all_final_event_timestamps_list:
            consolidated_event_timestamps = np.sort(np.unique(np.array(all_final_event_timestamps_list, dtype=np.int64))).tolist()
            print(f"  Generated a single list of {len(consolidated_event_timestamps)} unique event timestamps across all states, epochs, and reference regions.")
        else:
            consolidated_event_timestamps = []
            print("  No event timestamps found to consolidate for the final output file.")

        print("\n--- Step 12: Saving NPY Results (Reference Channels Only for Events) ---")
        np.save(output_path / f'{output_filename_base}_ripple_references_global.npy', reference_sites, allow_pickle=True)
        region_channels_df.to_csv(output_path / f'{output_filename_base}_ripple_analyzed_channels.csv', index=False)
        np.save(output_path / f'{output_filename_base}_consolidated_event_timestamps.npy', consolidated_event_timestamps, allow_pickle=True)
        print(f"Saved consolidated event timestamps to: {output_path / f'{output_filename_base}_consolidated_event_timestamps.npy'}")
        for state_code, state_name in STATES_TO_ANALYZE.items():
            state_suffix_refch = f"_{state_name}_RefCh" 
            state_suffix_analysis = f"_{state_name}" 
            np.save(output_path / f'{output_filename_base}_ripple_events{state_suffix_refch}_global.npy', ripple_events_by_state_global_ref[state_code], allow_pickle=True)
            np.save(output_path / f'{output_filename_base}_ca1_spwr_events{state_suffix_refch}_global.npy', ca1_spwr_events_by_state_global_ref[state_code], allow_pickle=True)
            np.save(output_path / f'{output_filename_base}_ripple_cooccurrence{state_suffix_analysis}_by_epoch.npy', cooccurrence_results_by_state_epoch[state_code], allow_pickle=True)
            np.save(output_path / f'{output_filename_base}_ripple_avg_spectrograms{state_suffix_analysis}_by_epoch.npy', averaged_spectrograms_by_state_epoch[state_code], allow_pickle=True)
            np.save(output_path / f'{output_filename_base}_ripple_rates{state_suffix_analysis}_by_epoch.npy', ripple_rates_by_state_epoch[state_code], allow_pickle=True) 
        print(f"NPY results saved to {output_path} with prefix '{output_filename_base}'")

    except Exception as e:
        print(f"\n!!!!!!!!!!!!!!!! ERROR PROCESSING FILE: {lfp_filepath.name} !!!!!!!!!!!!!!")
        print(f"Error details: {e}"); traceback.print_exc()
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
    finally:
        print("\n--- Cleaning up ---"); gc.collect()
        print("--- Analysis Complete ---")

# --- Script Execution (from original script) ---
if __name__ == "__main__":
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
    if not sleep_state_file_str: print("Sleep state file required for this analysis. Exiting."); exit() 
    SLEEP_STATE_FILE = Path(sleep_state_file_str); print(f"Selected Sleep State file: {SLEEP_STATE_FILE.name}")
    print("Select Epoch Boundaries file (*_epoch_boundaries*.npy) (Optional)...")
    epoch_boundaries_file_str = filedialog.askopenfilename(title="Select Epoch Boundaries File (Optional)", filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")])
    EPOCH_BOUNDARIES_FILE = Path(epoch_boundaries_file_str) if epoch_boundaries_file_str else None; print(f"Selected Epoch Boundaries file: {EPOCH_BOUNDARIES_FILE.name if EPOCH_BOUNDARIES_FILE else 'None'}")
    print("Select Output directory...")
    output_dir_str = filedialog.askdirectory(title="Select Output Directory")
    if not output_dir_str: print("Cancelled."); exit()
    OUTPUT_DIRECTORY = Path(output_dir_str)
    root.destroy()

    if not LFP_FILE.is_file(): print(f"Error: Selected LFP file not found: {LFP_FILE}"); exit()
    if not META_FILE.is_file(): print(f"Error: Selected Meta file not found: {META_FILE}"); exit()
    if not CHANNEL_INFO_FILE.is_file(): print(f"Error: Channel Info file not found: {CHANNEL_INFO_FILE}"); exit()
    if not SLEEP_STATE_FILE.is_file(): print(f"Error: Sleep State file not found: {SLEEP_STATE_FILE}"); exit() 
    if EPOCH_BOUNDARIES_FILE and not EPOCH_BOUNDARIES_FILE.is_file(): print(f"Error: Epoch Boundaries file not found: {EPOCH_BOUNDARIES_FILE}"); exit()
    try: OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True); print(f"Output directory: {OUTPUT_DIRECTORY}")
    except Exception as e: print(f"Error creating output directory: {e}"); exit()

    print(f"\n{'='*50}\nStarting processing for: {LFP_FILE.name}\n{'='*50}")
    run_ripple_analysis(lfp_filepath=LFP_FILE, meta_filepath=META_FILE, channel_info_filepath=CHANNEL_INFO_FILE,
                        sleep_state_filepath=SLEEP_STATE_FILE, epoch_boundaries_filepath=EPOCH_BOUNDARIES_FILE,
                        output_dir=OUTPUT_DIRECTORY)
    print(f"\n{'='*50}\nProcessing complete.\n{'='*50}")