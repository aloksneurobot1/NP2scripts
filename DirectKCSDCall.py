# -*- coding: utf-8 -*-
"""
Created on Wed May 14 17:04:43 2025

@author: HT_bo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy import signal
import quantities as pq
import neo
from pathlib import Path
import sys
import traceback
import os
import gc
# Removed: from numpy import LinAlgError 

# --- Try to import readMeta ---
try:
    # Assuming DemoReadSGLXData is in a location accessible by Python's import system
    # or in the same directory. If it's a module within a package, adjust the import.
    from DemoReadSGLXData.readSGLX import readMeta
except ImportError:
    print("ERROR: Could not import readMeta from DemoReadSGLXData.readSGLX.")
    print("Please ensure the 'DemoReadSGLXData' directory and 'readSGLX.py' are accessible,")
    print("or adjust the PYTHONPATH if necessary.")
    sys.exit(1)

# Import KCSD1D directly for more control
from elephant.current_source_density_src.KCSD import KCSD1D
# The estimate_csd wrapper is not used for KCSD1D in this revised script,
# but could be kept if other CSD methods are planned to be used via it.
# from elephant.current_source_density import estimate_csd


# --- Configuration Parameters ---
LFP_BIN_FILE_PATH_DEFAULT = "your_lfp_file.lf.bin"
LFP_META_FILE_PATH_DEFAULT = "your_lfp_file.lf.meta"
CHANNEL_INFO_CSV_PATH_DEFAULT = "your_channel_info.csv"
TIMESTAMPS_NPY_PATH_DEFAULT = "your_timestamps.nidq_timestamps.npy"

OUTPUT_DIR = Path("./csd_kcsd_output_epoch_f32_v5_final_direct_kcsd") 
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_FS_CSD = 1250.0  # Hz
LFP_BAND_LOWCUT_CSD = 1.0  # Hz
LFP_BAND_HIGHCUT_CSD = 300.0 # Hz
NUMTAPS_CSD_FILTER = 101 # Odd number

CSD_SIGMA_CONDUCTIVITY = 0.3  # Siemens per meter (S/m) -> will be passed as 'sigma'
KCSD_LAMBDAS_CV = np.logspace(-7, -2, 9)
KCSD_RS_CV_UM = np.logspace(np.log10(20), np.log10(500), 9) # in micrometers

EPOCH_SUB_CHUNK_DURATION_SECONDS = 10

# --- Voltage Scaling Function ---
def get_voltage_scaling_factor(meta):
    """Calculates the factor to convert int16 ADC values to microvolts (uV)."""
    try:
        v_max = float(meta['imAiRangeMax'])
        i_max_adc_val = float(meta['imMaxInt'])
        probe_type = int(meta.get('imDatPrb_type', 0))
        lfp_gain = None

        if probe_type in [21, 24, 2013]: # NP 2.0 single shank, NP 2.0 four shank, NP Ultra
            lfp_gain = 80.0
        else: # Includes NP 1.0 (type 0 or other)
            general_lfp_gain_key_str = "~imChanLFGain"
            if general_lfp_gain_key_str in meta:
                 lfp_gain = float(meta[general_lfp_gain_key_str])
            else:
                first_lfp_gain_key_found = None
                sorted_keys = sorted([key for key in meta.keys() if key.startswith('imChanLFGain') and key.endswith('lfGain')])
                if sorted_keys:
                    first_lfp_gain_key_found = sorted_keys[0]

                if first_lfp_gain_key_found:
                    lfp_gain = float(meta[first_lfp_gain_key_found])
                else:
                    lfp_gain = 250.0
                    print(f"  Probe type {probe_type}. No specific LFP gain key ('~imChanLFGain' or 'imChanLFGain*') found in meta.")
                    print(f"  Defaulting LFP gain to {lfp_gain} (Common for NP1.0). PLEASE VERIFY THIS IS CORRECT FOR YOUR PROBE AND RECORDING SYSTEM.")

        if lfp_gain is None: raise ValueError("LFP gain could not be determined.")
        if i_max_adc_val == 0 or lfp_gain == 0: raise ValueError("i_max_adc_val or LFP gain is zero, cannot calculate scaling factor.")
        scaling_factor_uv = (v_max / i_max_adc_val) * (1.0 / lfp_gain) * 1e6
        print(f"  Calculated uV scaling factor: {scaling_factor_uv:.6f} (v_max={v_max}, i_max_adc={i_max_adc_val}, lfp_gain={lfp_gain})")
        return scaling_factor_uv
    except Exception as e:
        print(f"Error calculating voltage scaling factor: {e}"); traceback.print_exc(); return None

# --- LFP Data Loading Function (Returns memmap) ---
def load_lfp_data_sglx_memmap(bin_file_path_str, meta_file_path_str):
    bin_file_path = Path(bin_file_path_str)
    meta_file_path = Path(meta_file_path_str)
    print(f"Setting up LFP data access from: {bin_file_path}")
    print(f"Using metadata from: {meta_file_path}")
    try:
        meta = readMeta(meta_file_path)
        fs_orig = float(meta['imSampRate'])
        n_channels_total = int(meta['nSavedChans'])
        print(f"  Meta: {n_channels_total} channels, FS: {fs_orig:.2f} Hz.")
        uv_scale_factor = get_voltage_scaling_factor(meta)
        if uv_scale_factor is None: print("  Warning: Voltage scaling factor could not be determined. LFP data will be in ADC units if scaling is skipped.")

        file_size = bin_file_path.stat().st_size
        item_size = np.dtype('int16').itemsize
        if n_channels_total == 0 or item_size == 0: raise ValueError("Meta: Invalid nSavedChans or itemsize (must be > 0).")
        num_samples_in_file = file_size // (n_channels_total * item_size)
        if file_size % (n_channels_total * item_size) != 0:
            print(f"  Warning: File size is not an integer multiple of (n_channels_total * item_size).")
            print(f"  This might indicate a corrupted file or incorrect metadata. Using {num_samples_in_file} full samples.")
        if num_samples_in_file <= 0: raise ValueError("Zero or negative number of samples calculated from file size. Check metadata and file integrity.")
        print(f"  Total samples in LFP file: {num_samples_in_file}. Memmap shape: ({num_samples_in_file}, {n_channels_total})")
        lfp_data_memmap = np.memmap(bin_file_path, dtype='int16', mode='r', shape=(num_samples_in_file, n_channels_total))
        print(f"  Successfully memory-mapped LFP data.")
        return lfp_data_memmap, fs_orig, n_channels_total, uv_scale_factor, num_samples_in_file
    except Exception as e:
        print(f"Error in load_lfp_data_sglx_memmap: {e}"); traceback.print_exc(); raise

# --- Channel Info Loading ---
def load_channel_info_kcsd(csv_filepath_str):
    csv_filepath = Path(csv_filepath_str)
    print(f"Loading channel info from {csv_filepath}")
    try:
        channel_df = pd.read_csv(csv_filepath)
        required_cols = ['global_channel_index', 'shank_index', 'ycoord_on_shank_um']
        if not all(col in channel_df.columns for col in required_cols):
            raise ValueError(f"Channel info CSV must contain columns: {required_cols}")
        for col in ['global_channel_index', 'shank_index']:
            channel_df[col] = pd.to_numeric(channel_df[col], errors='coerce').astype('Int64')
        channel_df['ycoord_on_shank_um'] = pd.to_numeric(channel_df['ycoord_on_shank_um'], errors='coerce').astype(float)
        channel_df.dropna(subset=required_cols, inplace=True)
        for col in ['global_channel_index', 'shank_index']:
            channel_df[col] = channel_df[col].astype(int)
        print(f"Loaded and validated channel info for {len(channel_df)} channels.")
        return channel_df
    except Exception as e:
        print(f"Error loading channel info: {e}"); traceback.print_exc(); sys.exit(1)

# --- Timestamp and Epoch Loading (Time-based) ---
def load_epoch_definitions_from_times(timestamps_npy_path_str, fs_lfp, total_lfp_samples_in_file):
    timestamps_npy_path = Path(timestamps_npy_path_str)
    print(f"Loading epoch definitions (time-based) from: {timestamps_npy_path}")
    if not timestamps_npy_path.exists():
        print(f"Error: Timestamps NPY file not found: {timestamps_npy_path}")
        return None
    try:
        loaded_npy = np.load(timestamps_npy_path, allow_pickle=True)
        
        data_to_process = None
        epoch_info_list = None

        if isinstance(loaded_npy, dict):
            data_to_process = loaded_npy
            print("  Loaded NPY as a direct dictionary.")
        elif isinstance(loaded_npy, np.ndarray) and loaded_npy.ndim == 0 and \
             hasattr(loaded_npy, 'item') and isinstance(loaded_npy.item(), dict):
            data_to_process = loaded_npy.item()
            print("  Loaded NPY as a 0-dim array; extracted dictionary item.")
        
        if data_to_process is not None and 'EpochFrameData' in data_to_process:
            epoch_info_list = data_to_process['EpochFrameData']
            if not isinstance(epoch_info_list, list):
                print(f"  Error: 'EpochFrameData' was found but it is not a list. Type: {type(epoch_info_list)}")
                return None
            print(f"  Found 'EpochFrameData' with {len(epoch_info_list)} entries.")
        
        elif isinstance(loaded_npy, np.ndarray) and loaded_npy.dtype.fields and \
             all(f in loaded_npy.dtype.fields for f in ['start_time_sec', 'end_time_sec', 'epoch_index']):
            epoch_info_list = list(loaded_npy) 
            print(f"  Interpreting loaded NPY as a structured array of {len(epoch_info_list)} epochs.")

        if epoch_info_list is None:
            print("Error: Timestamps NPY file format not recognized or 'EpochFrameData' not found in the expected structure.")
            print(f"  Loaded NPY type: {type(loaded_npy)}")
            if isinstance(loaded_npy, np.ndarray):
                print(f"  Numpy array details - shape: {loaded_npy.shape}, dtype: {loaded_npy.dtype}")
                if loaded_npy.ndim == 0 and hasattr(loaded_npy, 'item'):
                    try:
                        item_content = loaded_npy.item()
                        print(f"  Item from 0-dim array: {type(item_content)}")
                        if isinstance(item_content, dict):
                            print(f"  Keys in item dict: {list(item_content.keys())}")
                    except Exception as e_item:
                        print(f"  Could not retrieve or inspect item from 0-dim array: {e_item}")
            return None

        epochs = []
        for item_idx, epoch_info_item in enumerate(epoch_info_list):
            start_sec_val, end_sec_val, epoch_idx_val = None, None, None
            if isinstance(epoch_info_item, dict):
                start_sec_val = epoch_info_item.get('start_time_sec')
                end_sec_val = epoch_info_item.get('end_time_sec')
                epoch_idx_val = epoch_info_item.get('epoch_index')
            elif hasattr(epoch_info_item, 'dtype') and epoch_info_item.dtype.fields: 
                try:
                    start_sec_val = epoch_info_item['start_time_sec']
                    end_sec_val = epoch_info_item['end_time_sec']
                    epoch_idx_val = epoch_info_item['epoch_index']
                except (IndexError, KeyError) as e_field: 
                     print(f"  Warning: Missing expected field in structured array epoch item {item_idx}: {e_field}")
                     continue
            else:
                print(f"  Warning: Epoch info item {item_idx} format not recognized: {type(epoch_info_item)}")
                continue

            if start_sec_val is not None and end_sec_val is not None and epoch_idx_val is not None:
                start_sec = float(start_sec_val)
                end_sec = float(end_sec_val)
                epoch_abs_start_sample = int(round(start_sec * fs_lfp))
                epoch_abs_end_sample_inclusive = int(round(end_sec * fs_lfp)) -1
                if epoch_abs_end_sample_inclusive < epoch_abs_start_sample:
                     epoch_abs_end_sample_inclusive = epoch_abs_start_sample
                epoch_abs_start_sample_capped = max(0, epoch_abs_start_sample)
                epoch_abs_end_sample_inclusive_capped = min(epoch_abs_end_sample_inclusive, total_lfp_samples_in_file - 1)

                if epoch_abs_start_sample_capped > epoch_abs_end_sample_inclusive_capped or \
                   epoch_abs_start_sample_capped >= total_lfp_samples_in_file:
                    print(f"  Warning: Epoch {epoch_idx_val} (Requested: {start_sec:.3f}s - {end_sec:.3f}s) "
                          f"results in invalid/zero-duration sample range "
                          f"({epoch_abs_start_sample_capped} - {epoch_abs_end_sample_inclusive_capped}) "
                          f"after time conversion/capping. Skipping.")
                    continue
                
                epochs.append({
                    'epoch_index': int(epoch_idx_val),
                    'abs_start_sample': epoch_abs_start_sample_capped,
                    'abs_end_sample_inclusive': epoch_abs_end_sample_inclusive_capped,
                    'duration_sec_requested': end_sec - start_sec,
                    'duration_sec_actual': (epoch_abs_end_sample_inclusive_capped - epoch_abs_start_sample_capped + 1) / fs_lfp if fs_lfp > 0 else 0
                })
            else: print(f"  Warning: Epoch info item {item_idx} incomplete (missing start/end/index values).")
        
        if not epochs: print("No valid epoch definitions constructed from timestamps NPY."); return None
        epochs.sort(key=lambda e: e['abs_start_sample'])
        print(f"Loaded and validated {len(epochs)} epoch definitions based on times.")
        for ep_idx_print, ep in enumerate(epochs):
            print(f"  Using Epoch {ep['epoch_index']} (Sorted Idx {ep_idx_print}): "
                  f"Samples {ep['abs_start_sample']}-{ep['abs_end_sample_inclusive']} "
                  f"(Duration: {ep['duration_sec_actual']:.3f}s)")
        return epochs
    except Exception as e:
        print(f"Error loading/parsing timestamps NPY: {e}"); traceback.print_exc(); return None

# --- LFP Preprocessing for a SUB-CHUNK (using float32 for intermediate) ---
def preprocess_lfp_sub_chunk(lfp_sub_chunk_scaled_f32, fs_sub_chunk_in,
                             lowcut, highcut, numtaps, consistent_downsampling_factor,
                             target_overall_fs): 
    if lfp_sub_chunk_scaled_f32.ndim == 1:
        lfp_sub_chunk_scaled_f32 = lfp_sub_chunk_scaled_f32[:, np.newaxis]
    current_fs = fs_sub_chunk_in
    lfp_to_process_f32 = lfp_sub_chunk_scaled_f32
    if consistent_downsampling_factor > 1:
        min_samples_for_decimate = consistent_downsampling_factor * 30 
        if lfp_to_process_f32.shape[0] > min_samples_for_decimate:
            try:
                lfp_sub_chunk_downsampled_f64 = signal.decimate(lfp_to_process_f32.astype(np.float64),
                                                                consistent_downsampling_factor, axis=0, ftype='fir', zero_phase=True)
                lfp_to_process_f32 = lfp_sub_chunk_downsampled_f64.astype(np.float32)
                current_fs /= consistent_downsampling_factor
            except Exception as e_decimate:
                print(f"    Warning: Error during sub-chunk decimation (factor {consistent_downsampling_factor}): {e_decimate}. Using original sampling for this sub-chunk.")
        else:
            print(f"    Info: Sub-chunk too short ({lfp_to_process_f32.shape[0]} samples) for decimation by factor {consistent_downsampling_factor}. Skipping downsampling for this sub-chunk.")
    nyq = current_fs / 2.0
    actual_highcut = min(highcut, nyq * 0.98) 
    actual_lowcut = max(lowcut, 0.001) 
    current_numtaps = numtaps
    if current_numtaps >= lfp_to_process_f32.shape[0]:
        current_numtaps = lfp_to_process_f32.shape[0] - 1 
    if current_numtaps % 2 == 0 and current_numtaps > 0: 
        current_numtaps -=1
    if current_numtaps < 3 or actual_lowcut >= actual_highcut: 
        lfp_sub_chunk_filtered_f32 = lfp_to_process_f32
    else:
        try:
            fir_taps = signal.firwin(current_numtaps, [actual_lowcut, actual_highcut], 
                                     fs=current_fs, pass_zero='bandpass', window='hamming')
            lfp_sub_chunk_filtered_f64 = signal.filtfilt(fir_taps, 1.0, lfp_to_process_f32.astype(np.float64), axis=0)
            lfp_sub_chunk_filtered_f32 = lfp_sub_chunk_filtered_f64.astype(np.float32)
        except Exception as e_filter:
            print(f"    Warning: Error during sub-chunk filtering: {e_filter}. Using data before this filtering step for this sub-chunk.")
            lfp_sub_chunk_filtered_f32 = lfp_to_process_f32
    return lfp_sub_chunk_filtered_f32, current_fs

# --- Main CSD Analysis Script ---
def main_kcsd_analysis(lfp_bin_file, lfp_meta_file, channel_info_csv, timestamps_npy_file):
    base_filename = Path(lfp_bin_file).stem.replace('.lf', '') 
    output_file_prefix = OUTPUT_DIR / base_filename
    lfp_data_memmap_obj = None
    try:
        lfp_data_memmap_obj, fs_orig, n_channels_total_in_file, uv_scale_factor_loaded, num_samples_in_lfp_file = \
            load_lfp_data_sglx_memmap(lfp_bin_file, lfp_meta_file)
    except Exception as e:
        print(f"CRITICAL: Failed to load LFP data: {e}"); return

    epoch_definitions = load_epoch_definitions_from_times(timestamps_npy_file, fs_orig, num_samples_in_lfp_file)
    if not epoch_definitions:
        print("No valid epoch definitions. Exiting.");
        if lfp_data_memmap_obj is not None and hasattr(lfp_data_memmap_obj, '_mmap') and lfp_data_memmap_obj._mmap is not None:
            try: lfp_data_memmap_obj._mmap.close(); del lfp_data_memmap_obj
            except Exception: pass
        gc.collect()
        return
        
    if fs_orig <= TARGET_FS_CSD or abs(fs_orig - TARGET_FS_CSD) < 1e-3 : 
        overall_downsampling_factor = 1
    else:
        overall_downsampling_factor = int(round(fs_orig / TARGET_FS_CSD))
        if overall_downsampling_factor <= 0: overall_downsampling_factor = 1 
    final_effective_fs_for_csd_nominal = fs_orig / overall_downsampling_factor
    print(f"Original LFP FS: {fs_orig:.2f} Hz. Target CSD FS: {TARGET_FS_CSD:.2f} Hz.")
    print(f"Calculated overall downsampling factor: {overall_downsampling_factor}. Nominal final effective CSD FS: {final_effective_fs_for_csd_nominal:.2f} Hz")

    try:
        probe_channel_df = load_channel_info_kcsd(channel_info_csv)
        unique_shanks = sorted(probe_channel_df['shank_index'].unique())
        num_shanks_found = len(unique_shanks)
        if num_shanks_found == 0: print("No shanks found in channel info. Exiting."); return

        for epoch in epoch_definitions:
            epoch_idx = epoch['epoch_index']
            epoch_abs_start_sample = epoch['abs_start_sample']
            epoch_abs_end_sample_inclusive = epoch['abs_end_sample_inclusive']
            epoch_abs_end_slice = epoch_abs_end_sample_inclusive + 1
            print(f"\n>>> Processing Epoch {epoch_idx} (LFP Samples: {epoch_abs_start_sample} to {epoch_abs_end_sample_inclusive}, Duration: {epoch['duration_sec_actual']:.3f}s) <<<")
            fig_csd_epoch, axs_csd_epoch = plt.subplots(num_shanks_found, 1,
                                                          figsize=(12, 4 * num_shanks_found),
                                                          sharex=True, squeeze=False, constrained_layout=True)
            fig_csd_epoch.suptitle(f'kCSD1D - {base_filename} - Epoch {epoch_idx}', fontsize=16)
            actual_fs_of_processed_data_epoch = None 

            for i_shank, shank_id_val in enumerate(unique_shanks):
                ax_curr_csd_plot = axs_csd_epoch[i_shank, 0]
                print(f"\n  {'='*5} Shank {shank_id_val} (Epoch {epoch_idx}) {'='*5}")
                shank_channel_info = probe_channel_df[probe_channel_df['shank_index'] == shank_id_val].sort_values(
                    by='ycoord_on_shank_um', ascending=True) 
                if shank_channel_info.empty:
                    msg = f"Shank {shank_id_val} (Ep {epoch_idx}) - No Channel Data"
                    print(f"  {msg}"); ax_curr_csd_plot.set_title(msg); ax_curr_csd_plot.text(0.5,0.5, "No channels for shank", ha='center', va='center', transform=ax_curr_csd_plot.transAxes)
                    continue
                shank_global_indices = shank_channel_info['global_channel_index'].values
                electrode_coords_um_shank = shank_channel_info['ycoord_on_shank_um'].values 
                if len(shank_global_indices) < 2: 
                    msg = f"Shank {shank_id_val} (Ep {epoch_idx}) has {len(shank_global_indices)} channels. Needs >= 2 for kCSD."
                    print(f"  {msg} Skipping."); ax_curr_csd_plot.set_title(f"Shank {shank_id_val} (Ep {epoch_idx}) - Not Enough Channels"); ax_curr_csd_plot.text(0.5,0.5, msg, ha='center', va='center', transform=ax_curr_csd_plot.transAxes, wrap=True)
                    continue
                if np.any(shank_global_indices >= n_channels_total_in_file) or np.any(shank_global_indices < 0):
                    out_of_bounds_indices = shank_global_indices[(shank_global_indices >= n_channels_total_in_file) | (shank_global_indices < 0)]
                    msg = (f"Shank {shank_id_val} (Ep {epoch_idx}) - Channel Index Error. "
                           f"Indices {out_of_bounds_indices} out of bounds for {n_channels_total_in_file} total channels.")
                    print(f"  {msg}"); ax_curr_csd_plot.set_title(msg); ax_curr_csd_plot.text(0.5,0.5, "Channel index out of bounds.", ha='center', va='center', transform=ax_curr_csd_plot.transAxes, wrap=True)
                    continue

                processed_sub_chunks_for_shank_epoch = []
                epoch_duration_samples_for_slice = epoch_abs_end_slice - epoch_abs_start_sample
                sub_chunk_size_orig_fs = int(fs_orig * EPOCH_SUB_CHUNK_DURATION_SECONDS)
                if sub_chunk_size_orig_fs <= 0 : sub_chunk_size_orig_fs = epoch_duration_samples_for_slice 
                print(f"    Processing Epoch {epoch_idx}, Shank {shank_id_val} in sub-chunks of ~{EPOCH_SUB_CHUNK_DURATION_SECONDS}s (raw sample size: {sub_chunk_size_orig_fs})")
                sub_chunk_count = 0
                fs_of_first_processed_sub_chunk = None

                for i_sub_chunk_local_start in range(0, epoch_duration_samples_for_slice, sub_chunk_size_orig_fs):
                    sub_chunk_abs_start_in_file = epoch_abs_start_sample + i_sub_chunk_local_start
                    sub_chunk_abs_end_in_file_exclusive = min(sub_chunk_abs_start_in_file + sub_chunk_size_orig_fs, epoch_abs_end_slice)
                    current_sub_chunk_length_raw = sub_chunk_abs_end_in_file_exclusive - sub_chunk_abs_start_in_file
                    if current_sub_chunk_length_raw <= 0: continue
                    sub_chunk_count += 1
                    try:
                        if lfp_data_memmap_obj is None: raise ValueError("LFP data memory map is not available.")
                        temp_processed_channels_for_sub_chunk = []
                        fs_current_sub_chunk_processed = None
                        for ch_idx_in_shank, global_ch_idx in enumerate(shank_global_indices):
                            single_ch_sub_chunk_raw = lfp_data_memmap_obj[sub_chunk_abs_start_in_file:sub_chunk_abs_end_in_file_exclusive, global_ch_idx]
                            single_ch_sub_chunk_f32 = single_ch_sub_chunk_raw.astype(np.float32)
                            if uv_scale_factor_loaded is not None:
                                single_ch_sub_chunk_f32 *= uv_scale_factor_loaded 
                            processed_single_ch_sub_chunk_f32, fs_processed_this_channel_chunk = preprocess_lfp_sub_chunk(
                                single_ch_sub_chunk_f32, fs_sub_chunk_in=fs_orig,
                                lowcut=LFP_BAND_LOWCUT_CSD, highcut=LFP_BAND_HIGHCUT_CSD,
                                numtaps=NUMTAPS_CSD_FILTER,
                                consistent_downsampling_factor=overall_downsampling_factor,
                                target_overall_fs = final_effective_fs_for_csd_nominal)
                            if fs_current_sub_chunk_processed is None: 
                                fs_current_sub_chunk_processed = fs_processed_this_channel_chunk
                            elif abs(fs_current_sub_chunk_processed - fs_processed_this_channel_chunk) > 1e-3:
                                print(f"      Warning: Inconsistent processed FS within sub-chunk {sub_chunk_count} for shank {shank_id_val}. This should not happen.")
                            if processed_single_ch_sub_chunk_f32 is not None and processed_single_ch_sub_chunk_f32.shape[0] > 0:
                                temp_processed_channels_for_sub_chunk.append(processed_single_ch_sub_chunk_f32.flatten())
                            else: 
                                expected_len_ds = int(np.ceil(current_sub_chunk_length_raw / (fs_orig / fs_current_sub_chunk_processed if fs_current_sub_chunk_processed !=0 else 1) )) if fs_current_sub_chunk_processed is not None else 0
                                temp_processed_channels_for_sub_chunk.append(np.full(expected_len_ds, np.nan, dtype=np.float32))
                            del single_ch_sub_chunk_raw, single_ch_sub_chunk_f32, processed_single_ch_sub_chunk_f32 
                        if fs_of_first_processed_sub_chunk is None and fs_current_sub_chunk_processed is not None:
                            fs_of_first_processed_sub_chunk = fs_current_sub_chunk_processed
                            if actual_fs_of_processed_data_epoch is None: 
                                actual_fs_of_processed_data_epoch = fs_current_sub_chunk_processed
                        if temp_processed_channels_for_sub_chunk:
                            min_len = min(len(ch_arr) for ch_arr in temp_processed_channels_for_sub_chunk if ch_arr is not None and ch_arr.size > 0) if \
                                      any(ch_arr is not None and ch_arr.size > 0 for ch_arr in temp_processed_channels_for_sub_chunk) else 0
                            if min_len > 0:
                                aligned_channels = []
                                for ch_arr in temp_processed_channels_for_sub_chunk:
                                    if ch_arr is not None and ch_arr.size >= min_len:
                                        aligned_channels.append(ch_arr[:min_len])
                                    elif ch_arr is not None: 
                                        aligned_channels.append(np.pad(ch_arr, (0, min_len - len(ch_arr)), 'constant', constant_values=np.nan))
                                    else: 
                                        aligned_channels.append(np.full(min_len, np.nan, dtype=np.float32))
                                processed_sub_chunk_for_shank = np.vstack(aligned_channels).T.astype(np.float32) 
                                processed_sub_chunks_for_shank_epoch.append(processed_sub_chunk_for_shank)
                            else:
                                print(f"      Warning: No valid processed channel data with positive length for sub-chunk {sub_chunk_count} of shank {shank_id_val}. Skipping sub-chunk.")
                        del temp_processed_channels_for_sub_chunk; gc.collect()
                    except MemoryError as me_sub_chunk:
                        print(f"    MemoryError processing sub-chunk {sub_chunk_count} for Ep{epoch_idx},Shk{shank_id_val}: {me_sub_chunk}. Skipping rest of this shank/epoch.");
                        processed_sub_chunks_for_shank_epoch = [] 
                        break 
                    except Exception as e_sub_chunk:
                        print(f"    Error processing sub-chunk {sub_chunk_count} for Ep{epoch_idx},Shk{shank_id_val}: {e_sub_chunk}"); traceback.print_exc();
                        continue 
                if not processed_sub_chunks_for_shank_epoch:
                    msg = f"No sub-chunks successfully processed for Epoch {epoch_idx}, Shank {shank_id_val}."
                    print(f"  {msg} Skipping CSD calculation for this shank."); ax_curr_csd_plot.set_title(f"Shank {shank_id_val} (Ep {epoch_idx}) - Sub-chunk Proc. Failed"); ax_curr_csd_plot.text(0.5,0.5, msg, ha='center', va='center', transform=ax_curr_csd_plot.transAxes, wrap=True)
                    continue
                print(f"    Concatenating {len(processed_sub_chunks_for_shank_epoch)} sub-chunks for Ep{epoch_idx},Shk{shank_id_val}...")
                lfp_processed_shank_epoch_full = None
                try:
                    lfp_processed_shank_epoch_full = np.concatenate(processed_sub_chunks_for_shank_epoch, axis=0).astype(np.float32)
                except MemoryError as me_concat_epoch:
                    msg = f"MemoryError concatenating sub-chunks for Ep{epoch_idx},Shk{shank_id_val}: {me_concat_epoch}"
                    print(f"  {msg}"); ax_curr_csd_plot.set_title(f"Shank {shank_id_val} (Ep{epoch_idx}) - Concat Error"); ax_curr_csd_plot.text(0.5,0.5, msg, ha='center',va='center',wrap=True);
                finally:
                    del processed_sub_chunks_for_shank_epoch; gc.collect() 
                if lfp_processed_shank_epoch_full is None:
                    continue 
                final_fs_for_this_shank_csd = fs_of_first_processed_sub_chunk if fs_of_first_processed_sub_chunk is not None else final_effective_fs_for_csd_nominal
                if actual_fs_of_processed_data_epoch is None: actual_fs_of_processed_data_epoch = final_fs_for_this_shank_csd
                print(f"    Final LFP for CSD (Ep{epoch_idx},Shk{shank_id_val}): {lfp_processed_shank_epoch_full.shape}, FS: {final_fs_for_this_shank_csd:.2f} Hz, dtype: {lfp_processed_shank_epoch_full.dtype}")
                coords_quant_um_shank = electrode_coords_um_shank * pq.um 
                ele_pos_for_kcsd_meters = coords_quant_um_shank.rescale(pq.m).magnitude.reshape(-1, 1) 
                lfp_units = pq.uV if uv_scale_factor_loaded is not None else pq.dimensionless 
                lfp_for_kcsd_Volts = (lfp_processed_shank_epoch_full * lfp_units).rescale(pq.V).magnitude.T 
                min_coord_um = electrode_coords_um_shank.min()
                max_coord_um = electrode_coords_um_shank.max()
                xmin_kcsd_meters = (min_coord_um * pq.um).rescale(pq.m).magnitude
                xmax_kcsd_meters = (max_coord_um * pq.um).rescale(pq.m).magnitude
                if np.isclose(xmax_kcsd_meters, xmin_kcsd_meters):
                    print(f"    Warning: xmin and xmax are very close for KCSD ({xmin_kcsd_meters}m). Expanding slightly.")
                    padding_m = (10 * pq.um).rescale(pq.m).magnitude 
                    xmin_kcsd_meters -= padding_m
                    xmax_kcsd_meters += padding_m
                num_csd_est_pts_config = max(32, int(len(electrode_coords_um_shank) * 1.5)) 
                if num_csd_est_pts_config > 1 and (xmax_kcsd_meters - xmin_kcsd_meters) > 1e-9 : 
                    gdx_kcsd_meters = (xmax_kcsd_meters - xmin_kcsd_meters) / (num_csd_est_pts_config - 1)
                else: 
                    gdx_kcsd_meters = (10 * pq.um).rescale(pq.m).magnitude 
                if gdx_kcsd_meters <= 1e-9: 
                    gdx_kcsd_meters = (1 * pq.um).rescale(pq.m).magnitude
                ext_x_kcsd_meters = (0.0 * pq.um).rescale(pq.m).magnitude 
                if KCSD_RS_CV_UM.size > 0:
                    initial_R_meter = (KCSD_RS_CV_UM[0] * pq.um).rescale(pq.m).magnitude
                else: 
                    default_R_fraction = 0.1 
                    span_meters = xmax_kcsd_meters - xmin_kcsd_meters
                    initial_R_meter = default_R_fraction * span_meters if span_meters > 1e-9 else (20 * pq.um).rescale(pq.m).magnitude
                if initial_R_meter <= 1e-9: initial_R_meter = (1 * pq.um).rescale(pq.m).magnitude
                h_kcsd_meters = 1.0 
                kcsd_estimator_obj = None
                csd_result_neo = None
                csd_data_matrix = None
                print(f"    DEBUG: Instantiating KCSD1D for Ep{epoch_idx},Shk{shank_id_val}:")
                print(f"      LFP (transposed for KCSD) shape: {lfp_for_kcsd_Volts.shape}, units: Volts, sampling_rate (for neo): {final_fs_for_this_shank_csd:.2f} Hz")
                print(f"      Ele_pos (for KCSD) shape: {ele_pos_for_kcsd_meters.shape}, units: meters, values (um): {coords_quant_um_shank.magnitude.flatten()}")
                print(f"      KCSD Params: sigma={CSD_SIGMA_CONDUCTIVITY} S/m, n_src_init={num_csd_est_pts_config},") 
                print(f"                   xmin={xmin_kcsd_meters:.3e}m, xmax={xmax_kcsd_meters:.3e}m, gdx={gdx_kcsd_meters:.3e}m,")
                print(f"                   R_init={initial_R_meter:.3e}m, ext_x={ext_x_kcsd_meters:.3e}m, h={h_kcsd_meters:.1f}m")
                try:
                    kcsd_estimator_obj = KCSD1D(
                        ele_pos=ele_pos_for_kcsd_meters,    
                        pots=lfp_for_kcsd_Volts,            
                        sigma=CSD_SIGMA_CONDUCTIVITY,       
                        n_src_init=num_csd_est_pts_config,  
                        xmin=xmin_kcsd_meters,              
                        xmax=xmax_kcsd_meters,              
                        gdx=gdx_kcsd_meters,                
                        ext_x=ext_x_kcsd_meters,            
                        R_init=initial_R_meter,             
                        lambd=0.0,                          
                        h=h_kcsd_meters)
                    Rs_cv_meters = (KCSD_RS_CV_UM * pq.um).rescale(pq.m).magnitude
                    print(f"    Performing cross-validation for Ep{epoch_idx},Shk{shank_id_val} with {len(KCSD_LAMBDAS_CV)} lambdas and {len(Rs_cv_meters)} Rs (meters)...")
                    kcsd_estimator_obj.cross_validate(lambdas=KCSD_LAMBDAS_CV, Rs=Rs_cv_meters)
                    print(f"      CV Optimal Lambda: {kcsd_estimator_obj.lambd:.2e}")
                    print(f"      CV Optimal R: {kcsd_estimator_obj.R * 1e6:.2f} um") 
                    if hasattr(kcsd_estimator_obj, 'cv_error') and kcsd_estimator_obj.cv_error is not None:
                        fig_cv, ax_cv = plt.subplots(figsize=(8, 6))
                        cv_err_to_plot = np.log10(kcsd_estimator_obj.cv_error)
                        if np.isneginf(cv_err_to_plot).any():
                            finite_min = np.min(cv_err_to_plot[np.isfinite(cv_err_to_plot)]) if np.isfinite(cv_err_to_plot).any() else -10
                            cv_err_to_plot[np.isneginf(cv_err_to_plot)] = finite_min - 1 
                        vmin_cv = np.nanmin(cv_err_to_plot) if np.isfinite(cv_err_to_plot).any() else -10
                        vmax_cv = np.nanpercentile(cv_err_to_plot[np.isfinite(cv_err_to_plot)], 99) if np.isfinite(cv_err_to_plot).any() else vmin_cv + 1
                        if vmin_cv >= vmax_cv : vmax_cv = vmin_cv + (1e-9 if abs(vmin_cv) > 1e-10 else 1e-9) 
                        norm_cv = Normalize(vmin=vmin_cv, vmax=vmax_cv)
                        im_cv = ax_cv.imshow(cv_err_to_plot, aspect='auto', origin='lower', cmap='viridis', norm=norm_cv)
                        ax_cv.set_xticks(np.arange(len(KCSD_LAMBDAS_CV)))
                        ax_cv.set_xticklabels([f"{l:.1e}" for l in KCSD_LAMBDAS_CV], rotation=45, ha="right")
                        ax_cv.set_yticks(np.arange(len(KCSD_RS_CV_UM))) 
                        ax_cv.set_yticklabels([f"{r_val:.1f}" for r_val in KCSD_RS_CV_UM]) 
                        best_lambda_idx_list = np.where(np.isclose(KCSD_LAMBDAS_CV, kcsd_estimator_obj.lambd))[0]
                        best_R_idx_list = np.where(np.isclose(KCSD_RS_CV_UM, kcsd_estimator_obj.R * 1e6))[0] 
                        if len(best_lambda_idx_list) > 0 and len(best_R_idx_list) > 0:
                            ax_cv.scatter(best_lambda_idx_list[0], best_R_idx_list[0], marker='x', color='red', s=100, 
                                          label=f'Optimal\nL={kcsd_estimator_obj.lambd:.1e}\nR={kcsd_estimator_obj.R*1e6:.1f}um')
                            ax_cv.legend(fontsize='small', loc='best')
                        else:
                            print("      Warning: Could not find exact match for optimal lambda/R in CV grid for plotting marker.")
                        ax_cv.set_xlabel('Lambda (Regularization Parameter)')
                        ax_cv.set_ylabel('R (Basis Source Radius, µm)')
                        ax_cv.set_title(f'kCSD Cross-Validation Error (log10 scale)\nEpoch {epoch_idx}, Shank {shank_id_val}')
                        plt.colorbar(im_cv, ax=ax_cv, label='log10(CV Error)'); fig_cv.tight_layout()
                        cv_plot_filename = output_file_prefix.parent / f"{base_filename}_ep{epoch_idx}_shk{shank_id_val}_kCSD_CV_Error.png"
                        plt.savefig(cv_plot_filename); print(f"    Saved CV plot: {cv_plot_filename}"); plt.close(fig_cv)
                    else:
                        print("    CV error attribute (.cv_error) not found or is None on estimator after cross_validate call.")
                    csd_profile_values_A_per_m3 = kcsd_estimator_obj.values()
                    csd_profile_values_uA_per_mm3 = csd_profile_values_A_per_m3 * 1e-3
                    csd_estimation_coords_meters = kcsd_estimator_obj.estm_x.reshape(-1) * pq.m
                    csd_result_neo = neo.AnalogSignal(
                        csd_profile_values_uA_per_mm3.T, 
                        units=pq.uA / pq.mm**3,
                        sampling_rate=final_fs_for_this_shank_csd * pq.Hz,
                        coordinates=csd_estimation_coords_meters.rescale(pq.um))
                    csd_data_matrix = np.asarray(csd_result_neo).astype(np.float32) 
                    csd_times_vector = np.arange(csd_data_matrix.shape[0]) / final_fs_for_this_shank_csd
                    csd_positions_plot_um = csd_estimation_coords_meters.rescale(pq.um).magnitude.flatten()
                    csd_units_str = csd_result_neo.units.dimensionality.string 
                    if csd_data_matrix.size == 0:
                        msg = f"Empty CSD result after kCSD.values() (Ep{epoch_idx},Shk{shank_id_val})"
                        print(f"  {msg}"); ax_curr_csd_plot.set_title(msg); ax_curr_csd_plot.text(0.5,0.5,msg,ha='center',va='center',wrap=True);
                        if 'lfp_processed_shank_epoch_full' in locals(): del lfp_processed_shank_epoch_full
                        if kcsd_estimator_obj is not None: del kcsd_estimator_obj
                        if csd_result_neo is not None: del csd_result_neo
                        gc.collect()
                        continue
                    csd_data_for_plotting = csd_data_matrix.T 
                    print(f"      CSD data matrix for plotting (Ep{epoch_idx},Shk{shank_id_val}): {csd_data_for_plotting.shape}, units: {csd_units_str}, dtype: {csd_data_for_plotting.dtype}")
                    if csd_data_for_plotting.size > 0 and np.any(np.isfinite(csd_data_for_plotting)):
                        print(f"      CSD data stats: min={np.nanmin(csd_data_for_plotting):.2e}, max={np.nanmax(csd_data_for_plotting):.2e}, mean={np.nanmean(csd_data_for_plotting):.2e}")
                    abs_csd_finite = np.abs(csd_data_for_plotting[np.isfinite(csd_data_for_plotting)])
                    if abs_csd_finite.size == 0: 
                        print(f"      Warning: CSD data for Ep{epoch_idx},Shk{shank_id_val} is empty or all NaN/Inf after processing. Plot will be blank.")
                        clim_val = 1.0
                        ax_curr_csd_plot.text(0.5, 0.5, "CSD Data Empty/Invalid", ha='center', va='center', transform=ax_curr_csd_plot.transAxes)
                    else:
                        clim_val = np.percentile(abs_csd_finite, 99.0)
                        if clim_val < 1e-9: 
                           max_abs = np.nanmax(abs_csd_finite)
                           clim_val = max_abs if max_abs > 1e-9 else 1.0
                        if abs(clim_val) < 1e-9: clim_val = 1e-9 
                        print(f"      Plotting clim_val: {clim_val:.2e} {csd_units_str}")
                        time_edges = np.concatenate([csd_times_vector, [csd_times_vector[-1] + (1.0 / final_fs_for_this_shank_csd)]]) if len(csd_times_vector) > 0 else np.array([0, 1.0/final_fs_for_this_shank_csd])
                        if len(csd_positions_plot_um) > 1:
                            depth_diffs = np.diff(csd_positions_plot_um)
                            step_val = np.mean(depth_diffs) if len(depth_diffs) > 0 else ((csd_positions_plot_um[-1] - csd_positions_plot_um[0]) / (len(csd_positions_plot_um) -1) if len(csd_positions_plot_um) > 1 else 10.0)
                            if abs(step_val) < 1e-9 : step_val = 10.0 
                            first_edge = csd_positions_plot_um[0] - step_val/2
                            last_edge = csd_positions_plot_um[-1] + step_val/2
                            mid_edges = (csd_positions_plot_um[:-1] + csd_positions_plot_um[1:])/2
                            depth_edges = np.concatenate([[first_edge], mid_edges, [last_edge]])
                        elif len(csd_positions_plot_um) == 1: 
                            depth_edges = np.array([csd_positions_plot_um[0] - 10, csd_positions_plot_um[0] + 10]) 
                        else: 
                            depth_edges = np.array([0,1])
                        csd_data_plot_final = np.nan_to_num(csd_data_for_plotting, nan=0.0) 
                        img_csd = ax_curr_csd_plot.pcolormesh(time_edges, depth_edges, csd_data_plot_final, 
                                                              cmap='RdBu_r', shading='gouraud', 
                                                              vmin=-clim_val, vmax=clim_val, rasterized=True)
                        plt.colorbar(img_csd, ax=ax_curr_csd_plot, label=f'kCSD ({csd_units_str})', shrink=0.8, aspect=10)
                    ax_curr_csd_plot.set_ylabel('Depth (µm)'); ax_curr_csd_plot.set_title(f'Shank {shank_id_val}');
                    if len(csd_positions_plot_um) > 0 : ax_curr_csd_plot.set_ylim(max(csd_positions_plot_um), min(csd_positions_plot_um)) 
                # Removed specific LinAlgError catch block here
                except Exception as e_kcsd: # General exception will catch LinAlgError too
                    msg = f"kCSD Error (Ep{epoch_idx},Shk{shank_id_val}): {e_kcsd}"
                    print(f"  {msg}"); traceback.print_exc(); ax_curr_csd_plot.set_title(f"Shank {shank_id_val} (Ep {epoch_idx}) - kCSD Error"); ax_curr_csd_plot.text(0.5,0.5, "kCSD Error", ha='center',va='center',wrap=True);
                
                if 'lfp_processed_shank_epoch_full' in locals() and lfp_processed_shank_epoch_full is not None: del lfp_processed_shank_epoch_full
                if kcsd_estimator_obj is not None: del kcsd_estimator_obj
                if csd_result_neo is not None: del csd_result_neo
                if csd_data_matrix is not None: del csd_data_matrix
                if 'csd_data_for_plotting' in locals() and 'csd_data_plot_final' in locals():
                    del csd_data_for_plotting, csd_data_plot_final
                gc.collect()
            if num_shanks_found > 0 and axs_csd_epoch.size > 0: 
                axs_csd_epoch[-1, 0].set_xlabel('Time within epoch (s)')
            csd_epoch_plot_filename = output_file_prefix.parent / f"{base_filename}_epoch{epoch_idx}_kcsd1d_shanks.png"
            try:
                fig_csd_epoch.savefig(csd_epoch_plot_filename, dpi=150)
                print(f"  Saved CSD plot for Epoch {epoch_idx}: {csd_epoch_plot_filename}")
            except Exception as e_savefig:
                print(f"  Error saving CSD plot for Epoch {epoch_idx}: {e_savefig}")
            plt.close(fig_csd_epoch); gc.collect()
    except Exception as e_main_loop:
        print(f"An error occurred in the main processing loop: {e_main_loop}")
        traceback.print_exc()
    finally:
        if lfp_data_memmap_obj is not None:
            if hasattr(lfp_data_memmap_obj, '_mmap') and lfp_data_memmap_obj._mmap is not None:
                print("\nClosing LFP data memory map...")
                try: lfp_data_memmap_obj._mmap.close()
                except Exception as e_close_mmap: print(f"Warning: Error closing memory map: {e_close_mmap}")
            del lfp_data_memmap_obj; gc.collect(); print("LFP data memmap object deleted.")
        else: print("\nLFP data memory map was not active or already handled.")

# --- Main Execution Block ---
if __name__ == "__main__":
    try:
        from tkinter import Tk, filedialog
        use_gui = True
    except ImportError:
        print("tkinter not found. GUI for file selection will be disabled.")
        print(f"Please ensure LFP_BIN_FILE_PATH_DEFAULT ('{LFP_BIN_FILE_PATH_DEFAULT}'), etc., are correctly set in the script, or modify the script to accept paths as arguments.")
        use_gui = False
        lfp_bin_f_selected = LFP_BIN_FILE_PATH_DEFAULT
        lfp_meta_f_selected = LFP_META_FILE_PATH_DEFAULT
        channel_csv_f_selected = CHANNEL_INFO_CSV_PATH_DEFAULT
        timestamps_npy_f_selected = TIMESTAMPS_NPY_PATH_DEFAULT
        if not all(Path(f).exists() for f in [lfp_bin_f_selected, lfp_meta_f_selected, channel_csv_f_selected, timestamps_npy_f_selected]):
            print("One or more default file paths do not exist. Please check script configuration or provide files.")
    if use_gui:
        root = Tk(); root.withdraw(); root.attributes("-topmost", True)
        print("Please select the LFP binary file (*.lf.bin)...")
        lfp_bin_f_selected = filedialog.askopenfilename(title="Select LFP Binary File", initialfile=LFP_BIN_FILE_PATH_DEFAULT)
        if not lfp_bin_f_selected: sys.exit("LFP file selection cancelled.")
        suggested_meta_name = Path(lfp_bin_f_selected).stem + ".meta"
        print(f"Please select the corresponding Meta file (*.lf.meta)... (Suggested: {suggested_meta_name})")
        lfp_meta_f_selected = filedialog.askopenfilename(title="Select LFP Meta File", 
                                                         initialdir=Path(lfp_bin_f_selected).parent, 
                                                         initialfile=suggested_meta_name)
        if not lfp_meta_f_selected: sys.exit("Meta file selection cancelled.")
        print("Please select the Channel Info CSV file...")
        channel_csv_f_selected = filedialog.askopenfilename(title="Select Channel Info CSV", initialfile=CHANNEL_INFO_CSV_PATH_DEFAULT)
        if not channel_csv_f_selected: sys.exit("Channel info CSV selection cancelled.")
        print("Please select the Timestamps NPY file (*.nidq_timestamps.npy or similar)...")
        lfp_path_obj = Path(lfp_bin_f_selected)
        lfp_base_name_for_ts = lfp_path_obj.name
        if "_tcat.imec0.lf.bin" in lfp_base_name_for_ts: 
            suggested_ts_name = lfp_base_name_for_ts.replace(".imec0.lf.bin", ".nidq_timestamps.npy")
        elif "_tcat.imec1.lf.bin" in lfp_base_name_for_ts: 
            suggested_ts_name = lfp_base_name_for_ts.replace(".imec1.lf.bin", ".nidq_timestamps.npy")
        elif ".lf.bin" in lfp_base_name_for_ts: 
             suggested_ts_name = lfp_base_name_for_ts.replace(".lf.bin", ".nidq_timestamps.npy")
        else: 
            suggested_ts_name = lfp_path_obj.stem + ".nidq_timestamps.npy"
        timestamps_npy_f_selected = filedialog.askopenfilename(title="Select Timestamps NPY File",
                                                              initialdir=lfp_path_obj.parent,
                                                              initialfile=suggested_ts_name)
        if not timestamps_npy_f_selected: sys.exit("Timestamps NPY file selection cancelled.")
        root.destroy()

    print(f"\n--- Starting kCSD analysis script (Epoch-based, Time-Sliced, Float32, Direct KCSD1D) ---")
    print(f"LFP File: {lfp_bin_f_selected}")
    print(f"Meta File: {lfp_meta_f_selected}")
    print(f"Channel CSV: {channel_csv_f_selected}")
    print(f"Timestamps NPY: {timestamps_npy_f_selected}")
    print(f"Output Directory: {OUTPUT_DIR}\n")
    if not Path(lfp_bin_f_selected).exists(): print(f"ERROR: LFP file not found: {lfp_bin_f_selected}"); sys.exit(1)
    if not Path(lfp_meta_f_selected).exists(): print(f"ERROR: Meta file not found: {lfp_meta_f_selected}"); sys.exit(1)
    if not Path(channel_csv_f_selected).exists(): print(f"ERROR: Channel CSV not found: {channel_csv_f_selected}"); sys.exit(1)
    if not Path(timestamps_npy_f_selected).exists(): print(f"ERROR: Timestamps NPY not found: {timestamps_npy_f_selected}"); sys.exit(1)

    main_kcsd_analysis(lfp_bin_file=lfp_bin_f_selected,
                         lfp_meta_file=lfp_meta_f_selected,
                         channel_info_csv=channel_csv_f_selected,
                         timestamps_npy_file=timestamps_npy_f_selected)
    
    print(f"\n--- Script finished ---")

