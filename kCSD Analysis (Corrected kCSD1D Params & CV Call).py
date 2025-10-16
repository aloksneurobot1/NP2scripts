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

# --- Try to import readMeta ---
try:
    from DemoReadSGLXData.readSGLX import readMeta
except ImportError:
    print("ERROR: Could not import readMeta from DemoReadSGLXData.readSGLX.")
    print("Please ensure the 'DemoReadSGLXData' directory and 'readSGLX.py' are accessible.")
    sys.exit(1)

from elephant.current_source_density import estimate_csd # Use the public API

# --- Configuration Parameters ---
LFP_BIN_FILE_PATH_DEFAULT = "your_lfp_file.lf.bin"
LFP_META_FILE_PATH_DEFAULT = "your_lfp_file.lf.meta"
CHANNEL_INFO_CSV_PATH_DEFAULT = "your_channel_info.csv"
TIMESTAMPS_NPY_PATH_DEFAULT = "your_timestamps.nidq_timestamps.npy" 

OUTPUT_DIR = Path("./csd_kcsd_output_epoch_f32_v5_final") 
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_FS_CSD = 1000.0  # Hz
LFP_BAND_LOWCUT_CSD = 1.0  # Hz
LFP_BAND_HIGHCUT_CSD = 300.0 # Hz
NUMTAPS_CSD_FILTER = 101 # Odd number

CSD_SIGMA_CONDUCTIVITY = 0.3  # Siemens per meter (S/m) -> will be passed as 'sigma'
KCSD_LAMBDAS_CV = np.logspace(-7, -2, 9) 
KCSD_RS_CV_UM = np.logspace(np.log10(20), np.log10(500), 9) 

EPOCH_SUB_CHUNK_DURATION_SECONDS = 10 

# --- Voltage Scaling Function ---
def get_voltage_scaling_factor(meta):
    """Calculates the factor to convert int16 ADC values to microvolts (uV)."""
    try:
        v_max = float(meta['imAiRangeMax']) 
        i_max_adc_val = float(meta['imMaxInt'])
        probe_type = int(meta.get('imDatPrb_type', 0))
        lfp_gain = None

        if probe_type in [21, 24, 2013]: 
            lfp_gain = 80.0
        else: 
            general_lfp_gain_key_str = "~imChanLFGain" 
            if general_lfp_gain_key_str in meta: 
                 lfp_gain = float(meta[general_lfp_gain_key_str])
            else:
                first_lfp_gain_key_found = None
                sorted_keys = sorted([key for key in meta.keys() if key.endswith('lfGain')]) 
                if sorted_keys:
                    first_lfp_gain_key_found = sorted_keys[0]
                if first_lfp_gain_key_found:
                    lfp_gain = float(meta[first_lfp_gain_key_found])
                else:
                    lfp_gain = 250.0 
                    print(f"  Probe type {probe_type}, No specific LFP gain key found in meta.")
                    print(f"  Defaulting LFP gain to {lfp_gain} (Common for NP1.0). PLEASE VERIFY.")
        if lfp_gain is None: raise ValueError("LFP gain could not be determined.")
        if i_max_adc_val == 0 or lfp_gain == 0: raise ValueError("i_max_adc_val or LFP gain is zero.")
        scaling_factor_uv = (v_max / i_max_adc_val) * (1.0 / lfp_gain) * 1e6
        return scaling_factor_uv
    except Exception as e:
        print(f"Error calculating voltage scaling factor: {e}"); return None

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
        if uv_scale_factor is None: print("  Warning: Voltage scaling N/A.")

        file_size = bin_file_path.stat().st_size
        item_size = np.dtype('int16').itemsize
        if n_channels_total == 0 or item_size == 0: raise ValueError("Meta: Invalid chans/itemsize.")
        num_samples_in_file = file_size // (n_channels_total * item_size)
        if file_size % (n_channels_total * item_size) != 0:
            print(f"  Warning: File size non-integer multiple. Truncating to {num_samples_in_file} samples.")
        if num_samples_in_file <= 0: raise ValueError("Zero/negative samples in file.")
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
            raise ValueError(f"Channel info CSV must contain: {required_cols}")
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
    if not timestamps_npy_path.exists(): print(f"Error: Timestamps NPY not found: {timestamps_npy_path}"); return None
    try:
        ts_data = np.load(timestamps_npy_path, allow_pickle=True).item()
        if 'EpochFrameData' not in ts_data: print("Error: 'EpochFrameData' missing."); return None
        
        epochs = []
        for epoch_info_dict in ts_data['EpochFrameData']:
            if 'start_time_sec' in epoch_info_dict and 'end_time_sec' in epoch_info_dict and 'epoch_index' in epoch_info_dict:
                start_sec = float(epoch_info_dict['start_time_sec'])
                end_sec = float(epoch_info_dict['end_time_sec'])
                
                epoch_abs_start_sample = int(round(start_sec * fs_lfp))
                epoch_abs_end_sample_inclusive = int(round(end_sec * fs_lfp)) 
                
                original_start_sample_req = epoch_abs_start_sample 
                original_end_sample_req_inclusive = epoch_abs_end_sample_inclusive

                epoch_abs_start_sample = max(0, epoch_abs_start_sample)
                epoch_abs_end_sample_inclusive = min(epoch_abs_end_sample_inclusive, total_lfp_samples_in_file - 1)

                if epoch_abs_start_sample > epoch_abs_end_sample_inclusive or epoch_abs_start_sample >= total_lfp_samples_in_file:
                    print(f"Warning: Epoch {epoch_info_dict['epoch_index']} results in invalid/zero-duration after time conversion/capping. Skipping.")
                    continue
                
                epochs.append({
                    'epoch_index': int(epoch_info_dict['epoch_index']),
                    'abs_start_sample': epoch_abs_start_sample,
                    'abs_end_sample_inclusive': epoch_abs_end_sample_inclusive, 
                    'duration_sec_requested': end_sec - start_sec,
                    'duration_sec_actual': (epoch_abs_end_sample_inclusive - epoch_abs_start_sample + 1) / fs_lfp if fs_lfp > 0 else 0
                })
            else: print(f"Warning: Epoch info incomplete: {epoch_info_dict}")
        
        if not epochs: print("No valid epoch definitions constructed."); return None
        epochs.sort(key=lambda e: e['abs_start_sample'])
        print(f"Loaded and validated {len(epochs)} epoch definitions based on times.")
        for ep_idx_print, ep in enumerate(epochs): 
            print(f"  Using Epoch {ep['epoch_index']} (Sorted Idx {ep_idx_print}): Samples {ep['abs_start_sample']}-{ep['abs_end_sample_inclusive']} (Duration: {ep['duration_sec_actual']:.2f}s)")
        return epochs
    except Exception as e:
        print(f"Error loading/parsing timestamps NPY: {e}"); traceback.print_exc(); return None

# --- LFP Preprocessing for a SUB-CHUNK (using float32 for intermediate) ---
def preprocess_lfp_sub_chunk(lfp_sub_chunk_scaled_f32, fs_sub_chunk_in, 
                             lowcut, highcut, numtaps, consistent_downsampling_factor,
                             target_overall_fs): 
    if lfp_sub_chunk_scaled_f32.ndim == 1:
        lfp_sub_chunk_scaled_f32 = lfp_sub_chunk_scaled_f32[:, np.newaxis]
    
    downsampling_factor = consistent_downsampling_factor
    effective_fs_after_downsample = fs_sub_chunk_in / downsampling_factor

    if downsampling_factor > 1:
        min_samples_for_decimate = downsampling_factor * 30 
        if lfp_sub_chunk_scaled_f32.shape[0] <= min_samples_for_decimate : 
            lfp_sub_chunk_downsampled_f32 = lfp_sub_chunk_scaled_f32 
            effective_fs_after_downsample = fs_sub_chunk_in 
        else:
            try:
                lfp_sub_chunk_downsampled_f64 = signal.decimate(lfp_sub_chunk_scaled_f32, 
                                                       downsampling_factor, axis=0, ftype='fir', zero_phase=True)
                lfp_sub_chunk_downsampled_f32 = lfp_sub_chunk_downsampled_f64.astype(np.float32)
            except Exception as e_decimate:
                print(f"    Error during sub-chunk decimation: {e_decimate}. Using original sampling.")
                lfp_sub_chunk_downsampled_f32 = lfp_sub_chunk_scaled_f32 
                effective_fs_after_downsample = fs_sub_chunk_in
    else:
        lfp_sub_chunk_downsampled_f32 = lfp_sub_chunk_scaled_f32 
    
    nyq = effective_fs_after_downsample / 2.0
    actual_highcut = min(highcut, nyq * 0.99)
    actual_lowcut = max(lowcut, 0.01)
    
    current_numtaps = numtaps
    if current_numtaps >= lfp_sub_chunk_downsampled_f32.shape[0]:
        current_numtaps = lfp_sub_chunk_downsampled_f32.shape[0] - 1
    if current_numtaps % 2 == 0 and current_numtaps > 0: 
        current_numtaps -=1
    
    if current_numtaps < 3 or actual_lowcut >= actual_highcut: 
        lfp_sub_chunk_filtered_f32 = lfp_sub_chunk_downsampled_f32
    else:
        try:
            fir_taps = signal.firwin(current_numtaps, [actual_lowcut, actual_highcut], fs=effective_fs_after_downsample, pass_zero='bandpass', window='hamming')
            lfp_sub_chunk_filtered_f64 = signal.filtfilt(fir_taps, 1.0, lfp_sub_chunk_downsampled_f32, axis=0)
            lfp_sub_chunk_filtered_f32 = lfp_sub_chunk_filtered_f64.astype(np.float32)
        except Exception as e_filter:
            print(f"    Error during sub-chunk filtering: {e_filter}. Using downsampled data.")
            lfp_sub_chunk_filtered_f32 = lfp_sub_chunk_downsampled_f32
            
    return lfp_sub_chunk_filtered_f32, effective_fs_after_downsample


# --- Main CSD Analysis Script ---
def main_kcsd_analysis(lfp_bin_file, lfp_meta_file, channel_info_csv, timestamps_npy_file):
    base_filename = Path(lfp_bin_file).stem
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
        
    if abs(fs_orig - TARGET_FS_CSD) < 1e-3 or fs_orig < TARGET_FS_CSD:
        overall_downsampling_factor = 1
    else:
        overall_downsampling_factor = int(round(fs_orig / TARGET_FS_CSD))
        if overall_downsampling_factor <= 0: overall_downsampling_factor = 1
    final_effective_fs_for_csd = fs_orig / overall_downsampling_factor
    print(f"Overall target downsampling factor: {overall_downsampling_factor}, Final effective CSD FS: {final_effective_fs_for_csd:.2f} Hz")

    try:
        probe_channel_df = load_channel_info_kcsd(channel_info_csv)
        unique_shanks = sorted(probe_channel_df['shank_index'].unique())
        num_shanks_found = len(unique_shanks)
        if num_shanks_found == 0: print("No shanks in channel info. Exiting."); return 

        for epoch in epoch_definitions:
            epoch_idx = epoch['epoch_index']
            epoch_abs_start_sample = epoch['abs_start_sample']
            epoch_abs_end_sample_inclusive = epoch['abs_end_sample_inclusive']
            epoch_abs_end_slice = epoch_abs_end_sample_inclusive + 1 
            
            print(f"\n>>> Processing Epoch {epoch_idx} (LFP Samples: {epoch_abs_start_sample} to {epoch_abs_end_sample_inclusive}, Duration: {epoch['duration_sec_actual']:.2f}s) <<<")
            
            fig_csd_epoch, axs_csd_epoch = plt.subplots(num_shanks_found, 1,
                                                          figsize=(12, 4 * num_shanks_found),
                                                          sharex=True, squeeze=False, constrained_layout=True)
            fig_csd_epoch.suptitle(f'kCSD1D - {base_filename} - Epoch {epoch_idx}', fontsize=16)

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
                    msg = f"Shank {shank_id_val} (Ep {epoch_idx}) - Channel Index Error. Indices {out_of_bounds_indices} out of bounds for {n_channels_total_in_file} total channels."
                    print(f"  {msg}"); ax_curr_csd_plot.set_title(msg); ax_curr_csd_plot.text(0.5,0.5, "Channel index out of bounds.", ha='center', va='center', transform=ax_curr_csd_plot.transAxes, wrap=True)
                    continue

                processed_sub_chunks_for_shank_epoch = []
                epoch_duration_samples_for_slice = epoch_abs_end_slice - epoch_abs_start_sample
                sub_chunk_size_orig_fs = int(fs_orig * EPOCH_SUB_CHUNK_DURATION_SECONDS)
                if sub_chunk_size_orig_fs <= 0 : sub_chunk_size_orig_fs = epoch_duration_samples_for_slice 

                print(f"    Processing Epoch {epoch_idx}, Shank {shank_id_val} in sub-chunks of ~{EPOCH_SUB_CHUNK_DURATION_SECONDS}s")
                sub_chunk_count = 0
                for i_sub_chunk_local_start in range(0, epoch_duration_samples_for_slice, sub_chunk_size_orig_fs):
                    sub_chunk_abs_start_in_file = epoch_abs_start_sample + i_sub_chunk_local_start
                    sub_chunk_abs_end_in_file = min(sub_chunk_abs_start_in_file + sub_chunk_size_orig_fs, epoch_abs_end_slice)
                    current_sub_chunk_length = sub_chunk_abs_end_in_file - sub_chunk_abs_start_in_file
                    if current_sub_chunk_length <= 0: continue
                    sub_chunk_count += 1
                    
                    try:
                        if lfp_data_memmap_obj is None: raise ValueError("Memmap lost")
                        temp_processed_channels_for_sub_chunk = []
                        for ch_idx_in_shank, global_ch_idx in enumerate(shank_global_indices):
                            single_ch_sub_chunk_raw = lfp_data_memmap_obj[sub_chunk_abs_start_in_file:sub_chunk_abs_end_in_file, global_ch_idx]
                            single_ch_sub_chunk_f32 = single_ch_sub_chunk_raw.astype(np.float32) 
                            if uv_scale_factor_loaded is not None:
                                single_ch_sub_chunk_f32 *= uv_scale_factor_loaded
                            
                            processed_single_ch_sub_chunk_f32, _ = preprocess_lfp_sub_chunk(
                                single_ch_sub_chunk_f32, fs_sub_chunk_in=fs_orig, 
                                lowcut=LFP_BAND_LOWCUT_CSD, highcut=LFP_BAND_HIGHCUT_CSD, 
                                numtaps=NUMTAPS_CSD_FILTER,
                                consistent_downsampling_factor=overall_downsampling_factor,
                                target_overall_fs = TARGET_FS_CSD 
                            )
                            if processed_single_ch_sub_chunk_f32 is not None and processed_single_ch_sub_chunk_f32.shape[0] > 0:
                                temp_processed_channels_for_sub_chunk.append(processed_single_ch_sub_chunk_f32.flatten())
                            else:
                                expected_len_ds = int(np.ceil(current_sub_chunk_length / overall_downsampling_factor)) if overall_downsampling_factor > 0 else current_sub_chunk_length
                                temp_processed_channels_for_sub_chunk.append(np.full(expected_len_ds, np.nan, dtype=np.float32))
                            del single_ch_sub_chunk_raw, single_ch_sub_chunk_f32, processed_single_ch_sub_chunk_f32
                        
                        if temp_processed_channels_for_sub_chunk:
                            valid_channel_chunks = [ch_arr for ch_arr in temp_processed_channels_for_sub_chunk if ch_arr is not None and ch_arr.size > 0]
                            if not valid_channel_chunks:
                                print(f"      Warning: No valid processed channels for sub-chunk {sub_chunk_count}. Skipping sub-chunk.")
                            else:
                                min_len = min(len(ch_arr) for ch_arr in valid_channel_chunks)
                                aligned_channels = [ch_arr[:min_len] for ch_arr in valid_channel_chunks]
                                if not aligned_channels or not (hasattr(aligned_channels[0], 'ndim') and aligned_channels[0].ndim > 0 and aligned_channels[0].size > 0) or \
                                   not all(hasattr(arr, 'shape') and arr.shape[0] == aligned_channels[0].shape[0] for arr in aligned_channels if hasattr(arr, 'ndim') and arr.ndim > 0):
                                    print("      Error: Sub-chunk channels inconsistent lengths or invalid after alignment. Skipping sub-chunk.")
                                else:
                                    processed_sub_chunk_for_shank = np.vstack(aligned_channels).T.astype(np.float32)
                                    processed_sub_chunks_for_shank_epoch.append(processed_sub_chunk_for_shank)
                        del temp_processed_channels_for_sub_chunk; gc.collect()
                    except MemoryError as me_sub_chunk: 
                        print(f"    MemoryError processing sub-chunk for Ep{epoch_idx},Shk{shank_id_val}: {me_sub_chunk}. Skipping rest of this shank/epoch."); 
                        processed_sub_chunks_for_shank_epoch = [] 
                        break 
                    except Exception as e_sub_chunk: 
                        print(f"    Error processing sub-chunk for Ep{epoch_idx},Shk{shank_id_val}: {e_sub_chunk}"); 
                        continue
                
                if not processed_sub_chunks_for_shank_epoch: 
                    msg = f"No sub-chunks successfully processed for Epoch {epoch_idx}, Shank {shank_id_val}."
                    print(f"  {msg} Skipping CSD."); ax_curr_csd_plot.set_title(f"Shank {shank_id_val} (Ep {epoch_idx}) - Sub-chunk Proc. Failed"); ax_curr_csd_plot.text(0.5,0.5, msg, ha='center', va='center', transform=ax_curr_csd_plot.transAxes, wrap=True)
                    continue

                print(f"    Concatenating {len(processed_sub_chunks_for_shank_epoch)} sub-chunks for Ep{epoch_idx},Shk{shank_id_val}...")
                lfp_processed_shank_epoch_full = None 
                try:
                    lfp_processed_shank_epoch_full = np.concatenate(processed_sub_chunks_for_shank_epoch, axis=0).astype(np.float32)
                except MemoryError as me_concat_epoch: 
                    msg = f"MemoryError concatenating sub-chunks for Ep{epoch_idx},Shk{shank_id_val}: {me_concat_epoch}"
                    print(msg); ax_curr_csd_plot.set_title(f"Shank {shank_id_val} (Ep{epoch_idx}) - Concat Error"); ax_curr_csd_plot.text(0.5,0.5, msg, ha='center',va='center',wrap=True);
                finally: 
                    if 'processed_sub_chunks_for_shank_epoch' in locals(): 
                        del processed_sub_chunks_for_shank_epoch
                    gc.collect()
                
                if lfp_processed_shank_epoch_full is None: 
                    continue
                
                print(f"    Final LFP for CSD (Ep{epoch_idx},Shk{shank_id_val}): {lfp_processed_shank_epoch_full.shape}, FS: {final_effective_fs_for_csd:.2f} Hz, dtype: {lfp_processed_shank_epoch_full.dtype}")
                
                coords_quant_um = electrode_coords_um_shank.reshape(-1, 1) * pq.um

                # Initialize variables that might not be assigned if an error occurs early in the try block
                kcsd_estimator_obj = None
                pykcsd_obj = None # This is an attribute of kcsd_estimator_obj, not a separate object to get
                csd_result_neo = None
                csd_data_matrix = None
                shank_analog_signal = None


                try:
                    lfp_units_str = 'uV' if uv_scale_factor_loaded is not None else 'ADC_count'
                    # Data for neo.AnalogSignal is (samples, channels)
                    shank_analog_signal = neo.AnalogSignal(
                        lfp_processed_shank_epoch_full.astype(np.float64), # kCSD might prefer float64
                        units=lfp_units_str, 
                        sampling_rate=final_effective_fs_for_csd * pq.Hz
                    )
                except Exception as e_neo: 
                    msg = f"Neo Signal Error (Ep{epoch_idx},Shk{shank_id_val}): {e_neo}"
                    print(f"  {msg}"); ax_curr_csd_plot.set_title(f"Shank {shank_id_val} (Ep {epoch_idx}) - Neo Error"); ax_curr_csd_plot.text(0.5,0.5, "Neo Error", ha='center',va='center',wrap=True);
                    if 'lfp_processed_shank_epoch_full' in locals() and lfp_processed_shank_epoch_full is not None: del lfp_processed_shank_epoch_full; gc.collect()
                    continue
                
                # Delete the large concatenated array now that it's in the neo.AnalogSignal
                if 'lfp_processed_shank_epoch_full' in locals() and lfp_processed_shank_epoch_full is not None:
                    del lfp_processed_shank_epoch_full; gc.collect()

                print(f"    DEBUG: Before estimate_csd for Ep{epoch_idx},Shk{shank_id_val}:")
                print(f"      LFP (shank_analog_signal) shape: {shank_analog_signal.shape}, units: {shank_analog_signal.units}, sampling_rate: {shank_analog_signal.sampling_rate}")
                print(f"      Coordinates (coords_quant_um) shape: {coords_quant_um.shape}, units: {coords_quant_um.units}")

                try: 
                    num_csd_estimation_points = max(32, int(len(electrode_coords_um_shank) * 1.5))
                    
                    xmin_param = electrode_coords_um_shank.min() * pq.um
                    xmax_param = electrode_coords_um_shank.max() * pq.um
                    if num_csd_estimation_points > 1:
                        gdx_param = ((xmax_param - xmin_param) / (num_csd_estimation_points - 1))
                    else: 
                        gdx_param = 10.0 * pq.um 
                    if gdx_param.magnitude <= 1e-9: 
                        gdx_param = 1.0 * pq.um 
                        if num_csd_estimation_points < 1: num_csd_estimation_points = 1

                    print(f"    KCSD Params: n_src_init={num_csd_estimation_points}, xmin={xmin_param}, xmax={xmax_param}, gdx={gdx_param}")
                    
                    kcsd_estimator_obj = estimate_csd(
                        lfp=shank_analog_signal, 
                        coordinates=coords_quant_um, 
                        method='KCSD1D', 
                        sigma=CSD_SIGMA_CONDUCTIVITY * pq.S / pq.m, # Use 'sigma'
                        n_src_init=num_csd_estimation_points,       # Use 'n_src_init'
                        xmin=xmin_param,
                        xmax=xmax_param,
                        gdx=gdx_param,
                        ext_x = 0.0 * pq.um, # Explicitly pass ext_x with units
                        lambdas=KCSD_LAMBDAS_CV, 
                        Rs=KCSD_RS_CV_UM * pq.um,
                        process_estimate=False # Get the estimator object
                    )
                    
                    # The kcsd_estimator_obj is an instance of elephant.current_source_density_src.KCSD.KCSD1D
                    # It has a .cross_validate() method and .values() method.
                    # It also has attributes like .lambd and .R after cross_validate is called.

                    if kcsd_estimator_obj is not None and hasattr(kcsd_estimator_obj, 'cross_validate'):
                        print(f"    Performing cross-validation for Ep{epoch_idx},Shk{shank_id_val}...")
                        kcsd_estimator_obj.cross_validate(lambdas=KCSD_LAMBDAS_CV, Rs=KCSD_RS_CV_UM * pq.um) # Pass Rs with units
                        print(f"      CV Optimal Lambda: {kcsd_estimator_obj.lambd}") 
                        print(f"      CV Optimal R: {kcsd_estimator_obj.R * 1e6:.2f} um") # .R is in meters

                        if hasattr(kcsd_estimator_obj, 'cv_error') and kcsd_estimator_obj.cv_error is not None:
                            fig_cv, ax_cv = plt.subplots(figsize=(8, 6)) 
                            cv_err_to_plot = np.log10(kcsd_estimator_obj.cv_error) 
                            if np.isneginf(cv_err_to_plot).any():
                                finite_min = np.min(cv_err_to_plot[np.isfinite(cv_err_to_plot)]) if np.isfinite(cv_err_to_plot).any() else -10
                                cv_err_to_plot[np.isneginf(cv_err_to_plot)] = finite_min - 1 
                            vmin_cv = np.nanmin(cv_err_to_plot) if np.isfinite(cv_err_to_plot).any() else -10
                            vmax_cv = np.nanpercentile(cv_err_to_plot[np.isfinite(cv_err_to_plot)], 98) if np.isfinite(cv_err_to_plot).any() else vmin_cv + 1
                            if vmin_cv >= vmax_cv : vmax_cv = vmin_cv + (1e-9 if vmin_cv != 0 else 1e-9)
                            norm_cv = Normalize(vmin=vmin_cv, vmax=vmax_cv)
                            im_cv = ax_cv.imshow(cv_err_to_plot, aspect='auto', origin='lower', cmap='viridis', norm=norm_cv)
                            ax_cv.set_xticks(np.arange(len(KCSD_LAMBDAS_CV))) 
                            ax_cv.set_xticklabels([f"{l:.1e}" for l in KCSD_LAMBDAS_CV], rotation=45, ha="right")
                            ax_cv.set_yticks(np.arange(len(KCSD_RS_CV_UM))) 
                            ax_cv.set_yticklabels([f"{r_val:.1f}" for r_val in KCSD_RS_CV_UM])

                            best_lambda_idx_list = np.where(np.isclose(KCSD_LAMBDAS_CV, kcsd_estimator_obj.lambd))[0]
                            # kcsd_estimator_obj.R is in meters, KCSD_RS_CV_UM is in micrometers
                            best_R_idx_list = np.where(np.isclose(KCSD_RS_CV_UM, kcsd_estimator_obj.R * 1e6))[0]
                            
                            if len(best_lambda_idx_list) > 0 and len(best_R_idx_list) > 0:
                                ax_cv.scatter(best_lambda_idx_list[0], best_R_idx_list[0], marker='x', color='red', s=100, label=f'Optimal\nL={kcsd_estimator_obj.lambd:.1e}\nR={kcsd_estimator_obj.R*1e6:.1f}um')
                                ax_cv.legend(fontsize='small')
                            ax_cv.set_xlabel('Lambda'); ax_cv.set_ylabel('R (um)')
                            ax_cv.set_title(f'kCSD CV Error (Ep {epoch_idx}, Shk {shank_id_val})')
                            plt.colorbar(im_cv, ax=ax_cv, label='log10(CV Error)'); fig_cv.tight_layout()
                            cv_plot_filename = output_file_prefix.parent / f"{base_filename}_ep{epoch_idx}_shk{shank_id_val}_kCSDcv.png"
                            plt.savefig(cv_plot_filename); print(f"    Saved CV plot: {cv_plot_filename}"); plt.close(fig_cv)
                        else:
                            print("    CV error attribute not found on estimator after cross_validate call.")
                    else: 
                        print("    Estimator object kcsd_estimator_obj is None or does not have cross_validate method.")


                    # Get CSD values using the .values() method of the estimator
                    csd_profile_values = kcsd_estimator_obj.values() 
                    
                    # Manually create the neo.AnalogSignal for the CSD result
                    # The CSD values are typically in uA/mm^3 or similar
                    # The coordinates of the CSD estimate are on kcsd_estimator_obj.estm_x
                    csd_result_neo = neo.AnalogSignal(
                        csd_profile_values.T, # KCSD.values() returns (n_est, n_time), neo needs (n_time, n_est/n_chan)
                        units = pq.uA / pq.mm**3, # Typical CSD units
                        sampling_rate = final_effective_fs_for_csd * pq.Hz,
                        # Annotate with CSD estimation coordinates
                        coordinates = kcsd_estimator_obj.estm_x.rescale(pq.um) # estm_x is (n_est, 1)
                    )
                    
                    csd_data_matrix = np.asarray(csd_result_neo).astype(np.float32) 
                    csd_times_vector = np.arange(csd_data_matrix.shape[0]) / final_effective_fs_for_csd # Corrected for (time, csd_site)
                    if csd_data_matrix.ndim > 1: # If multiple CSD sites
                        csd_times_vector = np.arange(csd_data_matrix.shape[0]) / final_effective_fs_for_csd
                    else: # Single CSD site, or if shape is (time,)
                        csd_times_vector = np.arange(len(csd_data_matrix)) / final_effective_fs_for_csd


                    csd_units_str = csd_result_neo.units.dimensionality.string
                    
                    if hasattr(kcsd_estimator_obj, 'estm_x') and kcsd_estimator_obj.estm_x is not None:
                        csd_positions_plot_um = kcsd_estimator_obj.estm_x.rescale('um').magnitude.flatten()
                    else: 
                         csd_positions_plot_um = np.linspace(electrode_coords_um_shank.min(), electrode_coords_um_shank.max(), csd_data_matrix.shape[1] if csd_data_matrix.ndim > 1 else 1)


                    if csd_data_matrix.size == 0: 
                        msg = f"Empty CSD result (Ep{epoch_idx},Shk{shank_id_val})"
                        print(f"  {msg}"); ax_curr_csd_plot.set_title(msg); ax_curr_csd_plot.text(0.5,0.5,msg,ha='center',va='center',wrap=True);
                        # Clean up before continue
                        if shank_analog_signal is not None: del shank_analog_signal
                        if kcsd_estimator_obj is not None: del kcsd_estimator_obj
                        if csd_result_neo is not None: del csd_result_neo
                        gc.collect()
                        continue
                    
                    # Transpose CSD data matrix if it came out as (sites, time) from .values() for plotting
                    # pcolormesh expects (X, Y, C) where C matches X and Y dimensions, or C is (Y-1, X-1) for flat
                    # If csd_data_matrix from asarray(csd_result_neo) is (time, sites), it's good.
                    # If it's (sites, time) from kcsd_estimator_obj.values(), it needs transpose for plotting.
                    # neo.AnalogSignal stores as (time, channels/sites). So asarray(csd_result_neo) is (time, sites)
                    # kcsd_estimator_obj.values() returns (n_est_pts, n_time_pts)
                    
                    # So, after csd_result_neo = neo.AnalogSignal(csd_profile_values.T ...),
                    # csd_data_matrix = np.asarray(csd_result_neo) will be (time, csd_sites)
                    # For pcolormesh (X=time, Y=depth), CSD data should be (depth_sites, time_samples)
                    # So, we need csd_data_matrix.T for plotting.

                    csd_data_for_plotting = csd_data_matrix.T # Now (csd_sites, time_samples)

                    print(f"      CSD data matrix for plotting (Ep{epoch_idx},Shk{shank_id_val}): {csd_data_for_plotting.shape}, dtype: {csd_data_for_plotting.dtype}")
                    if csd_data_for_plotting.size > 0:
                        print(f"      CSD data stats: min={np.nanmin(csd_data_for_plotting):.2e}, max={np.nanmax(csd_data_for_plotting):.2e}, mean={np.nanmean(csd_data_for_plotting):.2e}")
                    
                    abs_csd = np.abs(csd_data_for_plotting)
                    if csd_data_for_plotting.size == 0 or np.all(np.isnan(abs_csd)) or not np.any(np.isfinite(abs_csd)):
                        print(f"      Warning: CSD data for Ep{epoch_idx},Shk{shank_id_val} is empty or all NaN/Inf. Plot will be blank.")
                        clim_val = 1.0 
                        ax_curr_csd_plot.text(0.5, 0.5, "CSD Data Empty/Invalid", ha='center', va='center', transform=ax_curr_csd_plot.transAxes)
                    else:
                        clim_val = np.percentile(abs_csd[np.isfinite(abs_csd)], 99.0) 
                        if clim_val < 1e-9: 
                           max_abs = np.nanmax(abs_csd[np.isfinite(abs_csd)])
                           clim_val = max_abs if max_abs > 1e-9 else 1.0 
                        if abs(clim_val) < 1e-9: clim_val = 1e-9 
                        print(f"      Plotting clim_val: {clim_val:.2e}")

                        time_edges = np.concatenate([csd_times_vector, [csd_times_vector[-1] + (1.0 / final_effective_fs_for_csd)]]) if len(csd_times_vector) > 0 else np.array([0, 1.0/final_effective_fs_for_csd])
                        if len(csd_positions_plot_um) > 1:
                            depth_diffs = np.diff(csd_positions_plot_um)
                            step_val = depth_diffs[0] if len(depth_diffs) > 0 else ( (csd_positions_plot_um[-1] - csd_positions_plot_um[0]) / (len(csd_positions_plot_um) -1 ) if len(csd_positions_plot_um) >1 else 10.0)
                            if abs(step_val) < 1e-9 : step_val = 10.0 
                            first_edge = csd_positions_plot_um[0] - step_val/2
                            last_edge = csd_positions_plot_um[-1] + step_val/2
                            mid_edges = (csd_positions_plot_um[:-1] + csd_positions_plot_um[1:])/2 if len(csd_positions_plot_um) > 1 else np.array([]) 
                            depth_edges = np.concatenate([[first_edge], mid_edges, [last_edge]])
                        elif len(csd_positions_plot_um) == 1: depth_edges = np.array([csd_positions_plot_um[0] - 10, csd_positions_plot_um[0] + 10])
                        else: depth_edges = np.array([0,1])
                        
                        if np.any(np.isnan(csd_data_for_plotting)):
                            print(f"      Warning: NaNs present in CSD data matrix for Ep{epoch_idx},Shk{shank_id_val}. Plot may have gaps.")
                            csd_data_plot_final = np.nan_to_num(csd_data_for_plotting, nan=0.0) 
                        else:
                            csd_data_plot_final = csd_data_for_plotting

                        img_csd = ax_curr_csd_plot.pcolormesh(time_edges, depth_edges, csd_data_plot_final, cmap='RdBu_r', shading='gouraud', vmin=-clim_val, vmax=clim_val, rasterized=True)
                        plt.colorbar(img_csd, ax=ax_curr_csd_plot, label=f'kCSD ({csd_units_str})', shrink=0.8)
                    
                    ax_curr_csd_plot.set_ylabel('Depth (µm)'); ax_curr_csd_plot.set_title(f'Shank {shank_id_val}'); ax_curr_csd_plot.invert_yaxis()

                except Exception as e_kcsd: 
                    msg = f"kCSD Error (Ep{epoch_idx},Shk{shank_id_val}): {e_kcsd}"
                    print(f"  {msg}"); traceback.print_exc(); ax_curr_csd_plot.set_title(f"Shank {shank_id_val} (Ep {epoch_idx}) - kCSD Error"); ax_curr_csd_plot.text(0.5,0.5, "kCSD Error", ha='center',va='center',wrap=True);
                
                # Cleanup for CSD specific variables
                if shank_analog_signal is not None: del shank_analog_signal 
                if kcsd_estimator_obj is not None: del kcsd_estimator_obj
                # pykcsd_obj is an attribute, no need to delete separately if kcsd_estimator_obj is deleted
                if csd_result_neo is not None: del csd_result_neo
                if csd_data_matrix is not None: del csd_data_matrix
                if 'csd_data_for_plot' in locals(): del csd_data_for_plot
                if 'csd_data_plot_final' in locals(): del csd_data_plot_final
                gc.collect()
            # End of shank loop for this epoch
            
            if num_shanks_found > 0:
                axs_csd_epoch[-1, 0].set_xlabel('Time within epoch (s)')
            csd_epoch_plot_filename = output_file_prefix.parent / f"{base_filename}_epoch{epoch_idx}_kcsd1d_shanks.png"
            fig_csd_epoch.savefig(csd_epoch_plot_filename, dpi=150)
            print(f"  Saved CSD plot for Epoch {epoch_idx}: {csd_epoch_plot_filename}")
            plt.close(fig_csd_epoch) # Close epoch figure
            gc.collect()
        # End of epoch loop

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
    from tkinter import Tk, filedialog
    root = Tk(); root.withdraw(); root.attributes("-topmost", True)

    print("Please select the LFP binary file (*.lf.bin)...")
    lfp_bin_f_selected = filedialog.askopenfilename(title="Select LFP Binary File", initialfile=LFP_BIN_FILE_PATH_DEFAULT)
    if not lfp_bin_f_selected: sys.exit("LFP file selection cancelled.")

    print("Please select the corresponding Meta file (*.lf.meta)...")
    lfp_meta_f_selected = filedialog.askopenfilename(title="Select LFP Meta File", initialdir=Path(lfp_bin_f_selected).parent, initialfile=Path(lfp_bin_f_selected).stem + ".meta")
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

    print(f"\n--- Starting kCSD analysis script (Epoch-based, Time-Sliced, Float32) ---")
    print(f"LFP File: {lfp_bin_f_selected}")
    print(f"Meta File: {lfp_meta_f_selected}")
    print(f"Channel CSV: {channel_csv_f_selected}")
    print(f"Timestamps NPY: {timestamps_npy_f_selected}")
    print(f"Output Directory: {OUTPUT_DIR}\n")
    
    main_kcsd_analysis(lfp_bin_file=lfp_bin_f_selected,
                         lfp_meta_file=lfp_meta_f_selected,
                         channel_info_csv=channel_csv_f_selected,
                         timestamps_npy_file=timestamps_npy_f_selected)
    
    print(f"\n--- Script finished ---")

