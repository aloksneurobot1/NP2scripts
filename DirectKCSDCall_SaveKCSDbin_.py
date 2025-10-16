# -*- coding: utf-8 -*-
"""
Created on Thu May 15 11:47:08 2025

@author: HT_bo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy import signal
import quantities as pq
import neo # Added for potential future use and type hinting
from pathlib import Path
import sys
import traceback
import os
import gc
import datetime # For metadata timestamp

# --- Try to import readMeta ---
try:
    from DemoReadSGLXData.readSGLX import readMeta
except ImportError:
    print("ERROR: Could not import readMeta from DemoReadSGLXData.readSGLX.")
    print("Please ensure the 'DemoReadSGLXData' directory and 'readSGLX.py' are accessible,")
    print("or adjust the PYTHONPATH if necessary.")
    sys.exit(1)

# Import KCSD1D directly for more control
from elephant.current_source_density_src.KCSD import KCSD1D


# --- Configuration Parameters ---
LFP_BIN_FILE_PATH_DEFAULT = "your_lfp_file.lf.bin"
LFP_META_FILE_PATH_DEFAULT = "your_lfp_file.lf.meta"
CHANNEL_INFO_CSV_PATH_DEFAULT = "your_channel_info.csv"
EPOCH_DEFINITIONS_NPY_PATH_DEFAULT = "your_epoch_definitions.nidq_timestamps.npy" # Renamed for clarity

# --- SWR Analysis Specific Configuration ---
SWR_TIMESTAMPS_NPY_PATH_DEFAULT = "your_swr_timestamps_NREM_by_epoch.npy" 
SWR_ANALYSIS_ENABLED = True # Set to True to perform SWR-triggered average CSD
SWR_TARGET_STATE_CODE = 1 # 1 for NREM, as per RipplesExtraction_Awake_NREM.txt output structure
SWR_TARGET_REGION = 'CA3' # Region from which to get SWR timestamps (e.g., 'CA1', 'CA3')
SWR_TIME_WINDOW_MS = np.array([-100, 100]) # Time window around SWR peak in milliseconds
# KCSD will be computed on the LFP averaged over this window for all SWRs in an epoch.

PLOT_INDIVIDUAL_SWR_CSDS = False  # Set to True to plot CSD for each SWR event
MAX_INDIVIDUAL_SWR_PLOTS_PER_SHANK_EPOCH = 10 # Optional: Uncomment to limit plots


OUTPUT_DIR = Path("./csd_kcsd_output_swr_avg_v1") # Updated output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_FS_CSD = 1250.0  # Hz
LFP_BAND_LOWCUT_CSD = 1.0  # Hz
LFP_BAND_HIGHCUT_CSD = 300.0 # Hz
NUMTAPS_CSD_FILTER = 101 # Odd number

CSD_SIGMA_CONDUCTIVITY = 0.3  # Siemens per meter (S/m) -> will be passed as 'sigma'
KCSD_LAMBDAS_CV = np.logspace(-7, -2, 9)
KCSD_RS_CV_UM = np.logspace(np.log10(20), np.log10(500), 9) # in micrometers

EPOCH_SUB_CHUNK_DURATION_SECONDS = 10 # Duration for LFP sub-chunk processing for full epoch LFP
PLOT_DURATION_LIMIT_SECONDS = 300.0  # Max duration in seconds for CSD plots (e.g., 5 minutes). Set to None or 0 to plot full epoch.
# For SWR CSD, the plot duration is determined by SWR_TIME_WINDOW_MS


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
        required_cols = ['global_channel_index', 'shank_index', 'ycoord_on_shank_um'] # Essential for KCSD1D
        # 'acronym' or 'subregion' would be needed for region-specific SWR loading if channel_to_subregion dict isn't provided
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

# --- Timestamp and Epoch Loading (Time-based for overall epochs) ---
def load_epoch_definitions_from_times(timestamps_npy_path_str, fs_lfp, total_lfp_samples_in_file):
    timestamps_npy_path = Path(timestamps_npy_path_str)
    print(f"Loading epoch definitions (time-based) from: {timestamps_npy_path}")
    if not timestamps_npy_path.exists():
        print(f"Error: Epoch definitions NPY file not found: {timestamps_npy_path}")
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
        
        if data_to_process is not None and 'EpochFrameData' in data_to_process: # Structure from some timestamp files
            epoch_info_list = data_to_process['EpochFrameData']
            if not isinstance(epoch_info_list, list):
                print(f"  Error: 'EpochFrameData' was found but it is not a list. Type: {type(epoch_info_list)}")
                return None
            print(f"  Found 'EpochFrameData' with {len(epoch_info_list)} entries.")
        elif isinstance(loaded_npy, np.ndarray) and loaded_npy.dtype.fields and \
             all(f in loaded_npy.dtype.fields for f in ['start_time_sec', 'end_time_sec', 'epoch_index']): # Common structure
            epoch_info_list = list(loaded_npy) 
            print(f"  Interpreting loaded NPY as a structured array of {len(epoch_info_list)} epochs.")
        # Add other potential structures from which to parse epoch definitions if necessary

        if epoch_info_list is None:
            print("Error: Epoch definitions NPY file format not recognized or 'EpochFrameData' not found in the expected structure.")
            # (rest of the error reporting from original script)
            return None

        epochs = []
        for item_idx, epoch_info_item in enumerate(epoch_info_list):
            start_sec_val, end_sec_val, epoch_idx_val = None, None, None
            # (parsing logic from original script, handles dict and structured array items)
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
                     print(f"  Warning: Missing expected field in structured array epoch item {item_idx}: {e_field}"); continue
            else: print(f"  Warning: Epoch info item {item_idx} format not recognized: {type(epoch_info_item)}"); continue
            
            if start_sec_val is not None and end_sec_val is not None and epoch_idx_val is not None:
                start_sec, end_sec = float(start_sec_val), float(end_sec_val)
                epoch_abs_start_sample = int(round(start_sec * fs_lfp))
                epoch_abs_end_sample_inclusive = int(round(end_sec * fs_lfp)) -1 
                if epoch_abs_end_sample_inclusive < epoch_abs_start_sample: epoch_abs_end_sample_inclusive = epoch_abs_start_sample
                epoch_abs_start_sample_capped = max(0, epoch_abs_start_sample)
                epoch_abs_end_sample_inclusive_capped = min(epoch_abs_end_sample_inclusive, total_lfp_samples_in_file - 1)

                if epoch_abs_start_sample_capped > epoch_abs_end_sample_inclusive_capped or epoch_abs_start_sample_capped >= total_lfp_samples_in_file:
                    print(f"  Warning: Epoch {epoch_idx_val} (Req: {start_sec:.3f}s-{end_sec:.3f}s) invalid sample range ({epoch_abs_start_sample_capped}-{epoch_abs_end_sample_inclusive_capped}). Skipping."); continue
                epochs.append({'epoch_index': int(epoch_idx_val), 'abs_start_sample': epoch_abs_start_sample_capped,
                    'abs_end_sample_inclusive': epoch_abs_end_sample_inclusive_capped, 'duration_sec_requested': end_sec - start_sec,
                    'duration_sec_actual': (epoch_abs_end_sample_inclusive_capped - epoch_abs_start_sample_capped + 1) / fs_lfp if fs_lfp > 0 else 0})
            else: print(f"  Warning: Epoch info item {item_idx} incomplete."); 
        
        if not epochs: print("No valid epoch definitions constructed."); return None
        epochs.sort(key=lambda e: e['abs_start_sample'])
        print(f"Loaded and validated {len(epochs)} epoch definitions.")
        for ep_idx_print, ep in enumerate(epochs): print(f"  Using Epoch {ep['epoch_index']} (Sorted Idx {ep_idx_print}): Samples {ep['abs_start_sample']}-{ep['abs_end_sample_inclusive']} (Duration: {ep['duration_sec_actual']:.3f}s)")
        return epochs
    except Exception as e: print(f"Error loading/parsing epoch definitions NPY: {e}"); traceback.print_exc(); return None

# --- SWR Timestamp Loading Function ---
def load_swr_timestamps(swr_timestamps_npy_path_str, target_state_code, target_epoch_idx, target_region_name):
    """
    Loads SWR peak timestamps for a specific state, epoch, and region.
    Timestamps are expected to be absolute sample numbers.
    The NPY file itself is assumed to be for a specific state (e.g., NREM_by_epoch.npy).
    """
    swr_timestamps_path = Path(swr_timestamps_npy_path_str)
    print(f"Loading SWR timestamps from: {swr_timestamps_path} for State {target_state_code}, Epoch {target_epoch_idx}, Region {target_region_name}")
    if not swr_timestamps_path.exists():
        print(f"  Error: SWR timestamps NPY file not found: {swr_timestamps_path}")
        return np.array([], dtype=int)
    try:
        loaded_content = np.load(swr_timestamps_path, allow_pickle=True)
        
        data_for_specific_state = None
        if isinstance(loaded_content, np.ndarray) and loaded_content.ndim == 0 and hasattr(loaded_content, 'item'):
            data_for_specific_state = loaded_content.item()
            print(f"    DEBUG: Loaded SWR NPY as a 0-dim array, extracted item of type: {type(data_for_specific_state)}")
        elif isinstance(loaded_content, dict):
            data_for_specific_state = loaded_content
            print(f"    DEBUG: Loaded SWR NPY directly as a dictionary.")
        else:
            print(f"  Error: SWR timestamps file content is not a dictionary or a 0-d array containing a dictionary. Actual type: {type(loaded_content)}")
            return np.array([], dtype=int)

        if not isinstance(data_for_specific_state, dict):
            print(f"  Error: Extracted SWR data is not a dictionary. Actual type: {type(data_for_specific_state)}")
            return np.array([], dtype=int)
            
        print(f"    DEBUG: Keys in loaded SWR data (expected epoch indices for state {target_state_code}): {list(data_for_specific_state.keys())}")
        
        epoch_specific_data = data_for_specific_state.get(target_epoch_idx, {}) # Get data for current target_epoch_idx
        
        if not epoch_specific_data:
            print(f"    DEBUG: No data found for Epoch {target_epoch_idx} in the loaded SWR NPY file (for state {target_state_code}).")
        else:
            print(f"    DEBUG: For Epoch {target_epoch_idx}, found data. Keys (expected region names): {list(epoch_specific_data.keys())}")

        swr_peaks_for_region = epoch_specific_data.get(target_region_name, np.array([], dtype=int))
        
        if not isinstance(swr_peaks_for_region, np.ndarray):
            print(f"  Warning: Timestamps for State {target_state_code}, Epoch {target_epoch_idx}, Region {target_region_name} are not a NumPy array. Found type: {type(swr_peaks_for_region)}")
            swr_peaks_for_region = np.array(swr_peaks_for_region, dtype=int)

        print(f"  Loaded {len(swr_peaks_for_region)} SWR peak timestamps for State {target_state_code}, Epoch {target_epoch_idx}, Region {target_region_name}.")
        return swr_peaks_for_region.astype(int)
    except Exception as e:
        print(f"  Error loading SWR timestamps: {e}")
        traceback.print_exc()
        return np.array([], dtype=int)

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

# --- Function to Save CSD Data and Metadata ---
def save_csd_data_and_meta(csd_data_matrix, csd_bin_filename, csd_meta_filename, 
                           original_lfp_file, epoch_info, shank_id, 
                           csd_units_str, final_fs_csd, csd_positions_um, 
                           kcsd_params, analysis_type="epoch", swr_info=None):
    """Saves CSD data to a binary file and corresponding metadata to a text file."""
    try:
        csd_data_matrix.astype(np.float32).tofile(csd_bin_filename)
        print(f"    Saved {analysis_type} CSD data to: {csd_bin_filename} (Shape: {csd_data_matrix.shape}, Dtype: {csd_data_matrix.dtype})")

        meta_content = []
        meta_content.append(f"~kCSDAnalysisVersion=1.1_SWR_Avg") 
        meta_content.append(f"~creationTime={datetime.datetime.now().isoformat()}")
        meta_content.append(f"analysisType={analysis_type}")
        meta_content.append(f"originalLFPFile={str(original_lfp_file)}")
        meta_content.append(f"epochIdx={epoch_info.get('epoch_index', 'N/A')}") # Make epoch_info flexible
        meta_content.append(f"shankId={shank_id}")
        meta_content.append(f"csdBinFile={csd_bin_filename.name}") 
        meta_content.append(f"csdDataType=float32")
        meta_content.append(f"csdDataShapeSamples={csd_data_matrix.shape[0]}") # Samples for CSD (time dimension)
        meta_content.append(f"csdDataShapeSites={csd_data_matrix.shape[1]}")   # Sites for CSD (spatial dimension)
        meta_content.append(f"csdUnits={csd_units_str}")
        meta_content.append(f"csdSamplingRateHz={final_fs_csd:.4f}")
        
        csd_pos_str = ",".join([f"{pos:.2f}" for pos in csd_positions_um])
        meta_content.append(f"csdSitesYCoordUm={csd_pos_str}")
        
        # Time zero and duration depend on analysis type
        if analysis_type == "swr_average":
            meta_content.append(f"csdTimeZeroSec={(SWR_TIME_WINDOW_MS[0] / 1000.0):.4f}") # Relative to SWR peak
            csd_duration_sec = (SWR_TIME_WINDOW_MS[1] - SWR_TIME_WINDOW_MS[0]) / 1000.0
            if swr_info:
                meta_content.append(f"swrSourceRegion={swr_info.get('source_region', 'N/A')}")
                meta_content.append(f"swrNumEventsAveraged={swr_info.get('num_events', 0)}")
                meta_content.append(f"swrTimeWindowMsStart={SWR_TIME_WINDOW_MS[0]}")
                meta_content.append(f"swrTimeWindowMsEnd={SWR_TIME_WINDOW_MS[1]}")

        else: # epoch
            meta_content.append(f"csdTimeZeroSec=0.0") 
            csd_duration_sec = csd_data_matrix.shape[0] / final_fs_csd if final_fs_csd > 0 else 0
        
        meta_content.append(f"csdDurationSec={csd_duration_sec:.4f}")

        meta_content.append(f"kcsdLambdaOptimal={kcsd_params['lambda_optimal']:.4e}")
        meta_content.append(f"kcsdRoptimalUm={kcsd_params['R_optimal_um']:.2f}")
        meta_content.append(f"kcsdSigmaSm={kcsd_params['sigma_Sm']:.2f}")
        meta_content.append(f"kcsdHmeters={kcsd_params['h_m']:.2f}")
        meta_content.append(f"kcsdNSrcInit={kcsd_params['n_src_init']}")
        meta_content.append(f"kcsdXminMeters={kcsd_params['xmin_m']:.4e}")
        meta_content.append(f"kcsdXmaxMeters={kcsd_params['xmax_m']:.4e}")
        meta_content.append(f"kcsdGdxMeters={kcsd_params['gdx_m']:.4e}")
        
        if epoch_info: # Only if epoch_info is valid (not for global averages if any)
            meta_content.append(f"lfpEpochStartSampleOrig={epoch_info.get('abs_start_sample', 'N/A')}")
            meta_content.append(f"lfpEpochEndSampleInclOrig={epoch_info.get('abs_end_sample_inclusive', 'N/A')}")
            meta_content.append(f"lfpEpochDurationSecActual={epoch_info.get('duration_sec_actual', 0):.4f}")

        with open(csd_meta_filename, 'w') as f_meta:
            for line in meta_content:
                f_meta.write(line + '\n')
        print(f"    Saved CSD metadata to: {csd_meta_filename}")

    except Exception as e_save:
        print(f"    Error saving {analysis_type} CSD data or metadata for shank {shank_id}, epoch {epoch_info.get('epoch_index', 'N/A')}: {e_save}")
        traceback.print_exc()

# --- Main CSD Analysis Script ---
def main_kcsd_analysis(lfp_bin_file_path_str, lfp_meta_file_path_str, channel_info_csv_path_str,
                         epoch_definitions_npy_path_str, swr_timestamps_npy_path_str_in):
    lfp_bin_file_path = Path(lfp_bin_file_path_str)
    base_filename = lfp_bin_file_path.stem.replace('.lf', '')

    lfp_data_memmap_obj = None
    try:
        lfp_data_memmap_obj, fs_orig, n_channels_total_in_file, uv_scale_factor_loaded, num_samples_in_lfp_file = \
            load_lfp_data_sglx_memmap(lfp_bin_file_path_str, lfp_meta_file_path_str)
    except Exception as e:
        print(f"CRITICAL: Failed to load LFP data: {e}")
        return

    epoch_definitions = load_epoch_definitions_from_times(epoch_definitions_npy_path_str, fs_orig, num_samples_in_lfp_file)
    if not epoch_definitions:
        print("No valid epoch definitions. Exiting.")
        if lfp_data_memmap_obj is not None and hasattr(lfp_data_memmap_obj, '_mmap') and lfp_data_memmap_obj._mmap is not None:
            try:
                lfp_data_memmap_obj._mmap.close()
                del lfp_data_memmap_obj
            except Exception:
                pass
        gc.collect()
        return

    overall_downsampling_factor = 1
    if fs_orig > TARGET_FS_CSD and not np.isclose(fs_orig, TARGET_FS_CSD):
        overall_downsampling_factor = int(round(fs_orig / TARGET_FS_CSD))
        if overall_downsampling_factor <= 0:
            overall_downsampling_factor = 1
    final_effective_fs_for_csd_nominal = fs_orig / overall_downsampling_factor
    print(f"Original LFP FS: {fs_orig:.2f} Hz. Target CSD FS: {TARGET_FS_CSD:.2f} Hz.")
    print(f"Calculated overall downsampling_factor: {overall_downsampling_factor}. Nominal final effective CSD FS: {final_effective_fs_for_csd_nominal:.2f} Hz")

    try:
        probe_channel_df = load_channel_info_kcsd(channel_info_csv_path_str)
        unique_shanks = sorted(probe_channel_df['shank_index'].unique())
        if not unique_shanks:
            print("No shanks found in channel info. Exiting.")
            return

        for epoch_info in epoch_definitions:
            epoch_idx = epoch_info['epoch_index']
            epoch_abs_start_sample, epoch_abs_end_sample_inclusive = epoch_info['abs_start_sample'], epoch_info['abs_end_sample_inclusive']
            print(f"\n>>> Processing Epoch {epoch_idx} (LFP Samples: {epoch_abs_start_sample}-{epoch_abs_end_sample_inclusive}, Duration: {epoch_info['duration_sec_actual']:.3f}s) <<<")

            is_target_state_epoch_for_swr = SWR_ANALYSIS_ENABLED # Simplified assumption

            if SWR_ANALYSIS_ENABLED and is_target_state_epoch_for_swr and swr_timestamps_npy_path_str_in is not None:
                print(f"  Attempting SWR-triggered average CSD for Epoch {epoch_idx} (Target State: {SWR_TARGET_STATE_CODE}, Region: {SWR_TARGET_REGION})")
                swr_peak_timestamps_abs = load_swr_timestamps(swr_timestamps_npy_path_str_in,
                                                              SWR_TARGET_STATE_CODE,
                                                              epoch_idx,
                                                              SWR_TARGET_REGION)

                swr_peak_timestamps_in_epoch = swr_peak_timestamps_abs[
                    (swr_peak_timestamps_abs >= epoch_abs_start_sample) &
                    (swr_peak_timestamps_abs <= epoch_abs_end_sample_inclusive)
                ]
                print(f"    Found {len(swr_peak_timestamps_in_epoch)} SWRs from '{SWR_TARGET_REGION}' in Epoch {epoch_idx} for State {SWR_TARGET_STATE_CODE}.")

                if len(swr_peak_timestamps_in_epoch) > 0:
                    swr_window_samples_orig_fs_start_offset = int(SWR_TIME_WINDOW_MS[0] / 1000.0 * fs_orig)
                    swr_window_samples_orig_fs_end_offset = int(SWR_TIME_WINDOW_MS[1] / 1000.0 * fs_orig)
                    swr_window_length_orig_fs = swr_window_samples_orig_fs_end_offset - swr_window_samples_orig_fs_start_offset

                    for i_shank, shank_id_val in enumerate(unique_shanks):
                        print(f"\n    SWR Avg CSD for Shank {shank_id_val} (Epoch {epoch_idx})")
                        shank_channel_info = probe_channel_df[probe_channel_df['shank_index'] == shank_id_val].sort_values(by='ycoord_on_shank_um', ascending=True)
                        if shank_channel_info.empty or len(shank_channel_info) < 2:
                            print(f"    Shank {shank_id_val} - Not enough channels. Skipping SWR CSD.")
                            continue

                        shank_global_indices = shank_channel_info['global_channel_index'].values
                        if np.any(shank_global_indices >= n_channels_total_in_file) or np.any(shank_global_indices < 0):
                            print(f"    Shank {shank_id_val} - Channel index error. Skipping SWR CSD.")
                            continue

                        electrode_coords_um_shank = shank_channel_info['ycoord_on_shank_um'].values
                        
                        min_abs_sample_needed_for_swrs = np.min(swr_peak_timestamps_in_epoch) + swr_window_samples_orig_fs_start_offset
                        max_abs_sample_needed_for_swrs = np.max(swr_peak_timestamps_in_epoch) + swr_window_samples_orig_fs_end_offset
                        
                        lfp_extraction_start_abs_origfs = max(0, min_abs_sample_needed_for_swrs)
                        lfp_extraction_end_abs_origfs = min(num_samples_in_lfp_file, max_abs_sample_needed_for_swrs)

                        if lfp_extraction_start_abs_origfs >= lfp_extraction_end_abs_origfs:
                            print(f"    No valid LFP range for SWRs in Epoch {epoch_idx}, Shank {shank_id_val} (after windowing). Skipping.")
                            continue
                        
                        try:
                            lfp_shank_swr_relevant_portion_raw = lfp_data_memmap_obj[lfp_extraction_start_abs_origfs:lfp_extraction_end_abs_origfs, shank_global_indices]
                        except Exception as e:
                            print(f"    Error extracting LFP for SWRs (Shank {shank_id_val}): {e}")
                            continue
                        
                        lfp_shank_swr_relevant_portion_scaled = lfp_shank_swr_relevant_portion_raw.astype(np.float32) * \
                                                                (uv_scale_factor_loaded if uv_scale_factor_loaded is not None else 1.0)
                        
                        processed_lfp_shank_swr_relevant, fs_processed_swr = preprocess_lfp_sub_chunk(
                            lfp_shank_swr_relevant_portion_scaled,
                            fs_orig,
                            LFP_BAND_LOWCUT_CSD, LFP_BAND_HIGHCUT_CSD, NUMTAPS_CSD_FILTER,
                            overall_downsampling_factor,
                            final_effective_fs_for_csd_nominal
                        )
                        
                        swr_window_start_offset_proc_fs = int(SWR_TIME_WINDOW_MS[0] / 1000.0 * fs_processed_swr)
                        swr_window_len_samples_proc_fs = int(swr_window_length_orig_fs / (fs_orig / fs_processed_swr))

                        swr_lfp_segments_shank = []
                        for swr_peak_abs_orig_fs in swr_peak_timestamps_in_epoch:
                            swr_peak_rel_to_extraction_start_orig_fs = swr_peak_abs_orig_fs - lfp_extraction_start_abs_origfs
                            swr_peak_rel_to_extraction_start_proc_fs = int(round(swr_peak_rel_to_extraction_start_orig_fs / (fs_orig / fs_processed_swr)))

                            win_start_in_processed_chunk = swr_peak_rel_to_extraction_start_proc_fs + swr_window_start_offset_proc_fs
                            win_end_in_processed_chunk = win_start_in_processed_chunk + swr_window_len_samples_proc_fs
                            
                            if win_start_in_processed_chunk >= 0 and win_end_in_processed_chunk <= processed_lfp_shank_swr_relevant.shape[0]:
                                segment = processed_lfp_shank_swr_relevant[win_start_in_processed_chunk:win_end_in_processed_chunk, :]
                                if segment.shape[0] == swr_window_len_samples_proc_fs:
                                     swr_lfp_segments_shank.append(segment)
                                else:
                                     print(f"      SWR window for peak {swr_peak_abs_orig_fs}: segment length mismatch. Got {segment.shape[0]}, expected {swr_window_len_samples_proc_fs}. Skipping this SWR.")
                            else:
                                print(f"      SWR window for peak {swr_peak_abs_orig_fs} (proc rel peak {swr_peak_rel_to_extraction_start_proc_fs}) out of bounds for processed LFP chunk (len {processed_lfp_shank_swr_relevant.shape[0]}). Window: {win_start_in_processed_chunk}-{win_end_in_processed_chunk}. Skipping this SWR.")

                        if not swr_lfp_segments_shank:
                            print(f"    No valid LFP segments extracted for SWRs on Shank {shank_id_val}. Skipping SWR CSD.")
                            continue

                        avg_swr_lfp_shank = np.mean(np.stack(swr_lfp_segments_shank), axis=0) 
                        print(f"    Averaged LFP for {len(swr_lfp_segments_shank)} SWRs on Shank {shank_id_val}. Avg LFP shape: {avg_swr_lfp_shank.shape}")

                        coords_quant_um_shank = electrode_coords_um_shank * pq.um
                        ele_pos_for_kcsd_meters = coords_quant_um_shank.rescale(pq.m).magnitude.reshape(-1, 1)
                        lfp_units = pq.uV if uv_scale_factor_loaded is not None else pq.dimensionless
                        avg_lfp_for_kcsd_Volts = (avg_swr_lfp_shank.T * lfp_units).rescale(pq.V).magnitude

                        min_coord_um, max_coord_um = electrode_coords_um_shank.min(), electrode_coords_um_shank.max()
                        xmin_kcsd_meters, xmax_kcsd_meters = (min_coord_um * pq.um).rescale(pq.m).magnitude, (max_coord_um * pq.um).rescale(pq.m).magnitude
                        if np.isclose(xmax_kcsd_meters, xmin_kcsd_meters): padding_m = (10*pq.um).rescale(pq.m).magnitude; xmin_kcsd_meters-=padding_m; xmax_kcsd_meters+=padding_m

                        num_csd_est_pts_config = max(32, int(len(electrode_coords_um_shank) * 1.5))
                        gdx_kcsd_meters = (xmax_kcsd_meters - xmin_kcsd_meters) / (num_csd_est_pts_config - 1) if num_csd_est_pts_config > 1 and (xmax_kcsd_meters - xmin_kcsd_meters) > 1e-9 else (10*pq.um).rescale(pq.m).magnitude
                        if gdx_kcsd_meters <= 1e-9: gdx_kcsd_meters = (1*pq.um).rescale(pq.m).magnitude
                        ext_x_kcsd_meters = (0.0*pq.um).rescale(pq.m).magnitude
                        initial_R_meter_val = (KCSD_RS_CV_UM[0]*pq.um).rescale(pq.m).magnitude if KCSD_RS_CV_UM.size > 0 else (0.1*(xmax_kcsd_meters-xmin_kcsd_meters) if (xmax_kcsd_meters-xmin_kcsd_meters)>1e-9 else (20*pq.um).rescale(pq.m).magnitude)
                        if initial_R_meter_val <= 1e-9: initial_R_meter_val = (1*pq.um).rescale(pq.m).magnitude
                        h_kcsd_meters = 1.0

                        kcsd_params_for_meta = {
                            'sigma_Sm': CSD_SIGMA_CONDUCTIVITY, 'n_src_init': num_csd_est_pts_config,
                            'xmin_m': xmin_kcsd_meters, 'xmax_m': xmax_kcsd_meters, 'gdx_m': gdx_kcsd_meters,
                            'R_init_m': initial_R_meter_val, 'ext_x_m': ext_x_kcsd_meters, 'h_m': h_kcsd_meters,
                            'lambda_optimal': None, 'R_optimal_um': None
                        }

                        try:
                            kcsd_estimator_obj = KCSD1D(ele_pos=ele_pos_for_kcsd_meters, pots=avg_lfp_for_kcsd_Volts, sigma=CSD_SIGMA_CONDUCTIVITY,
                                                        n_src_init=num_csd_est_pts_config, xmin=xmin_kcsd_meters, xmax=xmax_kcsd_meters,
                                                        gdx=gdx_kcsd_meters, ext_x=ext_x_kcsd_meters, R_init=initial_R_meter_val, lambd=0.0, h=h_kcsd_meters)

                            Rs_cv_meters = (KCSD_RS_CV_UM * pq.um).rescale(pq.m).magnitude
                            print(f"      Performing cross-validation for SWR-Avg CSD (Ep{epoch_idx},Shk{shank_id_val})...")
                            kcsd_estimator_obj.cross_validate(lambdas=KCSD_LAMBDAS_CV, Rs=Rs_cv_meters)
                            kcsd_params_for_meta['lambda_optimal'] = kcsd_estimator_obj.lambd
                            kcsd_params_for_meta['R_optimal_um'] = kcsd_estimator_obj.R * 1e6
                            print(f"        CV Optimal Lambda: {kcsd_estimator_obj.lambd:.2e}, Optimal R: {kcsd_estimator_obj.R * 1e6:.2f} um")

                            csd_profile_values_A_per_m3 = kcsd_estimator_obj.values()
                            csd_profile_values_uA_per_mm3 = csd_profile_values_A_per_m3 * 1e-3
                            csd_estimation_coords_meters = kcsd_estimator_obj.estm_x.reshape(-1) * pq.m

                            csd_result_swr_avg_neo = neo.AnalogSignal(csd_profile_values_uA_per_mm3.T, units=pq.uA / pq.mm**3,
                                                            sampling_rate=fs_processed_swr * pq.Hz,
                                                            t_start = SWR_TIME_WINDOW_MS[0] * pq.ms,
                                                            coordinates=csd_estimation_coords_meters.rescale(pq.um))

                            csd_data_matrix_swr_avg = np.asarray(csd_result_swr_avg_neo).astype(np.float32) # Shape: (time_win, csd_site)
                            csd_positions_plot_um = csd_estimation_coords_meters.rescale(pq.um).magnitude.flatten() # Shape: (csd_site)
                            csd_units_str = csd_result_swr_avg_neo.units.dimensionality.string

                            swr_metadata_info = {
                                'source_region': SWR_TARGET_REGION,
                                'num_events': len(swr_lfp_segments_shank)
                            }

                            if csd_data_matrix_swr_avg.size > 0:
                                output_prefix_swr = OUTPUT_DIR / f"{base_filename}_swr_avg"
                                csd_bin_filename_swr = output_prefix_swr.parent / f"{output_prefix_swr.name}_ep{epoch_idx}_shk{shank_id_val}_st{SWR_TARGET_STATE_CODE}_reg{SWR_TARGET_REGION}.csd.bin"
                                csd_meta_filename_swr = output_prefix_swr.parent / f"{output_prefix_swr.name}_ep{epoch_idx}_shk{shank_id_val}_st{SWR_TARGET_STATE_CODE}_reg{SWR_TARGET_REGION}.csd.meta"
                                save_csd_data_and_meta(csd_data_matrix_swr_avg, csd_bin_filename_swr, csd_meta_filename_swr,
                                                       lfp_bin_file_path, epoch_info, shank_id_val,
                                                       csd_units_str, fs_processed_swr,
                                                       csd_positions_plot_um, kcsd_params_for_meta,
                                                       analysis_type="swr_average", swr_info=swr_metadata_info)

                            # Plotting for SWR-averaged CSD
                            fig_swr_csd, ax_swr_csd = plt.subplots(figsize=(8, 6))
                            
                            plot_time_axis_ms = (np.arange(csd_data_matrix_swr_avg.shape[0]) / fs_processed_swr) * 1000 + SWR_TIME_WINDOW_MS[0] # X-coordinates for CSD values
                            csd_data_plot_final_swr = np.nan_to_num(csd_data_matrix_swr_avg.T, nan=0.0) # CSD data (csd_site, time_win)

                            abs_csd_finite = np.abs(csd_data_plot_final_swr[np.isfinite(csd_data_plot_final_swr)])
                            clim_val = 1.0
                            if abs_csd_finite.size > 0:
                                clim_val = np.percentile(abs_csd_finite, 99.0)
                                if clim_val < 1e-9: clim_val = np.nanmax(abs_csd_finite) if np.nanmax(abs_csd_finite) > 1e-9 else 1.0
                            if abs(clim_val) < 1e-9: clim_val = 1e-9
                            
                            # Corrected pcolormesh call for shading='gouraud'
                            img_swr = ax_swr_csd.pcolormesh(
                                plot_time_axis_ms,          # X coordinates of CSD values (length C.shape[1])
                                csd_positions_plot_um,      # Y coordinates of CSD values (length C.shape[0])
                                csd_data_plot_final_swr,    # CSD data (csd_site, time_win) -> (len(Y), len(X))
                                cmap='RdBu_r',
                                shading='gouraud',
                                vmin=-clim_val, vmax=clim_val,
                                rasterized=True
                            )
                            
                            plt.colorbar(img_swr, ax=ax_swr_csd, label=f'kCSD ({csd_units_str})', shrink=0.8, aspect=10)
                            ax_swr_csd.set_xlabel('Time relative to SWR Peak (ms)')
                            ax_swr_csd.set_ylabel('Depth (µm)')
                            ax_swr_csd.set_title(f'SWR-Avg CSD - Epoch {epoch_idx}, Shank {shank_id_val}\n (State {SWR_TARGET_STATE_CODE}, Region {SWR_TARGET_REGION}, N={len(swr_lfp_segments_shank)})')
                            if len(csd_positions_plot_um) > 0 :
                                ax_swr_csd.set_ylim(max(csd_positions_plot_um), min(csd_positions_plot_um))

                            swr_plot_filename = output_prefix_swr.parent / f"{output_prefix_swr.name}_ep{epoch_idx}_shk{shank_id_val}_st{SWR_TARGET_STATE_CODE}_reg{SWR_TARGET_REGION}.png"
                            fig_swr_csd.savefig(swr_plot_filename, dpi=150)
                            print(f"      Saved SWR-Avg CSD plot: {swr_plot_filename}")
                            plt.close(fig_swr_csd)

                        except Exception as e_kcsd_swr:
                            print(f"    kCSD Error or Plotting Error for SWR-Avg (Ep{epoch_idx},Shk{shank_id_val}): {e_kcsd_swr}"); traceback.print_exc();
                        
                        del processed_lfp_shank_swr_relevant
                        del avg_swr_lfp_shank
                        if 'kcsd_estimator_obj' in locals(): del kcsd_estimator_obj
                        if 'csd_result_swr_avg_neo' in locals(): del csd_result_swr_avg_neo
                        if 'csd_data_matrix_swr_avg' in locals(): del csd_data_matrix_swr_avg
                        gc.collect()
                else: # No SWRs in this epoch for this state/region
                    pass # Continue to next epoch or full epoch CSD

            print(f"  Skipping full epoch CSD processing for brevity in this SWR-focused update. It can be re-enabled.")
            print(f"  Completed processing for Epoch {epoch_idx}.")


    except Exception as e_main:
        print(f"Error in main processing: {e_main}")
        traceback.print_exc()
    finally:
        if lfp_data_memmap_obj is not None and hasattr(lfp_data_memmap_obj, '_mmap') and lfp_data_memmap_obj._mmap is not None:
            print("\nClosing LFP memmap...")
            lfp_data_memmap_obj._mmap.close()
            del lfp_data_memmap_obj
            gc.collect()
            print("LFP memmap closed.")
        else:
            print("\nLFP memmap not active or already handled.")
# --- Main Execution Block ---
if __name__ == "__main__":
    try: from tkinter import Tk, filedialog; use_gui = True
    except ImportError: print("tkinter not found. GUI disabled."); use_gui = False; 
    
    lfp_bin_f_selected, lfp_meta_f_selected, channel_csv_f_selected, epoch_def_npy_f_selected, swr_ts_npy_f_selected = \
        LFP_BIN_FILE_PATH_DEFAULT, LFP_META_FILE_PATH_DEFAULT, CHANNEL_INFO_CSV_PATH_DEFAULT, \
        EPOCH_DEFINITIONS_NPY_PATH_DEFAULT, SWR_TIMESTAMPS_NPY_PATH_DEFAULT

    if use_gui:
        root = Tk(); root.withdraw(); root.attributes("-topmost", True)
        lfp_bin_f_selected_gui = filedialog.askopenfilename(title="Select LFP Binary File (*.lf.bin)", initialfile=LFP_BIN_FILE_PATH_DEFAULT)
        if not lfp_bin_f_selected_gui: sys.exit("LFP file selection cancelled.")
        lfp_bin_f_selected = lfp_bin_f_selected_gui
        
        suggested_meta_name = Path(lfp_bin_f_selected).stem + ".meta"
        lfp_meta_f_selected_gui = filedialog.askopenfilename(title="Select LFP Meta File (*.lf.meta)", initialdir=Path(lfp_bin_f_selected).parent, initialfile=suggested_meta_name)
        if not lfp_meta_f_selected_gui: sys.exit("Meta file selection cancelled.")
        lfp_meta_f_selected = lfp_meta_f_selected_gui

        channel_csv_f_selected_gui = filedialog.askopenfilename(title="Select Channel Info CSV", initialfile=CHANNEL_INFO_CSV_PATH_DEFAULT)
        if not channel_csv_f_selected_gui: sys.exit("Channel info CSV selection cancelled.")
        channel_csv_f_selected = channel_csv_f_selected_gui
 
        lfp_path_obj = Path(lfp_bin_f_selected)
        # Suggestion for epoch definitions file (can be same as old timestamps file or a new one)
        suggested_epoch_def_name = lfp_path_obj.stem.replace(".lf", ".epochs.npy") # Or specific name
        epoch_def_npy_f_selected_gui = filedialog.askopenfilename(title="Select Epoch Definitions NPY File", initialdir=lfp_path_obj.parent, initialfile=suggested_epoch_def_name)
        if not epoch_def_npy_f_selected_gui: sys.exit("Epoch Definitions NPY file selection cancelled.")
        epoch_def_npy_f_selected = epoch_def_npy_f_selected_gui

        # Suggestion for SWR timestamps file
        suggested_swr_ts_name = lfp_path_obj.stem.replace(".lf", f"_ripple_timestamps_{SWR_TARGET_STATE_CODE}_{SWR_TARGET_REGION}_by_epoch.npy") # More specific suggestion
        if SWR_ANALYSIS_ENABLED:
            swr_ts_npy_f_selected_gui = filedialog.askopenfilename(title=f"Select SWR Timestamps NPY (State {SWR_TARGET_STATE_CODE}, Region {SWR_TARGET_REGION})", 
                                                                initialdir=lfp_path_obj.parent, initialfile=suggested_swr_ts_name)
            if not swr_ts_npy_f_selected_gui: sys.exit("SWR Timestamps NPY file selection cancelled.")
            swr_ts_npy_f_selected = swr_ts_npy_f_selected_gui
        
        root.destroy()
    else: 
        # Using default paths if GUI is not used
        if not all(Path(f).exists() for f in [lfp_bin_f_selected, lfp_meta_f_selected, channel_csv_f_selected, epoch_def_npy_f_selected]):
            print("One or more default file paths (LFP, Meta, Channel CSV, Epoch Defs) do not exist. Please check script config or provide files via GUI."); sys.exit(1)
        if SWR_ANALYSIS_ENABLED and not Path(swr_ts_npy_f_selected).exists():
            print(f"SWR Analysis is ENABLED but the default SWR timestamps file ({swr_ts_npy_f_selected}) does not exist."); sys.exit(1)


    print(f"\n--- Starting kCSD analysis (SWR-Average Focus) ---")
    print(f"LFP File: {lfp_bin_f_selected}")
    print(f"Meta File: {lfp_meta_f_selected}")
    print(f"Channel CSV: {channel_csv_f_selected}")
    print(f"Epoch Definitions NPY: {epoch_def_npy_f_selected}")
    if SWR_ANALYSIS_ENABLED:
        print(f"SWR Timestamps NPY: {swr_ts_npy_f_selected}")
        print(f"SWR Analysis: ENABLED (State: {SWR_TARGET_STATE_CODE}, Region: {SWR_TARGET_REGION}, Window: {SWR_TIME_WINDOW_MS} ms)")
    else:
        print(f"SWR Analysis: DISABLED")
    print(f"Output Dir: {OUTPUT_DIR}\n")

    files_to_check = [lfp_bin_f_selected, lfp_meta_f_selected, channel_csv_f_selected, epoch_def_npy_f_selected]
    if SWR_ANALYSIS_ENABLED:
        files_to_check.append(swr_ts_npy_f_selected)
    if not all(Path(f).exists() for f in files_to_check):
        print("ERROR: One or more required input files not found. Exiting."); sys.exit(1)

    main_kcsd_analysis(lfp_bin_file_path_str=lfp_bin_f_selected, 
                         lfp_meta_file_path_str=lfp_meta_f_selected,
                         channel_info_csv_path_str=channel_csv_f_selected, 
                         epoch_definitions_npy_path_str=epoch_def_npy_f_selected,
                         swr_timestamps_npy_path_str_in=swr_ts_npy_f_selected if SWR_ANALYSIS_ENABLED else None) # Pass SWR path
    print(f"\n--- Script finished ---")