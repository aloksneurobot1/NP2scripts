# -*- coding: utf-8 -*-
"""
Created on Fri May 16 11:45:23 2025
CSD plots and data saved per shank per epoch for a region ( wrt. peak ripple time of a sleep state)
 for electrodes in specific regions only (CA!, CA2, CA3, DG-mo)
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
import datetime # For metadata timestamp
from DemoReadSGLXData.readSGLX import readMeta

# Import KCSD1D from elephant and alias it
from elephant.current_source_density_src.KCSD import KCSD1D as ElephantKCSD1D


# --- Configuration Parameters ---
LFP_BIN_FILE_PATH_DEFAULT = "your_lfp_file.lf.bin"
LFP_META_FILE_PATH_DEFAULT = "your_lfp_file.lf.meta"
CHANNEL_INFO_CSV_PATH_DEFAULT = "your_channel_info.csv" 
EPOCH_DEFINITIONS_NPY_PATH_DEFAULT = "your_epoch_definitions.nidq_timestamps.npy"

REGIONS_TO_ANALYZE = ['CA1', 'CA2', 'CA3', 'DG-mo'] # Specify regions to include in analysis

# --- SWR Analysis Specific Configuration ---
SWR_TIMESTAMPS_NPY_PATH_DEFAULT = "your_swr_timestamps_NREM_by_epoch.npy"
SWR_ANALYSIS_ENABLED = True
SWR_TARGET_STATE_CODE = 1
SWR_TARGET_REGION = 'CA1'
SWR_TIME_WINDOW_MS = np.array([-100, 100])

LFP_PLOT_OVERLAY_SCALE = 0.5  # Scales LFP amplitude for visibility on CSD plot depth axis
RIPPLE_OVERLAY_LOWCUT = 100.0  # Hz, for LFP trace overlay filtering
RIPPLE_OVERLAY_HIGHCUT = 250.0 # Hz, for LFP trace overlay filtering
RIPPLE_OVERLAY_FILTER_NUMTAPS = 51 # Number of taps for the ripple overlay filter

PLOT_INDIVIDUAL_SWR_CSDS = True
MAX_INDIVIDUAL_SWR_PLOTS_PER_SHANK_EPOCH = 15 # Limit plots if > 0. Set to 0 or negative for no limit.

OUTPUT_DIR = Path("./csd_kcsd_output_swr_individual_v3") # Updated output directory name
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_FS_CSD = 1250.0
LFP_BAND_LOWCUT_CSD = 1.0 # For CSD calculation (broadband LFP for context)
LFP_BAND_HIGHCUT_CSD = 300.0 # For CSD calculation
NUMTAPS_CSD_FILTER = 101

CSD_SIGMA_CONDUCTIVITY = 0.3
KCSD_LAMBDAS_CV = np.logspace(-7, -2, 9)
KCSD_RS_CV_UM = np.logspace(np.log10(20), np.log10(500), 9)

EPOCH_SUB_CHUNK_DURATION_SECONDS = 10


# --- Custom KCSD1D Class for CV Plotting ---
class MyKCSD1D(ElephantKCSD1D):
    def __init__(self, ele_pos, pots, **kwargs):
        super().__init__(ele_pos, pots, **kwargs)
        self.cv_error_matrix = None

    def cross_validate(self, lambdas=None, Rs=None):
        if lambdas is None:
            lambdas = np.logspace(-2, -25, 25, base=10.)
            lambdas = np.hstack((lambdas, np.array((0.0))))
        elif hasattr(lambdas, 'size') and lambdas.size == 1: lambdas = lambdas.flatten()
        elif not hasattr(lambdas, 'size'): lambdas = np.array([lambdas])
        if Rs is None: Rs = np.array((self.R)).flatten()
        elif hasattr(Rs, 'size') and Rs.size == 1: Rs = Rs.flatten()
        elif not hasattr(Rs, 'size'): Rs = np.array([Rs])
        errs = np.zeros((Rs.size, lambdas.size)); index_generator = []
        for ii in range(self.n_ele):
            idx_test = [ii]; idx_train = list(range(self.n_ele)); idx_train.remove(ii)
            index_generator.append((idx_train, idx_test))
        for R_idx, R_val_meters in enumerate(Rs):
            self.update_R(R_val_meters)
            for lambd_idx, lambd_val in enumerate(lambdas):
                errs[R_idx, lambd_idx] = self.compute_cverror(lambd_val, index_generator)
        self.cv_error_matrix = errs
        min_error_indices = np.where(self.cv_error_matrix == np.min(self.cv_error_matrix))
        cv_R_meters = Rs[min_error_indices[0][0]]; cv_lambda = lambdas[min_error_indices[1][0]]
        self.cv_error = np.min(self.cv_error_matrix)
        self.update_R(cv_R_meters); self.update_lambda(cv_lambda)
        return cv_R_meters, cv_lambda

# --- Voltage Scaling Function ---
def get_voltage_scaling_factor(meta):
    try:
        v_max = float(meta['imAiRangeMax']); i_max_adc_val = float(meta['imMaxInt'])
        probe_type = int(meta.get('imDatPrb_type', 0)); lfp_gain = None
        if probe_type in [21, 24, 2013]: lfp_gain = 80.0
        else:
            general_lfp_gain_key_str = "~imChanLFGain"
            if general_lfp_gain_key_str in meta: lfp_gain = float(meta[general_lfp_gain_key_str])
            else:
                first_lfp_gain_key_found = None
                sorted_keys = sorted([key for key in meta.keys() if key.startswith('imChanLFGain') and key.endswith('lfGain')])
                if sorted_keys: first_lfp_gain_key_found = sorted_keys[0]
                if first_lfp_gain_key_found: lfp_gain = float(meta[first_lfp_gain_key_found])
                else: lfp_gain = 250.0; print(f"  Probe type {probe_type}. No specific LFP gain key found. Defaulting to {lfp_gain}.")
        if lfp_gain is None: raise ValueError("LFP gain undetermined.")
        if i_max_adc_val == 0 or lfp_gain == 0: raise ValueError("i_max_adc_val or LFP gain is zero.")
        scaling_factor_uv = (v_max / i_max_adc_val) * (1.0 / lfp_gain) * 1e6
        print(f"  Calculated uV scaling factor: {scaling_factor_uv:.6f} (v_max={v_max}, i_max_adc={i_max_adc_val}, lfp_gain={lfp_gain})")
        return scaling_factor_uv
    except Exception as e: print(f"Error calculating voltage scaling factor: {e}"); traceback.print_exc(); return None

# --- LFP Data Loading Function ---
def load_lfp_data_sglx_memmap(bin_file_path_str, meta_file_path_str):
    bin_file_path = Path(bin_file_path_str); meta_file_path = Path(meta_file_path_str)
    print(f"Setting up LFP data access from: {bin_file_path}\nUsing metadata from: {meta_file_path}")
    try:
        meta = readMeta(meta_file_path); fs_orig = float(meta['imSampRate']); n_channels_total = int(meta['nSavedChans'])
        print(f"  Meta: {n_channels_total} channels, FS: {fs_orig:.2f} Hz.")
        uv_scale_factor = get_voltage_scaling_factor(meta)
        if uv_scale_factor is None: print("  Warning: Voltage scaling factor undetermined. LFP in ADC units.")
        file_size = bin_file_path.stat().st_size; item_size = np.dtype('int16').itemsize
        if n_channels_total == 0 or item_size == 0: raise ValueError("Meta: Invalid nSavedChans or itemsize.")
        num_samples_in_file = file_size // (n_channels_total * item_size)
        if file_size % (n_channels_total * item_size) != 0: print(f"  Warning: File size not integer multiple. Using {num_samples_in_file} samples.")
        if num_samples_in_file <= 0: raise ValueError("Zero or negative samples calculated.")
        print(f"  Total samples in LFP file: {num_samples_in_file}. Memmap shape: ({num_samples_in_file}, {n_channels_total})")
        lfp_data_memmap = np.memmap(bin_file_path, dtype='int16', mode='r', shape=(num_samples_in_file, n_channels_total))
        print(f"  Successfully memory-mapped LFP data.")
        return lfp_data_memmap, fs_orig, n_channels_total, uv_scale_factor, num_samples_in_file
    except Exception as e: print(f"Error in load_lfp_data_sglx_memmap: {e}"); traceback.print_exc(); raise

# --- Channel Info Loading (Updated) ---
def load_channel_info_kcsd(csv_filepath_str):
    csv_filepath = Path(csv_filepath_str); print(f"Loading channel info from {csv_filepath}")
    try:
        channel_df = pd.read_csv(csv_filepath)
        required_cols = ['global_channel_index', 'shank_index', 'ycoord_on_shank_um']
        acronym_col_found = None; possible_acronym_cols = ['acronym', 'region', 'brain_area', 'area', 'allen_ontology_acronym']
        for col_name in possible_acronym_cols:
            if col_name in channel_df.columns:
                acronym_col_found = col_name
                if col_name != 'acronym': channel_df.rename(columns={col_name: 'acronym'}, inplace=True)
                break
        if acronym_col_found is None : print(f"Warning: No standard region/acronym column found (tried: {possible_acronym_cols}). Region filtering/labeling may fail.")
        if not all(col in channel_df.columns for col in required_cols):
            raise ValueError(f"Channel info CSV must at least contain columns: {required_cols}")
        for col in ['global_channel_index', 'shank_index']:
            channel_df[col] = pd.to_numeric(channel_df[col], errors='coerce').astype('Int64')
        channel_df['ycoord_on_shank_um'] = pd.to_numeric(channel_df['ycoord_on_shank_um'], errors='coerce').astype(float)
        channel_df.dropna(subset=required_cols, inplace=True)
        for col in ['global_channel_index', 'shank_index']: channel_df[col] = channel_df[col].astype(int)
        print(f"Loaded and validated channel info for {len(channel_df)} channels.")
        if acronym_col_found: print(f"  Found region information using column: '{acronym_col_found}' (standardized to 'acronym')")
        return channel_df
    except Exception as e: print(f"Error loading channel info: {e}"); traceback.print_exc(); sys.exit(1)

# --- Epoch Definitions Loading ---
def load_epoch_definitions_from_times(timestamps_npy_path_str, fs_lfp, total_lfp_samples_in_file):
    timestamps_npy_path = Path(timestamps_npy_path_str); print(f"Loading epoch definitions from: {timestamps_npy_path}")
    if not timestamps_npy_path.exists(): print(f"Error: Epoch definitions NPY file not found: {timestamps_npy_path}"); return None
    try:
        loaded_npy = np.load(timestamps_npy_path, allow_pickle=True); data_to_process = None; epoch_info_list = None
        if isinstance(loaded_npy, dict): data_to_process = loaded_npy; print("  Loaded NPY as direct dictionary.")
        elif isinstance(loaded_npy, np.ndarray) and loaded_npy.ndim == 0 and hasattr(loaded_npy, 'item') and isinstance(loaded_npy.item(), dict):
            data_to_process = loaded_npy.item(); print("  Loaded NPY as 0-dim array; extracted dictionary.")
        if data_to_process is not None and 'EpochFrameData' in data_to_process:
            epoch_info_list = data_to_process['EpochFrameData']
            if not isinstance(epoch_info_list, list): print(f"  Error: 'EpochFrameData' not a list."); return None
            print(f"  Found 'EpochFrameData' with {len(epoch_info_list)} entries.")
        elif isinstance(loaded_npy, np.ndarray) and loaded_npy.dtype.fields and all(f in loaded_npy.dtype.fields for f in ['start_time_sec', 'end_time_sec', 'epoch_index']):
            epoch_info_list = list(loaded_npy); print(f"  Interpreting NPY as structured array of {len(epoch_info_list)} epochs.")
        if epoch_info_list is None: print(f"Error: Epoch NPY format not recognized. Type: {type(loaded_npy)}"); return None
        epochs = []
        for item_idx, item in enumerate(epoch_info_list):
            start_s, end_s, ep_idx = item.get('start_time_sec') if isinstance(item,dict) else item['start_time_sec'], \
                                     item.get('end_time_sec') if isinstance(item,dict) else item['end_time_sec'], \
                                     item.get('epoch_index') if isinstance(item,dict) else item['epoch_index']
            if start_s is None or end_s is None or ep_idx is None: print(f"  Warning: Epoch item {item_idx} incomplete."); continue
            start_samp, end_samp_incl = int(round(float(start_s)*fs_lfp)), int(round(float(end_s)*fs_lfp))-1
            if end_samp_incl < start_samp: end_samp_incl = start_samp
            start_cap, end_cap_incl = max(0,start_samp), min(end_samp_incl, total_lfp_samples_in_file-1)
            if start_cap > end_cap_incl or start_cap >= total_lfp_samples_in_file: print(f"  Warning: Epoch {ep_idx} invalid sample range. Skipping."); continue
            epochs.append({'epoch_index':int(ep_idx), 'abs_start_sample':start_cap, 'abs_end_sample_inclusive':end_cap_incl,
                           'duration_sec_actual':(end_cap_incl-start_cap+1)/fs_lfp if fs_lfp>0 else 0})
        if not epochs: print("No valid epoch definitions."); return None
        epochs.sort(key=lambda e: e['abs_start_sample']); print(f"Loaded {len(epochs)} epoch definitions.")
        return epochs
    except Exception as e: print(f"Error loading epoch definitions: {e}"); traceback.print_exc(); return None

# --- SWR Timestamp Loading Function ---
def load_swr_timestamps(swr_timestamps_npy_path_str, target_state_code, target_epoch_idx, target_region_name):
    swr_path = Path(swr_timestamps_npy_path_str)
    print(f"Loading SWR timestamps from: {swr_path} for State {target_state_code}, Epoch {target_epoch_idx}, Region {target_region_name}")
    if not swr_path.exists(): print(f"  Error: SWR NPY file not found: {swr_path}"); return np.array([],dtype=int)
    try:
        loaded_content = np.load(swr_path, allow_pickle=True); data_for_state = None
        if isinstance(loaded_content, np.ndarray) and loaded_content.ndim==0 and hasattr(loaded_content,'item'): data_for_state=loaded_content.item()
        elif isinstance(loaded_content, dict): data_for_state = loaded_content
        else: print(f"  Error: SWR file content not dict/0-d array dict. Type: {type(loaded_content)}"); return np.array([],dtype=int)
        if not isinstance(data_for_state, dict): print(f"  Error: Extracted SWR data not dict. Type: {type(data_for_state)}"); return np.array([],dtype=int)
        # print(f"    DEBUG: Keys in loaded SWR data (expected epoch indices for state {target_state_code}): {list(data_for_state.keys())}")
        epoch_data = data_for_state.get(target_epoch_idx, {})
        # if not epoch_data: print(f"    DEBUG: No data for Epoch {target_epoch_idx} in NPY (state {target_state_code}).")
        # else: print(f"    DEBUG: For Epoch {target_epoch_idx}, found data. Keys (expected regions): {list(epoch_data.keys())}")
        swr_peaks = epoch_data.get(target_region_name, np.array([],dtype=int))
        if not isinstance(swr_peaks, np.ndarray):
            print(f"  Warning: Timestamps for S{target_state_code} E{target_epoch_idx} R{target_region_name} not NumPy array. Type: {type(swr_peaks)}")
            swr_peaks = np.array(swr_peaks, dtype=int)
        print(f"  Loaded {len(swr_peaks)} SWR peak timestamps for State {target_state_code}, Epoch {target_epoch_idx}, Region {target_region_name}.")
        return swr_peaks.astype(int)
    except Exception as e: print(f"  Error loading SWR timestamps: {e}"); traceback.print_exc(); return np.array([],dtype=int)

# --- LFP Preprocessing for a SUB-CHUNK ---
def preprocess_lfp_sub_chunk(lfp_sub_chunk_scaled_f32, fs_sub_chunk_in, lowcut, highcut, numtaps, consistent_downsampling_factor, target_overall_fs):
    if lfp_sub_chunk_scaled_f32.ndim == 1: lfp_sub_chunk_scaled_f32 = lfp_sub_chunk_scaled_f32[:, np.newaxis]
    current_fs, lfp_to_process_f32 = fs_sub_chunk_in, lfp_sub_chunk_scaled_f32
    if consistent_downsampling_factor > 1:
        min_samples_for_decimate = consistent_downsampling_factor * 30
        if lfp_to_process_f32.shape[0] > min_samples_for_decimate:
            try:
                lfp_sub_chunk_downsampled_f64 = signal.decimate(lfp_to_process_f32.astype(np.float64), consistent_downsampling_factor, axis=0, ftype='fir', zero_phase=True)
                lfp_to_process_f32, current_fs = lfp_sub_chunk_downsampled_f64.astype(np.float32), current_fs / consistent_downsampling_factor
            except Exception as e_decimate: print(f"    Warning: Decimation error (factor {consistent_downsampling_factor}): {e_decimate}.")
        # else: print(f"    Info: Sub-chunk too short ({lfp_to_process_f32.shape[0]} samples) for decimation by {consistent_downsampling_factor}.") # Can be verbose
    nyq = current_fs / 2.0; actual_highcut = min(highcut, nyq * 0.98); actual_lowcut = max(lowcut, 0.001)
    current_numtaps = numtaps
    if current_numtaps >= lfp_to_process_f32.shape[0]: current_numtaps = lfp_to_process_f32.shape[0] - 1
    if current_numtaps % 2 == 0 and current_numtaps > 0: current_numtaps -=1
    if current_numtaps < 3 or actual_lowcut >= actual_highcut: lfp_sub_chunk_filtered_f32 = lfp_to_process_f32
    else:
        try:
            fir_taps = signal.firwin(current_numtaps, [actual_lowcut, actual_highcut], fs=current_fs, pass_zero='bandpass', window='hamming')
            lfp_sub_chunk_filtered_f64 = signal.filtfilt(fir_taps, 1.0, lfp_to_process_f32.astype(np.float64), axis=0)
            lfp_sub_chunk_filtered_f32 = lfp_sub_chunk_filtered_f64.astype(np.float32)
        except Exception as e_filter: print(f"    Warning: Filtering error: {e_filter}."); lfp_sub_chunk_filtered_f32 = lfp_to_process_f32
    return lfp_sub_chunk_filtered_f32, current_fs

# --- Function to Save CSD Data and Metadata ---
def save_csd_data_and_meta(csd_data_matrix, csd_bin_filename, csd_meta_filename,
                           original_lfp_file, epoch_info, shank_id,
                           csd_units_str, final_fs_csd, csd_positions_um,
                           kcsd_params, analysis_type="epoch", swr_info=None):
    try:
        if csd_data_matrix.ndim == 2 :
            csd_data_matrix.astype(np.float32).tofile(csd_bin_filename)
            print(f"    Saved {analysis_type} CSD data to: {csd_bin_filename} (Shape: {csd_data_matrix.shape}, Dtype: {csd_data_matrix.dtype})")
        elif csd_data_matrix.ndim == 3:
             print(f"    Skipping .bin save for 3D array {analysis_type} CSD data (intended for .npy).")
        else: print(f"    Warning: CSD data matrix has unexpected shape {csd_data_matrix.shape}. Skipping .bin save.")
        meta_content = [f"~kCSDAnalysisVersion=1.3_IndivSWR_RegionFilt", f"~creationTime={datetime.datetime.now().isoformat()}",
                        f"analysisType={analysis_type}", f"originalLFPFile={str(original_lfp_file)}",
                        f"epochIdx={epoch_info.get('epoch_index', 'N/A')}", f"shankId={shank_id}",
                        f"csdUnits={csd_units_str}", f"csdSamplingRateHz={final_fs_csd:.4f}"]
        if csd_data_matrix.ndim == 2:
            meta_content.extend([f"csdBinFile={csd_bin_filename.name}", f"csdDataType=float32",
                                 f"csdDataShapeSamples={csd_data_matrix.shape[0]}", f"csdDataShapeSites={csd_data_matrix.shape[1]}"])
        meta_content.append(f"csdSitesYCoordUm={','.join([f'{pos:.2f}' for pos in csd_positions_um])}")
        csd_duration_sec = 0
        if analysis_type == "swr_average_for_cv":
            meta_content.append(f"csdTimeZeroSec={(SWR_TIME_WINDOW_MS[0] / 1000.0):.4f}")
            csd_duration_sec = (SWR_TIME_WINDOW_MS[1] - SWR_TIME_WINDOW_MS[0]) / 1000.0
            if swr_info: meta_content.extend([f"swrSourceRegion={swr_info.get('source_region', 'N/A')}", f"swrNumEventsAveraged={swr_info.get('num_events', 0)}", f"swrTimeWindowMsStart={SWR_TIME_WINDOW_MS[0]}", f"swrTimeWindowMsEnd={SWR_TIME_WINDOW_MS[1]}"])
        elif analysis_type == "individual_swr_csds_array":
             meta_content.append(f"csdTimeZeroSec_PerSWR={(SWR_TIME_WINDOW_MS[0] / 1000.0):.4f}")
             csd_duration_sec = (SWR_TIME_WINDOW_MS[1] - SWR_TIME_WINDOW_MS[0]) / 1000.0
             if swr_info: meta_content.extend([f"swrSourceRegion={swr_info.get('source_region', 'N/A')}", f"swrNumEventsInArray={swr_info.get('num_events_in_array', 0)}"])
        else: meta_content.append(f"csdTimeZeroSec=0.0"); csd_duration_sec = csd_data_matrix.shape[0] / final_fs_csd if final_fs_csd > 0 and csd_data_matrix.ndim == 2 else 0
        meta_content.append(f"csdDurationSec_PerWindow={csd_duration_sec:.4f}")
        meta_content.extend([f"kcsdOptimalLambdaUsed={kcsd_params['lambda_optimal']:.4e}", f"kcsdOptimalR_um_Used={kcsd_params['R_optimal_um']:.2f}",
                             f"kcsdSigmaSm={kcsd_params['sigma_Sm']:.2f}", f"kcsdHmeters={kcsd_params['h_m']:.2f}",
                             f"kcsdNSrcInit={kcsd_params['n_src_init']}", f"kcsdXminMeters={kcsd_params['xmin_m']:.4e}",
                             f"kcsdXmaxMeters={kcsd_params['xmax_m']:.4e}", f"kcsdGdxMeters={kcsd_params['gdx_m']:.4e}"])
        if epoch_info: meta_content.extend([f"lfpEpochStartSampleOrig={epoch_info.get('abs_start_sample', 'N/A')}", f"lfpEpochEndSampleInclOrig={epoch_info.get('abs_end_sample_inclusive', 'N/A')}", f"lfpEpochDurationSecActual={epoch_info.get('duration_sec_actual', 0):.4f}"])
        with open(csd_meta_filename, 'w') as f_meta:
            for line in meta_content: f_meta.write(line + '\n')
        print(f"    Saved CSD metadata to: {csd_meta_filename}")
    except Exception as e_save: print(f"    Error saving CSD data/metadata: {e_save}"); traceback.print_exc()

# --- Main CSD Analysis Script ---
def main_kcsd_analysis(lfp_bin_file_path_str, lfp_meta_file_path_str, channel_info_csv_path_str,
                         epoch_definitions_npy_path_str, swr_timestamps_npy_path_str_in):
    lfp_bin_file_path = Path(lfp_bin_file_path_str); base_filename = lfp_bin_file_path.stem.replace('.lf', '')
    lfp_data_memmap_obj = None
    try:
        lfp_data_memmap_obj, fs_orig, n_channels_total_in_file, uv_scale_factor_loaded, num_samples_in_lfp_file = \
            load_lfp_data_sglx_memmap(lfp_bin_file_path_str, lfp_meta_file_path_str)
    except Exception as e: print(f"CRITICAL: Failed to load LFP: {e}"); return
    epoch_definitions = load_epoch_definitions_from_times(epoch_definitions_npy_path_str, fs_orig, num_samples_in_lfp_file)
    if not epoch_definitions:
        if lfp_data_memmap_obj: hasattr(lfp_data_memmap_obj, '_mmap') and lfp_data_memmap_obj._mmap.close(); del lfp_data_memmap_obj
        gc.collect(); return
    overall_downsampling_factor = int(round(fs_orig / TARGET_FS_CSD)) if fs_orig > TARGET_FS_CSD and not np.isclose(fs_orig, TARGET_FS_CSD) else 1
    if overall_downsampling_factor <= 0: overall_downsampling_factor = 1
    final_effective_fs_for_csd_nominal = fs_orig / overall_downsampling_factor
    print(f"Original LFP FS: {fs_orig:.2f} Hz. Target CSD FS: {TARGET_FS_CSD:.2f} Hz. Downsampling: {overall_downsampling_factor}x. Nominal CSD FS: {final_effective_fs_for_csd_nominal:.2f} Hz")
    try:
        probe_channel_df_full = load_channel_info_kcsd(channel_info_csv_path_str)
        if REGIONS_TO_ANALYZE:
            if 'acronym' not in probe_channel_df_full.columns:
                print(f"ERROR: 'acronym' column required for region filtering not in '{channel_info_csv_path_str}'. Exiting."); sys.exit(1)
            probe_channel_df = probe_channel_df_full[probe_channel_df_full['acronym'].isin(REGIONS_TO_ANALYZE)].copy()
            if probe_channel_df.empty: print(f"ERROR: No channels in specified regions: {REGIONS_TO_ANALYZE}. Exiting."); sys.exit(1)
            print(f"Filtered to {len(probe_channel_df)} channels from regions: {REGIONS_TO_ANALYZE}")
        else: print("No specific regions to analyze. Using all loaded channels."); probe_channel_df = probe_channel_df_full.copy()
        unique_shanks = sorted(probe_channel_df['shank_index'].unique())
        if not unique_shanks: print("No shanks after region filter. Exiting."); return

        for epoch_info in epoch_definitions:
            epoch_idx = epoch_info['epoch_index']
            epoch_abs_start_sample, epoch_abs_end_sample_inclusive = epoch_info['abs_start_sample'], epoch_info['abs_end_sample_inclusive']
            print(f"\n>>> Processing Epoch {epoch_idx} (LFP Samples: {epoch_abs_start_sample}-{epoch_abs_end_sample_inclusive}, Duration: {epoch_info['duration_sec_actual']:.3f}s) <<<")
            if SWR_ANALYSIS_ENABLED and swr_timestamps_npy_path_str_in is not None:
                print(f"  Attempting SWR CSD analysis for Epoch {epoch_idx} (State: {SWR_TARGET_STATE_CODE}, Region: {SWR_TARGET_REGION})")
                swr_peak_timestamps_abs = load_swr_timestamps(swr_timestamps_npy_path_str_in, SWR_TARGET_STATE_CODE, epoch_idx, SWR_TARGET_REGION)
                swr_peak_timestamps_in_epoch = swr_peak_timestamps_abs[(swr_peak_timestamps_abs >= epoch_abs_start_sample) & (swr_peak_timestamps_abs <= epoch_abs_end_sample_inclusive)]
                print(f"    Found {len(swr_peak_timestamps_in_epoch)} SWRs from '{SWR_TARGET_REGION}' in Epoch {epoch_idx} for State {SWR_TARGET_STATE_CODE}.")
                if len(swr_peak_timestamps_in_epoch) > 0:
                    swr_win_start_offset_origfs = int(SWR_TIME_WINDOW_MS[0] / 1000.0 * fs_orig)
                    swr_win_len_origfs = int((SWR_TIME_WINDOW_MS[1] - SWR_TIME_WINDOW_MS[0]) / 1000.0 * fs_orig)
                    for i_shank, shank_id_val in enumerate(unique_shanks):
                        print(f"\n    SWR CSD for Shank {shank_id_val} (Epoch {epoch_idx})")
                        shank_channel_info_df = probe_channel_df[probe_channel_df['shank_index'] == shank_id_val].sort_values(by='ycoord_on_shank_um', ascending=True)
                        if shank_channel_info_df.empty or len(shank_channel_info_df) < 2: print(f"    Shank {shank_id_val}: Not enough channels after region filter. Skipping."); continue
                        shank_global_indices = shank_channel_info_df['global_channel_index'].values
                        electrode_coords_um_shank = shank_channel_info_df['ycoord_on_shank_um'].values
                        min_swr_sample_needed = np.min(swr_peak_timestamps_in_epoch) + swr_win_start_offset_origfs
                        max_swr_sample_needed = np.max(swr_peak_timestamps_in_epoch) + swr_win_start_offset_origfs + swr_win_len_origfs
                        lfp_extract_start, lfp_extract_end = max(0, min_swr_sample_needed), min(num_samples_in_lfp_file, max_swr_sample_needed)
                        if lfp_extract_start >= lfp_extract_end: print(f"    No valid LFP range for SWRs on Shank {shank_id_val}. Skipping."); continue
                        try: lfp_shank_relevant_raw = lfp_data_memmap_obj[lfp_extract_start:lfp_extract_end, shank_global_indices]
                        except Exception as e: print(f"    Error extracting LFP for SWRs (Shank {shank_id_val}): {e}"); continue
                        lfp_shank_relevant_scaled = lfp_shank_relevant_raw.astype(np.float32) * (uv_scale_factor_loaded if uv_scale_factor_loaded is not None else 1.0)
                        processed_lfp_relevant, fs_processed = preprocess_lfp_sub_chunk(lfp_shank_relevant_scaled, fs_orig, LFP_BAND_LOWCUT_CSD, LFP_BAND_HIGHCUT_CSD, NUMTAPS_CSD_FILTER, overall_downsampling_factor, final_effective_fs_for_csd_nominal)
                        swr_win_start_offset_procfs = int(SWR_TIME_WINDOW_MS[0] / 1000.0 * fs_processed)
                        swr_win_len_procfs = int(swr_win_len_origfs / (fs_orig / fs_processed))
                        swr_lfp_segments = []; valid_swr_peaks = []
                        for swr_peak_origfs in swr_peak_timestamps_in_epoch:
                            peak_rel_extract_origfs = swr_peak_origfs - lfp_extract_start
                            peak_rel_extract_procfs = int(round(peak_rel_extract_origfs / (fs_orig / fs_processed)))
                            win_start = peak_rel_extract_procfs + swr_win_start_offset_procfs
                            win_end = win_start + swr_win_len_procfs
                            if 0 <= win_start and win_end <= processed_lfp_relevant.shape[0]:
                                segment = processed_lfp_relevant[win_start:win_end, :]
                                if segment.shape[0] == swr_win_len_procfs: swr_lfp_segments.append(segment); valid_swr_peaks.append(swr_peak_origfs)
                                # else: print(f"      SWR window length mismatch for peak {swr_peak_origfs}. Skipping.") # Can be verbose
                            # else: print(f"      SWR window out of bounds for peak {swr_peak_origfs}. Skipping.") # Can be verbose
                        if not swr_lfp_segments: print(f"    No valid LFP segments for SWRs on Shank {shank_id_val}. Skipping."); continue
                        avg_swr_lfp = np.mean(np.stack(swr_lfp_segments), axis=0)
                        print(f"    Averaged LFP for {len(swr_lfp_segments)} SWRs on Shank {shank_id_val} for CV. Shape: {avg_swr_lfp.shape}")
                        coords_quant_um = electrode_coords_um_shank * pq.um; ele_pos_m = coords_quant_um.rescale(pq.m).magnitude.reshape(-1,1)
                        lfp_units = pq.uV if uv_scale_factor_loaded else pq.dimensionless
                        avg_lfp_cv_V = (avg_swr_lfp.T * lfp_units).rescale(pq.V).magnitude
                        min_coord_m, max_coord_m = ele_pos_m.min(), ele_pos_m.max()
                        if np.isclose(max_coord_m, min_coord_m): padding_m=(10*pq.um).rescale(pq.m).magnitude; min_coord_m-=padding_m; max_coord_m+=padding_m
                        n_src_config = max(32, int(len(electrode_coords_um_shank)*1.5))
                        gdx_m = (max_coord_m - min_coord_m)/(n_src_config-1) if n_src_config>1 and (max_coord_m-min_coord_m)>1e-9 else (10*pq.um).rescale(pq.m).magnitude
                        if gdx_m <= 1e-9: gdx_m = (1*pq.um).rescale(pq.m).magnitude
                        R_init_m = (KCSD_RS_CV_UM[0]*pq.um).rescale(pq.m).magnitude if KCSD_RS_CV_UM.size>0 else (0.1*(max_coord_m-min_coord_m) if (max_coord_m-min_coord_m)>1e-9 else (20*pq.um).rescale(pq.m).magnitude)
                        if R_init_m <=1e-9: R_init_m = (1*pq.um).rescale(pq.m).magnitude
                        optimal_lambda, optimal_R_um = None, None; output_prefix_cv = OUTPUT_DIR/f"{base_filename}_swr_avg_CV"
                        kcsd_cv_params = {'sigma_Sm':CSD_SIGMA_CONDUCTIVITY, 'n_src_init':n_src_config, 'xmin_m':min_coord_m, 'xmax_m':max_coord_m, 'gdx_m':gdx_m, 'R_init_m':R_init_m, 'ext_x_m':0.0, 'h_m':1.0, 'lambda_optimal':None, 'R_optimal_um':None}
                        try:
                            kcsd_cv = MyKCSD1D(ele_pos=ele_pos_m, pots=avg_lfp_cv_V, sigma=CSD_SIGMA_CONDUCTIVITY, n_src_init=n_src_config, xmin=min_coord_m, xmax=max_coord_m, gdx=gdx_m, ext_x=0.0, R_init=R_init_m, lambd=0.0, h=1.0)
                            Rs_m = (KCSD_RS_CV_UM*pq.um).rescale(pq.m).magnitude
                            print(f"      Performing CV on SWR-Avg LFP (Ep{epoch_idx},Shk{shank_id_val})..."); kcsd_cv.cross_validate(lambdas=KCSD_LAMBDAS_CV, Rs=Rs_m)
                            optimal_lambda, optimal_R_um = kcsd_cv.lambd, kcsd_cv.R*1e6
                            kcsd_cv_params['lambda_optimal']=optimal_lambda; kcsd_cv_params['R_optimal_um']=optimal_R_um
                            print(f"        CV Optimal Lambda: {optimal_lambda:.2e}, R: {optimal_R_um:.2f} um")
                            if hasattr(kcsd_cv,'cv_error_matrix') and kcsd_cv.cv_error_matrix is not None:
                                fig_cv,ax_cv=plt.subplots(figsize=(8,6)); cv_err_plot=np.log10(kcsd_cv.cv_error_matrix)
                                if np.isneginf(cv_err_plot).any(): fin_min=np.min(cv_err_plot[np.isfinite(cv_err_plot)]) if np.isfinite(cv_err_plot).any() else -10; cv_err_plot[np.isneginf(cv_err_plot)]=fin_min-1
                                vmin,vmax_p = (np.nanmin(cv_err_plot) if np.isfinite(cv_err_plot).any() else -10), (np.nanpercentile(cv_err_plot[np.isfinite(cv_err_plot)],99) if np.isfinite(cv_err_plot).any() else -10)
                                vmax=max(vmax_p,vmin+1e-9); norm=Normalize(vmin=vmin,vmax=vmax); im=ax_cv.imshow(cv_err_plot,aspect='auto',origin='lower',cmap='viridis',norm=norm)
                                ax_cv.set_xticks(np.arange(len(KCSD_LAMBDAS_CV))); ax_cv.set_xticklabels([f"{l:.1e}" for l in KCSD_LAMBDAS_CV],rotation=45,ha="right")
                                ax_cv.set_yticks(np.arange(len(KCSD_RS_CV_UM))); ax_cv.set_yticklabels([f"{r:.1f}" for r in KCSD_RS_CV_UM])
                                best_l_idx, best_R_idx = np.where(np.isclose(KCSD_LAMBDAS_CV,optimal_lambda))[0], np.where(np.isclose(KCSD_RS_CV_UM,optimal_R_um))[0]
                                if len(best_l_idx)>0 and len(best_R_idx)>0: ax_cv.scatter(best_l_idx[0],best_R_idx[0],marker='x',color='red',s=100,label=f'Optimal\nL={optimal_lambda:.1e}\nR={optimal_R_um:.1f}um'); ax_cv.legend(fontsize='small')
                                ax_cv.set_xlabel('Lambda'); ax_cv.set_ylabel('R (um)'); ax_cv.set_title(f'CV Error (SWR-Avg LFP)\nEp{epoch_idx} Shk{shank_id_val} St{SWR_TARGET_STATE_CODE} Reg{SWR_TARGET_REGION}'); plt.colorbar(im,ax=ax_cv,label='log10(CV Error)'); fig_cv.tight_layout()
                                cv_plot_fname=output_prefix_cv.parent/f"{output_prefix_cv.name}_ep{epoch_idx}_shk{shank_id_val}_st{SWR_TARGET_STATE_CODE}_reg{SWR_TARGET_REGION}.png"; fig_cv.savefig(cv_plot_fname); print(f"      Saved CV plot: {cv_plot_fname}"); plt.close(fig_cv)
                            del kcsd_cv
                        except Exception as e_cv: print(f"    Error during CV (Ep{epoch_idx},Shk{shank_id_val}): {e_cv}. Skipping shank."); traceback.print_exc(); continue
                        if optimal_lambda is None or optimal_R_um is None: print(f"    Failed to get optimal CV params for Shk{shank_id_val}. Skipping."); continue
                        optimal_R_m = optimal_R_um/1e6; indiv_csds=[]
                        print(f"    Calculating CSD for {len(swr_lfp_segments)} individual SWRs on Shk{shank_id_val} using fixed CV params...")
                        plots_made_shank_ep=0
                        for swr_i, indiv_lfp_uV in enumerate(swr_lfp_segments):
                            indiv_lfp_V=(indiv_lfp_uV.T*lfp_units).rescale(pq.V).magnitude
                            try:
                                kcsd_single=ElephantKCSD1D(ele_pos=ele_pos_m,pots=indiv_lfp_V,sigma=CSD_SIGMA_CONDUCTIVITY,n_src_init=n_src_config,xmin=min_coord_m,xmax=max_coord_m,gdx=gdx_m,ext_x=0.0,R_init=optimal_R_m,lambd=optimal_lambda,h=1.0)
                                csd_A_m3=kcsd_single.values(); csd_uA_mm3=csd_A_m3*1e-3; indiv_csds.append(csd_uA_mm3)
                                if PLOT_INDIVIDUAL_SWR_CSDS and (plots_made_shank_ep < MAX_INDIVIDUAL_SWR_PLOTS_PER_SHANK_EPOCH or MAX_INDIVIDUAL_SWR_PLOTS_PER_SHANK_EPOCH <= 0):
                                    lfp_overlay_broad=indiv_lfp_uV; lfp_overlay_ripple,_=preprocess_lfp_sub_chunk(lfp_overlay_broad.copy(),fs_processed,RIPPLE_OVERLAY_LOWCUT,RIPPLE_OVERLAY_HIGHCUT,RIPPLE_OVERLAY_FILTER_NUMTAPS,1,fs_processed)
                                    if lfp_overlay_ripple is None: lfp_overlay_ripple=lfp_overlay_broad
                                    csd_plot_data=csd_uA_mm3; peak_ts=valid_swr_peaks[swr_i]
                                    fig_indiv,ax_csd_i=plt.subplots(figsize=(10,8)); time_ms=(np.arange(csd_plot_data.shape[1])/fs_processed)*1000+SWR_TIME_WINDOW_MS[0]
                                    csd_pos_um_i=(kcsd_single.estm_x.reshape(-1)*pq.m).rescale(pq.um).magnitude.flatten()
                                    abs_csd_fin=np.abs(csd_plot_data[np.isfinite(csd_plot_data)]); clim_i=np.percentile(abs_csd_fin,99) if abs_csd_fin.size>0 else 1.0; clim_i=max(clim_i,1e-9)
                                    img_i=ax_csd_i.pcolormesh(time_ms,csd_pos_um_i,csd_plot_data,cmap='RdBu_r',shading='gouraud',vmin=-clim_i,vmax=clim_i,rasterized=True)
                                    plt.colorbar(img_i,ax=ax_csd_i,label='kCSD (uA/mm^3)',shrink=0.8,aspect=15)
                                    ax_csd_i.set_xlabel('Time re SWR Peak (ms)'); ax_csd_i.set_ylabel('CSD Depth (µm)')
                                    ax_csd_i.set_title(f'Indiv. SWR CSD & LFP ({RIPPLE_OVERLAY_LOWCUT:.0f}-{RIPPLE_OVERLAY_HIGHCUT:.0f}Hz) - Ep{epoch_idx} Shk{shank_id_val} SWR#{swr_i}\n(Peak@{peak_ts}) St{SWR_TARGET_STATE_CODE} Reg{SWR_TARGET_REGION}')
                                    if len(csd_pos_um_i)>0: ax_csd_i.set_ylim(max(csd_pos_um_i),min(csd_pos_um_i))
                                    ax_lfp_i=ax_csd_i.twinx(); y_tk_pos,y_tk_lbl=[],[]
                                    for i_el in range(lfp_overlay_ripple.shape[1]):
                                        lfp_tr=lfp_overlay_ripple[:,i_el]; el_depth=electrode_coords_um_shank[i_el]
                                        ch_props=shank_channel_info_df.iloc[i_el]; glob_ch_id=ch_props['global_channel_index']; acr=ch_props.get('acronym','N/A')
                                        ax_lfp_i.plot(time_ms,(lfp_tr*LFP_PLOT_OVERLAY_SCALE)+el_depth,color='black',linewidth=0.7,alpha=0.6)
                                        y_tk_pos.append(el_depth); y_tk_lbl.append(f"Ch{glob_ch_id} ({acr})")
                                    ax_lfp_i.set_yticks(y_tk_pos); ax_lfp_i.set_yticklabels(y_tk_lbl,fontsize=7)
                                    lfp_ymin,lfp_ymax=ax_csd_i.get_ylim(); ax_lfp_i.set_ylim(lfp_ymin,lfp_ymax); ax_lfp_i.invert_yaxis()
                                    fig_indiv.tight_layout(rect=[0,0,0.85,1])
                                    plot_pref_indiv=OUTPUT_DIR/f"{base_filename}_individual_SWR_plots_Shk{shank_id_val}"; plot_pref_indiv.mkdir(parents=True,exist_ok=True)
                                    indiv_fname=plot_pref_indiv/f"indivSWR_CSDLFP_ep{epoch_idx}_shk{shank_id_val}_st{SWR_TARGET_STATE_CODE}_reg{SWR_TARGET_REGION}_swr{swr_i}_ts{peak_ts}.png"
                                    fig_indiv.savefig(indiv_fname,dpi=100); plt.close(fig_indiv); plots_made_shank_ep+=1
                                del kcsd_single
                            except Exception as e_indiv: print(f"      Error processing/plotting indiv SWR#{swr_i} Shk{shank_id_val}: {e_indiv}"); traceback.print_exc();
                        if indiv_csds:
                            all_swr_csds_arr=np.stack(indiv_csds,axis=0)
                            print(f"    Stacked CSDs for {all_swr_csds_arr.shape[0]} SWRs on Shk{shank_id_val}. Shape: {all_swr_csds_arr.shape}")
                            out_pref_indiv_data=OUTPUT_DIR/f"{base_filename}_individual_SWR_CSDs"; out_pref_indiv_data.mkdir(parents=True,exist_ok=True)
                            csd_arr_fname=out_pref_indiv_data/f"all_SWR_CSDs_ep{epoch_idx}_shk{shank_id_val}_st{SWR_TARGET_STATE_CODE}_reg{SWR_TARGET_REGION}.npy"
                            np.save(csd_arr_fname,all_swr_csds_arr); print(f"      Saved array of indiv SWR CSDs: {csd_arr_fname}")
                        del processed_lfp_relevant, avg_swr_lfp, indiv_csds
                        if 'all_swr_csds_arr' in locals(): del all_swr_csds_arr
                        gc.collect()
            print(f"  Completed processing for Epoch {epoch_idx}.")
    except Exception as e_main: print(f"Error in main processing: {e_main}"); traceback.print_exc()
    finally:
        if lfp_data_memmap_obj is not None and hasattr(lfp_data_memmap_obj, '_mmap') and lfp_data_memmap_obj._mmap is not None:
            print("\nClosing LFP memmap..."); lfp_data_memmap_obj._mmap.close(); del lfp_data_memmap_obj; gc.collect(); print("LFP memmap closed.")
        else: print("\nLFP memmap not active or already handled.")

# --- Main Execution Block ---
if __name__ == "__main__":
    try: from tkinter import Tk, filedialog; use_gui = True
    except ImportError: print("tkinter not found. GUI disabled."); use_gui = False
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
        suggested_epoch_def_name = lfp_path_obj.stem.replace(".lf", ".epochs.npy") # More generic
        epoch_def_npy_f_selected_gui = filedialog.askopenfilename(title="Select Epoch Definitions NPY File", initialdir=lfp_path_obj.parent, initialfile=suggested_epoch_def_name)
        if not epoch_def_npy_f_selected_gui: sys.exit("Epoch Definitions NPY file selection cancelled.")
        epoch_def_npy_f_selected = epoch_def_npy_f_selected_gui
        swr_ts_npy_f_selected_gui = None # Initialize
        if SWR_ANALYSIS_ENABLED:
            state_name_for_file = "NREM" if SWR_TARGET_STATE_CODE == 1 else "Awake" if SWR_TARGET_STATE_CODE == 0 else f"State{SWR_TARGET_STATE_CODE}"
            suggested_swr_ts_name = f"{lfp_path_obj.name.split('.')[0]}_ripple_timestamps_{state_name_for_file}_by_epoch.npy"
            swr_ts_npy_f_selected_gui = filedialog.askopenfilename(
                title=f"Select SWR Timestamps NPY (e.g., for State {SWR_TARGET_STATE_CODE}, Region {SWR_TARGET_REGION})",
                initialdir=lfp_path_obj.parent, initialfile=suggested_swr_ts_name)
            if not swr_ts_npy_f_selected_gui: print("SWR Timestamps NPY file selection cancelled. Using default or previously set path if valid.")
            else: swr_ts_npy_f_selected = swr_ts_npy_f_selected_gui
        root.destroy()
    else: # Not using GUI
        if not all(Path(f).exists() for f in [lfp_bin_f_selected, lfp_meta_f_selected, channel_csv_f_selected, epoch_def_npy_f_selected]):
            print("One or more default file paths (LFP, Meta, Channel CSV, Epoch Defs) do not exist. Please check script config or provide files via GUI."); sys.exit(1)
        if SWR_ANALYSIS_ENABLED and (not swr_ts_npy_f_selected or not Path(swr_ts_npy_f_selected).exists()):
            print(f"SWR Analysis is ENABLED but the SWR timestamps file ('{swr_ts_npy_f_selected}') does not exist or is not specified."); sys.exit(1)

    print(f"\n--- Starting kCSD analysis (Individual SWR CSD Focus) ---")
    print(f"LFP File: {lfp_bin_f_selected}\nMeta File: {lfp_meta_f_selected}\nChannel CSV: {channel_csv_f_selected}\nEpoch Definitions NPY: {epoch_def_npy_f_selected}")
    if SWR_ANALYSIS_ENABLED: print(f"SWR Timestamps NPY: {swr_ts_npy_f_selected}\nSWR Analysis: ENABLED (State: {SWR_TARGET_STATE_CODE}, Region: {SWR_TARGET_REGION}, Window: {SWR_TIME_WINDOW_MS} ms)")
    else: print(f"SWR Analysis: DISABLED")
    print(f"Regions to Analyze: {REGIONS_TO_ANALYZE}")
    print(f"Plot individual SWR CSDs: {PLOT_INDIVIDUAL_SWR_CSDS}, Max per shank/epoch: {MAX_INDIVIDUAL_SWR_PLOTS_PER_SHANK_EPOCH}")
    print(f"LFP Overlay Filter: {RIPPLE_OVERLAY_LOWCUT}-{RIPPLE_OVERLAY_HIGHCUT} Hz")
    print(f"Output Dir: {OUTPUT_DIR}\n")

    files_to_check = [lfp_bin_f_selected, lfp_meta_f_selected, channel_csv_f_selected, epoch_def_npy_f_selected]
    if SWR_ANALYSIS_ENABLED:
        if swr_ts_npy_f_selected: files_to_check.append(swr_ts_npy_f_selected)
        else: print("ERROR: SWR Analysis enabled, but no SWR timestamp file path is configured."); sys.exit(1)
    if not all(Path(f).exists() for f in files_to_check):
        print("ERROR: One or more required input files not found. Exiting."); sys.exit(1)

    main_kcsd_analysis(lfp_bin_file_path_str=lfp_bin_f_selected,
                         lfp_meta_file_path_str=lfp_meta_f_selected,
                         channel_info_csv_path_str=channel_csv_f_selected,
                         epoch_definitions_npy_path_str=epoch_def_npy_f_selected,
                         swr_timestamps_npy_path_str_in=swr_ts_npy_f_selected if SWR_ANALYSIS_ENABLED else None)
    print(f"\n--- Script finished ---")