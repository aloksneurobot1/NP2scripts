import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy import signal
import quantities as pq
import neo # For type hinting and potential future use
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

# Import KCSD1D from elephant and alias it
from elephant.current_source_density_src.KCSD import KCSD1D as ElephantKCSD1D


# --- Configuration Parameters ---
LFP_BIN_FILE_PATH_DEFAULT = "your_lfp_file.lf.bin"
LFP_META_FILE_PATH_DEFAULT = "your_lfp_file.lf.meta"
CHANNEL_INFO_CSV_PATH_DEFAULT = "your_channel_info.csv"
EPOCH_DEFINITIONS_NPY_PATH_DEFAULT = "your_epoch_definitions.nidq_timestamps.npy"

# --- SWR Analysis Specific Configuration ---
SWR_TIMESTAMPS_NPY_PATH_DEFAULT = "your_swr_timestamps_NREM_by_epoch.npy"
SWR_ANALYSIS_ENABLED = True
SWR_TARGET_STATE_CODE = 1
SWR_TARGET_REGION = 'CA1' # As per user's last run
SWR_TIME_WINDOW_MS = np.array([-100, 100])
LFP_PLOT_OVERLAY_SCALE = 0.1
PLOT_INDIVIDUAL_SWR_CSDS = False
MAX_INDIVIDUAL_SWR_PLOTS_PER_SHANK_EPOCH = 5 # Optional: Uncomment to limit plots

OUTPUT_DIR = Path("./csd_kcsd_output_swr_individual_v3") #output directory name
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_FS_CSD = 1250.0
LFP_BAND_LOWCUT_CSD = 1.0
LFP_BAND_HIGHCUT_CSD = 300.0
NUMTAPS_CSD_FILTER = 101

RIPPLE_OVERLAY_LOWCUT = 100.0  # Hz, for LFP trace overlay
RIPPLE_OVERLAY_HIGHCUT = 250.0 # Hz, for LFP trace overlay
RIPPLE_OVERLAY_FILTER_NUMTAPS = 51 # Number of taps for the ripple overlay filter (adjust as needed)

CSD_SIGMA_CONDUCTIVITY = 0.3
KCSD_LAMBDAS_CV = np.logspace(-7, -2, 9)
KCSD_RS_CV_UM = np.logspace(np.log10(20), np.log10(500), 9)

EPOCH_SUB_CHUNK_DURATION_SECONDS = 10
# PLOT_DURATION_LIMIT_SECONDS = 300.0 # Not used for SWR plots currently


# --- Custom KCSD1D Class for CV Plotting ---
class MyKCSD1D(ElephantKCSD1D):
    """
    Custom KCSD1D class that overrides cross_validate to store the full CV error matrix.
    """
    def __init__(self, ele_pos, pots, **kwargs):
        super().__init__(ele_pos, pots, **kwargs)
        self.cv_error_matrix = None  # Initialize the new attribute

    def cross_validate(self, lambdas=None, Rs=None):
        if lambdas is None:
            lambdas = np.logspace(-2, -25, 25, base=10.)
            lambdas = np.hstack((lambdas, np.array((0.0))))
        elif hasattr(lambdas, 'size') and lambdas.size == 1:
            lambdas = lambdas.flatten()
        elif not hasattr(lambdas, 'size'):
             lambdas = np.array([lambdas])

        if Rs is None:
            Rs = np.array((self.R)).flatten() # self.R is in meters
        elif hasattr(Rs, 'size') and Rs.size == 1:
            Rs = Rs.flatten()
        elif not hasattr(Rs, 'size'):
            Rs = np.array([Rs])

        errs = np.zeros((Rs.size, lambdas.size))
        index_generator = []
        for ii in range(self.n_ele):
            idx_test = [ii]
            idx_train = list(range(self.n_ele))
            idx_train.remove(ii)
            index_generator.append((idx_train, idx_test))

        original_R = self.R # Store original R before iterating
        for R_idx, R_val_meters in enumerate(Rs):
            self.update_R(R_val_meters)
            # print(f'MyKCSD1D CV: Validating R={R_val_meters*1e6:.2f}um') # Optional debug
            for lambd_idx, lambd_val in enumerate(lambdas):
                errs[R_idx, lambd_idx] = self.compute_cverror(lambd_val, index_generator)
        
        self.cv_error_matrix = errs
        
        min_error_indices = np.where(self.cv_error_matrix == np.min(self.cv_error_matrix))
        cv_R_meters = Rs[min_error_indices[0][0]]
        cv_lambda = lambdas[min_error_indices[1][0]]
        
        self.cv_error = np.min(self.cv_error_matrix)
        
        self.update_R(cv_R_meters) # Set KCSD object to optimal R
        self.update_lambda(cv_lambda) # Set KCSD object to optimal lambda
        
        return cv_R_meters, cv_lambda

# --- Voltage Scaling Function ---
def get_voltage_scaling_factor(meta):
    try:
        v_max = float(meta['imAiRangeMax'])
        i_max_adc_val = float(meta['imMaxInt'])
        probe_type = int(meta.get('imDatPrb_type', 0))
        lfp_gain = None
        if probe_type in [21, 24, 2013]: lfp_gain = 80.0
        else:
            general_lfp_gain_key_str = "~imChanLFGain"
            if general_lfp_gain_key_str in meta: lfp_gain = float(meta[general_lfp_gain_key_str])
            else:
                first_lfp_gain_key_found = None
                sorted_keys = sorted([key for key in meta.keys() if key.startswith('imChanLFGain') and key.endswith('lfGain')])
                if sorted_keys: first_lfp_gain_key_found = sorted_keys[0]
                if first_lfp_gain_key_found: lfp_gain = float(meta[first_lfp_gain_key_found])
                else:
                    lfp_gain = 250.0
                    print(f"  Probe type {probe_type}. No specific LFP gain key found. Defaulting to {lfp_gain}.")
        if lfp_gain is None: raise ValueError("LFP gain undetermined.")
        if i_max_adc_val == 0 or lfp_gain == 0: raise ValueError("i_max_adc_val or LFP gain is zero.")
        scaling_factor_uv = (v_max / i_max_adc_val) * (1.0 / lfp_gain) * 1e6
        print(f"  Calculated uV scaling factor: {scaling_factor_uv:.6f} (v_max={v_max}, i_max_adc={i_max_adc_val}, lfp_gain={lfp_gain})")
        return scaling_factor_uv
    except Exception as e:
        print(f"Error calculating voltage scaling factor: {e}"); traceback.print_exc(); return None

# --- LFP Data Loading Function ---
def load_lfp_data_sglx_memmap(bin_file_path_str, meta_file_path_str):
    bin_file_path = Path(bin_file_path_str); meta_file_path = Path(meta_file_path_str)
    print(f"Setting up LFP data access from: {bin_file_path}\nUsing metadata from: {meta_file_path}")
    try:
        meta = readMeta(meta_file_path)
        fs_orig = float(meta['imSampRate']); n_channels_total = int(meta['nSavedChans'])
        print(f"  Meta: {n_channels_total} channels, FS: {fs_orig:.2f} Hz.")
        uv_scale_factor = get_voltage_scaling_factor(meta)
        if uv_scale_factor is None: print("  Warning: Voltage scaling factor undetermined. LFP in ADC units.")
        file_size = bin_file_path.stat().st_size; item_size = np.dtype('int16').itemsize
        if n_channels_total == 0 or item_size == 0: raise ValueError("Meta: Invalid nSavedChans or itemsize.")
        num_samples_in_file = file_size // (n_channels_total * item_size)
        if file_size % (n_channels_total * item_size) != 0:
            print(f"  Warning: File size not integer multiple of (n_channels_total * item_size). Using {num_samples_in_file} full samples.")
        if num_samples_in_file <= 0: raise ValueError("Zero or negative samples calculated.")
        print(f"  Total samples in LFP file: {num_samples_in_file}. Memmap shape: ({num_samples_in_file}, {n_channels_total})")
        lfp_data_memmap = np.memmap(bin_file_path, dtype='int16', mode='r', shape=(num_samples_in_file, n_channels_total))
        print(f"  Successfully memory-mapped LFP data.")
        return lfp_data_memmap, fs_orig, n_channels_total, uv_scale_factor, num_samples_in_file
    except Exception as e:
        print(f"Error in load_lfp_data_sglx_memmap: {e}"); traceback.print_exc(); raise

# --- Channel Info Loading ---
def load_channel_info_kcsd(csv_filepath_str):
    csv_filepath = Path(csv_filepath_str); print(f"Loading channel info from {csv_filepath}")
    try:
        channel_df = pd.read_csv(csv_filepath)
        required_cols = ['global_channel_index', 'shank_index', 'ycoord_on_shank_um']
        if not all(col in channel_df.columns for col in required_cols):
            raise ValueError(f"Channel info CSV must contain columns: {required_cols}")
        for col in ['global_channel_index', 'shank_index']:
            channel_df[col] = pd.to_numeric(channel_df[col], errors='coerce').astype('Int64')
        channel_df['ycoord_on_shank_um'] = pd.to_numeric(channel_df['ycoord_on_shank_um'], errors='coerce').astype(float)
        channel_df.dropna(subset=required_cols, inplace=True)
        for col in ['global_channel_index', 'shank_index']: channel_df[col] = channel_df[col].astype(int)
        print(f"Loaded and validated channel info for {len(channel_df)} channels.")
        return channel_df
    except Exception as e:
        print(f"Error loading channel info: {e}"); traceback.print_exc(); sys.exit(1)

# --- Epoch Definitions Loading ---
def load_epoch_definitions_from_times(timestamps_npy_path_str, fs_lfp, total_lfp_samples_in_file):
    timestamps_npy_path = Path(timestamps_npy_path_str)
    print(f"Loading epoch definitions (time-based) from: {timestamps_npy_path}")
    if not timestamps_npy_path.exists():
        print(f"Error: Epoch definitions NPY file not found: {timestamps_npy_path}"); return None
    try:
        loaded_npy = np.load(timestamps_npy_path, allow_pickle=True); data_to_process = None; epoch_info_list = None
        if isinstance(loaded_npy, dict): data_to_process = loaded_npy; print("  Loaded NPY as a direct dictionary.")
        elif isinstance(loaded_npy, np.ndarray) and loaded_npy.ndim == 0 and hasattr(loaded_npy, 'item') and isinstance(loaded_npy.item(), dict):
            data_to_process = loaded_npy.item(); print("  Loaded NPY as a 0-dim array; extracted dictionary item.")
        if data_to_process is not None and 'EpochFrameData' in data_to_process:
            epoch_info_list = data_to_process['EpochFrameData']
            if not isinstance(epoch_info_list, list): print(f"  Error: 'EpochFrameData' not a list."); return None
            print(f"  Found 'EpochFrameData' with {len(epoch_info_list)} entries.")
        elif isinstance(loaded_npy, np.ndarray) and loaded_npy.dtype.fields and all(f in loaded_npy.dtype.fields for f in ['start_time_sec', 'end_time_sec', 'epoch_index']):
            epoch_info_list = list(loaded_npy); print(f"  Interpreting loaded NPY as structured array of {len(epoch_info_list)} epochs.")
        if epoch_info_list is None:
            print(f"Error: Epoch definitions NPY format not recognized. Loaded type: {type(loaded_npy)}"); return None
        epochs = []
        for item_idx, epoch_info_item in enumerate(epoch_info_list):
            start_sec_val, end_sec_val, epoch_idx_val = None, None, None
            if isinstance(epoch_info_item, dict):
                start_sec_val, end_sec_val, epoch_idx_val = epoch_info_item.get('start_time_sec'), epoch_info_item.get('end_time_sec'), epoch_info_item.get('epoch_index')
            elif hasattr(epoch_info_item, 'dtype') and epoch_info_item.dtype.fields:
                try: start_sec_val, end_sec_val, epoch_idx_val = epoch_info_item['start_time_sec'], epoch_info_item['end_time_sec'], epoch_info_item['epoch_index']
                except (IndexError, KeyError) as e_field: print(f"  Warning: Missing field in item {item_idx}: {e_field}"); continue
            else: print(f"  Warning: Epoch item {item_idx} format unrecognized: {type(epoch_info_item)}"); continue
            if start_sec_val is not None and end_sec_val is not None and epoch_idx_val is not None:
                start_sec, end_sec = float(start_sec_val), float(end_sec_val)
                epoch_abs_start_sample = int(round(start_sec * fs_lfp)); epoch_abs_end_sample_inclusive = int(round(end_sec * fs_lfp)) - 1
                if epoch_abs_end_sample_inclusive < epoch_abs_start_sample: epoch_abs_end_sample_inclusive = epoch_abs_start_sample
                epoch_abs_start_sample_capped = max(0, epoch_abs_start_sample)
                epoch_abs_end_sample_inclusive_capped = min(epoch_abs_end_sample_inclusive, total_lfp_samples_in_file - 1)
                if epoch_abs_start_sample_capped > epoch_abs_end_sample_inclusive_capped or epoch_abs_start_sample_capped >= total_lfp_samples_in_file:
                    print(f"  Warning: Epoch {epoch_idx_val} invalid sample range. Skipping."); continue
                epochs.append({'epoch_index': int(epoch_idx_val), 'abs_start_sample': epoch_abs_start_sample_capped,
                    'abs_end_sample_inclusive': epoch_abs_end_sample_inclusive_capped, 'duration_sec_requested': end_sec - start_sec,
                    'duration_sec_actual': (epoch_abs_end_sample_inclusive_capped - epoch_abs_start_sample_capped + 1) / fs_lfp if fs_lfp > 0 else 0})
            else: print(f"  Warning: Epoch info item {item_idx} incomplete.")
        if not epochs: print("No valid epoch definitions constructed."); return None
        epochs.sort(key=lambda e: e['abs_start_sample'])
        print(f"Loaded and validated {len(epochs)} epoch definitions.")
        for ep_idx_print, ep in enumerate(epochs): print(f"  Using Epoch {ep['epoch_index']} (Sorted Idx {ep_idx_print}): Samples {ep['abs_start_sample']}-{ep['abs_end_sample_inclusive']} (Duration: {ep['duration_sec_actual']:.3f}s)")
        return epochs
    except Exception as e: print(f"Error loading epoch definitions NPY: {e}"); traceback.print_exc(); return None

# --- SWR Timestamp Loading Function (with enhanced debugging) ---
def load_swr_timestamps(swr_timestamps_npy_path_str, target_state_code, target_epoch_idx, target_region_name):
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
        epoch_specific_data = data_for_specific_state.get(target_epoch_idx, {})
        if not epoch_specific_data: print(f"    DEBUG: No data found for Epoch {target_epoch_idx} in loaded SWR NPY (state {target_state_code}).")
        else: print(f"    DEBUG: For Epoch {target_epoch_idx}, found data. Keys (expected region names): {list(epoch_specific_data.keys())}")
        swr_peaks_for_region = epoch_specific_data.get(target_region_name, np.array([], dtype=int))
        if not isinstance(swr_peaks_for_region, np.ndarray):
            print(f"  Warning: Timestamps for State {target_state_code}, Epoch {target_epoch_idx}, Region {target_region_name} not NumPy array. Type: {type(swr_peaks_for_region)}")
            swr_peaks_for_region = np.array(swr_peaks_for_region, dtype=int)
        print(f"  Loaded {len(swr_peaks_for_region)} SWR peak timestamps for State {target_state_code}, Epoch {target_epoch_idx}, Region {target_region_name}.")
        return swr_peaks_for_region.astype(int)
    except Exception as e:
        print(f"  Error loading SWR timestamps: {e}"); traceback.print_exc(); return np.array([], dtype=int)

# --- LFP Preprocessing for a SUB-CHUNK ---
def preprocess_lfp_sub_chunk(lfp_sub_chunk_scaled_f32, fs_sub_chunk_in, 
                             lowcut, highcut, numtaps, consistent_downsampling_factor, 
                             target_overall_fs):
    if lfp_sub_chunk_scaled_f32.ndim == 1: 
        lfp_sub_chunk_scaled_f32 = lfp_sub_chunk_scaled_f32[:, np.newaxis]
    current_fs, lfp_to_process_f32 = fs_sub_chunk_in, lfp_sub_chunk_scaled_f32
    if consistent_downsampling_factor > 1:
        min_samples_for_decimate = consistent_downsampling_factor * 30
        if lfp_to_process_f32.shape[0] > min_samples_for_decimate:
            try:
                lfp_sub_chunk_downsampled_f64 = signal.decimate(lfp_to_process_f32.astype(np.float64), 
                                                                consistent_downsampling_factor, axis=0, ftype='fir', zero_phase=True)
                lfp_to_process_f32, current_fs = lfp_sub_chunk_downsampled_f64.astype(np.float32), current_fs / consistent_downsampling_factor
            except Exception as e_decimate: print(f"    Warning: Decimation error (factor {consistent_downsampling_factor}): {e_decimate}.")
        else: print(f"    Info: Sub-chunk too short ({lfp_to_process_f32.shape[0]} samples) for decimation by {consistent_downsampling_factor}.")
    nyq = current_fs / 2.0; 
    actual_highcut = min(highcut, nyq * 0.98); 
    actual_lowcut = max(lowcut, 0.001)
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
    # This function is primarily for the averaged CSD if you choose to save it.
    # Individual SWR CSDs are saved as a single .npy array per shank/epoch.
    # Metadata for individual SWR CSDs might need a different approach if detailed per-SWR KCSD params are stored.
    # For now, this function can be used for the CV-determining average.
    try:
        # If csd_data_matrix is 3D (num_swr, sites, time), this save_csd_data_and_meta might need adjustment
        # Assuming it's called with 2D CSD matrix for now (e.g. for the average CSD used for CV)
        if csd_data_matrix.ndim == 2 : # (time, sites) for .bin file
            csd_data_matrix.astype(np.float32).tofile(csd_bin_filename)
            print(f"    Saved {analysis_type} CSD data to: {csd_bin_filename} (Shape: {csd_data_matrix.shape}, Dtype: {csd_data_matrix.dtype})")
        elif csd_data_matrix.ndim == 3: # (n_swr, sites, time) for .npy file (already saved as .npy)
             print(f"    Skipping .bin save for 3D array {analysis_type} CSD data (already saved as .npy).")
        else:
            print(f"    Warning: CSD data matrix has unexpected shape {csd_data_matrix.shape} for saving. Skipping .bin save.")

        meta_content = [f"~kCSDAnalysisVersion=1.3_IndivSWR_CVPlotFix", f"~creationTime={datetime.datetime.now().isoformat()}",
                        f"analysisType={analysis_type}", f"originalLFPFile={str(original_lfp_file)}",
                        f"epochIdx={epoch_info.get('epoch_index', 'N/A')}", f"shankId={shank_id}",
                        f"csdUnits={csd_units_str}", f"csdSamplingRateHz={final_fs_csd:.4f}"]
        if csd_data_matrix.ndim == 2: # Only add these if we saved a .bin
            meta_content.extend([f"csdBinFile={csd_bin_filename.name}", f"csdDataType=float32",
                                 f"csdDataShapeSamples={csd_data_matrix.shape[0]}", f"csdDataShapeSites={csd_data_matrix.shape[1]}"])

        meta_content.append(f"csdSitesYCoordUm={','.join([f'{pos:.2f}' for pos in csd_positions_um])}")
        
        # Duration and time zero depend on context
        if analysis_type == "swr_average_for_cv": # Distinguish the average used for CV
            meta_content.append(f"csdTimeZeroSec={(SWR_TIME_WINDOW_MS[0] / 1000.0):.4f}")
            csd_duration_sec = (SWR_TIME_WINDOW_MS[1] - SWR_TIME_WINDOW_MS[0]) / 1000.0
            if swr_info:
                meta_content.extend([f"swrSourceRegion={swr_info.get('source_region', 'N/A')}",
                                     f"swrNumEventsAveraged={swr_info.get('num_events', 0)}",
                                     f"swrTimeWindowMsStart={SWR_TIME_WINDOW_MS[0]}",
                                     f"swrTimeWindowMsEnd={SWR_TIME_WINDOW_MS[1]}"])
        elif analysis_type == "individual_swr_csds_array": # For the .npy file containing all CSDs
             meta_content.append(f"csdTimeZeroSec_PerSWR={(SWR_TIME_WINDOW_MS[0] / 1000.0):.4f}") # Time zero relative to each SWR peak
             csd_duration_sec = (SWR_TIME_WINDOW_MS[1] - SWR_TIME_WINDOW_MS[0]) / 1000.0 # Duration of each CSD window
             if swr_info: # swr_info would refer to the collection here
                meta_content.extend([f"swrSourceRegion={swr_info.get('source_region', 'N/A')}",
                                     f"swrNumEventsInArray={swr_info.get('num_events_in_array', 0)}"])

        else: # epoch
            meta_content.append(f"csdTimeZeroSec=0.0")
            csd_duration_sec = csd_data_matrix.shape[0] / final_fs_csd if final_fs_csd > 0 and csd_data_matrix.ndim == 2 else 0
        
        meta_content.append(f"csdDurationSec_PerWindow={csd_duration_sec:.4f}") # Clarify this is per CSD window

        # KCSD params are from the CV run (on average LFP)
        meta_content.extend([f"kcsdOptimalLambdaUsed={kcsd_params['lambda_optimal']:.4e}", 
                             f"kcsdOptimalR_um_Used={kcsd_params['R_optimal_um']:.2f}",
                             f"kcsdSigmaSm={kcsd_params['sigma_Sm']:.2f}", f"kcsdHmeters={kcsd_params['h_m']:.2f}",
                             f"kcsdNSrcInit={kcsd_params['n_src_init']}", f"kcsdXminMeters={kcsd_params['xmin_m']:.4e}",
                             f"kcsdXmaxMeters={kcsd_params['xmax_m']:.4e}", f"kcsdGdxMeters={kcsd_params['gdx_m']:.4e}"])
        if epoch_info:
            meta_content.extend([f"lfpEpochStartSampleOrig={epoch_info.get('abs_start_sample', 'N/A')}",
                                 f"lfpEpochEndSampleInclOrig={epoch_info.get('abs_end_sample_inclusive', 'N/A')}",
                                 f"lfpEpochDurationSecActual={epoch_info.get('duration_sec_actual', 0):.4f}"])
        with open(csd_meta_filename, 'w') as f_meta:
            for line in meta_content: f_meta.write(line + '\n')
        print(f"    Saved CSD metadata to: {csd_meta_filename}")
    except Exception as e_save:
        print(f"    Error saving {analysis_type} CSD data/metadata for shank {shank_id}, epoch {epoch_info.get('epoch_index', 'N/A')}: {e_save}"); traceback.print_exc()


# --- Main CSD Analysis Script ---
def main_kcsd_analysis(lfp_bin_file_path_str, lfp_meta_file_path_str, channel_info_csv_path_str,
                         epoch_definitions_npy_path_str, swr_timestamps_npy_path_str_in):
    lfp_bin_file_path = Path(lfp_bin_file_path_str)
    base_filename = lfp_bin_file_path.stem.replace('.lf', '')

    lfp_data_memmap_obj = None
    try:
        lfp_data_memmap_obj, fs_orig, n_channels_total_in_file, uv_scale_factor_loaded, num_samples_in_lfp_file = \
            load_lfp_data_sglx_memmap(lfp_bin_file_path_str, lfp_meta_file_path_str)
    except Exception as e: print(f"CRITICAL: Failed to load LFP data: {e}"); return

    epoch_definitions = load_epoch_definitions_from_times(epoch_definitions_npy_path_str, fs_orig, num_samples_in_lfp_file)
    if not epoch_definitions:
        print("No valid epoch definitions. Exiting."); 
        if lfp_data_memmap_obj is not None and hasattr(lfp_data_memmap_obj, '_mmap') and lfp_data_memmap_obj._mmap is not None:
            try: lfp_data_memmap_obj._mmap.close(); del lfp_data_memmap_obj
            except Exception: pass
        gc.collect(); return

    overall_downsampling_factor = 1
    if fs_orig > TARGET_FS_CSD and not np.isclose(fs_orig, TARGET_FS_CSD):
        overall_downsampling_factor = int(round(fs_orig / TARGET_FS_CSD))
        if overall_downsampling_factor <= 0: overall_downsampling_factor = 1
    final_effective_fs_for_csd_nominal = fs_orig / overall_downsampling_factor
    print(f"Original LFP FS: {fs_orig:.2f} Hz. Target CSD FS: {TARGET_FS_CSD:.2f} Hz.")
    print(f"Calculated overall downsampling_factor: {overall_downsampling_factor}. Nominal final effective CSD FS: {final_effective_fs_for_csd_nominal:.2f} Hz")

    try:
        probe_channel_df = load_channel_info_kcsd(channel_info_csv_path_str)
        unique_shanks = sorted(probe_channel_df['shank_index'].unique())
        if not unique_shanks: print("No shanks found in channel info. Exiting."); return

        for epoch_info in epoch_definitions:
            epoch_idx = epoch_info['epoch_index']
            epoch_abs_start_sample, epoch_abs_end_sample_inclusive = epoch_info['abs_start_sample'], epoch_info['abs_end_sample_inclusive']
            print(f"\n>>> Processing Epoch {epoch_idx} (LFP Samples: {epoch_abs_start_sample}-{epoch_abs_end_sample_inclusive}, Duration: {epoch_info['duration_sec_actual']:.3f}s) <<<")

            is_target_state_epoch_for_swr = SWR_ANALYSIS_ENABLED

            if SWR_ANALYSIS_ENABLED and is_target_state_epoch_for_swr and swr_timestamps_npy_path_str_in is not None:
                print(f"  Attempting SWR CSD analysis for Epoch {epoch_idx} (Target State: {SWR_TARGET_STATE_CODE}, Region: {SWR_TARGET_REGION})")
                swr_peak_timestamps_abs = load_swr_timestamps(swr_timestamps_npy_path_str_in,
                                                              SWR_TARGET_STATE_CODE, epoch_idx, SWR_TARGET_REGION)
                swr_peak_timestamps_in_epoch = swr_peak_timestamps_abs[
                    (swr_peak_timestamps_abs >= epoch_abs_start_sample) &
                    (swr_peak_timestamps_abs <= epoch_abs_end_sample_inclusive)]
                print(f"    Found {len(swr_peak_timestamps_in_epoch)} SWRs from '{SWR_TARGET_REGION}' in Epoch {epoch_idx} for State {SWR_TARGET_STATE_CODE}.")

                if len(swr_peak_timestamps_in_epoch) > 0:
                    swr_window_samples_orig_fs_start_offset = int(SWR_TIME_WINDOW_MS[0] / 1000.0 * fs_orig)
                    swr_window_samples_orig_fs_end_offset = int(SWR_TIME_WINDOW_MS[1] / 1000.0 * fs_orig)
                    swr_window_length_orig_fs = swr_window_samples_orig_fs_end_offset - swr_window_samples_orig_fs_start_offset

                    for i_shank, shank_id_val in enumerate(unique_shanks):
                        print(f"\n    SWR CSD for Shank {shank_id_val} (Epoch {epoch_idx})")
                        shank_channel_info = probe_channel_df[probe_channel_df['shank_index'] == shank_id_val].sort_values(by='ycoord_on_shank_um', ascending=True)
                        if shank_channel_info.empty or len(shank_channel_info) < 2:
                            print(f"    Shank {shank_id_val} - Not enough channels. Skipping SWR CSD."); continue
                        shank_global_indices = shank_channel_info['global_channel_index'].values
                        if np.any(shank_global_indices >= n_channels_total_in_file) or np.any(shank_global_indices < 0):
                            print(f"    Shank {shank_id_val} - Channel index error. Skipping SWR CSD."); continue
                        electrode_coords_um_shank = shank_channel_info['ycoord_on_shank_um'].values
                        
                        min_abs_sample_needed_for_swrs = np.min(swr_peak_timestamps_in_epoch) + swr_window_samples_orig_fs_start_offset
                        max_abs_sample_needed_for_swrs = np.max(swr_peak_timestamps_in_epoch) + swr_window_samples_orig_fs_end_offset
                        lfp_extraction_start_abs_origfs = max(0, min_abs_sample_needed_for_swrs)
                        lfp_extraction_end_abs_origfs = min(num_samples_in_lfp_file, max_abs_sample_needed_for_swrs)

                        if lfp_extraction_start_abs_origfs >= lfp_extraction_end_abs_origfs:
                            print(f"    No valid LFP range for SWRs in Epoch {epoch_idx}, Shank {shank_id_val}. Skipping."); continue
                        try:
                            lfp_shank_swr_relevant_portion_raw = lfp_data_memmap_obj[lfp_extraction_start_abs_origfs:lfp_extraction_end_abs_origfs, shank_global_indices]
                        except Exception as e:
                            print(f"    Error extracting LFP for SWRs (Shank {shank_id_val}): {e}"); continue
                        
                        lfp_shank_swr_relevant_portion_scaled = lfp_shank_swr_relevant_portion_raw.astype(np.float32) * (uv_scale_factor_loaded if uv_scale_factor_loaded is not None else 1.0)
                        processed_lfp_shank_swr_relevant, fs_processed_swr = preprocess_lfp_sub_chunk(
                            lfp_shank_swr_relevant_portion_scaled, fs_orig, LFP_BAND_LOWCUT_CSD, LFP_BAND_HIGHCUT_CSD, 
                            NUMTAPS_CSD_FILTER, overall_downsampling_factor, final_effective_fs_for_csd_nominal)
                        
                        swr_window_start_offset_proc_fs = int(SWR_TIME_WINDOW_MS[0] / 1000.0 * fs_processed_swr)
                        swr_window_len_samples_proc_fs = int(swr_window_length_orig_fs / (fs_orig / fs_processed_swr))

                        swr_lfp_segments_shank = []; valid_swr_peak_timestamps_for_shank = []
                        for swr_peak_abs_orig_fs in swr_peak_timestamps_in_epoch:
                            swr_peak_rel_to_extraction_start_orig_fs = swr_peak_abs_orig_fs - lfp_extraction_start_abs_origfs
                            swr_peak_rel_to_extraction_start_proc_fs = int(round(swr_peak_rel_to_extraction_start_orig_fs / (fs_orig / fs_processed_swr)))
                            win_start_in_processed_chunk = swr_peak_rel_to_extraction_start_proc_fs + swr_window_start_offset_proc_fs
                            win_end_in_processed_chunk = win_start_in_processed_chunk + swr_window_len_samples_proc_fs
                            if win_start_in_processed_chunk >= 0 and win_end_in_processed_chunk <= processed_lfp_shank_swr_relevant.shape[0]:
                                segment = processed_lfp_shank_swr_relevant[win_start_in_processed_chunk:win_end_in_processed_chunk, :]
                                if segment.shape[0] == swr_window_len_samples_proc_fs:
                                     swr_lfp_segments_shank.append(segment); valid_swr_peak_timestamps_for_shank.append(swr_peak_abs_orig_fs)
                                else: print(f"      SWR window for peak {swr_peak_abs_orig_fs}: segment length mismatch. Skipping.")
                            else: print(f"      SWR window for peak {swr_peak_abs_orig_fs} out of bounds for processed LFP chunk. Skipping.")
                        if not swr_lfp_segments_shank:
                            print(f"    No valid LFP segments extracted for SWRs on Shank {shank_id_val}. Skipping."); continue
                        
                        avg_swr_lfp_shank = np.mean(np.stack(swr_lfp_segments_shank), axis=0)
                        print(f"    Averaged LFP for {len(swr_lfp_segments_shank)} SWRs on Shank {shank_id_val} to determine CV params. Avg LFP shape: {avg_swr_lfp_shank.shape}")

                        coords_quant_um_shank = electrode_coords_um_shank * pq.um
                        ele_pos_for_kcsd_meters = coords_quant_um_shank.rescale(pq.m).magnitude.reshape(-1, 1)
                        lfp_units = pq.uV if uv_scale_factor_loaded is not None else pq.dimensionless
                        avg_lfp_for_kcsd_Volts_cv = (avg_swr_lfp_shank.T * lfp_units).rescale(pq.V).magnitude

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
                        
                        optimal_lambda_for_shank = None; optimal_R_um_for_shank = None
                        output_prefix_swr_cv = OUTPUT_DIR / f"{base_filename}_swr_avg_CV"

                        kcsd_params_for_meta_cv = { # Params for the object used for CV
                            'sigma_Sm': CSD_SIGMA_CONDUCTIVITY, 'n_src_init': num_csd_est_pts_config,
                            'xmin_m': xmin_kcsd_meters, 'xmax_m': xmax_kcsd_meters, 'gdx_m': gdx_kcsd_meters,
                            'R_init_m': initial_R_meter_val, 'ext_x_m': ext_x_kcsd_meters, 'h_m': h_kcsd_meters,
                            'lambda_optimal': None, 'R_optimal_um': None # These will be filled
                        }

                        try:
                            kcsd_cv_obj = MyKCSD1D(ele_pos=ele_pos_for_kcsd_meters, pots=avg_lfp_for_kcsd_Volts_cv, sigma=CSD_SIGMA_CONDUCTIVITY,
                                                 n_src_init=num_csd_est_pts_config, xmin=xmin_kcsd_meters, xmax=xmax_kcsd_meters,
                                                 gdx=gdx_kcsd_meters, ext_x=ext_x_kcsd_meters, R_init=initial_R_meter_val, lambd=0.0, h=h_kcsd_meters)
                            Rs_cv_meters = (KCSD_RS_CV_UM * pq.um).rescale(pq.m).magnitude
                            print(f"      Performing cross-validation ON SWR-AVERAGED LFP (Ep{epoch_idx},Shk{shank_id_val})...")
                            kcsd_cv_obj.cross_validate(lambdas=KCSD_LAMBDAS_CV, Rs=Rs_cv_meters)
                            optimal_lambda_for_shank = kcsd_cv_obj.lambd; optimal_R_um_for_shank = kcsd_cv_obj.R * 1e6
                            kcsd_params_for_meta_cv['lambda_optimal'] = optimal_lambda_for_shank
                            kcsd_params_for_meta_cv['R_optimal_um'] = optimal_R_um_for_shank
                            print(f"        CV Optimal Lambda (from avg): {optimal_lambda_for_shank:.2e}, Optimal R (from avg): {optimal_R_um_for_shank:.2f} um")
                            if hasattr(kcsd_cv_obj, 'cv_error_matrix') and kcsd_cv_obj.cv_error_matrix is not None:
                                fig_cv, ax_cv = plt.subplots(figsize=(8,6)); cv_err_to_plot = np.log10(kcsd_cv_obj.cv_error_matrix)
                                if np.isneginf(cv_err_to_plot).any():
                                    finite_min = np.min(cv_err_to_plot[np.isfinite(cv_err_to_plot)]) if np.isfinite(cv_err_to_plot).any() else -10
                                    cv_err_to_plot[np.isneginf(cv_err_to_plot)] = finite_min -1
                                vmin_cv = np.nanmin(cv_err_to_plot) if np.isfinite(cv_err_to_plot).any() else -10
                                vmax_percentile = np.nanpercentile(cv_err_to_plot[np.isfinite(cv_err_to_plot)], 99) if np.isfinite(cv_err_to_plot).any() else vmin_cv
                                vmax_cv = max(vmax_percentile, vmin_cv + 1e-9)
                                if vmin_cv >= vmax_cv: vmax_cv = vmin_cv + (1e-9 if abs(vmin_cv) > 1e-10 else 1e-9)
                                norm_cv = Normalize(vmin=vmin_cv, vmax=vmax_cv); im_cv = ax_cv.imshow(cv_err_to_plot, aspect='auto', origin='lower', cmap='viridis', norm=norm_cv)
                                ax_cv.set_xticks(np.arange(len(KCSD_LAMBDAS_CV))); ax_cv.set_xticklabels([f"{l:.1e}" for l in KCSD_LAMBDAS_CV], rotation=45, ha="right")
                                ax_cv.set_yticks(np.arange(len(KCSD_RS_CV_UM))); ax_cv.set_yticklabels([f"{r_val:.1f}" for r_val in KCSD_RS_CV_UM])
                                best_lambda_idx_list = np.where(np.isclose(KCSD_LAMBDAS_CV, optimal_lambda_for_shank))[0]
                                best_R_idx_list = np.where(np.isclose(KCSD_RS_CV_UM, optimal_R_um_for_shank))[0]
                                if len(best_lambda_idx_list) > 0 and len(best_R_idx_list) > 0:
                                    ax_cv.scatter(best_lambda_idx_list[0], best_R_idx_list[0], marker='x', color='red', s=100, label=f'Optimal\nL={optimal_lambda_for_shank:.1e}\nR={optimal_R_um_for_shank:.1f}um')
                                    ax_cv.legend(fontsize='small', loc='best')
                                else: print("      Warning: Could not find exact match for optimal lambda/R in CV grid for plotting marker on CV plot.")
                                ax_cv.set_xlabel('Lambda'); ax_cv.set_ylabel('R (um)'); ax_cv.set_title(f'CV Error (from SWR-Avg LFP)\nEp {epoch_idx}, Shk {shank_id_val}, St {SWR_TARGET_STATE_CODE}, Reg {SWR_TARGET_REGION}')
                                plt.colorbar(im_cv, ax=ax_cv, label='log10(CV Error)'); fig_cv.tight_layout()
                                cv_plot_filename = output_prefix_swr_cv.parent / f"{output_prefix_swr_cv.name}_ep{epoch_idx}_shk{shank_id_val}_st{SWR_TARGET_STATE_CODE}_reg{SWR_TARGET_REGION}.png"
                                fig_cv.savefig(cv_plot_filename); print(f"      Saved CV plot (from avg LFP): {cv_plot_filename}"); plt.close(fig_cv)
                            else: print(f"      CV error matrix not available for plotting (Ep{epoch_idx},Shk{shank_id_val}).")
                            del kcsd_cv_obj
                        except Exception as e_cv:
                            print(f"    Error during CV on SWR-averaged LFP (Ep{epoch_idx},Shk{shank_id_val}): {e_cv}. Cannot proceed with individual SWR CSDs for this shank."); traceback.print_exc(); continue
                        if optimal_lambda_for_shank is None or optimal_R_um_for_shank is None:
                             print(f"    Failed to determine optimal CV params for Shank {shank_id_val} from average. Skipping individual SWR CSDs."); continue
                        optimal_R_meters_for_shank = optimal_R_um_for_shank / 1e6

                        individual_csd_list_for_shank = []
                        print(f"    Calculating CSD for {len(swr_lfp_segments_shank)} individual SWRs on Shank {shank_id_val} using fixed CV params...")
                        for swr_idx, individual_swr_lfp_uV in enumerate(swr_lfp_segments_shank):
                            individual_swr_lfp_volts = (individual_swr_lfp_uV.T * lfp_units).rescale(pq.V).magnitude
                            try:
                                # Use ElephantKCSD1D here as we are not calling cross_validate on it
                                kcsd_single_swr_obj = ElephantKCSD1D( # Or just KCSD1D if it's not ambiguous
                                    ele_pos=ele_pos_for_kcsd_meters, pots=individual_swr_lfp_volts, sigma=CSD_SIGMA_CONDUCTIVITY,
                                    n_src_init=num_csd_est_pts_config, xmin=xmin_kcsd_meters, xmax=xmax_kcsd_meters,
                                    gdx=gdx_kcsd_meters, ext_x=ext_x_kcsd_meters, R_init=optimal_R_meters_for_shank,
                                    lambd=optimal_lambda_for_shank, h=h_kcsd_meters)
                                
                                csd_profile_single_swr_A_m3 = kcsd_single_swr_obj.values() # (num_csd_pts, time_win)
                                csd_profile_single_swr_uA_mm3 = csd_profile_single_swr_A_m3 * 1e-3
                                individual_csd_list_for_shank.append(csd_profile_single_swr_uA_mm3)

                                if PLOT_INDIVIDUAL_SWR_CSDS:
                                    csd_to_plot = csd_profile_single_swr_uA_mm3 # This is (num_csd_pts, time_win)
                                    lfp_to_overlay = individual_swr_lfp_uV # (time_win, num_elec)
                                    current_swr_peak_ts = valid_swr_peak_timestamps_for_shank[swr_idx]

                                    fig_indiv_swr, ax_csd_indiv = plt.subplots(figsize=(10, 7))
                                    plot_time_axis_ms = (np.arange(csd_to_plot.shape[1]) / fs_processed_swr) * 1000 + SWR_TIME_WINDOW_MS[0]
                                    
                                    # --- CORRECTED UNIT CONVERSION ---
                                    estm_x_meters_indiv = kcsd_single_swr_obj.estm_x.reshape(-1) * pq.m
                                    csd_positions_plot_um_indiv = estm_x_meters_indiv.rescale(pq.um).magnitude.flatten()
                                    # --- END CORRECTION ---
                                    
                                    abs_csd_indiv_finite = np.abs(csd_to_plot[np.isfinite(csd_to_plot)])
                                    clim_val_indiv = np.percentile(abs_csd_indiv_finite, 99.0) if abs_csd_indiv_finite.size > 0 else 1.0
                                    if clim_val_indiv < 1e-9 : clim_val_indiv = 1.0
                                    
                                    # For pcolormesh with C(M,N), X is N coords, Y is M coords for shading='gouraud'
                                    # csd_to_plot is (num_csd_pts/depth, time_win)
                                    img_indiv = ax_csd_indiv.pcolormesh(
                                        plot_time_axis_ms,             # X (time) coordinates
                                        csd_positions_plot_um_indiv,   # Y (depth) coordinates
                                        csd_to_plot,                   # CSD data
                                        cmap='RdBu_r', shading='gouraud', 
                                        vmin=-clim_val_indiv, vmax=clim_val_indiv, rasterized=True)
                                    
                                    plt.colorbar(img_indiv, ax=ax_csd_indiv, label=f'kCSD (uA/mm^3)', shrink=0.8, aspect=15)
                                    ax_csd_indiv.set_xlabel('Time relative to SWR Peak (ms)')
                                    ax_csd_indiv.set_ylabel('CSD Depth (µm)')
                                    ax_csd_indiv.set_title(f'Indiv. SWR CSD - Ep{epoch_idx} Shk{shank_id_val} SWR# {swr_idx} (Peak @{current_swr_peak_ts})\nSt{SWR_TARGET_STATE_CODE} Reg{SWR_TARGET_REGION}')
                                    if len(csd_positions_plot_um_indiv) > 0: 
                                        ax_csd_indiv.set_ylim(max(csd_positions_plot_um_indiv), min(csd_positions_plot_um_indiv))

                                    ax_lfp_indiv = ax_csd_indiv.twinx()
                                    min_depth_lfp, max_depth_lfp = np.min(electrode_coords_um_shank), np.max(electrode_coords_um_shank)
                                    for i_elec in range(lfp_to_overlay.shape[1]):
                                        lfp_trace = lfp_to_overlay[:, i_elec]
                                        elec_depth = electrode_coords_um_shank[i_elec]
                                        ax_lfp_indiv.plot(plot_time_axis_ms, (lfp_trace * LFP_PLOT_OVERLAY_SCALE) + elec_depth, color='black', linewidth=0.7, alpha=0.6)
                                    ax_lfp_indiv.set_ylim(max_depth_lfp + 50, min_depth_lfp - 50)
                                    ax_lfp_indiv.set_ylabel('Electrode Depth (µm) & LFP (a.u.)', color='black')
                                    ax_lfp_indiv.tick_params(axis='y', labelcolor='black')
                                    fig_indiv_swr.tight_layout(rect=[0, 0, 0.9, 1])
                                    
                                    plot_prefix_indiv = OUTPUT_DIR / f"{base_filename}_individual_SWR_plots"
                                    plot_prefix_indiv.mkdir(parents=True, exist_ok=True)
                                    indiv_plot_filename = plot_prefix_indiv / f"indivSWR_CSDLFP_ep{epoch_idx}_shk{shank_id_val}_st{SWR_TARGET_STATE_CODE}_reg{SWR_TARGET_REGION}_swr{swr_idx}_ts{current_swr_peak_ts}.png"
                                    fig_indiv_swr.savefig(indiv_plot_filename, dpi=100)
                                    plt.close(fig_indiv_swr)
                                del kcsd_single_swr_obj
                            except Exception as e_indiv_kcsd:
                                print(f"      Error processing or plotting individual SWR #{swr_idx} on Shank {shank_id_val}: {e_indiv_kcsd}"); traceback.print_exc()
                        if individual_csd_list_for_shank:
                            all_ripple_csds_shank_epoch_array = np.stack(individual_csd_list_for_shank, axis=0)
                            print(f"    Stacked CSDs for {all_ripple_csds_shank_epoch_array.shape[0]} SWRs on Shank {shank_id_val}. Array shape: {all_ripple_csds_shank_epoch_array.shape}")
                            output_prefix_indiv_csd_data = OUTPUT_DIR / f"{base_filename}_individual_SWR_CSDs"; output_prefix_indiv_csd_data.mkdir(parents=True, exist_ok=True)
                            csd_array_filename = output_prefix_indiv_csd_data / f"all_SWR_CSDs_ep{epoch_idx}_shk{shank_id_val}_st{SWR_TARGET_STATE_CODE}_reg{SWR_TARGET_REGION}.npy"
                            np.save(csd_array_filename, all_ripple_csds_shank_epoch_array)
                            print(f"      Saved array of individual SWR CSDs to: {csd_array_filename}")
                            # Optionally save metadata for this array
                            indiv_swr_meta_info = {'source_region': SWR_TARGET_REGION, 'num_events_in_array': all_ripple_csds_shank_epoch_array.shape[0]}
                            meta_array_filename = output_prefix_indiv_csd_data / f"all_SWR_CSDs_ep{epoch_idx}_shk{shank_id_val}_st{SWR_TARGET_STATE_CODE}_reg{SWR_TARGET_REGION}.meta"
                            # Use kcsd_params_for_meta_cv as it holds the lambda/R used for these individual CSDs
                            # The estm_x from the last individual SWR object could be used for csd_positions_um.
                            # CSD units str can be taken from one of the individual calculations if needed for a meta file.
                            # This save_csd_data_and_meta might need adaptation if we want a single meta for the .npy array
                            # For now, a simple .npy save is implemented.
                            
                        del processed_lfp_shank_swr_relevant, avg_swr_lfp_shank, individual_csd_list_for_shank
                        if 'all_ripple_csds_shank_epoch_array' in locals(): del all_ripple_csds_shank_epoch_array
                        gc.collect()
                else: pass
            print(f"  Skipping full epoch CSD processing for brevity in this SWR-focused update. It can be re-enabled.") # This line was from before, should be outside SWR_ANALYSIS_ENABLED block if full epoch is truly an alternative
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
        suggested_epoch_def_name = lfp_path_obj.stem.replace(".lf", ".epochs.npy")
        epoch_def_npy_f_selected_gui = filedialog.askopenfilename(title="Select Epoch Definitions NPY File", initialdir=lfp_path_obj.parent, initialfile=suggested_epoch_def_name)
        if not epoch_def_npy_f_selected_gui: sys.exit("Epoch Definitions NPY file selection cancelled.")
        epoch_def_npy_f_selected = epoch_def_npy_f_selected_gui
        swr_ts_npy_f_selected_gui = None
        if SWR_ANALYSIS_ENABLED:
            state_name_for_file = "NREM" if SWR_TARGET_STATE_CODE == 1 else "Awake" if SWR_TARGET_STATE_CODE == 0 else f"State{SWR_TARGET_STATE_CODE}"
            suggested_swr_ts_name = f"{lfp_path_obj.name.split('.')[0]}_ripple_timestamps_{state_name_for_file}_by_epoch.npy"
            swr_ts_npy_f_selected_gui = filedialog.askopenfilename(
                title=f"Select SWR Timestamps NPY (e.g., for State {SWR_TARGET_STATE_CODE}, Region {SWR_TARGET_REGION})",
                initialdir=lfp_path_obj.parent, initialfile=suggested_swr_ts_name)
            if not swr_ts_npy_f_selected_gui: print("SWR Timestamps NPY file selection cancelled. Using default or previously set path if valid.")
            else: swr_ts_npy_f_selected = swr_ts_npy_f_selected_gui
        root.destroy()
    else:
        if not all(Path(f).exists() for f in [lfp_bin_f_selected, lfp_meta_f_selected, channel_csv_f_selected, epoch_def_npy_f_selected]):
            print("One or more default file paths (LFP, Meta, Channel CSV, Epoch Defs) do not exist."); sys.exit(1)
        if SWR_ANALYSIS_ENABLED and (not swr_ts_npy_f_selected or not Path(swr_ts_npy_f_selected).exists()):
            print(f"SWR Analysis ENABLED but SWR timestamps file ('{swr_ts_npy_f_selected}') not found."); sys.exit(1)

    print(f"\n--- Starting kCSD analysis (Individual SWR CSD Focus) ---")
    print(f"LFP File: {lfp_bin_f_selected}\nMeta File: {lfp_meta_f_selected}\nChannel CSV: {channel_csv_f_selected}\nEpoch Definitions NPY: {epoch_def_npy_f_selected}")
    if SWR_ANALYSIS_ENABLED: print(f"SWR Timestamps NPY: {swr_ts_npy_f_selected}\nSWR Analysis: ENABLED (State: {SWR_TARGET_STATE_CODE}, Region: {SWR_TARGET_REGION}, Window: {SWR_TIME_WINDOW_MS} ms)")
    else: print(f"SWR Analysis: DISABLED")
    print(f"Plot individual SWR CSDs: {PLOT_INDIVIDUAL_SWR_CSDS}")
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