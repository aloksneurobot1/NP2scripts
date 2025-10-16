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

from elephant.current_source_density import estimate_csd 

# --- Configuration Parameters ---
LFP_BIN_FILE_PATH_DEFAULT = "your_lfp_file.lf.bin"
LFP_META_FILE_PATH_DEFAULT = "your_lfp_file.lf.meta"
CHANNEL_INFO_CSV_PATH_DEFAULT = "your_channel_info.csv"
TIMESTAMPS_NPY_PATH_DEFAULT = "your_timestamps.nidq_timestamps.npy" 

OUTPUT_DIR = Path("./csd_kcsd_output_v6_final") 
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_FS_CSD = 1000.0  # Hz
LFP_BAND_LOWCUT_CSD = 1.0  # Hz
LFP_BAND_HIGHCUT_CSD = 300.0 # Hz
NUMTAPS_CSD_FILTER = 101 

CSD_SIGMA_CONDUCTIVITY = 0.3  # Siemens per meter (S/m) 
KCSD_LAMBDAS_CV = np.logspace(-7, -2, 9) 
KCSD_RS_CV_UM = np.logspace(np.log10(20), np.log10(500), 9) 

EPOCH_SUB_CHUNK_DURATION_SECONDS = 10 

# --- Voltage Scaling Function ---
def get_voltage_scaling_factor(meta):
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
                if sorted_keys: first_lfp_gain_key_found = sorted_keys[0]
                if first_lfp_gain_key_found: lfp_gain = float(meta[first_lfp_gain_key_found])
                else:
                    lfp_gain = 250.0 
                    print(f"  Probe type {probe_type}, No LFP gain key. Defaulting to {lfp_gain} (NP1.0). VERIFY.")
        if lfp_gain is None: raise ValueError("LFP gain not determined.")
        if i_max_adc_val == 0 or lfp_gain == 0: raise ValueError("i_max_adc_val or LFP gain is zero.")
        return (v_max / i_max_adc_val) * (1.0 / lfp_gain) * 1e6
    except Exception as e: print(f"Error calculating voltage scaling factor: {e}"); return None

# --- LFP Data Loading Function (Returns memmap) ---
def load_lfp_data_sglx_memmap(bin_file_path_str, meta_file_path_str):
    bin_file_path = Path(bin_file_path_str)
    meta_file_path = Path(meta_file_path_str)
    print(f"Setting up LFP data access from: {bin_file_path}")
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
    except Exception as e: print(f"Error in load_lfp_data_sglx_memmap: {e}"); traceback.print_exc(); raise

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
        for col in ['global_channel_index', 'shank_index']: channel_df[col] = channel_df[col].astype(int)
        print(f"Loaded and validated channel info for {len(channel_df)} channels.")
        return channel_df
    except Exception as e: print(f"Error loading channel info: {e}"); traceback.print_exc(); sys.exit(1)

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
                start_sec, end_sec = float(epoch_info_dict['start_time_sec']), float(epoch_info_dict['end_time_sec'])
                epoch_abs_start_sample, epoch_abs_end_sample_inclusive = int(round(start_sec * fs_lfp)), int(round(end_sec * fs_lfp))
                original_start_sample_req, original_end_sample_req_inclusive = epoch_abs_start_sample, epoch_abs_end_sample_inclusive
                epoch_abs_start_sample = max(0, epoch_abs_start_sample)
                epoch_abs_end_sample_inclusive = min(epoch_abs_end_sample_inclusive, total_lfp_samples_in_file - 1)
                if epoch_abs_start_sample > epoch_abs_end_sample_inclusive or epoch_abs_start_sample >= total_lfp_samples_in_file:
                    print(f"Warning: Epoch {epoch_info_dict['epoch_index']} invalid after capping. Skipping.")
                    continue
                epochs.append({'epoch_index': int(epoch_info_dict['epoch_index']), 
                               'abs_start_sample': epoch_abs_start_sample, 
                               'abs_end_sample_inclusive': epoch_abs_end_sample_inclusive, 
                               'duration_sec_actual': (epoch_abs_end_sample_inclusive - epoch_abs_start_sample + 1) / fs_lfp if fs_lfp > 0 else 0})
            else: print(f"Warning: Epoch info incomplete: {epoch_info_dict}")
        if not epochs: print("No valid epoch definitions."); return None
        epochs.sort(key=lambda e: e['abs_start_sample'])
        print(f"Loaded {len(epochs)} valid epoch definitions.")
        for ep_idx_print, ep in enumerate(epochs): print(f"  Using Epoch {ep['epoch_index']} (Sorted {ep_idx_print}): Samples {ep['abs_start_sample']}-{ep['abs_end_sample_inclusive']} (Dur: {ep['duration_sec_actual']:.2f}s)")
        return epochs
    except Exception as e: print(f"Error loading timestamps: {e}"); traceback.print_exc(); return None

# --- LFP Preprocessing for a SUB-CHUNK ---
def preprocess_lfp_sub_chunk(lfp_sub_chunk_f32, fs_in, lowcut, highcut, numtaps, downsampling_factor): 
    if lfp_sub_chunk_f32.ndim == 1: lfp_sub_chunk_f32 = lfp_sub_chunk_f32[:, np.newaxis]
    effective_fs = fs_in / downsampling_factor
    if downsampling_factor > 1:
        min_samples = downsampling_factor * 30 
        if lfp_sub_chunk_f32.shape[0] <= min_samples : 
            lfp_downsampled_f32 = lfp_sub_chunk_f32; effective_fs = fs_in 
        else:
            try:
                temp_f64 = signal.decimate(lfp_sub_chunk_f32, downsampling_factor, axis=0, ftype='fir', zero_phase=True)
                lfp_downsampled_f32 = temp_f64.astype(np.float32)
            except Exception: lfp_downsampled_f32 = lfp_sub_chunk_f32; effective_fs = fs_in
    else: lfp_downsampled_f32 = lfp_sub_chunk_f32 
    
    nyq = effective_fs / 2.0; actual_highcut = min(highcut, nyq * 0.99); actual_lowcut = max(lowcut, 0.01)
    current_numtaps = numtaps
    if current_numtaps >= lfp_downsampled_f32.shape[0]: current_numtaps = lfp_downsampled_f32.shape[0] - 1
    if current_numtaps % 2 == 0 and current_numtaps > 0: current_numtaps -=1
    
    if current_numtaps < 3 or actual_lowcut >= actual_highcut: lfp_filtered_f32 = lfp_downsampled_f32
    else:
        try:
            taps = signal.firwin(current_numtaps, [actual_lowcut, actual_highcut], fs=effective_fs, pass_zero='bandpass', window='hamming')
            temp_f64 = signal.filtfilt(taps, 1.0, lfp_downsampled_f32, axis=0)
            lfp_filtered_f32 = temp_f64.astype(np.float32)
        except Exception: lfp_filtered_f32 = lfp_downsampled_f32
    return lfp_filtered_f32, effective_fs

# --- Main CSD Analysis Script ---
def main_kcsd_analysis(lfp_bin_file, lfp_meta_file, channel_info_csv, timestamps_npy_file):
    base_filename = Path(lfp_bin_file).stem
    output_file_prefix = OUTPUT_DIR / base_filename
    lfp_data_memmap_obj = None 
    try:
        lfp_data_memmap_obj, fs_orig, n_channels_total_in_file, uv_scale_factor_loaded, num_samples_in_lfp_file = \
            load_lfp_data_sglx_memmap(lfp_bin_file, lfp_meta_file)
    except Exception as e: print(f"CRITICAL: Failed to load LFP: {e}"); return

    epoch_definitions = load_epoch_definitions_from_times(timestamps_npy_file, fs_orig, num_samples_in_lfp_file)
    if not epoch_definitions: 
        print("No valid epochs. Exiting."); # Ensure memmap closure in finally
        if lfp_data_memmap_obj is not None and hasattr(lfp_data_memmap_obj, '_mmap') and lfp_data_memmap_obj._mmap is not None:
            try: lfp_data_memmap_obj._mmap.close(); del lfp_data_memmap_obj
            except Exception: pass
        gc.collect()
        return
        
    overall_ds_factor = int(round(fs_orig / TARGET_FS_CSD)) if fs_orig > TARGET_FS_CSD else 1
    if overall_ds_factor <= 0: overall_ds_factor = 1
    final_csd_fs = fs_orig / overall_ds_factor
    print(f"Overall DS factor: {overall_ds_factor}, Final CSD FS: {final_csd_fs:.2f} Hz")

    try:
        probe_channel_df = load_channel_info_kcsd(channel_info_csv)
        unique_shanks = sorted(probe_channel_df['shank_index'].unique())
        if not unique_shanks: print("No shanks in channel info. Exiting."); return 

        for epoch in epoch_definitions:
            ep_idx, ep_start_samp, ep_end_samp_inc = epoch['epoch_index'], epoch['abs_start_sample'], epoch['abs_end_sample_inclusive']
            ep_end_slice = ep_end_samp_inc + 1
            print(f"\n>>> Processing Epoch {ep_idx} (Samples: {ep_start_samp}-{ep_end_samp_inc}, Dur: {epoch['duration_sec_actual']:.2f}s) <<<")
            
            fig_ep, axs_ep = plt.subplots(len(unique_shanks), 1, figsize=(12, 4*len(unique_shanks)), sharex=True, squeeze=False, constrained_layout=True)
            fig_ep.suptitle(f'kCSD1D - {base_filename} - Epoch {ep_idx}', fontsize=16)

            for i_sh, shk_id in enumerate(unique_shanks):
                ax_csd = axs_ep[i_sh, 0]
                print(f"\n  Shank {shk_id} (Ep {ep_idx})")
                shk_ch_info = probe_channel_df[probe_channel_df['shank_index'] == shk_id].sort_values(by='ycoord_on_shank_um', ascending=True)
                if len(shk_ch_info) < 2: print(f"    Shank {shk_id} has <2 channels. Skipping."); continue
                
                shk_globals = shk_ch_info['global_channel_index'].values
                ele_coords_um = shk_ch_info['ycoord_on_shank_um'].values

                sub_chunks_processed = []
                ep_dur_samps = ep_end_slice - ep_start_samp
                sub_chunk_samps_orig = int(fs_orig * EPOCH_SUB_CHUNK_DURATION_SECONDS)
                if sub_chunk_samps_orig <= 0: sub_chunk_samps_orig = ep_dur_samps

                for sub_start_local in range(0, ep_dur_samps, sub_chunk_samps_orig):
                    abs_sub_start = ep_start_samp + sub_start_local
                    abs_sub_end = min(abs_sub_start + sub_chunk_samps_orig, ep_end_slice)
                    if abs_sub_start >= abs_sub_end: continue
                    
                    try:
                        ch_data_sub_chunk = []
                        for glob_ch_idx in shk_globals:
                            raw_ch_data = lfp_data_memmap_obj[abs_sub_start:abs_sub_end, glob_ch_idx]
                            scaled_ch_data_f32 = raw_ch_data.astype(np.float32)
                            if uv_scale_factor_loaded: scaled_ch_data_f32 *= uv_scale_factor_loaded
                            
                            proc_ch_data_f32, _ = preprocess_lfp_sub_chunk(
                                scaled_ch_data_f32, fs_orig, LFP_BAND_LOWCUT_CSD, LFP_BAND_HIGHCUT_CSD, 
                                NUMTAPS_CSD_FILTER, overall_ds_factor, TARGET_FS_CSD) # Pass overall_ds_factor
                            ch_data_sub_chunk.append(proc_ch_data_f32.flatten())
                        
                        if ch_data_sub_chunk:
                            min_len = min(len(arr) for arr in ch_data_sub_chunk)
                            aligned = [arr[:min_len] for arr in ch_data_sub_chunk]
                            sub_chunks_processed.append(np.vstack(aligned).T.astype(np.float32))
                        del ch_data_sub_chunk, aligned; gc.collect()
                    except MemoryError: print(f"    MemoryError in sub-chunk for Shk{shk_id},Ep{ep_idx}. Skipping rest of shank."); sub_chunks_processed=[]; break
                    except Exception as e_sub: print(f"    Error in sub-chunk for Shk{shk_id},Ep{ep_idx}: {e_sub}"); continue
                
                if not sub_chunks_processed: print(f"    No sub-chunks for Shk{shk_id},Ep{ep_idx}. Skipping CSD."); continue
                
                print(f"    Concatenating {len(sub_chunks_processed)} sub-chunks for Shk{shk_id},Ep{ep_idx}...")
                lfp_full_processed_f32 = None
                try: lfp_full_processed_f32 = np.concatenate(sub_chunks_processed, axis=0).astype(np.float32)
                except MemoryError: print(f"    MemoryError concatenating for Shk{shk_id},Ep{ep_idx}. Skipping CSD.");
                finally: del sub_chunks_processed; gc.collect()
                if lfp_full_processed_f32 is None: continue

                print(f"    Final LFP for CSD (Shk{shk_id},Ep{ep_idx}): {lfp_full_processed_f32.shape}, FS: {final_csd_fs:.2f} Hz")
                
                coords_q = ele_coords_um.reshape(-1, 1) * pq.um
                analog_sig = None; kcsd_est = None; csd_neo = None; csd_vals = None # Initialize
                try:
                    units = 'uV' if uv_scale_factor_loaded else 'ADC_count'
                    analog_sig = neo.AnalogSignal(lfp_full_processed_f32.astype(np.float64), units=units, sampling_rate=final_csd_fs * pq.Hz)
                    del lfp_full_processed_f32; gc.collect()

                    n_est_pts = max(32, int(len(ele_coords_um) * 1.5))
                    xmin_q, xmax_q = ele_coords_um.min()*pq.um, ele_coords_um.max()*pq.um
                    gdx_q = ((xmax_q - xmin_q) / (n_est_pts - 1)) if n_est_pts > 1 else 10.0*pq.um
                    if gdx_q.magnitude <= 1e-9: gdx_q = 1.0*pq.um
                    
                    print(f"    KCSD Params: n_src_init={n_est_pts}, xmin={xmin_q}, xmax={xmax_q}, gdx={gdx_q}")
                    
                    kcsd_est = estimate_csd(
                        analog_sig, coords_q, method='KCSD1D', 
                        sigma=CSD_SIGMA_CONDUCTIVITY*pq.S/pq.m, 
                        n_src_init=n_est_pts,
                        xmin=xmin_q, xmax=xmax_q, gdx=gdx_q, 
                        ext_x=0.0*pq.um, # ext_x is important for KCSD, provide with units
                        # Do not pass lambdas and Rs here if process_estimate=False
                        process_estimate=False)
                    
                    if hasattr(kcsd_est, 'cross_validate'):
                        print(f"    Performing CV for Shk{shk_id},Ep{ep_idx}...")
                        kcsd_est.cross_validate(lambdas=KCSD_LAMBDAS_CV, Rs=KCSD_RS_CV_UM*pq.um)
                        print(f"      CV Optimal Lambda: {kcsd_est.lambd}")
                        print(f"      CV Optimal R: {kcsd_est.R * 1e6:.2f} um")
                        if hasattr(kcsd_est, 'cv_error') and kcsd_est.cv_error is not None:
                            fig_cv, ax_cv = plt.subplots(figsize=(8,6)); 
                            cv_err_plot = np.log10(kcsd_est.cv_error);
                            if np.isneginf(cv_err_plot).any():
                                finite_min = np.min(cv_err_plot[np.isfinite(cv_err_plot)]) if np.isfinite(cv_err_plot).any() else -10
                                cv_err_plot[np.isneginf(cv_err_plot)] = finite_min - 1 
                            vmin_cv = np.nanmin(cv_err_plot) if np.isfinite(cv_err_plot).any() else -10
                            vmax_cv = np.nanpercentile(cv_err_plot[np.isfinite(cv_err_plot)], 98) if np.isfinite(cv_err_plot).any() else vmin_cv + 1
                            if vmin_cv >= vmax_cv : vmax_cv = vmin_cv + (1e-9 if vmin_cv != 0 else 1e-9)
                            norm_cv = Normalize(vmin=vmin_cv, vmax=vmax_cv)
                            im_cv = ax_cv.imshow(cv_err_plot, aspect='auto', origin='lower', cmap='viridis', norm=norm_cv)
                            ax_cv.set_xticks(np.arange(len(KCSD_LAMBDAS_CV)))
                            ax_cv.set_xticklabels([f"{l:.1e}" for l in KCSD_LAMBDAS_CV], rotation=45, ha="right")
                            ax_cv.set_yticks(np.arange(len(KCSD_RS_CV_UM))) 
                            ax_cv.set_yticklabels([f"{r_val:.1f}" for r_val in KCSD_RS_CV_UM])
                            best_lambda_idx_list = np.where(np.isclose(KCSD_LAMBDAS_CV, kcsd_est.lambd))[0]
                            best_R_idx_list = np.where(np.isclose(KCSD_RS_CV_UM, kcsd_est.R * 1e6))[0] 
                            if len(best_lambda_idx_list) > 0 and len(best_R_idx_list) > 0:
                                ax_cv.scatter(best_lambda_idx_list[0], best_R_idx_list[0], marker='x', color='red', s=100, label=f'Optimal\nL={kcsd_est.lambd:.1e}\nR={kcsd_est.R*1e6:.1f}um')
                                ax_cv.legend(fontsize='small')
                            ax_cv.set_xlabel('Lambda'); ax_cv.set_ylabel('R (um)')
                            ax_cv.set_title(f'kCSD CV Error (Ep {ep_idx}, Shk {shk_id})')
                            plt.colorbar(im_cv, ax=ax_cv, label='log10(CV Error)'); fig_cv.tight_layout()
                            cv_plot_filename = output_file_prefix.parent / f"{base_filename}_ep{ep_idx}_shk{shk_id}_kCSDcv.png"
                            plt.savefig(cv_plot_filename); print(f"    Saved CV plot: {cv_plot_filename}"); plt.close(fig_cv)
                    else: print("    Estimator object does not support cross_validate, or CV parameters not supplied to it.")

                    csd_vals = kcsd_est.values() # Get CSD values (n_est_pts, n_time_pts)
                    csd_neo = neo.AnalogSignal(csd_vals.T, units=pq.uA/pq.mm**3, sampling_rate=final_csd_fs*pq.Hz,
                                               coordinates=kcsd_est.estm_x.rescale(pq.um))
                    csd_matrix_plot = np.asarray(csd_neo).astype(np.float32).T 
                    csd_times = csd_neo.times.rescale("s").magnitude
                    csd_pos_um = csd_neo.annotations['coordinates'].rescale("um").magnitude.flatten()

                    if csd_matrix_plot.size == 0: print(f"      CSD result empty for Shk{shk_id},Ep{ep_idx}."); continue
                    
                    print(f"      CSD data matrix for plotting (Ep{ep_idx},Shk{shk_id}): {csd_matrix_plot.shape}, dtype: {csd_matrix_plot.dtype}")
                    if csd_matrix_plot.size > 0: print(f"      CSD stats: min={np.nanmin(csd_matrix_plot):.2e}, max={np.nanmax(csd_matrix_plot):.2e}, mean={np.nanmean(csd_matrix_plot):.2e}")
                    
                    abs_csd_plot = np.abs(csd_matrix_plot)
                    if csd_matrix_plot.size == 0 or np.all(np.isnan(abs_csd_plot)) or not np.any(np.isfinite(abs_csd_plot)):
                        print(f"      Warning: CSD data for Ep{ep_idx},Shk{shk_id} is empty or all NaN/Inf. Plot will be blank.")
                        clim_val_plot = 1.0; ax_csd.text(0.5, 0.5, "CSD Data Empty/Invalid", ha='center', va='center', transform=ax_csd.transAxes)
                    else:
                        clim_val_plot = np.percentile(abs_csd_plot[np.isfinite(abs_csd_plot)], 99.0)
                        if clim_val_plot < 1e-9: clim_val_plot = np.nanmax(abs_csd_plot[np.isfinite(abs_csd_plot)]) if np.isfinite(abs_csd_plot).any() else 1.0
                        if abs(clim_val_plot) < 1e-9: clim_val_plot = 1e-9 
                        print(f"      Plotting clim_val: {clim_val_plot:.2e}")
                        time_edges_plot = np.concatenate([csd_times, [csd_times[-1] + (1.0 / final_csd_fs)]]) if len(csd_times) > 0 else np.array([0, 1.0/final_csd_fs])
                        if len(csd_pos_um) > 1:
                            depth_diffs = np.diff(csd_pos_um); step_val = depth_diffs[0] if len(depth_diffs) > 0 else 10.0
                            if abs(step_val) < 1e-9 : step_val = 10.0 
                            first_edge = csd_pos_um[0] - step_val/2; last_edge = csd_pos_um[-1] + step_val/2
                            mid_edges = (csd_pos_um[:-1] + csd_pos_um[1:])/2 if len(csd_pos_um) > 1 else np.array([]) 
                            depth_edges = np.concatenate([[first_edge], mid_edges, [last_edge]])
                        elif len(csd_pos_um) == 1: depth_edges = np.array([csd_pos_um[0] - 10, csd_pos_um[0] + 10])
                        else: depth_edges = np.array([0,1])
                        csd_plot_data = np.nan_to_num(csd_matrix_plot, nan=0.0) if np.any(np.isnan(csd_matrix_plot)) else csd_matrix_plot
                        img = ax_csd.pcolormesh(time_edges_plot, depth_edges, csd_plot_data, cmap='RdBu_r', shading='gouraud', vmin=-clim_val_plot, vmax=clim_val_plot, rasterized=True)
                        plt.colorbar(img, ax=ax_csd, label=f'kCSD ({csd_neo.units.dimensionality.string})', shrink=0.8)
                    ax_csd.set_ylabel('Depth (µm)'); ax_csd.set_title(f'Shank {shk_id}'); ax_csd.invert_yaxis()

                except Exception as e_csd_proc: print(f"    Error during CSD processing for Shk{shk_id},Ep{ep_idx}: {e_csd_proc}"); traceback.print_exc()
                finally:
                    del analog_sig, kcsd_est, csd_neo, csd_vals, csd_matrix_plot
                    if 'csd_plot_data' in locals(): del csd_plot_data
                    gc.collect()
            # End shank loop
            if len(unique_shanks) > 0: axs_ep[-1,0].set_xlabel('Time within epoch (s)')
            plot_fname = output_file_prefix.parent / f"{base_filename}_epoch{ep_idx}_kcsd1d_shanks.png"
            fig_ep.savefig(plot_fname, dpi=150); print(f"  Saved CSD plot for Epoch {ep_idx}: {plot_fname}"); plt.close(fig_ep)
            gc.collect()
        # End epoch loop
    finally: 
        if lfp_data_memmap_obj is not None:
            if hasattr(lfp_data_memmap_obj, '_mmap') and lfp_data_memmap_obj._mmap is not None:
                print("\nClosing LFP memmap..."); 
                try: lfp_data_memmap_obj._mmap.close() 
                except Exception as e: print(f"Warn: Error closing memmap: {e}")
            del lfp_data_memmap_obj; gc.collect(); print("LFP memmap object deleted.")

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
    if "_tcat.imec0.lf.bin" in lfp_base_name_for_ts: suggested_ts_name = lfp_base_name_for_ts.replace(".imec0.lf.bin", ".nidq_timestamps.npy")
    elif "_tcat.imec1.lf.bin" in lfp_base_name_for_ts: suggested_ts_name = lfp_base_name_for_ts.replace(".imec1.lf.bin", ".nidq_timestamps.npy")
    elif ".lf.bin" in lfp_base_name_for_ts: suggested_ts_name = lfp_base_name_for_ts.replace(".lf.bin", ".nidq_timestamps.npy")
    else: suggested_ts_name = lfp_path_obj.stem + ".nidq_timestamps.npy"
    timestamps_npy_f_selected = filedialog.askopenfilename(title="Select Timestamps NPY File", initialdir=lfp_path_obj.parent, initialfile=suggested_ts_name)
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
