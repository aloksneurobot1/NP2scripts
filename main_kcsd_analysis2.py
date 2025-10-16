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
import gc # For garbage collection
from DemoReadSGLXData.readSGLX import readMeta

from elephant.current_source_density import estimate_csd

# --- Configuration Parameters ---
# (Default file paths can be edited or will be prompted for by filedialog)
LFP_BIN_FILE_PATH_DEFAULT = "your_lfp_file.lf.bin"
LFP_META_FILE_PATH_DEFAULT = "your_lfp_file.lf.meta"
CHANNEL_INFO_CSV_PATH_DEFAULT = "your_channel_info.csv"

OUTPUT_DIR = Path("./csd_kcsd_output_memfix_complete") # Changed output dir name
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_FS_CSD = 1000.0  # Hz, 
LFP_BAND_LOWCUT_CSD = 1.0  # Hz
LFP_BAND_HIGHCUT_CSD = 300.0 # Hz
NUMTAPS_CSD_FILTER = 101 

# kCSD Parameters
CSD_CONDUCTIVITY = 0.3  # Siemens per meter (S/m)
KCSD_LAMBDAS_CV = np.logspace(-7, -2, 9) # Range for lambda regularization
KCSD_RS_CV_UM = np.logspace(np.log10(20), np.log10(500), 9) # Range for R parameter in micrometers

# --- Voltage Scaling Function ---
def get_voltage_scaling_factor(meta):
    """Calculates the factor to convert int16 ADC values to microvolts (uV)."""
    try:
        v_max = float(meta['imAiRangeMax'])
        i_max_adc_val = float(meta['imMaxInt'])
        probe_type = int(meta.get('imDatPrb_type', 0))
        lfp_gain = None

        if probe_type in [21, 24, 2013]: # NP2.0 variants
            lfp_gain = 80.0
            print(f"  Probe type {probe_type} (NP2.0 family) detected, using fixed LFP gain: {lfp_gain}")
        else: # For NP1.0 (type 0) and potentially others
            general_lfp_gain_key = meta.get("~imChanLFGain") # SpikeGLX convention for shared gain
            if general_lfp_gain_key is not None:
                 lfp_gain = float(general_lfp_gain_key)
                 print(f"  Probe type {probe_type}, using general LFP gain from meta key '~imChanLFGain': {lfp_gain}")
                                        
        scaling_factor_uv = (v_max / i_max_adc_val) * (1.0 / lfp_gain) * 1e6
        print(f"  Calculated uV scaling factor: {scaling_factor_uv:.8f} uV/ADC_unit (Vmax={v_max}, i_max_dig={i_max_adc_val}, LFP Gain={lfp_gain})")
        return scaling_factor_uv
    except KeyError as e:
        print(f"Error (KeyError): Missing essential key in metadata for voltage scaling: {e}")
        return None
    except ValueError as e:
        print(f"Error (ValueError): Invalid value in metadata for voltage scaling: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error calculating voltage scaling factor: {e}")
        traceback.print_exc()
        return None

# --- LFP Data Loading Function (Memory Efficient) ---
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
        if uv_scale_factor is None:
            print("  Warning: Voltage scaling factor could not be determined. CSD will be in arbitrary units relative to ADC counts.")

        file_size = bin_file_path.stat().st_size
        item_size = np.dtype('int16').itemsize
        if n_channels_total == 0 or item_size == 0:
            raise ValueError("Invalid number of channels or item size from meta.")

        expected_total_bytes_per_sample_set = n_channels_total * item_size
        num_samples = file_size // expected_total_bytes_per_sample_set
        
        if file_size % expected_total_bytes_per_sample_set != 0:
            print(f"  Warning: File size {file_size} non-integer multiple of samples. Truncating to {num_samples} samples.")
        if num_samples <= 0:
            raise ValueError("Calculated number of samples is zero or negative.")

        print(f"  Calculated samples: {num_samples}. Expected data shape for memmap: ({num_samples}, {n_channels_total})")
        lfp_data_memmap = np.memmap(bin_file_path, dtype='int16', mode='r',
                                 shape=(num_samples, n_channels_total))

        print(f"  Successfully memory-mapped LFP data. Access will be on-demand.")
        return lfp_data_memmap, fs_orig, n_channels_total, uv_scale_factor
    except FileNotFoundError:
        print(f"Error: File not found - {bin_file_path} or {meta_file_path}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred in load_lfp_data_sglx_memmap: {e}")
        traceback.print_exc()
        raise

# --- Channel Info Loading ---
def load_channel_info_kcsd(csv_filepath_str):
    csv_filepath = Path(csv_filepath_str)
    print(f"Loading channel info from {csv_filepath}")
    try:
        channel_df = pd.read_csv(csv_filepath)
        required_cols = ['global_channel_index', 'shank_index', 'ycoord_on_shank_um']
        if not all(col in channel_df.columns for col in required_cols):
            raise ValueError(f"Channel info CSV must contain: {required_cols}")
        channel_df['global_channel_index'] = channel_df['global_channel_index'].astype(int)
        channel_df['shank_index'] = channel_df['shank_index'].astype(int)
        channel_df['ycoord_on_shank_um'] = channel_df['ycoord_on_shank_um'].astype(float)
        print(f"Loaded channel info for {len(channel_df)} channels.")
        return channel_df
    except FileNotFoundError:
        print(f"Error: Channel info file not found: {csv_filepath}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading channel info: {e}")
        traceback.print_exc()
        sys.exit(1)

# --- LFP Preprocessing ---
def preprocess_lfp_for_csd_kcsd(lfp_shank_data_scaled, fs_orig, target_fs, lowcut, highcut, numtaps):
    if lfp_shank_data_scaled.ndim == 1:
        lfp_shank_data_scaled = lfp_shank_data_scaled[:, np.newaxis]

    if abs(fs_orig - target_fs) < 1e-3 or fs_orig < target_fs : # If already at or below target, or very close
        downsampling_factor = 1
    else:
        downsampling_factor = int(round(fs_orig / target_fs))
        if downsampling_factor <= 0 : downsampling_factor = 1

    effective_csd_fs = fs_orig / downsampling_factor
    print(f"  Original FS: {fs_orig:.2f} Hz. Target CSD FS: {target_fs:.2f} Hz. Factor: {downsampling_factor}. Effective FS: {effective_csd_fs:.2f} Hz.")

    if downsampling_factor > 1:
        if lfp_shank_data_scaled.shape[0] <= downsampling_factor * 20: # Adjusted heuristic for decimation filter length
            print(f"  Warning: Data for shank is too short ({lfp_shank_data_scaled.shape[0]} samples) for robust decimation. Using original FS for this shank.")
            lfp_shank_downsampled = lfp_shank_data_scaled.astype(np.float64)
            effective_csd_fs = fs_orig 
        else:
            try:
                lfp_shank_downsampled = signal.decimate(lfp_shank_data_scaled.astype(np.float64),
                                                       downsampling_factor, axis=0, ftype='fir', zero_phase=True)
            except Exception as e_decimate:
                print(f"  Error during decimation: {e_decimate}. Using original data at original sampling rate for this shank.")
                lfp_shank_downsampled = lfp_shank_data_scaled.astype(np.float64)
                effective_csd_fs = fs_orig 
    else:
        lfp_shank_downsampled = lfp_shank_data_scaled.astype(np.float64)
    print(f"  LFP shape after downsampling: {lfp_shank_downsampled.shape}")

    print(f"  Filtering LFP ({lowcut}-{highcut} Hz)...")
    nyq = effective_csd_fs / 2.0
    actual_highcut = min(highcut, nyq * 0.99) 
    actual_lowcut = max(lowcut, 0.01) 
    
    if actual_lowcut >= actual_highcut:
        print(f"  Filter Error: lowcut ({actual_lowcut:.2f}) >= highcut ({actual_highcut:.2f}). Skipping filtering.")
        lfp_shank_filtered = lfp_shank_downsampled
    elif numtaps >= lfp_shank_downsampled.shape[0]:
        print(f"  Filter Warning: numtaps ({numtaps}) >= data length ({lfp_shank_downsampled.shape[0]}). Skipping filtering.")
        lfp_shank_filtered = lfp_shank_downsampled
    else:
        try:
            fir_taps = signal.firwin(numtaps, [actual_lowcut, actual_highcut], fs=effective_csd_fs, pass_zero='bandpass', window='hamming')
            lfp_shank_filtered = signal.filtfilt(fir_taps, 1.0, lfp_shank_downsampled, axis=0)
        except Exception as e_filter:
            print(f"  Error during filtering: {e_filter}. Using downsampled data without further filtering.")
            traceback.print_exc()
            lfp_shank_filtered = lfp_shank_downsampled
            
    print(f"  LFP shape after filtering: {lfp_shank_filtered.shape}")
    return lfp_shank_filtered, effective_csd_fs

# --- Main CSD Analysis Script ---
def main_kcsd_analysis(lfp_bin_file, lfp_meta_file, channel_info_csv):
    base_filename = Path(lfp_bin_file).stem
    output_file_prefix = OUTPUT_DIR / base_filename

    lfp_data_memmap_obj = None 
    try:
        lfp_data_memmap_obj, fs_orig, n_channels_total_in_file, uv_scale_factor_loaded = \
            load_lfp_data_sglx_memmap(lfp_bin_file, lfp_meta_file)
    except Exception as e:
        print(f"CRITICAL: Failed to load or memory-map LFP data: {e}")
        return

    try:
        probe_channel_df = load_channel_info_kcsd(channel_info_csv)
        unique_shanks = sorted(probe_channel_df['shank_index'].unique())
        num_shanks_found = len(unique_shanks)

        if num_shanks_found == 0:
            print("No shanks found in the probe channel info file. Exiting.")
            return

        print(f"\nFound {num_shanks_found} shanks: {unique_shanks}. Processing kCSD for each.")
        fig_csd_all_shanks, axs_csd_all_shanks = plt.subplots(num_shanks_found, 1,
                                                              figsize=(12, 4 * num_shanks_found),
                                                              sharex=True, squeeze=False)

        for i_shank, shank_id_val in enumerate(unique_shanks):
            ax_curr_csd_plot = axs_csd_all_shanks[i_shank, 0]
            print(f"\n{'='*10} Processing kCSD for Shank {shank_id_val} {'='*10}")

            shank_channel_info = probe_channel_df[probe_channel_df['shank_index'] == shank_id_val].sort_values(
                by='ycoord_on_shank_um', ascending=True
            )

            if shank_channel_info.empty:
                msg = f"Shank {shank_id_val} - No Channel Data"
                print(msg)
                ax_curr_csd_plot.set_title(msg)
                ax_curr_csd_plot.text(0.5, 0.5, "No channels found for this shank", ha='center', va='center', transform=ax_curr_csd_plot.transAxes)
                continue

            shank_global_indices = shank_channel_info['global_channel_index'].values
            electrode_coords_um_shank = shank_channel_info['ycoord_on_shank_um'].values

            if len(shank_global_indices) < 2:
                msg = f"Shank {shank_id_val} has {len(shank_global_indices)} channels. Needs >= 2 for kCSD."
                print(msg + " Skipping.")
                ax_curr_csd_plot.set_title(f"Shank {shank_id_val} - Not Enough Channels for kCSD")
                ax_curr_csd_plot.text(0.5, 0.5, msg, ha='center', va='center', transform=ax_curr_csd_plot.transAxes, wrap=True)
                continue

            if np.any(shank_global_indices >= n_channels_total_in_file) or np.any(shank_global_indices < 0):
                msg = f"Error: Shank {shank_id_val} global indices {shank_global_indices} are out of bounds for data with {n_channels_total_in_file} channels."
                print(msg)
                ax_curr_csd_plot.set_title(f"Shank {shank_id_val} - Channel Index Error")
                ax_curr_csd_plot.text(0.5, 0.5, "Channel index out of bounds.", ha='center', va='center', transform=ax_curr_csd_plot.transAxes, wrap=True)
                continue
            
            print(f"  Loading data for shank {shank_id_val} channels: {shank_global_indices}")
            try:
                lfp_this_shank_raw = lfp_data_memmap_obj[:, shank_global_indices]
                lfp_this_shank_scaled = lfp_this_shank_raw.astype(np.float64)
                if uv_scale_factor_loaded is not None:
                    lfp_this_shank_scaled *= uv_scale_factor_loaded
                else:
                    print(f"  Proceeding with ADC units for shank {shank_id_val}.")
                print(f"  Data for shank {shank_id_val} loaded and scaled. Shape: {lfp_this_shank_scaled.shape}")
            except Exception as e_load_shank:
                msg = f"Error loading/scaling data for shank {shank_id_val}: {e_load_shank}"
                print(msg)
                traceback.print_exc()
                ax_curr_csd_plot.set_title(f"Shank {shank_id_val} - Data Load Error")
                ax_curr_csd_plot.text(0.5, 0.5, "Error loading shank data", ha='center', va='center', transform=ax_curr_csd_plot.transAxes, wrap=True)
                continue
            
            lfp_processed_shank, effective_fs_csd = preprocess_lfp_for_csd_kcsd(
                lfp_this_shank_scaled, fs_orig, TARGET_FS_CSD,
                LFP_BAND_LOWCUT_CSD, LFP_BAND_HIGHCUT_CSD, NUMTAPS_CSD_FILTER
            )
            del lfp_this_shank_scaled # Free memory
            gc.collect()

            if lfp_processed_shank is None or lfp_processed_shank.shape[0] == 0:
                msg = f"Preprocessing failed for shank {shank_id_val}."
                print(msg + " Skipping.")
                ax_curr_csd_plot.set_title(f"Shank {shank_id_val} - Preprocessing Failed")
                ax_curr_csd_plot.text(0.5, 0.5, msg, ha='center', va='center', transform=ax_curr_csd_plot.transAxes, wrap=True)
                continue

            lfp_input_neo = lfp_processed_shank.T
            del lfp_processed_shank
            gc.collect()
            
            coords_quant_um = electrode_coords_um_shank.reshape(-1, 1) * pq.um

            try:
                lfp_units_str = 'uV' if uv_scale_factor_loaded is not None else 'ADC_count'
                shank_analog_signal = neo.AnalogSignal(
                    lfp_input_neo, units=lfp_units_str, sampling_rate=effective_fs_csd * pq.Hz
                )
            except Exception as e_neo:
                msg = f"Error creating neo.AnalogSignal for shank {shank_id_val}: {e_neo}"
                print(f"  {msg}. Skipping CSD.")
                ax_curr_csd_plot.set_title(f"Shank {shank_id_val} - Neo Signal Error")
                ax_curr_csd_plot.text(0.5, 0.5, "Neo Signal Error", ha='center', va='center', transform=ax_curr_csd_plot.transAxes, wrap=True)
                continue

            print(f"  Estimating kCSD1D for shank {shank_id_val} ({lfp_input_neo.shape[0]} chans, {lfp_input_neo.shape[1]} samples)")
            print(f"  Electrode y-coords (um, sorted deep to superficial): {electrode_coords_um_shank}")

            try:
                num_csd_estimation_points = max(32, int(len(electrode_coords_um_shank) * 1.5))
                csd_est_coords_um = np.linspace(electrode_coords_um_shank.min(), electrode_coords_um_shank.max(), num_csd_estimation_points)
                
                print(f"  kCSD CV Lambdas: {KCSD_LAMBDAS_CV}")
                print(f"  kCSD CV Rs (um): {KCSD_RS_CV_UM}")

                kcsd_estimator_obj = estimate_csd(
                    lfp=shank_analog_signal,
                    coordinates=coords_quant_um,
                    method='kCSD1D',
                    conductivity=CSD_CONDUCTIVITY * pq.S / pq.m,
                    lambdas=KCSD_LAMBDAS_CV,
                    Rs=KCSD_RS_CV_UM * pq.um,
                    n_sources_est=num_csd_estimation_points,
                    est_x_coords=csd_est_coords_um.reshape(-1, 1) * pq.um,
                    gdx=(csd_est_coords_um[1] - csd_est_coords_um[0]) * pq.um if len(csd_est_coords_um) > 1 else 10.0 * pq.um,
                    process_estimate=False 
                )
                
                if not hasattr(kcsd_estimator_obj, 'pykcsd_obj'):
                    msg = "Estimator object missing 'pykcsd_obj'. Cannot get CV details."
                    print(f"  Error: {msg}")
                    ax_curr_csd_plot.set_title(f"Shank {shank_id_val} - kCSD Estimator Error")
                    ax_curr_csd_plot.text(0.5, 0.5, msg, ha='center', va='center', transform=ax_curr_csd_plot.transAxes, wrap=True)
                    continue

                pykcsd_obj = kcsd_estimator_obj.pykcsd_obj
                
                if hasattr(pykcsd_obj, 'cv_error') and pykcsd_obj.cv_error is not None:
                    print(f"  Optimal Lambda from CV: {pykcsd_obj.lambd}")
                    print(f"  Optimal R from CV: {pykcsd_obj.R_selected * 1e6:.2f} um")

                    fig_cv, ax_cv = plt.subplots(figsize=(8, 6))
                    cv_err_to_plot = np.log10(pykcsd_obj.cv_error)
                    if np.isinf(cv_err_to_plot).any():
                        min_finite_cv_log = np.min(cv_err_to_plot[np.isfinite(cv_err_to_plot)]) if np.isfinite(cv_err_to_plot).any() else -10
                        cv_err_to_plot[np.isinf(cv_err_to_plot)] = min_finite_cv_log -1

                    vmin_cv = np.nanmin(cv_err_to_plot) if np.isfinite(cv_err_to_plot).any() else -10
                    vmax_cv = np.nanpercentile(cv_err_to_plot[np.isfinite(cv_err_to_plot)], 98) if np.isfinite(cv_err_to_plot).any() else vmin_cv + 1
                    if vmin_cv >= vmax_cv: vmax_cv = vmin_cv + 1e-9 if vmin_cv !=0 else 1e-9


                    norm_cv = Normalize(vmin=vmin_cv, vmax=vmax_cv)
                    im_cv = ax_cv.imshow(cv_err_to_plot, aspect='auto', origin='lower', cmap='viridis', norm=norm_cv)
                    
                    ax_cv.set_xticks(np.arange(len(pykcsd_obj.lambdas_CV)))
                    ax_cv.set_xticklabels([f"{l:.1e}" for l in pykcsd_obj.lambdas_CV], rotation=45, ha="right")
                    ax_cv.set_yticks(np.arange(len(pykcsd_obj.Rs_CV))) # Rs_CV in pykcsd_obj are in meters
                    ax_cv.set_yticklabels([f"{r_m * 1e6:.1f}" for r_m in pykcsd_obj.Rs_CV])

                    best_lambda_idx_list = np.where(np.isclose(pykcsd_obj.lambdas_CV, pykcsd_obj.lambd))[0]
                    best_R_idx_list = np.where(np.isclose(pykcsd_obj.Rs_CV, pykcsd_obj.R_selected))[0]
                    
                    if len(best_lambda_idx_list) > 0 and len(best_R_idx_list) > 0:
                        ax_cv.scatter(best_lambda_idx_list[0], best_R_idx_list[0], marker='x', color='red', s=100, label=f'Optimal\nL={pykcsd_obj.lambd:.1e}\nR={pykcsd_obj.R_selected*1e6:.1f}um')
                        ax_cv.legend(fontsize='small')

                    ax_cv.set_xlabel('Lambda (Regularization)')
                    ax_cv.set_ylabel('R (Source Extent / Kernel Width, um)')
                    ax_cv.set_title(f'kCSD1D CV Error (log10) - Shank {shank_id_val}')
                    plt.colorbar(im_cv, ax=ax_cv, label='log10(CV Error)')
                    fig_cv.tight_layout()
                    cv_plot_filename = output_file_prefix.parent / f"{base_filename}_shank_{shank_id_val}_kcsd1d_cv_error.png"
                    plt.savefig(cv_plot_filename)
                    print(f"  Saved CV plot: {cv_plot_filename}")
                    plt.close(fig_cv)
                else:
                    print("  CV error information not found or CV not performed.")

                print("  Estimating CSD with optimized parameters...")
                csd_result_neo = kcsd_estimator_obj.estimate_csd()
                
                csd_data_matrix = np.asarray(csd_result_neo)
                csd_times_vector = np.arange(csd_data_matrix.shape[1]) / effective_fs_csd
                csd_units_str = csd_result_neo.units.dimensionality.string
                
                if hasattr(csd_result_neo, 'annotations') and 'coordinates' in csd_result_neo.annotations:
                     csd_positions_plot_um = csd_result_neo.annotations['coordinates'].rescale('um').magnitude.flatten()
                else:
                     csd_positions_plot_um = csd_est_coords_um # Fallback to requested estimation coordinates

                if csd_data_matrix.size == 0:
                    msg = f"kCSD result empty for Shank {shank_id_val}"
                    print(f"  {msg}")
                    ax_curr_csd_plot.set_title(f'Shank {shank_id_val} - Empty kCSD Result')
                    ax_curr_csd_plot.text(0.5, 0.5, msg, ha='center', va='center', transform=ax_curr_csd_plot.transAxes)
                    continue

                abs_csd = np.abs(csd_data_matrix)
                clim_val = np.percentile(abs_csd[np.isfinite(abs_csd)], 99.5) if np.isfinite(abs_csd).any() else 1.0
                if clim_val == 0 or not np.isfinite(clim_val):
                    clim_val = np.nanmax(abs_csd) if np.isfinite(abs_csd).any() else 1.0
                if clim_val == 0 or not np.isfinite(clim_val):
                    clim_val = 1.0
                
                time_edges = np.concatenate([csd_times_vector, [csd_times_vector[-1] + (1.0 / effective_fs_csd)]]) if len(csd_times_vector) > 0 else np.array([0, 1.0/effective_fs_csd])
                if len(csd_positions_plot_um) > 1:
                    depth_diffs = np.diff(csd_positions_plot_um)
                    first_edge = csd_positions_plot_um[0] - (depth_diffs[0]/2 if len(depth_diffs) > 0 else 5.0)
                    last_edge = csd_positions_plot_um[-1] + (depth_diffs[-1]/2 if len(depth_diffs) > 0 else 5.0)
                    mid_edges = csd_positions_plot_um[:-1] + depth_diffs/2 if len(depth_diffs) > 0 else np.array([])
                    depth_edges = np.concatenate([[first_edge], mid_edges, [last_edge]])
                elif len(csd_positions_plot_um) == 1:
                     depth_edges = np.array([csd_positions_plot_um[0] - 10, csd_positions_plot_um[0] + 10])
                else: 
                    depth_edges = np.array([0,1])


                img_csd = ax_curr_csd_plot.pcolormesh(time_edges, depth_edges, csd_data_matrix,
                                                 cmap='RdBu_r', shading='gouraud',
                                                 vmin=-clim_val, vmax=clim_val, rasterized=True)
                plt.colorbar(img_csd, ax=ax_curr_csd_plot, label=f'kCSD ({csd_units_str})', shrink=0.8)
                ax_curr_csd_plot.set_ylabel('Depth along shank (µm)')
                ax_curr_csd_plot.set_title(f'kCSD1D - Shank {shank_id_val}')
                ax_curr_csd_plot.invert_yaxis()

            except Exception as e_kcsd:
                msg = f"kCSD Processing Error (Shank {shank_id_val}): {e_kcsd}"
                print(f"  {msg}")
                traceback.print_exc()
                ax_curr_csd_plot.set_title(f"Shank {shank_id_val} - kCSD Processing Error")
                ax_curr_csd_plot.text(0.5, 0.5, "kCSD Error", ha='center', va='center', transform=ax_curr_csd_plot.transAxes, wrap=True)

        # End of shank loop
        gc.collect() # Collect garbage after each shank to free memory

    finally: # Ensure memmap is closed even if errors occur in the loop
        if lfp_data_memmap_obj is not None and hasattr(lfp_data_memmap_obj, '_mmap') and lfp_data_memmap_obj._mmap is not None:
            print("\nClosing LFP data memory map...")
            try:
                lfp_data_memmap_obj._mmap.close()
                # Try to explicitly delete the memmap object to help with releasing file lock on Windows
                del lfp_data_memmap_obj
                gc.collect()
                print("Memory map closed and object deleted.")
            except Exception as e_close:
                print(f"Warning: Error closing memory map: {e_close}")
        else:
            print("\nLFP data memory map was not active or already handled.")


    if num_shanks_found > 0:
        axs_csd_all_shanks[-1, 0].set_xlabel('Time (s)')
    fig_csd_all_shanks.suptitle(f'kCSD1D Analysis - {base_filename}', fontsize=16)
    fig_csd_all_shanks.tight_layout(rect=[0, 0.03, 1, 0.96])

    csd_plot_filename_all = output_file_prefix.parent / f"{base_filename}_kcsd1d_all_shanks_memfix_complete.png"
    plt.savefig(csd_plot_filename_all, dpi=150)
    print(f"Saved combined CSD plot: {csd_plot_filename_all}")
    plt.show()
    plt.close(fig_csd_all_shanks)


if __name__ == "__main__":
    from tkinter import Tk, filedialog
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    print("Please select the LFP binary file (*.lf.bin)...")
    lfp_bin_f_selected = filedialog.askopenfilename(title="Select LFP Binary File", initialfile=LFP_BIN_FILE_PATH_DEFAULT)
    if not lfp_bin_f_selected: print("LFP file selection cancelled. Exiting."); sys.exit()

    print("Please select the corresponding Meta file (*.lf.meta)...")
    lfp_meta_f_selected = filedialog.askopenfilename(title="Select LFP Meta File", initialdir=Path(lfp_bin_f_selected).parent, initialfile=Path(lfp_bin_f_selected).stem + ".meta")
    if not lfp_meta_f_selected: print("Meta file selection cancelled. Exiting."); sys.exit()
    
    print("Please select the Channel Info CSV file (e.g., channel_brain_regions.csv)...")
    channel_csv_f_selected = filedialog.askopenfilename(title="Select Channel Info CSV", initialfile=CHANNEL_INFO_CSV_PATH_DEFAULT)
    if not channel_csv_f_selected: print("Channel info CSV selection cancelled. Exiting."); sys.exit()
    
    root.destroy()

    # current_timestamp = pd.Timestamp.now(tz='America/New_York') if 'America/New_York' in pd.DatetimeTZDtype.get_zoneinfo_keys() else pd.Timestamp.now()
    # print(f"\nStarting kCSD analysis script at {current_timestamp}")
    print(f"LFP File: {lfp_bin_f_selected}")
    print(f"Meta File: {lfp_meta_f_selected}")
    print(f"Channel CSV: {channel_csv_f_selected}")
    print(f"Output Directory: {OUTPUT_DIR}\n")
    
    main_kcsd_analysis(lfp_bin_file=lfp_bin_f_selected,
                         lfp_meta_file=lfp_meta_f_selected,
                         channel_info_csv=channel_csv_f_selected)
    
    # current_timestamp_end = pd.Timestamp.now(tz='America/New_York') if 'America/New_York' in pd.DatetimeTZDtype.get_zoneinfo_keys() else pd.Timestamp.now()
    # print(f"\nScript finished at {current_timestamp_end}")