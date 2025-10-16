# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 10:00:00 2025

Shank-wise LFP Power Spectral Density Heatmap with Overlaid LFP Traces
(Plots +/-100ms around a user-specified time. Calculations based on the first
 30 minutes of Epoch 1. Targets specified brain areas, 50% Spectrogram Overlap,
 Fixed LFP Gain of 80. Heatmap colorscale dynamic to plot window.)
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, welch, iirnotch, spectrogram
from scipy.stats import zscore
import plotly.graph_objects as go
import plotly.io as pio
import os
from pathlib import Path
import warnings
import re 
from tkinter import Tk, filedialog
from DemoReadSGLXData.readSGLX import readMeta

# --- Configure Plotly ---
pio.renderers.default = "browser"

# --- Helper Functions ---
def get_voltage_scaling_factor(meta):
    """
    Calculates the voltage scaling factor. LFP gain is fixed to 80.
    """
    try:
        v_max = float(meta['imAiRangeMax'])
        i_max = int(meta['imMaxInt'])
        lfp_gain = 80.0 # Fixed LFP gain        
        sf = (v_max * 1e6) / (i_max * lfp_gain) 
        return sf
    except KeyError as e: 
        warnings.warn(f"KeyError for voltage scaling (imAiRangeMax or imMaxInt missing from meta): {e}. None returned.")
        return None
           
def load_lfp_data_memmap(file_path, meta_path, data_type='int16'):
    sf_o, lfp_m, n_c, n_s, uv_sf = None, None, 0, 0, None 
    file_path, meta_path = Path(file_path), Path(meta_path)
    print(f"LFP Load: {file_path}, Meta: {meta_path}")
    try:
        meta = readMeta(meta_path)
        n_c = int(meta['nSavedChans'])
        srate_key = 'imSampRate' if 'imSampRate' in meta else 'niSampRate'
        if srate_key not in meta: print(f"Error: Sampling rate key not found."); return None, None, 0, 0, None
        sf_o = float(meta[srate_key])
        if sf_o <= 0: print(f"Error: Invalid SampRate ({sf_o})."); return None, None, n_c, 0, None
        print(f"  Meta: {n_c} ch, Rate: {sf_o:.2f} Hz")
        uv_sf = get_voltage_scaling_factor(meta) 
        if uv_sf is not None: print(f"  uV scaling factor obtained: {uv_sf:.6f}")
        fsize, isize = file_path.stat().st_size, np.dtype(data_type).itemsize
        if n_c <= 0 or isize <= 0: print(f"Error: Invalid n_ch/isize."); return None, sf_o, n_c, 0, uv_sf
        if fsize % (n_c * isize) != 0: warnings.warn(f"File size not exact multiple.")
        n_s = fsize // (n_c * isize)
        if n_s <= 0: print(f"Error: Zero samples."); return None, sf_o, n_c, 0, uv_sf
        shape = (n_s, n_c); print(f"  Samples: {n_s}. Memmap shape: {shape}")
        lfp_m = np.memmap(file_path, dtype=data_type, mode='r', shape=shape, offset=0)
        print(f"  Memmap successful: {file_path}")
        return lfp_m, sf_o, n_c, n_s, uv_sf
    except Exception as e: print(f"LFP Load Error: {e}"); import traceback; traceback.print_exc(); return None,None,0,0,None

def load_epoch_data(epoch_boundaries_filepath, fs_orig):
    """Loads epoch boundaries from a .npy file."""
    epoch_samp = []
    if epoch_boundaries_filepath and Path(epoch_boundaries_filepath).exists():
        try:
            ep_s = np.load(epoch_boundaries_filepath, allow_pickle=True)
            if isinstance(ep_s, np.ndarray) and ep_s.ndim == 2 and ep_s.shape[1] == 2:
                valid_ep_s = [tuple(ep) for ep in ep_s if ep[1] >= ep[0]]
            elif isinstance(ep_s, list): 
                 valid_ep_s = [tuple(ep) for ep in ep_s if isinstance(ep, (list, tuple, np.ndarray)) and len(ep) == 2 and all(isinstance(t, (int, float)) for t in ep) and ep[1] >= ep[0]]
            else:
                valid_ep_s = []
                warnings.warn(f"Epoch file format not as expected (should be list/array of [start, end] pairs). Path: {epoch_boundaries_filepath}")

            epoch_samp = [(int(s * fs_orig), int(e * fs_orig)) for s, e in valid_ep_s] 
            print(f"Loaded {len(epoch_samp)} epochs from {Path(epoch_boundaries_filepath).name}.")
        except Exception as e: 
            print(f"Epoch load error from {Path(epoch_boundaries_filepath).name}: {e}")
            epoch_samp = [] 
    else:
        print(f"Epoch boundaries file not found or not provided: {epoch_boundaries_filepath}")
    return epoch_samp

def apply_notch_filters(data_segment, fs, notch_freqs_list, Q_factor):
    notched_data = data_segment.copy()
    for freq_to_notch in notch_freqs_list:
        if freq_to_notch <= 0 or freq_to_notch >= fs / 2:
            warnings.warn(f"Notch frequency {freq_to_notch} Hz is invalid for fs={fs} Hz. Skipping this notch.")
            continue
        try:
            b_notch, a_notch = iirnotch(freq_to_notch, Q_factor, fs)
            notched_data = filtfilt(b_notch, a_notch, notched_data)
        except ValueError as e:
            warnings.warn(f"Could not apply notch filter at {freq_to_notch} Hz: {e}")
    return notched_data

def plot_shank_psd_heatmap_with_lfp(
    lfp_data_for_calc_shank, 
    fs, 
    channel_info_on_shank_df, 
    depth_col_name_in_df,     
    plot_start_idx_in_calc,   
    plot_duration_samples,    
    title_prefix="",
    output_dir=None, 
    base_filename_prefix="", 
    psd_freq_range=(1, 200),
    lfp_filter_range=(1, 200),
    nperseg_spec=256, 
    noverlap_spec=None, 
    trace_amplitude_display_scale=1,
    user_specified_time_s=None 
    ):

    if lfp_data_for_calc_shank is None or lfp_data_for_calc_shank.shape[0] == 0:
        print(f"No LFP data provided for calculation in {title_prefix}. Skipping plot.")
        return
    if channel_info_on_shank_df.empty:
        print(f"No channel info provided for {title_prefix}. Skipping plot.")
        return
    if plot_duration_samples == 0:
        print(f"Zero samples requested for plotting in {title_prefix}. Skipping plot.")
        return

    num_channels_on_shank = lfp_data_for_calc_shank.shape[1]
    samples_for_calc = lfp_data_for_calc_shank.shape[0]
    
    if plot_start_idx_in_calc < 0 or (plot_start_idx_in_calc + plot_duration_samples) > samples_for_calc:
        warnings.warn(f"Plot window [{plot_start_idx_in_calc}, {plot_start_idx_in_calc + plot_duration_samples}] "
                      f"is out of bounds for calculation data [0, {samples_for_calc}]. Skipping plot for {title_prefix}")
        return

    if noverlap_spec is None: 
        if nperseg_spec > 0:
            noverlap_calc = int(nperseg_spec * 0.50) 
            if noverlap_calc >= nperseg_spec: noverlap_calc = max(0, nperseg_spec -1) 
            noverlap_spec = noverlap_calc
        else: noverlap_spec = 0 
    elif noverlap_spec >= nperseg_spec and nperseg_spec > 0: 
        warnings.warn(f"Provided noverlap_spec ({noverlap_spec}) >= nperseg_spec ({nperseg_spec}). Adjusting.")
        noverlap_spec = max(0, nperseg_spec - 1)

    time_vector_lfp_plotting_ms = (np.arange(plot_duration_samples) - plot_duration_samples // 2) / fs * 1000

    t_spec_vec_s_full_calc = np.array([])
    num_time_bins_spectrogram_full_calc = 0
    if samples_for_calc > 0 and nperseg_spec > 0 and samples_for_calc >= nperseg_spec:
        _dummy_seg_calc = np.zeros(samples_for_calc)
        try:
            _, t_spec_vec_s_full_calc, _ = spectrogram(_dummy_seg_calc, fs=fs, nperseg=nperseg_spec, noverlap=noverlap_spec, scaling='density')
            num_time_bins_spectrogram_full_calc = len(t_spec_vec_s_full_calc)
        except ValueError as e_spec_init:
            warnings.warn(f"Error initializing spectrogram for full calc time vector: {e_spec_init}.")
    else:
        warnings.warn(f"Not enough samples ({samples_for_calc}) or invalid nperseg_spec ({nperseg_spec}) for full spectrogram.")

    heatmap_psd_data_zscored_full_calc = np.full((num_channels_on_shank, num_time_bins_spectrogram_full_calc), np.nan) if num_time_bins_spectrogram_full_calc > 0 else np.array([[] for _ in range(num_channels_on_shank)])
    lfp_lines_data_filtered_full_calc = np.full((num_channels_on_shank, samples_for_calc), np.nan)

    b_line, a_line = None, None
    if lfp_filter_range and len(lfp_filter_range) == 2 and lfp_filter_range[0] < lfp_filter_range[1] and lfp_filter_range[1] < fs/2 and lfp_filter_range[0] > 0:
        nyq = fs/2.0; low_lfp = lfp_filter_range[0]/nyq; high_lfp = lfp_filter_range[1]/nyq
        try: b_line, a_line = butter(3, [low_lfp, high_lfp], btype='band')
        except ValueError as e: warnings.warn(f"Could not create LFP line filter ({lfp_filter_range} Hz): {e}")

    print(f"  Processing {num_channels_on_shank} channels for {title_prefix} (Full calc duration {samples_for_calc/fs:.1f}s, nperseg={nperseg_spec}, noverlap={noverlap_spec})...")
    for i_ch_local in range(num_channels_on_shank): 
        lfp_ch_calc = lfp_data_for_calc_shank[:, i_ch_local] 

        if num_time_bins_spectrogram_full_calc > 0 and len(lfp_ch_calc) >= nperseg_spec :
            try:
                f_spec, t_spec, Sxx_spec = spectrogram(lfp_ch_calc, fs=fs, nperseg=nperseg_spec, noverlap=noverlap_spec, scaling='density')
                if len(t_spec) == num_time_bins_spectrogram_full_calc: 
                    Sxx_db = 10 * np.log10(Sxx_spec + np.finfo(float).eps)
                    psd_freq_mask = (f_spec >= psd_freq_range[0]) & (f_spec <= psd_freq_range[1])
                    Sxx_db_masked_freq = Sxx_db[psd_freq_mask, :]
                    
                    if Sxx_db_masked_freq.shape[0] > 0: 
                        mean_psd_power_time_series = np.mean(Sxx_db_masked_freq, axis=0) 
                        mean_val = np.nanmean(mean_psd_power_time_series); std_val = np.nanstd(mean_psd_power_time_series)
                        if std_val > 1e-9 and not np.isnan(mean_val): 
                            heatmap_psd_data_zscored_full_calc[i_ch_local, :] = (mean_psd_power_time_series - mean_val) / std_val
                        elif not np.isnan(mean_val): 
                             heatmap_psd_data_zscored_full_calc[i_ch_local, :] = np.zeros_like(mean_psd_power_time_series)
                
            except Exception as e_spec: warnings.warn(f"Error computing spectrogram/PSD for Ch {channel_info_on_shank_df.iloc[i_ch_local]['global_channel_index']}: {e_spec}")
        
        if b_line is not None and a_line is not None:
            try: lfp_lines_data_filtered_full_calc[i_ch_local, :] = filtfilt(b_line, a_line, lfp_ch_calc)
            except Exception as e_filt_line:
                 warnings.warn(f"Error filtering LFP line for Ch {channel_info_on_shank_df.iloc[i_ch_local]['global_channel_index']}: {e_filt_line}")
                 lfp_lines_data_filtered_full_calc[i_ch_local, :] = lfp_ch_calc 
        else: lfp_lines_data_filtered_full_calc[i_ch_local, :] = lfp_ch_calc 

    lfp_lines_data_plotting = lfp_lines_data_filtered_full_calc[:, plot_start_idx_in_calc : plot_start_idx_in_calc + plot_duration_samples]

    t_spec_vec_s_plotting_centered = np.array([])
    heatmap_psd_data_plotting = np.array([[] for _ in range(num_channels_on_shank)])

    if num_time_bins_spectrogram_full_calc > 0 and t_spec_vec_s_full_calc.size > 0:
        plot_window_start_time_in_calc_s = plot_start_idx_in_calc / fs
        plot_window_end_time_in_calc_s = (plot_start_idx_in_calc + plot_duration_samples) / fs
        
        heatmap_time_indices_to_plot_bool = (t_spec_vec_s_full_calc >= plot_window_start_time_in_calc_s) & \
                                            (t_spec_vec_s_full_calc < plot_window_end_time_in_calc_s)
        heatmap_time_indices_to_plot = np.where(heatmap_time_indices_to_plot_bool)[0]

        if heatmap_time_indices_to_plot.size > 0:
            heatmap_psd_data_plotting = heatmap_psd_data_zscored_full_calc[:, heatmap_time_indices_to_plot]
            selected_times_s = t_spec_vec_s_full_calc[heatmap_time_indices_to_plot]
            center_of_plot_window_in_calc_s = plot_window_start_time_in_calc_s + (plot_duration_samples / (2 * fs))
            t_spec_vec_s_plotting_centered = (selected_times_s - center_of_plot_window_in_calc_s) * 1000 
        else: 
            warnings.warn(f"No spectrogram time bins fall within the {plot_duration_samples/fs*1000:.0f}ms plot window. Heatmap will be empty.")
    
    fig = go.Figure()
    y_axis_labels_heatmap = [f"Ch {channel_info_on_shank_df.iloc[i_ch_local]['global_channel_index']} ({channel_info_on_shank_df.iloc[i_ch_local]['acronym']} - {channel_info_on_shank_df.iloc[i_ch_local][depth_col_name_in_df]:.0f}\u00b5m)" 
                             for i_ch_local in range(num_channels_on_shank)]
    
    # --- Dynamic Heatmap Scaling ---
    zmin_plot, zmax_plot = -2.5, 2.5 # Default fallback
    if np.any(np.isfinite(heatmap_psd_data_plotting)) and heatmap_psd_data_plotting.shape[1] > 0:
        finite_values_in_plot_window = heatmap_psd_data_plotting[np.isfinite(heatmap_psd_data_plotting)]
        if finite_values_in_plot_window.size > 0:
            zmin_plot = np.min(finite_values_in_plot_window)
            zmax_plot = np.max(finite_values_in_plot_window)
            if np.isclose(zmin_plot, zmax_plot): # Handle case where all values are the same
                zmin_plot -= 0.5 
                zmax_plot += 0.5
        # else: fallback to default zmin_plot, zmax_plot defined above
            
        fig.add_trace(go.Heatmap(
            z=heatmap_psd_data_plotting, x=t_spec_vec_s_plotting_centered, y=np.arange(num_channels_on_shank), 
            colorscale='RdBu_r', 
            zmid=0, # Keep zmid=0 as data is Z-scored from a larger context
            zmin=zmin_plot, 
            zmax=zmax_plot, 
            colorbar=dict(title=f'Z-Scored Mean PSD<br>({psd_freq_range[0]}-{psd_freq_range[1]}Hz)', thickness=15, len=0.8, y=0.5, x=1.03)
        ))
    else: print(f"No valid data for PSD heatmap plotting for {title_prefix}. Heatmap may be empty.")

    valid_lfp_data_full_calc = lfp_lines_data_filtered_full_calc[np.isfinite(lfp_lines_data_filtered_full_calc)]
    max_abs_lfp_val = np.percentile(np.abs(valid_lfp_data_full_calc), 98) if valid_lfp_data_full_calc.size > 0 else 1.0
    if max_abs_lfp_val < 1e-9: max_abs_lfp_val = 1.0 
    lfp_line_scaler = trace_amplitude_display_scale / max_abs_lfp_val

    for i_ch_local in range(num_channels_on_shank):
        lfp_trace_plotting = lfp_lines_data_plotting[i_ch_local, :]
        if np.all(np.isnan(lfp_trace_plotting)): continue
        
        y_base_for_lfp = i_ch_local 
        scaled_lfp_for_plot = lfp_trace_plotting * lfp_line_scaler
        
        fig.add_trace(go.Scatter(
            x=time_vector_lfp_plotting_ms, y=y_base_for_lfp + scaled_lfp_for_plot, mode='lines',
            line=dict(color='rgba(0,0,0,0.7)', width=0.75), 
            name=f"LFP Ch {channel_info_on_shank_df.iloc[i_ch_local]['global_channel_index']}", showlegend=False 
        ))
    
    plot_title_detail = f"User Time {user_specified_time_s:.2f}s in Epoch 1" if user_specified_time_s is not None else "Epoch 1 Center"
    fig.update_layout(
        title=f"{title_prefix}PSD Heatmap & LFP ({plot_title_detail} +/-{plot_duration_samples/fs*1000/2:.0f}ms)",
        xaxis_title=f"Time from {plot_title_detail} (ms)", 
        yaxis_title=f"Channel (Sorted by Depth - {depth_col_name_in_df})", 
        yaxis=dict(tickmode='array', tickvals=np.arange(num_channels_on_shank), ticktext=y_axis_labels_heatmap, autorange="reversed", showgrid=False, zeroline=False),
        template="plotly_white", height=max(500, num_channels_on_shank * 20 + 150), width=1200
    )
    fig.show()

    if output_dir and base_filename_prefix:
        plot_filepath_html = Path(output_dir) / f"{base_filename_prefix}_UserTime_Epoch1Plot.html"
        try: fig.write_html(str(plot_filepath_html)); print(f"    Saved Plot to: {plot_filepath_html}")
        except Exception as e: print(f"    Error saving plot: {e}")

# --- Main script logic ---
def main_shank_psd_heatmap_visualization():
    root = Tk(); root.withdraw(); root.attributes("-topmost", True)
    print("Starting Shank-wise LFP PSD Heatmap & Traces Visualization Script...")
    
    lfp_bin_p = Path(filedialog.askopenfilename(title="Select LFP Binary File (*.lf.bin)", filetypes=[("SGLX LFP files", "*.lf.bin")]))
    if not lfp_bin_p.name: print("Cancelled."); return
    meta_p = Path(filedialog.askopenfilename(title="Select Meta File (*.meta)", initialdir=lfp_bin_p.parent, initialfile=lfp_bin_p.with_suffix('.meta').name, filetypes=[("Meta files", "*.meta")]))
    if not meta_p.name: print("Cancelled."); return
    chan_info_p = Path(filedialog.askopenfilename(title="Select Channel Info CSV", filetypes=[("CSV files", "*.csv")]))
    if not chan_info_p.name: print("Cancelled."); return
    epoch_p_str = filedialog.askopenfilename(title="Select Epoch Boundaries File (*_epoch_boundaries.npy)", filetypes=[("NPY files", "*.npy")], initialdir=lfp_bin_p.parent)
    epoch_p = Path(epoch_p_str) if epoch_p_str else None 
        
    output_d = Path(filedialog.askdirectory(title="Select Output Directory for Plots"))
    if not output_d.name: print("Cancelled."); return
    root.destroy()

    apply_notch_filter_user_choice = False
    notch_frequencies_to_use = [60, 180]; notch_q_to_use = 60 
    if input(f"Apply notch filter at {notch_frequencies_to_use} Hz (Q={notch_q_to_use})? (yes/no, default no): ").strip().lower() == 'yes':
        apply_notch_filter_user_choice = True; print(f"Notch filtering ENABLED for {notch_frequencies_to_use} Hz.")
    else: print("Notch filtering DISABLED.")

    base_fname = lfp_bin_p.stem.replace('.lf', '')
    plots_output_dir = output_d / f"{base_fname}_shank_UserTime_Epoch1_Plots"; os.makedirs(plots_output_dir, exist_ok=True)

    lfp_mm, fs_o, n_ch_total, n_samp_total, uv_sf = load_lfp_data_memmap(lfp_bin_p, meta_p)
    if lfp_mm is None or fs_o is None: print("LFP load failed. Exiting."); return
    fs_eff = fs_o
    
    chan_df = None; depth_column_to_use = None 
    try:
        chan_df = pd.read_csv(chan_info_p)
        req_cols = ['global_channel_index','acronym','shank_index']
        if not all(c in chan_df.columns for c in req_cols): raise ValueError(f"Channel Info CSV missing one or more required columns: {req_cols}.")
        
        if 'ycoord_on_shank_um' in chan_df.columns: depth_column_to_use = 'ycoord_on_shank_um'
        elif 'depth' in chan_df.columns: depth_column_to_use = 'depth' 
        elif 'y_coord' in chan_df.columns: depth_column_to_use = 'y_coord' 
        else:
            warnings.warn("No standard depth column found. Channels will not be sorted by depth."); chan_df['dummy_depth'] = chan_df['global_channel_index']; depth_column_to_use = 'dummy_depth'
    except Exception as e: 
        print(f"Error loading or validating Channel Info CSV: {e}")
        if hasattr(lfp_mm, '_mmap') and lfp_mm._mmap: 
            try: lfp_mm._mmap.close() 
            except Exception: pass
        return

    epoch_boundaries_samples = load_epoch_data(epoch_p, fs_eff)
    if not epoch_boundaries_samples:
        print("No epoch boundaries loaded or file not provided. Cannot proceed with epoch-specific plot.")
        if hasattr(lfp_mm, '_mmap') and lfp_mm._mmap: lfp_mm._mmap.close()
        return
    if len(epoch_boundaries_samples) == 0:
        print("Epoch boundaries file was empty or invalid. Cannot proceed.")
        if hasattr(lfp_mm, '_mmap') and lfp_mm._mmap: lfp_mm._mmap.close()
        return
        
    epoch1_start_samp, epoch1_end_samp = epoch_boundaries_samples[0]
    calc_window_max_duration_s = 1800 
    epoch1_duration_s = (epoch1_end_samp - epoch1_start_samp) / fs_eff
    
    actual_calc_duration_s = min(calc_window_max_duration_s, epoch1_duration_s)
    actual_samples_for_calc_overall = int(actual_calc_duration_s * fs_eff)
    calc_abs_start_samp_overall = epoch1_start_samp 
    calc_abs_end_samp_overall = epoch1_start_samp + actual_samples_for_calc_overall
    
    calc_abs_end_samp_overall = min(calc_abs_end_samp_overall, n_samp_total)
    actual_samples_for_calc_overall = calc_abs_end_samp_overall - calc_abs_start_samp_overall
    actual_calc_duration_s = actual_samples_for_calc_overall / fs_eff

    if actual_samples_for_calc_overall <= 0:
        print(f"Calculation window (first {actual_calc_duration_s:.1f}s of Epoch 1) is zero or negative. Cannot proceed.")
        if hasattr(lfp_mm, '_mmap') and lfp_mm._mmap: lfp_mm._mmap.close()
        return
    print(f"Calculation window: Samples {calc_abs_start_samp_overall}-{calc_abs_end_samp_overall} (Duration: {actual_calc_duration_s:.1f}s from start of Epoch 1)")

    user_specified_time_relative_to_calc_window_s = None
    while user_specified_time_relative_to_calc_window_s is None:
        try:
            input_str = input(f"Enter a time in seconds (0.0 to {actual_calc_duration_s:.2f}) within the calculation window (first part of Epoch 1) to center the plot: ")
            val = float(input_str)
            if 0.0 <= val <= actual_calc_duration_s:
                user_specified_time_relative_to_calc_window_s = val
            else:
                print(f"Time out of range. Please enter a value between 0.0 and {actual_calc_duration_s:.2f} s.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    user_specified_center_samp_in_calc_window = int(user_specified_time_relative_to_calc_window_s * fs_eff)
    user_specified_center_abs_samp = calc_abs_start_samp_overall + user_specified_center_samp_in_calc_window
    
    print(f"Targeting user-specified time: {user_specified_time_relative_to_calc_window_s:.2f}s within calc window (Absolute sample: {user_specified_center_abs_samp})")

    plot_window_duration_s = 10 
    plot_window_half_samples = int((plot_window_duration_s / 2) * fs_eff)
    
    plot_abs_start_samp = max(0, user_specified_center_abs_samp - plot_window_half_samples)
    plot_abs_end_samp = min(n_samp_total, user_specified_center_abs_samp + plot_window_half_samples)
    actual_plot_duration_samples = plot_abs_end_samp - plot_abs_start_samp

    if actual_plot_duration_samples <= 0:
        print(f"Plot window around user time is zero or negative duration. Cannot plot.")
        if hasattr(lfp_mm, '_mmap') and lfp_mm._mmap: lfp_mm._mmap.close()
        return
    print(f"Plotting window: Samples {plot_abs_start_samp}-{plot_abs_end_samp} (Duration: {actual_plot_duration_samples/fs_eff:.3f}s)")

    plot_start_idx_within_calc = plot_abs_start_samp - calc_abs_start_samp_overall
    
    if plot_start_idx_within_calc < 0:
        warnings.warn("Plot window start is before calculation window start. Adjusting plot window start.")
        plot_abs_start_samp = calc_abs_start_samp_overall
        plot_start_idx_within_calc = 0
        plot_abs_end_samp = min(plot_abs_start_samp + int(plot_window_duration_s * fs_eff), calc_abs_end_samp_overall, n_samp_total)
        actual_plot_duration_samples = plot_abs_end_samp - plot_abs_start_samp
        if actual_plot_duration_samples <= 0:
             print("Adjusted plot duration is zero or negative after aligning with calculation window. Cannot plot."); return
        print(f"Adjusted plotting window: Samples {plot_abs_start_samp}-{plot_abs_end_samp} (Duration: {actual_plot_duration_samples/fs_eff:.3f}s)")
    
    if (plot_start_idx_within_calc + actual_plot_duration_samples) > actual_samples_for_calc_overall:
        warnings.warn("Plot window extends beyond calculation window. Truncating plot duration.")
        actual_plot_duration_samples = actual_samples_for_calc_overall - plot_start_idx_within_calc
        if actual_plot_duration_samples <= 0:
            print("Adjusted plot duration is zero or negative after aligning with calculation window end. Cannot plot."); return
        print(f"Truncated plotting duration to {actual_plot_duration_samples/fs_eff:.3f}s to fit within calculation window.")

    custom_nperseg_spec = int(2 * fs_eff) 
    if custom_nperseg_spec > actual_samples_for_calc_overall : 
        custom_nperseg_spec = actual_samples_for_calc_overall
        warnings.warn(f"nperseg for spectrogram (2*fs = {int(2*fs_eff)}) was greater than "
                      f"calculation data length ({actual_samples_for_calc_overall} samples). "
                      f"Setting nperseg to data length ({custom_nperseg_spec}).")
    
    custom_noverlap_spec = 0 
    if custom_nperseg_spec > 0:
        noverlap_target = int(custom_nperseg_spec * 0.50)
        if custom_nperseg_spec == actual_samples_for_calc_overall:
            custom_noverlap_spec = noverlap_target
            if custom_noverlap_spec >= custom_nperseg_spec: custom_noverlap_spec = max(0, custom_nperseg_spec - 1)
        else:
            custom_noverlap_spec = noverlap_target
            if custom_noverlap_spec >= custom_nperseg_spec and custom_nperseg_spec > 0 :
                custom_noverlap_spec = max(0, custom_nperseg_spec - 1) 
                warnings.warn(f"Calculated 50% noverlap ({noverlap_target}) was >= nperseg ({custom_nperseg_spec}). Adjusted to {custom_noverlap_spec}.")
    else: warnings.warn(f"nperseg_spec is {custom_nperseg_spec}, setting noverlap_spec to 0.")

    target_acronyms_plot = ["CA1", "CA2", "CA3", "DG-mo"] 
    
    if chan_df is not None and 'shank_index' in chan_df.columns and depth_column_to_use: 
        unique_shanks = sorted(chan_df['shank_index'].unique())
        print(f"\n--- Plotting for user time {user_specified_time_relative_to_calc_window_s:.2f}s in calc window (+/-{plot_window_duration_s*1000/2:.0f}ms) ---")
        print(f"--- Targeting areas: {', '.join(target_acronyms_plot)} ---")

        for shank_id in unique_shanks:
            print(f"  Processing Shank {shank_id}...")
            shank_channels_all_areas_df = chan_df[chan_df['shank_index'] == shank_id].copy()
            
            is_dg_mo = shank_channels_all_areas_df['acronym'].str.upper() == "DG-MO"
            is_other_target_areas = shank_channels_all_areas_df['acronym'].apply(lambda x: x in target_acronyms_plot if isinstance(x, str) else False)
            current_shank_channels_df = shank_channels_all_areas_df[is_dg_mo | is_other_target_areas].sort_values(by=depth_column_to_use, ascending=False)

            if current_shank_channels_df.empty:
                print(f"    No channels in target areas for Shank {shank_id}. Skipping."); continue
            
            shank_channel_indices = current_shank_channels_df['global_channel_index'].tolist()
            valid_shank_channel_indices = [idx for idx in shank_channel_indices if 0 <= idx < n_ch_total]
            
            if not valid_shank_channel_indices:
                print(f"    No valid global channel indices for Shank {shank_id} after filtering. Skipping."); continue

            valid_channels_on_shank_df = current_shank_channels_df[current_shank_channels_df['global_channel_index'].isin(valid_shank_channel_indices)].copy()

            lfp_data_for_calc_shank_raw = lfp_mm[calc_abs_start_samp_overall:calc_abs_end_samp_overall, valid_shank_channel_indices].astype(np.float64)
            
            lfp_data_for_calc_shank_processed = np.zeros_like(lfp_data_for_calc_shank_raw)
            for i_ch_proc in range(lfp_data_for_calc_shank_raw.shape[1]):
                temp_ch_data = lfp_data_for_calc_shank_raw[:, i_ch_proc].copy()
                if uv_sf is not None: 
                    temp_ch_data *= uv_sf
                if apply_notch_filter_user_choice:
                    temp_ch_data = apply_notch_filters(temp_ch_data, fs_eff, notch_frequencies_to_use, notch_q_to_use)
                lfp_data_for_calc_shank_processed[:, i_ch_proc] = temp_ch_data
            
            plot_shank_psd_heatmap_with_lfp(
                lfp_data_for_calc_shank=lfp_data_for_calc_shank_processed, 
                fs=fs_eff, 
                channel_info_on_shank_df=valid_channels_on_shank_df, 
                depth_col_name_in_df=depth_column_to_use, 
                plot_start_idx_in_calc = plot_start_idx_within_calc,
                plot_duration_samples = actual_plot_duration_samples,
                title_prefix=f"Shank {shank_id} (Areas: {', '.join(target_acronyms_plot)}) - ",
                output_dir=plots_output_dir, 
                base_filename_prefix=f"{base_fname}_Shank{shank_id}_UserTime_Plot",
                psd_freq_range=(1, 200), lfp_filter_range=(1, 200), 
                nperseg_spec=custom_nperseg_spec, noverlap_spec=custom_noverlap_spec,
                user_specified_time_s = user_specified_time_relative_to_calc_window_s 
            )
    else: print("Channel info DataFrame not loaded, 'shank_index', or depth column missing. Cannot plot shank-wise.")

    if lfp_mm is not None and hasattr(lfp_mm, '_mmap') and lfp_mm._mmap is not None:
        print("Closing LFP memmap..."); 
        try: lfp_mm._mmap.close(); print("LFP memmap closed.")
        except Exception as e: print(f"Error closing memmap: {e}")
    print("\nScript finished.")

if __name__ == "__main__":
    main_shank_psd_heatmap_visualization()
