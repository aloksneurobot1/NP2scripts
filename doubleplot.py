# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 10:00:00 2025

Shank-wise LFP Power Spectral Density Heatmap with Overlaid LFP Traces
(Plots data for the first minute for channels in specified brain areas on each shank)
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

# Assuming DemoReadSGLXData.readSGLX is in the same directory or accessible via PYTHONPATH
try:
    from DemoReadSGLXData.readSGLX import readMeta
except ImportError:
    print("Warning: DemoReadSGLXData.readSGLX module not found. Ensure it's in the correct path.")
    def readMeta(meta_path):
        print(f"Dummy readMeta called for {meta_path}. Real implementation missing.")
        return {
            'imAiRangeMax': '5', 'imMaxInt': '32767', 'nSavedChans': '0', 
            'imSampRate': '1000', 'niSampRate': '1000', 'imroTbl': "{'probeOpt': {'lfpGain': 80}}"
        }

# --- Configure Plotly ---
pio.renderers.default = "browser"

# --- Helper Functions ---
def get_voltage_scaling_factor(meta):
    try:
        v_max = float(meta['imAiRangeMax'])
        i_max = int(meta['imMaxInt'])
        
        imroTbl_str = meta.get('imroTbl', '{}')
        lfp_gain = 80 # Default
        try:
            if isinstance(imroTbl_str, str):
                imroTbl_str_cleaned = imroTbl_str.replace("'", "\"")
                gain_match = re.search(r"['\"]lfpGain['\"]\s*:\s*(\d+)", imroTbl_str)
                if gain_match:
                    lfp_gain = int(gain_match.group(1))
                else: 
                    import ast
                    imro_dict = ast.literal_eval(imroTbl_str_cleaned)
                    lfp_gain = imro_dict.get('probeOpt', {}).get('lfpGain', 80)
            elif isinstance(imroTbl_str, dict): 
                 lfp_gain = imroTbl_str.get('probeOpt', {}).get('lfpGain', 80)

        except Exception as e_imro:
            warnings.warn(f"Could not parse 'lfpGain' from 'imroTbl': {e_imro}. Using default LFP gain {lfp_gain}.")

        if not isinstance(lfp_gain, (int, float)) or lfp_gain <=0:
            lfp_gain = 80 
            warnings.warn(f"LFP gain {lfp_gain} is invalid, using default 80.")
        
        sf = (v_max * 1e6) / (i_max * float(lfp_gain)) 
        return sf
    except KeyError as e: warnings.warn(f"KeyError for voltage scaling: {e}. None returned."); return None
    except Exception as e: warnings.warn(f"General error in voltage scaling: {e}. None returned."); return None
            
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
    lfp_memmap_obj, fs, uv_scale_factor, 
    shank_channels_df, 
    depth_col_name,
    samples_to_use, 
    title_prefix="",
    output_dir=None, 
    base_filename_prefix="", 
    apply_notch_filter_flag=False, 
    notch_freqs=None, Q_notch=None,
    psd_freq_range=(1, 200),
    lfp_filter_range=(1, 200),
    nperseg_spec=512, 
    noverlap_spec=None,
    trace_amplitude_display_scale=0.7 
    ):

    if shank_channels_df.empty:
        print(f"No channels provided for {title_prefix}. Skipping plot.")
        return

    if noverlap_spec is None:
        noverlap_spec = nperseg_spec // 2

    sorted_shank_channels_df = shank_channels_df.sort_values(by=depth_col_name, ascending=False)
    channel_indices_ordered = sorted_shank_channels_df['global_channel_index'].tolist()
    num_channels_on_shank = len(channel_indices_ordered)

    if num_channels_on_shank == 0:
        print(f"Zero channels to plot for {title_prefix} after filtering and sorting. Skipping.")
        return

    time_vector_lfp_s = np.arange(samples_to_use) / fs

    _dummy_seg = np.zeros(samples_to_use)
    _, t_spec_vec_s, _ = spectrogram(_dummy_seg, fs=fs, nperseg=nperseg_spec, noverlap=noverlap_spec, scaling='density')
    num_time_bins_spectrogram = len(t_spec_vec_s)

    heatmap_psd_data = np.full((num_channels_on_shank, num_time_bins_spectrogram), np.nan)
    lfp_lines_data = np.full((num_channels_on_shank, samples_to_use), np.nan)

    b_line, a_line = None, None
    if lfp_filter_range and lfp_filter_range[0] < lfp_filter_range[1] and lfp_filter_range[1] < fs/2:
        nyq = fs/2.0
        low_lfp = lfp_filter_range[0]/nyq
        high_lfp = lfp_filter_range[1]/nyq
        try:
            b_line, a_line = butter(3, [low_lfp, high_lfp], btype='band')
        except ValueError as e:
            warnings.warn(f"Could not create LFP line filter ({lfp_filter_range} Hz): {e}")

    print(f"  Processing {num_channels_on_shank} channels for {title_prefix}...")
    for i_plot_ch, ch_idx in enumerate(channel_indices_ordered):
        if not (0 <= ch_idx < lfp_memmap_obj.shape[1]):
            warnings.warn(f"Channel index {ch_idx} out of bounds. Skipping for this plot.")
            continue

        lfp_segment_i16 = lfp_memmap_obj[0 : samples_to_use, ch_idx]
        lfp_segment_f64 = lfp_segment_i16.astype(np.float64)

        if uv_scale_factor is not None:
            lfp_segment_f64 *= uv_scale_factor
        
        lfp_segment_notched = lfp_segment_f64
        if apply_notch_filter_flag and notch_freqs and Q_notch:
            lfp_segment_notched = apply_notch_filters(lfp_segment_f64, fs, notch_freqs, Q_notch)

        try:
            f_spec, t_spec, Sxx_spec = spectrogram(lfp_segment_notched, fs=fs, nperseg=nperseg_spec, noverlap=noverlap_spec, scaling='density')
            Sxx_db = 10 * np.log10(Sxx_spec + np.finfo(float).eps)
            
            psd_freq_mask = (f_spec >= psd_freq_range[0]) & (f_spec <= psd_freq_range[1])
            Sxx_db_masked_freq = Sxx_db[psd_freq_mask, :]
            
            if Sxx_db_masked_freq.shape[0] > 0: 
                mean_psd_power_time_series = np.mean(Sxx_db_masked_freq, axis=0) 
                
                mean_val = np.nanmean(mean_psd_power_time_series)
                std_val = np.nanstd(mean_psd_power_time_series)
                if std_val > 1e-9 and not np.isnan(mean_val): 
                    heatmap_psd_data[i_plot_ch, :] = (mean_psd_power_time_series - mean_val) / std_val
                elif not np.isnan(mean_val): 
                     heatmap_psd_data[i_plot_ch, :] = np.zeros_like(mean_psd_power_time_series)
            else:
                warnings.warn(f"No PSD data in range {psd_freq_range} for Ch {ch_idx}. Row will be NaN.")
        except Exception as e_spec:
            warnings.warn(f"Error computing spectrogram/PSD for Ch {ch_idx}: {e_spec}")

        if b_line is not None and a_line is not None:
            try:
                lfp_lines_data[i_plot_ch, :] = filtfilt(b_line, a_line, lfp_segment_notched)
            except Exception as e_filt_line:
                 warnings.warn(f"Error filtering LFP line for Ch {ch_idx}: {e_filt_line}")
                 lfp_lines_data[i_plot_ch, :] = lfp_segment_notched 
        else:
            lfp_lines_data[i_plot_ch, :] = lfp_segment_notched 

    fig = go.Figure()

    y_axis_labels_heatmap = []
    for ch_idx_ordered in channel_indices_ordered:
        ch_row = sorted_shank_channels_df[sorted_shank_channels_df['global_channel_index'] == ch_idx_ordered].iloc[0]
        label = f"Ch {ch_idx_ordered} ({ch_row['acronym']} - {ch_row[depth_col_name]:.0f}\u00b5m)"
        y_axis_labels_heatmap.append(label)
    
    if np.any(np.isfinite(heatmap_psd_data)):
        fig.add_trace(go.Heatmap(
            z=heatmap_psd_data,
            x=t_spec_vec_s, 
            y=np.arange(num_channels_on_shank), 
            colorscale='RdBu_r', 
            zmid=0, 
            zmin=-2.5, zmax=2.5, 
            colorbar=dict(title=f'Z-Scored Mean PSD<br>({psd_freq_range[0]}-{psd_freq_range[1]}Hz)', thickness=15, len=0.8, y=0.5, x=1.03)
        ))
    else:
        print(f"No valid data for PSD heatmap for {title_prefix}. Heatmap will be empty.")

    valid_lfp_data = lfp_lines_data[np.isfinite(lfp_lines_data)]
    if valid_lfp_data.size > 0:
        max_abs_lfp_val = np.percentile(np.abs(valid_lfp_data), 98) 
        if max_abs_lfp_val < 1e-9: max_abs_lfp_val = 1.0 
    else:
        max_abs_lfp_val = 1.0

    lfp_line_scaler = trace_amplitude_display_scale / max_abs_lfp_val

    for i_plot_ch in range(num_channels_on_shank):
        ch_global_idx = channel_indices_ordered[i_plot_ch]
        lfp_trace = lfp_lines_data[i_plot_ch, :]
        
        if np.all(np.isnan(lfp_trace)):
            continue
        
        y_base_for_lfp = i_plot_ch 
        scaled_lfp_for_plot = lfp_trace * lfp_line_scaler
        
        fig.add_trace(go.Scatter(
            x=time_vector_lfp_s,
            y=y_base_for_lfp + scaled_lfp_for_plot,
            mode='lines',
            line=dict(color='rgba(0,0,0,0.7)', width=0.75), 
            name=f"LFP Ch {ch_global_idx}",
            showlegend=False 
        ))

    fig.update_layout(
        title=f"{title_prefix}PSD Heatmap & LFP Traces (First {samples_to_use/fs:.1f}s)",
        xaxis_title="Time (s)",
        yaxis_title=f"Channel (Sorted by Depth - {depth_col_name})",
        yaxis=dict(
            tickmode='array',
            tickvals=np.arange(num_channels_on_shank),
            ticktext=y_axis_labels_heatmap,
            autorange="reversed", 
            showgrid=False,
            zeroline=False
        ),
        legend_title_text='Channels',
        template="plotly_white",
        height=max(500, num_channels_on_shank * 20 + 150), 
        width=1200
    )
    fig.show()

    if output_dir and base_filename_prefix:
        plot_filepath_html = Path(output_dir) / f"{base_filename_prefix}_PSDHeatmapLFP.html"
        try: 
            fig.write_html(str(plot_filepath_html))
            print(f"    Saved Plot to: {plot_filepath_html}")
        except Exception as e: 
            print(f"    Error saving plot: {e}")

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
        
    output_d = Path(filedialog.askdirectory(title="Select Output Directory for Plots"))
    if not output_d.name: print("Cancelled."); return
    root.destroy()

    apply_notch_filter_user_choice = False
    notch_frequencies_to_use = [60, 180] 
    notch_q_to_use = 30 
    if input(f"Apply notch filter at {notch_frequencies_to_use} Hz (Q={notch_q_to_use})? (yes/no, default no): ").strip().lower() == 'yes':
        apply_notch_filter_user_choice = True
        print(f"Notch filtering ENABLED for {notch_frequencies_to_use} Hz.")
    else: print("Notch filtering DISABLED.")

    base_fname = lfp_bin_p.stem.replace('.lf', '')
    plots_output_dir = output_d / f"{base_fname}_shank_PSD_Heatmaps_LFP_first_minute"; os.makedirs(plots_output_dir, exist_ok=True)

    lfp_mm, fs_o, n_ch_total, n_samp_total, uv_sf = load_lfp_data_memmap(lfp_bin_p, meta_p)
    if lfp_mm is None or fs_o is None: print("LFP load failed. Exiting."); return
    fs_eff = fs_o
    if uv_sf is None: print("WARNING: No voltage scaling factor. LFP data in ADC units.")

    chan_df = None
    depth_column_to_use = None
    try:
        chan_df = pd.read_csv(chan_info_p)
        req_cols = ['global_channel_index','acronym','shank_index']
        if not all(c in chan_df.columns for c in req_cols):
            raise ValueError(f"Channel Info CSV missing one or more required columns: {req_cols}.")
        
        if 'ycoord_on_shank_um' in chan_df.columns: depth_column_to_use = 'ycoord_on_shank_um'
        elif 'depth' in chan_df.columns: depth_column_to_use = 'depth' 
        elif 'y_coord' in chan_df.columns: depth_column_to_use = 'y_coord' 
        else:
            warnings.warn("No standard depth column ('ycoord_on_shank_um', 'depth', 'y_coord') found in Channel Info. Channels will not be sorted by depth.")
            chan_df['dummy_depth'] = chan_df['global_channel_index'] # Use GID for order if no depth
            depth_column_to_use = 'dummy_depth'

    except Exception as e: 
        print(f"Error loading or validating Channel Info CSV: {e}")
        if hasattr(lfp_mm, '_mmap') and lfp_mm._mmap: 
            try: lfp_mm._mmap.close() 
            except Exception: pass
        return
    
    duration_for_plot_seconds = 60
    samples_for_plot = int(fs_eff * duration_for_plot_seconds)
    actual_samples_to_use = min(n_samp_total, samples_for_plot)

    if n_samp_total < samples_for_plot:
        warnings.warn(f"Recording duration ({n_samp_total/fs_eff:.2f}s) is less than requested {duration_for_plot_seconds}s. Using full available duration ({actual_samples_to_use/fs_eff:.2f}s).")
    
    if actual_samples_to_use == 0:
        print("Error: No LFP samples available. Exiting.")
        if hasattr(lfp_mm, '_mmap') and lfp_mm._mmap: lfp_mm._mmap.close()
        return

    custom_nperseg_spec = int(fs_eff * 0.256) 
    custom_noverlap_spec = custom_nperseg_spec // 2 

    # Define target brain areas
    target_acronyms_plot = ["CA1", "CA2", "CA3", "DG-mo"] # Case-sensitive for direct match
    # For "DG-mo", we often need case-insensitive matching if source data varies
    target_acronyms_upper = [acro.upper() for acro in target_acronyms_plot if acro.upper() != "DG-MO"]
    
    if chan_df is not None and 'shank_index' in chan_df.columns and depth_column_to_use:
        unique_shanks = sorted(chan_df['shank_index'].unique())
        print(f"\n--- Plotting PSD Heatmaps with LFP Traces for First {actual_samples_to_use/fs_eff:.2f} Seconds (Shank-wise) ---")
        print(f"--- Targeting areas: {', '.join(target_acronyms_plot)} ---")

        for shank_id in unique_shanks:
            print(f"  Processing Shank {shank_id}...")
            shank_channels_all_areas_df = chan_df[chan_df['shank_index'] == shank_id].copy()
            
            # Filter for target acronyms
            # For DG-mo, allow case-insensitive match; for others, use the list as is.
            is_dg_mo = shank_channels_all_areas_df['acronym'].str.upper() == "DG-MO"
            is_other_target_areas = shank_channels_all_areas_df['acronym'].isin(target_acronyms_plot) # Direct match for CA1, CA2, CA3
            
            current_shank_channels_df = shank_channels_all_areas_df[is_dg_mo | is_other_target_areas]

            if current_shank_channels_df.empty:
                print(f"    No channels found in target areas ({', '.join(target_acronyms_plot)}) for Shank {shank_id}. Skipping plot for this shank.")
                continue
            
            plot_shank_psd_heatmap_with_lfp(
                lfp_memmap_obj=lfp_mm, 
                fs=fs_eff, 
                uv_scale_factor=uv_sf, 
                shank_channels_df=current_shank_channels_df, # Pass the filtered DataFrame
                depth_col_name=depth_column_to_use,
                samples_to_use = actual_samples_to_use,
                title_prefix=f"Shank {shank_id} (Areas: {', '.join(target_acronyms_plot)}) - ",
                output_dir=plots_output_dir, 
                base_filename_prefix=f"{base_fname}_Shank{shank_id}_FilteredAreas",
                apply_notch_filter_flag=apply_notch_filter_user_choice,
                notch_freqs=notch_frequencies_to_use, 
                Q_notch=notch_q_to_use,
                psd_freq_range=(1, 200), 
                lfp_filter_range=(1, 200), 
                nperseg_spec=custom_nperseg_spec,
                noverlap_spec=custom_noverlap_spec
            )
    else:
        print("Channel info DataFrame not loaded, 'shank_index', or depth column missing. Cannot plot shank-wise.")

    if lfp_mm is not None and hasattr(lfp_mm, '_mmap') and lfp_mm._mmap is not None:
        print("Closing LFP memmap..."); 
        try: lfp_mm._mmap.close(); print("LFP memmap closed.")
        except Exception as e: print(f"Error closing memmap: {e}")
    print("\nScript finished.")

if __name__ == "__main__":
    main_shank_psd_heatmap_visualization()
