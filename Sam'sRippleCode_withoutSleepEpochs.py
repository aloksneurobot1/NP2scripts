# -*- coding: utf-8 -*-
"""
Ripple Detection and Visualization Script
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, welch
from scipy.stats import zscore
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib # Import matplotlib itself
import matplotlib.pyplot as plt # Then pyplot
import os
from pathlib import Path
import warnings 
from tkinter import Tk, filedialog
from DemoReadSGLXData.readSGLX import readMeta
# --- Configure Plotly to open in browser ---
pio.renderers.default = "browser"

# --- Configure Matplotlib to use an interactive backend ---
try:
    matplotlib.use('Qt5Agg') 
    print("Matplotlib backend set to Qt5Agg.")
except ImportError:
    try:
        matplotlib.use('TkAgg') 
        print("Matplotlib backend set to TkAgg.")
    except ImportError:
        print("Warning: Could not set an interactive Matplotlib backend (Qt5Agg or TkAgg). Ripple example plots might not display.")
except Exception as e:
    print(f"Warning: Error setting Matplotlib backend: {e}. Ripple example plots might not display.")




# --- Helper Function for Voltage Scaling (Corrected) ---
def get_voltage_scaling_factor(meta):
    """Calculates the factor to convert int16 ADC values to microvolts (uV)."""
    try:
        v_max = float(meta['imAiRangeMax'])
        i_max = int(meta['imMaxInt'])
        lfp_gain = 80.0

        if i_max == 0 or lfp_gain == 0: 
            warnings.warn(f"Imax ({i_max}) or LFP gain ({lfp_gain}) is zero in metadata. Cannot calculate voltage scaling factor.")
            return None        
        scaling_factor_uv = (v_max * 1e6) / (i_max * lfp_gain)
        print(f"  Calculated uV scaling factor: {scaling_factor_uv:.6f} uV/ADC_unit, LFP Gain: {lfp_gain})")
        return scaling_factor_uv
    except KeyError as e:
        warnings.warn(f"Warning: Missing key in metadata for voltage scaling: {e}. Returning None.")
        return None
           
# --- Data Loading Function for SpikeGLX ---
def load_lfp_data_memmap(file_path, meta_path, data_type='int16'):
    """Loads LFP data using memory-mapping and extracts scaling factor."""
    sampling_rate = None
    lfp_memmap = None 
    n_channels = 0
    n_samples = 0
    uv_scale_factor = None
    
    file_path = Path(file_path)
    meta_path = Path(meta_path)

    print(f"Attempting to load LFP data from: {file_path}")
    print(f"Using metadata from: {meta_path}")
    try:
        meta = readMeta(meta_path)
        n_channels = int(meta['nSavedChans'])
        
        if 'imSampRate' in meta:
            sampling_rate = float(meta['imSampRate'])
        elif 'niSampRate' in meta:
            sampling_rate = float(meta['niSampRate'])
        else:
            print(f"Error: Sampling rate key ('imSampRate' or 'niSampRate') not found in meta file.")
            return None, None, 0, 0, None
            
        if sampling_rate <= 0:
            print(f"Error: Invalid sampling rate ({sampling_rate}).")
            return None, None, n_channels, 0, None
        print(f"  Metadata: {n_channels} channels. Original Rate: {sampling_rate:.2f} Hz")

        uv_scale_factor = get_voltage_scaling_factor(meta)

        file_size = file_path.stat().st_size
        item_size = np.dtype(data_type).itemsize
        
        if n_channels <= 0 or item_size <= 0:
            print(f"Error: Invalid n_channels ({n_channels}) or item_size ({item_size}).")
            return None, sampling_rate, n_channels, 0, uv_scale_factor
            
        if file_size % (n_channels * item_size) != 0:
            warnings.warn(f"Warning: File size {file_size} is not an exact multiple of n_channels ({n_channels}) * item_size ({item_size}). Data might be incomplete if recording was interrupted.")
        
        n_samples = file_size // (n_channels * item_size)
        
        if n_samples <= 0:
            print(f"Error: Zero samples calculated from file size.")
            return None, sampling_rate, n_channels, 0, uv_scale_factor
            
        shape = (n_samples, n_channels) 
        print(f"  Calculated samples: {n_samples}. Memmap shape: {shape}")
        
        lfp_memmap = np.memmap(file_path, dtype=data_type, mode='r', shape=shape, offset=0)
        print(f"  Successfully memory-mapped file: {file_path}")
        return lfp_memmap, sampling_rate, n_channels, n_samples, uv_scale_factor

    except FileNotFoundError:
        print(f"Error: File not found - {file_path} or {meta_path}")
        return None, None, 0, 0, None
    except KeyError as e:
        print(f"Error: Metadata key missing - {e}")
        return None, sampling_rate, n_channels, 0, None
    except ValueError as e:
        print(f"Error: Memmap shape/dtype error or metadata parsing error - {e}")
        return None, sampling_rate, n_channels, 0, None
    except Exception as e:
        print(f"Unexpected error loading LFP via memmap: {e}")
        import traceback
        traceback.print_exc()
        return None, None, 0, 0, None

# --- User-provided find_swr function ---
def find_swr(lfp, timestamps, fs=1250, thresholds=(2, 5), durations=(20, 40, 500),
             freq_range=(100, 250), noise=None):

    low_thresh, high_thresh = thresholds
    min_isi_ms, min_dur_ms, max_dur_ms = durations

    nyquist = fs / 2
    low_f = max(0.1, freq_range[0]) 
    high_f = min(nyquist - 0.1, freq_range[1]) 
    if low_f >= high_f:
        if high_f <= low_f : 
             return [], None, None 

    try:
        b, a = butter(3, [low_f / nyquist, high_f / nyquist], btype='band')
    except ValueError as e:
        return [], None, None
    filtered = filtfilt(b, a, lfp)
    rectified = filtered ** 2
    env_low_f = 0.5 
    env_high_f = 20 
    if env_low_f >= env_high_f or env_high_f >= nyquist: 
        try:
            b_env, a_env = butter(3, env_high_f / nyquist, btype='low')
        except ValueError as e:
            return [], None, None
    else:
        try: 
            b_env, a_env = butter(3, [env_low_f / nyquist, env_high_f / nyquist], btype='band')
        except ValueError as e:
            try:
                b_env, a_env = butter(3, env_high_f / nyquist, btype='low') 
            except ValueError as e_lp:
                return [], None, None

    envelope = filtfilt(b_env, a_env, rectified)
    if len(envelope) < 2:
        return [], filtered, None
    envelope_z = zscore(envelope)
    above_thresh = envelope_z > low_thresh
    rising = np.where(np.diff(above_thresh.astype(int)) == 1)[0] + 1 
    falling = np.where(np.diff(above_thresh.astype(int)) == -1)[0]

    if len(rising) == 0 or len(falling) == 0:
        return [], filtered, envelope_z 
    if rising[0] > falling[0]: 
        falling = falling[1:]
        if len(falling) == 0: return [], filtered, envelope_z
    if rising[-1] > falling[-1]: 
        rising = rising[:-1]
        if len(rising) == 0: return [], filtered, envelope_z
    if len(rising) != len(falling): 
        return [], filtered, envelope_z
    events = np.column_stack((rising, falling))
    if events.shape[0] == 0:
        return [], filtered, envelope_z
    merged = []
    if len(events) > 0:
        ripple = list(events[0])
        min_isi_samples = int(min_isi_ms / 1000 * fs)
        for start, end in events[1:]:
            if start - ripple[1] < min_isi_samples:
                ripple[1] = end
            else:
                merged.append(ripple)
                ripple = [start, end]
        merged.append(ripple)
        if not merged: 
             return [], filtered, envelope_z
        merged = np.array(merged)
    else:
        return [], filtered, envelope_z
    kept, peaks, peak_amps = [], [], []
    for start, end in merged:
        if start >= end: continue 
        segment_env = envelope_z[start:end]
        segment_filt = filtered[start:end]
        if len(segment_env) == 0: continue
        max_val_env = np.max(segment_env)
        if max_val_env >= high_thresh:
            peak_idx_in_segment = np.argmin(segment_filt) 
            peak_abs_idx = start + peak_idx_in_segment
            kept.append([start, end])
            peaks.append(peak_abs_idx)
            peak_amps.append(max_val_env) 
    if not kept:
        return [], filtered, envelope_z
    kept = np.array(kept)
    peaks = np.array(peaks)
    peak_amps = np.array(peak_amps)
    durations_s = (kept[:, 1] - kept[:, 0]) / fs
    mask = (durations_s >= min_dur_ms / 1000) & (durations_s <= max_dur_ms / 1000)
    kept = kept[mask]
    peaks = peaks[mask]
    peak_amps = peak_amps[mask]
    if not kept.any():
        return [], filtered, envelope_z
    if noise is not None and len(noise) == len(lfp):
        try:
            noise_filt_b, noise_filt_a = butter(3, [low_f / nyquist, high_f / nyquist], btype='band')
            noise_filtered = filtfilt(noise_filt_b, noise_filt_a, noise)
            noise_rectified = noise_filtered ** 2 
            noise_env = filtfilt(b_env, a_env, noise_rectified)
            if len(noise_env) < 2:
                 raise ValueError("Noise envelope too short for z-scoring")
            noise_z = zscore(noise_env)
            valid_indices = []
            for i, (start_event, end_event) in enumerate(kept): 
                if start_event >= end_event : continue
                noise_segment_z = noise_z[start_event:end_event]
                if len(noise_segment_z) == 0 or not np.any(noise_segment_z > high_thresh): 
                    valid_indices.append(i)
            kept = kept[valid_indices]
            peaks = peaks[valid_indices]
            peak_amps = peak_amps[valid_indices]
        except Exception as e:
            print(f"Error during noise rejection: {e}. Proceeding without noise rejection for this call.")
    if not kept.any():
        return [], filtered, envelope_z
    ripple_events = []
    for start_idx, peak_idx, end_idx, amp_val in zip(kept[:, 0], peaks, kept[:, 1], peak_amps):
        if start_idx < len(timestamps) and peak_idx < len(timestamps) and end_idx < len(timestamps):
            ripple_events.append({
                'start_sample': int(start_idx),
                'peak_sample': int(peak_idx),
                'end_sample': int(end_idx),
                'start_time': timestamps[start_idx],
                'peak_time': timestamps[peak_idx],
                'end_time': timestamps[end_idx],
                'peak_amplitude_z': amp_val, 
                'duration_ms': (end_idx - start_idx) / fs * 1000
            })
        else:
            warnings.warn(f"Event indices [{start_idx}, {peak_idx}, {end_idx}] out of bounds for timestamps array of length {len(timestamps)}. Skipping event.")

    return ripple_events, filtered, envelope_z

# --- Helper function to plot PSDs ---
def plot_channel_psds(lfp_memmap_obj, fs, uv_scale_factor, channel_indices_to_plot, channel_info_df, title_prefix=""):
    nperseg = int(2 * fs)
    fig = go.Figure()
    
    try:
        from matplotlib.cm import get_cmap
        try:
            cmap = plt.colormaps['tab10']
        except AttributeError: 
            cmap = get_cmap('tab10')
        plot_colors = [f'rgb({r*255},{g*255},{b*255})' for r,g,b,a_ in cmap(np.linspace(0, 1, len(channel_indices_to_plot)))] # Renamed 'a' to 'a_'
    except Exception as e:
        print(f"Matplotlib colormap error: {e}. Using default colors.")
        plot_colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

    for i, ch_idx in enumerate(channel_indices_to_plot):
        if not (0 <= ch_idx < lfp_memmap_obj.shape[1]):
            print(f"Warning: Channel index {ch_idx} out of bounds. Skipping PSD plot.")
            continue
        
        max_samples_for_psd = int(fs * 120) 
        num_samples_to_take = min(lfp_memmap_obj.shape[0], max_samples_for_psd)
        
        if lfp_memmap_obj.shape[0] > num_samples_to_take:
            start_idx_psd = (lfp_memmap_obj.shape[0] - num_samples_to_take) // 2
        else:
            start_idx_psd = 0
        
        signal_int16 = lfp_memmap_obj[start_idx_psd : start_idx_psd + num_samples_to_take, ch_idx]
        signal_float = signal_int16.astype(np.float64)
        
        y_label_psd_units = "ADC\u00b2/Hz"
        if uv_scale_factor is not None:
            signal_float *= uv_scale_factor
            y_label_psd_units = "\u00b5V\u00b2/Hz"

        try:
            current_nperseg = nperseg
            if len(signal_float) < nperseg:
                current_nperseg = len(signal_float)
                if current_nperseg > 0: # Only warn if we are actually adjusting
                     warnings.warn(f"Channel {ch_idx} has fewer samples ({len(signal_float)}) than nperseg ({nperseg}). Adjusting nperseg for this channel.")
            
            if current_nperseg == 0 : 
                print(f"Channel {ch_idx} has 0 samples for PSD. Skipping.")
                continue

            frequencies, psd = welch(signal_float, fs=fs, nperseg=current_nperseg)
        except ValueError as e:
            print(f"Could not compute PSD for channel {ch_idx}: {e}")
            continue

        psd_db = 10 * np.log10(psd + np.finfo(float).eps)
        mask = (frequencies >= 1) & (frequencies <= 300)

        ch_row = channel_info_df[channel_info_df['global_channel_index'] == ch_idx]
        ch_name = f"Ch {ch_idx} ({ch_row['acronym'].iloc[0]} S{ch_row['shank_index'].iloc[0]})" if not ch_row.empty else f'Ch {ch_idx}'

        fig.add_trace(go.Scatter(
            x=frequencies[mask],
            y=psd_db[mask],
            mode='lines',
            line=dict(width=1.5, color=plot_colors[i % len(plot_colors)]),
            name=ch_name
        ))

    fig.update_layout(
        title=f"{title_prefix}Welch PSD (1-300 Hz)",
        xaxis_title="Frequency (Hz)",
        yaxis_title=f"Power Spectral Density ({y_label_psd_units.replace('/Hz', ' dB')})",
        font=dict(size=12),
        template="plotly_white",
        showlegend=True,
        height=600,
        width=1000
    )
    fig.show()

# --- Helper function to plot example ripple events ---
def plot_lfp_with_ripples(raw_lfp_segment_scaled, filtered_lfp_segment, envelope_segment_z,
                          ripple_event_samples, fs, channel_name, title_suffix="", uv_scale_factor_present=True):
    time_axis = np.arange(len(raw_lfp_segment_scaled)) / fs * 1000 

    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(12, 8))
    channel_name_str = str(channel_name)
    fig.suptitle(f"Ripple Event on {channel_name_str} - {title_suffix}", fontsize=14)

    y_label_raw = "Raw LFP (\u00b5V)" if uv_scale_factor_present else "Raw LFP (ADC units)"
    axs[0].plot(time_axis, raw_lfp_segment_scaled, color='black', label='Raw LFP')
    axs[0].set_ylabel(y_label_raw)
    axs[0].grid(True, linestyle=':', alpha=0.7)

    y_label_filt = "Filtered LFP (\u00b5V)" if uv_scale_factor_present else "Filtered LFP (ADC units)"
    axs[1].plot(time_axis, filtered_lfp_segment, color='red', label='Ripple-band LFP')
    axs[1].set_ylabel(y_label_filt)
    axs[1].grid(True, linestyle=':', alpha=0.7)

    axs[2].plot(time_axis, envelope_segment_z, color='blue', label='Z-scored Envelope Power')
    axs[2].set_ylabel("Z-score Power Env.", color='blue')
    axs[2].tick_params(axis='y', labelcolor='blue')
    axs[2].grid(True, linestyle=':', alpha=0.7)
    
    event_start_ms = ripple_event_samples['start_sample_in_segment'] / fs * 1000
    event_peak_ms = ripple_event_samples['peak_sample_in_segment'] / fs * 1000
    event_end_ms = ripple_event_samples['end_sample_in_segment'] / fs * 1000

    vline_labels = ['Ripple Start', 'Ripple Peak (Trough)', 'Ripple End'] 
    vline_colors = ['green', 'magenta', 'purple']
    
    for ax_idx, ax in enumerate(axs):
        ax.axvline(event_start_ms, color=vline_colors[0], linestyle='--', lw=1, label=vline_labels[0] if ax_idx==0 else "_nolegend_")
        ax.axvline(event_peak_ms, color=vline_colors[1], linestyle='--', lw=1.5, label=vline_labels[1] if ax_idx==0 else "_nolegend_")
        ax.axvline(event_end_ms, color=vline_colors[2], linestyle='--', lw=1, label=vline_labels[2] if ax_idx==0 else "_nolegend_")
    
    axs[0].legend(loc='upper right') 
    axs[1].legend(loc='upper right')
    axs[2].legend(loc='upper left')

    axs[2].set_xlabel("Time (ms)")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# --- Main script logic ---
def main_ripple_analysis_and_visualization():
    root = Tk()
    root.withdraw() 
    root.attributes("-topmost", True)

    print("Starting Ripple Analysis and Visualization Script...")

    print("Please select the LFP binary file (*.lf.bin)...")
    lfp_binary_file_path_str = filedialog.askopenfilename(title="Select LFP Binary File (*.lf.bin)", filetypes=[("SpikeGLX LFP files", "*.lf.bin"), ("All files", "*.*")])
    if not lfp_binary_file_path_str: print("File selection cancelled. Exiting."); return
    lfp_binary_file_path = Path(lfp_binary_file_path_str)

    default_meta_filename = lfp_binary_file_path.with_suffix('.meta').name 
    print(f"Please select the corresponding Meta file (e.g., {default_meta_filename})...")
    meta_file_path_str = filedialog.askopenfilename(title="Select Meta File (*.meta)", initialdir=lfp_binary_file_path.parent, initialfile=default_meta_filename, filetypes=[("Meta files", "*.meta"), ("All files", "*.*")])
    if not meta_file_path_str: print("File selection cancelled. Exiting."); return
    meta_file_path = Path(meta_file_path_str)

    print("Please select the Channel Info CSV file...")
    channel_info_file_path_str = filedialog.askopenfilename(title="Select Channel Info CSV File", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
    if not channel_info_file_path_str: print("File selection cancelled. Exiting."); return
    channel_info_file_path = Path(channel_info_file_path_str)

    print("Please select the Output Directory...")
    output_dir_str = filedialog.askdirectory(title="Select Output Directory")
    if not output_dir_str: print("Directory selection cancelled. Exiting."); return
    output_dir = Path(output_dir_str)
    
    root.destroy()

    ripple_thresholds = (2, 4)
    ripple_durations = (20, 40, 500) 
    ripple_freq_range = (100, 250)
    target_areas = ["CA1", "CA3", "CA2"] 
    dg_area_name = "DG-mo"

    os.makedirs(output_dir, exist_ok=True)

    lfp_memmap_object, fs_orig, n_channels, n_samples, uv_scale_factor = load_lfp_data_memmap(lfp_binary_file_path, meta_file_path)

    if lfp_memmap_object is None or fs_orig is None:
        print("Failed to load LFP data. Exiting.")
        return
    
    fs = fs_orig 
    print(f"LFP data loaded: {n_channels} channels, {n_samples} samples. Effective FS: {fs:.2f} Hz.")
    uv_scale_factor_present = uv_scale_factor is not None
    if not uv_scale_factor_present:
        print("WARNING: No voltage scaling factor. LFP data is in ADC units.")

    # Load channel_df and ensure it's available before further use
    channel_df = None # Initialize to None
    try:
        print(f"Attempting to load channel info from: {channel_info_file_path}")
        channel_df = pd.read_csv(channel_info_file_path)
        print(f"Successfully read CSV, columns: {channel_df.columns.tolist()}")
        required_cols = ['global_channel_index', 'acronym', 'shank_index']
        if not all(col in channel_df.columns for col in required_cols):
            print(f"ERROR: Channel info CSV must contain columns: {required_cols}")
            if hasattr(lfp_memmap_object, '_mmap') and lfp_memmap_object._mmap is not None : 
                try: lfp_memmap_object._mmap.close()
                except Exception: pass
            return
        print("Channel info CSV validated successfully.")
    except FileNotFoundError: 
        print(f"ERROR: Channel info file not found at {channel_info_file_path}. Please check the path.")
        if hasattr(lfp_memmap_object, '_mmap') and lfp_memmap_object._mmap is not None : 
            try: lfp_memmap_object._mmap.close()
            except Exception: pass
        return 
    except Exception as e: 
        print(f"ERROR: Could not load or validate channel info: {e}")
        if hasattr(lfp_memmap_object, '_mmap') and lfp_memmap_object._mmap is not None : 
            try: lfp_memmap_object._mmap.close()
            except Exception: pass
        return 

    # Ensure channel_df is loaded before proceeding
    if channel_df is None:
        print("ERROR: channel_df was not loaded. Exiting.")
        if hasattr(lfp_memmap_object, '_mmap') and lfp_memmap_object._mmap is not None : 
            try: lfp_memmap_object._mmap.close()
            except Exception: pass
        return

    timestamps = np.arange(n_samples) / fs
    unique_shanks = sorted(channel_df['shank_index'].unique())
    print(f"Identified shanks from CSV: {unique_shanks}")

    selected_noise_channel_idx = None
    selected_reference_channels = {} 

    print("\n--- Channel Selection Phase ---")
    dg_channels_df = channel_df[channel_df['acronym'].str.contains(dg_area_name, case=False, na=False)]
    if not dg_channels_df.empty:
        dg_indices_to_plot = sorted(dg_channels_df['global_channel_index'].unique().tolist())
        print(f"\nDisplaying PSDs for {dg_area_name} channels (potential noise channels): {dg_indices_to_plot}")
        plot_channel_psds(lfp_memmap_object, fs, uv_scale_factor, dg_indices_to_plot, channel_df, title_prefix=f"{dg_area_name} Channels ")
        while selected_noise_channel_idx is None:
            try:
                val = input(f"Enter 'global_channel_index' for {dg_area_name} NOISE (from {dg_indices_to_plot}, or -1 to skip): ")
                idx = int(val)
                if idx == -1: selected_noise_channel_idx = -1; print("Skipping noise rejection."); break
                if idx in dg_indices_to_plot: selected_noise_channel_idx = idx; print(f"Noise channel: {idx}"); break
                else: print(f"Invalid index. Must be one of {dg_indices_to_plot} or -1.")
            except ValueError: print("Invalid input. Please enter a number.")
    else:
        print(f"No channels with acronym containing '{dg_area_name}' found. Skipping noise channel selection.")
        selected_noise_channel_idx = -1

    for shank in unique_shanks:
        print(f"\n--- Shank {shank} ---")
        for area in target_areas:
            area_shank_df = channel_df[(channel_df['acronym'].str.upper() == area.upper()) & (channel_df['shank_index'] == shank)]
            if area_shank_df.empty: print(f"No channels for Area {area} on Shank {shank}."); continue
            
            area_shank_indices = sorted(area_shank_df['global_channel_index'].unique().tolist())
            print(f"\nPSDs for Area {area}, Shank {shank}: {area_shank_indices}")
            plot_channel_psds(lfp_memmap_object, fs, uv_scale_factor, area_shank_indices, channel_df, title_prefix=f"Shank {shank} Area {area} ")
            while True:
                try:
                    val = input(f"Enter 'global_channel_index' for {area} S{shank} (from {area_shank_indices}, or -1 to skip): ")
                    idx = int(val)
                    if idx == -1: selected_reference_channels[(shank, area)] = -1; print(f"Skipping {area} S{shank}."); break
                    if idx in area_shank_indices: selected_reference_channels[(shank, area)] = idx; print(f"Ref for {area} S{shank}: {idx}"); break
                    else: print(f"Invalid index. Must be one of {area_shank_indices} or -1.")
                except ValueError: print("Invalid input. Please enter a number.")
    
    print("\n--- Channel selection complete. ---")

    all_shank_area_ripple_data = {}
    lfp_dg_noise_scaled = None
    if selected_noise_channel_idx not in [None, -1]:
        if 0 <= selected_noise_channel_idx < n_channels:
            lfp_dg_noise_scaled = lfp_memmap_object[:, selected_noise_channel_idx].astype(np.float64)
            if uv_scale_factor is not None:
                lfp_dg_noise_scaled *= uv_scale_factor
            print(f"Using LFP from channel {selected_noise_channel_idx} (scaled) for noise rejection.")
        else:
            print(f"Warn: Noise channel index {selected_noise_channel_idx} out of bounds. No noise rejection.")
            selected_noise_channel_idx = -1

    print("\n--- Ripple Detection and Visualization Phase ---")
    for (shank, area), ref_ch_idx in selected_reference_channels.items():
        if ref_ch_idx == -1: continue

        print(f"\nProcessing Shank {shank}, Area {area}, Ref Ch {ref_ch_idx}...")
        lfp_ref_trace_full = lfp_memmap_object[:, ref_ch_idx].astype(np.float64)
        if uv_scale_factor is not None:
            lfp_ref_trace_full *= uv_scale_factor
        
        ripple_events, filtered_lfp_full, envelope_z_full = find_swr(
            lfp=lfp_ref_trace_full, timestamps=timestamps, fs=fs,
            thresholds=ripple_thresholds, durations=ripple_durations,
            freq_range=ripple_freq_range, noise=lfp_dg_noise_scaled
        )
        shank_area_key = f"Shank{shank}_Area{area}_Chan{ref_ch_idx}"
        all_shank_area_ripple_data[shank_area_key] = ripple_events
        print(f"Found {len(ripple_events)} ripples for {shank_area_key}.")

        if ripple_events and filtered_lfp_full is not None and envelope_z_full is not None:
            print(f"Plotting example ripples for {shank_area_key}...")
            num_examples_to_plot = min(len(ripple_events), 3)
            indices_to_plot = np.linspace(0, len(ripple_events) - 1, num_examples_to_plot, dtype=int)

            for i, event_idx_to_plot in enumerate(indices_to_plot):
                event = ripple_events[event_idx_to_plot]
                start_s, peak_s, end_s = event['start_sample'], event['peak_sample'], event['end_sample']
                
                plot_padding_samples = int(0.1 * fs) 
                plot_start = max(0, start_s - plot_padding_samples)
                plot_end = min(n_samples, end_s + plot_padding_samples)

                if plot_start >= plot_end: continue

                raw_lfp_segment_int16 = lfp_memmap_object[plot_start:plot_end, ref_ch_idx]
                raw_lfp_segment_scaled = raw_lfp_segment_int16.astype(np.float64)
                if uv_scale_factor is not None:
                    raw_lfp_segment_scaled *= uv_scale_factor
                
                filt_lfp_seg = filtered_lfp_full[plot_start:plot_end]
                env_z_seg = envelope_z_full[plot_start:plot_end]
                
                event_samples_in_segment = {
                    'start_sample_in_segment': start_s - plot_start,
                    'peak_sample_in_segment': peak_s - plot_start,
                    'end_sample_in_segment': end_s - plot_start
                }
                plot_lfp_with_ripples(
                    raw_lfp_segment_scaled, filt_lfp_seg, env_z_seg,
                    event_samples_in_segment, fs,
                    channel_name=f"Ch {ref_ch_idx} ({area} S{shank})",
                    title_suffix=f"Event peak at {event['peak_time']:.3f}s (Example {i+1})",
                    uv_scale_factor_present=uv_scale_factor_present
                )
    
    print("\n--- Data Saving Phase ---")
    if not all_shank_area_ripple_data or all(not v for v in all_shank_area_ripple_data.values()):
        print("No ripple data was generated or detected to save.")
    else:
        save_choice = input("Save detected ripple data? (yes/no): ").strip().lower()
        if save_choice == 'yes':
            all_ripples_list_for_df = []
            for key, events in all_shank_area_ripple_data.items():
                try:
                    key_parts = key.replace("Shank","").replace("Area","").replace("Chan","")
                    shank_val, area_val, chan_val = key_parts.split("_")
                except ValueError:
                    print(f"Warning: Could not parse key '{key}' for shank/area/channel. Using placeholders.")
                    shank_val, area_val, chan_val = "Unknown", "Unknown", key.split("Chan")[-1] if "Chan" in key else "Unknown"

                for event_dict in events:
                    event_dict_copy = event_dict.copy()
                    event_dict_copy['shank'] = shank_val
                    event_dict_copy['area'] = area_val
                    event_dict_copy['reference_channel_idx'] = chan_val
                    all_ripples_list_for_df.append(event_dict_copy)
            
            if all_ripples_list_for_df:
                ripple_df = pd.DataFrame(all_ripples_list_for_df)
                base_filename = lfp_binary_file_path.stem.replace('.lf', '')
                
                csv_output_path = output_dir / f"{base_filename}_detected_ripple_events.csv"
                npy_output_path = output_dir / f"{base_filename}_detected_ripple_events_dict.npy"
                try:
                    ripple_df.to_csv(csv_output_path, index=False)
                    print(f"Ripple data saved to CSV: {csv_output_path}")
                    np.save(npy_output_path, all_shank_area_ripple_data, allow_pickle=True)
                    print(f"Ripple data (dict) saved to NPY: {npy_output_path}")
                except Exception as e:
                    print(f"Error saving ripple data: {e}")
            else:
                print("No ripples to save after compiling list for DataFrame.")
        else:
            print("Ripple data not saved.")

    if lfp_memmap_object is not None and hasattr(lfp_memmap_object, '_mmap') and lfp_memmap_object._mmap is not None:
        print("Closing LFP memmap object...")
        try:
            lfp_memmap_object._mmap.close()
            print("LFP memmap object closed.")
        except Exception as e:
            print(f"Error closing memmap: {e}")
            
    print("\nScript finished.")

if __name__ == "__main__":
    main_ripple_analysis_and_visualization()