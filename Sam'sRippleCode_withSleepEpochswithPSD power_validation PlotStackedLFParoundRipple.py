# -*- coding: utf-8 -*-
"""
Advanced Ripple Detection, Feature Extraction, and Visualization Script
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, welch, hilbert, stft
from scipy.stats import zscore, circmean
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm # For divergent colormap if needed
import os
from pathlib import Path
import warnings
import re 
from tkinter import Tk, filedialog
from DemoReadSGLXData.readSGLX import readMeta
# --- Configure Plotly & Matplotlib ---
pio.renderers.default = "browser"
try:
    matplotlib.use('Qt5Agg') 
    print("Matplotlib backend set to Qt5Agg.")
except ImportError:
    try: matplotlib.use('TkAgg'); print("Matplotlib backend set to TkAgg.")
    except ImportError: print("Warning: Could not set interactive Matplotlib backend. Plots might not display but will be saved.")
except Exception as e: print(f"Warning: Error setting Matplotlib backend: {e}. Plots might not display but will be saved.")

# --- Helper Functions (get_voltage_scaling_factor, load_lfp_data_memmap, load_sleep_and_epoch_data, find_swr, calculate_advanced_ripple_features, calculate_state_durations_in_epochs, plot_channel_psds, plot_lfp_with_ripples) ---
# (These functions are the same as in the previous version of the script you approved. For brevity, I'm not repeating their full code here, but they are assumed to be present and correct.)
# [[ Ensure the previous full definitions of these functions are here ]]
def get_voltage_scaling_factor(meta):
    try:
        v_max = float(meta['imAiRangeMax'])
        i_max = int(meta['imMaxInt'])
        lfp_gain = 250.0 
        probe_type_str = meta.get('imDatPrb_type', '0')
        probe_type = int(probe_type_str) if probe_type_str.isdigit() else 0
        if probe_type in [21, 24, 2013]: lfp_gain = 80.0
        elif probe_type == 0: lfp_gain = 250.0 
        elif probe_type != 0 and probe_type not in [21, 24, 2013]: 
             warnings.warn(f"LFP Gain: Probe type {probe_type} unknown. Using default LFP gain {lfp_gain}.")
        if i_max == 0 or lfp_gain == 0: 
            warnings.warn(f"Imax ({i_max}) or LFP gain ({lfp_gain}) is zero. Cannot scale voltage.")
            return None
        sf = (v_max * 1e6) / (i_max * lfp_gain)
        print(f"  uV scaling factor: {sf:.6f} (Probe: {probe_type}, Gain: {lfp_gain})")
        return sf
    except KeyError as e: warnings.warn(f"KeyError for voltage scaling: {e}. None returned."); return None
    except ValueError as e: warnings.warn(f"ValueError for voltage scaling: {e}. None returned."); return None
    except Exception as e: warnings.warn(f"Unexpected error in voltage scaling: {e}. None returned."); return None
            
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

def load_sleep_and_epoch_data(sleep_state_filepath, epoch_boundaries_filepath, fs_orig):
    sleep_lookup = None 
    epoch_samp = []
    if epoch_boundaries_filepath and epoch_boundaries_filepath.exists():
        try:
            ep_s = np.load(epoch_boundaries_filepath, allow_pickle=True)
            valid_ep_s = [tuple(ep) for ep in ep_s if isinstance(ep, (list, tuple, np.ndarray)) and len(ep) == 2 and all(isinstance(t, (int, float)) for t in ep) and ep[1] >= ep[0]]
            epoch_samp = [(int(s * fs_orig), int(e * fs_orig)) for s, e in valid_ep_s]
            print(f"Loaded {len(epoch_samp)} epochs (samples).")
        except Exception as e: print(f"Epoch load error {epoch_boundaries_filepath.name}: {e}")
    if sleep_state_filepath and sleep_state_filepath.exists():
        try:
            codes = np.load(sleep_state_filepath, allow_pickle=True)
            times_p, base_n = None, sleep_state_filepath.stem
            possible_tn = [ base_n.replace('_sleep_states', '_sleep_state_times') + ".npy", base_n.replace('_states', '_times') + ".npy",
                            re.sub(r'_sleep_states.*$', '_sleep_state_times.npy', sleep_state_filepath.name),
                            re.sub(r'_states.*$', '_times.npy', sleep_state_filepath.name) ]
            base_m = re.match(r"^(.*?)_sleep_states.*\.npy$", sleep_state_filepath.name)
            if base_m: possible_tn.append(f"{base_m.group(1)}_sleep_state_times_EMG.npy")
            for name_v in possible_tn:
                pot_p = sleep_state_filepath.parent / name_v
                if pot_p.exists(): times_p = pot_p; break
            if times_p and times_p.exists():
                times_s = np.load(times_p, allow_pickle=True)
                print(f"Loaded state codes ({len(codes)}) & times ({len(times_s)}) from {times_p.name}.")
                if len(codes) == len(times_s):
                    s_idx = np.argsort(times_s); times_ss, codes_s = times_s[s_idx], codes[s_idx]
                    sleep_lookup = {'times_sec': times_ss, 'codes': codes_s}
                    print(f"Sleep lookup created ({len(codes_s)} entries).")
                else: print("Warning: Sleep codes/times mismatch.")
            else: print("Warning: Sleep times file not found.")
        except Exception as e: print(f"Sleep load error: {e}"); sleep_lookup = None
    return sleep_lookup, epoch_samp

def find_swr(lfp, timestamps, fs=2500, thresholds=(2, 5), durations=(20, 40, 500),
             freq_range=(100, 250), noise=None):
    low_thresh, high_thresh = thresholds
    min_isi_ms, min_dur_ms, max_dur_ms = durations
    nyquist = fs / 2.0
    low_f = max(0.1, freq_range[0]); high_f = min(nyquist - 0.1, freq_range[1])
    if low_f >= high_f: return [], None, None, None
    try:
        b, a = butter(3, [low_f / nyquist, high_f / nyquist], btype='band')
        env_low_f_filt = max(0.1, 0.5); env_high_f_filt = min(nyquist - 0.1, 20)
        if env_low_f_filt >= env_high_f_filt: b_env, a_env = butter(3, env_high_f_filt / nyquist, btype='low')
        else: b_env, a_env = butter(3, [env_low_f_filt / nyquist, env_high_f_filt / nyquist], btype='band')
    except ValueError as e: print(f"Filter creation error in find_swr: {e}"); return [], None, None, None
    filtered = filtfilt(b, a, lfp); rectified = filtered ** 2
    envelope_raw = filtfilt(b_env, a_env, rectified) 
    if len(envelope_raw) < 2: return [], filtered, None, None
    envelope_z = zscore(envelope_raw)
    above_thresh = envelope_z > low_thresh
    rising = np.where(np.diff(above_thresh.astype(int)) == 1)[0] + 1
    falling = np.where(np.diff(above_thresh.astype(int)) == -1)[0]
    if not (len(rising) > 0 and len(falling) > 0): return [], filtered, envelope_raw, envelope_z
    if rising[0] > falling[0]: falling = falling[1:]
    if not (len(rising) > 0 and len(falling) > 0): return [], filtered, envelope_raw, envelope_z
    if rising[-1] > falling[-1]: rising = rising[:-1]
    if not (len(rising) == len(falling) and len(rising) > 0): return [], filtered, envelope_raw, envelope_z
    events = np.column_stack((rising, falling)); merged = []
    if len(events) > 0:
        ripple = list(events[0]); min_isi_samps = int(min_isi_ms / 1000 * fs)
        for start, end in events[1:]:
            if start - ripple[1] < min_isi_samps: ripple[1] = end
            else: merged.append(ripple); ripple = [start, end]
        merged.append(ripple)
        merged = np.array([r for r in merged if r[0] < r[1]]) 
    if len(merged) == 0: return [], filtered, envelope_raw, envelope_z
    kept, peaks_idx, peak_amps_z, peak_amps_power = [], [], [], []
    for start, end in merged:
        if start >= end or end > len(envelope_z): continue
        seg_env_z = envelope_z[start:end]; seg_env_raw = envelope_raw[start:end]; seg_filt = filtered[start:end]
        if len(seg_env_z) == 0: continue
        max_val_z = np.max(seg_env_z)
        if max_val_z >= high_thresh:
            peak_idx_rel = np.argmin(seg_filt); peak_abs = start + peak_idx_rel
            if peak_abs >= len(envelope_raw): continue 
            kept.append([start, end]); peaks_idx.append(peak_abs)
            peak_amps_z.append(max_val_z); peak_amps_power.append(envelope_raw[peak_abs]) 
    if not kept: return [], filtered, envelope_raw, envelope_z
    kept_arr = np.array(kept); peaks_arr = np.array(peaks_idx); p_amps_z_arr = np.array(peak_amps_z); p_amps_pow_arr = np.array(peak_amps_power)
    durations_s = (kept_arr[:, 1] - kept_arr[:, 0]) / fs
    mask = (durations_s >= min_dur_ms / 1000) & (durations_s <= max_dur_ms / 1000)
    final_kept = kept_arr[mask]; final_peaks = peaks_arr[mask]; final_amps_z = p_amps_z_arr[mask]; final_amps_pow = p_amps_pow_arr[mask]
    if not final_kept.any(): return [], filtered, envelope_raw, envelope_z
    if noise is not None and len(noise) == len(lfp):
        try:
            noise_filt = filtfilt(b, a, noise); noise_rect = noise_filt**2
            noise_env_raw = filtfilt(b_env, a_env, noise_rect)
            if len(noise_env_raw) < 2: raise ValueError("Noise envelope too short")
            noise_env_z = zscore(noise_env_raw)
            valid_event_indices = []
            for i, (start, end) in enumerate(final_kept):
                if start >= end or end > len(noise_env_z): continue
                if not np.any(noise_env_z[start:end] > high_thresh): valid_event_indices.append(i)
            final_kept = final_kept[valid_event_indices]; final_peaks = final_peaks[valid_event_indices]
            final_amps_z = final_amps_z[valid_event_indices]; final_amps_pow = final_amps_pow[valid_event_indices]
        except Exception as e: print(f"Noise rejection error: {e}.")
    ripple_events = []
    for (s, e), p, az, apow in zip(final_kept, final_peaks, final_amps_z, final_amps_pow):
        if s < len(timestamps) and p < len(timestamps) and e < len(timestamps):
            ripple_events.append({'start_sample': int(s), 'peak_sample': int(p), 'end_sample': int(e),
                                  'start_time': timestamps[s], 'peak_time': timestamps[p], 'end_time': timestamps[e],
                                  'peak_amplitude_z': az, 'peak_amplitude_power': apow,
                                  'duration_ms': (e - s) / fs * 1000 })
    return ripple_events, filtered, envelope_raw, envelope_z

def calculate_advanced_ripple_features(filtered_event_lfp, fs, ripple_freq_range_tuple):
    peak_freq, n_cycles, peak_phase = np.nan, np.nan, np.nan
    if len(filtered_event_lfp) < max(4, 2 * (fs / ripple_freq_range_tuple[0])): return peak_freq, n_cycles, peak_phase
    try: 
        nperseg_r = min(len(filtered_event_lfp), int(fs * 0.05)); nperseg_r = max(4, nperseg_r)
        if nperseg_r > len(filtered_event_lfp): nperseg_r = len(filtered_event_lfp)
        if nperseg_r > 0:
            freqs_r, psd_r = welch(filtered_event_lfp, fs=fs, nperseg=nperseg_r, noverlap=nperseg_r//2 if nperseg_r > 4 else 0)
            band_mask = (freqs_r >= ripple_freq_range_tuple[0]) & (freqs_r <= ripple_freq_range_tuple[1])
            if np.any(band_mask) and len(psd_r[band_mask]) > 0:
                peak_freq = freqs_r[band_mask][np.argmax(psd_r[band_mask])]
                duration_s = len(filtered_event_lfp) / fs; n_cycles = peak_freq * duration_s 
    except Exception as e: warnings.warn(f"Peak freq/cycles error: {e}")
    try: 
        analytic_signal = hilbert(filtered_event_lfp)
        instantaneous_phase = np.angle(analytic_signal)
        peak_idx_in_event = np.argmin(filtered_event_lfp) 
        peak_phase = instantaneous_phase[peak_idx_in_event]
    except Exception as e: warnings.warn(f"Peak phase error: {e}")
    return peak_freq, n_cycles, peak_phase

def calculate_state_durations_in_epochs(epoch_bounds_samples_list, sleep_lookup_dict, fs_orig, states_to_analyze_dict):
    state_duration_in_epoch = {state_code: {} for state_code in states_to_analyze_dict}
    if not sleep_lookup_dict: return state_duration_in_epoch
    state_times_sec, state_codes = sleep_lookup_dict['times_sec'], sleep_lookup_dict['codes']
    for epoch_idx, (ep_start_samp, ep_end_samp) in enumerate(epoch_bounds_samples_list):
        ep_start_sec, ep_end_sec = ep_start_samp / fs_orig, ep_end_samp / fs_orig
        for target_state_code in states_to_analyze_dict:
            current_epoch_state_duration_sec = 0.0
            relevant_indices = np.where((state_times_sec < ep_end_sec))[0]
            for i in relevant_indices:
                if state_codes[i] == target_state_code:
                    int_start_s = state_times_sec[i]
                    int_end_s = state_times_sec[i+1] if i + 1 < len(state_times_sec) else ep_end_sec 
                    overlap_start = max(int_start_s, ep_start_sec)
                    overlap_end = min(int_end_s, ep_end_sec)
                    if overlap_end > overlap_start:
                        current_epoch_state_duration_sec += (overlap_end - overlap_start)
            state_duration_in_epoch[target_state_code][epoch_idx] = current_epoch_state_duration_sec
    return state_duration_in_epoch

def plot_channel_psds(lfp_memmap_obj, fs, uv_scale_factor, 
                      channel_indices_to_plot, channel_info_df, 
                      ripple_freq_range_tuple, 
                      title_prefix=""):
    nperseg = int(2 * fs) 
    fig = go.Figure()
    try:
        from matplotlib.cm import get_cmap
        try: cmap = plt.colormaps['tab10']
        except AttributeError: cmap = get_cmap('tab10')
        plot_colors = [f'rgb({r*255},{g*255},{b*255})' for r,g,b,a_ in cmap(np.linspace(0, 1, len(channel_indices_to_plot)))]
    except Exception as e:
        print(f"Matplotlib colormap error: {e}. Using default colors.")
        plot_colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    for i, ch_idx in enumerate(channel_indices_to_plot):
        if not (0 <= ch_idx < lfp_memmap_obj.shape[1]):
            print(f"Warning: Ch {ch_idx} out of bounds. Skipping PSD plot.")
            continue
        max_s = int(fs * 120); num_s = min(lfp_memmap_obj.shape[0], max_s)
        start_s_psd = (lfp_memmap_obj.shape[0] - num_s) // 2 if lfp_memmap_obj.shape[0] > num_s else 0
        sig_i16 = lfp_memmap_obj[start_s_psd : start_s_psd + num_s, ch_idx]
        sig_f64 = sig_i16.astype(np.float64)
        y_units_linear = "ADC\u00b2/Hz" 
        if uv_scale_factor is not None:
            sig_f64 *= uv_scale_factor
            y_units_linear = "\u00b5V\u00b2/Hz"
        try:
            curr_nperseg = nperseg if len(sig_f64) >= nperseg else len(sig_f64)
            if curr_nperseg == 0 : 
                print(f"Ch {ch_idx} has 0 samples for PSD. Skipping PSD."); continue
            freqs, psd_v = welch(sig_f64, fs=fs, nperseg=curr_nperseg)
        except ValueError as e:
            print(f"Could not compute PSD for channel {ch_idx}: {e}"); continue
        avg_power_db_str = "N/A"
        if ripple_freq_range_tuple is not None and len(ripple_freq_range_tuple) == 2:
            band_mask_for_avg = (freqs >= ripple_freq_range_tuple[0]) & (freqs <= ripple_freq_range_tuple[1])
            if np.any(band_mask_for_avg) and len(psd_v[band_mask_for_avg]) > 0:
                avg_linear_power = np.mean(psd_v[band_mask_for_avg])
                avg_power_db = 10 * np.log10(avg_linear_power + np.finfo(float).eps)
                avg_power_db_str = f"{avg_power_db:.2f} dB"
            else: avg_power_db_str = "NoDataInBand"
        psd_db_plot = 10 * np.log10(psd_v + np.finfo(float).eps) 
        plot_mask = (freqs >= 1) & (freqs <= 300)
        ch_r = channel_info_df[channel_info_df['global_channel_index'] == ch_idx]
        ch_n_base = f"Ch {ch_idx} ({ch_r['acronym'].iloc[0]} S{ch_r['shank_index'].iloc[0]})" if not ch_r.empty else f'Ch {ch_idx}'
        ch_n_legend = f"{ch_n_base} (Avg Rip Pwr: {avg_power_db_str})" 
        fig.add_trace(go.Scatter(x=freqs[plot_mask], y=psd_db_plot[plot_mask], mode='lines', line=dict(width=1.5, color=plot_colors[i % len(plot_colors)]), name=ch_n_legend))
    y_axis_title = f"Power Spectral Density ({y_units_linear.replace('/Hz', ' dB')})" 
    fig.update_layout(title=f"{title_prefix}Welch PSD (1-300 Hz)", xaxis_title="Frequency (Hz)", yaxis_title=y_axis_title, font=dict(size=12), template="plotly_white", showlegend=True, legend_title_text='Channels (Avg Ripple Power in Band)', height=700, width=1200)
    fig.show()

def plot_lfp_with_ripples(raw_lfp_segment_scaled, filtered_lfp_segment, envelope_segment_z,
                          ripple_event_samples, fs, channel_name, title_suffix="", 
                          uv_scale_factor_present=True, output_plot_path=None):
    time_axis = np.arange(len(raw_lfp_segment_scaled)) / fs * 1000 
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(12, 8))
    channel_name_str = str(channel_name)
    fig.suptitle(f"Ripple Event on {channel_name_str} - {title_suffix}", fontsize=14)
    y_label_raw = "Raw LFP (\u00b5V)" if uv_scale_factor_present else "Raw LFP (ADC units)"
    axs[0].plot(time_axis, raw_lfp_segment_scaled, color='black', label='Raw LFP'); axs[0].set_ylabel(y_label_raw); axs[0].grid(True, linestyle=':', alpha=0.7)
    y_label_filt = "Filtered LFP (\u00b5V)" if uv_scale_factor_present else "Filtered LFP (ADC units)"
    axs[1].plot(time_axis, filtered_lfp_segment, color='red', label='Ripple-band LFP'); axs[1].set_ylabel(y_label_filt); axs[1].grid(True, linestyle=':', alpha=0.7)
    axs[2].plot(time_axis, envelope_segment_z, color='blue', label='Z-scored Envelope Power'); axs[2].set_ylabel("Z-score Power Env.", color='blue'); axs[2].tick_params(axis='y', labelcolor='blue'); axs[2].grid(True, linestyle=':', alpha=0.7)
    event_start_ms = ripple_event_samples['start_sample_in_segment'] / fs * 1000
    event_peak_ms = ripple_event_samples['peak_sample_in_segment'] / fs * 1000
    event_end_ms = ripple_event_samples['end_sample_in_segment'] / fs * 1000
    vline_labels = ['Ripple Start', 'Ripple Peak (Trough)', 'Ripple End']; vline_colors = ['green', 'magenta', 'purple']
    for ax_idx, ax_val in enumerate(axs):
        ax_val.axvline(event_start_ms, color=vline_colors[0], linestyle='--', lw=1, label=vline_labels[0] if ax_idx==0 else "_nolegend_")
        ax_val.axvline(event_peak_ms, color=vline_colors[1], linestyle='--', lw=1.5, label=vline_labels[1] if ax_idx==0 else "_nolegend_")
        ax_val.axvline(event_end_ms, color=vline_colors[2], linestyle='--', lw=1, label=vline_labels[2] if ax_idx==0 else "_nolegend_")
    axs[0].legend(loc='upper right'); axs[1].legend(loc='upper right'); axs[2].legend(loc='upper left')
    axs[2].set_xlabel("Time (ms)"); plt.tight_layout(rect=[0, 0, 1, 0.96])
    if output_plot_path:
        try: plt.savefig(output_plot_path); print(f"    Saved ripple plot to: {output_plot_path}")
        except Exception as e: print(f"    Error saving plot to {output_plot_path}: {e}")
    else: plt.show()
    plt.close(fig)

# --- NEW: Function to plot Aggregate Ripple-Locked LFP/Spectrogram ---
def plot_ripple_locked_lfp_spectrogram(
    region_shorthand_name, 
    channels_in_region_df_sorted, # DataFrame of channels for this region, sorted by depth
    all_ripple_peak_samples_abs,  # List of absolute peak samples of ripples for averaging
    lfp_memmap_obj, 
    fs, 
    uv_scale_factor,
    ref_channel_global_idx,       # Global index of the reference channel for this region/shank
    window_ms,                    # Total window width in ms (e.g., 400 for +/- 200ms)
    output_dir,
    output_filename_base,
    depth_column_name='y_coord',  # Column name for depth in channels_in_region_df_sorted
    spec_nperseg_mult=0.02,       # Spectrogram nperseg as a multiple of fs (e.g., 20ms window)
    spec_noverlap_mult=0.01,      # Spectrogram noverlap as a multiple of fs (e.g., 10ms overlap)
    spec_freq_max=300):           # Max frequency for spectrogram display
    
    print(f"  Generating aggregate LFP/Spectrogram for {region_shorthand_name} (Ref Ch: {ref_channel_global_idx})...")
    
    if not all_ripple_peak_samples_abs:
        print(f"    No ripple peaks provided for {region_shorthand_name}. Skipping plot.")
        return
    if channels_in_region_df_sorted.empty:
        print(f"    No channels provided for {region_shorthand_name}. Skipping plot.")
        return

    window_samples_half = int((window_ms / 2) / 1000 * fs)
    num_channels_to_plot = len(channels_in_region_df_sorted)
    
    all_avg_lfps = []
    all_avg_spectrograms = []
    
    # Spectrogram parameters
    nperseg = int(fs * spec_nperseg_mult)
    noverlap = int(fs * spec_noverlap_mult)
    if nperseg == 0 : nperseg = 128 # Fallback
    if noverlap >= nperseg : noverlap = nperseg // 2


    for _, ch_row in channels_in_region_df_sorted.iterrows():
        ch_idx = int(ch_row['global_channel_index'])
        
        lfp_segments_for_ch = []
        spectrogram_segments_for_ch = []
        
        for peak_samp_abs in all_ripple_peak_samples_abs:
            start_samp = peak_samp_abs - window_samples_half
            end_samp = peak_samp_abs + window_samples_half
            
            if start_samp < 0 or end_samp > lfp_memmap_obj.shape[0]:
                continue # Skip ripple if window is out of bounds
            
            lfp_segment_int16 = lfp_memmap_obj[start_samp:end_samp, ch_idx]
            lfp_segment_float = lfp_segment_int16.astype(np.float64)
            if uv_scale_factor is not None:
                lfp_segment_float *= uv_scale_factor
            lfp_segments_for_ch.append(lfp_segment_float)
            
            # Calculate Spectrogram for this segment
            if len(lfp_segment_float) >= nperseg : # Check if segment is long enough for STFT
                try:
                    f_spec, t_spec, Sxx = stft(lfp_segment_float, fs=fs, nperseg=nperseg, noverlap=noverlap)
                    # Keep only up to spec_freq_max
                    freq_mask_spec = f_spec <= spec_freq_max
                    Sxx_abs_db = 10 * np.log10(np.abs(Sxx[freq_mask_spec, :])**2 + np.finfo(float).eps)
                    spectrogram_segments_for_ch.append(Sxx_abs_db)
                except ValueError as e_stft:
                    warnings.warn(f"STFT error for ch {ch_idx}, peak {peak_samp_abs}: {e_stft}")
                    pass # Continue if one spectrogram fails

        if lfp_segments_for_ch:
            avg_lfp_for_ch = np.mean(np.array(lfp_segments_for_ch), axis=0)
            all_avg_lfps.append(avg_lfp_for_ch)
        else: # Add a dummy trace if no valid segments (should match window length)
            all_avg_lfps.append(np.full(window_samples_half * 2, np.nan))
            
        if spectrogram_segments_for_ch:
            # Ensure all spectrograms have the same time dimension before averaging
            min_t_len = min(s.shape[1] for s in spectrogram_segments_for_ch)
            spectrograms_to_avg = [s[:, :min_t_len] for s in spectrogram_segments_for_ch]
            avg_spectrogram_for_ch = np.mean(np.array(spectrograms_to_avg), axis=0)
            all_avg_spectrograms.append(avg_spectrogram_for_ch)
        else: # Add a dummy spectrogram
            # Attempt to get dummy f_spec, t_spec dimensions
            dummy_f_spec, dummy_t_spec, _ = stft(np.random.randn(window_samples_half * 2), fs=fs, nperseg=nperseg, noverlap=noverlap)
            dummy_freq_mask = dummy_f_spec <= spec_freq_max
            num_freq_bins = np.sum(dummy_freq_mask)
            num_time_bins = len(dummy_t_spec)
            all_avg_spectrograms.append(np.full((num_freq_bins, num_time_bins), np.nan))


    if not all_avg_lfps or not all_avg_spectrograms:
        print(f"    No data to plot for {region_shorthand_name}. Skipping."); return

    # Plotting
    fig, ax = plt.subplots(figsize=(10, max(6, num_channels_to_plot * 0.5))) # Adjust height based on num channels
    
    time_axis_ms = (np.arange(window_samples_half * 2) - window_samples_half) / fs * 1000
    
    # Determine overall min/max for spectrogram color scaling for consistency
    valid_specs = [s for s in all_avg_spectrograms if not np.all(np.isnan(s))]
    if not valid_specs: 
        print(f"    All spectrograms are NaN for {region_shorthand_name}. Skipping plot."); return

    vmin = np.nanmin([np.nanmin(s) for s in valid_specs])
    vmax = np.nanmax([np.nanmax(s) for s in valid_specs])
    if vmin == vmax: vmax = vmin + 1 # Avoid error if all values are same

    # Y-ticks will be channel depths or indices
    y_ticks_pos = []
    y_tick_labels = []

    for i, ch_row in enumerate(channels_in_region_df_sorted.iterrows()):
        idx, row_data = ch_row
        ch_global_idx = int(row_data['global_channel_index'])
        depth_val = row_data[depth_column_name]
        y_pos = i # Simple stack position
        y_ticks_pos.append(y_pos)
        y_tick_labels.append(f"{ch_global_idx} ({depth_val:.0f})")

        # Plot Spectrogram/Heatmap for this channel
        # We need extent for imshow: [time_min, time_max, y_bottom, y_top] for this channel's strip
        # Let each channel strip have a height of 1 unit on the plot
        spectrogram_data = all_avg_spectrograms[i]
        if not np.all(np.isnan(spectrogram_data)):
            # `t_spec` needs to be relative to the window
            # The number of time bins in spectrogram is spectrogram_data.shape[1]
            # The time axis for spectrogram extent should match time_axis_ms
            spec_time_bins = np.linspace(time_axis_ms[0], time_axis_ms[-1], spectrogram_data.shape[1])

            ax.imshow(spectrogram_data, aspect='auto', origin='lower',
                      extent=[spec_time_bins[0], spec_time_bins[-1], y_pos - 0.5, y_pos + 0.5],
                      cmap='viridis', vmin=vmin, vmax=vmax, interpolation='nearest') # or 'bilinear' for smoother

        # Overlay LFP trace
        lfp_trace = all_avg_lfps[i]
        if not np.all(np.isnan(lfp_trace)):
            # Scale LFP trace to fit nicely within its strip (e.g., +/- 0.4 units of the y_pos)
            lfp_min, lfp_max = np.nanmin(lfp_trace), np.nanmax(lfp_trace)
            if lfp_min != lfp_max:
                lfp_scaled = ((lfp_trace - lfp_min) / (lfp_max - lfp_min) - 0.5) * 0.8 # Scale to roughly +/- 0.4
            else:
                lfp_scaled = np.zeros_like(lfp_trace) # Flat line if no variation
            
            linewidth = 2.0 if ch_global_idx == ref_channel_global_idx else 0.75
            linecolor = 'white' if ch_global_idx == ref_channel_global_idx else 'black'
            ax.plot(time_axis_ms, y_pos + lfp_scaled, color=linecolor, linewidth=linewidth)

    ax.axvline(0, color='red', linestyle='--', linewidth=1, label='Ripple Peak (t=0)')
    ax.set_yticks(y_ticks_pos)
    ax.set_yticklabels(y_tick_labels)
    ax.set_ylabel(f"Channel (Global Index / {depth_column_name})")
    ax.set_xlabel("Time from Ripple Peak (ms)")
    ax.set_title(f"Ripple-Locked Avg LFP & Spectrogram - {region_shorthand_name} (Ref Ch: {ref_channel_global_idx})")
    ax.set_xlim(time_axis_ms[0], time_axis_ms[-1])
    ax.set_ylim(-0.5, num_channels_to_plot - 0.5) # Adjust y-limits for stacked plots
    
    # Add a colorbar
    # Need a mappable for the colorbar if not using a direct imshow for all
    # Create a dummy mappable for the colorbar if imshow was per strip
    if valid_specs: # only add colorbar if there was data
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
        sm.set_array([]) # You need to set a dummy array for the mappable
        cbar = fig.colorbar(sm, ax=ax, orientation='vertical', label='Spectrogram Power (dB or AU^2/Hz)') # Adjust label
        cbar.ax.tick_params(labelsize=8)

    plt.legend(loc='upper right', fontsize='small')
    plt.tight_layout()
    
    output_filepath = output_dir / f"{output_filename_base}.png"
    try:
        plt.savefig(output_filepath, dpi=300)
        print(f"    Saved aggregate LFP/Spectrogram plot to: {output_filepath}")
    except Exception as e:
        print(f"    Error saving aggregate plot to {output_filepath}: {e}")
    plt.close(fig)

# --- Main script logic ---
def main_ripple_analysis_and_visualization():
    root = Tk(); root.withdraw(); root.attributes("-topmost", True)
    print("Starting Advanced Ripple Analysis Script...")
    
    STATES_TO_ANALYZE = { 1: "NREM", 0: "Awake" } 
    
    lfp_bin_p = Path(filedialog.askopenfilename(title="Select LFP Binary File (*.lf.bin)", filetypes=[("SGLX LFP files", "*.lf.bin")]))
    if not lfp_bin_p.name: print("Cancelled."); return
    meta_p = Path(filedialog.askopenfilename(title="Select Meta File (*.meta)", initialdir=lfp_bin_p.parent, initialfile=lfp_bin_p.with_suffix('.meta').name, filetypes=[("Meta files", "*.meta")]))
    if not meta_p.name: print("Cancelled."); return
    chan_info_p = Path(filedialog.askopenfilename(title="Select Channel Info CSV", filetypes=[("CSV files", "*.csv")]))
    if not chan_info_p.name: print("Cancelled."); return
    sleep_p_str = filedialog.askopenfilename(title="Select Sleep State File (Optional, *_sleep_states.npy)", filetypes=[("NPY", "*.npy")], initialdir=lfp_bin_p.parent)
    sleep_p = Path(sleep_p_str) if sleep_p_str else None
    epoch_p_str = filedialog.askopenfilename(title="Select Epoch Boundaries File (Optional, *_epoch_boundaries.npy)", filetypes=[("NPY", "*.npy")], initialdir=lfp_bin_p.parent)
    epoch_p = Path(epoch_p_str) if epoch_p_str else None
    output_d = Path(filedialog.askdirectory(title="Select Output Directory"))
    if not output_d.name: print("Cancelled."); return
    root.destroy()

    ripple_thr = (2, 4); ripple_dur = (20, 40, 500); ripple_f_range = (100, 250) 
    target_areas_list = ["CA1", "CA3", "CA2"] 
    dg_area_name_str = "DG-mo"
    base_fname = lfp_bin_p.stem.replace('.lf', '')
    plots_d = output_d / f"{base_fname}_ripple_plots"; os.makedirs(plots_d, exist_ok=True)
    aggregate_plots_d = output_d / f"{base_fname}_aggregate_event_plots"; os.makedirs(aggregate_plots_d, exist_ok=True)

    lfp_mm, fs_o, n_ch, n_samp, uv_sf = load_lfp_data_memmap(lfp_bin_p, meta_p)
    if lfp_mm is None or fs_o is None: print("LFP load failed. Exiting."); return
    fs_eff = fs_o; uv_sf_present = uv_sf is not None
    if not uv_sf_present: print("WARNING: No voltage scaling. LFP in ADC units.")

    chan_df = None
    try:
        chan_df = pd.read_csv(chan_info_p)
        if not all(c in chan_df.columns for c in ['global_channel_index','acronym','shank_index']):
            raise ValueError("ChanInfo CSV missing required columns.")
        # UPDATED depth column check
        if 'depth' not in chan_df.columns and \
           'y_coord' not in chan_df.columns and \
           'ycoord_on_shank_um' not in chan_df.columns:
            warnings.warn("Channel Info CSV does not contain 'depth', 'y_coord', or 'ycoord_on_shank_um'. Aggregate plots needing depth will be skipped.")
            
    except Exception as e: 
        print(f"ChanInfo Error: {e}")
        if hasattr(lfp_mm, '_mmap') and lfp_mm._mmap: 
            try: lfp_mm._mmap.close() 
            except Exception: pass
        return
    
    sleep_lkp, epoch_b_samp = load_sleep_and_epoch_data(sleep_p, epoch_p, fs_eff)
    _get_state_func = None
    if sleep_lkp:
        def _get_state_internal(s_idx, fs_v, times_s_lkp, codes_lkp):
            t_s = s_idx / fs_v; idx = np.searchsorted(times_s_lkp, t_s, side='right') - 1
            return codes_lkp[idx] if 0 <= idx < len(codes_lkp) else -1 
        _get_state_func = lambda s, f_val: _get_state_internal(s,f_val,sleep_lkp['times_sec'],sleep_lkp['codes'])
        print("Sleep state lookup active.")
    else: print("No sleep data. Ripples not filtered by state unless STATES_TO_ANALYZE is empty/None.")

    if not epoch_b_samp: epoch_b_samp = [(0, n_samp)]; print("No epochs. Processing whole recording.")
    state_dur_in_ep = calculate_state_durations_in_epochs(epoch_b_samp, sleep_lkp, fs_eff, STATES_TO_ANALYZE)
    full_ts_abs = np.arange(n_samp) / fs_eff
    
    sel_noise_ch = None; sel_ref_chs = {}
    unique_shks = sorted(chan_df['shank_index'].unique())
    print("\n--- Channel Selection Phase ---")
    dg_df = chan_df[chan_df['acronym'].str.contains(dg_area_name_str, case=False, na=False)]
    if not dg_df.empty:
        dg_idxs = sorted(dg_df['global_channel_index'].unique().tolist())
        print(f"\nPSDs for {dg_area_name_str} channels: {dg_idxs}")
        plot_channel_psds(lfp_mm, fs_eff, uv_sf, dg_idxs, chan_df, ripple_f_range, title_prefix=f"{dg_area_name_str} Channels ")
        while sel_noise_ch is None:
            try:
                val = input(f"Enter 'global_channel_index' for {dg_area_name_str} NOISE ({dg_idxs}, or -1 skip): ")
                idx = int(val)
                if idx == -1: sel_noise_ch = -1; print("Skip noise reject."); break
                if idx in dg_idxs: sel_noise_ch = idx; print(f"Noise ch: {idx}"); break
                else: print(f"Invalid index ({dg_idxs} or -1).")
            except ValueError: print("Invalid input.")
    else: print(f"No '{dg_area_name_str}' channels. Skip noise select."); sel_noise_ch = -1
    
    for shk in unique_shks:
        print(f"\n--- Shank {shk} ---")
        for area_n in target_areas_list:
            area_shk_df = chan_df[(chan_df['acronym'].str.upper()==area_n.upper())&(chan_df['shank_index']==shk)]
            if area_shk_df.empty: print(f"No ch for {area_n} S{shk}."); continue
            area_shk_idxs = sorted(area_shk_df['global_channel_index'].unique().tolist())
            print(f"\nPSDs for {area_n} S{shk}: {area_shk_idxs}")
            plot_channel_psds(lfp_mm, fs_eff, uv_sf, area_shk_idxs, chan_df, ripple_f_range, title_prefix=f"S{shk} {area_n} ")
            while True:
                try:
                    val = input(f"Enter 'global_channel_index' for {area_n} S{shk} ({area_shk_idxs}, or -1 skip): ")
                    idx = int(val)
                    if idx == -1: sel_ref_chs[(shk,area_n)] = -1; print(f"Skip {area_n} S{shk}."); break
                    if idx in area_shk_idxs: sel_ref_chs[(shk,area_n)] = idx; print(f"Ref {area_n} S{shk}: {idx}"); break
                    else: print(f"Invalid index ({area_shk_idxs} or -1).")
                except ValueError: print("Invalid input.")
    print("\n--- Channel selection complete. ---")

    all_ripples_final_list = []
    noise_ep_scaled = None 
    states_iter = STATES_TO_ANALYZE if _get_state_func and STATES_TO_ANALYZE else {"AllData": -2}

    for st_code, st_name in states_iter.items():
        print(f"\nProcessing State: {st_name} (Code: {st_code})")
        for ep_idx, (ep_start, ep_end) in enumerate(epoch_b_samp):
            print(f"  Epoch {ep_idx} (Samples: {ep_start}-{ep_end})")
            if sel_noise_ch not in [None, -1] and 0 <= sel_noise_ch < n_ch:
                noise_ep_unsc = lfp_mm[ep_start:ep_end, sel_noise_ch]
                noise_ep_scaled = noise_ep_unsc.astype(np.float64) 
                if uv_sf: noise_ep_scaled *= uv_sf
            else: noise_ep_scaled = None
            for (shk, area_n), ref_ch in sel_ref_chs.items():
                if ref_ch == -1: continue
                print(f"    Detecting S{shk} {area_n} Ch{ref_ch}...")
                lfp_ep_unsc = lfp_mm[ep_start:ep_end, ref_ch]
                if len(lfp_ep_unsc) == 0: print(f"      Empty LFP seg Ch{ref_ch} ep {ep_idx}. Skip."); continue
                lfp_ep_scaled = lfp_ep_unsc.astype(np.float64)
                if uv_sf: lfp_ep_scaled *= uv_sf
                ts_ep_abs = full_ts_abs[ep_start:ep_end]
                if len(lfp_ep_scaled) != len(ts_ep_abs): warnings.warn(f"LFP/TS mismatch Ep{ep_idx} Ch{ref_ch}. Skip."); continue
                ep_rips, filt_lfp_ep, env_raw_ep, env_z_ep = find_swr(
                    lfp=lfp_ep_scaled, timestamps=ts_ep_abs, fs=fs_eff, thresholds=ripple_thr,
                    durations=ripple_dur, freq_range=ripple_f_range, noise=noise_ep_scaled)
                print(f"      Found {len(ep_rips)} raw rips in ep {ep_idx} for Ch {ref_ch}.")
                ep_rips_sorted = sorted(ep_rips, key=lambda x: x['peak_sample'])
                plotted_count = 0
                for i_rip, ev_dict in enumerate(ep_rips_sorted):
                    ev_dict['abs_start_sample'] = ep_start + ev_dict['start_sample']
                    ev_dict['abs_peak_sample'] = ep_start + ev_dict['peak_sample']
                    ev_dict['abs_end_sample'] = ep_start + ev_dict['end_sample']
                    ev_dict['epoch_rel_start_sample'] = ev_dict.pop('start_sample')
                    ev_dict['epoch_rel_peak_sample'] = ev_dict.pop('peak_sample')
                    ev_dict['epoch_rel_end_sample'] = ev_dict.pop('end_sample')
                    curr_ev_st_code = _get_state_func(ev_dict['abs_peak_sample'], fs_eff) if _get_state_func else -2
                    if st_name == "AllData" or curr_ev_st_code == st_code:
                        ev_filt_lfp = filt_lfp_ep[ev_dict['epoch_rel_start_sample'] : ev_dict['epoch_rel_end_sample']+1]
                        if len(ev_filt_lfp) > 0 :
                             pk_f, n_cyc, pk_ph = calculate_advanced_ripple_features(ev_filt_lfp, fs_eff, ripple_f_range)
                        else: pk_f, n_cyc, pk_ph = np.nan, np.nan, np.nan
                        ev_dict['peak_frequency_hz'] = pk_f; ev_dict['n_cycles'] = n_cyc; ev_dict['peak_phase_rad'] = pk_ph
                        ev_dict['iri_s'] = (ev_dict['peak_time'] - ep_rips_sorted[i_rip-1]['peak_time']) if i_rip > 0 else np.nan
                        ev_to_store = ev_dict.copy()
                        ev_to_store.update({'shank': shk, 'area': area_n, 'reference_channel_idx': ref_ch, 
                                            'epoch_idx': ep_idx, 'state_code': curr_ev_st_code, 
                                            'state_name': STATES_TO_ANALYZE.get(curr_ev_st_code, "Unknown") if _get_state_func else "AllData"})
                        all_ripples_final_list.append(ev_to_store)
                        if plotted_count < 3 and filt_lfp_ep is not None and env_z_ep is not None and len(ev_filt_lfp) > 0 :
                            plot_pad_s = int(0.1 * fs_eff)
                            plot_s_ep = max(0, ev_dict['epoch_rel_start_sample'] - plot_pad_s)
                            plot_e_ep = min(len(lfp_ep_scaled), ev_dict['epoch_rel_end_sample'] + plot_pad_s)
                            if plot_s_ep >= plot_e_ep: continue
                            raw_lfp_plot_unsc = lfp_mm[ep_start + plot_s_ep : ep_start + plot_e_ep, ref_ch]
                            raw_lfp_plot_sc = raw_lfp_plot_unsc.astype(np.float64)
                            if uv_sf: raw_lfp_plot_sc *= uv_sf
                            filt_plot_seg = filt_lfp_ep[plot_s_ep:plot_e_ep]; env_z_plot_seg = env_z_ep[plot_s_ep:plot_e_ep]
                            ev_samps_for_plot = {'start_sample_in_segment': ev_dict['epoch_rel_start_sample'] - plot_s_ep,
                                                 'peak_sample_in_segment': ev_dict['epoch_rel_peak_sample'] - plot_s_ep,
                                                 'end_sample_in_segment': ev_dict['epoch_rel_end_sample'] - plot_s_ep}
                            plot_fname = f"{base_fname}_State{ev_to_store['state_name']}_Ep{ep_idx}_Sh{shk}A{area_n}C{ref_ch}_PeakS{ev_dict['abs_peak_sample']}.png"
                            plot_fpath = plots_d / plot_fname
                            plot_lfp_with_ripples(raw_lfp_plot_sc, filt_plot_seg, env_z_plot_seg, ev_samps_for_plot, 
                                                  fs_eff, f"Ch {ref_ch} ({area_n} S{shk})", 
                                                  f"State:{ev_to_store['state_name']},Ep{ep_idx},Peak(abs):{ev_dict['abs_peak_sample']}",
                                                  uv_sf_present, output_plot_path=plot_fpath)
                            plotted_count += 1
    
    # --- Aggregate LFP/Spectrogram Plotting ---
    if all_ripples_final_list:
        print("\n--- Generating Aggregate LFP/Spectrogram Plots ---")
        # Ensure chan_df is available
        if chan_df is None:
            print("Channel DataFrame (chan_df) is not available. Skipping aggregate plots.")
        else:
            ripple_df_for_agg_plot = pd.DataFrame(all_ripples_final_list)
            # Ensure necessary columns exist before groupby
            group_cols_agg = ['state_name', 'shank', 'area', 'reference_channel_idx']
            if not all(col in ripple_df_for_agg_plot.columns for col in group_cols_agg):
                print(f"Warning: One or more grouping columns ({group_cols_agg}) not in ripple data. Cannot generate aggregate plots.")
            else:
                for (st_name_agg, shk_agg, area_agg, ref_ch_agg), group_df in ripple_df_for_agg_plot.groupby(group_cols_agg):
                    # ref_ch_agg is already an int from the ripple data
                    print(f"  Preparing aggregate plot for State: {st_name_agg}, Shank: {shk_agg}, Area: {area_agg} (Ref Ch: {ref_ch_agg})")

                    current_shank_area_channels_df = chan_df[
                        (chan_df['shank_index'] == shk_agg) & # Ensure shk_agg is correct type for comparison if chan_df['shank_index'] is not int
                        (chan_df['acronym'].str.upper() == area_agg.upper())
                    ]

                    depth_col_name = None
                    if 'depth' in current_shank_area_channels_df.columns: depth_col_name = 'depth'
                    elif 'y_coord' in current_shank_area_channels_df.columns: depth_col_name = 'y_coord'
                    elif 'ycoord_on_shank_um' in current_shank_area_channels_df.columns: depth_col_name = 'ycoord_on_shank_um' # YOUR COLUMN
                    
                    if not depth_col_name:
                        warnings.warn(f"No depth column for S{shk_agg} A{area_agg}. Skipping aggregate plot for this group.")
                        continue
                    
                    channels_to_plot_df_sorted = current_shank_area_channels_df.sort_values(by=depth_col_name, ascending=False) 
                    if channels_to_plot_df_sorted.empty :
                        warnings.warn(f"No channels found for plotting in S{shk_agg} A{area_agg} after depth sort. Skipping aggregate plot.")
                        continue
                        
                    ripple_peak_samples_for_group = group_df['abs_peak_sample'].tolist()
                    if not ripple_peak_samples_for_group:
                        print(f"    No ripples for S{shk_agg} A{area_agg} State {st_name_agg}. Skipping aggregate plot.")
                        continue

                    agg_plot_filename_base = f"{base_fname}_Aggregate_State{st_name_agg}_Sh{shk_agg}A{area_agg}C{ref_ch_agg}"
                    
                    plot_ripple_locked_lfp_spectrogram(
                        region_shorthand_name=f"S{shk_agg}-{area_agg}",
                        channels_in_region_df_sorted=channels_to_plot_df_sorted,
                        all_ripple_peak_samples_abs=ripple_peak_samples_for_group,
                        lfp_memmap_obj=lfp_mm, fs=fs_eff, uv_scale_factor=uv_sf,
                        ref_channel_global_idx=ref_ch_agg, window_ms=400, 
                        output_dir=aggregate_plots_d, output_filename_base=agg_plot_filename_base,
                        depth_column_name=depth_col_name, spec_freq_max=250 
                    )
    
    print("\n--- Data Saving ---")
    # ... (Data saving part remains the same as your last approved version) ...
    if not all_ripples_final_list: print("No ripples to save."); 
    else:
        save_opt = input("Save detected ripple data & summary? (yes/no): ").strip().lower()
        if save_opt == 'yes':
            ripple_df_out = pd.DataFrame(all_ripples_final_list)
            csv_path = output_d / f"{base_fname}_ripple_events_detailed.csv"
            npy_path = output_d / f"{base_fname}_ripple_events_detailed_list.npy"
            try:
                ripple_df_out_sorted = ripple_df_out.sort_values(by=['state_name','epoch_idx','reference_channel_idx','abs_peak_sample']).reset_index(drop=True)
                ripple_df_out_sorted.to_csv(csv_path, index=False, float_format='%.4f')
                print(f"Detailed ripples saved to CSV: {csv_path}")
                np.save(npy_path, all_ripples_final_list, allow_pickle=True)
                print(f"Detailed ripples (list of dicts) saved to NPY: {npy_path}")
                summary_data = []
                group_by_cols = ['state_name', 'state_code', 'epoch_idx', 'shank', 'area', 'reference_channel_idx']
                # Ensure all groupby columns exist
                for col in group_by_cols:
                    if col not in ripple_df_out.columns:
                        print(f"Critical Error: Groupby column '{col}' missing from ripple_df_out. Cannot create summary.")
                        # Decide how to handle: skip summary, or try with available cols, or error out.
                        # For now, let's try to proceed if some non-essential ones are missing but warn heavily.
                        # If essential like 'state_name' is missing, it will fail.
                        # This should ideally not happen if data population is correct.
                
                # Attempt groupby, catch error if columns still an issue after data population
                try:
                    grouped = ripple_df_out.groupby(group_by_cols)
                except KeyError as ke:
                    print(f"KeyError during groupby for summary: {ke}. Some essential grouping columns might be missing. Cannot create summary.")
                    grouped = None # Ensure grouped is None if it fails

                if grouped:
                    for name, group in grouped:
                        state_n, state_c, ep_i, shk_v, area_v, ref_c_v = name # Ensure name has 6 elements
                        dur_of_state_in_ep = state_dur_in_ep.get(state_c, {}).get(ep_i, np.nan)
                        num_rips = len(group)
                        rate_hz = num_rips / dur_of_state_in_ep if pd.notna(dur_of_state_in_ep) and dur_of_state_in_ep > 0 else 0.0
                        valid_ph = group['peak_phase_rad'].dropna()
                        avg_pk_ph = circmean(valid_ph, high=np.pi, low=-np.pi) if len(valid_ph) > 0 else np.nan
                        summary_row = {'state_name': state_n, 'state_code': state_c, 'epoch_idx': ep_i,
                                    'shank': shk_v, 'area': area_v, 'reference_channel_idx': ref_c_v,
                                    'num_ripples': num_rips, 'duration_of_state_in_epoch_s': dur_of_state_in_ep,
                                    'ripple_rate_hz': rate_hz, 'avg_duration_ms': group['duration_ms'].mean(),
                                    'avg_peak_amplitude_z': group['peak_amplitude_z'].mean(),
                                    'avg_peak_amplitude_power': group['peak_amplitude_power'].mean(),
                                    'avg_peak_frequency_hz': group['peak_frequency_hz'].mean(),
                                    'avg_n_cycles': group['n_cycles'].mean(),
                                    'avg_iri_s': group['iri_s'].mean(), 
                                    'resultant_peak_phase_rad': avg_pk_ph }
                        summary_data.append(summary_row)
                if summary_data:
                    summary_df = pd.DataFrame(summary_data)
                    summary_csv_path = output_d / f"{base_fname}_ripple_summary_stats.csv"
                    summary_df.to_csv(summary_csv_path, index=False, float_format='%.4f')
                    print(f"Summary stats saved to CSV: {summary_csv_path}")
                else: print("No data for summary stats CSV (possibly due to grouping issues or no ripples).")
            except Exception as e: print(f"Error saving ripple data/summary: {e}"); traceback.print_exc()
        else: print("Ripple data not saved.")

    if lfp_mm is not None and hasattr(lfp_mm, '_mmap') and lfp_mm._mmap is not None:
        print("Closing LFP memmap..."); 
        try: lfp_mm._mmap.close(); print("LFP memmap closed.")
        except Exception as e: print(f"Error closing memmap: {e}")
    print("\nScript finished.")

if __name__ == "__main__":
    main_ripple_analysis_and_visualization()