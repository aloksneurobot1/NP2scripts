# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 18:13:47 2025

Advanced Ripple Detection, Feature Extraction, and Visualization Script
(Added Optional Notch Filtering 60 Hz and 180 Hz only)
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, welch, hilbert, iirnotch 
from scipy.stats import zscore, circmean
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib
import matplotlib.pyplot as plt
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
    print("Matplotlib backend set to Qt5Agg (for individual ripple LFP traces).")
except ImportError:
    try: matplotlib.use('TkAgg'); print("Matplotlib backend set to TkAgg (for individual ripple LFP traces).")
    except ImportError: print("Warning: Could not set interactive Matplotlib backend. Plots might not display but will be saved.")
except Exception as e: print(f"Warning: Error setting Matplotlib backend: {e}. Plots might not display but will be saved.")


# --- Helper Functions ---
def get_voltage_scaling_factor(meta):
    try:
        v_max = float(meta['imAiRangeMax'])
        i_max = int(meta['imMaxInt'])
        lfp_gain = 80 # for NP2 4 shank probe gain is 80 , NP1 probe single shank check steinmetz lab github for details or spikeglx help 
        sf = (v_max * 1e6) / (i_max * lfp_gain)
        return sf
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

# --- NEW: Notch Filter Function ---
def apply_notch_filters(data_segment, fs, notch_freqs_list, Q_factor):
    """Applies a series of notch filters to the data segment."""
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

def find_swr(lfp, timestamps, fs=1000, thresholds=(2, 5), durations=(20, 40, 500),
             freq_range=(100, 250), noise=None):
    # ... (find_swr definition remains the same as the version that returns noise_excluded_count) ...
    low_thresh, high_thresh = thresholds
    min_isi_ms, min_dur_ms, max_dur_ms = durations
    nyquist = fs / 2.0
    low_f = max(0.1, freq_range[0]); high_f = min(nyquist - 0.1, freq_range[1])
    if low_f >= high_f: return [], None, None, None, 0 
    try:
        b, a = butter(3, [low_f / nyquist, high_f / nyquist], btype='band')
        env_low_f_filt = max(0.1, 0.5); env_high_f_filt = min(nyquist - 0.1, 20)
        if env_low_f_filt >= env_high_f_filt: b_env, a_env = butter(3, env_high_f_filt / nyquist, btype='low')
        else: b_env, a_env = butter(3, [env_low_f_filt / nyquist, env_high_f_filt / nyquist], btype='band')
    except ValueError as e: print(f"Filter creation error in find_swr: {e}"); return [], None, None, None, 0
    filtered = filtfilt(b, a, lfp); rectified = filtered ** 2
    envelope_raw = filtfilt(b_env, a_env, rectified) 
    if len(envelope_raw) < 2: return [], filtered, None, None, 0
    envelope_z = zscore(envelope_raw)
    above_thresh = envelope_z > low_thresh
    rising = np.where(np.diff(above_thresh.astype(int)) == 1)[0] + 1
    falling = np.where(np.diff(above_thresh.astype(int)) == -1)[0]
    if not (len(rising) > 0 and len(falling) > 0): return [], filtered, envelope_raw, envelope_z, 0
    if rising[0] > falling[0]: falling = falling[1:]
    if not (len(rising) > 0 and len(falling) > 0): return [], filtered, envelope_raw, envelope_z, 0
    if rising[-1] > falling[-1]: rising = rising[:-1]
    if not (len(rising) == len(falling) and len(rising) > 0): return [], filtered, envelope_raw, envelope_z, 0
    events = np.column_stack((rising, falling)); merged = []
    if len(events) > 0:
        ripple = list(events[0]); min_isi_samps = int(min_isi_ms / 1000 * fs)
        for start, end in events[1:]:
            if start - ripple[1] < min_isi_samps: ripple[1] = end
            else: merged.append(ripple); ripple = [start, end]
        merged.append(ripple)
        merged = np.array([r for r in merged if r[0] < r[1]]) 
    if len(merged) == 0: return [], filtered, envelope_raw, envelope_z, 0
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
    if not kept: return [], filtered, envelope_raw, envelope_z, 0
    kept_arr = np.array(kept); peaks_arr = np.array(peaks_idx); p_amps_z_arr = np.array(peak_amps_z); p_amps_pow_arr = np.array(peak_amps_power)
    durations_s = (kept_arr[:, 1] - kept_arr[:, 0]) / fs
    mask = (durations_s >= min_dur_ms / 1000) & (durations_s <= max_dur_ms / 1000)
    final_kept_before_noise_rejection = kept_arr[mask] 
    final_peaks = peaks_arr[mask]; final_amps_z = p_amps_z_arr[mask]; final_amps_pow = p_amps_pow_arr[mask]
    if not final_kept_before_noise_rejection.any(): return [], filtered, envelope_raw, envelope_z, 0
    noise_excluded_count = 0 
    final_kept_after_noise_rejection = final_kept_before_noise_rejection.copy() 
    if noise is not None and len(noise) == len(lfp):
        try:
            noise_filt = filtfilt(b, a, noise); noise_rect = noise_filt**2
            noise_env_raw = filtfilt(b_env, a_env, noise_rect)
            if len(noise_env_raw) < 2: raise ValueError("Noise envelope too short")
            noise_env_z = zscore(noise_env_raw)
            valid_event_indices_after_noise = []
            for i, (start, end) in enumerate(final_kept_before_noise_rejection): 
                if start >= end or end > len(noise_env_z): continue 
                if not np.any(noise_env_z[start:end] > high_thresh): 
                    valid_event_indices_after_noise.append(i)
                else: noise_excluded_count += 1 
            final_kept_after_noise_rejection = final_kept_before_noise_rejection[valid_event_indices_after_noise]
            final_peaks = final_peaks[valid_event_indices_after_noise]
            final_amps_z = final_amps_z[valid_event_indices_after_noise]
            final_amps_pow = final_amps_pow[valid_event_indices_after_noise]
        except Exception as e: 
            print(f"Noise rejection error: {e}.")
            noise_excluded_count = 0 
    ripple_events = []
    if final_kept_after_noise_rejection.any(): 
        for (s, e), p, az, apow in zip(final_kept_after_noise_rejection, final_peaks, final_amps_z, final_amps_pow):
            if s < len(timestamps) and p < len(timestamps) and e < len(timestamps):
                ripple_events.append({'start_sample': int(s), 'peak_sample': int(p), 'end_sample': int(e),
                                      'start_time': timestamps[s], 'peak_time': timestamps[p], 'end_time': timestamps[e],
                                      'peak_amplitude_z': az, 'peak_amplitude_power': apow,
                                      'duration_ms': (e - s) / fs * 1000 })
    return ripple_events, filtered, envelope_raw, envelope_z, noise_excluded_count


def calculate_advanced_ripple_features(filtered_event_lfp, fs, ripple_freq_range_tuple):
    # ... (same as before) ...
    peak_freq, n_cycles, peak_phase = np.nan, np.nan, np.nan
    min_samples_for_welch = max(8, int(fs / ripple_freq_range_tuple[0] * 1.5)) 
    if len(filtered_event_lfp) < min_samples_for_welch : 
        return peak_freq, n_cycles, peak_phase
    try: 
        nperseg_r = min(len(filtered_event_lfp), max(16, int(fs * 0.025))) 
        if nperseg_r > len(filtered_event_lfp): nperseg_r = len(filtered_event_lfp)
        noverlap_r = nperseg_r // 2
        if nperseg_r <= 0 : raise ValueError(f"Invalid nperseg_r ({nperseg_r})")
        freqs_r, psd_r = welch(filtered_event_lfp, fs=fs, nperseg=nperseg_r, noverlap=noverlap_r, detrend=False)
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
    # ... (same as before) ...
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

# --- MODIFIED plot_channel_psds function to incorporate notch filter option if active ---
def plot_channel_psds(lfp_memmap_obj, fs, uv_scale_factor, 
                      channel_indices_to_plot, channel_info_df, 
                      ripple_freq_range_for_avg, 
                      title_prefix="",
                      apply_notch_filter_flag=False, # New flag
                      notch_freqs=None, Q_notch=None): # New notch params
    nperseg = int(2 * fs) 
    fig = go.Figure()
    # ... (Colormap logic same as before) ...
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
        
        if uv_scale_factor is not None: 
            sig_f64 *= uv_scale_factor
        
        # Apply Notch Filter if flag is True
        if apply_notch_filter_flag and notch_freqs and Q_notch:
            # print(f"    Applying notch filter to LFP segment for PSD Ch {ch_idx}...") # Optional print
            sig_f64 = apply_notch_filters(sig_f64, fs, notch_freqs, Q_notch)

        y_units_linear = "ADC\u00b2/Hz" if uv_scale_factor is None else "\u00b5V\u00b2/Hz"
        
        try:
            current_nperseg = nperseg if len(sig_f64) >= nperseg else len(sig_f64)
            if current_nperseg == 0 : 
                print(f"Ch {ch_idx} has 0 samples for PSD. Skipping PSD."); continue
            frequencies, psd_v = welch(sig_f64, fs=fs, nperseg=current_nperseg) 
        except ValueError as e:
            print(f"Could not compute PSD for channel {ch_idx}: {e}"); continue

        psd_db_full_spectrum = 10 * np.log10(psd_v + np.finfo(float).eps) 
        avg_power_db_str_ripple_band = "N/A"
        if ripple_freq_range_for_avg is not None and len(ripple_freq_range_for_avg) == 2:
            band_mask_for_avg = (frequencies >= ripple_freq_range_for_avg[0]) & (frequencies <= ripple_freq_range_for_avg[1])
            if np.any(band_mask_for_avg) and len(psd_v[band_mask_for_avg]) > 0:
                avg_linear_power_rip_band = np.mean(psd_v[band_mask_for_avg])
                avg_db_power_rip_band = 10 * np.log10(avg_linear_power_rip_band + np.finfo(float).eps)
                avg_power_db_str_ripple_band = f"{avg_db_power_rip_band:.2f} dB"
            else: avg_power_db_str_ripple_band = "NoDataInBand"
        
        if len(psd_db_full_spectrum) >= 2: psd_db_zscored = zscore(psd_db_full_spectrum)
        else: psd_db_zscored = psd_db_full_spectrum 
        plot_mask_1_200Hz = (frequencies >= 1) & (frequencies <= 200)
        ch_r = channel_info_df[channel_info_df['global_channel_index'] == ch_idx]
        ch_n_base = f"Ch {ch_idx} ({ch_r['acronym'].iloc[0]} S{ch_r['shank_index'].iloc[0]})" if not ch_r.empty else f'Ch {ch_idx}'
        ch_n_legend = f"{ch_n_base} (Avg Pwr {ripple_freq_range_for_avg[0]}-{ripple_freq_range_for_avg[1]}Hz: {avg_power_db_str_ripple_band})"
        
        fig.add_trace(go.Scatter(x=frequencies[plot_mask_1_200Hz], y=psd_db_zscored[plot_mask_1_200Hz], mode='lines', 
                                 line=dict(width=1, color=plot_colors[i % len(plot_colors)]), 
                                 opacity=0.7, name=ch_n_legend))
    
    y_axis_title = "Z-scored Power Spectral Density (dB)" 
    fig.update_layout(title=f"{title_prefix}Welch PSD (1-200 Hz)", xaxis_title="Frequency (Hz)", 
                      yaxis_title=y_axis_title, font=dict(size=12), 
                      template="plotly_white", showlegend=True, 
                      legend_title_text=f'Channels (Avg Power in {ripple_freq_range_for_avg[0]}-{ripple_freq_range_for_avg[1]}Hz Band)', 
                      height=700, width=1200)
    fig.show()

def plot_lfp_with_ripples(raw_lfp_segment_scaled, filtered_lfp_segment, envelope_segment_z,
                          ripple_event_samples, fs, channel_name, title_suffix="", 
                          uv_scale_factor_present=True, output_plot_path=None):
    # ... (plot_lfp_with_ripples definition - unchanged) ...
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
        try: plt.savefig(output_plot_path); 
        except Exception as e: print(f"    Error saving plot to {output_plot_path}: {e}")
    else: plt.show()
    plt.close(fig)

# --- RENAMED: Function to plot Ripple-Locked Average LFP Stack with Voltage Heatmap (using Plotly) ---
def plot_ripple_locked_avg_lfp_plotly( 
    plot_context_title, all_ripple_peak_samples_abs, channels_df_for_plot, ref_channel_for_highlighting,       
    lfp_memmap_obj, fs, uv_scale_factor, depth_col_name, output_dir, base_filename_prefix,               
    uv_scale_factor_present=True, lfp_filter_range=(1, 250), window_ms_around_peak=400,          
    trace_amplitude_display_scale= 0.8,
    apply_notch_filter_flag=False, notch_freqs=None, Q_notch=None): # Added notch params

    plot_title_filter_text = f"Filtered {lfp_filter_range[0]}-{lfp_filter_range[1]} Hz" if lfp_filter_range else "Raw"
    if apply_notch_filter_flag and notch_freqs: plot_title_filter_text += " +Notch"
    print(f"  Generating Plotly Ripple-Locked Avg LFP: {plot_context_title} ({plot_title_filter_text})...")
    
    if channels_df_for_plot.empty or not all_ripple_peak_samples_abs:
        print(f"    No channels or no ripple peaks for {plot_context_title}. Skipping plot.")
        return

    fig = go.Figure(); num_channels_plot = len(channels_df_for_plot)
    window_samples_half = int((window_ms_around_peak / 2) / 1000 * fs); total_window_samples = window_samples_half * 2
    if total_window_samples <= 0: print(f"    Window is 0 samples for {plot_context_title}. Skip."); return
    time_vector_rel_ms = (np.arange(total_window_samples) - window_samples_half) / fs * 1000 
    
    b_lfp, a_lfp = None, None; can_bandpass_filter = False # Renamed can_filter
    if lfp_filter_range:
        nyquist = fs / 2.0; low_cut = lfp_filter_range[0]/nyquist; high_cut = lfp_filter_range[1]/nyquist
        if high_cut >= 1.0: high_cut = 0.99
        if low_cut <= 0 : low_cut = 0.001
        if low_cut < high_cut:
            try: b_lfp, a_lfp = butter(3, [low_cut, high_cut], btype='band'); can_bandpass_filter = True
            except ValueError as e: print(f"    Filter ({lfp_filter_range} Hz) error: {e}. Using un(bandpass)filtered LFP.")
        else: print(f"    Invalid LFP filter range ({lfp_filter_range} Hz). Using un(bandpass)filtered LFP.")
    else: print(f"    No LFP filter range specified. Using un(bandpass)filtered LFP for averages.")

    averaged_lfps_matrix = np.full((num_channels_plot, total_window_samples), np.nan)
    y_heatmap_indices = list(range(num_channels_plot)); y_tick_vals_for_plot, y_tick_text_for_plot = [], []
    
    for i_ch_plot, (_, ch_row) in enumerate(channels_df_for_plot.iterrows()):
        ch_global_idx = int(ch_row['global_channel_index']); segments_for_this_ch = []
        for peak_samp_abs in all_ripple_peak_samples_abs:
            start_abs = peak_samp_abs - window_samples_half; end_abs = peak_samp_abs + window_samples_half
            if start_abs >= 0 and end_abs <= lfp_memmap_obj.shape[0] and end_abs > start_abs :
                segment_raw = lfp_memmap_obj[start_abs:end_abs, ch_global_idx].astype(np.float64)
                if uv_scale_factor is not None: segment_raw *= uv_scale_factor
                
                segment_processed = segment_raw # Start with scaled (or raw if no_uv)
                if apply_notch_filter_flag and notch_freqs and Q_notch: # Apply Notch first
                    segment_processed = apply_notch_filters(segment_processed, fs, notch_freqs, Q_notch)
                if can_bandpass_filter and b_lfp is not None and a_lfp is not None: # Then Bandpass for display
                    try: segment_processed = filtfilt(b_lfp, a_lfp, segment_processed)
                    except ValueError: pass # Keep already (potentially) notched if bandpass fails
                
                if len(segment_processed) == total_window_samples: segments_for_this_ch.append(segment_processed)
        if segments_for_this_ch:
            try: averaged_lfps_matrix[i_ch_plot, :] = np.mean(np.array(segments_for_this_ch), axis=0)
            except ValueError as e_avg: warnings.warn(f"Avg LFP error Ch {ch_global_idx} {plot_context_title}: {e_avg}")
        y_tick_vals_for_plot.append(y_heatmap_indices[i_ch_plot])
        y_tick_text_for_plot.append(f"{ch_global_idx} ({ch_row[depth_col_name]:.0f}\u00b5m)")

    finite_heatmap_data = averaged_lfps_matrix[np.isfinite(averaged_lfps_matrix)]
    zmin_h, zmax_h = None, None; max_abs_lfp_val_for_scaling = 1.0
    # ... (zmin_h, zmax_h, line_deflection_scaler calculation - same as before) ...
    if finite_heatmap_data.size > 1:
        p_low, p_high = np.percentile(finite_heatmap_data, [2, 98])
        if np.isclose(p_low, p_high):
            center_val = p_low; current_std = np.std(finite_heatmap_data) if len(finite_heatmap_data)>1 and np.std(finite_heatmap_data) > 1e-9 else 1.0
            span = max(abs(center_val * 0.1), 0.1 * current_std); span = max(span, 0.5) 
            zmin_h, zmax_h = center_val - span, center_val + span
        else: zmin_h, zmax_h = p_low, p_high
        max_abs_lfp_val_for_scaling = max(abs(zmin_h if zmin_h is not None else 0), abs(zmax_h if zmax_h is not None else 0), 1.0) 
    elif finite_heatmap_data.size == 1: val = finite_heatmap_data[0]; zmin_h, zmax_h = val - 0.5, val + 0.5; max_abs_lfp_val_for_scaling = max(abs(val), 1.0)
    else: print(f"    WARNING: All heatmap data NaN/empty for {plot_context_title}. Heatmap blank."); zmin_h, zmax_h = -1.0, 1.0 
    line_deflection_scaler = trace_amplitude_display_scale / max_abs_lfp_val_for_scaling if max_abs_lfp_val_for_scaling > 1e-9 else 0 

    if finite_heatmap_data.size > 0:
        fig.add_trace(go.Heatmap(z=averaged_lfps_matrix, x=time_vector_rel_ms, y=y_heatmap_indices, 
                                 colorscale='RdBu_r', zmid=0, zmin=zmin_h, zmax=zmax_h, showscale=True,
                                 colorbar=dict(title='Avg LFP (\u00b5V)' if uv_scale_factor_present else 'Avg LFP (ADC)', 
                                               thickness=15, len=0.75, y=0.5, x=1.02, tickfont=dict(size=10))))
    for i_ch_plot in range(num_channels_plot):
        ch_row = channels_df_for_plot.iloc[i_ch_plot] 
        ch_global_idx = int(ch_row['global_channel_index'])
        avg_lfp_trace = averaged_lfps_matrix[i_ch_plot,:] 
        if np.all(np.isnan(avg_lfp_trace)): continue
        y_values_for_line = y_heatmap_indices[i_ch_plot] + (avg_lfp_trace * line_deflection_scaler)
        is_ref = ch_global_idx == ref_channel_for_highlighting 
        fig.add_trace(go.Scatter(x=time_vector_rel_ms, y=y_values_for_line, mode='lines',
                                 name=f"Ch {ch_global_idx} ({ch_row['acronym']} D:{ch_row[depth_col_name]:.0f}){' REF' if is_ref else ''}",
                                 line=dict(color='black', width=2.0 if is_ref else 0.75), hoverinfo='name', legendgroup=f"ch_{ch_global_idx}" ))
    fig.update_layout(title=f"{plot_context_title} ({plot_title_filter_text})", xaxis_title="Time from Ripple Peak (ms)",
        yaxis_title=f"Channels by Depth ({depth_col_name})",
        yaxis=dict(tickmode='array', tickvals=y_tick_vals_for_plot, ticktext=y_tick_text_for_plot,
                   showgrid=False, zeroline=False, autorange="reversed"),
        showlegend=True, legend_title_text="Channels", template="plotly_white",
        height=max(600, num_channels_plot * 25 + 200), width=1600 )
    plot_filepath_html = output_dir / f"{base_filename_prefix}.html"
    try: fig.write_html(str(plot_filepath_html)); print(f"    Saved Ripple-Locked Avg LFP plot to: {plot_filepath_html}")
    except Exception as e: print(f"    Error saving Ripple-Locked Avg LFP plot: {e}")

# --- Main script logic (Corrected Notch Frequencies and Call to plot_channel_psds) ---
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

    # --- Notch Filter Configuration ---
    apply_notch_filter_user_choice = False
    notch_frequencies_to_use = [60, 180] # Default frequencies to only 60 Hz and 180 Hz as major noise observed in these frq when plotting psd plots
    notch_q_to_use = 60 # Default Q factor
    
    # Prompt user about notch filtering
    if input(f"Apply notch filter at {notch_frequencies_to_use} Hz (Q={notch_q_to_use})? (yes/no, default no): ").strip().lower() == 'yes':
        apply_notch_filter_user_choice = True
        print(f"Notch filtering ENABLED for {notch_frequencies_to_use} Hz.")
        # Optional: Add prompts here to customize frequencies and Q if needed
        # custom_freqs_str = input(f"Enter comma-separated notch frequencies or press Enter for default ({notch_frequencies_to_use}): ").strip()
        # if custom_freqs_str:
        #     try: notch_frequencies_to_use = [float(f.strip()) for f in custom_freqs_str.split(',')]
        #     except ValueError: print(f"Invalid frequency list. Using default: {notch_frequencies_to_use}")
        # custom_q_str = input(f"Enter Q factor or press Enter for default ({notch_q_to_use}): ").strip()
        # if custom_q_str:
        #     try: notch_q_to_use = float(custom_q_str)
        #     except ValueError: print(f"Invalid Q factor. Using default: {notch_q_to_use}")
    else:
        print("Notch filtering DISABLED.")

    ripple_thr = (2, 4); ripple_dur = (20, 40, 500); ripple_f_range = (100, 250) 
    target_areas_list = ["CA1", "CA3", "CA2"]
    dg_area_name_str = "DG-mo" 
    base_fname = lfp_bin_p.stem.replace('.lf', '')
    
    lfp_example_plots_d = output_d / f"{base_fname}_ripple_example_traces"; os.makedirs(lfp_example_plots_d, exist_ok=True)
    ripple_locked_avg_plots_d = output_d / f"{base_fname}_shank_ripple_avg_LFP_plots"; os.makedirs(ripple_locked_avg_plots_d, exist_ok=True)

    lfp_mm, fs_o, n_ch, n_samp, uv_sf = load_lfp_data_memmap(lfp_bin_p, meta_p)
    if lfp_mm is None or fs_o is None: print("LFP load failed. Exiting."); return
    fs_eff = fs_o; uv_sf_present = uv_sf is not None
    if not uv_sf_present: print("WARNING: No voltage scaling. LFP in ADC units.")

    chan_df = None; depth_column_to_use = None
    try:
        chan_df = pd.read_csv(chan_info_p)
        if not all(c in chan_df.columns for c in ['global_channel_index','acronym','shank_index']):
            raise ValueError("ChanInfo CSV missing required columns.")
        if 'ycoord_on_shank_um' in chan_df.columns: depth_column_to_use = 'ycoord_on_shank_um'
        elif 'depth' in chan_df.columns: depth_column_to_use = 'depth' 
        elif 'y_coord' in chan_df.columns: depth_column_to_use = 'y_coord' 
        if not depth_column_to_use:
            warnings.warn("No depth column found in ChanInfo. Plots requiring depth sorting will be skipped.")
    except Exception as e: 
        print(f"ChanInfo Error: {e}")
        if hasattr(lfp_mm, '_mmap') and lfp_mm._mmap:
            try: lfp_mm._mmap.close() 
            except Exception: pass
        return
    
    sleep_lkp, epoch_b_samp_loaded = load_sleep_and_epoch_data(sleep_p, epoch_p, fs_eff)
    _get_state_func = None
    if sleep_lkp:
        def _get_state_internal(s_idx, fs_v, times_s_lkp, codes_lkp):
            t_s = s_idx / fs_v; idx = np.searchsorted(times_s_lkp, t_s, side='right') - 1
            return codes_lkp[idx] if 0 <= idx < len(codes_lkp) else -1 
        _get_state_func = lambda s, f_val: _get_state_internal(s,f_val,sleep_lkp['times_sec'],sleep_lkp['codes'])
        print("Sleep state lookup active.")
    else: 
        print("No sleep data loaded. Ripple-locked average plots will use 'AllData' if any ripples found, unless STATES_TO_ANALYZE is empty.")
        # STATES_TO_ANALYZE = {} # This was here before, but could be problematic if user expects NREM plots still

    processing_epochs = epoch_b_samp_loaded if epoch_b_samp_loaded else [(0, n_samp)]
    if not epoch_b_samp_loaded: print("No user-defined epochs for ripple detection. Processing entire recording as one epoch.")

    state_dur_in_ep = calculate_state_durations_in_epochs(processing_epochs, sleep_lkp, fs_eff, STATES_TO_ANALYZE)
    full_ts_abs = np.arange(n_samp) / fs_eff
    
    sel_noise_ch = None; sel_ref_chs = {}
    unique_shks = sorted(chan_df['shank_index'].unique())
    
    print("\n--- Channel Selection Phase ---")
    dg_df = chan_df[chan_df['acronym'].str.contains(dg_area_name_str, case=False, na=False)]
    if not dg_df.empty:
        dg_idxs = sorted(dg_df['global_channel_index'].unique().tolist())
        print(f"\nPSDs for {dg_area_name_str} channels: {dg_idxs}")
        plot_channel_psds(lfp_mm, fs_eff, uv_sf, dg_idxs, chan_df, 
                          ripple_freq_range_for_avg=ripple_f_range, # Corrected keyword
                          title_prefix=f"{dg_area_name_str} Channels ",
                          apply_notch_filter_flag=apply_notch_filter_user_choice,
                          notch_freqs=notch_frequencies_to_use, Q_notch=notch_q_to_use)
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
            plot_channel_psds(lfp_mm, fs_eff, uv_sf, area_shk_idxs, chan_df, 
                              ripple_freq_range_for_avg=ripple_f_range, # Corrected keyword
                              title_prefix=f"S{shk} {area_n} ",
                              apply_notch_filter_flag=apply_notch_filter_user_choice,
                              notch_freqs=notch_frequencies_to_use, Q_notch=notch_q_to_use)
            while True:
                try:
                    val = input(f"Enter 'global_channel_index' for {area_n} S{shk} ({area_shk_idxs}, or -1 skip): ")
                    idx = int(val)
                    if idx == -1: sel_ref_chs[(shk,area_n)] = -1; print(f"Skip {area_n} S{shk}."); break
                    if idx in area_shk_idxs: sel_ref_chs[(shk,area_n)] = idx; print(f"Ref {area_n} S{shk}: {idx}"); break
                    else: print(f"Invalid index ({area_shk_idxs} or -1).")
                except ValueError: print("Invalid input.")
    print("\n--- Channel selection complete. ---")

    total_noise_excluded_ripples = 0
    all_ripples_final_list = []
    
    states_iter_for_detection = STATES_TO_ANALYZE if _get_state_func and STATES_TO_ANALYZE else {"AllData": -2}

    for st_code_detect, st_name_detect in states_iter_for_detection.items():
        print(f"\nProcessing Ripple Detection for State: {st_name_detect} (Code: {st_code_detect})")
        for ep_idx, (ep_start, ep_end) in enumerate(processing_epochs): 
            print(f"  Epoch {ep_idx} (Samples: {ep_start}-{ep_end})")
            noise_ep_scaled = None
            if sel_noise_ch not in [None, -1] and 0 <= sel_noise_ch < n_ch:
                noise_ep_unsc = lfp_mm[ep_start:ep_end, sel_noise_ch]
                noise_ep_scaled = noise_ep_unsc.astype(np.float64); 
                if uv_sf: noise_ep_scaled *= uv_sf
                if apply_notch_filter_user_choice: # Apply notch if chosen
                    noise_ep_scaled = apply_notch_filters(noise_ep_scaled, fs_eff, notch_frequencies_to_use, notch_q_to_use)
            
            for (shk, area_n), ref_ch in sel_ref_chs.items():
                if ref_ch == -1: continue
                print(f"    Detecting S{shk} {area_n} Ch{ref_ch}...")
                lfp_ep_unsc = lfp_mm[ep_start:ep_end, ref_ch]
                if len(lfp_ep_unsc) == 0: print(f"      Empty LFP seg Ch{ref_ch} ep {ep_idx}. Skip."); continue
                lfp_ep_scaled_ref_chan = lfp_ep_unsc.astype(np.float64)
                if uv_sf: lfp_ep_scaled_ref_chan *= uv_sf
                if apply_notch_filter_user_choice: # Apply notch if chosen
                     lfp_ep_scaled_ref_chan = apply_notch_filters(lfp_ep_scaled_ref_chan, fs_eff, notch_frequencies_to_use, notch_q_to_use)

                ts_ep_abs = full_ts_abs[ep_start:ep_end]
                if len(lfp_ep_scaled_ref_chan) != len(ts_ep_abs): warnings.warn(f"LFP/TS mismatch Ep{ep_idx} Ch{ref_ch}. Skip."); continue
                
                ep_rips, filt_lfp_ep, _, env_z_ep, excluded_this_call = find_swr( 
                    lfp=lfp_ep_scaled_ref_chan, timestamps=ts_ep_abs, fs=fs_eff, thresholds=ripple_thr,
                    durations=ripple_dur, freq_range=ripple_f_range, noise=noise_ep_scaled)
                total_noise_excluded_ripples += excluded_this_call 
                print(f"      Found {len(ep_rips)} raw rips (after noise rejection) in ep {ep_idx} for Ch {ref_ch}.")
                if excluded_this_call > 0: print(f"        ({excluded_this_call} ripples excluded by noise for this segment/channel)")
                
                ep_rips_sorted = sorted(ep_rips, key=lambda x: x['peak_sample'])
                plotted_lfp_examples_this_combo = 0 
                for i_rip, ev_dict_original in enumerate(ep_rips_sorted):
                    ev_dict = ev_dict_original.copy() 
                    rel_start = ev_dict.pop('start_sample'); rel_peak = ev_dict.pop('peak_sample'); rel_end = ev_dict.pop('end_sample')
                    ev_dict.update({'abs_start_sample': ep_start + rel_start, 'abs_peak_sample': ep_start + rel_peak, 'abs_end_sample': ep_start + rel_end,
                                    'epoch_rel_start_sample': rel_start, 'epoch_rel_peak_sample': rel_peak, 'epoch_rel_end_sample': rel_end})
                    curr_ev_st_code = _get_state_func(ev_dict['abs_peak_sample'], fs_eff) if _get_state_func else -2
                    if st_name_detect == "AllData" or curr_ev_st_code == st_code_detect:
                        # For advanced features, use the ripple-band filtered LFP from SWR (derived from potentially notched LFP)
                        ev_filt_lfp_segment = filt_lfp_ep[ev_dict['epoch_rel_start_sample'] : ev_dict['epoch_rel_end_sample']+1] if filt_lfp_ep is not None else np.array([])
                        pk_f, n_cyc, pk_ph = calculate_advanced_ripple_features(ev_filt_lfp_segment, fs_eff, ripple_f_range) if len(ev_filt_lfp_segment) > 0 else (np.nan, np.nan, np.nan)
                        ev_dict.update({'peak_frequency_hz': pk_f, 'n_cycles': n_cyc, 'peak_phase_rad': pk_ph, 
                                        'iri_s': (ev_dict['peak_time'] - ep_rips_sorted[i_rip-1]['peak_time']) if i_rip > 0 else np.nan})
                        ev_to_store = ev_dict.copy()
                        ev_to_store.update({'shank': shk, 'area': area_n, 'reference_channel_idx': ref_ch, 
                                            'epoch_idx': ep_idx, 'state_code': curr_ev_st_code, 
                                            'state_name': STATES_TO_ANALYZE.get(curr_ev_st_code, "AllData") if _get_state_func and curr_ev_st_code != -2 else "AllData"})
                        all_ripples_final_list.append(ev_to_store)
                        
                        # --- Individual Ripple LFP Trace Plotting ---
                        if plotted_lfp_examples_this_combo < 3 and filt_lfp_ep is not None and env_z_ep is not None and len(ev_filt_lfp_segment) > 0:
                            plot_pad_s = int(0.1 * fs_eff)
                            # Use epoch_rel_... for slicing within the current epoch's LFP segment
                            plot_s_ep_rel = ev_dict['epoch_rel_start_sample'] - plot_pad_s
                            plot_e_ep_rel = ev_dict['epoch_rel_end_sample'] + plot_pad_s
                            
                            plot_s_ep_rel_clipped = max(0, plot_s_ep_rel)
                            plot_e_ep_rel_clipped = min(len(lfp_ep_scaled_ref_chan), plot_e_ep_rel)

                            if plot_s_ep_rel_clipped >= plot_e_ep_rel_clipped: continue

                            # The 'raw' LFP for this plot is the lfp_ep_scaled_ref_chan (which is already scaled and *notched* if option was chosen)
                            raw_lfp_for_plot_window = lfp_ep_scaled_ref_chan[plot_s_ep_rel_clipped:plot_e_ep_rel_clipped]
                                                        
                            filt_plot_seg = filt_lfp_ep[plot_s_ep_rel_clipped:plot_e_ep_rel_clipped] 
                            env_z_plot_seg = env_z_ep[plot_s_ep_rel_clipped:plot_e_ep_rel_clipped]
                            
                            ev_samps_for_plot = {
                                'start_sample_in_segment': ev_dict['epoch_rel_start_sample'] - plot_s_ep_rel_clipped,
                                'peak_sample_in_segment': ev_dict['epoch_rel_peak_sample'] - plot_s_ep_rel_clipped,
                                'end_sample_in_segment': ev_dict['epoch_rel_end_sample'] - plot_s_ep_rel_clipped
                            }
                            plot_fname = f"{base_fname}_State{ev_to_store['state_name']}_Ep{ep_idx}_Sh{shk}A{area_n}C{ref_ch}_PeakS{ev_dict['abs_peak_sample']}.png"
                            plot_fpath = lfp_example_plots_d / plot_fname 
                            plot_lfp_with_ripples(raw_lfp_for_plot_window, filt_plot_seg, env_z_plot_seg, ev_samps_for_plot, 
                                                  fs_eff, f"Ch {ref_ch} ({area_n} S{shk})", 
                                                  f"State:{ev_to_store['state_name']},Ep{ep_idx},Peak(abs):{ev_dict['abs_peak_sample']}",
                                                  uv_sf_present, output_plot_path=plot_fpath)
                            plotted_lfp_examples_this_combo += 1

    # --- Generate Ripple-Locked Average LFP Plots for NREM state in User-Chosen Epoch(s) ---
    if all_ripples_final_list and depth_column_to_use and epoch_b_samp_loaded and STATES_TO_ANALYZE:
        print("\n--- Generating NREM Ripple-Locked Average LFP Plots ---")
        nrem_state_code_for_plot = None; nrem_state_name_for_plot = "NREM" 
        for code, name in STATES_TO_ANALYZE.items():
            if name.upper() == "NREM": nrem_state_code_for_plot = code; nrem_state_name_for_plot = name; break
        
        if nrem_state_code_for_plot is not None:
            valid_epoch_indices_str = ", ".join(map(str, range(len(epoch_b_samp_loaded))))
            chosen_epochs_input_str = input(f"Enter Epoch Index(es) for NREM ripple-locked plots (e.g., 0 or 0,2,4 or -1 to skip) [valid: {valid_epoch_indices_str}]: ")
            
            if chosen_epochs_input_str.strip() == '-1' or not chosen_epochs_input_str.strip():
                print("Skipping NREM ripple-locked average plots by user choice or no input.")
            else:
                try:
                    chosen_epoch_indices_list_str = [x.strip() for x in chosen_epochs_input_str.split(',') if x.strip()]
                    chosen_epoch_indices_for_plotting = [] 
                    valid_input_for_epochs = True
                    for s_idx_str in chosen_epoch_indices_list_str:
                        if not s_idx_str.isdigit(): valid_input_for_epochs = False; break
                        idx_val = int(s_idx_str)
                        if not (0 <= idx_val < len(epoch_b_samp_loaded)): valid_input_for_epochs = False; break
                        chosen_epoch_indices_for_plotting.append(idx_val)
                    
                    if not valid_input_for_epochs or not chosen_epoch_indices_for_plotting:
                        print(f"Invalid epoch index or format. Please enter comma-separated numbers from [{valid_epoch_indices_str}]. Skipping NREM plots.")
                    else:
                        ripple_df_for_plotting = pd.DataFrame(all_ripples_final_list)
                        print(f"Generating NREM ripple-locked plots for user-selected Epoch(s): {chosen_epoch_indices_for_plotting}")

                        for current_chosen_epoch_idx in chosen_epoch_indices_for_plotting: 
                            print(f"\n  Processing for NREM, Epoch {current_chosen_epoch_idx}:")
                            for (shk_plot_idx, area_plot_name), ref_ch_plot_idx in sel_ref_chs.items():
                                if ref_ch_plot_idx == -1: continue
                                ripples_for_avg = ripple_df_for_plotting[
                                    (ripple_df_for_plotting['reference_channel_idx'] == ref_ch_plot_idx) &
                                    (ripple_df_for_plotting['epoch_idx'] == current_chosen_epoch_idx) &
                                    (ripple_df_for_plotting['state_code'] == nrem_state_code_for_plot) ]
                                if ripples_for_avg.empty:
                                    print(f"    No NREM ripples in Epoch {current_chosen_epoch_idx} for RefCh {ref_ch_plot_idx} (S{shk_plot_idx} {area_plot_name}). Skipping plot.")
                                    continue
                                all_ripple_peaks_for_avg = ripples_for_avg['abs_peak_sample'].tolist()
                                all_relevant_areas_on_shank_plot = [a.upper() for a in target_areas_list] + [dg_area_name_str.upper()]
                                channels_to_display_df = chan_df[
                                    (chan_df['shank_index'] == shk_plot_idx) &
                                    (chan_df['acronym'].str.upper().isin(all_relevant_areas_on_shank_plot))
                                ].sort_values(by=depth_column_to_use, ascending=False)
                                if channels_to_display_df.empty: 
                                    print(f"    No channels from target regions found on Shank {shk_plot_idx} for RefCh {ref_ch_plot_idx}. Skipping plot.")
                                    continue
                                plot_filename_base = f"{base_fname}_{nrem_state_name_for_plot}_Ep{current_chosen_epoch_idx}_S{shk_plot_idx}RefA{area_plot_name}C{ref_ch_plot_idx}_RippleLockedAvg"
                                plot_ripple_locked_avg_lfp_plotly( 
                                    plot_context_title=f"{nrem_state_name_for_plot} Ripple-Locked Avg LFP - Shank {shk_plot_idx} (Locked to {area_plot_name} RefCh {ref_ch_plot_idx}) - Epoch {current_chosen_epoch_idx}",
                                    all_ripple_peak_samples_abs=all_ripple_peaks_for_avg, 
                                    channels_df_for_plot=channels_to_display_df,
                                    ref_channel_for_highlighting=ref_ch_plot_idx, 
                                    lfp_memmap_obj=lfp_mm, fs=fs_eff, uv_scale_factor=uv_sf,
                                    depth_col_name=depth_column_to_use, 
                                    output_dir=ripple_locked_avg_plots_d, 
                                    base_filename_prefix=plot_filename_base, 
                                    uv_scale_factor_present=uv_sf_present,
                                    lfp_filter_range=(1, 250), window_ms_around_peak=400,
                                    apply_notch_filter_flag=apply_notch_filter_user_choice, # Pass notch option
                                    notch_freqs=notch_frequencies_to_use, Q_notch=notch_q_to_use )
                except ValueError: print("Invalid epoch index format. Please enter comma-separated numbers. Skipping NREM ripple-locked plots.")
        else: print("    NREM state not defined in STATES_TO_ANALYZE. Cannot generate NREM plots.")
    elif not depth_column_to_use: print("Skipping Ripple-Locked Average LFP plots (no depth column).")
    elif not epoch_b_samp_loaded : print("Skipping Ripple-Locked Average LFP plots (no user-defined epochs).")
    elif not STATES_TO_ANALYZE : print("Skipping Ripple-Locked Average LFP plots (no sleep states to analyze).")
    
    if sel_noise_ch not in [None, -1]:
        print(f"\n--- Noise Exclusion Summary ---")
        print(f"Total number of potential ripple events excluded by noise channel: {total_noise_excluded_ripples}")
    
    print("\n--- Data Saving ---")
    if not all_ripples_final_list: print("No ripples to save."); 
    else:
        save_opt = input("Save detected ripple data & summary? (yes/no): ").strip().lower()
        if save_opt == 'yes':
            # ... (Full data saving logic from previous correct version) ...
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
                for col in group_by_cols:
                    if col not in ripple_df_out.columns: print(f"Warning: Groupby col '{col}' not in detailed data. Summary incomplete.")
                try: grouped = ripple_df_out.groupby(group_by_cols)
                except KeyError as ke: print(f"KeyError for summary groupby: {ke}. Cannot create summary."); grouped = None
                if grouped:
                    for name, group in grouped:
                        state_n, state_c, ep_i, shk_v, area_v, ref_c_v = name 
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
                else: print("No data for summary stats CSV.")
            except Exception as e: print(f"Error saving ripple data/summary: {e}"); traceback.print_exc()
        else: print("Ripple data not saved.")

    if lfp_mm is not None and hasattr(lfp_mm, '_mmap') and lfp_mm._mmap is not None:
        print("Closing LFP memmap..."); 
        try: lfp_mm._mmap.close(); print("LFP memmap closed.")
        except Exception as e: print(f"Error closing memmap: {e}")
    print("\nScript finished.")

if __name__ == "__main__":
    main_ripple_analysis_and_visualization()