# -*- coding: utf-8 -*-
"""
Advanced Ripple Detection, Feature Extraction, and Visualization Script
(Channel selection PSDs Z-scored, 1-200Hz, with Avg 100-250Hz Power in Legend)
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, welch, hilbert
from scipy.stats import zscore, circmean
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib
import matplotlib.pyplot as plt
# from matplotlib.patches import Rectangle 
import os
from pathlib import Path
import warnings
import re 
from tkinter import Tk, filedialog

# --- Configure Plotly & Matplotlib ---
pio.renderers.default = "browser"
try:
    matplotlib.use('Qt5Agg') 
    print("Matplotlib backend set to Qt5Agg (for individual ripple LFP traces).")
except ImportError:
    try: matplotlib.use('TkAgg'); print("Matplotlib backend set to TkAgg (for individual ripple LFP traces).")
    except ImportError: print("Warning: Could not set interactive Matplotlib backend. Plots might not display but will be saved.")
except Exception as e: print(f"Warning: Error setting Matplotlib backend: {e}. Plots might not display but will be saved.")

try:
    from DemoReadSGLXData.readSGLX import readMeta
except ImportError: print("ERROR: readMeta import failed. Ensure DemoReadSGLXData is accessible."); exit()

# --- Helper Functions ---
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

def find_swr(lfp, timestamps, fs=1000, thresholds=(2, 5), durations=(20, 40, 500),
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

# --- MODIFIED plot_channel_psds function ---
def plot_channel_psds(lfp_memmap_obj, fs, uv_scale_factor, 
                      channel_indices_to_plot, channel_info_df, 
                      ripple_freq_range_for_avg, # Specific band for averaging power
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
        
        # Use a representative segment of data for PSD calculation
        max_samples_for_psd = int(fs * 120) # e.g., 2 minutes of data for PSD
        num_samples_to_take = min(lfp_memmap_obj.shape[0], max_samples_for_psd)
        start_s_psd = (lfp_memmap_obj.shape[0] - num_samples_to_take) // 2 if lfp_memmap_obj.shape[0] > num_samples_to_take else 0
        
        signal_int16 = lfp_memmap_obj[start_s_psd : start_s_psd + num_samples_to_take, ch_idx]
        signal_float = signal_int16.astype(np.float64)
        
        scaled_units_label = "ADC\u00b2/Hz"
        if uv_scale_factor is not None: # Apply scaling BEFORE PSD calculation
            signal_float *= uv_scale_factor
            scaled_units_label = "\u00b5V\u00b2/Hz"
        
        try:
            current_nperseg = nperseg if len(signal_float) >= nperseg else len(signal_float)
            if current_nperseg == 0 : 
                print(f"Ch {ch_idx} has 0 samples for PSD. Skipping PSD."); continue
            
            frequencies, psd_v = welch(signal_float, fs=fs, nperseg=current_nperseg) # psd_v is linear power
        except ValueError as e:
            print(f"Could not compute PSD for channel {ch_idx}: {e}"); continue

        psd_db_full_spectrum = 10 * np.log10(psd_v + np.finfo(float).eps) # Full dB spectrum
        
        # Calculate average power in 100-250 Hz band from linear psd_v
        avg_power_db_str_ripple_band = "N/A"
        if ripple_freq_range_for_avg is not None and len(ripple_freq_range_for_avg) == 2:
            band_mask_for_avg = (frequencies >= ripple_freq_range_for_avg[0]) & (frequencies <= ripple_freq_range_for_avg[1])
            if np.any(band_mask_for_avg) and len(psd_v[band_mask_for_avg]) > 0:
                avg_linear_power_rip_band = np.mean(psd_v[band_mask_for_avg])
                avg_db_power_rip_band = 10 * np.log10(avg_linear_power_rip_band + np.finfo(float).eps)
                avg_power_db_str_ripple_band = f"{avg_db_power_rip_band:.2f} dB"
            else:
                avg_power_db_str_ripple_band = "NoDataInBand"
        
        # Z-score the PSD in dB (over all its frequencies)
        if len(psd_db_full_spectrum) >= 2: 
            psd_db_zscored = zscore(psd_db_full_spectrum)
        else:
            psd_db_zscored = psd_db_full_spectrum # Cannot zscore if less than 2 points

        # Mask for plotting (1-200 Hz) - applied AFTER Z-scoring the full dB spectrum
        plot_mask_1_200Hz = (frequencies >= 1) & (frequencies <= 200)
        
        ch_r = channel_info_df[channel_info_df['global_channel_index'] == ch_idx]
        ch_n_base = f"Ch {ch_idx} ({ch_r['acronym'].iloc[0]} S{ch_r['shank_index'].iloc[0]})" if not ch_r.empty else f'Ch {ch_idx}'
        ch_n_legend = f"{ch_n_base} (Avg Pwr {ripple_freq_range_for_avg[0]}-{ripple_freq_range_for_avg[1]}Hz: {avg_power_db_str_ripple_band})"

        fig.add_trace(go.Scatter(
            x=frequencies[plot_mask_1_200Hz], 
            y=psd_db_zscored[plot_mask_1_200Hz], # Plot the masked part of Z-scored PSD
            mode='lines', 
            line=dict(width=1, color=plot_colors[i % len(plot_colors)]), 
            opacity=0.7, 
            name=ch_n_legend
        ))
    
    fig.update_layout(
        title=f"{title_prefix}Welch PSD (1-200 Hz)", 
        xaxis_title="Frequency (Hz)", 
        yaxis_title="Z-scored Power Spectral Density (dB)", # Y-axis label reflects Z-scoring
        font=dict(size=12), # Smaller font for potentially long legend
        template="plotly_white", 
        showlegend=True, 
        legend_title_text=f'Channels (Avg Power in {ripple_freq_range_for_avg[0]}-{ripple_freq_range_for_avg[1]}Hz Band)',
        height=700, 
        width=1200
    )
    fig.show()

# --- Function to plot PSDs of LFP segments from multiple regional channels during a single ripple event ---
def plot_regional_psds_for_single_ripple(
    ripple_event,                   # Single event dictionary (with 'abs_start_sample', 'abs_end_sample', 'peak_time')
    channels_in_region_df_sorted,   # DataFrame of all channels in the event's region/shank, sorted by depth
    lfp_memmap_obj,
    fs,
    uv_scale_factor,
    ref_channel_global_idx,         # The channel on which this ripple_event was detected
    output_dir,                     # Base output directory for these plots
    plot_filename_base_prefix,      # e.g., "M2_file_StateNREM_Ep0_RefCh123"
    uv_scale_factor_present=True):

    if channels_in_region_df_sorted.empty:
        warnings.warn("No channels provided for regional PSD plot. Skipping.")
        return

    event_start_samp = ripple_event['abs_start_sample']
    event_end_samp = ripple_event['abs_end_sample']
    event_peak_time = ripple_event['peak_time']

    if event_start_samp >= event_end_samp:
        warnings.warn(f"Invalid event duration for ripple at {event_peak_time:.3f}s. Skipping PSD plot.")
        return

    fig = go.Figure()
    
    # Define a color sequence for traces
    try:
        from matplotlib.cm import get_cmap
        try: cmap = plt.colormaps['viridis'] # Using viridis for potentially many traces
        except AttributeError: cmap = get_cmap('viridis')
        plot_colors = [f'rgb({int(r*255)},{int(g*255)},{int(b*255)})' for r,g,b,a_ in cmap(np.linspace(0, 1, len(channels_in_region_df_sorted)))]
    except Exception as e:
        print(f"Matplotlib colormap error for regional PSD: {e}. Using default plotly sequence.")
        plot_colors = None # Let Plotly use its default sequence

    for i, (_, ch_row) in enumerate(channels_in_region_df_sorted.iterrows()):
        ch_global_idx = int(ch_row['global_channel_index'])
        ch_acronym = ch_row['acronym']
        ch_shank = ch_row['shank_index']
        # Assuming depth_col_name is available if channels_in_region_df_sorted is passed correctly sorted
        depth_col_name_local = 'ycoord_on_shank_um' if 'ycoord_on_shank_um' in ch_row else 'y_coord' if 'y_coord' in ch_row else 'depth' if 'depth' in ch_row else None
        depth_val_str = f" D:{ch_row[depth_col_name_local]:.0f}" if depth_col_name_local else ""


        if not (0 <= ch_global_idx < lfp_memmap_obj.shape[1]):
            warnings.warn(f"Channel {ch_global_idx} out of bounds for LFP data. Skipping its PSD.")
            continue
            
        lfp_event_segment_int16 = lfp_memmap_obj[event_start_samp : event_end_samp + 1, ch_global_idx]
        lfp_event_segment_float = lfp_event_segment_int16.astype(np.float64)

        if uv_scale_factor is not None:
            lfp_event_segment_float *= uv_scale_factor
        
        if len(lfp_event_segment_float) < 4: # Need a few samples for Welch
            # warnings.warn(f"LFP segment for Ch {ch_global_idx} during event at {event_peak_time:.3f}s is too short ({len(lfp_event_segment_float)} samples). Skipping PSD.")
            continue

        # For Welch on short event segments, nperseg should be <= segment length
        # Using full segment length for nperseg effectively computes a periodogram of the event.
        nperseg_event = len(lfp_event_segment_float) 
        min_nperseg_for_welch = 8 # A very small minimum, Welch might still struggle
        if nperseg_event < min_nperseg_for_welch:
            # warnings.warn(f"Segment for Ch {ch_global_idx} too short ({nperseg_event} samples) for reliable Welch PSD. Min {min_nperseg_for_welch} needed. Skipping PSD.")
            continue
        
        try:
            # For very short segments, detrend='constant' or detrend=False might be better than 'linear' (default)
            frequencies, psd_event_linear = welch(lfp_event_segment_float, fs=fs, nperseg=nperseg_event, noverlap=0, detrend='constant')
        except ValueError as e:
            # warnings.warn(f"PSD calculation error for Ch {ch_global_idx} during event at {event_peak_time:.3f}s: {e}. Skipping.")
            continue
            
        psd_event_db = 10 * np.log10(psd_event_linear + np.finfo(float).eps)
        
        # Z-score the PSD in dB for this specific event, across frequencies (as per user's example)
        if len(psd_event_db) >=2: 
             psd_event_db_zscored = zscore(psd_event_db)
        else:
             psd_event_db_zscored = psd_event_db # Cannot zscore single point

        line_width = 2.0 if ch_global_idx == ref_channel_global_idx else 1.0
        opacity_val = 1.0 if ch_global_idx == ref_channel_global_idx else 0.7
        
        line_color_val = plot_colors[i % len(plot_colors)] if plot_colors else None

        fig.add_trace(go.Scatter(
            x=frequencies, 
            y=psd_event_db_zscored, # Plotting Z-scored dB PSD
            mode='lines',
            name=f"Ch {ch_global_idx} ({ch_acronym} S{ch_shank}{depth_val_str})",
            line=dict(color=line_color_val, width=line_width),
            opacity=opacity_val
        ))

    y_axis_unit_str = "Z-scored PSD (dB)"
    fig.update_layout(
        title=f"Regional PSDs during Ripple on RefCh {ref_channel_global_idx} (Peak @ {event_peak_time:.3f}s)",
        xaxis_title="Frequency (Hz)",
        yaxis_title=y_axis_unit_str,
        font=dict(size=12),
        template="plotly_white",
        showlegend=True,
        legend_title_text="Channels in Region",
        xaxis_range=[0, 300] # Show up to 300 Hz or adjust as needed
    )
    
    peak_time_str = f"{event_peak_time:.3f}".replace(".", "p") # Sanitize for filename
    plot_html_filename = f"{plot_filename_base_prefix}_RipplePeak{peak_time_str}s_RegionalPSDs.html"
    plot_html_path = output_dir / plot_html_filename
    try:
        fig.write_html(str(plot_html_path))
        print(f"    Saved regional PSDs for ripple event to: {plot_html_path}")
    except Exception as e:
        print(f"    Error saving regional PSDs plot: {e}")

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
        try: plt.savefig(output_plot_path); 
        except Exception as e: print(f"    Error saving plot to {output_plot_path}: {e}")
    else: plt.show()
    plt.close(fig)

# --- CORRECTED Function Definition to plot Stacked LFP Traces ---
def plot_shank_lfp_traces_with_states( 
    plot_context_title,                 # Main title part, e.g., "Shank X - Epoch Y Overview"
    lfp_data_for_epoch_dict,            # Dict: {ch_global_idx: lfp_trace_array}
    channels_df_for_plot,               
    ref_channels_global_indices, 
    time_vector_sec,                  
    sleep_state_intervals_for_background, 
    depth_col_name,
    target_areas_list_param,            # EXPECTED PARAMETER
    dg_area_name_str_param,             # EXPECTED PARAMETER
    output_dir,
    base_filename_prefix,               
    fs,                                 
    uv_scale_factor_present=True,
    lfp_filter_range=(1, 200),      
    trace_amplitude_display_scale=0.4,  
    is_ripple_locked_plot=False 
    ):

    plot_title_filter_text = f"Filtered {lfp_filter_range[0]}-{lfp_filter_range[1]} Hz" if lfp_filter_range else "Raw"
    # Construct the full title using the passed parameters
    full_plot_title = f"{plot_context_title} ({plot_title_filter_text})<br>Regions: {', '.join(target_areas_list_param)}, {dg_area_name_str_param}"
    
    print(f"  Generating Plotly LFP stack: {plot_context_title} ({plot_title_filter_text})...")
    
    if channels_df_for_plot.empty or not lfp_data_for_epoch_dict:
        print(f"    No channel data or LFP data for {plot_context_title}. Skipping plot.")
        return

    fig = go.Figure()
    num_channels_plot = len(channels_df_for_plot)
    
    nyquist = fs / 2.0
    low_cut_filt = lfp_filter_range[0] / nyquist if lfp_filter_range else 0
    high_cut_filt = lfp_filter_range[1] / nyquist if lfp_filter_range else nyquist -0.1
    if high_cut_filt >= 1.0: high_cut_filt = 0.99 
    if low_cut_filt <=0 : low_cut_filt = 0.001 
    
    b_lfp, a_lfp = None, None; can_filter = False
    if lfp_filter_range and low_cut_filt < high_cut_filt: 
        try:
            b_lfp, a_lfp = butter(3, [low_cut_filt, high_cut_filt], btype='band')
            can_filter = True
        except ValueError as e: print(f"    Could not create LFP bandpass filter ({lfp_filter_range} Hz): {e}. Using raw traces.")
    elif lfp_filter_range: 
        print(f"    Invalid LFP filter range ({lfp_filter_range} Hz). Using raw traces.")
    else: 
        print("    No LFP filter range specified. Using raw traces for heatmap/lines.")

    processed_lfp_for_heatmap_and_lines = [] 
    y_heatmap_indices = list(range(num_channels_plot)) 
    y_tick_vals_for_plot, y_tick_text_for_plot = [], []
    temp_filtered_traces_for_scaling = []

    for i, (_, ch_row) in enumerate(channels_df_for_plot.iterrows()):
        ch_global_idx = int(ch_row['global_channel_index'])
        lfp_trace_raw = lfp_data_for_epoch_dict.get(ch_global_idx)
        current_trace_for_processing = np.full(len(time_vector_sec), np.nan) if time_vector_sec is not None and len(time_vector_sec)>0 else np.array([])
        if lfp_trace_raw is not None and len(lfp_trace_raw) == len(time_vector_sec):
            lfp_to_filter_plot = lfp_trace_raw.copy() 
            if can_filter and b_lfp is not None and a_lfp is not None:
                try: current_trace_for_processing = filtfilt(b_lfp, a_lfp, lfp_to_filter_plot)
                except ValueError as filter_err: 
                    warnings.warn(f"Could not filter Ch {ch_global_idx}: {filter_err}. Using raw trace."); 
                    current_trace_for_processing = lfp_to_filter_plot 
            else: current_trace_for_processing = lfp_to_filter_plot
        processed_lfp_for_heatmap_and_lines.append(current_trace_for_processing)
        if not np.all(np.isnan(current_trace_for_processing)):
             temp_filtered_traces_for_scaling.append(current_trace_for_processing)
    lfp_matrix_for_heatmap_np = np.array(processed_lfp_for_heatmap_and_lines)
    
    finite_heatmap_data = lfp_matrix_for_heatmap_np[np.isfinite(lfp_matrix_for_heatmap_np)]
    zmin_h, zmax_h = None, None; max_abs_lfp_val_for_scaling = 1.0
    if finite_heatmap_data.size > 1:
        p_low, p_high = np.percentile(finite_heatmap_data, [2, 98])
        if np.isclose(p_low, p_high):
            center_val = p_low 
            current_std = np.std(finite_heatmap_data) if len(finite_heatmap_data)>1 and np.std(finite_heatmap_data) > 1e-9 else 1.0
            span = max(abs(center_val * 0.1), 0.1 * current_std); span = max(span, 0.5) 
            zmin_h, zmax_h = center_val - span, center_val + span
        else: zmin_h, zmax_h = p_low, p_high
        max_abs_lfp_val_for_scaling = max(abs(zmin_h if zmin_h is not None else 0), abs(zmax_h if zmax_h is not None else 0), 1.0) 
    elif finite_heatmap_data.size == 1: 
        val = finite_heatmap_data[0]; zmin_h, zmax_h = val - 0.5, val + 0.5
        max_abs_lfp_val_for_scaling = max(abs(val), 1.0)
    else: 
        print(f"    WARNING: All heatmap data is NaN or empty for {plot_context_title}. Heatmap might be blank."); 
        zmin_h, zmax_h = -1.0, 1.0 
    
    line_deflection_scaler = trace_amplitude_display_scale / max_abs_lfp_val_for_scaling if max_abs_lfp_val_for_scaling > 1e-9 else 0 

    if finite_heatmap_data.size > 0:
        fig.add_trace(go.Heatmap(z=lfp_matrix_for_heatmap_np, x=time_vector_sec, y=y_heatmap_indices, 
                                 colorscale='RdBu_r', zmid=0, zmin=zmin_h, zmax=zmax_h, showscale=True,
                                 colorbar=dict(title='LFP Voltage (\u00b5V)' if uv_scale_factor_present else 'LFP (ADC)', 
                                               thickness=15, len=0.75, y=0.5, x=1.02, tickfont=dict(size=10))))
    
    for i_ch_plot in range(num_channels_plot):
        ch_row = channels_df_for_plot.iloc[i_ch_plot] 
        ch_global_idx = int(ch_row['global_channel_index'])
        lfp_trace_to_plot_line = lfp_matrix_for_heatmap_np[i_ch_plot,:] 
        y_baseline = y_heatmap_indices[i_ch_plot] 
        y_tick_vals_for_plot.append(y_baseline); 
        y_tick_text_for_plot.append(f"{ch_global_idx} ({ch_row[depth_col_name]:.0f}\u00b5m)")
        if np.all(np.isnan(lfp_trace_to_plot_line)): continue
        y_values_for_line = y_baseline + (lfp_trace_to_plot_line * line_deflection_scaler)
        is_ref = ch_global_idx in ref_channels_global_indices
        fig.add_trace(go.Scatter(x=time_vector_sec, y=y_values_for_line, mode='lines',
                                 name=f"Ch {ch_global_idx} ({ch_row['acronym']} D:{ch_row[depth_col_name]:.0f}){' REF' if is_ref else ''}",
                                 line=dict(color='black', width=2.0 if is_ref else 0.75), hoverinfo='name', legendgroup=f"ch_{ch_global_idx}" ))

    if not is_ripple_locked_plot and sleep_state_intervals_for_background: 
        for start_s, end_s, state_name, color_matplotlib in sleep_state_intervals_for_background:
            if color_matplotlib == 'lightblue': color_rgba_plotly = 'rgba(173, 216, 230, 0.2)'
            elif color_matplotlib == 'lightcoral': color_rgba_plotly = 'rgba(240, 128, 128, 0.2)'
            else: color_rgba_plotly = 'rgba(211, 211, 211, 0.2)' 
            fig.add_shape(type="rect", xref="x", yref="paper", x0=start_s, y0=0, x1=end_s, y1=1,
                          fillcolor=color_rgba_plotly, layer="below", line_width=0, name=state_name)
        added_states_to_legend = set()
        for _, _, state_name, color_matplotlib in sleep_state_intervals_for_background:
            if state_name not in added_states_to_legend:
                if color_matplotlib == 'lightblue': color_rgba_plotly_legend = 'rgba(173,216,230,0.7)'
                elif color_matplotlib == 'lightcoral': color_rgba_plotly_legend = 'rgba(240,128,128,0.7)'
                else: color_rgba_plotly_legend = 'rgba(211,211,211,0.7)'
                fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', 
                    marker=dict(size=10, color=color_rgba_plotly_legend), name=f"{state_name} (BG)", legendgroup="states_bg" ))
                added_states_to_legend.add(state_name)

    xaxis_title_text = "Time from Ripple Peak (ms)" if is_ripple_locked_plot else "Time (s)"
    fig.update_layout(
        title=full_plot_title, # Use the fully constructed title
        xaxis_title=xaxis_title_text,
        yaxis_title=f"Channels by Depth ({depth_col_name})",
        yaxis=dict(tickmode='array', tickvals=y_tick_vals_for_plot, ticktext=y_tick_text_for_plot,
                   showgrid=False, zeroline=False, autorange="reversed"),
        showlegend=True, legend_title_text="Channels/States", template="plotly_white",
        height=max(600, num_channels_plot * 25 + 200), width=1600 )
            
    plot_filepath_html = output_dir / f"{base_filename_prefix}.html"
    try: fig.write_html(str(plot_filepath_html)); print(f"    Saved Plotly LFP stack with heatmap to: {plot_filepath_html}")
    except Exception as e: print(f"    Error saving Plotly LFP stack plot: {e}")
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
    lfp_example_plots_d = output_d / f"{base_fname}_ripple_example_traces"; os.makedirs(lfp_example_plots_d, exist_ok=True)
    shank_overview_plots_d = output_d / f"{base_fname}_shank_LFP_overview_plots"; os.makedirs(shank_overview_plots_d, exist_ok=True)

    lfp_mm, fs_o, n_ch, n_samp, uv_sf = load_lfp_data_memmap(lfp_bin_p, meta_p)
    if lfp_mm is None or fs_o is None: print("LFP load failed. Exiting."); return
    fs_eff = fs_o; uv_sf_present = uv_sf is not None
    if not uv_sf_present: print("WARNING: No voltage scaling. LFP in ADC units.")

    chan_df = None; depth_column_to_use = None
    try:
        chan_df = pd.read_csv(chan_info_p)
        if not all(c in chan_df.columns for c in ['global_channel_index','acronym','shank_index']):
            raise ValueError("ChanInfo CSV missing required columns.")
        if 'depth' in chan_df.columns: depth_column_to_use = 'depth'
        elif 'y_coord' in chan_df.columns: depth_column_to_use = 'y_coord'
        elif 'ycoord_on_shank_um' in chan_df.columns: depth_column_to_use = 'ycoord_on_shank_um'
        if not depth_column_to_use:
            warnings.warn("No depth column found in ChanInfo. Shank LFP overview plots requiring depth sorting will be skipped.")
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
    else: print("No sleep data loaded.")
    
    processing_epochs = epoch_b_samp_loaded if epoch_b_samp_loaded else [(0, n_samp)]
    if not epoch_b_samp_loaded: print("No epoch boundaries for ripple detection. Processing entire recording as one epoch.")

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
                          ripple_freq_range_for_avg=ripple_f_range,  # Pass the correct ripple_f_range
                          title_prefix=f"{dg_area_name_str} Channels ")
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
                              ripple_freq_range_for_avg=ripple_f_range, # Pass the correct ripple_f_range
                              title_prefix=f"S{shk} {area_n} ")
            while True:
                try:
                    val = input(f"Enter 'global_channel_index' for {area_n} S{shk} ({area_shk_idxs}, or -1 skip): ")
                    idx = int(val)
                    if idx == -1: sel_ref_chs[(shk,area_n)] = -1; print(f"Skip {area_n} S{shk}."); break
                    if idx in area_shk_idxs: sel_ref_chs[(shk,area_n)] = idx; print(f"Ref {area_n} S{shk}: {idx}"); break
                    else: print(f"Invalid index ({area_shk_idxs} or -1).")
                except ValueError: print("Invalid input.")
    print("\n--- Channel selection complete. ---")

    # --- Shank LFP Overview Plot Options ---
    plot_overview_for_all_epochs = False
    if epoch_b_samp_loaded and len(epoch_b_samp_loaded) > 1 : 
        if input("Generate shank LFP overview plots for ALL defined epochs (Default: first epoch only)? (yes/no): ").strip().lower() == 'yes':
            plot_overview_for_all_epochs = True
    
    epochs_to_plot_overview_tuples = [] 
    if epoch_b_samp_loaded: # Check if user provided epoch file
        epochs_to_plot_overview_tuples.append(epoch_b_samp_loaded[0]) 
        if plot_overview_for_all_epochs:
            epochs_to_plot_overview_tuples.extend(epoch_b_samp_loaded[1:])
    elif depth_column_to_use : # No user epochs, but depth exists, so plot default initial segment
         epochs_to_plot_overview_tuples.append((0, min(n_samp, int(10 * 60 * fs_eff))))
         print("No specific epochs defined by user; shank overview plot will use first part of recording.")

    if depth_column_to_use and epochs_to_plot_overview_tuples:
        print(f"\n--- Generating Shank LFP Overview Plot(s) (Plotly with Heatmap, 1-200Hz LFP) ---")
        # THE LOOP WHERE i_ep_plot IS DEFINED
        for i_ep_plot, (ep_plot_start_raw, ep_plot_end_raw) in enumerate(epochs_to_plot_overview_tuples):
            max_plot_duration_samps = int(0.01 * 60 * fs_eff)
            ep_plot_start = ep_plot_start_raw
            ep_plot_end = min(ep_plot_end_raw, ep_plot_start + max_plot_duration_samps)
            ep_plot_end = min(ep_plot_end, n_samp)
            if ep_plot_start >= ep_plot_end: continue
            
            # Use i_ep_plot from enumerate() for consistent file/title naming
            # If epoch_b_samp_loaded is empty, i_ep_plot will be 0 for the default segment
            actual_epoch_idx_for_filename_and_title = i_ep_plot 

            print(f"  For Overview Plot - Epoch {actual_epoch_idx_for_filename_and_title} (Samples: {ep_plot_start} to {ep_plot_end})")
            time_vector_for_overview = full_ts_abs[ep_plot_start:ep_plot_end]
            
            overview_sleep_intervals = []
            if sleep_lkp:
                s_t_sec, s_c = sleep_lkp['times_sec'], sleep_lkp['codes']
                p_start_s, p_end_s = ep_plot_start/fs_eff, ep_plot_end/fs_eff
                for i_st, st_s_abs in enumerate(s_t_sec):
                    st_code_val = s_c[i_st]; st_e_abs = s_t_sec[i_st+1] if i_st+1<len(s_t_sec) else p_end_s+1.0 # Add a buffer
                    ov_s = max(st_s_abs, p_start_s); ov_e = min(st_e_abs, p_end_s)
                    if ov_e > ov_s and st_code_val in STATES_TO_ANALYZE:
                        st_n_plot = STATES_TO_ANALYZE[st_code_val]
                        clr = 'lightblue' if st_n_plot=="Awake" else 'lightcoral' if st_n_plot=="NREM" else 'lightgrey'
                        overview_sleep_intervals.append((ov_s, ov_e, st_n_plot, clr))
            
            for shk_idx_plot in unique_shks:
                shank_ch_df_plot = chan_df[(chan_df['shank_index'] == shk_idx_plot) & (chan_df['acronym'].str.upper().isin([a.upper() for a in target_areas_list] + [dg_area_name_str.upper()]))].sort_values(by=depth_column_to_use, ascending=False) 
                if shank_ch_df_plot.empty: continue
                shank_lfp_data_dict = {}
                ref_chans_on_shk = [sel_ref_chs.get((shk_idx_plot, area_val), -1) for area_val in target_areas_list]
                ref_chans_on_shk = [ch for ch in ref_chans_on_shk if ch != -1]
                for _, ch_r_plot in shank_ch_df_plot.iterrows():
                    g_idx = int(ch_r_plot['global_channel_index'])
                    # Ensure slicing is within bounds of lfp_mm
                    safe_ep_plot_end = min(ep_plot_end, lfp_mm.shape[0])
                    if ep_plot_start >= safe_ep_plot_end: continue # Skip if slice is invalid

                    lfp_s = lfp_mm[ep_plot_start:safe_ep_plot_end, g_idx].astype(np.float64)
                    if uv_sf: lfp_s *= uv_sf
                    shank_lfp_data_dict[g_idx] = lfp_s
                
                # Make sure time_vector_for_overview matches the actual LFP segment length if adjusted by safe_ep_plot_end
                current_time_vector = full_ts_abs[ep_plot_start:safe_ep_plot_end]
                if not shank_lfp_data_dict: continue # Skip if no LFP data collected for any channel

                plot_shank_lfp_traces_with_states( 
                    plot_context_title=f"Shank {shk_idx_plot} - Epoch {actual_epoch_idx_for_filename_and_title} LFP Overview",
                    lfp_data_for_epoch_dict=shank_lfp_data_dict, 
                    channels_df_for_plot=shank_ch_df_plot, 
                    ref_channels_global_indices=ref_chans_on_shk,
                    time_vector_sec=current_time_vector, 
                    sleep_state_intervals_for_background=overview_sleep_intervals, 
                    depth_col_name=depth_column_to_use,
                    target_areas_list_param=target_areas_list, 
                    dg_area_name_str_param=dg_area_name_str,    
                    output_dir=shank_overview_plots_d,
                    base_filename_prefix=f"{base_fname}_Shank{shk_idx_plot}_Epoch{actual_epoch_idx_for_filename_and_title}_Overview", 
                    fs=fs_eff, 
                    uv_scale_factor_present=uv_sf_present,
                    is_ripple_locked_plot=False 
                )
    elif not depth_column_to_use: 
        print("Skipping shank LFP overview plots (no depth column found).")
    else: 
        print("Skipping shank LFP overview plots (no epochs specified for overview and/or depth column missing).")

    # ... (Rest of the main function: ripple detection, example plots, data saving) ...

    # --- Ripple Detection and Individual Example LFP Plots ---
    # ... (This section remains the same as the previously corrected version for ripple detection and plotting example LFP traces) ...
    plot_ripple_examples_for_all_defined_epochs = False
    if epoch_b_samp_loaded : 
        if input("Generate example ripple LFP traces for ALL defined epochs? (Default: first only) (yes/no): ").strip().lower() == 'yes':
            plot_ripple_examples_for_all_defined_epochs = True
            
    all_ripples_final_list = []
    noise_ep_scaled = None 
    states_iter = STATES_TO_ANALYZE if _get_state_func and STATES_TO_ANALYZE else {"AllData": -2}
    for st_code, st_name in states_iter.items():
        print(f"\nProcessing State: {st_name} (Code: {st_code})")
        for ep_idx, (ep_start, ep_end) in enumerate(processing_epochs): 
            print(f"  Epoch {ep_idx} (Samples: {ep_start}-{ep_end}) for Ripple Detection")
            if sel_noise_ch not in [None, -1] and 0 <= sel_noise_ch < n_ch:
                noise_ep_unsc = lfp_mm[ep_start:ep_end, sel_noise_ch]
                noise_ep_scaled = noise_ep_unsc.astype(np.float64); 
                if uv_sf: noise_ep_scaled *= uv_sf
            else: noise_ep_scaled = None
            for (shk, area_n), ref_ch in sel_ref_chs.items():
                if ref_ch == -1: continue
                print(f"    Detecting S{shk} {area_n} Ch{ref_ch}...")
                lfp_ep_unsc = lfp_mm[ep_start:ep_end, ref_ch]
                if len(lfp_ep_unsc) == 0: print(f"      Empty LFP seg Ch{ref_ch} ep {ep_idx}. Skip."); continue
                lfp_ep_scaled_ref_chan = lfp_ep_unsc.astype(np.float64)
                if uv_sf: lfp_ep_scaled_ref_chan *= uv_sf
                ts_ep_abs = full_ts_abs[ep_start:ep_end]
                if len(lfp_ep_scaled_ref_chan) != len(ts_ep_abs): warnings.warn(f"LFP/TS mismatch Ep{ep_idx} Ch{ref_ch}. Skip."); continue
                ep_rips, filt_lfp_ep, _, env_z_ep = find_swr( 
                    lfp=lfp_ep_scaled_ref_chan, timestamps=ts_ep_abs, fs=fs_eff, thresholds=ripple_thr,
                    durations=ripple_dur, freq_range=ripple_f_range, noise=noise_ep_scaled)
                print(f"      Found {len(ep_rips)} raw rips in ep {ep_idx} for Ch {ref_ch}.")
                ep_rips_sorted = sorted(ep_rips, key=lambda x: x['peak_sample'])
                plotted_lfp_examples_this_combo = 0 
                for i_rip, ev_dict in enumerate(ep_rips_sorted):
                    ev_dict['abs_start_sample'] = ep_start + ev_dict['start_sample']
                    ev_dict['abs_peak_sample'] = ep_start + ev_dict['peak_sample']
                    ev_dict['abs_end_sample'] = ep_start + ev_dict['end_sample']
                    ev_dict['epoch_rel_start_sample'] = ev_dict.pop('start_sample')
                    ev_dict['epoch_rel_peak_sample'] = ev_dict.pop('peak_sample')
                    ev_dict['epoch_rel_end_sample'] = ev_dict.pop('end_sample')
                    curr_ev_st_code = _get_state_func(ev_dict['abs_peak_sample'], fs_eff) if _get_state_func else -2
                    if st_name == "AllData" or curr_ev_st_code == st_code:
                        ev_filt_lfp_segment = filt_lfp_ep[ev_dict['epoch_rel_start_sample'] : ev_dict['epoch_rel_end_sample']+1] if filt_lfp_ep is not None else np.array([])
                        if len(ev_filt_lfp_segment) > 0 :
                             pk_f, n_cyc, pk_ph = calculate_advanced_ripple_features(ev_filt_lfp_segment, fs_eff, ripple_f_range)
                        else: pk_f, n_cyc, pk_ph = np.nan, np.nan, np.nan
                        ev_dict['peak_frequency_hz'] = pk_f; ev_dict['n_cycles'] = n_cyc; ev_dict['peak_phase_rad'] = pk_ph
                        ev_dict['iri_s'] = (ev_dict['peak_time'] - ep_rips_sorted[i_rip-1]['peak_time']) if i_rip > 0 else np.nan
                        ev_to_store = ev_dict.copy()
                        ev_to_store.update({'shank': shk, 'area': area_n, 'reference_channel_idx': ref_ch, 
                                            'epoch_idx': ep_idx, 'state_code': curr_ev_st_code, 
                                            'state_name': STATES_TO_ANALYZE.get(curr_ev_st_code, "Unknown") if _get_state_func else "AllData"})
                        all_ripples_final_list.append(ev_to_store)
                        should_plot_example_lfp = False
                        if not epoch_b_samp_loaded: should_plot_example_lfp = True
                        elif plot_ripple_examples_for_all_defined_epochs: should_plot_example_lfp = True
                        elif ep_idx == 0: should_plot_example_lfp = True
                        if should_plot_example_lfp and plotted_lfp_examples_this_combo < 3 and \
                           filt_lfp_ep is not None and env_z_ep is not None and len(ev_filt_lfp_segment) > 0 :
                            plot_pad_s = int(0.1 * fs_eff)
                            plot_s_ep = max(0, ev_dict['epoch_rel_start_sample'] - plot_pad_s)
                            plot_e_ep = min(len(lfp_ep_scaled_ref_chan), ev_dict['epoch_rel_end_sample'] + plot_pad_s)
                            if plot_s_ep >= plot_e_ep: continue
                            raw_lfp_plot_unsc = lfp_mm[ep_start + plot_s_ep : ep_start + plot_e_ep, ref_ch]
                            raw_lfp_plot_sc = raw_lfp_plot_unsc.astype(np.float64)
                            if uv_sf: raw_lfp_plot_sc *= uv_sf
                            filt_plot_seg = filt_lfp_ep[plot_s_ep:plot_e_ep]; env_z_plot_seg = env_z_ep[plot_s_ep:plot_e_ep]
                            ev_samps_for_plot = {'start_sample_in_segment': ev_dict['epoch_rel_start_sample'] - plot_s_ep,
                                                 'peak_sample_in_segment': ev_dict['epoch_rel_peak_sample'] - plot_s_ep,
                                                 'end_sample_in_segment': ev_dict['epoch_rel_end_sample'] - plot_s_ep}
                            plot_fname = f"{base_fname}_State{ev_to_store['state_name']}_Ep{ep_idx}_Sh{shk}A{area_n}C{ref_ch}_PeakS{ev_dict['abs_peak_sample']}.png"
                            plot_fpath = lfp_example_plots_d / plot_fname 
                            plot_lfp_with_ripples(raw_lfp_plot_sc, filt_plot_seg, env_z_plot_seg, ev_samps_for_plot, 
                                                  fs_eff, f"Ch {ref_ch} ({area_n} S{shk})", 
                                                  f"State:{ev_to_store['state_name']},Ep{ep_idx},Peak(abs):{ev_dict['abs_peak_sample']}",
                                                  uv_sf_present, output_plot_path=plot_fpath)
                            plotted_lfp_examples_this_combo += 1
    
    # --- Data Saving --- 
    print("\n--- Data Saving ---")
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
                for col in group_by_cols:
                    if col not in ripple_df_out.columns:
                        print(f"Warning: Groupby col '{col}' not in detailed data. Summary incomplete.")
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