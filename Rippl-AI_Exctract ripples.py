# -*- coding: utf-8 -*-
"""
Script to extract sharp-wave ripples (SWRs) using the rippl-AI library.

This script is adapted to use rippl-AI as the detection engine,
explicitly calling the 1D-CNN model, while maintaining the original workflow for
data loading, feature extraction, and saving.
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, welch, hilbert
from scipy.stats import circmean
import os
from pathlib import Path
import warnings
from tkinter import Tk, filedialog

# Attempt to import the core rippl-AI function
try:
    from rippl_ai.main import find_ripples
    print("Successfully imported rippl-AI.")
except ImportError:
    print("FATAL ERROR: Could not import 'find_ripples' from 'rippl_ai.main'.")
    print("Please ensure rippl-AI is installed and your Spyder IDE is using the correct Python interpreter.")
    find_ripples = None # Set to None to prevent script from running further

# --- Helper Functions from Original Script ---
# These functions are useful for data loading and feature calculation.

def readMeta(meta_path):
    """Reads a SpikeGLX meta file and returns a dictionary of parameters."""
    meta = {}
    with open(meta_path, 'r') as f:
        for line in f:
            if '=' in line:
                parts = line.strip().split('=')
                meta[parts[0]] = parts[1]
    return meta

def get_voltage_scaling_factor(meta):
    """Calculates the voltage scaling factor from metadata."""
    try:
        v_max = float(meta['imAiRangeMax'])
        i_max = int(meta['imMaxInt'])
        # NP2 4-shank probe gain. Adjust if using a different probe.
        lfp_gain = 80
        sf = (v_max * 1e6) / (i_max * lfp_gain)
        return sf
    except KeyError as e:
        warnings.warn(f"KeyError for voltage scaling: {e}. None returned.")
        return None

def load_lfp_data_memmap(file_path, meta_path):
    """Loads LFP data using numpy.memmap for memory efficiency."""
    file_path, meta_path = Path(file_path), Path(meta_path)
    print(f"LFP Load: {file_path}, Meta: {meta_path}")
    try:
        meta = readMeta(meta_path)
        n_c = int(meta['nSavedChans'])
        srate_key = 'imSampRate' if 'imSampRate' in meta else 'niSampRate'
        fs_o = float(meta[srate_key])
        uv_sf = get_voltage_scaling_factor(meta)
        
        fsize = file_path.stat().st_size
        isize = np.dtype('int16').itemsize
        n_s = fsize // (n_c * isize)
        shape = (n_s, n_c)
        print(f"  Meta: {n_c} ch, Rate: {fs_o:.2f} Hz, Samples: {n_s}")
        if uv_sf is not None:
            print(f"  uV scaling factor obtained: {uv_sf:.6f}")
            
        lfp_m = np.memmap(file_path, dtype='int16', mode='r', shape=shape)
        print(f"  Memmap successful: {file_path}")
        return lfp_m, fs_o, n_c, n_s, uv_sf
    except Exception as e:
        print(f"LFP Load Error: {e}")
        return None, None, 0, 0, None

def calculate_advanced_ripple_features(filtered_event_lfp, fs, ripple_freq_range_tuple):
    """
    Calculates advanced features for a single ripple event.
    This function is preserved from your original script.
    """
    peak_freq, n_cycles, peak_phase = np.nan, np.nan, np.nan
    # Check if the segment is long enough for reliable PSD calculation
    min_samples_for_welch = max(8, int(fs / ripple_freq_range_tuple[0] * 1.5))
    if len(filtered_event_lfp) < min_samples_for_welch:
        return peak_freq, n_cycles, peak_phase
    
    # Peak Frequency and Number of Cycles
    try:
        nperseg_r = min(len(filtered_event_lfp), 256)
        freqs_r, psd_r = welch(filtered_event_lfp, fs=fs, nperseg=nperseg_r)
        band_mask = (freqs_r >= ripple_freq_range_tuple[0]) & (freqs_r <= ripple_freq_range_tuple[1])
        if np.any(band_mask):
            peak_freq = freqs_r[band_mask][np.argmax(psd_r[band_mask])]
            duration_s = len(filtered_event_lfp) / fs
            n_cycles = peak_freq * duration_s
    except Exception as e:
        warnings.warn(f"Peak freq/cycles error: {e}")
        
    # Peak Phase
    try:
        analytic_signal = hilbert(filtered_event_lfp)
        instantaneous_phase = np.angle(analytic_signal)
        # In the original script, the peak was the trough of the filtered LFP
        peak_idx_in_event = np.argmin(filtered_event_lfp)
        peak_phase = instantaneous_phase[peak_idx_in_event]
    except Exception as e:
        warnings.warn(f"Peak phase error: {e}")
        
    return peak_freq, n_cycles, peak_phase


# --- Main Analysis Function ---
def run_ripple_ai_analysis():
    """Main function to run the ripple detection and analysis pipeline."""
    if find_ripples is None:
        print("rippl-AI not available. Aborting analysis.")
        return

    # --- 1. User Inputs (Same as your original script) ---
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    
    print("Select the required files for ripple analysis.")
    
    lfp_bin_p = Path(filedialog.askopenfilename(title="Select LFP Binary File (*.lf.bin)", filetypes=[("SGLX LFP files", "*.lf.bin")]))
    if not lfp_bin_p.name: print("Cancelled."); return
    
    meta_p = Path(lfp_bin_p.with_suffix('.meta'))
    if not meta_p.exists():
        meta_p = Path(filedialog.askopenfilename(title="Could not find .meta file. Please select it.", initialdir=lfp_bin_p.parent, filetypes=[("Meta files", "*.meta")]))
    if not meta_p.name: print("Cancelled."); return
        
    chan_info_p = Path(filedialog.askopenfilename(title="Select Channel Info CSV", filetypes=[("CSV files", "*.csv")]))
    if not chan_info_p.name: print("Cancelled."); return

    output_d = Path(filedialog.askdirectory(title="Select Output Directory"))
    if not output_d.name: print("Cancelled."); return
    
    root.destroy()

    # --- 2. Configuration & Parameters ---
    # Frequency range for post-hoc feature calculation (e.g., peak frequency)
    RIPPLE_FREQ_RANGE = (100, 250)
    
    base_fname = lfp_bin_p.stem.replace('.lf', '')

    # --- 3. Load Data ---
    lfp_mm, fs, n_ch, _, uv_sf = load_lfp_data_memmap(lfp_bin_p, meta_p)
    if lfp_mm is None: print("LFP load failed. Exiting."); return
    
    uv_sf_present = uv_sf is not None
    if not uv_sf_present:
        print("\nWARNING: No voltage scaling factor found. LFP is in raw ADC units.")
        print("rippl-AI models are trained on µV data. Results may be suboptimal.")
        print("Continuing with raw data...\n")

    try:
        chan_df = pd.read_csv(chan_info_p)
    except Exception as e:
        print(f"Channel Info CSV Error: {e}. Exiting."); return

    # --- 4. Select Channels for Detection ---
    print("\n--- Channel Selection ---")
    print("Select one or more reference channels for ripple detection.")
    print("You can enter multiple channel indices separated by commas (e.g., 3, 15, 28)")
    
    all_ch_indices = chan_df['global_channel_index'].tolist()
    ref_channels_to_process = []
    while True:
        val = input(f"Enter global channel indices to process (e.g. 128,132) from available channels 0-{n_ch-1}: ")
        try:
            indices = [int(x.strip()) for x in val.split(',') if x.strip()]
            if all(idx in all_ch_indices for idx in indices):
                ref_channels_to_process = indices
                print(f"Selected channels for processing: {ref_channels_to_process}")
                break
            else:
                print("Error: One or more selected channels are not valid.")
        except ValueError:
            print("Invalid input. Please enter numbers separated by commas.")

    # --- 5. Run Ripple Detection ---
    all_ripples_final_list = []

    for ref_ch in ref_channels_to_process:
        print(f"\n--- Processing Channel {ref_ch} ---")
        
        # Extract the LFP data for the selected channel
        lfp_ch_data = lfp_mm[:, ref_ch].astype(np.float64)
        
        # Apply voltage scaling factor if it exists
        if uv_sf_present:
            lfp_ch_data *= uv_sf
        
        print(f"Running rippl-AI on Ch {ref_ch}... (This may take a while)")
        
        # Core `rippl-AI` call
        # Explicitly select the 1D-CNN architecture as requested.
        ripple_dfs = find_ripples([lfp_ch_data],
                                  fs=[fs],
                                  arch='CNN1D') # Use arch='CNN1D'
        
        # The result is a list of dataframes, one for each signal we passed.
        # Since we passed only one signal, we take the first element.
        ripple_df_ai = ripple_dfs[0]
        
        print(f"  Found {len(ripple_df_ai)} ripples on Ch {ref_ch}.")

        if ripple_df_ai.empty:
            continue

        # --- 6. Process `rippl-AI` Output & Calculate Features ---
        # Create a bandpass filter to get the filtered LFP trace for feature calculation
        # This is the same filtering as in your original `find_swr` function
        nyquist = fs / 2.0
        low, high = RIPPLE_FREQ_RANGE
        b, a = butter(3, [low / nyquist, high / nyquist], btype='band')
        filtered_lfp_ch = filtfilt(b, a, lfp_ch_data)
        
        # The `find_ripples` output gives 'Start', 'Stop', and 'Peak' times in seconds.
        # We convert them to sample indices.
        ripple_df_ai['start_sample'] = (ripple_df_ai['Start'] * fs).astype(int)
        ripple_df_ai['peak_sample'] = (ripple_df_ai['Peak'] * fs).astype(int)
        ripple_df_ai['end_sample'] = (ripple_df_ai['Stop'] * fs).astype(int)

        for _, ripple in ripple_df_ai.iterrows():
            s, p, e = ripple['start_sample'], ripple['peak_sample'], ripple['end_sample']
            
            # Ensure indices are within bounds
            if not (0 <= s < p < e < len(lfp_ch_data)):
                continue

            # Extract the filtered LFP segment for this event
            event_filt_lfp = filtered_lfp_ch[s : e+1]

            # Calculate advanced features using your original function
            pk_f, n_cyc, pk_ph = calculate_advanced_ripple_features(event_filt_lfp, fs, RIPPLE_FREQ_RANGE)

            # Get area info from the channel dataframe
            ch_info = chan_df[chan_df['global_channel_index'] == ref_ch]
            area = ch_info['acronym'].iloc[0] if not ch_info.empty else "N/A"
            shank = ch_info['shank_index'].iloc[0] if not ch_info.empty else "N/A"

            # Store results in a dictionary, similar to your original format
            ev_to_store = {
                'start_time': ripple['Start'],
                'peak_time': ripple['Peak'],
                'end_time': ripple['Stop'],
                'abs_start_sample': s,
                'abs_peak_sample': p,
                'abs_end_sample': e,
                'duration_ms': (ripple['Stop'] - ripple['Start']) * 1000,
                'peak_amplitude_power': np.nan, # rippl-AI does not provide this
                'peak_amplitude_z': np.nan,    # rippl-AI does not provide this
                'peak_frequency_hz': pk_f,
                'n_cycles': n_cyc,
                'peak_phase_rad': pk_ph,
                'shank': shank,
                'area': area,
                'reference_channel_idx': ref_ch
            }
            all_ripples_final_list.append(ev_to_store)

    # --- 7. Save Data ---
    print("\n--- Data Saving ---")
    if not all_ripples_final_list:
        print("No ripples were detected across any selected channels.")
    else:
        # Create a single dataframe from all detected events
        ripple_df_out = pd.DataFrame(all_ripples_final_list)
        ripple_df_out = ripple_df_out.sort_values(by='peak_time').reset_index(drop=True)

        # Calculate Inter-Ripple-Interval (IRI)
        ripple_df_out['iri_s'] = ripple_df_out.groupby('reference_channel_idx')['peak_time'].diff()

        # Save detailed event list
        csv_path = output_d / f"{base_fname}_ripplai_events_detailed.csv"
        ripple_df_out.to_csv(csv_path, index=False, float_format='%.4f')
        print(f"Detailed ripples saved to CSV: {csv_path}")

        # Save summary statistics
        summary_data = []
        for ch, group in ripple_df_out.groupby('reference_channel_idx'):
            rec_duration_s = len(lfp_mm) / fs
            num_rips = len(group)
            rate_hz = num_rips / rec_duration_s if rec_duration_s > 0 else 0.0

            valid_ph = group['peak_phase_rad'].dropna()
            avg_pk_ph = circmean(valid_ph, high=np.pi, low=-np.pi) if len(valid_ph) > 0 else np.nan
            
            summary_row = {
                'reference_channel_idx': ch,
                'shank': group['shank'].iloc[0],
                'area': group['area'].iloc[0],
                'num_ripples': num_rips,
                'total_recording_duration_s': rec_duration_s,
                'ripple_rate_hz': rate_hz,
                'avg_duration_ms': group['duration_ms'].mean(),
                'avg_peak_frequency_hz': group['peak_frequency_hz'].mean(),
                'avg_n_cycles': group['n_cycles'].mean(),
                'avg_iri_s': group['iri_s'].mean(),
                'resultant_peak_phase_rad': avg_pk_ph
            }
            summary_data.append(summary_row)
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_csv_path = output_d / f"{base_fname}_ripplai_summary_stats.csv"
            summary_df.to_csv(summary_csv_path, index=False, float_format='%.4f')
            print(f"Summary stats saved to CSV: {summary_csv_path}")

    # --- 8. Cleanup ---
    if lfp_mm is not None and hasattr(lfp_mm, '_mmap'):
        lfp_mm._mmap.close()
        print("\nLFP memmap closed.")

    print("\nScript finished.")


if __name__ == "__main__":
    run_ripple_ai_analysis()