# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 13:30:00 2025

This script calculates firing properties by concatenating spike times for a
given sleep state within each user-selected epoch. This provides a single,
aggregated result based on a continuous virtual spike train.
"""

import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog, Frame, Label, Button, Checkbutton, BooleanVar, Toplevel
from pathlib import Path
from tqdm import tqdm
import warnings
from numba import jit

# =============================================================================
# --- CORE CALCULATION AND HELPER FUNCTIONS ---
# =============================================================================

@jit(nopython=True)
def calculate_royer_burst_index(spike_times_sec):
    """
    Calculates the burst index based on the definition by Royer et al., 2012.
    This function is JIT-compiled with Numba for high performance.
    """
    if len(spike_times_sec) < 10:
        return np.nan

    spike_times_ms = spike_times_sec * 1000
    bin_size_ms = 1
    window_ms = 51
    bins = np.arange(0, window_ms, bin_size_ms)
    
    lags = []
    for i in range(len(spike_times_ms)):
        j = i + 1
        while j < len(spike_times_ms) and (spike_times_ms[j] - spike_times_ms[i]) < window_ms:
            lags.append(spike_times_ms[j] - spike_times_ms[i])
            j += 1
            
    if len(lags) == 0:
        return np.nan

    acg_counts, _ = np.histogram(np.array(lags), bins=bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    peak_mask = (bin_centers > 0) & (bin_centers <= 10)
    if not np.any(peak_mask):
        return 0.0
    peak_val = np.max(acg_counts[peak_mask])

    baseline_mask = (bin_centers >= 40) & (bin_centers < 50)
    if not np.any(baseline_mask) or np.sum(acg_counts[baseline_mask]) == 0:
        return 1.0 if peak_val > 0 else 0.0
    baseline_val = np.mean(acg_counts[baseline_mask])

    amplitude = peak_val - baseline_val

    burst_index = 0.0
    if amplitude > 0:
        if peak_val > 0: burst_index = amplitude / peak_val
    else:
        if baseline_val > 0: burst_index = amplitude / baseline_val
    return burst_index

def calculate_firing_properties(spike_times_sec, segment_duration_sec):
    """
    Calculates firing rate, CV of ISI, and the Royer et al. (2012) burst index.
    """
    num_spikes = len(spike_times_sec)
    mean_firing_rate_hz = num_spikes / segment_duration_sec if segment_duration_sec > 0 else 0
    
    cv_isi = np.nan
    if num_spikes >= 2:
        # Spike times are already concatenated, so np.diff is valid
        isis = np.diff(spike_times_sec)
        mean_isi = np.mean(isis)
        if mean_isi > 0:
            cv_isi = np.std(isis) / mean_isi
            
    royer_burst_index = calculate_royer_burst_index(spike_times_sec)
    
    return {
        "mean_firing_rate_hz": mean_firing_rate_hz, 
        "cv_isi": cv_isi, 
        "royer_burst_index": royer_burst_index
    }

def reconstruct_sleep_segments(states_array, times_array, epoch_boundaries, step_size=1.0):
    """
    Combines flat arrays of states and times into a structured dictionary of
    sleep segments, organized by epoch.
    """
    state_map = {0: 'Awake', 1: 'NREM', 2: 'REM'}
    sleep_data_by_epoch = {f"epoch_{i}": [] for i in range(len(epoch_boundaries))}

    if len(states_array) == 0 or len(times_array) == 0:
        return sleep_data_by_epoch

    change_indices = np.where(np.diff(states_array) != 0)[0]
    segment_starts = np.concatenate(([0], change_indices + 1))
    segment_ends = np.concatenate((change_indices, [len(states_array) - 1]))

    all_segments = []
    for start_idx, end_idx in zip(segment_starts, segment_ends):
        if start_idx > end_idx: continue
        state_int = states_array[start_idx]
        state_str = state_map.get(state_int, 'Unknown')
        start_time = times_array[start_idx]
        end_time = times_array[end_idx] + step_size
        all_segments.append({'state': state_str, 'start_time': start_time, 'end_time': end_time})
    
    for segment in all_segments:
        seg_start = segment['start_time']
        for i, (epoch_start, epoch_end) in enumerate(epoch_boundaries):
            if epoch_start <= seg_start < epoch_end:
                epoch_key = f"epoch_{i}"
                clipped_segment = segment.copy()
                clipped_segment['start_time'] = max(seg_start, epoch_start)
                clipped_segment['end_time'] = min(segment['end_time'], epoch_end)
                sleep_data_by_epoch[epoch_key].append(clipped_segment)
                break
    return sleep_data_by_epoch

def get_user_selections(regions, types, epochs, sleep_states):
    """
    Creates a Tkinter GUI for selecting analysis parameters.
    """
    result = {}
    window = Toplevel()
    window.title("Select Analysis Parameters")
    window.attributes("-topmost", True)
    
    frames = {
        "Regions": Frame(window, relief='sunken', borderwidth=1, padx=5, pady=5),
        "Types": Frame(window, relief='sunken', borderwidth=1, padx=5, pady=5),
        "Epochs": Frame(window, relief='sunken', borderwidth=1, padx=5, pady=5),
        "States": Frame(window, relief='sunken', borderwidth=1, padx=5, pady=5)
    }
    for frame in frames.values(): frame.pack(pady=10, padx=10, fill='x')

    def create_checkbox_group(parent, title, items_list):
        Label(parent, text=title, font=('Helvetica', 10, 'bold')).pack()
        item_vars = {item: BooleanVar() for item in items_list}
        all_var = BooleanVar()
        def toggle_all():
            for var in item_vars.values(): var.set(all_var.get())
        Checkbutton(parent, text="-- SELECT ALL --", variable=all_var, command=toggle_all).pack(anchor='w')
        for item, var in item_vars.items():
            Checkbutton(parent, text=str(item), variable=var).pack(anchor='w')
        return item_vars

    region_vars = create_checkbox_group(frames["Regions"], "Select Brain Regions", regions)
    type_vars = create_checkbox_group(frames["Types"], "Select Neuron Types", types)
    epoch_labels = [f"Epoch {e}" for e in epochs]
    epoch_vars = create_checkbox_group(frames["Epochs"], "Select Epochs", epoch_labels)
    state_vars = create_checkbox_group(frames["States"], "Select Sleep States", sleep_states)

    def on_submit():
        result['regions'] = [r for r, v in region_vars.items() if v.get()]
        result['types'] = [t for t, v in type_vars.items() if v.get()]
        selected_epoch_labels = [label for label, var in epoch_vars.items() if var.get()]
        result['epochs'] = [int(label.replace("Epoch ", "")) for label in selected_epoch_labels]
        result['sleep_states'] = [s for s, v in state_vars.items() if v.get()]
        if not all(result.values()):
            print("Warning: Please select at least one item from each category.")
        else:
            window.quit(); window.destroy()

    Button(window, text="Run Analysis", command=on_submit, font=('Helvetica', 10, 'bold')).pack(pady=20)
    window.mainloop()
    return result

# =============================================================================
# --- MAIN ANALYSIS SCRIPT ---
# =============================================================================
def main():
    warnings.filterwarnings("ignore", category=UserWarning, module='numba')
    root = tk.Tk(); root.withdraw()
    
    print("--- Firing Property Analysis (Concatenated Spike Times) ---")
    data_dir = filedialog.askdirectory(title="Select Root Folder (containing unit, timestamp, and sleep state files)")
    if not data_dir: print("No directory selected. Exiting."); return
    data_dir = Path(data_dir)

    print("Scanning recordings...")
    all_regions, all_types, all_epochs, all_sleep_states = set(), set(), set(), set()
    recording_paths = []
    
    for unit_path in data_dir.glob('good_clusters_processed_*_CellExplorerACG.npy'):
        name_part = unit_path.name.replace('good_clusters_processed_', '').split('_imec0')[0]
        ts_path = data_dir / f"{name_part}_tcat.nidq_timestamps.npy"
        sleep_states_path = data_dir / f"{name_part}_tcat_sleep_states_EMG.npy"
        sleep_times_path = data_dir / f"{name_part}_tcat_sleep_state_times_EMG.npy"

        if ts_path.exists() and sleep_states_path.exists() and sleep_times_path.exists():
            recording_paths.append({ "name": name_part, "units": unit_path, "timestamps": ts_path, "sleep_states": sleep_states_path, "sleep_times": sleep_times_path})
            try:
                units_df = pd.DataFrame(list(np.load(unit_path, allow_pickle=True)))
                all_regions.update(units_df['acronym'].dropna().unique())
                all_types.update(units_df['cell_type'].dropna().unique())
                
                ts_data = np.load(ts_path, allow_pickle=True).item()
                epoch_data = ts_data.get('EpochFrameData', [])
                all_epochs.update([e['epoch_index'] for e in epoch_data])
                
                states_array = np.load(sleep_states_path, allow_pickle=True)
                state_map = {0: 'Awake', 1: 'NREM', 2: 'REM'}
                for state_int in np.unique(states_array):
                    all_sleep_states.add(state_map.get(state_int, 'Unknown'))
            except Exception as e:
                print(f"Warning: Could not read metadata from {name_part}: {e}")
        else:
            print(f"Info: Skipping '{name_part}' because a required file is missing.")

    if not recording_paths: print("No valid recordings with all required files found. Exiting."); return

    selections = get_user_selections(sorted(list(all_regions)), sorted(list(all_types)), sorted(list(all_epochs)), sorted(list(all_sleep_states)))
    if not all(selections.values()): print("No valid selections made. Exiting."); return

    all_results = []
    print("\nProcessing selected data...")
    for rec_info in tqdm(recording_paths, desc="Recordings"):
        try:
            units_df = pd.DataFrame(list(np.load(rec_info["units"], allow_pickle=True)))
            ts_data = np.load(rec_info["timestamps"], allow_pickle=True).item()
            states_array = np.load(rec_info["sleep_states"], allow_pickle=True)
            times_array = np.load(rec_info["sleep_times"], allow_pickle=True)
            
            epoch_boundaries = [(e['start_time_sec'], e['end_time_sec']) for e in ts_data.get('EpochFrameData', [])]
            sleep_data = reconstruct_sleep_segments(states_array, times_array, epoch_boundaries)
        except Exception as e:
            print(f"\nError loading files for {rec_info['name']}, skipping: {e}"); continue

        filtered_units = units_df[units_df['acronym'].isin(selections['regions']) & units_df['cell_type'].isin(selections['types'])]
        
        # --- CONCATENATION LOGIC BLOCK ---
        for epoch_idx in selections['epochs']:
            epoch_key = f"epoch_{epoch_idx}"
            if epoch_key not in sleep_data: continue

            for _, neuron in filtered_units.iterrows():
                spikes_all = neuron['spike_times_sec']
                
                # Dictionaries to hold concatenated data for the current neuron
                spikes_by_state = {}
                duration_by_state = {}
                cumulative_duration_by_state = {}

                # Step 1: Iterate through segments to build concatenated spike trains
                for segment in sleep_data[epoch_key]:
                    state_name = segment['state']
                    if state_name in selections['sleep_states']:
                        start_time, end_time = segment['start_time'], segment['end_time']
                        segment_duration = end_time - start_time
                        if segment_duration <= 0: continue

                        segment_spikes = spikes_all[(spikes_all >= start_time) & (spikes_all < end_time)]
                        
                        if state_name not in spikes_by_state:
                            spikes_by_state[state_name] = []
                            duration_by_state[state_name] = 0.0
                            cumulative_duration_by_state[state_name] = 0.0

                        # Get the time offset for stitching
                        time_offset = cumulative_duration_by_state[state_name]
                        # Shift spikes relative to segment start and add to the offset
                        shifted_spikes = (segment_spikes - start_time) + time_offset
                        
                        spikes_by_state[state_name].extend(shifted_spikes)
                        duration_by_state[state_name] += segment_duration
                        cumulative_duration_by_state[state_name] += segment_duration

                # Step 2: Perform one calculation per state using the concatenated data
                for state_name, concatenated_spikes in spikes_by_state.items():
                    total_duration = duration_by_state[state_name]
                    if total_duration > 0:
                        final_spikes_array = np.array(concatenated_spikes)
                        
                        properties = calculate_firing_properties(final_spikes_array, total_duration)
                        
                        result_row = {
                            "recording": rec_info['name'], 
                            "brain_region": neuron['acronym'],
                            "cell_type": neuron['cell_type'], 
                            "cell_id": neuron['cluster_id'], 
                            "epoch": epoch_idx, 
                            "sleep_state": state_name, 
                            **properties
                        }
                        all_results.append(result_row)

    if not all_results: print("\nAnalysis complete, but no data was generated based on your selections."); return

    results_df = pd.DataFrame(all_results)
    # output_path = filedialog.asksaveasfilename(title="Save Analysis Results As", defaultextension=".csv", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
    
    print("\nAutomatically saving results...")
    output_filename = "concatenated_firing_properties.csv"
    output_path = data_dir / output_filename
    
    if output_path:
        results_df.to_csv(output_path, index=False, float_format='%.4f')
        print(f"\nSUCCESS: Analysis complete. Results saved to:\n{output_path}")
    else:
        print("\nAnalysis complete. No output file was saved.")

if __name__ == "__main__":
    main()
    input("\nPress Enter to exit.")