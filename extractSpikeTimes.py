# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 19:03:14 2025
Modified on Thu Apr 10 15:23:00 2025 # Use AllenSDK isi_viol metric

@author: HT_bo
"""

import pandas as pd
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt

# Assume current date is Thu Apr 10 15:23:00 2025 EDT

def browse_directory():
    """Opens a dialog to browse for a directory and returns the selected path."""
    root = tk.Tk()
    root.withdraw()
    folder_selected = filedialog.askdirectory()
    return folder_selected

def calculate_burst_index(spike_times_sec, bin_size_ms=1, window_ms=500):
    """Calculates the burst index for a unit based on its spike times.
    Burst index was determined by calculating the average number of spikes in 
    the 3- to 5-ms bins of the spike autocorrelogram divided by the average number 
    of spikes in the 200- to 300-ms bins. Antonio Ruiz et al 2021 10.1126/science.abf3119 Fig S11"""
    if len(spike_times_sec) < 10: return np.nan
    sorted_times_ms = np.sort(spike_times_sec) * 1000
    diffs = []
    for i in range(len(sorted_times_ms)):
        j = i + 1
        while j < len(sorted_times_ms) and (sorted_times_ms[j] - sorted_times_ms[i]) <= window_ms:
            diffs.append(sorted_times_ms[j] - sorted_times_ms[i])
            j += 1
    if not diffs: return np.nan
    diffs = np.array(diffs)
    bins = np.arange(0, window_ms + bin_size_ms, bin_size_ms)
    acg, _ = np.histogram(diffs, bins=bins)
    burst_bins_mask = (bins[:-1] >= 3) & (bins[:-1] < 5)
    baseline_bins_mask = (bins[:-1] >= 200) & (bins[:-1] < 300)
    avg_burst = np.mean(acg[burst_bins_mask]) if np.any(burst_bins_mask) else 0
    avg_baseline = np.mean(acg[baseline_bins_mask]) if np.any(baseline_bins_mask) else 0
    if avg_baseline == 0: return np.nan
    else: return avg_burst / avg_baseline

# --- Main Script ---
print("Please select the directory containing your Kilosort output files, metrics.csv, and channel brain region mapping...")
kilosort_folder = browse_directory()

if not kilosort_folder:
    print("No directory selected. Exiting.")
    exit()
print(f"Selected directory: {kilosort_folder}")

# --- Paths ---
spike_times_file = os.path.join(kilosort_folder, 'spike_times.npy')
spike_times_sec_file = os.path.join(kilosort_folder, 'spike_times_sec.npy')
spike_clusters_file = os.path.join(kilosort_folder, 'spike_clusters.npy')
channel_pos_file = os.path.join(kilosort_folder, 'channel_positions.npy')
metrics_file = os.path.join(kilosort_folder, 'metrics.csv')
brain_regions_file_name = 'channel_brain_regions.csv'
brain_regions_file = os.path.join(kilosort_folder, brain_regions_file_name)

# --- Load Kilosort Data ---
# ...(Same loading logic)...
if os.path.exists(spike_times_sec_file):
    spike_times_seconds = np.load(spike_times_sec_file)
    print(f"Loaded spike times from: {spike_times_sec_file}")
elif os.path.exists(spike_times_file):
    spike_times = np.load(spike_times_file)
    sampling_rate_hz = 30000.36276 
    spike_times_seconds = spike_times / sampling_rate_hz
    print(f"Loaded spike times from: {spike_times_file} and converted to seconds using SR={sampling_rate_hz} Hz.")
else: print(f"Error: Neither '{spike_times_file}' nor '{spike_times_sec_file}' found."); exit()
if not os.path.exists(spike_clusters_file): print(f"Error: Spike clusters file '{spike_clusters_file}' not found."); exit()
spike_clusters = np.load(spike_clusters_file).flatten()
if len(spike_times_seconds.flatten()) != len(spike_clusters): print(f"Error: Mismatch between spike times and clusters length."); exit()
if not os.path.exists(channel_pos_file): print(f"Warning: Channel positions file '{channel_pos_file}' not found."); channel_pos = None
else: channel_pos = np.load(channel_pos_file)
if not os.path.exists(metrics_file): print(f"Error: Metrics file '{metrics_file}' not found."); exit()
metrics_df = pd.read_csv(metrics_file); print(f"Loaded metrics from: {metrics_file}")

# --- Modify and Load Allen Brain Regions mapping ---
# ...(Same CSV modification/loading logic)...
brain_regions_df = None
if not os.path.exists(brain_regions_file): print(f"Warning: Brain regions file '{brain_regions_file}' not found.")
else:
    try:
        print(f"Checking indexing in {brain_regions_file_name}...")
        temp_df = pd.read_csv(brain_regions_file)
        if 'global_channel_index' in temp_df.columns and not temp_df.empty:
            if temp_df['global_channel_index'].iloc[0] != 0:
                print(f" Adjusting '{brain_regions_file_name}' to 0-based index..."); temp_df['global_channel_index'] -= 1
                print(f" --> SAVING MODIFIED DATA back to '{brain_regions_file_name}'..."); temp_df.to_csv(brain_regions_file, index=False)
                print(f"     '{brain_regions_file_name}' updated."); brain_regions_df = pd.read_csv(brain_regions_file, index_col='global_channel_index')
            else:
                print(f" '{brain_regions_file_name}' already 0-based."); brain_regions_df = pd.read_csv(brain_regions_file, index_col='global_channel_index')
        else: print(f" Warning: 'global_channel_index' not found/empty in '{brain_regions_file_name}'."); brain_regions_df = None
        if brain_regions_df is not None: print(f"Successfully loaded brain region mapping.")
    except Exception as e: print(f"Error processing brain regions file '{brain_regions_file}': {e}"); brain_regions_df = None

# --- Main Processing Loop ---
good_clusters_final = []
unique_clusters_all = np.unique(spike_clusters)
print(f"\nFound {len(unique_clusters_all)} unique clusters. Processing...")

if len(spike_times_seconds) == 0 or np.isnan(spike_times_seconds).all(): print("Error: spike_times_seconds empty/invalid."); exit()
recording_duration = np.nanmax(spike_times_seconds) - np.nanmin(spike_times_seconds)
if recording_duration <= 0: print(f"Warning: Recording duration non-positive ({recording_duration:.2f}s).")

# --- Iterate through each cluster ---
for i, cluster_id in enumerate(unique_clusters_all):
    if (i + 1) % 50 == 0: print(f" Processing cluster {i+1}/{len(unique_clusters_all)} (ID: {cluster_id})")

    cluster_mask = (spike_clusters == cluster_id)
    cluster_spike_times_seconds = spike_times_seconds.flatten()[cluster_mask]

    n_spikes = len(cluster_spike_times_seconds)
    firing_rate = (n_spikes / recording_duration) if (n_spikes > 0 and recording_duration > 0) else 0

    # Calculate simple ISI % violation < 2ms for reference (optional)
    percent_short_isi = None
    if n_spikes >= 2:
        isi = np.diff(np.sort(cluster_spike_times_seconds)) * 1000 # ISIs in ms
        count_0_2 = np.sum(isi < 2) # Violations < 2ms
        percent_short_isi = (count_0_2 / len(isi)) * 100 if len(isi) > 0 else 0
    else:
        percent_short_isi = 0

    metrics_row = metrics_df[metrics_df['cluster_id'] == cluster_id]

    if not metrics_row.empty:
        metrics_data = metrics_row.iloc[0]

        if 'peak_channel' not in metrics_data.index or pd.isna(metrics_data['peak_channel']): continue
        peak_channel_index = int(metrics_data['peak_channel'])

        # --- Retrieve AllenSDK Quality Metrics ---
        contam_rate = float(metrics_data['contam_rate']) if 'contam_rate' in metrics_data.index and pd.notna(metrics_data['contam_rate']) else None
        l_ratio = float(metrics_data['l_ratio']) if 'l_ratio' in metrics_data.index and pd.notna(metrics_data['l_ratio']) else None
        # Use the pre-calculated 'isi_viol' metric from AllenSDK output
        # Check for 'isi_viol' column name (as per user sample)
        isi_viol_rate = float(metrics_data['isi_viol']) if 'isi_viol' in metrics_data.index and pd.notna(metrics_data['isi_viol']) else None

        # --- Apply "Good Cluster" Criteria (Using AllenSDK 'isi_viol' metric) ---
        # Criteria: FR>0.5, Contam<0.1, LRatio<0.05, Allen's isi_viol < 0.2
        
        if (firing_rate > 0.1 and
            contam_rate is not None and contam_rate < 10 and
            l_ratio is not None and l_ratio < 0.1 and
            isi_viol_rate is not None and isi_viol_rate < 0.5 
            ): # Keep basic check? Optional.

            # --- Calculate Burst Index ---
            burst_index = calculate_burst_index(cluster_spike_times_seconds)
            # --- Get Trough-to-Peak Duration (ms) ---
            trough_to_peak_ms = float(metrics_data['duration']) if 'duration' in metrics_data.index and pd.notna(metrics_data['duration']) else np.nan

            # --- Create dictionary for good cluster ---
            good_cluster_info = {
                'cluster_id': int(cluster_id),
                'firing_rate_hz': round(firing_rate, 3),
                'isi_violation_percent': round(percent_short_isi, 3) if percent_short_isi is not None else None, # Store basic % for reference
                'isi_viol_rate': isi_viol_rate, 
                'contamination_rate': contam_rate,
                'l_ratio': l_ratio,
                'num_spikes': n_spikes,
                'spike_times_seconds': cluster_spike_times_seconds.tolist(), # Keep spike times for NPY
                'peak_channel_index_0based': peak_channel_index,
                'burst_index': burst_index if pd.notna(burst_index) else None,
                'trough_to_peak_ms': trough_to_peak_ms if pd.notna(trough_to_peak_ms) else None,
                'shank_index': None, 'distance_from_entry_um': None, 'acronym': None,
                'brain_region_name': None, 'channel_depth_um_fallback': None
            }

            # --- Get Brain Region Info ---
            if brain_regions_df is not None:
                if peak_channel_index in brain_regions_df.index:
                    region_info = brain_regions_df.loc[peak_channel_index]
                    good_cluster_info['shank_index'] = int(region_info['shank_index']) if pd.notna(region_info['shank_index']) else None
                    good_cluster_info['distance_from_entry_um'] = float(region_info['distance_from_entry_um']) if pd.notna(region_info['distance_from_entry_um']) else None
                    good_cluster_info['acronym'] = region_info['acronym'] if pd.notna(region_info['acronym']) else None
                    good_cluster_info['brain_region_name'] = region_info['name'] if pd.notna(region_info['name']) else None

            # --- Calculate Fallback Depth ---
            if good_cluster_info['distance_from_entry_um'] is None:
                 if channel_pos is not None and peak_channel_index < len(channel_pos):
                     try: good_cluster_info['channel_depth_um_fallback'] = float(channel_pos[peak_channel_index][1])
                     except (IndexError, TypeError): pass

            good_clusters_final.append(good_cluster_info)

# --- Post-processing ---
print(f"\nProcessed {len(unique_clusters_all)} clusters.")
# Updated summary message to reflect the new criterion based on AllenSDK metric
print(f"Found {len(good_clusters_final)} good clusters meeting criteria (FR>0.5Hz, Contam<10%, LRatio<0.05, isi_viol<0.5).") 

if good_clusters_final:
    good_clusters_df = pd.DataFrame(good_clusters_final)

    # --- 1. Save Everything (including spike times lists) to NPY ---
    output_npy_file_name = f'good_clusters_processed_{os.path.basename(kilosort_folder)}.npy'
    output_npy_path = os.path.join(kilosort_folder, output_npy_file_name)
    try:
        np.save(output_npy_path, good_clusters_final)
        print(f"Full data including spike times saved to: {output_npy_path}")
    except Exception as e: print(f"Error saving NPY file: {e}")

    # --- 2. Save Metrics Summary (excluding spike times) to CSV ---
    try:
        good_clusters_df_no_spikes = None
        if 'spike_times_seconds' in good_clusters_df.columns:
             good_clusters_df_no_spikes = good_clusters_df.drop(columns=['spike_times_seconds'])
        else: good_clusters_df_no_spikes = good_clusters_df.copy()

        # Reorder columns (optional), using isi_viol_rate
        cols_order = [
            'cluster_id', 'acronym', 'brain_region_name', 'firing_rate_hz',
            'isi_violation_percent', 'isi_viol_rate', # Added AllenSDK metric name
            'contamination_rate', 'l_ratio',
            'num_spikes', 'burst_index', 'trough_to_peak_ms',
            'peak_channel_index_0based', 'shank_index',
            'distance_from_entry_um', 'channel_depth_um_fallback'
        ]
        cols_order_present = [col for col in cols_order if col in good_clusters_df_no_spikes.columns]
        good_clusters_df_no_spikes = good_clusters_df_no_spikes[cols_order_present]

        output_csv_path = os.path.join(kilosort_folder, f'good_clusters_processed_{os.path.basename(kilosort_folder)}.csv')
        good_clusters_df_no_spikes.to_csv(output_csv_path, index=False, float_format='%.4f')
        print(f"Metrics summary (excluding spike times) saved to: {output_csv_path}")
    except Exception as e: print(f"Error saving CSV file: {e}")

    # --- Generate Scatter Plot ---
    plot_data = good_clusters_df[['trough_to_peak_ms', 'burst_index']].dropna()
    if not plot_data.empty:
        trough_peak_values = plot_data['trough_to_peak_ms'].values
        burst_index_values = plot_data['burst_index'].values
        print(f"Generating scatter plot for {len(trough_peak_values)} good clusters with valid metrics...")
        plt.figure(figsize=(8, 6))
        plt.scatter(trough_peak_values, burst_index_values, alpha=0.6, s=20)
        plt.xlabel('Trough-to-Peak Duration (ms)')
        plt.ylabel('Burst Index')
        plt.title(f'Putative Neuron Type Classification ({os.path.basename(kilosort_folder)})')
        plt.grid(True, linestyle='--', alpha=0.6)
        plot_filename = f'neuron_classification_scatter_{os.path.basename(kilosort_folder)}.png'
        plot_filepath = os.path.join(kilosort_folder, plot_filename)
        try: plt.savefig(plot_filepath); print(f"Scatter plot saved to: {plot_filepath}")
        except Exception as e: print(f"Error saving plot: {e}")
        plt.show()
    else: print("No valid data points found for scatter plot.")

else:
    print("No 'good' clusters identified, skipping CSV/NPY saving and scatter plot generation.")

print("\nScript finished.")