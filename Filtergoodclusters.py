# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 19:03:14 2025

@author: HT_bo
"""

import pandas as pd
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog

def browse_directory():
    """Opens a dialog to browse for a directory and returns the selected path."""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    folder_selected = filedialog.askdirectory()
    return folder_selected

# --- Let the user select the directory ---
print("Please select the directory containing your Kilosort output files and metrics.csv...")
kilosort_folder = browse_directory()

if not kilosort_folder:
    print("No directory selected. Exiting.")
    exit()

print(f"Selected directory: {kilosort_folder}")

# --- Paths to your Kilosort output files ---
spike_times_file = os.path.join(kilosort_folder, 'spike_times.npy')
spike_times_sec_file = os.path.join(kilosort_folder, 'spike_times_sec.npy')
spike_clusters_file = os.path.join(kilosort_folder, 'spike_clusters.npy')
channel_pos_file = os.path.join(kilosort_folder, 'channel_positions.npy')
metrics_file = os.path.join(kilosort_folder, 'metrics.csv')
channel_shank_file = os.path.join(kilosort_folder, 'channel_shanks.npy')  # Assuming the file is named 'channel_shanks.npy'

# --- Load the data ---
if os.path.exists(spike_times_sec_file):
    spike_times_seconds = np.load(spike_times_sec_file)
    print(f"Loaded spike times from: {spike_times_sec_file}")
elif os.path.exists(spike_times_file):
    spike_times = np.load(spike_times_file)
    spike_times_seconds = spike_times / 30000.36276  # Assuming a sampling rate of 30kHz, adjust if different
    print(f"Loaded spike times from: {spike_times_file} and converted to seconds.")
else:
    print(f"Error: Neither '{spike_times_file}' nor '{spike_times_sec_file}' found in the selected directory.")
    exit()

if not os.path.exists(spike_clusters_file):
    print(f"Error: Spike clusters file '{spike_clusters_file}' not found in the selected directory.")
    exit()
spike_clusters = np.load(spike_clusters_file)

if not os.path.exists(channel_pos_file):
    print(f"Warning: Channel positions file '{channel_pos_file}' not found in the selected directory. Channel depth and probe area will not be included.")
    channel_pos = None
else:
    channel_pos = np.load(channel_pos_file)

if not os.path.exists(metrics_file):
    print(f"Error: Metrics file '{metrics_file}' not found in the selected directory. Firing rate and peak channel information are needed.")
    exit()
metrics_df = pd.read_csv(metrics_file)

if not os.path.exists(channel_shank_file):
    print(f"Warning: Channel shanks file '{channel_shank_file}' not found in the selected directory. Channel shank information will not be included.")
    channel_shank = None
else:
    channel_shank = np.load(channel_shank_file)

good_clusters_final = []
unique_clusters_all = np.unique(spike_clusters)
print(f"Number of unique clusters found: {len(unique_clusters_all)}")

# --- Iterate through each cluster ID found in spike_clusters ---
for cluster_id in unique_clusters_all:
    cluster_spike_times_seconds = spike_times_seconds[spike_clusters == cluster_id]

    # --- 1. Calculate Firing Rate ---
    if len(cluster_spike_times_seconds) > 0:
        duration = np.max(spike_times_seconds) - np.min(spike_times_seconds)
        firing_rate = len(cluster_spike_times_seconds) / duration
    else:
        firing_rate = 0

    # --- 2. Calculate ISI Violations (% < 2ms) ---
    if len(cluster_spike_times_seconds) >= 2:
        isi = np.diff(cluster_spike_times_seconds) * 1000  # Convert to milliseconds
        short_isi_count = np.sum(isi < 2)  # Count ISIs less than 2ms
        total_isi = len(isi)
        percent_short_isi = (short_isi_count / total_isi) * 100 if total_isi > 0 else 0
    else:
        percent_short_isi = 0

    print(f"Cluster ID: {cluster_id}, Firing Rate: {firing_rate:.2f} Hz, ISI Violations (<2ms): {percent_short_isi:.2f}%")

    # --- Get Peak Channel from metrics.csv ---
    metrics_row = metrics_df[metrics_df['cluster_id'] == cluster_id]
    if not metrics_row.empty:
        peak_channel_index = metrics_row['peak_channel'].iloc[0]

        # --- Apply Criteria ---
        if firing_rate > 0.5 and percent_short_isi <= 1: # Corrected condition
            good_cluster_info = {
                'cluster_id': int(cluster_id),
                'spike_times_seconds': cluster_spike_times_seconds.tolist(),
                'peak_channel_index': int(peak_channel_index) if pd.notna(peak_channel_index) else None
            }

            # --- 4. Calculate Channel Depth ---
            if channel_pos is not None and good_cluster_info['peak_channel_index'] is not None and good_cluster_info['peak_channel_index'] < len(channel_pos):
                good_cluster_info['channel_depth_um'] = channel_pos[good_cluster_info['peak_channel_index']][1]  # Assuming y-coordinate is depth
            else:
                good_cluster_info['channel_depth_um'] = None

            # --- 6. Get Channel Shank ---
            if channel_shank is not None and good_cluster_info['peak_channel_index'] is not None and good_cluster_info['peak_channel_index'] < len(channel_shank):
                good_cluster_info['peak_channel_shank'] = int(channel_shank[good_cluster_info['peak_channel_index']])
            else:
                good_cluster_info['peak_channel_shank'] = None

            # --- 5. Optional: Determine Probe Area based on channel depth ---
            if 'channel_depth_um' in good_cluster_info and good_cluster_info['channel_depth_um'] is not None:
                depth = good_cluster_info['channel_depth_um']
                # Example probe area mapping (adjust based on your probe)
                if depth < 500:
                    good_cluster_info['probe_area'] = 'Superficial'
                elif 500 <= depth < 1000:
                    good_cluster_info['probe_area'] = 'Middle'
                else:
                    good_cluster_info['probe_area'] = 'Deep'
            else:
                good_cluster_info['probe_area'] = None

            good_clusters_final.append(good_cluster_info)
            print(f"Found a good cluster: {cluster_id}")

# --- Save the good clusters data to a .npy file ---
output_file = 'good_clusters_final.npy'
np.save(output_file, good_clusters_final)

print(f"Information for good clusters (FR > 0.5 Hz and < 1% ISI > 2ms) saved to: {output_file}")