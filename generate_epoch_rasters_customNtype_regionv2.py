# -*- coding: utf-8 -*-
"""
Created on Thu Aug 07 14:00:00 2025

This script integrates processed neuronal unit data with epoch timestamps to
generate detailed, epoch-aligned raster plots and quantitative reports.

It requires two input files:
1. The output from the ACG analysis script (good_clusters_processed..._CellExplorerACG.npy)
2. The output from the timestamp extraction script (..._timestamps.npy)

For each epoch, it generates a TIFF image containing:
- A binned heatmap showing firing rate over time for each neuron.
- A firing rate plot with semi-transparent lines for individual neurons and a
  bold line for the group average.

It also produces a CSV file with each neuron's firing rate and CV per epoch.
"""

import pandas as pd
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import warnings

# --- Configuration ---
CELL_TYPE_COLORS = {
    'NS Interneuron': 'blue',
    'WS Interneuron': 'green',
    'Pyramidal': 'red',
    'Unclassified': 'gray'
}
CELL_TYPE_ORDER = ['NS Interneuron', 'WS Interneuron', 'Pyramidal', 'Unclassified']

# --- MODIFICATION: Bin size for raster heatmap and firing rate plots ---
RASTER_HEATMAP_BIN_SIZE_SEC = 1.0 # Bin size in seconds for the raster plot
FIRING_RATE_BIN_SIZE_SEC = 10.0    # Bin size in seconds for the FR plot

# --- Helper Functions ---

def select_file(title):
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    file_path = filedialog.askopenfilename(title=title)
    root.destroy()
    if not file_path:
        print(f"No file selected for: {title}. Exiting.")
        return None
    return Path(file_path)

def calculate_cv(spike_times_sec):
    if len(spike_times_sec) < 2:
        return np.nan
    isis = np.diff(np.sort(spike_times_sec))
    if np.mean(isis) == 0:
        return np.nan
    return np.std(isis) / np.mean(isis)

def create_output_directory(base_path, dir_name="Raster_Analysis_Output"):
    output_dir = base_path / dir_name
    output_dir.mkdir(exist_ok=True)
    print(f"Output will be saved to: {output_dir}")
    return output_dir

def get_user_filter(options, item_name):
    if not options:
        return []
    print(f"\n--- Filter by {item_name} ---")
    print(f"Available {item_name}s: {', '.join(options)}")
    prompt = f"Enter {item_name}s to include (comma-separated), or press Enter for all: "
    user_input = input(prompt).strip()
    if not user_input:
        print(f"Including all {item_name}s.")
        return []
    selected_items = [item.strip() for item in user_input.split(',')]
    options_lower = [opt.lower() for opt in options]
    valid_selection = []
    for item in selected_items:
        try:
            match_index = options_lower.index(item.lower())
            valid_selection.append(options[match_index])
        except ValueError:
            print(f"Warning: '{item}' is not a valid {item_name}. It will be ignored.")
    if valid_selection:
        print(f"Filtering for: {', '.join(valid_selection)}")
    else:
        print(f"No valid {item_name}s selected. Including all.")
    return valid_selection

# --- Main Analysis Function ---

def generate_epoch_rasters():
    print("--- Starting Epoch-Aligned Raster and Firing Rate Analysis ---")

    # 1. --- Load Data Files ---
    print("\nPlease select the required input files.")
    unit_data_path = select_file("Select the processed unit data file (good_clusters..._CellExplorerACG.npy)")
    if not unit_data_path: return
    timestamp_path = select_file("Select the timestamp file (..._timestamps.npy)")
    if not timestamp_path: return

    try:
        print(f"\nLoading unit data from: {unit_data_path.name}")
        unit_data_raw = np.load(unit_data_path, allow_pickle=True)
        units_df = pd.DataFrame(list(unit_data_raw))
        print(f"  Loaded {len(units_df)} total units.")
        required_cols = ['cluster_id', 'cell_type', 'acronym', 'spike_times_sec', 'firing_rate_hz', 'burst_index']
        if not all(col in units_df.columns for col in required_cols):
            print(f"Error: Unit data file is missing one of the required columns.")
            return
    except Exception as e:
        print(f"Error loading or processing unit data file: {e}")
        return

    try:
        print(f"Loading timestamp data from: {timestamp_path.name}")
        timestamps_data = np.load(timestamp_path, allow_pickle=True).item()
        epochs = timestamps_data.get('EpochFrameData')
        if not epochs:
            print("Error: No 'EpochFrameData' found in the timestamp file.")
            return
        print(f"  Found {len(epochs)} epochs to analyze.")
    except Exception as e:
        print(f"Error loading or processing timestamp file: {e}")
        return

    units_df['cell_type'] = pd.Categorical(units_df['cell_type'], categories=CELL_TYPE_ORDER, ordered=True)

    # --- Get user input for filtering ---
    available_types = sorted(units_df['cell_type'].cat.categories.tolist())
    available_regions = sorted(units_df['acronym'].dropna().unique().tolist())
    selected_types = get_user_filter(available_types, "neuron type")
    selected_regions = get_user_filter(available_regions, "region")

    if selected_types:
        units_df = units_df[units_df['cell_type'].isin(selected_types)]
    if selected_regions:
        units_df = units_df[units_df['acronym'].isin(selected_regions)]

    if units_df.empty:
        print("\nNo units remaining after filtering. Exiting.")
        return
    else:
        print(f"\nProceeding with {len(units_df)} units after filtering.")

    # 2. --- Setup Output ---
    filter_name_parts = selected_types + selected_regions
    dir_suffix = "_".join(filter_name_parts) if filter_name_parts else "All"
    output_dir_name = f"Raster_Analysis_Output_{dir_suffix}"
    output_dir = create_output_directory(unit_data_path.parent, dir_name=output_dir_name)
    all_epochs_data_for_csv = []

    # 3. --- Loop Through Epochs ---
    for epoch in epochs:
        epoch_idx = epoch['epoch_index']
        start_time = epoch.get('start_time_sec')
        end_time = epoch.get('end_time_sec')

        if start_time is None or end_time is None or start_time >= end_time:
            continue

        print(f"\n--- Processing Epoch {epoch_idx} (Time: {start_time:.2f}s to {end_time:.2f}s) ---")
        epoch_duration = end_time - start_time

        epoch_units_data = []
        for _, unit in units_df.iterrows():
            spikes_in_epoch = unit['spike_times_sec'][(unit['spike_times_sec'] >= start_time) & (unit['spike_times_sec'] < end_time)]
            epoch_fr = len(spikes_in_epoch) / epoch_duration
            epoch_cv = calculate_cv(spikes_in_epoch)

            unit_epoch_info = unit.to_dict()
            unit_epoch_info['epoch_spikes'] = spikes_in_epoch
            unit_epoch_info['epoch_firing_rate'] = epoch_fr
            unit_epoch_info['epoch_cv'] = epoch_cv
            epoch_units_data.append(unit_epoch_info)

        epoch_df = pd.DataFrame(epoch_units_data)
        for _, row in epoch_df.iterrows():
            all_epochs_data_for_csv.append({
                'cluster_id': row['cluster_id'], 'epoch_index': epoch_idx,
                'cell_type': row['cell_type'], 'acronym': row['acronym'],
                'epoch_firing_rate_hz': row['epoch_firing_rate'],
                'epoch_cv': row['epoch_cv'], 'burst_index': row['burst_index']
            })

        plot_df = epoch_df.sort_values(by=['acronym', 'cell_type', 'firing_rate_hz'], ascending=[True, True, False]).reset_index(drop=True)
        if plot_df.empty:
            print(f"  No units with activity in Epoch {epoch_idx}. Skipping plot generation.")
            continue
        print(f"  Sorted {len(plot_df)} units for plotting.")

        # --- MODIFICATION: Setup 2 plots instead of 3 ---
        fig, (ax_raster, ax_fr) = plt.subplots(2, 1, figsize=(20, 10), sharex=True, gridspec_kw={'height_ratios': [3, 2]})
        fig.suptitle(f"Neuronal Activity during Epoch {epoch_idx} ({start_time:.2f}s - {end_time:.2f}s) - Filter: {dir_suffix}", fontsize=16, y=0.98)

        # --- MODIFICATION: Generate Raster Plot as a Binned Heatmap ---
        time_bins_raster = np.arange(0, epoch_duration, RASTER_HEATMAP_BIN_SIZE_SEC)
        n_neurons = len(plot_df)
        n_bins = len(time_bins_raster) -1
        raster_matrix = np.zeros((n_neurons, n_bins))

        for i, unit in plot_df.iterrows():
            spikes_relative = unit['epoch_spikes'] - start_time
            spike_counts, _ = np.histogram(spikes_relative, bins=time_bins_raster)
            raster_matrix[i, :] = spike_counts

        im = ax_raster.imshow(raster_matrix, aspect='auto', interpolation='none', cmap='viridis',
                              extent=[0, epoch_duration, n_neurons, 0])
        fig.colorbar(im, ax=ax_raster, label=f'Spikes per {RASTER_HEATMAP_BIN_SIZE_SEC}s bin', pad=0.01)

        # --- Draw boundary lines and labels for raster ---
        y_ticks, y_tick_labels = [], []
        area_boundaries = plot_df.groupby('acronym').tail(1).index.values
        for boundary in area_boundaries[:-1]:
            ax_raster.axhline(y=boundary + 0.5, color='white', linestyle='-', linewidth=1.5)
        for area_name, area_df in plot_df.groupby('acronym'):
            middle_idx = area_df.index.min() + (area_df.index.max() - area_df.index.min()) / 2
            y_ticks.append(middle_idx)
            y_tick_labels.append(area_name if pd.notna(area_name) else "N/A")
        
        ax_raster.set_yticks(y_ticks)
        ax_raster.set_yticklabels(y_tick_labels, rotation=0, fontsize=10)
        ax_raster.set_ylabel("Brain Area / Units", fontsize=12)

        # --- MODIFICATION: Generate Firing Rate Plot with Individual and Average Traces ---
        time_bins_fr = np.arange(0, epoch_duration, FIRING_RATE_BIN_SIZE_SEC)
        bin_centers_fr = (time_bins_fr[:-1] + time_bins_fr[1:]) / 2

        # Plot individual neuron traces (semi-transparent)
        for _, unit in plot_df.iterrows():
            spikes_relative = unit['epoch_spikes'] - start_time
            spike_counts, _ = np.histogram(spikes_relative, bins=time_bins_fr)
            fr_trace = spike_counts / FIRING_RATE_BIN_SIZE_SEC
            color = CELL_TYPE_COLORS.get(unit['cell_type'], 'gray')
            ax_fr.plot(bin_centers_fr, fr_trace, color=color, alpha=0.15, linewidth=1.0)

        # Plot bold average traces on top
        grouped = plot_df.groupby(['acronym', 'cell_type'])
        for (area, cell_type), group_df in grouped:
            if group_df.empty: continue
            all_group_spikes = np.concatenate(group_df['epoch_spikes'].values) - start_time
            spike_counts, _ = np.histogram(all_group_spikes, bins=time_bins_fr)
            avg_fr = spike_counts / (FIRING_RATE_BIN_SIZE_SEC * len(group_df))
            color = CELL_TYPE_COLORS.get(cell_type, 'gray')
            label = f"{area or 'N/A'} - {cell_type} (Avg)"
            ax_fr.plot(bin_centers_fr, avg_fr, color=color, label=label, linewidth=2.5, alpha=1.0)

        ax_fr.set_ylabel("Firing Rate (Hz)", fontsize=12)
        ax_fr.set_xlabel("Time (s)", fontsize=12)
        ax_fr.grid(axis='y', linestyle=':', alpha=0.5)
        ax_fr.legend(loc='upper right', fontsize=8, ncol=2)
        ax_fr.set_xlim(0, epoch_duration)

        # --- Finalize and Save Figure ---
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        output_image_path = output_dir / f"Epoch_{epoch_idx}_RasterAnalysis.tiff"
        try:
            fig.savefig(output_image_path, dpi=300, format='tiff', bbox_inches='tight')
            print(f"  SUCCESS: Plot for Epoch {epoch_idx} saved to {output_image_path.name}")
        except Exception as e:
            print(f"  ERROR: Could not save plot for Epoch {epoch_idx}. Reason: {e}")
        plt.close(fig)

    # 4. --- Save Report ---
    if all_epochs_data_for_csv:
        report_df = pd.DataFrame(all_epochs_data_for_csv)
        report_csv_path = output_dir / "neuron_data_per_epoch.csv"
        report_df.to_csv(report_csv_path, index=False, float_format='%.4f')
        print(f"\nSUCCESS: Full report for plotted neurons saved to {report_csv_path.name}")

    print("\n--- Analysis Finished ---")

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
    generate_epoch_rasters()
    input("\nPress Enter to exit.")