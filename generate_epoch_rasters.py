# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 13:53:54 2025

@author: HT_bo

This script integrates processed neuronal unit data with epoch timestamps to
generate detailed, epoch-aligned raster plots and quantitative reports.

It requires two input files:
1. The output from the ACG analysis script (good_clusters_processed..._CellExplorerACG.npy),
   which contains unit properties and spike times.
2. The output from the timestamp extraction script (..._timestamps.npy), which
   contains the start and end times for each experimental epoch.

For each epoch, it generates a TIFF image containing:
- A raster plot of spikes, with units sorted by area, type, and firing rate.
- An average firing rate trace for each cell type group.
- A bar plot of the average Coefficient of Variation (CV) for each group.

It also produces two CSV files:
- A detailed report of each neuron's firing rate in every epoch.
- A summary report of the average CV for each cell type/area group per epoch.

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
# Define colors and order for cell types, consistent with previous script
CELL_TYPE_COLORS = {
    'NS Interneuron': 'blue',
    'WS Interneuron': 'green',
    'Pyramidal': 'red',
    'Unclassified': 'gray'
}
CELL_TYPE_ORDER = ['NS Interneuron', 'WS Interneuron', 'Pyramidal', 'Unclassified']

# Configuration for the firing rate trace
FIRING_RATE_BIN_SIZE_SEC = 1.0

# --- Helper Functions ---

def select_file(title):
    """Opens a dialog to select a file and returns its path."""
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
    """Calculates the Coefficient of Variation (CV) for a set of spike times."""
    if len(spike_times_sec) < 2:
        return np.nan
    isis = np.diff(np.sort(spike_times_sec))
    if np.mean(isis) == 0:
        return np.nan
    return np.std(isis) / np.mean(isis)

def create_output_directory(base_path, dir_name="Raster_Analysis_Output"):
    """Creates a directory to save the output files."""
    output_dir = base_path / dir_name
    output_dir.mkdir(exist_ok=True)
    print(f"Output will be saved to: {output_dir}")
    return output_dir

# --- Main Analysis Function ---

def generate_epoch_rasters():
    """Main function to load data, process epochs, and generate plots/reports."""
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
        # Convert to DataFrame for easier manipulation
        units_df = pd.DataFrame(list(unit_data_raw))
        print(f"  Loaded {len(units_df)} total units.")
        # Ensure required columns are present
        required_cols = ['cluster_id', 'cell_type', 'acronym', 'spike_times_sec', 'firing_rate_hz', 'burst_index']
        if not all(col in units_df.columns for col in required_cols):
            print(f"Error: Unit data file is missing one of the required columns: {required_cols}")
            return
    except Exception as e:
        print(f"Error loading or processing unit data file: {e}")
        return

    try:
        print(f"Loading timestamp data from: {timestamp_path.name}")
        timestamps_data = np.load(timestamp_path, allow_pickle=True).item()
        epochs = timestamps_data.get('EpochFrameData')
        if not epochs or len(epochs) == 0:
            print("Error: No 'EpochFrameData' found in the timestamp file. Cannot proceed.")
            return
        print(f"  Found {len(epochs)} epochs to analyze.")
    except Exception as e:
        print(f"Error loading or processing timestamp file: {e}")
        return

    # 2. --- Setup Output ---
    output_dir = create_output_directory(unit_data_path.parent)
    all_epochs_firing_rates = []
    all_epochs_cv_summary = []
    
    # Convert cell_type to a categorical for stable sorting
    units_df['cell_type'] = pd.Categorical(units_df['cell_type'], categories=CELL_TYPE_ORDER, ordered=True)

    # 3. --- Loop Through Epochs ---
    for epoch in epochs:
        epoch_idx = epoch['epoch_index']
        start_time = epoch.get('start_time_sec')
        end_time = epoch.get('end_time_sec')

        if start_time is None or end_time is None or start_time >= end_time:
            print(f"\nSkipping Epoch {epoch_idx}: Invalid start/end times.")
            continue

        print(f"\n--- Processing Epoch {epoch_idx} (Time: {start_time:.2f}s to {end_time:.2f}s) ---")
        epoch_duration = end_time - start_time

        # --- Filter data for the current epoch ---
        epoch_units_data = []
        for _, unit in units_df.iterrows():
            # Filter spikes within the epoch
            spikes_in_epoch = unit['spike_times_sec'][(unit['spike_times_sec'] >= start_time) & (unit['spike_times_sec'] < end_time)]
            
            # Calculate epoch-specific metrics
            epoch_fr = len(spikes_in_epoch) / epoch_duration
            epoch_cv = calculate_cv(spikes_in_epoch)

            unit_epoch_info = unit.to_dict()
            unit_epoch_info['epoch_spikes'] = spikes_in_epoch
            unit_epoch_info['epoch_firing_rate'] = epoch_fr
            unit_epoch_info['epoch_cv'] = epoch_cv
            epoch_units_data.append(unit_epoch_info)

        if not epoch_units_data:
            print(f"No unit data for Epoch {epoch_idx}. Skipping.")
            continue
            
        epoch_df = pd.DataFrame(epoch_units_data)
        
        # Add to the global firing rate report list
        for _, row in epoch_df.iterrows():
            all_epochs_firing_rates.append({
                'cluster_id': row['cluster_id'],
                'epoch_index': epoch_idx,
                'cell_type': row['cell_type'],
                'acronym': row['acronym'],
                'epoch_firing_rate_hz': row['epoch_firing_rate'],
                'burst_index': row['burst_index']
            })

        # --- Sort neurons for plotting ---
        # Sort by: Area (acronym), Neuron Type, Firing Rate (overall)
        plot_df = epoch_df.sort_values(by=['acronym', 'cell_type', 'firing_rate_hz'], ascending=[True, True, False]).reset_index(drop=True)
        print(f"  Sorted {len(plot_df)} units for plotting.")

        # --- Plotting Setup ---
        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 1], hspace=0.1)
        ax_raster = fig.add_subplot(gs[0])
        ax_fr = fig.add_subplot(gs[1], sharex=ax_raster)
        ax_cv = fig.add_subplot(gs[2], sharex=ax_raster)
        
        fig.suptitle(f"Neuronal Activity during Epoch {epoch_idx} ({start_time:.2f}s - {end_time:.2f}s)", fontsize=16, y=0.98)

        # --- Generate Raster Plot ---
        y_ticks, y_tick_labels = [], []
        area_boundaries = plot_df.groupby('acronym').tail(1).index.values
        type_boundaries = plot_df.groupby(['acronym', 'cell_type']).tail(1).index.values

        for i, unit in plot_df.iterrows():
            color = CELL_TYPE_COLORS.get(unit['cell_type'], 'gray')
            spikes_to_plot = unit['epoch_spikes']
            # Offset spikes relative to epoch start time for plotting
            ax_raster.eventplot(spikes_to_plot - start_time, lineoffsets=i, linelengths=0.8, color=color)

        # Draw boundary lines and labels
        for boundary in area_boundaries[:-1]:
            ax_raster.axhline(y=boundary + 0.5, color='black', linestyle='-', linewidth=1.5)
        for boundary in type_boundaries[:-1]:
            ax_raster.axhline(y=boundary + 0.5, color='dimgray', linestyle=':', linewidth=0.8)

        # Set y-axis labels for brain areas
        for area_name, area_df in plot_df.groupby('acronym'):
            middle_idx = area_df.index.min() + (area_df.index.max() - area_df.index.min()) / 2
            y_ticks.append(middle_idx)
            y_tick_labels.append(area_name if pd.notna(area_name) else "N/A")

        ax_raster.set_yticks(y_ticks)
        ax_raster.set_yticklabels(y_tick_labels, rotation=0, fontsize=10)
        ax_raster.set_ylabel("Brain Area / Units", fontsize=12)
        ax_raster.set_ylim(-1, len(plot_df))
        plt.setp(ax_raster.get_xticklabels(), visible=False) # Hide x-axis labels on raster
        ax_raster.grid(axis='x', linestyle=':', alpha=0.5)

        # --- Generate Average Firing Rate Plot ---
        time_bins = np.arange(0, epoch_duration + FIRING_RATE_BIN_SIZE_SEC, FIRING_RATE_BIN_SIZE_SEC)
        bin_centers = (time_bins[:-1] + time_bins[1:]) / 2

        grouped = plot_df.groupby(['acronym', 'cell_type'])
        for (area, cell_type), group_df in grouped:
            if group_df.empty: continue
            
            # Aggregate all spikes from this group and offset them
            all_group_spikes = np.concatenate(group_df['epoch_spikes'].values) - start_time
            
            # Calculate histogram
            spike_counts, _ = np.histogram(all_group_spikes, bins=time_bins)
            
            # Normalize by bin size and number of neurons in the group
            avg_fr = spike_counts / (FIRING_RATE_BIN_SIZE_SEC * len(group_df))
            
            color = CELL_TYPE_COLORS.get(cell_type, 'gray')
            label = f"{area or 'N/A'} - {cell_type}"
            ax_fr.plot(bin_centers, avg_fr, color=color, label=label, linewidth=1.5)

        ax_fr.set_ylabel("Avg. Firing Rate (Hz)", fontsize=12)
        ax_fr.grid(axis='y', linestyle=':', alpha=0.5)
        plt.setp(ax_fr.get_xticklabels(), visible=False)
        ax_fr.legend(loc='upper right', fontsize=8, ncol=2)

        # --- Generate CV Bar Plot ---
        cv_summary = plot_df.groupby(['acronym', 'cell_type'])['epoch_cv'].mean().dropna()
        bar_labels = [f"{idx[0] or 'N/A'}-{idx[1]}" for idx in cv_summary.index]
        bar_colors = [CELL_TYPE_COLORS.get(idx[1], 'gray') for idx in cv_summary.index]
        
        # Append CV data to the global summary list
        for (area, cell_type), mean_cv in cv_summary.items():
            all_epochs_cv_summary.append({
                'epoch_index': epoch_idx,
                'acronym': area,
                'cell_type': cell_type,
                'mean_cv': mean_cv
            })

        ax_cv.bar(range(len(cv_summary)), cv_summary.values, color=bar_colors)
        ax_cv.set_xticks(range(len(cv_summary)))
        ax_cv.set_xticklabels(bar_labels, rotation=45, ha='right', fontsize=8)
        ax_cv.set_ylabel("Mean CV of ISI", fontsize=12)
        ax_cv.set_xlabel("Time (s)", fontsize=12)
        ax_cv.set_xlim(-0.5, epoch_duration) # Align x-axis with plots above

        # --- Finalize and Save Figure ---
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        output_image_path = output_dir / f"Epoch_{epoch_idx}_RasterAnalysis.tiff"
        try:
            fig.savefig(output_image_path, dpi=300, format='tiff', bbox_inches='tight')
            print(f"  SUCCESS: Plot for Epoch {epoch_idx} saved to {output_image_path.name}")
        except Exception as e:
            print(f"  ERROR: Could not save plot for Epoch {epoch_idx}. Reason: {e}")
        plt.close(fig) # Close figure to free memory

    # 4. --- Save CSV Reports ---
    if all_epochs_firing_rates:
        fr_df = pd.DataFrame(all_epochs_firing_rates)
        fr_csv_path = output_dir / "firing_rates_per_epoch.csv"
        fr_df.to_csv(fr_csv_path, index=False, float_format='%.4f')
        print(f"\nSUCCESS: Firing rate report saved to {fr_csv_path.name}")

    if all_epochs_cv_summary:
        cv_df = pd.DataFrame(all_epochs_cv_summary)
        cv_csv_path = output_dir / "cv_summary_per_epoch.csv"
        cv_df.to_csv(cv_csv_path, index=False, float_format='%.4f')
        print(f"SUCCESS: CV summary report saved to {cv_csv_path.name}")

    print("\n--- Analysis Finished ---")


if __name__ == "__main__":
    # Suppress RuntimeWarning from mean of empty slice in CV calculation
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
    generate_epoch_rasters()
    input("\nPress Enter to exit.")
