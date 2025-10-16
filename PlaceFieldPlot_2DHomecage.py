# -*- coding: utf-8 -*-
"""
Calculates, filters, saves, and plots place fields for neuronal clusters.

Combines spike data (.npy) with animal tracking data from DLC (.csv),
using precise frame timestamps from a NIDAQ alignment file (.npy)
for accurate alignment.

Based on user-provided description and parameters.
Followed Azahara Oliva et al 2019 https://doi.org/10.1038/s41586-020-2758-y
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.stats import binned_statistic_2d
import scipy.ndimage as ndi # For labelling connected regions
import os
import sys
import tkinter as tk
from tkinter import filedialog

# ==============================================================
# --- Adjust these parameters as needed ---

# --- Video/Tracking Parameters ---
PIXEL_WIDTH = 1440
PIXEL_HEIGHT = 1080
REAL_WIDTH_CM = 28.5
REAL_HEIGHT_CM = 17.0 # Ensure it's a float
# FRAME_RATE = 30.0 # No longer primary source, loaded from timestamp file

# --- Place Field Calculation Parameters ---
BIN_SIZE_CM = 1.0 # Bin width for maps (cm)
GAUSSIAN_SD_CM = 5.0 # Standard deviation for Gaussian smoothing (cm)
# Minimum time (seconds) in bin to be considered valid (using loaded FPS)
# MIN_OCCUPANCY_S = 1.0 / FRAME_RATE # Calculated dynamically now

# --- Place Field Filtering Criteria ---
MIN_FIELD_SIZE_CM2 = 15.0 # Minimum continuous area for a place field (cm^2)
RATE_THRESH_OF_PEAK = 0.10 # Firing rate must be > X% of peak rate
PEAK_RATE_THRESH_HZ = 2.0 # Peak firing rate must be > X Hz
SPATIAL_COHERENCE_THRESH = 0.7 # Spatial coherence must be > X

# --- Tracking Data Settings ---
# Which bodypart's coordinates to use for position? (Must match CSV column names)
POS_X_COL = 'Neck_x'
POS_Y_COL = 'Neck_y'
# Which speed column to use for filtering?
SPEED_COL = 'Neck Speed (cm/s)'
# Speed threshold: only analyze data when speed is above this value (cm/s)
# Set to 0 or None to disable speed filtering.
SPEED_THRESHOLD_CM_S = 2.0

# --- Output Settings ---
OUTPUT_SUBDIR = "place_field_analysis_aligned" # Subdirectory name for results

# ==============================================================

# --- Helper Functions ---

def select_file(title="Select File", filetypes=[("All files", "*.*")]):
    """Opens a file dialog for the user to select a file."""
    root = tk.Tk()
    root.withdraw() # Hide the main tkinter window
    filepath = filedialog.askopenfilename(title=title, filetypes=filetypes)
    if not filepath:
        print("No file selected. Exiting.")
        sys.exit()
    print(f"Selected file: {filepath}")
    return filepath

def load_spike_data(filepath):
    """Loads cluster spike data from .npy file into DataFrame."""
    try:
        data_struct = np.load(filepath, allow_pickle=True)
        if data_struct.dtype.names:
            df = pd.DataFrame([dict(zip(data_struct.dtype.names, row)) for row in data_struct])
        elif isinstance(data_struct[0], dict):
             df = pd.DataFrame(list(data_struct))
        else:
             print("Error: Spike data .npy structure not recognized as structured array or list of dicts.")
             sys.exit() # More specific error handling might be needed

        print(f"Loaded spike data for {df['cluster_id'].nunique()} clusters.")
        if 'cluster_id' not in df.columns or 'spike_times_seconds' not in df.columns:
            print("Error: Spike data DataFrame missing 'cluster_id' or 'spike_times_seconds'.")
            sys.exit()
        # Ensure spike times are numeric arrays
        df['spike_times_seconds'] = df['spike_times_seconds'].apply(lambda x: np.asarray(x, dtype=float))
        return df

    except FileNotFoundError:
        print(f"Error: Spike data file not found at '{filepath}'.")
        sys.exit()
    except Exception as e:
        print(f"Error loading or processing spike data file '{filepath}': {e}")
        sys.exit()

def load_tracking_data(filepath):
    """Loads tracking data from DLC CSV file, assuming a single header row."""
    try:
        # Explicitly use the first row (index 0) as header
        # and the first column (index 0) as the row index.
        df = pd.read_csv(filepath, header=0, index_col=0)

        # --- DEBUG: Print columns immediately after loading ---
        print("Columns loaded by pandas:", list(df.columns))
        # -----------------------------------------------------

        # Define necessary columns based on configuration
        relevant_cols = [POS_X_COL, POS_Y_COL]
        if SPEED_THRESHOLD_CM_S is not None and SPEED_COL:
             relevant_cols.append(SPEED_COL)

        # Check if columns exist (case-insensitive check might be robust)
        available_cols_lower = [c.lower() for c in df.columns]
        missing_cols = []
        final_cols_to_use = {} # Store mapping from config name to actual name

        # Check for configured columns in the loaded columns
        for col_config_name in relevant_cols:
             found = False
             # Try exact match first
             if col_config_name in df.columns:
                 final_cols_to_use[col_config_name] = col_config_name
                 found = True
             else:
                 # Try case-insensitive match if exact fails
                 for actual_col_name in df.columns:
                     if actual_col_name.lower() == col_config_name.lower():
                         final_cols_to_use[col_config_name] = actual_col_name
                         found = True
                         print(f"  Info: Matched config '{col_config_name}' to actual column '{actual_col_name}' (case-insensitive).")
                         break
             if not found:
                 missing_cols.append(col_config_name)


        if missing_cols:
            print(f"\n--- ERROR ---")
            print(f"Tracking CSV missing required columns (after loading): {missing_cols}")
            print(f"Columns found in CSV by pandas: {list(df.columns)}")
            print(f"Please check the CSV file header and the CONFIGURATION section (POS_X_COL, POS_Y_COL, SPEED_COL).")
            print(f"-------------\n")
            sys.exit()

        # Select and rename columns using the found actual names
        df_selected = df[[final_cols_to_use[col] for col in relevant_cols]].copy()
        # Rename columns back to the names expected by the script configuration
        df_selected.rename(columns={v: k for k, v in final_cols_to_use.items()}, inplace=True)

        # Add frame numbers based on index (assuming index is sequential frames)
        # Reset index if it's not already simple frame numbers
        if not pd.api.types.is_integer_dtype(df_selected.index):
             print("Warning: Index column is not integer type, resetting index for frame numbers.")
             df_selected.reset_index(drop=True, inplace=True)

        df_selected.insert(0, 'frame', df_selected.index)

        print(f"Loaded tracking data with {len(df_selected)} frames.")
        return df_selected
    except FileNotFoundError:
        print(f"Error: Tracking data file not found at '{filepath}'.")
        sys.exit()
    except Exception as e:
        print(f"Error loading or processing tracking data file '{filepath}': {e}")
        # Print more details for debugging CSV issues
        if isinstance(e, pd.errors.ParserError):
             print("  Pandas ParserError: Check CSV formatting, delimiters, or extra header/footer rows.")
        elif isinstance(e, KeyError):
             print(f"  KeyError: Problem accessing expected column after loading. Columns found: {list(df.columns)}")
        sys.exit()

def load_timestamp_data(filepath):
    """Loads frame timestamp data from the alignment .npy file."""
    try:
        # This file contains a dictionary saved as a numpy object
        data = np.load(filepath, allow_pickle=True).item()
        print(f"Timestamp file successfully loaded!")

        required_keys = ['FirstFrameTimeInSec', 'FramesTimesInSec', 'FPS_CameraTTLChannel']
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
             print(f"Error: Timestamp file dictionary is missing required keys: {missing_keys}")
             print(f"Available keys: {list(data.keys())}")
             sys.exit()

        print(f"  First frame time: {data['FirstFrameTimeInSec']:.4f} s")
        print(f"  Number of frame timestamps: {len(data['FramesTimesInSec'])}")
        print(f"  Camera FPS (from TTL): {data['FPS_CameraTTLChannel']:.4f} Hz")
        return data

    except FileNotFoundError:
        print(f"Error: Timestamp file not found at '{filepath}'.")
        sys.exit()
    except Exception as e:
        print(f"Error loading or processing timestamp file '{filepath}': {e}")
        sys.exit()

def calculate_spatial_coherence(rate_map, occupancy_map, occupancy_thresh_s):
    """ Calculates spatial coherence of a rate map. """
    
    valid_occupancy_mask = occupancy_map > occupancy_thresh_s
    if not np.any(valid_occupancy_mask): return np.nan
    padded_rate_map = np.pad(rate_map, pad_width=1, mode='constant', constant_values=np.nan)
    neighbor_sum = np.zeros_like(rate_map, dtype=float)
    neighbor_count = np.zeros_like(rate_map, dtype=int)
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0: continue
            neighbor_values = padded_rate_map[1+dy : rate_map.shape[0]+1+dy, 1+dx : rate_map.shape[1]+1+dx]
            valid_neighbors = ~np.isnan(neighbor_values)
            neighbor_sum[valid_neighbors] += neighbor_values[valid_neighbors]
            neighbor_count[valid_neighbors] += 1
    mean_neighbor_rate = np.full_like(rate_map, np.nan)
    valid_counts = neighbor_count > 0
    mean_neighbor_rate[valid_counts] = neighbor_sum[valid_counts] / neighbor_count[valid_counts]
    valid_mask = valid_occupancy_mask & ~np.isnan(rate_map) & ~np.isnan(mean_neighbor_rate)
    if np.sum(valid_mask) < 2: return np.nan
    center_rates = rate_map[valid_mask]
    neighbor_rates = mean_neighbor_rate[valid_mask]
    if np.std(center_rates) < 1e-9 or np.std(neighbor_rates) < 1e-9: return 0.0
    coherence = np.corrcoef(center_rates, neighbor_rates)[0, 1]
    return coherence

# --- Main Analysis ---
if __name__ == "__main__":

    # 1. Select Input Files
    spike_filepath = select_file(
        title="Select Cluster Spike Data file (e.g., good_clusters_processed...npy)",
        filetypes=[("Numpy files", "*.npy")]
    )
    tracking_filepath = select_file(
        title="Select Tracking Data file (e.g., ...speed_data_cm_per_s.csv)",
        filetypes=[("CSV files", "*.csv")]
    )
    timestamp_filepath = select_file(
        title="Select Frame Timestamp file (e.g., NeuropixelNIdaqFile_TimeStamps...npy)",
        filetypes=[("Numpy files", "*.npy")]
    )

    # 2. Create Output Directory
    output_dir = os.path.join(os.path.dirname(tracking_filepath), OUTPUT_SUBDIR)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved in: {output_dir}")

    # 3. Load Data
    df_spikes = load_spike_data(spike_filepath)
    df_tracking = load_tracking_data(tracking_filepath)
    timestamp_data = load_timestamp_data(timestamp_filepath)

    # 4. Verify Data Consistency and Align
    frame_times_s = timestamp_data['FramesTimesInSec']
    first_frame_time_s = timestamp_data['FirstFrameTimeInSec']
    actual_fps = timestamp_data['FPS_CameraTTLChannel']

    # Check length match
    if len(df_tracking) != len(frame_times_s):
        print("\n--- ERROR ---")
        print(f"Mismatch in frame counts:")
        print(f"  Tracking CSV has {len(df_tracking)} rows (frames).")
        print(f"  Timestamp file has {len(frame_times_s)} frame timestamps.")
        print("Cannot proceed with alignment. Please check your input files.")
        print("-------------\n")
        sys.exit()

    # Assign precise timestamps to tracking data
    df_tracking['timestamp_s'] = frame_times_s

    # Filter spike times BEFORE the first frame timestamp
    print(f"\nAligning spike data: Removing spikes before first frame ({first_frame_time_s:.4f}s)...")
    original_spike_counts = df_spikes['spike_times_seconds'].apply(len).sum()
    df_spikes['spike_times_seconds'] = df_spikes['spike_times_seconds'].apply(
        lambda times: times[times >= first_frame_time_s]
    )
    aligned_spike_counts = df_spikes['spike_times_seconds'].apply(len).sum()
    print(f"  Removed {original_spike_counts - aligned_spike_counts} spikes.")

    # Optional: Remove clusters that now have zero spikes after alignment
    original_cluster_count = len(df_spikes)
    df_spikes = df_spikes[df_spikes['spike_times_seconds'].apply(len) > 0].copy() # Use .copy() to avoid SettingWithCopyWarning
    print(f"  Kept {len(df_spikes)} / {original_cluster_count} clusters with remaining spikes.\n")


    # 5. Prepare Tracking Data (Coordinates, Speed Filter)
    # Convert pixel coordinates to cm
    scale_x = REAL_WIDTH_CM / PIXEL_WIDTH
    scale_y = REAL_HEIGHT_CM / PIXEL_HEIGHT
    df_tracking['pos_x_cm'] = df_tracking[POS_X_COL] * scale_x
    df_tracking['pos_y_cm'] = df_tracking[POS_Y_COL] * scale_y
    # Flip Y? Often needed if pixel origin (0,0) is top-left but real-world Y increases upwards
    df_tracking['pos_y_cm'] = REAL_HEIGHT_CM - df_tracking['pos_y_cm']

    # --- Speed Filtering ---
    if SPEED_THRESHOLD_CM_S is not None and SPEED_COL in df_tracking.columns:
        valid_speed_mask = df_tracking[SPEED_COL] > SPEED_THRESHOLD_CM_S
        df_tracking_filtered = df_tracking[valid_speed_mask].copy() # Use .copy()
        print(f"Applied speed filter: > {SPEED_THRESHOLD_CM_S} cm/s. Kept {len(df_tracking_filtered)} / {len(df_tracking)} frames.")
        if len(df_tracking_filtered) == 0:
             print("Warning: No tracking data points above speed threshold. Place field analysis might fail.")
    else:
        df_tracking_filtered = df_tracking.copy() # Use .copy()
        print("No speed filtering applied.")

    tracking_timestamps = df_tracking_filtered['timestamp_s'].values
    tracking_pos_x = df_tracking_filtered['pos_x_cm'].values
    tracking_pos_y = df_tracking_filtered['pos_y_cm'].values

    # 6. Define Bins for Histograms
    n_bins_x = int(np.ceil(REAL_WIDTH_CM / BIN_SIZE_CM))
    n_bins_y = int(np.ceil(REAL_HEIGHT_CM / BIN_SIZE_CM))
    x_edges = np.linspace(0, n_bins_x * BIN_SIZE_CM, n_bins_x + 1)
    y_edges = np.linspace(0, n_bins_y * BIN_SIZE_CM, n_bins_y + 1)
    bin_area_cm2 = BIN_SIZE_CM * BIN_SIZE_CM

    # Define minimum occupancy threshold in seconds using loaded FPS
    MIN_OCCUPANCY_S = 1.0 / actual_fps

    # 7. Calculate Overall Occupancy Map (using filtered data)
    total_occupancy_counts, _, _, _ = binned_statistic_2d(
        tracking_pos_x, tracking_pos_y, None,
        statistic='count', bins=[x_edges, y_edges], expand_binnumbers=False)
    # Convert counts to seconds using precise FPS
    total_occupancy_s = total_occupancy_counts / actual_fps

    # Smooth the occupancy map
    sigma_bins = GAUSSIAN_SD_CM / BIN_SIZE_CM
    smoothed_occupancy_s = gaussian_filter(total_occupancy_s, sigma=sigma_bins, mode='constant', cval=0.0)
    # Avoid division by zero later: set very low occupancy to NaN
    smoothed_occupancy_s[smoothed_occupancy_s < 1e-9] = np.nan # Use a small number close to zero

    # 8. Process Each Cluster (using aligned spike data)
    results = []
    for index, row in df_spikes.iterrows():
        cluster_id = row['cluster_id']
        spike_times = row['spike_times_seconds'] # These are already filtered

        print(f"\nProcessing Cluster {cluster_id} ({len(spike_times)} aligned spikes)...")

        if len(spike_times) == 0:
             # This check is now redundant due to earlier filtering, but safe to keep
             print("  Skipping - Cluster has no aligned spikes.")
             continue

        # Find position corresponding to each spike time (using filtered tracking data & precise timestamps)
        indices = np.searchsorted(tracking_timestamps, spike_times, side='right') - 1
        valid_spike_mask = (indices >= 0) & (indices < len(tracking_timestamps))
        spike_pos_x = tracking_pos_x[indices[valid_spike_mask]]
        spike_pos_y = tracking_pos_y[indices[valid_spike_mask]]
        num_valid_spikes = len(spike_pos_x)

        print(f"  Found positions for {num_valid_spikes} spikes during movement periods.")

        if num_valid_spikes < 5:
            print("  Skipping - Too few spikes occurred during movement periods.")
            continue

        # Calculate Spike Map for this cluster
        raw_spike_map, _, _, _ = binned_statistic_2d(
            spike_pos_x, spike_pos_y, None,
            statistic='count', bins=[x_edges, y_edges], expand_binnumbers=False)

        # Smooth the spike map
        smoothed_spike_map = gaussian_filter(raw_spike_map, sigma=sigma_bins, mode='constant', cval=0.0)

        # Calculate Rate Map
        with np.errstate(divide='ignore', invalid='ignore'):
            rate_map_hz = smoothed_spike_map / smoothed_occupancy_s
        rate_map_hz[np.isnan(rate_map_hz)] = 0

        # --- Apply Place Field Criteria ---
        peak_rate = np.nanmax(rate_map_hz) if np.any(~np.isnan(rate_map_hz)) else 0
        # Calculate mean rate only over bins with valid occupancy
        valid_occupancy_mask_for_mean = smoothed_occupancy_s > MIN_OCCUPANCY_S
        if np.nansum(smoothed_occupancy_s[valid_occupancy_mask_for_mean]) > 0:
             mean_rate_overall = np.nansum(rate_map_hz[valid_occupancy_mask_for_mean] * smoothed_occupancy_s[valid_occupancy_mask_for_mean]) / np.nansum(smoothed_occupancy_s[valid_occupancy_mask_for_mean])
        else:
            mean_rate_overall = 0

        print(f"  Peak Rate: {peak_rate:.2f} Hz, Mean Rate (valid bins): {mean_rate_overall:.2f} Hz")

        is_place_cell = False
        field_properties = {}

        if peak_rate > PEAK_RATE_THRESH_HZ:
            potential_field_mask = rate_map_hz > (peak_rate * RATE_THRESH_OF_PEAK)
            labeled_regions, num_labels = ndi.label(potential_field_mask)

            if num_labels > 0:
                region_sizes = ndi.sum_labels(potential_field_mask, labeled_regions, index=np.arange(1, num_labels + 1))
                largest_region_label = np.argmax(region_sizes) + 1
                largest_region_mask = (labeled_regions == largest_region_label)
                field_area_bins = np.sum(largest_region_mask)
                field_area_cm2 = field_area_bins * bin_area_cm2
                print(f"  Largest potential field: Area={field_area_cm2:.2f} cm2")

                if field_area_cm2 >= MIN_FIELD_SIZE_CM2:
                    # Calculate coherence using original rate map but considering only valid occupancy bins
                    spatial_coherence = calculate_spatial_coherence(rate_map_hz, total_occupancy_s, MIN_OCCUPANCY_S)
                    print(f"  Spatial Coherence: {spatial_coherence:.3f}")

                    if spatial_coherence is not None and not np.isnan(spatial_coherence) and spatial_coherence > SPATIAL_COHERENCE_THRESH:
                        is_place_cell = True
                        field_properties = {
                            'field_mask': largest_region_mask,
                            'field_area_cm2': field_area_cm2,
                            'peak_rate_hz': peak_rate,
                            'mean_rate_hz': mean_rate_overall, # Mean over valid occupancy bins
                            'spatial_coherence': spatial_coherence
                        }
                        print(f"  Cluster {cluster_id} meets criteria: IS a place cell.")
                    else:
                        print(f"  Cluster {cluster_id} fails criteria: Spatial Coherence ({spatial_coherence:.3f}) <= {SPATIAL_COHERENCE_THRESH}.")
                else:
                    print(f"  Cluster {cluster_id} fails criteria: Field Area ({field_area_cm2:.2f} cm2) < {MIN_FIELD_SIZE_CM2} cm2.")
            else:
                print(f"  Cluster {cluster_id} fails criteria: No contiguous region above {RATE_THRESH_OF_PEAK*100}% rate threshold.")
        else:
            print(f"  Cluster {cluster_id} fails criteria: Peak Rate ({peak_rate:.2f} Hz) <= {PEAK_RATE_THRESH_HZ} Hz.")


        # --- Save Rate Map ---
        map_filename = os.path.join(output_dir, f"cluster_{cluster_id}_ratemap.npy")
        np.save(map_filename, rate_map_hz)

        # --- Plotting ---
        # (Plotting code remains largely the same as before)
        plt.figure(figsize=(7, 6))
        extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
        # Handle potential all-zero or all-NaN maps for plotting limits
        vmax = np.nanmax(rate_map_hz) if np.any(~np.isnan(rate_map_hz)) else 1.0
        vmin = 0
        im = plt.imshow(rate_map_hz, origin='lower', cmap='jet', extent=extent, interpolation='gaussian', vmin=vmin, vmax=max(vmax, 0.1)) # Ensure vmax is > 0
        plt.colorbar(im, label='Firing Rate (Hz)')
        plot_title = f"Cluster {cluster_id} Place Field"
        if is_place_cell:
            field_mask = field_properties.get('field_mask')
            if field_mask is not None: plt.contour(field_mask, levels=[0.5], colors='white', linewidths=1.5, extent=extent)
            plot_title += " (Place Cell)"
            plt.text(0.05, 0.95, f"Peak: {field_properties['peak_rate_hz']:.1f} Hz\nArea: {field_properties['field_area_cm2']:.0f} cm2\nCoh: {field_properties['spatial_coherence']:.2f}",
                     transform=plt.gca().transAxes, color='white', verticalalignment='top', fontsize=8, bbox=dict(boxstyle='round,pad=0.3', fc='black', alpha=0.5))
        else:
            plot_title += " (Not Place Cell)"
            plt.text(0.05, 0.95, f"Peak: {peak_rate:.1f} Hz",
                     transform=plt.gca().transAxes, color='white', verticalalignment='top', fontsize=8, bbox=dict(boxstyle='round,pad=0.3', fc='black', alpha=0.5))
        plt.title(plot_title)
        plt.xlabel("Position X (cm)")
        plt.ylabel("Position Y (cm)")
        plt.axis('on')
        plot_filename = os.path.join(output_dir, f"cluster_{cluster_id}_placefield.png")
        plt.savefig(plot_filename, dpi=150)
        plt.close()

        results.append({
            'cluster_id': cluster_id,
            'is_place_cell': is_place_cell,
            'peak_rate_hz': peak_rate,
            'mean_rate_hz': mean_rate_overall,
            'spatial_coherence': field_properties.get('spatial_coherence', np.nan),
            'field_area_cm2': field_properties.get('field_area_cm2', 0),
            'ratemap_file': map_filename,
            'plot_file': plot_filename
        })

    # 9. Save Summary Results
    if results:
        df_results = pd.DataFrame(results)
        summary_filename = os.path.join(output_dir, "place_field_summary.csv")
        df_results.to_csv(summary_filename, index=False)
        print(f"\nSummary saved to: {summary_filename}")
        print(f"Found {df_results['is_place_cell'].sum()} place cells out of {len(df_results)} processed clusters.")
    else:
        print("\nNo clusters processed or results generated.")

    print("\nScript finished.")