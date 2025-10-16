# -*- coding: utf-8 -*-
"""
Calculates, filters, saves, and plots place fields for neuronal clusters.

Combines spike data (.npy) with animal tracking data from DLC (.csv),
using precise frame timestamps from a NIDAQ alignment file (.npy)
for accurate alignment.

Generates a place field heatmap and a trajectory plot with spike overlays
for each cluster, using the position data specified in the configuration
(e.g., Neck) for both.

Handles potential mismatch between tracking frame count and timestamp count
by truncating the tracking data if it's longer, as nidq recording was closed before actual video.

Based on user-provided description and parameters.
Followed Azahara Oliva et al 2019 https://doi.org/10.1038/s41586-020-2758-y
Spiking data were binned into 1-cm-wide segments of the camera field projected onto the maze floor,
 generating raw maps of spike counts and occupancy probability. A Gaussian kernel (standard deviation = 5 cm) was applied to both raw maps of spike and occupancy, 
 and a smoothed rate map was constructed by dividing the spike map by the occupancy map. A place field was defined as a continuous region of at least 15 cm2, 
 where the mean firing rate was greater than 10% of the peak rate in the maze, the peak firing rate was greater than 2 Hz, and the spatial coherence was larger than 0.7.
 
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
REAL_HEIGHT_CM = 17.0 


# --- Place Field Calculation Parameters ---
BIN_SIZE_CM = 1.0 # Bin width for maps (cm) 1.0
GAUSSIAN_SD_CM = 5.0 # Standard deviation for Gaussian smoothing (cm) 5.0
# Minimum time (seconds) in bin to be considered valid (using loaded FPS)
# MIN_OCCUPANCY_S = 1.0 / actual_fps # Calculated dynamically now

# --- Place Field Filtering Criteria ---
MIN_FIELD_SIZE_CM2 = 15.0 # Minimum continuous area for a place field (cm^2) 15.0
RATE_THRESH_OF_PEAK = 0.10 # Firing rate must be > X% of peak rate
PEAK_RATE_THRESH_HZ = 2.0 # Peak firing rate must be > X Hz 2
SPATIAL_COHERENCE_THRESH = 0.7 # Spatial coherence must be > X

# --- Tracking Data Settings ---
# Which bodypart's coordinates to use for position? (Must match CSV column names)
# THIS WILL BE USED FOR BOTH PLACE FIELDS AND TRAJECTORY PLOTS
POS_X_COL = 'Neck_x'
POS_Y_COL = 'Neck_y'
# Which speed column to use for filtering? (Usually based on the same bodypart)
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
            sys.exit()

        print(f"Loaded spike data for {df['cluster_id'].nunique()} clusters.")
        if 'cluster_id' not in df.columns or 'spike_times_seconds' not in df.columns:
            print("Error: Spike data DataFrame missing 'cluster_id' or 'spike_times_seconds'.")
            sys.exit()
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
        df = pd.read_csv(filepath, header=0, index_col=0)
        print("Columns loaded by pandas:", list(df.columns))

        # Define necessary columns based on configuration
        relevant_cols = [POS_X_COL, POS_Y_COL]
        if SPEED_THRESHOLD_CM_S is not None and SPEED_COL:
            relevant_cols.append(SPEED_COL)

        # Check if columns exist (case-insensitive check)
        missing_cols = []
        final_cols_to_use = {} # Map config name to actual CSV column name

        for col_config_name in relevant_cols:
            found = False
            if col_config_name in df.columns: # Exact match
                final_cols_to_use[col_config_name] = col_config_name
                found = True
            else: # Case-insensitive match
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
            print(f"Tracking CSV missing required columns: {missing_cols}")
            print(f"Required based on config: POS_X='{POS_X_COL}', POS_Y='{POS_Y_COL}'" + (f", SPEED='{SPEED_COL}'" if SPEED_COL else ""))
            print(f"Columns found in CSV: {list(df.columns)}")
            print(f"Please check the CSV file header and the script CONFIGURATION section.")
            print(f"-------------\n")
            sys.exit()

        # Select columns using the found actual names
        df_selected = df[[final_cols_to_use[col] for col in relevant_cols]].copy()

        # Rename columns back to the consistent names used in the script (config names)
        rename_map = {v: k for k, v in final_cols_to_use.items()}
        df_selected.rename(columns=rename_map, inplace=True)

        # Add frame numbers
        if not pd.api.types.is_integer_dtype(df_selected.index):
            print("Warning: Index column is not integer type, resetting index for frame numbers.")
            df_selected.reset_index(drop=True, inplace=True)
        df_selected.insert(0, 'frame', df_selected.index)

        print(f"Loaded tracking data with {len(df_selected)} frames and required columns.")
        print(f"  Columns available after selection/rename: {list(df_selected.columns)}")
        return df_selected

    except FileNotFoundError:
        print(f"Error: Tracking data file not found at '{filepath}'.")
        sys.exit()
    except Exception as e:
        print(f"Error loading or processing tracking data file '{filepath}': {e}")
        if isinstance(e, pd.errors.ParserError):
            print("  Pandas ParserError: Check CSV formatting, delimiters, or extra header/footer rows.")
        elif isinstance(e, KeyError):
             print(f"  KeyError: Problem accessing expected column during selection/rename.")
             print(f"  Needed columns (config names): {relevant_cols}")
             print(f"  Mapping used (ConfigName: ActualCSVName): {final_cols_to_use}")
        sys.exit()


def load_timestamp_data(filepath):
    """Loads frame timestamp data from the alignment .npy file."""
    try:
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
    df_tracking = load_tracking_data(tracking_filepath) # Uses updated function
    timestamp_data = load_timestamp_data(timestamp_filepath)

    # 4. Verify Data Consistency and Align
    frame_times_s_orig = timestamp_data['FramesTimesInSec']
    first_frame_time_s = timestamp_data['FirstFrameTimeInSec']
    actual_fps = timestamp_data['FPS_CameraTTLChannel']

    # === FRAME COUNT MISMATCH HANDLING START ===
    n_tracking_frames = len(df_tracking)
    n_timestamps = len(frame_times_s_orig)
    frame_times_s = frame_times_s_orig

    if n_tracking_frames != n_timestamps:
        print("\n--- WARNING: Frame Count Mismatch Detected ---")
        print(f"  Tracking CSV has {n_tracking_frames} rows (frames).")
        print(f"  Timestamp file has {n_timestamps} frame timestamps.")

        if n_tracking_frames > n_timestamps:
            diff = n_tracking_frames - n_timestamps
            print(f"  Tracking data has {diff} more frames than timestamps.")
            print(f"  Assuming extra frames are at the END, truncating tracking data...")
            df_tracking = df_tracking.iloc[:n_timestamps].copy()
            print(f"  Tracking data truncated to {len(df_tracking)} frames.")

        elif n_timestamps > n_tracking_frames:
            diff = n_timestamps - n_tracking_frames
            print(f"  Timestamp file has {diff} more timestamps than tracking frames.")
            print(f"  This scenario is less common. Attempting to truncate timestamps...")
            frame_times_s = frame_times_s[:n_tracking_frames]
            if len(frame_times_s) > 0:
                first_frame_time_s = frame_times_s[0] # Update first frame time
                print(f"  Timestamps truncated to {len(frame_times_s)}. Check results carefully.")
            else:
                 print(f"  Error: Truncation resulted in zero timestamps. Cannot proceed.")
                 sys.exit()
        else:
             pass # Counts match

        # Re-check after attempting correction
        if len(df_tracking) != len(frame_times_s):
             print("\n--- ERROR ---")
             print(f"Frame count mismatch PERSISTS after attempted truncation:")
             print(f"  Tracking CSV now has {len(df_tracking)} rows.")
             print(f"  Timestamp array now has {len(frame_times_s)} timestamps.")
             print("Cannot proceed. Please investigate source files/truncation logic.")
             print("-------------\n")
             sys.exit()
        else:
            print("--- Frame counts now match after truncation. Proceeding with alignment. ---")
    # === FRAME COUNT MISMATCH HANDLING END ===

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
    df_spikes = df_spikes[df_spikes['spike_times_seconds'].apply(len) > 0].copy()
    print(f"  Kept {len(df_spikes)} / {original_cluster_count} clusters with remaining spikes.\n")


    # 5. Prepare Tracking Data (Coordinates, Speed Filter)
    # Convert configured pixel coordinates (e.g., Neck) to cm
    scale_x = REAL_WIDTH_CM / PIXEL_WIDTH
    scale_y = REAL_HEIGHT_CM / PIXEL_HEIGHT
    df_tracking['pos_x_cm'] = df_tracking[POS_X_COL] * scale_x
    df_tracking['pos_y_cm'] = df_tracking[POS_Y_COL] * scale_y
    df_tracking['pos_y_cm'] = REAL_HEIGHT_CM - df_tracking['pos_y_cm'] # Flip Y

    # --- Speed Filtering (based on configured SPEED_COL, e.g., Neck Speed) ---
    if SPEED_THRESHOLD_CM_S is not None and SPEED_COL in df_tracking.columns:
        valid_speed_mask = df_tracking[SPEED_COL] > SPEED_THRESHOLD_CM_S
        df_tracking_filtered = df_tracking[valid_speed_mask].copy()
        print(f"Applied speed filter (based on {SPEED_COL}): > {SPEED_THRESHOLD_CM_S} cm/s. Kept {len(df_tracking_filtered)} / {len(df_tracking)} frames.")
        if len(df_tracking_filtered) == 0:
            print("Warning: No tracking data points above speed threshold. Analysis might fail.")
    else:
        df_tracking_filtered = df_tracking.copy()
        print("No speed filtering applied.")

    # Extract Filtered Timestamps and Positions (e.g., Neck) for analysis and plotting
    tracking_timestamps = df_tracking_filtered['timestamp_s'].values
    tracking_pos_x = df_tracking_filtered['pos_x_cm'].values # e.g., Neck X
    tracking_pos_y = df_tracking_filtered['pos_y_cm'].values # e.g., Neck Y

    # Check if enough data remains after filtering
    if len(tracking_timestamps) < 10:
         print("\n--- WARNING ---")
         print(f"Very few tracking data points ({len(tracking_timestamps)}) remain after speed filtering.")
         print("Analysis might be unreliable or fail.")
         # Optionally exit: sys.exit()

    # 6. Define Bins for Histograms
    n_bins_x = int(np.ceil(REAL_WIDTH_CM / BIN_SIZE_CM))
    n_bins_y = int(np.ceil(REAL_HEIGHT_CM / BIN_SIZE_CM))
    x_edges = np.linspace(0, n_bins_x * BIN_SIZE_CM, n_bins_x + 1)
    y_edges = np.linspace(0, n_bins_y * BIN_SIZE_CM, n_bins_y + 1)
    bin_area_cm2 = BIN_SIZE_CM * BIN_SIZE_CM
    MIN_OCCUPANCY_S = 1.0 / actual_fps

    # 7. Calculate Overall Occupancy Map (using filtered configured positions, e.g., Neck)
    if len(tracking_pos_x) == 0:
         print("\n--- ERROR ---")
         print("No tracking data points available to calculate occupancy map.")
         sys.exit()
    total_occupancy_counts, _, _, _ = binned_statistic_2d(
        tracking_pos_x, tracking_pos_y, None, # Use Neck positions
        statistic='count', bins=[x_edges, y_edges], expand_binnumbers=False)
    total_occupancy_s = total_occupancy_counts / actual_fps
    sigma_bins = GAUSSIAN_SD_CM / BIN_SIZE_CM
    smoothed_occupancy_s = gaussian_filter(total_occupancy_s, sigma=sigma_bins, mode='constant', cval=0.0)
    smoothed_occupancy_s[smoothed_occupancy_s < 1e-9] = np.nan


    # 8. Process Each Cluster
    results = []
    for index, row in df_spikes.iterrows():
        cluster_id = row['cluster_id']
        spike_times = row['spike_times_seconds']

        print(f"\nProcessing Cluster {cluster_id} ({len(spike_times)} aligned spikes)...")

        # Find spike indices corresponding to filtered movement times
        if len(spike_times) == 0: continue
        if len(tracking_timestamps) == 0: continue
        indices = np.searchsorted(tracking_timestamps, spike_times, side='right') - 1
        valid_spike_mask = (indices >= 0) & (indices < len(tracking_timestamps))
        valid_indices = indices[valid_spike_mask]
        if len(valid_indices) == 0:
            print("  Skipping - No spikes occurred during valid movement periods.")
            continue

        # Get Spike Positions (using filtered configured bodypart, e.g., Neck)
        spike_pos_x = tracking_pos_x[valid_indices] # e.g., Neck X at spike times
        spike_pos_y = tracking_pos_y[valid_indices] # e.g., Neck Y at spike times
        num_valid_spikes = len(spike_pos_x)

        print(f"  Found positions for {num_valid_spikes} spikes using {POS_X_COL}/{POS_Y_COL} during movement.")
        if num_valid_spikes < 5:
            print(f"  Skipping - Too few spikes ({num_valid_spikes}) occurred during movement periods.")
            continue

        # Calculate Place Field using configured positions (e.g., Neck)
        raw_spike_map, _, _, _ = binned_statistic_2d(
            spike_pos_x, spike_pos_y, None, # Use Neck spike positions
            statistic='count', bins=[x_edges, y_edges], expand_binnumbers=False)
        smoothed_spike_map = gaussian_filter(raw_spike_map, sigma=sigma_bins, mode='constant', cval=0.0)
        with np.errstate(divide='ignore', invalid='ignore'):
            rate_map_hz = smoothed_spike_map / smoothed_occupancy_s # Occupancy based on Neck
        rate_map_hz[np.isnan(rate_map_hz)] = 0

        # Apply Place Field Criteria
        peak_rate = np.nanmax(rate_map_hz[smoothed_occupancy_s > 1e-9]) if np.any(smoothed_occupancy_s > 1e-9) else 0
        valid_occupancy_mask_for_mean = total_occupancy_s > MIN_OCCUPANCY_S
        if np.nansum(smoothed_occupancy_s[valid_occupancy_mask_for_mean]) > 0:
            mean_rate_overall = np.nansum(rate_map_hz[valid_occupancy_mask_for_mean] * smoothed_occupancy_s[valid_occupancy_mask_for_mean]) / np.nansum(smoothed_occupancy_s[valid_occupancy_mask_for_mean])
        else:
            mean_rate_overall = 0
        print(f"  Peak Rate: {peak_rate:.2f} Hz, Mean Rate (valid bins): {mean_rate_overall:.2f} Hz")
        is_place_cell = False
        field_properties = {}
        spatial_coherence = np.nan
        if peak_rate > PEAK_RATE_THRESH_HZ:
            potential_field_mask = (rate_map_hz > (peak_rate * RATE_THRESH_OF_PEAK)) & (smoothed_occupancy_s > 1e-9)
            labeled_regions, num_labels = ndi.label(potential_field_mask)
            if num_labels > 0:
                region_sizes = ndi.sum_labels(potential_field_mask, labeled_regions, index=np.arange(1, num_labels + 1))
                largest_region_label = np.argmax(region_sizes) + 1
                largest_region_mask = (labeled_regions == largest_region_label)
                field_area_bins = np.sum(largest_region_mask)
                field_area_cm2 = field_area_bins * bin_area_cm2
                print(f"  Largest potential field: Area={field_area_cm2:.2f} cm2")
                if field_area_cm2 >= MIN_FIELD_SIZE_CM2:
                    spatial_coherence = calculate_spatial_coherence(rate_map_hz, total_occupancy_s, MIN_OCCUPANCY_S)
                    print(f"  Spatial Coherence: {spatial_coherence:.3f}")
                    if spatial_coherence is not None and not np.isnan(spatial_coherence) and spatial_coherence > SPATIAL_COHERENCE_THRESH:
                        is_place_cell = True
                        field_properties = {
                            'field_mask': largest_region_mask,
                            'field_area_cm2': field_area_cm2,
                            'peak_rate_hz': peak_rate,
                            'mean_rate_hz': mean_rate_overall,
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

        # --- Plotting (Place Field Map) ---
        plt.figure(figsize=(7, 6))
        extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
        vmax = peak_rate if peak_rate > 0 else 1.0
        vmin = 0
        plot_map = np.where(smoothed_occupancy_s > 1e-9, rate_map_hz, np.nan)
        cmap = plt.cm.jet
        cmap.set_bad(color='lightgray')
        im = plt.imshow(plot_map, origin='lower', cmap=cmap, extent=extent, interpolation='gaussian', vmin=vmin, vmax=vmax)
        plt.colorbar(im, label='Firing Rate (Hz)')
        plot_title = f"Cluster {cluster_id} Place Field ({POS_X_COL}/{POS_Y_COL})" # Indicate bodypart used
        if is_place_cell:
            field_mask = field_properties.get('field_mask')
            if field_mask is not None: plt.contour(field_mask, levels=[0.5], colors='white', linewidths=1.5, extent=extent)
            plot_title += " (Place Cell)"
            coh_text = f"{spatial_coherence:.2f}" if not np.isnan(spatial_coherence) else "N/A"
            plt.text(0.05, 0.95, f"Peak: {field_properties['peak_rate_hz']:.1f} Hz\nArea: {field_properties['field_area_cm2']:.0f} cm2\nCoh: {coh_text}",
                     transform=plt.gca().transAxes, color='white', verticalalignment='top', fontsize=8, bbox=dict(boxstyle='round,pad=0.3', fc='black', alpha=0.5))
        else:
            plot_title += " (Not Place Cell)"
            coh_text = f"{spatial_coherence:.2f}" if not np.isnan(spatial_coherence) else "N/A"
            plt.text(0.05, 0.95, f"Peak: {peak_rate:.1f} Hz\nCoh: {coh_text}",
                     transform=plt.gca().transAxes, color='white', verticalalignment='top', fontsize=8, bbox=dict(boxstyle='round,pad=0.3', fc='black', alpha=0.5))
        plt.title(plot_title)
        plt.xlabel("Position X (cm)")
        plt.ylabel("Position Y (cm)")
        plt.axis('on')
        plt.gca().set_facecolor('lightgray')
        plot_filename_pf = os.path.join(output_dir, f"cluster_{cluster_id}_placefield.png")
        plt.savefig(plot_filename_pf, dpi=150)
        plt.close() # Close the place field plot

        # --- Plotting (Trajectory + Spikes - using configured positions e.g., Neck) ---
        plt.figure(figsize=(7, 6))
        # Plot the filtered trajectory (e.g., Neck path during movement)
        plt.plot(tracking_pos_x, tracking_pos_y, color='grey', alpha=0.5, zorder=1, label=f'{POS_X_COL}/{POS_Y_COL} Path (movement)')
        # Plot the spike locations (using same bodypart, e.g., Neck)
        plt.scatter(spike_pos_x, spike_pos_y, color='red', s=15, zorder=2, label=f'Spikes (at {POS_X_COL}/{POS_Y_COL} pos)')
        plt.xlabel("Position X (cm)")
        plt.ylabel("Position Y (cm)")
        plt.title(f"Cluster {cluster_id} - Trajectory ({POS_X_COL}/{POS_Y_COL}) and Spikes ({num_valid_spikes} spikes)")
        plt.xlim(x_edges[0], x_edges[-1])
        plt.ylim(y_edges[0], y_edges[-1])
        plt.gca().set_aspect('equal', adjustable='box')
        plt.legend(loc='upper right', fontsize='small')
        plt.grid(True, linestyle='--', alpha=0.6)
        plot_filename_traj = os.path.join(output_dir, f"cluster_{cluster_id}_trajectory_spikes.png")
        plt.savefig(plot_filename_traj, dpi=150)
        plt.close() # Close the trajectory plot

        # --- Append Results ---
        results.append({
            'cluster_id': cluster_id,
            'is_place_cell': is_place_cell,
            'peak_rate_hz': peak_rate,
            'mean_rate_hz': mean_rate_overall,
            'spatial_coherence': spatial_coherence,
            'field_area_cm2': field_properties.get('field_area_cm2', 0),
            'ratemap_file': map_filename,
            'plot_file_placefield': plot_filename_pf,
            'plot_file_trajectory': plot_filename_traj
        })

    # 9. Save Summary Results
    if results:
        df_results = pd.DataFrame(results)
        summary_filename = os.path.join(output_dir, "place_field_summary.csv")
        df_results.to_csv(summary_filename, index=False)
        print(f"\nSummary saved to: {summary_filename}")
        print(f"Found {df_results['is_place_cell'].sum()} place cells out of {len(df_results)} processed clusters.")
    else:
         print("\nNo clusters processed or results generated (potentially due to filtering or lack of spikes during movement).")

    print("\nScript finished.")