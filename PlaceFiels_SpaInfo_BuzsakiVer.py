# -*- coding: utf-8 -*-
"""
Calculates place fields and spatial information (Skaggs Index) based on
a specific method combining peak-rate relative boundaries, size constraints,
peak rate thresholds, and spatial coherence.

Method Overview (per user description v2):
1. Load spikes, tracking (e.g., Neck), timestamps. Align.
2. Filter tracking data for speed > 2 cm/s.
3. Bin data (5x5 cm) for occupancy (time) and spikes (counts).
4. Smooth occupancy time map (Gaussian sigma = 10cm, i.e., 2 bins * 5cm/bin).
5. Smooth spike count map (Gaussian sigma = 10cm).
6. Calculate smoothed rate map = smooth_spikes / smooth_occupancy_time.
7. For each cell:
    a. Calculate Peak Firing Rate on smoothed map.
    b. Define Field Boundary: Rate >= 20% of Peak Rate.
    c. Identify contiguous regions above boundary threshold.
    d. Filter regions by Size (2% to 60% of cage area).
    e. Calculate Spatial Coherence.
    f. Classify Place Cell if:
        - Peak Rate >= 3 Hz
        - AND At least one valid field exists (passes size filter)
        - AND Spatial Coherence > 0.7
    g. Calculate Spatial Information (Skaggs Index - bits/spike) using
       smoothed rate map and unsmoothed occupancy, following provided logic in SkaggsIndex code in repository.
    h. Save maps, plots, and summary statistics.

NOTE: Trial separation (forward/backward runs) mentioned in the source
      description is NOT implemented here; the analysis assumes one session.
NOTE: Smoothing sigma=3cm (2 bins * 1.5cm/bin) is used as interpreted,
      but may be large; consider reducing if results are over-smoothed.
following https://github.com/valegarman/HippoCookBook
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.stats import binned_statistic_2d
# from scipy.stats import median_abs_deviation # No longer needed
from skimage.measure import label as ski_label
from skimage.measure import regionprops
import os
import sys
import time
import tkinter as tk
from tkinter import filedialog

# ==============================================================
# --- Configuration Parameters --- [UPDATED FOR SKAGGS METHOD] ---

# --- Video/Tracking Parameters ---
PIXEL_WIDTH = 1440
PIXEL_HEIGHT = 1080
REAL_WIDTH_CM = 28.5 # User's cage size
REAL_HEIGHT_CM = 17.0 # User's cage size

# --- Tracking Data Settings ---
POS_X_COL = 'Neck_x' # Bodypart for position
POS_Y_COL = 'Neck_y'
SPEED_COL = 'Neck Speed (cm/s)' # Speed column for filtering
SPEED_THRESHOLD_CM_S = 2.0 # Speed threshold (cm/s)

# --- Place Field Calculation Parameters ---
BIN_SIZE_CM = 1.5 
# Gaussian smoothing SD. Assuming description "2-bins" refers to sigma.
GAUSSIAN_SD_CM = 3 # CHANGED (2 bins * 1.5 cm/bin)

# --- Place Field Criteria ---
RATE_THRESH_OF_PEAK = 0.20 # - Relative rate for boundary
PEAK_RATE_THRESH_HZ = 3.0 # - Minimum peak firing rate
SPATIAL_COHERENCE_THRESH = 0.7 # Spatial coherence threshold
# Field Size Constraints (% of total cage area)
MIN_FIELD_SIZE_PRCT = 2.0 
MAX_FIELD_SIZE_PRCT = 60.0 

# --- Output Settings ---
OUTPUT_SUBDIR = "place_field_analysis_BuzsakiLab_method" # Updated subdir name

# ==============================================================

# --- Helper Functions ---

def select_file(title="Select File", filetypes=[("All files", "*.*")]):
    """Opens a file dialog."""
    root = tk.Tk(); root.withdraw()
    filepath = filedialog.askopenfilename(title=title, filetypes=filetypes)
    if not filepath: print("No file selected. Exiting."); sys.exit()
    print(f"Selected file: {filepath}"); return filepath

def load_spike_data(filepath):
    """Loads cluster spike data."""
    try:
        data_struct = np.load(filepath, allow_pickle=True)
        if data_struct.dtype.names: df = pd.DataFrame([dict(zip(data_struct.dtype.names, row)) for row in data_struct])
        elif isinstance(data_struct[0], dict): df = pd.DataFrame(list(data_struct))
        else: print("Error: Spike data .npy structure not recognized."); sys.exit()
        print(f"Loaded spike data for {df['cluster_id'].nunique()} clusters.")
        if not {'cluster_id', 'spike_times_seconds'}.issubset(df.columns): print("Error: Spike data missing required columns."); sys.exit()
        df['spike_times_seconds'] = df['spike_times_seconds'].apply(lambda x: np.asarray(x, dtype=float))
        return df
    except FileNotFoundError: print(f"Error: Spike data file not found: '{filepath}'."); sys.exit()
    except Exception as e: print(f"Error loading spike data '{filepath}': {e}"); sys.exit()

def load_tracking_data(filepath):
    """Loads tracking data, ensuring required columns."""
    try:
        df = pd.read_csv(filepath, header=0, index_col=0)
        print("Columns loaded:", list(df.columns))
        relevant_cols = [POS_X_COL, POS_Y_COL]
        if SPEED_COL: relevant_cols.append(SPEED_COL)
        else: print("Warning: SPEED_COL not defined.")
        missing_cols, final_cols_to_use = [], {}
        for col_config_name in relevant_cols:
            if col_config_name is None: continue
            found = False
            if col_config_name in df.columns: final_cols_to_use[col_config_name], found = col_config_name, True
            else:
                for actual_col_name in df.columns:
                    if actual_col_name.lower() == col_config_name.lower():
                        final_cols_to_use[col_config_name], found = actual_col_name, True
                        print(f"  Info: Matched '{col_config_name}' to '{actual_col_name}' (case-insensitive).")
                        break
            if not found: missing_cols.append(col_config_name)
        if missing_cols: print(f"\n--- ERROR --- Tracking CSV missing columns: {missing_cols}. Required: {relevant_cols}. Found: {list(df.columns)}. Check config/headers. ---"); sys.exit()
        df_selected = df[[final_cols_to_use[col] for col in relevant_cols if col in final_cols_to_use]].copy()
        df_selected.rename(columns={v: k for k, v in final_cols_to_use.items()}, inplace=True)
        if not pd.api.types.is_integer_dtype(df_selected.index): print("Warning: Index not integer, resetting index."); df_selected.reset_index(drop=True, inplace=True)
        df_selected.insert(0, 'frame', df_selected.index)
        print(f"Loaded tracking data: {len(df_selected)} frames. Cols available: {list(df_selected.columns)}")
        return df_selected
    except FileNotFoundError: print(f"Error: Tracking data file not found: '{filepath}'."); sys.exit()
    except Exception as e: print(f"Error loading tracking data '{filepath}': {e}"); sys.exit()

def load_timestamp_data(filepath):
    """Loads NIDAQ frame timestamp data."""
    try:
        data = np.load(filepath, allow_pickle=True).item()
        print("Timestamp file loaded.")
        required_keys = ['FirstFrameTimeInSec', 'FramesTimesInSec', 'FPS_CameraTTLChannel']
        if not all(key in data for key in required_keys): print(f"Error: Timestamp file missing keys. Need: {required_keys}. Found: {list(data.keys())}"); sys.exit()
        print(f"  First frame: {data['FirstFrameTimeInSec']:.4f}s, Num timestamps: {len(data['FramesTimesInSec'])}, FPS: {data['FPS_CameraTTLChannel']:.4f}Hz")
        return data
    except FileNotFoundError: print(f"Error: Timestamp file not found: '{filepath}'."); sys.exit()
    except Exception as e: print(f"Error loading timestamp data '{filepath}': {e}"); sys.exit()

def calculate_spatial_coherence(rate_map, occupancy_map_unsm, occupancy_thresh_s):
    """ Calculates spatial coherence. """
    # ... (Same as the previous version) ...
    valid_occupancy_mask = occupancy_map_unsm > occupancy_thresh_s
    if not np.any(valid_occupancy_mask): return np.nan
    rate_map_valid = np.where(valid_occupancy_mask, rate_map, np.nan)
    padded_rate_map = np.pad(rate_map_valid, pad_width=1, mode='constant', constant_values=np.nan)
    neighbor_sum = np.zeros_like(rate_map_valid, dtype=float); neighbor_count = np.zeros_like(rate_map_valid, dtype=int)
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0: continue
            neighbor_values = padded_rate_map[1+dy : rate_map_valid.shape[0]+1+dy, 1+dx : rate_map_valid.shape[1]+1+dx]
            valid_neighbors = ~np.isnan(neighbor_values)
            neighbor_sum[valid_neighbors] += neighbor_values[valid_neighbors]; neighbor_count[valid_neighbors] += 1
    mean_neighbor_rate = np.full_like(rate_map_valid, np.nan); valid_counts = neighbor_count > 0
    mean_neighbor_rate[valid_counts] = neighbor_sum[valid_counts] / neighbor_count[valid_counts]
    correlation_mask = valid_occupancy_mask & ~np.isnan(rate_map_valid) & ~np.isnan(mean_neighbor_rate)
    if np.sum(correlation_mask) < 2: return np.nan
    center_rates = rate_map_valid[correlation_mask]; neighbor_rates = mean_neighbor_rate[correlation_mask]
    if np.std(center_rates) < 1e-9 or np.std(neighbor_rates) < 1e-9: return 0.0
    coherence = np.corrcoef(center_rates, neighbor_rates)[0, 1]
    return coherence

# --- UPDATED Helper Function: calculate_skaggs_index ---
def calculate_skaggs_index(rate_map_hz, occupancy_time_s_unsm, min_occupancy_s):
    """
    Calculates Skaggs spatial information index based on provided Matlab logic.

    Args:
        rate_map_hz (np.ndarray): Smoothed firing rate map (Hz) (z in matlab code).
        occupancy_time_s_unsm (np.ndarray): Unsmoothed occupancy time map (seconds)
                                            (occupancy or time in matlab code).
        min_occupancy_s (float): Minimum time in bin to be included (minTime).

    Returns:
        tuple: (bits_per_spike, bits_per_sec)
    """
    bits_per_sec_skaggs = np.nan
    bits_per_spike_skaggs = np.nan

    # nanmask = (z > 0) & time >= minTime;
    # Use a small epsilon for rate > 0 check
    nanmask = (rate_map_hz > 1e-9) & (occupancy_time_s_unsm >= min_occupancy_s)

    if not np.any(nanmask):
        # print("Warning: No valid bins found for Skaggs index calculation.")
        return 0.0, 0.0 # Return 0 if no valid area

    # duration = sum(occupancy(nanmask));
    # This is the total time spent in *valid bins*
    duration = np.nansum(occupancy_time_s_unsm[nanmask])

    if duration <= 1e-9:
        # print("Warning: Zero duration in valid bins for Skaggs index.")
        return 0.0, 0.0 # Return 0 if no time spent in valid areas

    # meanRate = sum(z(nanmask).*occupancy(nanmask))/duration;
    # This is the occupancy-weighted mean firing rate across *valid bins* (lambda)
    total_spikes_in_valid_bins = np.nansum(rate_map_hz[nanmask] * occupancy_time_s_unsm[nanmask])
    mean_rate = total_spikes_in_valid_bins / duration # lambda

    if mean_rate <= 1e-9:
        # print("Warning: Zero mean rate for Skaggs index.")
        return 0.0, 0.0 # Return 0 if cell is silent in valid areas

    # p_x = occupancy ./ duration; --> Prob density P(x_i) relative to time in valid bins
    # Needs to be same shape as rate_map for element-wise ops later
    p_x = np.zeros_like(rate_map_hz)
    # Calculate probability ONLY for valid bins, relative to total time in valid bins
    p_x[nanmask] = occupancy_time_s_unsm[nanmask] / duration

    # p_r = z ./ meanRate; --> lambda_i / lambda
    p_r = rate_map_hz / mean_rate

    # dummy = p_x .* z; --> P(x_i) * lambda_i
    dummy = p_x * rate_map_hz

    # idx = dummy > 0; --> Further mask for only bins contributing to the sum
    # Combine with nanmask to ensure we only use originally valid bins
    # Also check p_r > 0 for log2 calculation
    idx_mask = nanmask & (dummy > 1e-12) & (p_r > 1e-12)

    if not np.any(idx_mask):
        # print("Warning: No bins satisfying conditions for Skaggs sum.")
        return 0.0, 0.0

    # skaggs.bitsPerSec = sum(dummy(idx).*log2(p_r(idx)));
    bits_per_sec_skaggs = np.nansum(dummy[idx_mask] * np.log2(p_r[idx_mask]))

    # skaggs.bitsPerSpike = skaggs.bitsPerSec / meanRate;
    bits_per_spike_skaggs = bits_per_sec_skaggs / mean_rate

    return bits_per_spike_skaggs, bits_per_sec_skaggs

# --- REMOVED Helper Functions: find_firing_fields (MAD based), shuffle_spikes ---

# --- Main Analysis ---
if __name__ == "__main__":

    start_time_analysis = time.time()
    # 1. Select Input Files & 2. Create Output Directory
    spike_filepath = select_file(title="Select Cluster Spike Data (.npy)")
    tracking_filepath = select_file(title="Select Tracking Data (.csv)")
    timestamp_filepath = select_file(title="Select Frame Timestamp (.npy)")
    output_dir = os.path.join(os.path.dirname(tracking_filepath), OUTPUT_SUBDIR)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results saved in: {output_dir}")

    # 3. Load Data
    df_spikes = load_spike_data(spike_filepath)
    df_tracking = load_tracking_data(tracking_filepath)
    timestamp_data = load_timestamp_data(timestamp_filepath)

    # 4. Align Data & Handle Frame Mismatches
    frame_times_s_orig = timestamp_data['FramesTimesInSec']
    first_frame_time_s = timestamp_data['FirstFrameTimeInSec']
    actual_fps = timestamp_data['FPS_CameraTTLChannel']
    n_tracking_frames, n_timestamps = len(df_tracking), len(frame_times_s_orig)
    frame_times_s = np.copy(frame_times_s_orig)
    # ... (Frame Mismatch Handling code - same as before) ...
    if n_tracking_frames != n_timestamps:
        print("\n--- WARNING: Frame Count Mismatch ---")
        # ... (Truncation logic same as before) ...
        if n_tracking_frames > n_timestamps: diff = n_tracking_frames - n_timestamps; print(f"  Truncating {diff} tracking frames..."); df_tracking = df_tracking.iloc[:n_timestamps].copy()
        elif n_timestamps > n_tracking_frames: diff = n_timestamps - n_tracking_frames; print(f"  Truncating {diff} timestamps..."); frame_times_s = frame_times_s[:n_tracking_frames]; first_frame_time_s = frame_times_s[0] if len(frame_times_s)>0 else first_frame_time_s
        if len(df_tracking) != len(frame_times_s): print("\n--- ERROR: Mismatch PERSISTS. Exiting. ---"); sys.exit()
        else: print("--- Frame counts match after truncation. ---")

    df_tracking['timestamp_s'] = frame_times_s
    session_start_time = first_frame_time_s; session_end_time = frame_times_s[-1]
    df_spikes['spike_times_seconds'] = df_spikes['spike_times_seconds'].apply(lambda t: t[(t >= session_start_time) & (t <= session_end_time)])

    # 5. Prepare Tracking Data
    scale_x = REAL_WIDTH_CM / PIXEL_WIDTH; scale_y = REAL_HEIGHT_CM / PIXEL_HEIGHT
    df_tracking['pos_x_cm'] = df_tracking[POS_X_COL] * scale_x
    df_tracking['pos_y_cm'] = df_tracking[POS_Y_COL] * scale_y
    df_tracking['pos_y_cm'] = REAL_HEIGHT_CM - df_tracking['pos_y_cm']

    if SPEED_THRESHOLD_CM_S is not None and SPEED_COL in df_tracking.columns:
        valid_speed_mask = df_tracking[SPEED_COL] > SPEED_THRESHOLD_CM_S
        df_tracking_filtered = df_tracking[valid_speed_mask].copy()
        print(f"Applied speed filter (> {SPEED_THRESHOLD_CM_S} cm/s): Kept {len(df_tracking_filtered)} / {len(df_tracking)} frames.")
    else: df_tracking_filtered = df_tracking.copy(); print("No speed filtering.")

    tracking_timestamps_filt = df_tracking_filtered['timestamp_s'].values
    tracking_pos_x_filt = df_tracking_filtered['pos_x_cm'].values
    tracking_pos_y_filt = df_tracking_filtered['pos_y_cm'].values
    if len(tracking_timestamps_filt) < 10: print("Warning: Few points remain after speed filtering.")

    # 6. Define Bins & Calculate Area Constraints
    n_bins_x = int(np.ceil(REAL_WIDTH_CM / BIN_SIZE_CM))
    n_bins_y = int(np.ceil(REAL_HEIGHT_CM / BIN_SIZE_CM))
    x_edges = np.linspace(0, n_bins_x * BIN_SIZE_CM, n_bins_x + 1)
    y_edges = np.linspace(0, n_bins_y * BIN_SIZE_CM, n_bins_y + 1)
    bin_area_cm2 = BIN_SIZE_CM * BIN_SIZE_CM
    cage_area_cm2 = REAL_WIDTH_CM * REAL_HEIGHT_CM
    min_field_cm2 = cage_area_cm2 * (MIN_FIELD_SIZE_PRCT / 100.0)
    max_field_cm2 = cage_area_cm2 * (MAX_FIELD_SIZE_PRCT / 100.0)
    print(f"Using {BIN_SIZE_CM}cm bins -> {n_bins_x}x{n_bins_y} grid.")
    print(f"Cage Area: {cage_area_cm2:.1f} cm2. Field size range: {min_field_cm2:.1f} - {max_field_cm2:.1f} cm2 ({MIN_FIELD_SIZE_PRCT}%-{MAX_FIELD_SIZE_PRCT}%)")

    # 7. Calculate Overall Occupancy Maps
    if len(tracking_pos_x_filt) == 0: sys.exit("ERROR: No tracking data for occupancy map.")
    occupancy_counts, _, _, _ = binned_statistic_2d(tracking_pos_x_filt, tracking_pos_y_filt, None, 'count', [x_edges, y_edges])
    occupancy_time_s_unsm = occupancy_counts / actual_fps
    sigma_bins = GAUSSIAN_SD_CM / BIN_SIZE_CM
    print(f"Smoothing sigma: {GAUSSIAN_SD_CM} cm = {sigma_bins:.2f} bins")
    smoothed_occupancy_time_s = gaussian_filter(occupancy_time_s_unsm, sigma=sigma_bins, mode='constant', cval=0.0)
    occupancy_for_division = smoothed_occupancy_time_s.copy()
    occupancy_for_division[occupancy_for_division < 1e-9] = np.nan
    MIN_OCCUPANCY_S = 1.0 / actual_fps # Min time for coherence & skaggs calculation

    # 8. Process Each Cluster
    results = []
    min_spikes_for_processing = 5 # Min spikes during movement

    for index, row in df_spikes.iterrows():
        cluster_id = row['cluster_id']; spike_times = row['spike_times_seconds']
        total_spikes_session = len(spike_times)
        print(f"\nProcessing Cluster {cluster_id} ({total_spikes_session} spikes)...")

        if len(tracking_timestamps_filt) == 0: print("  Skipping - No movement data."); continue
        indices = np.searchsorted(tracking_timestamps_filt, spike_times, 'right') - 1
        valid_spike_mask = (indices >= 0) & (indices < len(tracking_timestamps_filt))
        valid_indices = indices[valid_spike_mask]
        num_valid_spikes = len(valid_indices)

        if num_valid_spikes < min_spikes_for_processing: print(f"  Skipping - Only {num_valid_spikes} spikes during movement."); continue

        spike_pos_x = tracking_pos_x_filt[valid_indices]; spike_pos_y = tracking_pos_y_filt[valid_indices]

        # Calculate Maps
        spike_counts_unsm, _, _, _ = binned_statistic_2d(spike_pos_x, spike_pos_y, None, 'count', [x_edges, y_edges])
        smoothed_spike_counts = gaussian_filter(spike_counts_unsm, sigma=sigma_bins, mode='constant', cval=0.0)
        with np.errstate(divide='ignore', invalid='ignore'): rate_map_hz = smoothed_spike_counts / occupancy_for_division
        rate_map_hz[~np.isfinite(rate_map_hz)] = 0.0; rate_map_hz[rate_map_hz < 0] = 0.0
        peak_rate_actual = np.nanmax(rate_map_hz)
        print(f"  Peak Rate: {peak_rate_actual:.3f} Hz")

        # Calculate Metrics
        spatial_coherence = calculate_spatial_coherence(rate_map_hz, occupancy_time_s_unsm, MIN_OCCUPANCY_S)
        # Use the NEW Skaggs function
        skaggs_bits_per_spike, skaggs_bits_per_sec = calculate_skaggs_index(rate_map_hz, occupancy_time_s_unsm, MIN_OCCUPANCY_S)
        print(f"  Spatial Coherence: {spatial_coherence:.3f}")
        print(f"  Skaggs Index: {skaggs_bits_per_spike:.4f} bits/spike")

        # --- Identify Firing Fields (Peak Threshold Method) ---
        field_area_total_cm2 = 0.0
        num_fields = 0
        fields_found_size_ok = False # Flag if any field meets size criteria
        field_details = []
        potential_field_mask = np.zeros_like(rate_map_hz, dtype=bool) # Initialize mask

        # Check peak rate criterion first
        peak_rate_ok = peak_rate_actual >= PEAK_RATE_THRESH_HZ
        if peak_rate_ok:
            # Find regions above relative threshold
            field_thresh = peak_rate_actual * RATE_THRESH_OF_PEAK
            potential_field_mask = (rate_map_hz >= field_thresh) & (occupancy_time_s_unsm > 1e-9)
            labeled_map, num_labels = ski_label(potential_field_mask, connectivity=2, return_num=True)

            if num_labels > 0:
                regions = regionprops(labeled_map, intensity_image=rate_map_hz)
                for region in regions:
                    area_cm2_region = region.area * bin_area_cm2
                    # Check size constraint for this specific region
                    if area_cm2_region >= min_field_cm2 and area_cm2_region <= max_field_cm2:
                        num_fields += 1
                        field_area_total_cm2 += area_cm2_region # Accumulate area of valid fields
                        fields_found_size_ok = True # Mark that at least one field passed size check
                        field_details.append({'label': region.label, 'area_cm2': area_cm2_region, 'peak_rate': region.max_intensity})
        else:
             print(f"  Skipping field detection - Peak rate {peak_rate_actual:.3f} < {PEAK_RATE_THRESH_HZ} Hz")


        print(f"  Field Finding (Rate>={RATE_THRESH_OF_PEAK*100}%; Size {MIN_FIELD_SIZE_PRCT}-{MAX_FIELD_SIZE_PRCT}%): Found {num_fields} fields meeting size criteria.")

        # --- Final Place Cell Classification ---
        coherence_ok = (spatial_coherence is not None and not np.isnan(spatial_coherence) and spatial_coherence > SPATIAL_COHERENCE_THRESH)
        # Criteria: Peak Rate OK? Field(s) Found (Size OK)? Coherence OK?
        is_place_cell = peak_rate_ok and fields_found_size_ok and coherence_ok
        print(f"  Criteria Met: Peak Rate [{peak_rate_ok}], Field Size [{fields_found_size_ok}], Coherence [{coherence_ok}]")
        print(f"  Final Place Cell Classification: {is_place_cell}")

        # --- Save & Plot ---
        map_filename = os.path.join(output_dir, f"cluster_{cluster_id}_ratemap.npy"); np.save(map_filename, rate_map_hz)
        plt.figure(figsize=(7, 6)); extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
        plot_map = rate_map_hz; vmin = 0; vmax = peak_rate_actual if peak_rate_actual > 0 else 1.0
        cmap = plt.get_cmap('jet'); im = plt.imshow(plot_map, origin='lower', cmap=cmap, extent=extent, interpolation='gaussian', vmin=vmin, vmax=vmax)
        plt.colorbar(im, label='Firing Rate (Hz)')
        title_str = f"Cluster {cluster_id} (Peak: {peak_rate_actual:.2f} Hz)"
        if is_place_cell: title_str += f" - PLACE CELL"
        else: title_str += f" - Not Place Cell"
        # Display Skaggs and Coherence
        title_str += f"\nSkaggs: {skaggs_bits_per_spike:.3f} b/sp | Coh: {spatial_coherence:.3f}"
        if fields_found_size_ok: title_str += f" | Fields: {num_fields}" # Indicate if fields meeting criteria found
        plt.title(title_str); plt.xlabel("Position X (cm)"); plt.ylabel("Position Y (cm)"); plt.axis('on')
        # Optional: Draw contours of identified fields meeting size criteria
        if fields_found_size_ok and field_details:
             contour_mask = np.zeros_like(labeled_map, dtype=bool)
             for field in field_details:
                 contour_mask[labeled_map == field['label']] = True
             plt.contour(contour_mask, levels=[0.5], colors='white', linewidths=1.0, extent=extent)

        plot_filename = os.path.join(output_dir, f"cluster_{cluster_id}_placefield.png"); plt.savefig(plot_filename, dpi=150); plt.close()

        # --- Append Results ---
        results.append({
            'cluster_id': cluster_id,
            'is_place_cell': is_place_cell, # Final classification based on Peak Rate, Field Size, Coherence
            'peak_rate_hz': peak_rate_actual,
            'peak_rate_criterion_met': peak_rate_ok,
            'spatial_coherence': spatial_coherence,
            'coherence_criterion_met': coherence_ok,
            'skaggs_bits_per_spike': skaggs_bits_per_spike,
            'skaggs_bits_per_sec': skaggs_bits_per_sec,
            'found_fields_size_ok': fields_found_size_ok, # Based on size criterion only
            'num_fields_size_ok': num_fields,
            'total_field_area_cm2': field_area_total_cm2, # Sum area of fields meeting size criteria
            'num_spikes_session': total_spikes_session,
            'num_spikes_movement': num_valid_spikes,
            'ratemap_file': map_filename,
            'plot_file': plot_filename
        })

    # 9. Save Summary Results
    if results:
        df_results = pd.DataFrame(results)
        summary_filename = os.path.join(output_dir, "place_field_summary.csv")
        df_results.to_csv(summary_filename, index=False)
        print(f"\nSummary saved to: {summary_filename}")
        print(f"Processed {len(df_results)} clusters.")
        if 'is_place_cell' in df_results.columns:
             print(f"Found {df_results['is_place_cell'].sum()} place cells based on combined criteria.")
    else: print("\nNo clusters processed or results generated.")
    print(f"\nTotal analysis time: {time.time() - start_time_analysis:.2f} seconds.")
    print("Script finished.")