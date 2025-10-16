# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 16:31:00 2025 
Burst index Antonio, 
Cell type classification Check Methods
Jeong, N., Zheng, X., Paulson, A.L. et al. Nature (2025). https://doi.org/10.1038/s41586-025-08868-5
Plots similar to Supplementary Fig 1
@author: ALOK
"""

import pandas as pd
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import warnings
import sys 
try:
    from scipy.optimize import curve_fit, OptimizeWarning # Import OptimizeWarning
    # from scipy.special import exponential # Not needed now
except ImportError:
    print("Error: SciPy library not found. Please install it: pip install scipy")
    sys.exit(1) 

# --- Configuration ---
NS_PYWS_THRESHOLD_MS = 0.425
# Threshold applies to the fitted parameter 'd' from the specific formula provided
WS_PY_D_PARAM_THRESHOLD = 6.0
MIN_SPIKES_FOR_ACG_FIT = 100
ACG_BIN_SIZE_MS = 0.5
ACG_WINDOW_MS = 500 # Adjust ACG window if needed (e.g., 500ms)

CELL_TYPE_COLORS = {
    'NS Interneuron': 'blue',
    'WS Interneuron': 'green',
    'Pyramidal': 'red',
    'Unclassified': 'gray'
}
# Quality Criteria
MIN_FIRING_RATE_HZ = 0.1
MAX_CONTAM_RATE = 10
MAX_L_RATIO = 0.1
MAX_ISI_VIOL_RATE = 0.5

# --- Helper Functions ---

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
    of spikes in the 200- to 300-ms bins. Antonio Ruiz et al 2021 10.1126/science.abf3119 Fig S11
    """
    if len(spike_times_sec) < 10: return np.nan
    if not isinstance(spike_times_sec, np.ndarray): spike_times_sec = np.array(spike_times_sec)
    sorted_times_ms = np.sort(spike_times_sec) * 1000
    diffs = np.diff(sorted_times_ms)
    valid_diffs = diffs[diffs <= window_ms]
    if len(valid_diffs) == 0: return np.nan
    bins = np.arange(0, window_ms + bin_size_ms, bin_size_ms)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        acg, _ = np.histogram(valid_diffs, bins=bins)
        if len(acg) == 0: return np.nan
    burst_bins_mask = (bins[:-1] >= 3) & (bins[:-1] < 5)
    baseline_bins_mask = (bins[:-1] >= 200) & (bins[:-1] < 300)
    avg_burst = np.mean(acg[burst_bins_mask]) if np.any(burst_bins_mask) and np.sum(burst_bins_mask)>0 else 0
    avg_baseline = np.mean(acg[baseline_bins_mask]) if np.any(baseline_bins_mask) and np.sum(baseline_bins_mask)>0 else 0
    if avg_baseline == 0: return np.nan
    else: return avg_burst / avg_baseline


def calculate_acg(spike_times_sec, bin_size_ms, window_ms):
    """Calculates the autocorrelogram (ACG) for a given set of spike times.
    Formula(1) ACGfit = max(c*(exp(-(x-t_refrac)/τ_rise) - d*exp(-(x-t_refrac)/τ_rise)) + h*exp(-(x-t_refrac)/τ_rise) + rate_asymptote, 0)
    where c is the ACG τ decay amplitude, d is the ACG τ rise amplitude, 
    h is the burst amplitude and trefrac is the ACG refractory period (ms).
    Adapted from CellExplorer package Petersen, P. C., Siegle, J. H., Steinmetz, N. A., Mahallati, S. & Buzsáki, G. 
     Neuron 109, 3594–3608 (2021). 10.1016/j.neuron.2021.09.002 
    
    for common exponential term rather than three exponential terms, we can factorize !!!
    Consider E = exp(-(x - t_refrac) / τ_rise)
    then Formula(1) becomes ACGfit = max((c - c*d + h) * E + rate_asymptote, 0)
    which can be simplified as: ACGfit = max((c*(1 - d) + h) * exp(-(x - t_refrac) / τ_rise) + rate_asymptote, 0)
    
    
    """
    if len(spike_times_sec) < 2: return None, None
    if not isinstance(spike_times_sec, np.ndarray): spike_times_sec = np.array(spike_times_sec)
    spike_times_ms = np.sort(spike_times_sec) * 1000
    diffs = []
    max_spikes_for_broadcast = 15000 # Limit for memory efficiency
    if len(spike_times_ms) < max_spikes_for_broadcast:
        # Efficient broadcast method for fewer spikes
        time_diffs = np.abs(np.subtract.outer(spike_times_ms, spike_times_ms))
        diffs = time_diffs[np.triu_indices_from(time_diffs, k=1)] # Unique diffs > 0
        diffs = diffs[diffs <= window_ms]
    else:
         # Fallback loop for very large spike counts (memory safe)
         print(f" Note: Using looped ACG calculation for unit with {len(spike_times_ms)} spikes.")
         for i in range(len(spike_times_ms)):
              j = i + 1
              # Optimization: Check if even the next spike is beyond the window
              if j < len(spike_times_ms) and (spike_times_ms[j] - spike_times_ms[i]) > window_ms:
                  continue # No need to check further for this 'i'
              while j < len(spike_times_ms) and (spike_times_ms[j] - spike_times_ms[i]) <= window_ms:
                   diffs.append(spike_times_ms[j] - spike_times_ms[i])
                   j += 1
         diffs = np.array(diffs)

    if len(diffs) == 0: return None, None
    bins = np.arange(0, window_ms + bin_size_ms, bin_size_ms)
    acg_counts, _ = np.histogram(diffs, bins=bins)
    bin_centers = bins[:-1] + bin_size_ms / 2.0
    return acg_counts, bin_centers

# Define the ACG fitting function EXACTLY as provided in the image/user text
def custom_acg_fit_func(x, c, d, h, tau_rise, t_refrac, rate_asymptote):
    """Custom ACG function based on user-provided formula (single exponential structure)."""
    safe_tau_rise = max(1e-6, tau_rise)
    t = np.maximum(0, x - t_refrac)
    exp_term = np.exp(-t / safe_tau_rise)
    amplitude_factor = c * (1.0 - d) + h
    fit_values = amplitude_factor * exp_term + rate_asymptote
    final_values = np.where(x <= t_refrac, 0, fit_values)
    return np.maximum(0, final_values)


def fit_acg(acg_counts, bin_centers, n_spikes, total_time_sec):
    """Attempts to fit the custom ACG function."""
    if acg_counts is None or len(acg_counts) < 5: # Need min points for fit
        return None, None

    xdata = bin_centers
    ydata = acg_counts

    # --- Initial Guesses (p0) and Bounds ---
    # Parameters: [c, d, h, tau_rise, t_refrac, rate_asymptote]
    peak_acg = np.max(ydata) if len(ydata) > 0 else 1.0
    baseline_est = 0
    if total_time_sec > 0 and n_spikes > 0 and len(bin_centers) > 1:
        # Estimate baseline rate^2 * bin_width * num_spikes
        baseline_est = (n_spikes / total_time_sec)**2 * (bin_centers[1] - bin_centers[0]) * 1e-3 * n_spikes

    p0 = [ peak_acg * 0.6, 0.5, peak_acg * 0.4, 20, 2.0, max(0.01, baseline_est) ]
    bounds = (
        [0, -np.inf, 0, 0.5, 0, 0], # Lower bounds
        [peak_acg*5, np.inf, peak_acg*5, 1000, 10, max(peak_acg, baseline_est * 2 + 0.1)] # Upper bounds
    )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            warnings.simplefilter("ignore", category=OptimizeWarning)
            popt, pcov = curve_fit(
                custom_acg_fit_func, xdata, ydata, p0=p0, bounds=bounds, maxfev=5000, method='trf'
             )
        # Check covariance validity (optional, indicates fit reliability)
        if pcov is None or np.any(np.isinf(np.diag(pcov))) or np.any(np.isnan(np.diag(pcov))):
            print(f"  Warning: Fit converged but covariance matrix is invalid. Parameters might be unreliable.")
            pass # Still return popt, but maybe handle classification differently?
        return popt, pcov
    except RuntimeError: return None, None
    except ValueError: return None, None
    except Exception: return None, None


def classify_cell_type_full(trough_to_peak_ms, d_parameter_fit):
    """Classifies cell type using TTP and the fitted 'd' parameter."""
    if pd.isna(trough_to_peak_ms): return 'Unclassified'
    ttp = trough_to_peak_ms
    if ttp <= NS_PYWS_THRESHOLD_MS: return 'NS Interneuron'
    else: # ttp > NS_PYWS_THRESHOLD_MS
        if pd.notna(d_parameter_fit) and d_parameter_fit > WS_PY_D_PARAM_THRESHOLD: return 'WS Interneuron'
        else: return 'Pyramidal' # Default if 'd' invalid or below threshold


def generate_classification_plots(df, output_dir, base_filename):
    """Generates and saves classification plots based on the dataframe."""
    print("\n--- Generating Classification Plots ---")
    if df.empty: print("Dataframe is empty. Skipping plot generation."); return
    required_plot_cols = ['trough_to_peak_ms', 'firing_rate_hz', 'burst_index', 'isi_viol_rate_allensdk', 'cell_type']
    missing_cols = [col for col in required_plot_cols if col not in df.columns]
    if missing_cols: print(f"Warning: Missing columns required for plotting: {missing_cols}. Skipping plots."); return
    valid_df = df.dropna(subset=required_plot_cols)
    if valid_df.empty: print("No valid data points for plotting."); return
    print(f"Plotting data for {len(valid_df)} units.")
    types = sorted(valid_df['cell_type'].unique())
    data_by_type = {t: valid_df[valid_df['cell_type'] == t] for t in types if t != 'Unclassified'}
    unclassified_data = valid_df[valid_df['cell_type'] == 'Unclassified']

    # --- Plot b: Trough-to-peak vs Burst Index ---
    plt.figure(figsize=(8, 6))
    for cell_type, type_data in data_by_type.items(): plt.scatter(type_data['trough_to_peak_ms'], type_data['burst_index'], alpha=0.6, s=20, label=cell_type, color=CELL_TYPE_COLORS.get(cell_type, 'grey'))
    if not unclassified_data.empty: plt.scatter(unclassified_data['trough_to_peak_ms'], unclassified_data['burst_index'], alpha=0.3, s=15, label='Unclassified', color=CELL_TYPE_COLORS['Unclassified'])
    plt.axvline(NS_PYWS_THRESHOLD_MS, color='k', linestyle='--', linewidth=1, label=f'NS/PY-WS Threshold ({NS_PYWS_THRESHOLD_MS} ms)')
    plt.xlabel('Trough-to-Peak Duration (ms)');
    plt.ylabel('Burst Index'); plt.title(f'Trough-to-Peak vs Burst Index ({base_filename})')
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.6);
    plot_filepath = os.path.join(output_dir, f'plot_b_TTP_vs_BurstIdx_{base_filename}.png')
    try:
        plt.savefig(plot_filepath)
        print(f" Plot (b) saved to: {os.path.basename(plot_filepath)}")
    except Exception as e: print(f" Error saving plot (b): {e}")
    finally: plt.close()

    # --- Plot c: Distribution of Firing Rates ---
    plt.figure(figsize=(8, 6))
    # Calculate min/max safely, ensuring min_fr > 0 for log scale
    positive_fr = valid_df['firing_rate_hz'][valid_df['firing_rate_hz'] > 0]
    if positive_fr.empty:
        print(" Warning: No positive firing rates found for plot (c). Skipping plot.")
        plt.close() # Close the empty figure
        # Need to skip the rest of plot c logic, maybe continue to plot d?
        # Or handle differently depending on desired behavior for no positive FRs
    else:
        min_fr = positive_fr.min()
        max_fr = valid_df['firing_rate_hz'].max() # Max can still include 0 if present
        # Ensure min_fr is slightly positive if it ended up zero somehow
        if min_fr <= 0: min_fr = 0.01

        bins_c = np.logspace(np.log10(min_fr), np.log10(max_fr + 1), 30) # Use different var name bins_c

        # Loop through classified types
        for cell_type, type_data in data_by_type.items():
            # Get positive firing rates for this type
            fr_data = type_data['firing_rate_hz'][type_data['firing_rate_hz'] > 0]
            # Only plot if there's data for this type after filtering
            if not fr_data.empty:
                 plt.hist(fr_data, bins=bins_c, alpha=0.7, label=cell_type, color=CELL_TYPE_COLORS.get(cell_type, 'grey'), density=True)

        # Handle unclassified data
        if not unclassified_data.empty:
            # Get positive firing rates for unclassified
            fr_data_uncl = unclassified_data['firing_rate_hz'][unclassified_data['firing_rate_hz'] > 0]
            # --- CORRECTED INDENTATION and added empty check ---
            # Only plot histogram if fr_data_uncl is not empty after filtering
            if not fr_data_uncl.empty:
                plt.hist(fr_data_uncl, bins=bins_c, alpha=0.4, label='Unclassified', color=CELL_TYPE_COLORS['Unclassified'], density=True)
        # --- End Corrected Indentation ---

        # Set plot properties only if we plotted something
        plt.gca().set_xscale('log')
        plt.xlabel('Firing Rate (Hz) [Log Scale]')
        plt.ylabel('Density')
        plt.title(f'Distribution of Firing Rates by Cell Type ({base_filename})')
        plt.legend()
        plt.grid(True, axis='y', linestyle='--', alpha=0.6)

        # Save the plot
        plot_filepath = os.path.join(output_dir, f'plot_c_FiringRate_Dist_{base_filename}.png')
        try:
            plt.savefig(plot_filepath)
            print(f" Plot (c) saved to: {os.path.basename(plot_filepath)}")
        except Exception as e:
            print(f" Error saving plot (c): {e}")
        finally:
            plt.close() # Close plot figure

    # --- Plot d: Distribution of Burst Index ---
    plt.figure(figsize=(8, 6))
    valid_burst_indices = valid_df['burst_index'].replace([np.inf, -np.inf], np.nan).dropna()
    if valid_burst_indices.empty:
        print(" No valid burst indices for plot (d).")
        plt.close() # Close the empty figure
    else:
        # --- Start of Corrected Bin Range Logic for Plot d ---
        # Determine robust range for bins, avoiding extreme outliers maybe
        q1, q99 = valid_burst_indices.quantile([0.01, 0.99]) 
        min_bi_q = max(0, q1)
        max_bi_q = q99 * 1.1

        # Use quantiles only if they provide a valid range, otherwise use actual min/max
        if min_bi_q < max_bi_q:
             min_bi = min_bi_q
             max_bi = max_bi_q
        else:
            # Fallback to actual min/max if quantile approach failed or gave invalid range
            min_bi = valid_burst_indices.min()
            max_bi = valid_burst_indices.max()

        # Handle the edge case where min and max are identical (e.g., all values are the same)
        if np.isclose(min_bi, max_bi): # Use isclose for floating point comparison
            offset = 0.1
            min_bi -= offset / 2
            max_bi += offset / 2
            if np.isclose(valid_burst_indices.iloc[0], 0.0): # Check if the original value was zero
                 min_bi = 0
                 max_bi = offset

        # Final safety check: Ensure min_bi is strictly less than max_bi for linspace
        if not min_bi < max_bi:
            min_bi = max_bi - 0.1

        bins_d = np.linspace(min_bi, max_bi, 30) # Use different var name bins_d
        # --- End of Corrected Bin Range Logic ---

        for cell_type, type_data in data_by_type.items():
             bi_data = type_data['burst_index'].replace([np.inf, -np.inf], np.nan).dropna()
             if not bi_data.empty: plt.hist(bi_data, bins=bins_d, alpha=0.7, label=cell_type, color=CELL_TYPE_COLORS.get(cell_type, 'grey'), density=True)
        if not unclassified_data.empty:
             bi_data_uncl = unclassified_data['burst_index'].replace([np.inf, -np.inf], np.nan).dropna()
             if not bi_data_uncl.empty: plt.hist(bi_data_uncl, bins=bins_d, alpha=0.4, label='Unclassified', color=CELL_TYPE_COLORS['Unclassified'], density=True)
        plt.xlabel('Burst Index'); plt.ylabel('Density'); plt.title(f'Distribution of Burst Index by Cell Type ({base_filename})')
        plt.legend(); plt.grid(True, axis='y', linestyle='--', alpha=0.6);
        plot_filepath = os.path.join(output_dir, f'plot_d_BurstIndex_Dist_{base_filename}.png')
        try:
            plt.savefig(plot_filepath)
            print(f" Plot (d) saved to: {os.path.basename(plot_filepath)}")
        except Exception as e: print(f" Error saving plot (d): {e}")
        finally: plt.close()

    # --- Plot e: Distribution of ISI Violation Rate ---
    plt.figure(figsize=(8, 6)); valid_isi_viol = valid_df['isi_viol_rate_allensdk'].dropna()
    if valid_isi_viol.empty: print(" No valid ISI violation rates for plot (e)."); plt.close()
    else:
        min_isi = 0; q99 = valid_isi_viol.quantile(0.99); max_isi = max(0.5, q99 * 1.2);
        bins_e = np.linspace(min_isi, max_isi, 30); # Use different var name bins_e
        plt.hist(valid_isi_viol, bins=bins_e, alpha=0.5, label='All Good Units', color='darkgrey', density=True); 
        perc_95 = np.percentile(valid_isi_viol, 95); plt.axvline(perc_95, color='red', linestyle=':', linewidth=2, label=f'95th Percentile ({perc_95:.3f})')
        plt.xlabel('ISI Violation Rate (AllenSDK `isi_viol`)'); 
        plt.ylabel('Density'); plt.title(f'Distribution of Refractory Period Violations ({base_filename})')
        plt.legend(); plt.grid(True, axis='y', linestyle='--', alpha=0.6); 
        plt.xlim(left=min_isi);
        plot_filepath = os.path.join(output_dir, f'plot_e_ISIViol_Dist_{base_filename}.png')
        try:
            plt.savefig(plot_filepath)
            print(f" Plot (e) saved to: {os.path.basename(plot_filepath)}")
        except Exception as e: print(f" Error saving plot (e): {e}")
        finally: plt.close()

    print("--- Plot generation finished ---")

# --- Main Script ---
print("Please select the directory containing your Kilosort output files, metrics.csv, and channel brain region mapping...")
kilosort_folder = browse_directory()
if not kilosort_folder:
    print("No directory selected. Exiting.")
    sys.exit(1) # Exit if no folder selected
print(f"Selected directory: {kilosort_folder}")
folder_basename = os.path.basename(kilosort_folder)

# Paths
spike_times_file = os.path.join(kilosort_folder, 'spike_times.npy')
spike_times_sec_file = os.path.join(kilosort_folder, 'spike_times_sec.npy')
spike_clusters_file = os.path.join(kilosort_folder, 'spike_clusters.npy')
channel_pos_file = os.path.join(kilosort_folder, 'channel_positions.npy')
metrics_file = os.path.join(kilosort_folder, 'metrics.csv')
brain_regions_file_name = 'channel_brain_regions.csv'
brain_regions_file = os.path.join(kilosort_folder, brain_regions_file_name)

# Load Kilosort Data
spike_times_seconds = None
sampling_rate_hz = 30000.0 # Default sampling rate

if os.path.exists(spike_times_sec_file):
    try:
        spike_times_seconds = np.load(spike_times_sec_file).flatten()
        print(f"Loaded spike times from: {os.path.basename(spike_times_sec_file)}")
    except Exception as e:
        print(f"Error loading '{os.path.basename(spike_times_sec_file)}': {e}")
        sys.exit(1)
elif os.path.exists(spike_times_file):
    print(f"'{os.path.basename(spike_times_sec_file)}' not found. Looking for params.py and '{os.path.basename(spike_times_file)}'.")
    params_file = os.path.join(kilosort_folder, 'params.py')
    if os.path.exists(params_file):
        try:
            with open(params_file, 'r') as f: params_content = f.read()
            local_namespace = {}
            exec(params_content, {}, local_namespace)
            sampling_rate_hz = local_namespace.get('sample_rate', sampling_rate_hz)
            print(f" Found sampling rate in params.py: {sampling_rate_hz} Hz")
        except Exception as e:
            print(f" Warning: Could not read sampling rate from params.py ({e}). Using default: {sampling_rate_hz} Hz")

    try:
        spike_times = np.load(spike_times_file).flatten()
        if sampling_rate_hz <= 0:
            # *** CORRECTED INDENTATION for sys.exit ***
            print("Error: Invalid sampling rate (<=0) obtained.")
            sys.exit(1)
        # Only proceed if sampling rate is valid
        spike_times_seconds = spike_times / sampling_rate_hz
        print(f"Loaded spike times from: {os.path.basename(spike_times_file)} and converted to seconds using SR={sampling_rate_hz} Hz.")
        # Optionally save converted seconds file
        # try:
        #     np.save(spike_times_sec_file, spike_times_seconds)
        #     print(f" Saved converted spike times to: {os.path.basename(spike_times_sec_file)}")
        # except Exception as e:
        #     print(f" Warning: Could not save converted spike times: {e}")
    except Exception as e:
         print(f"Error loading '{os.path.basename(spike_times_file)}': {e}")
         sys.exit(1)
else:
    print(f"Error: Neither '{os.path.basename(spike_times_sec_file)}' nor '{os.path.basename(spike_times_file)}' found.")
    sys.exit(1)

# Load spike clusters
if not os.path.exists(spike_clusters_file):
    # *** CORRECTED INDENTATION for sys.exit ***
    print(f"Error: Spike clusters file '{os.path.basename(spike_clusters_file)}' not found.")
    sys.exit(1)
try:
    spike_clusters = np.load(spike_clusters_file).flatten()
    print(f"Loaded spike clusters from: {os.path.basename(spike_clusters_file)}")
except Exception as e:
    print(f"Error loading spike clusters file '{os.path.basename(spike_clusters_file)}': {e}")
    sys.exit(1)


# Check lengths
if len(spike_times_seconds) != len(spike_clusters):
    # *** CORRECTED INDENTATION for sys.exit ***
    print(f"Error: Mismatch spike times ({len(spike_times_seconds)}) /clusters ({len(spike_clusters)}) length.")
    sys.exit(1)

# Load channel positions
# *** CORRECTED if/else STRUCTURE ***
if not os.path.exists(channel_pos_file):
    print(f"Warning: Channel positions file '{os.path.basename(channel_pos_file)}' not found.")
    channel_pos = None
else:
    try:
        channel_pos = np.load(channel_pos_file)
        print(f"Loaded channel positions from: {os.path.basename(channel_pos_file)}")
    except Exception as e:
        print(f"Error loading channel positions file '{os.path.basename(channel_pos_file)}': {e}")
        channel_pos = None # Set to None if loading fails

# Load metrics
if not os.path.exists(metrics_file):
    print(f"Error: Metrics file '{os.path.basename(metrics_file)}' not found.")
    sys.exit(1) # Indentation already correct here
try:
    metrics_df = pd.read_csv(metrics_file)
    id_col = None
    if 'cluster_id' in metrics_df.columns: id_col = 'cluster_id'
    elif 'id' in metrics_df.columns: id_col = 'id'; print("Warning: Using 'id' column from metrics.csv.")
    else: raise ValueError("Neither 'cluster_id' nor 'id' found in metrics.")
    metrics_df.rename(columns={id_col: 'cluster_id'}, inplace=True)
    metrics_df['cluster_id'] = pd.to_numeric(metrics_df['cluster_id'], errors='coerce')
    metrics_df.dropna(subset=['cluster_id'], inplace=True) # Remove rows where cluster_id became NaN
    metrics_df['cluster_id'] = metrics_df['cluster_id'].astype(int)
    print(f"Loaded metrics from: {os.path.basename(metrics_file)}")
except Exception as e:
    print(f"Error loading or processing metrics file '{os.path.basename(metrics_file)}': {e}")
    sys.exit(1) # Indentation already correct here

# Load Brain Regions
brain_regions_df = None
if not os.path.exists(brain_regions_file):
    print(f"Warning: Brain regions file '{brain_regions_file_name}' not found.")
else:
    try:
        print(f"Checking indexing in {brain_regions_file_name}...")
        temp_df = pd.read_csv(brain_regions_file)
        if 'global_channel_index' in temp_df.columns and not temp_df.empty:
            # Check if adjustment is needed AND the first value is indeed 1 (or not 0)
            if temp_df['global_channel_index'].iloc[0] != 0 and pd.notna(temp_df['global_channel_index'].iloc[0]):
                print(f" Adjusting '{brain_regions_file_name}' to 0-based index..."); temp_df['global_channel_index'] -= 1
                print(f" --> SAVING MODIFIED DATA back to '{brain_regions_file_name}'..."); temp_df.to_csv(brain_regions_file, index=False)
                print(f"     '{brain_regions_file_name}' updated."); brain_regions_df = pd.read_csv(brain_regions_file, index_col='global_channel_index')
            else:
                print(f" '{brain_regions_file_name}' seems already 0-based or starts with NaN/0."); brain_regions_df = pd.read_csv(brain_regions_file, index_col='global_channel_index')
        else:
             print(f" Warning: 'global_channel_index' not found/empty in '{brain_regions_file_name}'."); brain_regions_df = None

        if brain_regions_df is not None:
            print(f"Successfully loaded and processed brain region mapping.")
    except Exception as e:
        print(f"Error processing brain regions file '{brain_regions_file}': {e}")
        brain_regions_df = None # Ensure it's None if loading fails

# --- Main Processing Loop ---
print(f"\nCalculating recording duration...")
good_clusters_final = []
unique_clusters_all = np.unique(spike_clusters)
print(f"Found {len(unique_clusters_all)} unique clusters in '{os.path.basename(spike_clusters_file)}'. Processing (fitting ACGs)...")

# Calculate recording duration safely
min_time = np.nanmin(spike_times_seconds)
max_time = np.nanmax(spike_times_seconds)
if np.isnan(min_time) or np.isnan(max_time):
     print("Warning: Cannot determine recording duration from spike times (NaN found). Setting duration to 0.")
     recording_duration = 0.0
else:
     recording_duration = max_time - min_time

if recording_duration <= 0:
     print(f"Warning: Recording duration is non-positive ({recording_duration:.2f}s). Firing rates will be affected.")
     # Decide whether to exit or continue with potentially weird rates
     # sys.exit(1) # Optional: Exit if duration is invalid

# --- Iterate through each cluster ---
fit_failures = 0
skipped_fits = 0
for i, cluster_id in enumerate(unique_clusters_all):
    cluster_id = int(cluster_id)
    if (i + 1) % 50 == 0: print(f" Processing cluster {i+1}/{len(unique_clusters_all)} (ID: {cluster_id})...")

    cluster_mask = (spike_clusters == cluster_id)
    cluster_spike_times_seconds = spike_times_seconds[cluster_mask]
    n_spikes = len(cluster_spike_times_seconds)

    # Skip processing if no spikes (shouldn't happen with Kilosort output but safety check)
    if n_spikes == 0:
        continue

    # Calculate firing rate (handle zero duration)
    firing_rate = (n_spikes / recording_duration) if recording_duration > 0 else np.inf

    # Calculate percent_short_isi
    percent_short_isi = np.nan
    if n_spikes >= 2:
        isi = np.diff(np.sort(cluster_spike_times_seconds)) * 1000
        len_isi = len(isi)
        percent_short_isi = (np.sum(isi < 2) / len_isi * 100) if len_isi > 0 else 0.0
    elif n_spikes == 1: percent_short_isi = 0.0

    # Find metrics row
    metrics_row = metrics_df[metrics_df['cluster_id'] == cluster_id]
    if not metrics_row.empty:
        metrics_data = metrics_row.iloc[0]
        if 'peak_channel' not in metrics_data.index or pd.isna(metrics_data['peak_channel']): continue
        try: peak_channel_index = int(metrics_data['peak_channel'])
        except ValueError: continue

        # Get quality metrics
        # Get quality metrics
        contam_rate = metrics_data.get('contam_rate', None)
        l_ratio = metrics_data.get('l_ratio', None)
        isi_viol_rate = metrics_data.get('isi_viol', None) # AllenSDK metric name

        # --- CORRECTED Try/Except Blocks ---
        # Convert quality metrics to float with error handling
        try:
            # Use pd.notna() to check for NaN before conversion
            contam_rate = float(contam_rate) if pd.notna(contam_rate) else None
        except (ValueError, TypeError):
            # Handle cases where conversion to float fails (e.g., if value was text)
            contam_rate = None # Set to None if conversion fails

        try:
            l_ratio = float(l_ratio) if pd.notna(l_ratio) else None
        except (ValueError, TypeError):
            l_ratio = None # Set to None if conversion fails

        try:
            isi_viol_rate = float(isi_viol_rate) if pd.notna(isi_viol_rate) else None
        except (ValueError, TypeError):
            isi_viol_rate = None # Set to None if conversion fails
        # --- End Corrected Blocks ---

        # Check quality criteria using the potentially updated variables
        is_good = (firing_rate > MIN_FIRING_RATE_HZ and
                   contam_rate is not None and contam_rate < MAX_CONTAM_RATE and
                   l_ratio is not None and l_ratio < MAX_L_RATIO and
                   isi_viol_rate is not None and isi_viol_rate < MAX_ISI_VIOL_RATE)

        if is_good:
            # ... (rest of the code for good clusters) ...
            burst_index = calculate_burst_index(cluster_spike_times_seconds, window_ms=500)
            trough_to_peak_ms = metrics_data.get('duration', None)
            try: trough_to_peak_ms = float(trough_to_peak_ms) if pd.notna(trough_to_peak_ms) else None
            except (ValueError, TypeError): trough_to_peak_ms = None

            # --- ACG Calculation and Fitting ---
            d_parameter = np.nan # Initialize
            popt = None
            if n_spikes >= MIN_SPIKES_FOR_ACG_FIT:
                acg_counts, bin_centers = calculate_acg(cluster_spike_times_seconds, ACG_BIN_SIZE_MS, ACG_WINDOW_MS)
                if acg_counts is not None:
                     popt, pcov = fit_acg(acg_counts, bin_centers, n_spikes, recording_duration)
                     if popt is not None:
                         d_parameter = popt[1] # Extract the 'd' parameter (index 1)
                     else:
                         fit_failures += 1
            else:
                skipped_fits +=1

            # --- Classify Cell Type ---
            cell_type = classify_cell_type_full(trough_to_peak_ms, d_parameter)

            # --- Create dictionary ---
            good_cluster_info = {
                'cluster_id': int(cluster_id), 'cell_type': cell_type, 'firing_rate_hz': round(firing_rate, 3),
                'isi_violation_percent_manual': round(percent_short_isi, 3) if pd.notna(percent_short_isi) else None,
                'isi_viol_rate_allensdk': isi_viol_rate, 'contamination_rate': contam_rate, 'l_ratio': l_ratio,
                'num_spikes': n_spikes, 'spike_times_seconds': cluster_spike_times_seconds.tolist(), # NPY Only
                'peak_channel_index_0based': peak_channel_index, 'burst_index': burst_index if pd.notna(burst_index) else None,
                'trough_to_peak_ms': trough_to_peak_ms if pd.notna(trough_to_peak_ms) else None,
                'acg_d_parameter': d_parameter if pd.notna(d_parameter) else None, # Store fitted 'd'
                'shank_index': None, 'distance_from_entry_um': None, 'acronym': None, 'brain_region_name': None, 'channel_depth_um_fallback': None
            }

            # --- Get Brain Region Info & Fallback Depth ---
            if brain_regions_df is not None:
                 if peak_channel_index in brain_regions_df.index:
                      region_info = brain_regions_df.loc[peak_channel_index]
                      good_cluster_info['shank_index'] = int(region_info.get('shank_index')) if pd.notna(region_info.get('shank_index')) else None
                      good_cluster_info['distance_from_entry_um'] = float(region_info.get('distance_from_entry_um')) if pd.notna(region_info.get('distance_from_entry_um')) else None
                      good_cluster_info['acronym'] = region_info.get('acronym') if pd.notna(region_info.get('acronym')) else None
                      good_cluster_info['brain_region_name'] = region_info.get('name') if pd.notna(region_info.get('name')) else None
            if good_cluster_info['distance_from_entry_um'] is None:
                  if channel_pos is not None and 0 <= peak_channel_index < len(channel_pos):
                       try: good_cluster_info['channel_depth_um_fallback'] = float(channel_pos[peak_channel_index][1])
                       except (IndexError, TypeError, ValueError): pass

            good_clusters_final.append(good_cluster_info)

# --- Post-processing ---
print(f"\nProcessed {len(unique_clusters_all)} unique clusters.")
print(f"ACG fitting: Skipped for {skipped_fits} units (low spike count), Failed for {fit_failures} units.")
n_good = len(good_clusters_final)
print(f"Found {n_good} good clusters meeting quality criteria.")

if good_clusters_final:
    good_clusters_df = pd.DataFrame(good_clusters_final)

    # Cell Type Summary
    if 'cell_type' in good_clusters_df.columns:
         type_counts = good_clusters_df['cell_type'].value_counts()
         print("Cell Type Summary (using provided ACG formula fit):")
         for ctype, count in type_counts.items(): print(f" - {ctype}: {count} units")
         print(f"   (WS Interneuron requires TTP>{NS_PYWS_THRESHOLD_MS}ms & fitted 'd'>{WS_PY_D_PARAM_THRESHOLD})")
         print(f"   (Pyramidal if TTP>{NS_PYWS_THRESHOLD_MS}ms but 'd' condition not met/fit failed)")
         print(f"   WARNING: Classification uses the literal formula provided. The meaning of 'd > {WS_PY_D_PARAM_THRESHOLD}' with this formula might need verification from the source.")
    else: print("Warning: 'cell_type' column missing.")

    # Save NPY
    output_npy_file_name = f'good_clusters_processed_{folder_basename}_customACG.npy'
    output_npy_path = os.path.join(kilosort_folder, output_npy_file_name)
    try:
        np.save(output_npy_path, np.array(good_clusters_final, dtype=object), allow_pickle=True)
        print(f"\nFull data saved to: {output_npy_path}")
    except Exception as e: print(f"Error saving NPY file: {e}")

    # Save CSV
    try:
        good_clusters_df_no_spikes = good_clusters_df.drop(columns=['spike_times_seconds']) if 'spike_times_seconds' in good_clusters_df.columns else good_clusters_df.copy()
        cols_order = [ # Updated column name
            'cluster_id', 'cell_type', 'acronym', 'brain_region_name', 'firing_rate_hz',
            'isi_violation_percent_manual', 'isi_viol_rate_allensdk', 'contamination_rate', 'l_ratio',
            'num_spikes', 'burst_index', 'trough_to_peak_ms', 'acg_d_parameter', # Use new name
            'peak_channel_index_0based', 'shank_index', 'distance_from_entry_um', 'channel_depth_um_fallback'
        ]
        # Make column ordering robust
        cols_order_present = [col for col in cols_order if col in good_clusters_df_no_spikes.columns]
        missing_cols = [col for col in good_clusters_df_no_spikes.columns if col not in cols_order_present]
        final_cols_order = cols_order_present + missing_cols
        good_clusters_df_no_spikes = good_clusters_df_no_spikes[final_cols_order]

        output_csv_path = os.path.join(kilosort_folder, f'good_clusters_processed_{folder_basename}_customACG.csv')
        good_clusters_df_no_spikes.to_csv(output_csv_path, index=False, float_format='%.4f')
        print(f"Metrics summary saved to: {output_csv_path}")
    except Exception as e: print(f"Error saving CSV file: {e}")

    # Generate Plots
    generate_classification_plots(good_clusters_df, kilosort_folder, folder_basename)

else:
    print("\nNo 'good' clusters identified, skipping saving and plotting.")

print("\nScript finished.")