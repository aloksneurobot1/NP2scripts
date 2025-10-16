# -*- coding: utf-8 -*-
"""
Created Wed Apr 23 15:45:00 2025 # Set fixed sampling rate

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
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches 
import warnings
import sys
import glob
import traceback
from scipy.optimize import curve_fit, OptimizeWarning
from scipy.stats import gaussian_kde
import seaborn as sns


# --- Configuration ---
SAMPLING_RATE_HZ = 30000

NS_PYWS_THRESHOLD_MS = 0.425
WS_PY_TAU_RISE_THRESHOLD = 6.0 # Unit is ms
MIN_SPIKES_FOR_ACG_FIT = 100
ACG_BIN_SIZE_MS = 0.5
ACG_WINDOW_MS = 50

CELL_TYPE_COLORS = {
    'NS Interneuron': 'blue',
    'WS Interneuron': 'green',
    'Pyramidal': 'red',
    'Unclassified': 'gray'
}
CELL_TYPE_ORDER = ['NS Interneuron', 'WS Interneuron', 'Pyramidal', 'Unclassified']

REGION_MARKERS = { 
    'CA1': 'o', 'CA2': 'P', 'CA3': '^', 'DG-MO': 's', 'DG': 's',
    'SSp-bfd5': 'D', 'SSp-tr5': 'v', 'SSp-tr6a': '<', 'SSp-tr6b': '>',
    'ALV': '*', 'CCB': 'h', 'CING': 'X', 'ROOT': '.', 'SCWM': 'p',
    'DEFAULT': 'x', 'NAN/OTHER': 'x', 'UNCLASSIFIED':'x' 
}

# Quality Criteria
MIN_FIRING_RATE_HZ = 0.05
MAX_CONTAM_RATE = 10
MAX_L_RATIO = 0.1
MAX_ISI_VIOL_RATE = 0.5

# Multi-directory Processing Config
BASE_OUTPUT_PARENT_DIR = r"H:\CatGT_Stitched_Output\fileslist_output Copy\Obj"
DIRECTORY_SEARCH_PATTERN = os.path.join("*", "imec0_ks4")
SAVE_GLOBAL_PLOTS_IN_PARENT_DIR = True


# --- Helper Functions ---
# calculate_burst_index, calculate_acg, triple_exponential_acg_fit_func, fit_acg, classify_cell_type_full
def calculate_burst_index(spike_times_sec, bin_size_ms=1, window_ms=500):
    """Calculates the burst index for a unit based on its spike times."""
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
    """
    Calculates the autocorrelogram (ACG) for a given set of spike times
    up to window_ms using specified bin_size_ms.
    Returns counts and bin centers for positive lags only.
    """
    if len(spike_times_sec) < 2: return None, None
    if not isinstance(spike_times_sec, np.ndarray): spike_times_sec = np.array(spike_times_sec)
    spike_times_ms = np.sort(spike_times_sec) * 1000
    diffs = []
    max_spikes_for_broadcast = 15000
    if len(spike_times_ms) < max_spikes_for_broadcast:
        time_diffs = np.abs(np.subtract.outer(spike_times_ms, spike_times_ms))
        diffs = time_diffs[np.triu_indices_from(time_diffs, k=1)]
        diffs = diffs[diffs <= window_ms]
    else:
         print(f" Note: Using looped ACG calculation for unit with {len(spike_times_ms)} spikes.")
         for i in range(len(spike_times_ms)):
              j = i + 1
              if j < len(spike_times_ms) and (spike_times_ms[j] - spike_times_ms[i]) > window_ms: continue
              while j < len(spike_times_ms) and (spike_times_ms[j] - spike_times_ms[i]) <= window_ms:
                   diffs.append(spike_times_ms[j] - spike_times_ms[i]); j += 1
         diffs = np.array(diffs)

    if len(diffs) == 0: return None, None
    # Ensure bins start just above 0 if bin size allows, up to window_ms
    # Bins from bin_size_ms to window_ms+bin_size_ms with step bin_size_ms
    # Example: binsize=0.5, window=50 -> bins = [0.5, 1.0, ..., 50.0, 50.5]
    bins = np.arange(bin_size_ms, window_ms + bin_size_ms, bin_size_ms)

    
    hist_bins = np.arange(0, window_ms + bin_size_ms, bin_size_ms)
    acg_counts, _ = np.histogram(diffs, bins=hist_bins)

    # Bin centers correspond to the positive lags we fit (e.g., 0.25, 0.75, ... 49.75 for 0.5ms bins)
    bin_centers = hist_bins[:-1] + bin_size_ms / 2.0

    # Return only counts corresponding to positive lags > 0 (MATLAB fits x > 0)
    # If hist_bins[0] is 0, skip the first count bin (0 to bin_size_ms)
    if np.isclose(hist_bins[0], 0):
        return acg_counts[1:], bin_centers[1:] # Return counts/centers for bins starting > 0
    else: # Should not happen with arange(0,...) but safety check
        return acg_counts, bin_centers

def triple_exponential_acg_fit_func(x, a, b, c, d, e, f, g, h):
    """Triple exponential function matching CellExplorer's fit_ACG.m:
    max(c*(exp(-(x-f)/a)-d*exp(-(x-f)/b))+h*exp(-(x-f)/g)+e,0)
    a = tau_decay, b = tau_rise, c = amp_decay, d = multiplier_rise,
    e = asymptote, f = t_refrac, g = tau_burst, h = amp_burst"""
    safe_a = max(1e-6, a); 
    safe_b = max(1e-6, b); 
    safe_g = max(1e-6, g); 
    t = np.maximum(0, x - f)
    term1 = c * np.exp(-t / safe_a); 
    term2 = c * d * np.exp(-t / safe_b); 
    term3 = h * np.exp(-t / safe_g); 
    fit_values = term1 - term2 + term3 + e
    final_values = np.where(x <= f, 0, fit_values); 
    return np.maximum(0, final_values)

def fit_acg(acg_counts, bin_centers):
    if acg_counts is None or bin_centers is None or len(acg_counts) < 8: return None, None
    min_len = min(len(acg_counts), len(bin_centers));
    if min_len == 0: return None, None
    xdata = bin_centers[:min_len]; ydata = acg_counts[:min_len]; valid_indices = xdata > 0;
    if not np.any(valid_indices): return None, None
    xdata = xdata[valid_indices]; ydata = ydata[valid_indices]
    if len(xdata) < 8: return None, None
    p0 = [ 20,  1,  30,  2, 0.5,  5, 1.5,  2]; lb = list([1, 0.1, 0, 0, -30, 0, 0.1, 0]); 
    ub = list([500, 50, 500, 15, 50, 20, 5, 100])
    max_y = np.max(ydata); 
    ub[2] = max(ub[2], max_y * 1.5); ub[7] = max(ub[7], max_y * 1.5);
    ub[4] = max(ub[4], max_y * 0.5);
    bounds = (lb, ub)
    try:
        with warnings.catch_warnings(): warnings.simplefilter("ignore", category=RuntimeWarning); 
        warnings.simplefilter("ignore", category=OptimizeWarning)
        popt, pcov = curve_fit(triple_exponential_acg_fit_func, xdata, ydata, p0=p0, bounds=bounds, maxfev=5000, method='trf')
        if pcov is None or np.any(np.isinf(np.diag(pcov))) or np.any(np.isnan(np.diag(pcov))): pass
        return popt, pcov
    except: return None, None

def classify_cell_type_full(trough_to_peak_ms, tau_rise_fit):
    if pd.isna(trough_to_peak_ms): return 'Unclassified'
    ttp = trough_to_peak_ms
    if ttp <= NS_PYWS_THRESHOLD_MS: return 'NS Interneuron'
    else:
        if pd.notna(tau_rise_fit) and tau_rise_fit > WS_PY_TAU_RISE_THRESHOLD: return 'WS Interneuron'
        else: return 'Pyramidal'
        
# --- Complete Global Plotting Function ---
def generate_global_plots(all_df, output_dir, global_basename, region_filter=None):
    """Generates global plots using aggregated data from all directories."""
    print(f"\n--- Generating Global Plots (Data from {len(all_df)} units) ---")

    # --- Define required columns (ensure these match columns generated in process_kilosort_directory) ---
    basic_plot_cols = ['trough_to_peak_ms', 'firing_rate_hz', 'burst_index', 'isi_viol_rate_allensdk', 'cell_type', 'num_spikes', 'acronym']
    compound_plot_cols = ['trough_to_peak_ms', 'acg_tau_rise', 'cell_type', 'acronym']
    all_req_cols = list(set(basic_plot_cols + compound_plot_cols))

    # --- Filter by region ---
    if region_filter:
        if isinstance(region_filter, str): region_filter = [region_filter]
        original_count = len(all_df)
        if 'acronym' not in all_df.columns:
            print("  Warning: 'acronym' column not found. Cannot filter by region.")
        else:
            # Case-insensitive matching for filtering
            available_regions_upper = [str(r).upper() for r in all_df['acronym'].dropna().unique()]
            region_filter_upper = [r.upper() for r in region_filter]
            all_df['acronym_upper'] = all_df['acronym'].fillna('N/A').str.upper() # Treat NaN as 'N/A' temporarily
            all_df = all_df[all_df['acronym_upper'].isin(region_filter_upper)].copy()
            all_df.drop(columns=['acronym_upper'], inplace=True) # Remove temporary column
            print(f"  Filtered data to regions: {', '.join(region_filter)}. Kept {len(all_df)} out of {original_count} units.")
            if all_df.empty: print("  No data remaining after region filtering. Skipping global plots."); return
            global_basename += f"_{'_'.join(region_filter)}"
    else:
        print("  Using data from all regions.")

    # Check if all required columns exist AFTER filtering
    missing_cols = [col for col in all_req_cols if col not in all_df.columns]
    if missing_cols: print(f"Warning: Missing columns required for plotting: {missing_cols}. Skipping plots."); return


    plot_types = ['NS Interneuron', 'WS Interneuron', 'Pyramidal']
    # Determine types present in the potentially filtered dataframe
    types_present_in_data = [t for t in CELL_TYPE_ORDER if t in all_df['cell_type'].unique()]


    # --- Plot a: Compound Scatter (TTP vs TauRise) ---
    print("  Generating plot a: Compound scatter...")
    # Data Prep for Plot A - use required columns
    plot_data_a = all_df[compound_plot_cols].copy()
    # Drop NaNs only for columns essential for this specific plot's axes/color/markers
    plot_data_a.dropna(subset=['trough_to_peak_ms', 'acg_tau_rise', 'cell_type'], inplace=True)
    plot_data_a = plot_data_a[plot_data_a['acg_tau_rise'] > 1e-3] # Filter for log scale safety

    if plot_data_a.empty: print("   No valid data points found for compound scatter plot (TTP vs TauRise).")
    else:
        print(f"   Plotting compound scatter for {len(plot_data_a)} units.")
        # Get types/regions present in the *filtered* data for this plot
        types_present_a = [t for t in CELL_TYPE_ORDER if t in plot_data_a['cell_type'].unique()]
        unclassified_present_a = 'Unclassified' in plot_data_a['cell_type'].unique()
        present_acronyms_a = sorted(plot_data_a['acronym'].dropna().astype(str).unique()) # Ensure string acronyms

        fig = plt.figure(figsize=(8, 7)); # Adjusted size slightly for external legend
        gs = gridspec.GridSpec(2, 2, width_ratios=[1.2, 4], height_ratios=[4, 1.2], wspace=0.05, hspace=0.05)
        ax_scatter = fig.add_subplot(gs[0, 1]);
        ax_histx = fig.add_subplot(gs[1, 1], sharex=ax_scatter);
        ax_histy = fig.add_subplot(gs[0, 0], sharey=ax_scatter);

        # --- Scatter Plot - With Region Markers ---
        ax_scatter.set_yscale('log')
        type_handles = [] # Use simple patches for type legend
        region_marker_map_used = {} # Track used markers/regions

        # Loop through Cell Types first
        for cell_type in types_present_a:
            color = CELL_TYPE_COLORS.get(cell_type, 'grey')
            cell_type_data = plot_data_a[plot_data_a['cell_type'] == cell_type]
            type_handles.append(mpatches.Patch(color=color, label=cell_type)) # Add handle for type legend

            # Plot units with known regions for this cell type
            for region in present_acronyms_a:
                region_cell_data = cell_type_data[cell_type_data['acronym'] == region]
                if not region_cell_data.empty:
                    marker = REGION_MARKERS.get(str(region).upper(), REGION_MARKERS['DEFAULT']) # Case-insensitive lookup
                    ax_scatter.scatter(region_cell_data['trough_to_peak_ms'], region_cell_data['acg_tau_rise'],
                                       alpha=0.7, s=25, color=color, marker=marker, edgecolors='face', linewidth=0.5)
                    region_marker_map_used[region] = marker # Record marker used

            # Plot units with NaN/missing region for this cell type
            nan_region_cell_data = cell_type_data[cell_type_data['acronym'].isna()]
            if not nan_region_cell_data.empty:
                 marker = REGION_MARKERS['DEFAULT']
                 ax_scatter.scatter(nan_region_cell_data['trough_to_peak_ms'], nan_region_cell_data['acg_tau_rise'],
                                       alpha=0.6, s=25, color=color, marker=marker, edgecolors='face', linewidth=0.5)
                 region_marker_map_used['NaN/Other'] = marker

        # Plot unclassified units (using default marker)
        if unclassified_present_a:
            unclassified_data = plot_data_a[plot_data_a['cell_type'] == 'Unclassified']
            color = CELL_TYPE_COLORS['Unclassified']
            marker = REGION_MARKERS['DEFAULT']
            ax_scatter.scatter(unclassified_data['trough_to_peak_ms'], unclassified_data['acg_tau_rise'],
                               alpha=0.4, s=15, color=color, marker=marker, edgecolors='none', label='Unclassified')
            region_marker_map_used['Unclassified'] = marker
            # type_handles.append(mpatches.Patch(color=color, label='Unclassified')) # Optionally add unclassified

        # Scatter plot cosmetics
        ax_scatter.tick_params(axis="y", labelleft=False)
        ax_scatter.tick_params(axis="x", labelbottom=False)
        ax_scatter.grid(True, linestyle=':', alpha=0.5)

        # --- Create Legends ---
        # 1. Cell Type Legend (using patches, inside plot area)
        type_legend = ax_scatter.legend(handles=type_handles, loc='upper right',
                                         frameon=True, framealpha=0.8, edgecolor='none',
                                         facecolor='white', title="Cell Type")

        # 2. Region Marker Legend (outside plot area)
        marker_legend_handles = []
        marker_legend_labels = []
        # Sort unique markers used for consistent legend order
        unique_markers_used = sorted(list(set(region_marker_map_used.values())))

        for marker in unique_markers_used:
             # Find regions associated with this marker, sort them
             regions = sorted([r for r, m in region_marker_map_used.items() if m == marker])
             proxy = plt.Line2D([0], [0], marker=marker, color='grey', linestyle='None', markersize=6)
             marker_legend_handles.append(proxy)
             marker_legend_labels.append(f"{marker}: {', '.join(regions)}")

        if marker_legend_handles:
             # Place legend outside the main axes using figure coordinates
             region_legend = fig.legend(handles=marker_legend_handles, labels=marker_legend_labels,
                                        loc='upper left',
                                        bbox_to_anchor=(0.99, 0.95), # Position it to the right, adjust as needed
                                        frameon=False, title="Region Marker", fontsize=8) # Smaller font

        # Get scatter plot limits AFTER plotting ALL points
        xlim = ax_scatter.get_xlim(); ylim = ax_scatter.get_ylim()
        # Apply limits firmly AFTER determining them
        ax_scatter.set_xlim(xlim); ax_scatter.set_ylim(ylim)

        # --- X-Marginal (TTP) - Color-coded by type & VISIBLE AXIS ---
        for cell_type in types_present_a: # Loop through types
            type_data = plot_data_a[plot_data_a['cell_type'] == cell_type]['trough_to_peak_ms'].values
            if len(type_data) >= 2:
                color = CELL_TYPE_COLORS.get(cell_type, 'grey') # Get color
                try:
                    kde = gaussian_kde(type_data); xs = np.linspace(xlim[0], xlim[1], 300); kde_y = kde(xs)
                    if np.max(kde_y) > 1e-9: # Check density is not effectively zero
                        scaled_y = kde_y / np.max(kde_y)
                        ax_histx.plot(xs, scaled_y, color=color, lw=1.5) # PLOT COLOR-CODED
                        ax_histx.fill_between(xs, 0, scaled_y, color=color, alpha=0.3) # FILL COLOR-CODED
                except Exception as e: print(f"   Warning: Could not compute/plot KDE for TTP of {cell_type}: {e}")
        # X-Marginal cosmetics - Visible axis
        ax_histx.set_xlabel('Trough-to-peak (ms)'); 
        ax_histx.set_ylabel('Peak'); 
        ax_histx.tick_params(axis="y", direction='in', left=True, labelleft=True); 
        ax_histx.spines['top'].set_visible(False); 
        ax_histx.spines['right'].set_visible(False); 
        ax_histx.spines['left'].set_visible(True); 
        ax_histx.set_yticks([0, 0.5, 1]); 
        ax_histx.set_ylim(0, 1.1)

        # --- Y-Marginal (ACG Tau Rise) - Color-coded by type & VISIBLE AXIS ---
        print(f"   DEBUG: Generating Y-Marginal for types: {types_present_a}")
        any_y_marginal_plotted = False # Flag to check if anything gets plotted
        for cell_type in types_present_a:
            print(f"    DEBUG Y-Marg: Processing {cell_type}")
            type_data = plot_data_a[plot_data_a['cell_type'] == cell_type]['acg_tau_rise'].values
            print(f"     DEBUG Y-Marg: Found {len(type_data)} points for {cell_type}")
            if len(type_data) < 2: print(f"     DEBUG Y-Marg: Skipping {cell_type} (fewer than 2 points)"); continue

            # Check for positive values *before* log10
            if not np.all(type_data > 0):
                 print(f"     DEBUG Y-Marg: Skipping {cell_type} (contains non-positive values before log)")
                 continue

            log_data = np.log10(type_data)
            color = CELL_TYPE_COLORS.get(cell_type, 'grey')

            # Check for finite values *after* log10
            finite_mask = np.isfinite(log_data)
            if not np.all(finite_mask):
                 print(f"     DEBUG Y-Marg: {cell_type} - Found non-finite values in log10(Tau Rise): {log_data[~finite_mask]}")
                 log_data = log_data[finite_mask] # Keep only finite values
                 if len(log_data) < 2: print(f"     DEBUG Y-Marg: Skipping {cell_type} (fewer than 2 finite log points)"); continue

            # Check for variance *after* filtering
            if np.var(log_data) < 1e-10: print(f"     DEBUG Y-Marg: Skipping {cell_type} (zero variance in log data)"); continue

            # Check ylim validity before linspace
            if ylim[0] <= 0 or ylim[1] <= 0 or ylim[0] >= ylim[1]:
                 print(f"     DEBUG Y-Marg: Skipping {cell_type} due to invalid ylim for logspace: {ylim}")
                 continue

            print(f"     DEBUG Y-Marg: Attempting KDE for {cell_type}...")
            try:
                kde = gaussian_kde(log_data)
                ys_log = np.linspace(np.log10(ylim[0]), np.log10(ylim[1]), 300)
                kde_x = kde(ys_log)
                if np.max(kde_x) > 1e-9:
                    scaled_x = kde_x / np.max(kde_x)
                    print(f"     DEBUG Y-Marg: Plotting KDE for {cell_type}...")
                    ax_histy.plot(scaled_x, 10**ys_log, color=color, lw=1.5)
                    ax_histy.fill_betweenx(10**ys_log, 0, scaled_x, color=color, alpha=0.3)
                    any_y_marginal_plotted = True # Mark that something was plotted
                else: print(f"     DEBUG Y-Marg: Skipping {cell_type} (KDE max value too low)")
            except Exception as e: print(f"     DEBUG Y-Marg: ERROR during KDE/Plotting for {cell_type}: {e}")
        # End Y-Marginal loop

        # Y-Marginal cosmetics
        ax_histy.set_xlabel('Peak'); ax_histy.set_ylabel(r'ACG $\tau_{rise}$ (ms)'); ax_histy.tick_params(axis="x", direction='in', bottom=True, labelbottom=True); ax_histy.spines['top'].set_visible(False); ax_histy.spines['right'].set_visible(False); ax_histy.spines['bottom'].set_visible(True); ax_histy.set_xticks([0, 0.5, 1]); ax_histy.set_xlim(0, 1.1)
        if not any_y_marginal_plotted: print("   WARNING: No Y-marginal distributions were successfully plotted.") # Add warning if nothing got plotted

        # Figure adjustments and saving
        fig.suptitle(f'Cell Type vs ACG $\\tau_{{rise}}$ & TTP ({global_basename})', y=0.98, fontsize=12);
        plt.subplots_adjust(left=0.15, right=0.75, bottom=0.1, top=0.9)
        plot_filepath_a = os.path.join(output_dir, f'plot_a_ScatterMarginals_{global_basename}.png');
        try: plt.savefig(plot_filepath_a, bbox_inches='tight', dpi=300); print(f"  Plot (a - Compound) saved to: {os.path.basename(plot_filepath_a)}");
        except Exception as e: print(f"  Error saving plot a: {e}")
        finally: plt.close(fig)

    # --- Plots b, c, d, e ---
    # Use valid_df_basic which contains all necessary columns for these plots
    valid_df_basic = all_df.dropna(subset=basic_plot_cols)
    if valid_df_basic.empty: print("No valid data points for basic plots (b, c, d, e)."); return # Exit plotting early
    print(f"Plotting basic plots for {len(valid_df_basic)} units.")
    types_basic = sorted(valid_df_basic['cell_type'].unique()); data_by_type_basic = {t: valid_df_basic[valid_df_basic['cell_type'] == t] for t in types_basic if t != 'Unclassified'}; 
    unclassified_data_basic = valid_df_basic[valid_df_basic['cell_type'] == 'Unclassified']

    # Plot b 
    print("  Generating plot b: TTP vs Burst Index...")
    plot_data_b = valid_df_basic[['trough_to_peak_ms', 'burst_index', 'cell_type']].copy(); plot_data_b.dropna(inplace=True) # Use valid_df_basic
    if plot_data_b.empty: print("   No valid data points found for plot b.")
    else: plt.figure(figsize=(8, 6)); 
    types_present_b = [t for t in CELL_TYPE_ORDER if t in plot_data_b['cell_type'].unique()]; 
    unclassified_present_b = 'Unclassified' in plot_data_b['cell_type'].unique(); 
    [plt.scatter(plot_data_b[plot_data_b['cell_type'] == cell_type]['trough_to_peak_ms'], 
                 plot_data_b[plot_data_b['cell_type'] == cell_type]['burst_index'], alpha=0.6, s=20, label=cell_type, 
                 color=CELL_TYPE_COLORS.get(cell_type, 'grey')) for cell_type in types_present_b]; 
    plt.axvline(NS_PYWS_THRESHOLD_MS, color='k', linestyle='--', linewidth=1, label=f'NS/PY-WS Threshold ({NS_PYWS_THRESHOLD_MS} ms)'); 
    plt.xlabel('Trough-to-Peak Duration (ms)'); 
    plt.ylabel('Burst Index'); 
    plt.title(f'Trough-to-Peak vs Burst Index ({global_basename})'); 
    plt.legend(); 
    plt.grid(True, linestyle='--', alpha=0.6); 
    plot_filepath_b = os.path.join(output_dir, f'plot_b_TTP_vs_BurstIdx_{global_basename}.png'); 
    try: plt.savefig(plot_filepath_b, dpi=300); print(f"  Plot (b) saved to: {os.path.basename(plot_filepath_b)}"); 
    except Exception as e: print(f"  Error saving plot b: {e}"); 
    finally: plt.close()

    # Plot c (Unchanged - legend fix already applied)
    print("  Generating plot c: Firing Rate Distribution...")
    plot_data_c = valid_df_basic[['firing_rate_hz', 'cell_type']].copy(); plot_data_c.dropna(inplace=True); plot_data_c = plot_data_c[plot_data_c['firing_rate_hz'] > 1e-3]
    if plot_data_c.empty: print("   No valid positive firing rate data found for plot c.")
    elif sns is None: print("   Seaborn not found, skipping KDE plot c.")
    else: 
        types_present_c = [t for t in CELL_TYPE_ORDER if t in plot_data_c['cell_type'].unique()]; 
        palette_c = {t: CELL_TYPE_COLORS.get(t, 'grey') for t in types_present_c}; 
        fig_c, ax_c = plt.subplots(figsize=(8, 5)); 
        try: 
            sns.kdeplot(data=plot_data_c, x='firing_rate_hz', hue='cell_type', hue_order=types_present_c,
                        palette=palette_c, common_norm=False, log_scale=True, fill=True, alpha=0.4, linewidth=1.5, ax=ax_c); 
            sns.rugplot(data=plot_data_c, x='firing_rate_hz', hue='cell_type', hue_order=types_present_c, 
                        palette=palette_c, height=0.05, ax=ax_c, legend=False); 
            ax_c.set_xlabel("Firing rate (spikes/s) [Log Scale]"); ax_c.set_ylabel("Density"); 
            ax_c.set_title(f"Firing Rate Distribution ({global_basename})"); 
            ax_c.spines['top'].set_visible(False); 
            ax_c.spines['right'].set_visible(False); 
            legend = ax_c.get_legend(); 
            if legend: 
                legend.set_title("Cell Type"); 
                legend.set_frame_on(False); 
                plot_filepath_c = os.path.join(output_dir, f'plot_c_FiringRate_Dist_{global_basename}.png'); 
                plt.savefig(plot_filepath_c, bbox_inches='tight', dpi=300); 
                print(f"  Plot (c) saved to: {os.path.basename(plot_filepath_c)}"); 
        except Exception as e: 
            print(f"  Error generating plot c: {e}"); 
        finally: plt.close(fig_c)

    # Plot d (Unchanged - legend fix already applied)
    print("  Generating plot d: Burst Index Distribution...")
    plot_data_d = valid_df_basic[['burst_index', 'cell_type']].copy(); 
    plot_data_d['burst_index'] = plot_data_d['burst_index'].replace([np.inf, -np.inf], np.nan); 
    plot_data_d.dropna(inplace=True)
    if plot_data_d.empty: print("   No valid burst index data found for plot d.")
    elif sns is None: print("   Seaborn not found, skipping KDE plot d.")
    else: 
        types_present_d = [t for t in CELL_TYPE_ORDER if t in plot_data_d['cell_type'].unique()]; 
        palette_d = {t: CELL_TYPE_COLORS.get(t, 'grey') for t in types_present_d}; 
        fig_d, ax_d = plt.subplots(figsize=(8, 5)); 
        try:
            min_bi_d = plot_data_d['burst_index'].min(); max_bi_d = plot_data_d['burst_index'].max()
            sns.kdeplot(data=plot_data_d, x='burst_index', hue='cell_type', hue_order=types_present_d,
                        palette=palette_d, common_norm=False, fill=True, alpha=0.4, linewidth=1.5, ax=ax_d, clip=(min_bi_d, max_bi_d))

            sns.rugplot(data=plot_data_d, x='burst_index', hue='cell_type', hue_order=types_present_d,
                        palette=palette_d, height=0.05, ax=ax_d, legend=False)

            ax_d.set_xlabel("Burst Index")
            ax_d.set_ylabel("Density")
            ax_d.set_title(f"Burst Index Distribution ({global_basename})")
            ax_d.spines['top'].set_visible(False)
            ax_d.spines['right'].set_visible(False)

            legend = ax_d.get_legend()
            if legend:
                legend.set_title("Cell Type")
                legend.set_frame_on(False)

            plot_filepath_d = os.path.join(output_dir, f'plot_d_BurstIndex_Dist_{global_basename}.png')
            plt.savefig(plot_filepath_d, bbox_inches='tight', dpi=300)
            print(f"  Plot (d) saved to: {os.path.basename(plot_filepath_d)}")
        except Exception as e:
            print(f"  Error generating plot d: {e}")
        finally:
            plt.close(fig_d)
            
    # --- Plot e: Histograms (ISI Violations, Num Spikes Proxy) ---
    print("  Generating plot e: ISI Viol./Num Spikes Histograms...")
    # Ensure valid_df_basic is used here
    plot_data_e_isi = valid_df_basic['isi_viol_rate_allensdk'].dropna()
    plot_data_e_nspikes = valid_df_basic['num_spikes'].dropna()

    if plot_data_e_isi.empty and plot_data_e_nspikes.empty:
        print("   No valid data for plot e.")
    else:
        fig_e, axes_e = plt.subplots(1, 2, figsize=(10, 4)) # 1 row, 2 columns

        # --- Left plot: ISI Violations ---
        if not plot_data_e_isi.empty:
            ax = axes_e[0]
            # Determine bins dynamically, using log scale
            # Use a small epsilon to handle potential zero values after filtering if needed
            epsilon = 1e-10
            min_isi_val = plot_data_e_isi[plot_data_e_isi > epsilon].min() if np.any(plot_data_e_isi > epsilon) else epsilon
            max_isi_val = plot_data_e_isi.max()

            # Ensure valid range for logspace
            min_isi_log = np.log10(max(epsilon, min_isi_val)) # Ensure min is positive
            max_isi_log = np.log10(max(min_isi_val + epsilon, max_isi_val)) # Ensure max > min
            # Add buffer and ensure min reasonable tick value
            max_isi_log = max(np.log10(0.5), max_isi_log) + 0.1
            if max_isi_log <= min_isi_log:
                 max_isi_log = min_isi_log + 1 # Ensure range > 0

            bins_e_isi = np.logspace(min_isi_log, max_isi_log, 30)

            ax.hist(plot_data_e_isi, bins=bins_e_isi, color='grey', edgecolor='black', alpha=0.7)
            perc_95_isi = np.percentile(plot_data_e_isi, 95)
            # Plot vertical line only if perc_95_isi is within plot limits
            if perc_95_isi >= 10**min_isi_log and perc_95_isi <= 10**max_isi_log:
                 ax.axvline(perc_95_isi, color='red', linestyle='--', linewidth=1.5, label=f'95th ({perc_95_isi:.3f})')
                 ax.legend(frameon=False) # Show legend only if line is plotted

            ax.set_xscale('log')
            ax.set_xlabel('Refractory period violation\n(AllenSDK `isi_viol`)')
            ax.set_ylabel('Units')
            # ax.legend(frameon=False) # Moved inside if block above
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        else:
            # Handle case with no ISI data
            ax = axes_e[0]
            ax.text(0.5, 0.5, "No ISI viol data", ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)

        # --- Right plot: Num Spikes (Proxy for stability) ---
        if not plot_data_e_nspikes.empty:
            ax = axes_e[1]
            # Use linear bins, maybe dynamic range or log scale if range is huge? Linear first.
            min_nsp = plot_data_e_nspikes.min()
            max_nsp = plot_data_e_nspikes.max()
            # Handle case where all values are the same
            if np.isclose(max_nsp, min_nsp):
                 bins_e_nsp = np.linspace(min_nsp - 1, max_nsp + 1, 20)
            else:
                 bins_e_nsp = np.linspace(min_nsp, max_nsp, 30) # Linear bins

            ax.hist(plot_data_e_nspikes, bins=bins_e_nsp, color='grey', edgecolor='black', alpha=0.7)
            perc_95_nsp = np.percentile(plot_data_e_nspikes, 95)
            # Plot vertical line only if perc_95_nsp is within plot limits
            if perc_95_nsp >= min_nsp and perc_95_nsp <= max_nsp :
                 ax.axvline(perc_95_nsp, color='red', linestyle='--', linewidth=1.5, label=f'95th ({perc_95_nsp:.0f})')
                 ax.legend(frameon=False) # Show legend only if line is plotted

            # Optional: Use log scale if range is very large
            # if max_nsp / max(1, min_nsp) > 1000:
            #    ax.set_xscale('log')
            #    ax.set_xlabel('Number of Spikes [Log Scale]')
            # else:
            ax.set_xlabel('Number of Spikes')

            ax.set_ylabel('Units')
            # ax.legend(frameon=False) # Moved inside if block above
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        else:
            # Handle case with no Num Spikes data
            ax = axes_e[1]
            ax.text(0.5, 0.5, "No Num Spikes data", ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)

        # --- Figure title and saving ---
        fig_e.suptitle(f'Data Quality Histograms ({global_basename})', y=1.02)
        plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust layout

        plot_filepath_e = os.path.join(output_dir, f'plot_e_QualityHists_{global_basename}.png')
        # Corrected try/except/finally for saving
        try:
            plt.savefig(plot_filepath_e, bbox_inches='tight', dpi=300)
            print(f"  Plot (e) saved to: {os.path.basename(plot_filepath_e)}")
        except Exception as e:
            print(f"  Error saving plot e: {e}")
        finally:
            plt.close(fig_e) 

    print("--- Global plot generation finished ---")


def process_kilosort_directory(kilosort_folder):
    """Loads data, classifies units, saves results FOR A SINGLE Kilosort directory. Returns dataframe."""
    print(f"\n=================================================")
    print(f" Processing Directory: {kilosort_folder}")
    print(f"=================================================")
    folder_basename = os.path.basename(kilosort_folder)
    parent_dir_name = os.path.basename(os.path.dirname(kilosort_folder))
    output_base_name = f"{parent_dir_name}_{folder_basename}"

    try: # Outer try block for directory-level errors
        # --- Paths ---
        spike_times_file = os.path.join(kilosort_folder, 'spike_times.npy')
        spike_times_sec_file = os.path.join(kilosort_folder, 'spike_times_sec.npy')
        spike_clusters_file = os.path.join(kilosort_folder, 'spike_clusters.npy')
        channel_pos_file = os.path.join(kilosort_folder, 'channel_positions.npy')
        metrics_file = os.path.join(kilosort_folder, 'metrics.csv')
        brain_regions_file_name = 'channel_brain_regions.csv'
        brain_regions_file = os.path.join(kilosort_folder, brain_regions_file_name)

        # --- Load Kilosort Data ---
        print("  Loading data...")
        spike_times_seconds = None

        # *** Enhanced Debugging for Spike Time Loading ***
        print(f"   DEBUG: Full path check 1: '{spike_times_sec_file}'")
        sec_file_exists = os.path.exists(spike_times_sec_file)
        print(f"   DEBUG: Result 1 (sec file exists?): {sec_file_exists}")

        print(f"   DEBUG: Full path check 2: '{spike_times_file}'")
        npy_file_exists = os.path.exists(spike_times_file)
        print(f"   DEBUG: Result 2 (npy file exists?): {npy_file_exists}")

        # DEBUG: If npy check failed, list directory contents
        if not npy_file_exists:
             print(f"   DEBUG: os.path.exists returned False for spike_times.npy. Listing directory contents for {kilosort_folder}:")
             try:
                 contents = os.listdir(kilosort_folder)
                 print(f"     DEBUG Contents: {contents}")
                 if 'spike_times.npy' in contents:
                     print("     DEBUG WARNING: 'spike_times.npy' is IN os.listdir() but os.path.exists() returned False!")
                 else:
                     print("     DEBUG: 'spike_times.npy' NOT FOUND in os.listdir().")
             except Exception as list_err:
                 print(f"     DEBUG ERROR listing directory contents: {list_err}")
        # *** End Enhanced Debugging Init Checks ***

        try: # Inner try for the loading logic based on checks
            if sec_file_exists:
                print("   DEBUG Action: Trying to load spike_times_sec.npy")
                spike_times_seconds = np.load(spike_times_sec_file).flatten()
                print(f"   Loaded spike times from: {os.path.basename(spike_times_sec_file)}")
                print(f"   (Using fixed SR: {SAMPLING_RATE_HZ} Hz for calculations)")

            elif npy_file_exists:
                # This block *should* execute
                print("   DEBUG Action: Trying to load spike_times.npy")
                print(f"   '{os.path.basename(spike_times_sec_file)}' not found. Loading '{os.path.basename(spike_times_file)}'.") # Normal message
                spike_times = np.load(spike_times_file).flatten()
                print("   DEBUG: np.load('spike_times.npy') successful.") # Confirm load worked
                if SAMPLING_RATE_HZ <= 0:
                    raise ValueError("Invalid fixed SAMPLING_RATE_HZ defined.")
                spike_times_seconds = spike_times / SAMPLING_RATE_HZ
                print(f"   Loaded spike times from: {os.path.basename(spike_times_file)} and converted to seconds using fixed SR={SAMPLING_RATE_HZ} Hz.")

            else:
                # This block should only run if both variables sec_file_exists and npy_file_exists are False
                print("   DEBUG Action: Raising FileNotFoundError as neither file existed based on checks above.")
                raise FileNotFoundError(f"Neither '{os.path.basename(spike_times_sec_file)}' nor '{os.path.basename(spike_times_file)}' found based on os.path.exists checks.")

        except Exception as e:
            # This catches errors from the specific if/elif/else block above it
            print(f"   DEBUG: Caught exception during spike time loading attempt: {type(e).__name__}")
            # This is the error message you are seeing in the log
            print(f"  ERROR loading spike times: {e}. Skipping directory.")
            return None # Exit this function for this directory
        # *** End Spike Time Loading Section ***

        # --- Load Clusters (No changes needed below this point in this function) ---
        try:
            if not os.path.exists(spike_clusters_file): raise FileNotFoundError(f"Spike clusters file '{os.path.basename(spike_clusters_file)}' not found.")
            spike_clusters = np.load(spike_clusters_file).flatten(); print(f"   Loaded spike clusters from: {os.path.basename(spike_clusters_file)}")
        except Exception as e: print(f"  ERROR loading spike clusters: {e}. Skipping directory."); return None

        if len(spike_times_seconds) != len(spike_clusters): print(f"  ERROR: Mismatch spike times/clusters length. Skipping directory."); return None

        channel_pos = None
        if os.path.exists(channel_pos_file):
            try: channel_pos = np.load(channel_pos_file); print(f"   Loaded channel positions from: {os.path.basename(channel_pos_file)}");
            except Exception as e: print(f"   Warning: Error loading channel positions file '{os.path.basename(channel_pos_file)}': {e}")
        else: print(f"   Warning: Channel positions file '{os.path.basename(channel_pos_file)}' not found.")

        try:
            if not os.path.exists(metrics_file): raise FileNotFoundError(f"Metrics file '{os.path.basename(metrics_file)}' not found.")
            metrics_df = pd.read_csv(metrics_file); id_col = None
            if 'cluster_id' in metrics_df.columns: id_col = 'cluster_id'
            elif 'id' in metrics_df.columns: id_col = 'id'; print("   Warning: Using 'id' column from metrics.csv.")
            else: raise ValueError("Neither 'cluster_id' nor 'id' found in metrics.")
            metrics_df.rename(columns={id_col: 'cluster_id'}, inplace=True); metrics_df['cluster_id'] = pd.to_numeric(metrics_df['cluster_id'], errors='coerce'); metrics_df.dropna(subset=['cluster_id'], inplace=True); metrics_df['cluster_id'] = metrics_df['cluster_id'].astype(int); print(f"   Loaded metrics from: {os.path.basename(metrics_file)}");
        except Exception as e: print(f"  ERROR loading or processing metrics file: {e}. Skipping directory."); return None

        brain_regions_df = None
        if os.path.exists(brain_regions_file):
            try: brain_regions_df = pd.read_csv(brain_regions_file, index_col='global_channel_index'); print(f"   Loaded brain region mapping.");
            except Exception as e: print(f"   Warning: Error processing brain regions file '{brain_regions_file}': {e}")
        else: print(f"   Warning: Brain regions file '{brain_regions_file_name}' not found.")


        # --- Main Processing Loop ---
        # ... (Rest of function is unchanged) ...
        print(f"  Calculating recording duration..."); good_clusters_final = []; unique_clusters_all = np.unique(spike_clusters)
        print(f"  Found {len(unique_clusters_all)} unique clusters. Processing units...")
        min_time = np.nanmin(spike_times_seconds); max_time = np.nanmax(spike_times_seconds)
        if np.isnan(min_time) or np.isnan(max_time): print("   Warning: Cannot determine recording duration."); recording_duration = 0.0
        else: recording_duration = max_time - min_time
        if recording_duration <= 0: print(f"   Warning: Recording duration non-positive ({recording_duration:.2f}s).")
        fit_failures = 0; skipped_fits = 0
        for i, cluster_id in enumerate(unique_clusters_all):
            cluster_id = int(cluster_id)
            cluster_mask = (spike_clusters == cluster_id); cluster_spike_times_seconds = spike_times_seconds[cluster_mask]; n_spikes = len(cluster_spike_times_seconds)
            if n_spikes == 0: continue
            firing_rate = (n_spikes / recording_duration) if recording_duration > 0 else np.inf; percent_short_isi = np.nan
            if n_spikes >= 2: isi = np.diff(np.sort(cluster_spike_times_seconds)) * 1000; len_isi = len(isi); percent_short_isi = (np.sum(isi < 2) / len_isi * 100) if len_isi > 0 else 0.0
            elif n_spikes == 1: percent_short_isi = 0.0
            metrics_row = metrics_df[metrics_df['cluster_id'] == cluster_id]
            if not metrics_row.empty:
                metrics_data = metrics_row.iloc[0];
                if 'peak_channel' not in metrics_data.index or pd.isna(metrics_data['peak_channel']): continue
                try: peak_channel_index = int(metrics_data['peak_channel'])
                except ValueError: continue
                contam_rate = metrics_data.get('contam_rate', None); l_ratio = metrics_data.get('l_ratio', None); isi_viol_rate = metrics_data.get('isi_viol', None)
                try: contam_rate = float(contam_rate) if pd.notna(contam_rate) else None; 
                except (ValueError, TypeError): contam_rate = None
                try: l_ratio = float(l_ratio) if pd.notna(l_ratio) else None; 
                except (ValueError, TypeError): l_ratio = None
                try: isi_viol_rate = float(isi_viol_rate) if pd.notna(isi_viol_rate) else None; 
                except (ValueError, TypeError): isi_viol_rate = None
                is_good = (firing_rate > MIN_FIRING_RATE_HZ and contam_rate is not None and contam_rate < MAX_CONTAM_RATE and l_ratio is not None and l_ratio < MAX_L_RATIO and isi_viol_rate is not None and isi_viol_rate < MAX_ISI_VIOL_RATE)
                # is_good = (contam_rate is not None and contam_rate < MAX_CONTAM_RATE and l_ratio is not None and l_ratio < MAX_L_RATIO and isi_viol_rate is not None and isi_viol_rate < MAX_ISI_VIOL_RATE)
                if is_good:
                    burst_index = calculate_burst_index(cluster_spike_times_seconds, window_ms=500)
                    trough_to_peak_ms = metrics_data.get('duration', None)
                    try: trough_to_peak_ms = float(trough_to_peak_ms) if pd.notna(trough_to_peak_ms) else None
                    except (ValueError, TypeError): trough_to_peak_ms = None
                    tau_rise_parameter = np.nan; acg_fit_params = None; popt = None
                    if n_spikes >= MIN_SPIKES_FOR_ACG_FIT:
                        acg_counts, bin_centers = calculate_acg(cluster_spike_times_seconds, ACG_BIN_SIZE_MS, ACG_WINDOW_MS)
                        if acg_counts is not None and bin_centers is not None:
                             popt, pcov = fit_acg(acg_counts, bin_centers)
                             if popt is not None: tau_rise_parameter = popt[1]; acg_fit_params = popt
                             else: fit_failures += 1;
                        # else: skipped_fits += 1 
                    else: skipped_fits +=1
                    cell_type = classify_cell_type_full(trough_to_peak_ms, tau_rise_parameter)
                    good_cluster_info = { 'cluster_id': int(cluster_id), 'cell_type': cell_type, 'firing_rate_hz': round(firing_rate, 3), 'isi_violation_percent_manual': round(percent_short_isi, 3) if pd.notna(percent_short_isi) else None, 'isi_viol_rate_allensdk': isi_viol_rate, 'contamination_rate': contam_rate, 'l_ratio': l_ratio, 'num_spikes': n_spikes, 'peak_channel_index_0based': peak_channel_index, 'burst_index': burst_index if pd.notna(burst_index) else None, 'trough_to_peak_ms': trough_to_peak_ms if pd.notna(trough_to_peak_ms) else None, 'acg_tau_rise': tau_rise_parameter if pd.notna(tau_rise_parameter) else None, 'acg_tau_decay': acg_fit_params[0] if acg_fit_params is not None else None, 'acg_tau_burst': acg_fit_params[6] if acg_fit_params is not None else None, 'acg_refrac': acg_fit_params[5] if acg_fit_params is not None else None, 'acg_fit_rsquare': None, 'shank_index': None, 'distance_from_entry_um': None, 'acronym': None, 'brain_region_name': None, 'channel_depth_um_fallback': None }
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

        print(f"\n  Processed {len(unique_clusters_all)} unique clusters in this directory.")
        print(f"  ACG fitting (Triple Exp): Skipped for {skipped_fits} units, Failed for {fit_failures} units.")
        n_good = len(good_clusters_final)
        print(f"  Found {n_good} good clusters meeting quality criteria.")

        if good_clusters_final:
            good_clusters_df = pd.DataFrame(good_clusters_final)
            output_npy_file_name = f'good_clusters_processed_{output_base_name}_CellExplorerACG.npy'; output_npy_path = os.path.join(kilosort_folder, output_npy_file_name)
            try: np.save(output_npy_path, np.array(good_clusters_final, dtype=object), allow_pickle=True); print(f"\n  Per-directory data saved to: {output_npy_path}");
            except Exception as e: print(f"  Error saving per-directory NPY file: {e}")
            try: good_clusters_df_no_spikes = good_clusters_df; cols_order = ['cluster_id', 'cell_type', 'acronym', 'brain_region_name', 'firing_rate_hz', 'trough_to_peak_ms', 'burst_index', 'acg_tau_rise', 'acg_tau_decay', 'acg_tau_burst', 'acg_refrac', 'acg_fit_rsquare', 'isi_violation_percent_manual', 'isi_viol_rate_allensdk', 'contamination_rate', 'l_ratio', 'num_spikes', 'peak_channel_index_0based', 'shank_index', 'distance_from_entry_um', 'channel_depth_um_fallback']; cols_order_present = [col for col in cols_order if col in good_clusters_df_no_spikes.columns]; missing_cols = [col for col in good_clusters_df_no_spikes.columns if col not in cols_order_present]; final_cols_order = cols_order_present + missing_cols; good_clusters_df_no_spikes = good_clusters_df_no_spikes[final_cols_order]; output_csv_path = os.path.join(kilosort_folder, f'good_clusters_processed_{output_base_name}_CellExplorerACG.csv'); good_clusters_df_no_spikes.to_csv(output_csv_path, index=False, float_format='%.4f'); print(f"  Per-directory metrics summary saved to: {output_csv_path}");
            except Exception as e: print(f"  Error saving per-directory CSV file: {e}")
            print(f"--- Finished processing {kilosort_folder}. Found {len(good_clusters_df)} good units. ---")
            return good_clusters_df
        else: print("\n  No 'good' clusters identified in this directory."); return None
    except Exception as e:
         print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"); print(f"!!! UNHANDLED ERROR processing directory {kilosort_folder}: {e}"); print(f"!!! Skipping to next directory."); traceback.print_exc(); print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"); return None

# --- Main Execution Script ---
if __name__ == "__main__":
    
    print(f"Using fixed sampling rate: {SAMPLING_RATE_HZ} Hz")
    if SAMPLING_RATE_HZ <= 0:
        print("Error: Invalid fixed SAMPLING_RATE_HZ defined in script configuration.")
        sys.exit(1)

    search_path = os.path.join(BASE_OUTPUT_PARENT_DIR, DIRECTORY_SEARCH_PATTERN)
    print(f"\nSearching for Kilosort directories matching: {search_path}")
    directories_to_process = glob.glob(search_path, recursive=False)
    if not directories_to_process: print("\nError: No directories found matching the specified pattern."); sys.exit(1)
    else: print(f"\nFound {len(directories_to_process)} directories to process:"); directories_to_process.sort(); [print(f"  - {d}") for d in directories_to_process]

    print("\n--- Starting Batch Processing ---")
    all_good_clusters_list = []
    for current_kilosort_folder in directories_to_process:
        result_df = process_kilosort_directory(current_kilosort_folder) 
        if result_df is not None and not result_df.empty:
            result_df['source_directory'] = os.path.basename(os.path.dirname(current_kilosort_folder))
            all_good_clusters_list.append(result_df)

    if not all_good_clusters_list: print("\nNo good clusters found in any directory. Cannot generate global plots."); sys.exit(0)
    else:
        print(f"\n--- Aggregating data from {len(all_good_clusters_list)} directories ---")
        global_df = pd.concat(all_good_clusters_list, ignore_index=True)
        print(f"Total good clusters found across all directories: {len(global_df)}")
        selected_regions = None
        if 'acronym' not in global_df.columns: print("\nWarning: 'acronym' column missing. Cannot filter by region.")
        else:
            available_regions = sorted(global_df['acronym'].dropna().astype(str).unique())
            if not available_regions: print("\nNo brain region acronyms found.")
            else:
                print("\nAvailable brain region acronyms:"); print(", ".join(available_regions))
                print("\nEnter region acronym(s) to filter by (comma-separated), or leave blank for all:")
                region_input = input("> ").strip()
                if region_input:
                    selected_regions_input = [r.strip().upper() for r in region_input.split(',')]
                    available_regions_upper = [ar.upper() for ar in available_regions]
                    valid_selected = [r for r in selected_regions_input if r in available_regions_upper]
                    invalid_selected = [r for r in selected_regions_input if r not in available_regions_upper]
                    if invalid_selected: print(f"  Warning: Invalid region(s): {', '.join(invalid_selected)}")
                    if not valid_selected: print("  No valid regions selected. Using all regions."); selected_regions = None
                    else: selected_regions = valid_selected 

        global_plot_dir = BASE_OUTPUT_PARENT_DIR if SAVE_GLOBAL_PLOTS_IN_PARENT_DIR else os.path.join(BASE_OUTPUT_PARENT_DIR, "global_plots"); 
        os.makedirs(global_plot_dir, exist_ok=True)
        global_plot_basename = "GLOBAL";

        # Call global plotting function
        generate_global_plots(global_df, global_plot_dir, global_plot_basename, region_filter=selected_regions)

    print("\n\nScript finished processing all directories.")