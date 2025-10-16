# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 12:15:00 2025 
# Added compound scatter+marginal plot
Metrics.csv from ecephys repository https://github.com/AllenInstitute/ecephys_spike_sorting/tree/main/ecephys_spike_sorting/modules/mean_waveforms
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
import warnings
import sys
from scipy.optimize import curve_fit, OptimizeWarning 
from scipy.stats import gaussian_kde 

# --- Configuration ---
NS_PYWS_THRESHOLD_MS = 0.425
# Threshold applies to the fitted ACG tau_rise time constant (parameter 'b')
WS_PY_TAU_RISE_THRESHOLD = 6.0 # Unit is ms
MIN_SPIKES_FOR_ACG_FIT = 100 # Min spikes to attempt ACG fit
ACG_BIN_SIZE_MS = 0.5 # Bin size (matches CellExplorer)
ACG_WINDOW_MS = 50 # Window for ACG calculation/fit (matches CellExplorer fit range 0-50ms)

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
    """Calculates the burst index for a unit based on its spike times."""
    # ... (Function unchanged) ...
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
    # ... (Function largely unchanged, but now uses window_ms=50) ...
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

    # Add a 0 bin edge for histogram function if needed, but we only fit positive lags
    # Adjust range slightly to ensure first bin center is >= bin_size_ms / 2
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


# *** NEW Fitting Function: Triple Exponential (Matches MATLAB fit_ACG.m) ***
# x: time points (bin centers in ms, starting > 0)
# Parameters: a, b, c, d, e, f, g, h (matching MATLAB coefficient order)
def triple_exponential_acg_fit_func(x, a, b, c, d, e, f, g, h):
    """
    Triple exponential function matching CellExplorer's fit_ACG.m:
    max(c*(exp(-(x-f)/a)-d*exp(-(x-f)/b))+h*exp(-(x-f)/g)+e,0)
    a = tau_decay, b = tau_rise, c = amp_decay, d = multiplier_rise,
    e = asymptote, f = t_refrac, g = tau_burst, h = amp_burst
    """
    # Ensure time constants are positive
    safe_a = max(1e-6, a) # tau_decay
    safe_b = max(1e-6, b) # tau_rise
    safe_g = max(1e-6, g) # tau_burst

    # Calculate time relative to refractory period end
    # Note: MATLAB's fit starts x from 0.5ms, Python's x starts from bin_center (e.g., 0.25ms)
    # The 'f' (t_refrac) parameter handles the effective start time shift.
    t = np.maximum(0, x - f) # Time since refractory period

    # Calculate each exponential term
    term1 = c * np.exp(-t / safe_a)
    term2 = c * d * np.exp(-t / safe_b) # Note: c*d is the effective amplitude here
    term3 = h * np.exp(-t / safe_g)

    # Combine terms and add baseline
    fit_values = term1 - term2 + term3 + e

    # Apply max(..., 0) as in MATLAB formula
    # Also, ensure output is 0 for x <= f (refractory period)
    final_values = np.where(x <= f, 0, fit_values)
    return np.maximum(0, final_values)


def fit_acg(acg_counts, bin_centers):
    """
    Attempts to fit the triple exponential ACG function matching CellExplorer.
    Requires ACG counts and bin centers for positive lags > 0.
    """
    if acg_counts is None or bin_centers is None or len(acg_counts) < 8: # Need points > #params
        return None, None

    # Ensure equal length
    min_len = min(len(acg_counts), len(bin_centers))
    if min_len == 0: return None, None
    xdata = bin_centers[:min_len]
    ydata = acg_counts[:min_len]

    # Ensure xdata starts > 0
    valid_indices = xdata > 0
    if not np.any(valid_indices): return None, None # No points > 0 to fit
    xdata = xdata[valid_indices]
    ydata = ydata[valid_indices]
    if len(xdata) < 8: return None, None # Check again after filtering

    # --- Initial Guesses (p0) and Bounds from MATLAB fit_ACG.m ---
    # Parameters: [a, b, c, d, e, f, g, h]
    # MATLAB a0 = [20, 1, 30, 2, 0.5, 5, 1.5, 2]; -> Order matches Python signature now
    # Indices:       0, 1,  2, 3,   4, 5,   6, 7
    p0 = [ 20,  1,  30,  2, 0.5,  5, 1.5,  2] # Direct copy from MATLAB a0

    # MATLAB lb = [1, 0.1, 0, 0, -30, 0, 0.1, 0];
    # MATLAB ub = [500, 50, 500, 15, 50, 20, 5, 100];
    bounds = (
        [  1, 0.1,   0,   0, -30,   0, 0.1,   0], # Lower bounds
        [500,  50, 500,  15,  50,  20,   5, 100]  # Upper bounds
    )
    # Check if ydata max requires adjusting upper bounds for amplitudes (c, h) or baseline (e)?
    max_y = np.max(ydata)
    bounds[1][2] = max(bounds[1][2], max_y * 1.5) # Upper bound for c
    bounds[1][7] = max(bounds[1][7], max_y * 1.5) # Upper bound for h
    bounds[1][4] = max(bounds[1][4], max_y * 0.5) # Upper bound for e
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            warnings.simplefilter("ignore", category=OptimizeWarning)
            popt, pcov = curve_fit(
                triple_exponential_acg_fit_func, # *** Use  triple exponential function ***
                xdata,
                ydata,
                p0=p0,
                bounds=bounds,
                maxfev=5000, # Keep high max iterations
                method='trf' # Keep method
             )
        if pcov is None or np.any(np.isinf(np.diag(pcov))) or np.any(np.isnan(np.diag(pcov))):
             pass # print(f"  Warning: Fit converged but covariance matrix is invalid.")
        return popt, pcov
    except RuntimeError: return None, None
    except ValueError: return None, None
    except Exception: return None, None

def classify_cell_type_full(trough_to_peak_ms, tau_rise_fit):
    """
    Classifies cell type using TTP and the fitted ACG tau_rise time constant.
    Matches CellExplorer logic (using tau_rise > 6ms).
    """
    if pd.isna(trough_to_peak_ms): return 'Unclassified'
    ttp = trough_to_peak_ms

    if ttp <= NS_PYWS_THRESHOLD_MS:
        return 'NS Interneuron'
    else: # ttp > NS_PYWS_THRESHOLD_MS
        # Check the fitted ACG parameter 'b' (tau_rise) against the threshold
        if pd.notna(tau_rise_fit) and tau_rise_fit > WS_PY_TAU_RISE_THRESHOLD:
            return 'WS Interneuron'
        else:
            # If tau_rise is missing, NaN, or below threshold, classify as Pyramidal
            return 'Pyramidal'

def generate_classification_plots(df, output_dir, base_filename):
    """Generates and saves classification plots based on the dataframe."""
    print("\n--- Generating Classification Plots ---")
    if df.empty: print("Dataframe is empty. Skipping plot generation."); return

    # --- Define required columns for different plots ---
    basic_plot_cols = ['trough_to_peak_ms', 'firing_rate_hz', 'burst_index', 'isi_viol_rate_allensdk', 'cell_type']
    compound_plot_cols = ['trough_to_peak_ms', 'acg_tau_rise', 'cell_type']
    all_req_cols = list(set(basic_plot_cols + compound_plot_cols)) # Unique columns needed overall

    # Check if all required columns exist
    missing_cols = [col for col in all_req_cols if col not in df.columns]
    if missing_cols: print(f"Warning: Missing columns required for plotting: {missing_cols}. Skipping plots."); return

    # --- Plot a: Compound Scatter + Marginal Distributions Plot ---
    print(" Generating compound scatter plot with marginal distributions...")
    plot_data_a = df[compound_plot_cols].copy()
    plot_data_a.dropna(subset=compound_plot_cols, inplace=True)
    plot_data_a = plot_data_a[plot_data_a['acg_tau_rise'] > 0] # Filter for log scale

    if plot_data_a.empty:
        print(" No valid data points found for compound scatter plot (need TTP and positive ACG Tau Rise).")
    else:
        print(f" Plotting compound scatter for {len(plot_data_a)} units.")
        plot_types = ['NS Interneuron', 'WS Interneuron', 'Pyramidal'] # Order for plotting
        plot_data_a_filtered = plot_data_a[plot_data_a['cell_type'].isin(plot_types + ['Unclassified'])]
        types_present_a = [t for t in plot_types if t in plot_data_a_filtered['cell_type'].unique()]
        unclassified_present_a = 'Unclassified' in plot_data_a_filtered['cell_type'].unique()

        fig = plt.figure(figsize=(7, 7)) # Slightly adjusted size
        gs = gridspec.GridSpec(2, 2, width_ratios=[1, 4], height_ratios=[4, 1], wspace=0.02, hspace=0.02) # Reduced space
        ax_scatter = fig.add_subplot(gs[0, 1])
        ax_histx = fig.add_subplot(gs[1, 1], sharex=ax_scatter)
        ax_histy = fig.add_subplot(gs[0, 0], sharey=ax_scatter)

        # Scatter Plot
        ax_scatter.set_yscale('log')
        legend_handles = []
        for cell_type in types_present_a:
            type_data = plot_data_a_filtered[plot_data_a_filtered['cell_type'] == cell_type]
            color = CELL_TYPE_COLORS.get(cell_type, 'grey')
            handle = ax_scatter.scatter(type_data['trough_to_peak_ms'], type_data['acg_tau_rise'], alpha=0.7, s=15, label=cell_type, color=color, edgecolors='none') # Slightly smaller points
            legend_handles.append(handle)
        if unclassified_present_a:
            type_data = plot_data_a_filtered[plot_data_a_filtered['cell_type'] == 'Unclassified']
            color = CELL_TYPE_COLORS['Unclassified']
            ax_scatter.scatter(type_data['trough_to_peak_ms'], type_data['acg_tau_rise'], alpha=0.4, s=10, color=color, edgecolors='none')

        ax_scatter.tick_params(axis="y", labelleft=False, left=False) # Hide y-ticks on scatter
        ax_scatter.tick_params(axis="x", labelbottom=False, bottom=False) # Hide x-ticks on scatter
        ax_scatter.grid(False) 
        ax_scatter.legend(handles=legend_handles, loc='upper right', frameon=False)
        xlim = ax_scatter.get_xlim()
        ylim = ax_scatter.get_ylim()

        # X-Marginal (TTP)
        for cell_type in types_present_a:
            type_data = plot_data_a_filtered[plot_data_a_filtered['cell_type'] == cell_type]['trough_to_peak_ms'].values
            if len(type_data) < 2: continue
            color = CELL_TYPE_COLORS.get(cell_type, 'grey')
            try:
                kde = gaussian_kde(type_data)
                xs = np.linspace(xlim[0], xlim[1], 300)
                kde_y = kde(xs)
                if np.max(kde_y) > 1e-6: # Avoid division by zero if KDE is flat
                    scaled_y = kde_y / np.max(kde_y)
                    ax_histx.plot(xs, scaled_y, color=color, lw=1.5)
                    ax_histx.fill_between(xs, 0, scaled_y, color=color, alpha=0.3) # Fill area
            except Exception as e: print(f" Warning: Could not compute/plot KDE for TTP of {cell_type}: {e}")
        ax_histx.set_xlabel('Trough-to-peak (ms)')
        ax_histx.set_ylabel('Peak', rotation=0, ha='right', va='center', labelpad=10)
        ax_histx.tick_params(axis="y", left=False, labelleft=False)
        ax_histx.spines['top'].set_visible(False); ax_histx.spines['right'].set_visible(False); ax_histx.spines['left'].set_visible(False)
        ax_histx.set_ylim(0, 1.1)

        # Y-Marginal (ACG Tau Rise - Log space KDE)
        for cell_type in types_present_a:
            type_data = plot_data_a_filtered[plot_data_a_filtered['cell_type'] == cell_type]['acg_tau_rise'].values
            if len(type_data) < 2: continue
            log_data = np.log10(type_data)
            color = CELL_TYPE_COLORS.get(cell_type, 'grey')
            try:
                kde = gaussian_kde(log_data)
                ys_log = np.linspace(np.log10(ylim[0]), np.log10(ylim[1]), 300)
                kde_x = kde(ys_log)
                if np.max(kde_x) > 1e-6:
                    scaled_x = kde_x / np.max(kde_x)
                    ax_histy.plot(scaled_x, 10**ys_log, color=color, lw=1.5) # Plot density vs original value
                    ax_histy.fill_betweenx(10**ys_log, 0, scaled_x, color=color, alpha=0.3) # Fill area
            except Exception as e: print(f" Warning: Could not compute/plot KDE for log10(Tau Rise) of {cell_type}: {e}")
        ax_histy.set_xlabel('Peak', rotation=90, ha='center', va='bottom', labelpad=10)
        ax_histy.set_ylabel(r'ACG $\tau_{rise}$ (ms)')
        ax_histy.tick_params(axis="x", bottom=False, labelbottom=False)
        ax_histy.spines['top'].set_visible(False); ax_histy.spines['right'].set_visible(False); ax_histy.spines['bottom'].set_visible(False)
        ax_histy.set_xlim(1.1, 0) # Reverse x-axis

        # Set limits based on scatter plot AFTER plotting marginals
        ax_scatter.set_xlim(xlim)
        ax_scatter.set_ylim(ylim)

        fig.suptitle(f'Cell Type vs ACG $\\tau_{{rise}}$ & TTP ({base_filename})', y=0.97, fontsize=12)

        # Save the compound plot
        plot_filepath_compound = os.path.join(output_dir, f'plot_a_ScatterMarginals_{base_filename}.png')
        try: plt.savefig(plot_filepath_compound, bbox_inches='tight', dpi=300); print(f" Plot (a - Compound) saved to: {os.path.basename(plot_filepath_compound)}");
        except Exception as e: print(f" Error saving compound plot: {e}")
        finally: plt.close(fig)

    # --- Continue with Plots b, c, d, e using basic_plot_cols ---
    # Ensure valid_df used here has the necessary basic columns
    valid_df_basic = df.dropna(subset=basic_plot_cols)
    if valid_df_basic.empty: print("No valid data points for basic plots (b, c, d, e)."); return
    print(f"Plotting basic plots for {len(valid_df_basic)} units.")
    types_basic = sorted(valid_df_basic['cell_type'].unique())
    data_by_type_basic = {t: valid_df_basic[valid_df_basic['cell_type'] == t] for t in types_basic if t != 'Unclassified'}
    unclassified_data_basic = valid_df_basic[valid_df_basic['cell_type'] == 'Unclassified']


    # --- Plot b: Trough-to-peak vs Burst Index ---
    plt.figure(figsize=(8, 6))
    for cell_type, type_data in data_by_type_basic.items(): plt.scatter(type_data['trough_to_peak_ms'], type_data['burst_index'], alpha=0.6, s=20, label=cell_type, color=CELL_TYPE_COLORS.get(cell_type, 'grey'))
    if not unclassified_data_basic.empty: plt.scatter(unclassified_data_basic['trough_to_peak_ms'], unclassified_data_basic['burst_index'], alpha=0.3, s=15, label='Unclassified', color=CELL_TYPE_COLORS['Unclassified'])
    plt.axvline(NS_PYWS_THRESHOLD_MS, color='k', linestyle='--', linewidth=1, label=f'NS/PY-WS Threshold ({NS_PYWS_THRESHOLD_MS} ms)')
    plt.xlabel('Trough-to-Peak Duration (ms)'); plt.ylabel('Burst Index'); 
    plt.title(f'Trough-to-Peak vs Burst Index ({base_filename})')
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.6);
    plot_filepath = os.path.join(output_dir, f'plot_b_TTP_vs_BurstIdx_{base_filename}.png')
    try: plt.savefig(plot_filepath); print(f" Plot (b) saved to: {os.path.basename(plot_filepath)}");
    except Exception as e: print(f" Error saving plot (b): {e}")
    finally: plt.close()

    # --- Plot c: Distribution of Firing Rates ---
    plt.figure(figsize=(8, 6))
    # Calculate min/max safely, ensuring min_fr > 0 for log scale
    # Use valid_df_basic which contains columns needed for this plot
    positive_fr = valid_df_basic['firing_rate_hz'][valid_df_basic['firing_rate_hz'] > 0]
    if positive_fr.empty:
        print(" Warning: No positive firing rates found for plot (c). Skipping plot.")
        plt.close() # Close the empty figure
    else:
        min_fr = positive_fr.min()
        max_fr = valid_df_basic['firing_rate_hz'].max() # Max can still include 0 if present
        # Ensure min_fr is slightly positive if it ended up zero somehow
        if min_fr <= 0: min_fr = 0.01

        bins_c = np.logspace(np.log10(min_fr), np.log10(max_fr + 1), 30)

        print("  Plot C: Looping through classified types for histogram...") # Diagnostic print
        # Loop through classified types found in data_by_type_basic
        for cell_type, type_data in data_by_type_basic.items():
            # Get positive firing rates for this specific type
            fr_data = type_data['firing_rate_hz'][type_data['firing_rate_hz'] > 0]
            # Only plot if there's data for this type after filtering
            if not fr_data.empty:
                 # *** ADDED DIAGNOSTIC PRINT HERE ***
                 print(f"    -> Plotting histogram for: {cell_type} ({len(fr_data)} units)")
                 plt.hist(fr_data, bins=bins_c, alpha=0.7, label=cell_type, color=CELL_TYPE_COLORS.get(cell_type, 'grey'), density=True)
            else:
                 print(f"    -> Skipping histogram for: {cell_type} (no positive FR data)") # Diagnostic print


        # Handle unclassified data (Code unchanged, should be correct)
        if not unclassified_data_basic.empty:
            fr_data_uncl = unclassified_data_basic['firing_rate_hz'][unclassified_data_basic['firing_rate_hz'] > 0]
            if not fr_data_uncl.empty:
                print(f"    -> Plotting histogram for: Unclassified ({len(fr_data_uncl)} units)") # Diagnostic print
                plt.hist(fr_data_uncl, bins=bins_c, alpha=0.4, label='Unclassified', color=CELL_TYPE_COLORS['Unclassified'], density=True)
            else:
                 print(f"    -> Skipping histogram for: Unclassified (no positive FR data)") # Diagnostic print

        # Set plot properties only if we plotted something
        plt.gca().set_xscale('log')
        plt.xlabel('Firing Rate (Hz) [Log Scale]')
        plt.ylabel('Density')
        plt.title(f'Distribution of Firing Rates by Cell Type ({base_filename})')
        # Ensure legend includes plotted types
        handles, labels = plt.gca().get_legend_handles_labels()
        if handles: # Only show legend if something was plotted
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
    # Indentation level 1 (inside generate_classification_plots function)
    plt.figure(figsize=(8, 6))    
    valid_burst_indices = valid_df_basic['burst_index'].replace([np.inf, -np.inf], np.nan).dropna()
    if valid_burst_indices.empty:       
        print(" No valid burst indices for plot (d).")
        plt.close() # Close the empty figure
    else:
        q1, q99 = valid_burst_indices.quantile([0.01, 0.99])
        min_bi_q = max(0, q1)
        max_bi_q = q99 * 1.1
        if min_bi_q < max_bi_q:             
             min_bi = min_bi_q
             max_bi = max_bi_q
        else:            
            min_bi = valid_burst_indices.min()
            max_bi = valid_burst_indices.max()
        if np.isclose(min_bi, max_bi): 
            offset = 0.1
            min_bi -= offset / 2
            max_bi += offset / 2
            if np.isclose(valid_burst_indices.iloc[0], 0.0):
                 min_bi = 0 # Ensure min doesn't become negative
                 max_bi = offset # Set range 0 to 0.1
        # Final safety check: Ensure min_bi is strictly less than max_bi for linspace
        if not min_bi < max_bi:
            # Indentation level 4
            min_bi = max_bi - 0.1 # Force a minimal range if something went wrong
        bins_d = np.linspace(min_bi, max_bi, 30) # Use different var name bins_d
        # --- End of Corrected Bin Range Logic ---

        # --- Start of Corrected Histogram Plotting Logic ---
        # Plot histograms for classified types
        for cell_type, type_data in data_by_type_basic.items():
             # Indentation level 4
             bi_data = type_data['burst_index'].replace([np.inf, -np.inf], np.nan).dropna()
             # Only plot if data remains after dropping NaNs
             if not bi_data.empty:
                  plt.hist(bi_data, bins=bins_d, alpha=0.7, label=cell_type, color=CELL_TYPE_COLORS.get(cell_type, 'grey'), density=True)
        # Handle unclassified data - CORRECTED BLOCK
        if not unclassified_data_basic.empty:
             bi_data_uncl = unclassified_data_basic['burst_index'].replace([np.inf, -np.inf], np.nan).dropna()
             # Only plot histogram if bi_data_uncl is not empty after dropping NaNs
             if not bi_data_uncl.empty:
                 plt.hist(bi_data_uncl, bins=bins_d, alpha=0.4, label='Unclassified', color=CELL_TYPE_COLORS['Unclassified'], density=True)
        # --- End Corrected Histogram Plotting Logic ---

        plt.xlabel('Burst Index')
        plt.ylabel('Density')
        plt.title(f'Distribution of Burst Index by Cell Type ({base_filename})')
        plt.legend()
        plt.grid(True, axis='y', linestyle='--', alpha=0.6)

        plot_filepath = os.path.join(output_dir, f'plot_d_BurstIndex_Dist_{base_filename}.png')
        try:
             plt.savefig(plot_filepath)
             print(f" Plot (d) saved to: {os.path.basename(plot_filepath)}")
        except Exception as e:
             print(f" Error saving plot (d): {e}")
        finally:
             plt.close() 

    # --- Plot e: Distribution of ISI Violation Rate ---
    plt.figure(figsize=(8, 6)); valid_isi_viol = valid_df_basic['isi_viol_rate_allensdk'].dropna()
    if valid_isi_viol.empty: print(" No valid ISI violation rates for plot (e)."); plt.close()
    else:
        min_isi = 0; q99 = valid_isi_viol.quantile(0.99); max_isi = max(0.5, q99 * 1.2); bins_e = np.linspace(min_isi, max_isi, 30);
        plt.hist(valid_isi_viol, bins=bins_e, alpha=0.5, label='All Good Units', color='darkgrey', density=True); perc_95 = np.percentile(valid_isi_viol, 95); plt.axvline(perc_95, color='red', linestyle=':', linewidth=2, label=f'95th Percentile ({perc_95:.3f})')
        plt.xlabel('ISI Violation Rate (AllenSDK `isi_viol`)'); plt.ylabel('Density'); plt.title(f'Distribution of Refractory Period Violations ({base_filename})')
        plt.legend(); plt.grid(True, axis='y', linestyle='--', alpha=0.6); plt.xlim(left=min_isi);
        plot_filepath = os.path.join(output_dir, f'plot_e_ISIViol_Dist_{base_filename}.png')
        try: plt.savefig(plot_filepath); print(f" Plot (e) saved to: {os.path.basename(plot_filepath)}");
        except Exception as e: print(f" Error saving plot (e): {e}")
        finally: plt.close()

    print("--- Plot generation finished ---")


# --- Main Script ---
print("Please select the directory containing your Kilosort output files, metrics.csv, and channel brain region mapping...")
kilosort_folder = browse_directory()
if not kilosort_folder:
    print("No directory selected. Exiting.")
    sys.exit(1)
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
spike_times_seconds = None; sampling_rate_hz = 30000.0
if os.path.exists(spike_times_sec_file):
    try: spike_times_seconds = np.load(spike_times_sec_file).flatten(); print(f"Loaded spike times from: {os.path.basename(spike_times_sec_file)}");
    except Exception as e: print(f"Error loading '{os.path.basename(spike_times_sec_file)}': {e}"); sys.exit(1)
elif os.path.exists(spike_times_file):
    print(f"'{os.path.basename(spike_times_sec_file)}' not found. Looking for params.py and '{os.path.basename(spike_times_file)}'.")
    params_file = os.path.join(kilosort_folder, 'params.py')
    if os.path.exists(params_file):
        try:
            with open(params_file, 'r') as f: params_content = f.read(); local_namespace = {}; exec(params_content, {}, local_namespace); sampling_rate_hz = local_namespace.get('sample_rate', sampling_rate_hz); print(f" Found sampling rate in params.py: {sampling_rate_hz} Hz");
        except Exception as e: print(f" Warning: Could not read sampling rate from params.py ({e}). Using default: {sampling_rate_hz} Hz")
    try:
        spike_times = np.load(spike_times_file).flatten()
        if sampling_rate_hz <= 0: print("Error: Invalid sampling rate (<=0) obtained."); sys.exit(1)
        spike_times_seconds = spike_times / sampling_rate_hz
        print(f"Loaded spike times from: {os.path.basename(spike_times_file)} and converted to seconds using SR={sampling_rate_hz} Hz.")
    except Exception as e: print(f"Error loading '{os.path.basename(spike_times_file)}': {e}"); sys.exit(1)
else: print(f"Error: Neither spike time file found."); sys.exit(1)
if not os.path.exists(spike_clusters_file): print(f"Error: Spike clusters file '{os.path.basename(spike_clusters_file)}' not found."); sys.exit(1)
try: spike_clusters = np.load(spike_clusters_file).flatten(); print(f"Loaded spike clusters from: {os.path.basename(spike_clusters_file)}");
except Exception as e: print(f"Error loading spike clusters file '{os.path.basename(spike_clusters_file)}': {e}"); sys.exit(1)
if len(spike_times_seconds) != len(spike_clusters): print(f"Error: Mismatch spike times ({len(spike_times_seconds)}) /clusters ({len(spike_clusters)}) length."); sys.exit(1)
if not os.path.exists(channel_pos_file): print(f"Warning: Channel positions file '{os.path.basename(channel_pos_file)}' not found."); channel_pos = None
else:
    try: channel_pos = np.load(channel_pos_file); print(f"Loaded channel positions from: {os.path.basename(channel_pos_file)}");
    except Exception as e: print(f"Error loading channel positions file '{os.path.basename(channel_pos_file)}': {e}"); channel_pos = None
if not os.path.exists(metrics_file): print(f"Error: Metrics file '{os.path.basename(metrics_file)}' not found."); sys.exit(1)
try:
    metrics_df = pd.read_csv(metrics_file); id_col = None
    if 'cluster_id' in metrics_df.columns: id_col = 'cluster_id'
    elif 'id' in metrics_df.columns: id_col = 'id'; print("Warning: Using 'id' column from metrics.csv.")
    else: raise ValueError("Neither 'cluster_id' nor 'id' found in metrics.")
    metrics_df.rename(columns={id_col: 'cluster_id'}, inplace=True); metrics_df['cluster_id'] = pd.to_numeric(metrics_df['cluster_id'], errors='coerce'); metrics_df.dropna(subset=['cluster_id'], inplace=True); metrics_df['cluster_id'] = metrics_df['cluster_id'].astype(int); print(f"Loaded metrics from: {os.path.basename(metrics_file)}");
except Exception as e: print(f"Error loading or processing metrics file '{os.path.basename(metrics_file)}': {e}"); sys.exit(1)
brain_regions_df = None
if not os.path.exists(brain_regions_file): print(f"Warning: Brain regions file '{brain_regions_file_name}' not found.")
else:
    try:
        print(f"Checking indexing in {brain_regions_file_name}..."); temp_df = pd.read_csv(brain_regions_file)
        if 'global_channel_index' in temp_df.columns and not temp_df.empty:
            if temp_df['global_channel_index'].iloc[0] != 0 and pd.notna(temp_df['global_channel_index'].iloc[0]): print(f" Adjusting '{brain_regions_file_name}' to 0-based index..."); temp_df['global_channel_index'] -= 1; print(f" --> SAVING MODIFIED DATA back to '{brain_regions_file_name}'..."); temp_df.to_csv(brain_regions_file, index=False); print(f"     '{brain_regions_file_name}' updated."); brain_regions_df = pd.read_csv(brain_regions_file, index_col='global_channel_index')
            else: print(f" '{brain_regions_file_name}' seems already 0-based or starts with NaN/0."); brain_regions_df = pd.read_csv(brain_regions_file, index_col='global_channel_index')
        else: print(f" Warning: 'global_channel_index' not found/empty in '{brain_regions_file_name}'."); brain_regions_df = None
        if brain_regions_df is not None: print(f"Successfully loaded and processed brain region mapping.")
    except Exception as e: print(f"Error processing brain regions file '{brain_regions_file}': {e}"); brain_regions_df = None

# --- Main Processing Loop ---
print(f"\nCalculating recording duration...")
good_clusters_final = []
unique_clusters_all = np.unique(spike_clusters)
print(f"Found {len(unique_clusters_all)} unique clusters in '{os.path.basename(spike_clusters_file)}'. Processing (fitting ACGs)...")
min_time = np.nanmin(spike_times_seconds); max_time = np.nanmax(spike_times_seconds)
if np.isnan(min_time) or np.isnan(max_time): print("Warning: Cannot determine recording duration. Setting duration to 0."); recording_duration = 0.0
else: recording_duration = max_time - min_time
if recording_duration <= 0: print(f"Warning: Recording duration non-positive ({recording_duration:.2f}s).")

fit_failures = 0
skipped_fits = 0
for i, cluster_id in enumerate(unique_clusters_all):
    cluster_id = int(cluster_id)
    if (i + 1) % 50 == 0: print(f" Processing cluster {i+1}/{len(unique_clusters_all)} (ID: {cluster_id})...")

    cluster_mask = (spike_clusters == cluster_id)
    cluster_spike_times_seconds = spike_times_seconds[cluster_mask]
    n_spikes = len(cluster_spike_times_seconds)
    if n_spikes == 0: continue

    firing_rate = (n_spikes / recording_duration) if recording_duration > 0 else np.inf
    percent_short_isi = np.nan
    if n_spikes >= 2: isi = np.diff(np.sort(cluster_spike_times_seconds)) * 1000; len_isi = len(isi); percent_short_isi = (np.sum(isi < 2) / len_isi * 100) if len_isi > 0 else 0.0
    elif n_spikes == 1: percent_short_isi = 0.0

    metrics_row = metrics_df[metrics_df['cluster_id'] == cluster_id]
    if not metrics_row.empty:
        metrics_data = metrics_row.iloc[0]
        if 'peak_channel' not in metrics_data.index or pd.isna(metrics_data['peak_channel']): continue
        try: peak_channel_index = int(metrics_data['peak_channel'])
        except ValueError: continue

        # Get quality metrics and convert 
        contam_rate = metrics_data.get('contam_rate', None); l_ratio = metrics_data.get('l_ratio', None); isi_viol_rate = metrics_data.get('isi_viol', None)
        try: contam_rate = float(contam_rate) if pd.notna(contam_rate) else None;
        except (ValueError, TypeError): contam_rate = None
        try: l_ratio = float(l_ratio) if pd.notna(l_ratio) else None;
        except (ValueError, TypeError): l_ratio = None
        try: isi_viol_rate = float(isi_viol_rate) if pd.notna(isi_viol_rate) else None;
        except (ValueError, TypeError): isi_viol_rate = None

        # Check quality criteria
        is_good = (firing_rate > MIN_FIRING_RATE_HZ and contam_rate is not None and contam_rate < MAX_CONTAM_RATE and l_ratio is not None and l_ratio < MAX_L_RATIO and isi_viol_rate is not None and isi_viol_rate < MAX_ISI_VIOL_RATE)

        if is_good:
            burst_index = calculate_burst_index(cluster_spike_times_seconds, window_ms=500)
            trough_to_peak_ms = metrics_data.get('duration', None)
            try: trough_to_peak_ms = float(trough_to_peak_ms) if pd.notna(trough_to_peak_ms) else None
            except (ValueError, TypeError): trough_to_peak_ms = None

            # --- ACG Calculation and Fitting (Triple Exponential) ---
            tau_rise_parameter = np.nan # Initialize
            acg_fit_params = None
            popt = None
            if n_spikes >= MIN_SPIKES_FOR_ACG_FIT:
                acg_counts, bin_centers = calculate_acg(cluster_spike_times_seconds, ACG_BIN_SIZE_MS, ACG_WINDOW_MS)
                if acg_counts is not None and bin_centers is not None:
                     popt, pcov = fit_acg(acg_counts, bin_centers) # Use triple exp fit
                     if popt is not None:
                         tau_rise_parameter = popt[1] # Extract tau_rise (parameter 'b', index 1)
                         acg_fit_params = popt # Store all params
                     else: fit_failures += 1
                else: skipped_fits += 1 # Treat inability to calc ACG as skipped fit
            else: skipped_fits +=1

            # --- Classify Cell Type (using tau_rise) ---
            cell_type = classify_cell_type_full(trough_to_peak_ms, tau_rise_parameter)

            # --- Create dictionary ---
            good_cluster_info = {
                'cluster_id': int(cluster_id), 'cell_type': cell_type, 'firing_rate_hz': round(firing_rate, 3),
                'isi_violation_percent_manual': round(percent_short_isi, 3) if pd.notna(percent_short_isi) else None,
                'isi_viol_rate_allensdk': isi_viol_rate, 'contamination_rate': contam_rate, 'l_ratio': l_ratio,
                'num_spikes': n_spikes, 'spike_times_seconds': cluster_spike_times_seconds.tolist(), # NPY Only
                'peak_channel_index_0based': peak_channel_index, 'burst_index': burst_index if pd.notna(burst_index) else None,
                'trough_to_peak_ms': trough_to_peak_ms if pd.notna(trough_to_peak_ms) else None,                
                'acg_tau_rise': tau_rise_parameter if pd.notna(tau_rise_parameter) else None,
                'acg_tau_decay': acg_fit_params[0] if acg_fit_params is not None else None,
                'acg_tau_burst': acg_fit_params[6] if acg_fit_params is not None else None,
                'acg_refrac': acg_fit_params[5] if acg_fit_params is not None else None,
                'acg_fit_rsquare': None, # R-square calculation not implemented here
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
print(f"ACG fitting (Triple Exp): Skipped for {skipped_fits} units, Failed for {fit_failures} units.")
n_good = len(good_clusters_final)
print(f"Found {n_good} good clusters meeting quality criteria.")

if good_clusters_final:
    good_clusters_df = pd.DataFrame(good_clusters_final)

    # Cell Type Summary
    if 'cell_type' in good_clusters_df.columns:
         type_counts = good_clusters_df['cell_type'].value_counts()
         print("Cell Type Summary (using CellExplorer Triple Exp ACG fit):")
         for ctype, count in type_counts.items(): print(f" - {ctype}: {count} units")
         print(f"   (WS Interneuron requires TTP>{NS_PYWS_THRESHOLD_MS}ms & fitted tau_rise>{WS_PY_TAU_RISE_THRESHOLD}ms)")
         print(f"   (Pyramidal if TTP>{NS_PYWS_THRESHOLD_MS}ms but tau_rise condition not met/fit failed)")
    else: print("Warning: 'cell_type' column missing.")
    
    output_npy_file_name = f'good_clusters_processed_{folder_basename}_CellExplorerACG.npy'
    output_npy_path = os.path.join(kilosort_folder, output_npy_file_name)
    try: np.save(output_npy_path, np.array(good_clusters_final, dtype=object), allow_pickle=True); print(f"\nFull data saved to: {output_npy_path}");
    except Exception as e: print(f"Error saving NPY file: {e}")
   
    try:
        good_clusters_df_no_spikes = good_clusters_df.drop(columns=['spike_times_seconds']) if 'spike_times_seconds' in good_clusters_df.columns else good_clusters_df.copy()
        cols_order = [ # Updated/Reordered Columns for CSV
            'cluster_id', 'cell_type', 'acronym', 'brain_region_name', 'firing_rate_hz',
            'trough_to_peak_ms', 'burst_index',
            'acg_tau_rise', 'acg_tau_decay', 'acg_tau_burst', 'acg_refrac', 'acg_fit_rsquare',
            'isi_violation_percent_manual', 'isi_viol_rate_allensdk', 'contamination_rate', 'l_ratio', 'num_spikes',
            'peak_channel_index_0based', 'shank_index', 'distance_from_entry_um', 'channel_depth_um_fallback'
        ]
        cols_order_present = [col for col in cols_order if col in good_clusters_df_no_spikes.columns]
        missing_cols = [col for col in good_clusters_df_no_spikes.columns if col not in cols_order_present]
        final_cols_order = cols_order_present + missing_cols
        good_clusters_df_no_spikes = good_clusters_df_no_spikes[final_cols_order]
        output_csv_path = os.path.join(kilosort_folder, f'good_clusters_processed_{folder_basename}_CellExplorerACG.csv')
        good_clusters_df_no_spikes.to_csv(output_csv_path, index=False, float_format='%.4f')
        print(f"Metrics summary saved to: {output_csv_path}")
    except Exception as e: print(f"Error saving CSV file: {e}") # Catch and print error, but continue

    # Generate Plots
    generate_classification_plots(good_clusters_df, kilosort_folder, folder_basename)

else:
    print("\nNo 'good' clusters identified, skipping saving and plotting.")

print("\nScript finished.")