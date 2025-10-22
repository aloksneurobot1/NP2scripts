# -*- coding: utf-8 -*-
"""
Created on Sat Aug 23 10:00:00 2025

This is the final, comprehensive script for spike gain analysis.
It implements TWO parallel analysis methods:
    1. Deconvolution (Stark Lab, 2022)
    2. Poisson-based statistics (Stark & Abeles, 2009)

Generates separate plots/CSVs for each method, including advanced plots
for distance, asymmetry, bursting, convergence, and spatial distributions.

Updated to use user-specific CSV headers:
global_channel_index, shank_index, xcoord_on_shank_um, ycoord_on_shank_um, acronym, etc.
"""

import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog, Toplevel, Label, Button, Checkbutton, Frame, BooleanVar
from pathlib import Path
import warnings
from itertools import product
import multiprocessing as mp
import os
from tqdm import tqdm
from scipy.stats import poisson, zscore, linregress
from scipy.optimize import curve_fit
from scipy.signal.windows import gaussian # Corrected import
import numba
from numba import jit
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
MIN_SPIKES_FOR_ANALYSIS = 10
MIN_SPIKES_FOR_CONDITIONAL_ANALYSIS = 30
RUN_CONDITIONAL_ANALYSIS = True
N_SHUFFLES_FOR_MODEL_FIT = 1000

# --- Conditional Analysis Parameters ---
N_ISI_BINS = 40
ISI_RANGE_MS = (0.5, 2000)

# --- Advanced Plot Parameters ---
DISTANCE_BINS_UM = [0, 50, 100, 150, 200, 300, 500, np.inf] # Bins for plot I

# =============================================================================
# --- CORE HELPER FUNCTIONS (Shared) ---
# =============================================================================
# ... (find_lags_fast, calculate_cch, calculate_burst_index, calculate_asymmetry_index remain the same) ...
@jit(nopython=True)
def find_lags_fast(ref_spikes, target_spikes_sorted, window_size):
    all_lags = []
    for ref_time in ref_spikes:
        start_time, end_time = ref_time - window_size, ref_time + window_size
        start_idx = np.searchsorted(target_spikes_sorted, start_time, side='left')
        end_idx = np.searchsorted(target_spikes_sorted, end_time, side='right')
        if end_idx > start_idx:
            nearby_spikes = target_spikes_sorted[start_idx:end_idx]
            diffs = nearby_spikes - ref_time
            for d in diffs: all_lags.append(d)
    return np.array(all_lags)

def calculate_cch(ref_spikes, target_spikes, window_size, bin_size):
    is_acg = np.array_equal(ref_spikes, target_spikes)
    target_spikes_sorted = np.sort(target_spikes)
    lags = find_lags_fast(ref_spikes, target_spikes_sorted, window_size)
    if is_acg: lags = lags[lags != 0]
    num_bins = int(2 * window_size / bin_size)
    bins = np.linspace(-window_size, window_size, num_bins + 1)
    counts, _ = np.histogram(lags, bins=bins)
    return counts, bins

@jit(nopython=True)
def calculate_burst_index(spikes, burst_isi_ms=6.0):
    """Calculates a simple burst index."""
    if len(spikes) < 2:
        return 0.0
    isis_ms = np.diff(spikes) * 1000
    burst_spikes = np.sum(isis_ms < burst_isi_ms)
    return burst_spikes / (len(spikes) -1) if len(spikes) > 1 else 0.0 # Normalize by number of ISIs

def calculate_asymmetry_index(cch, causal_bins, anticausal_bins):
    """Calculates (C-A)/(C+A) for peak counts."""
    # Ensure indices are within bounds
    valid_causal = causal_bins[(causal_bins >= 0) & (causal_bins < len(cch))]
    valid_anticausal = anticausal_bins[(anticausal_bins >= 0) & (anticausal_bins < len(cch))]

    causal_peak = np.max(cch[valid_causal]) if valid_causal.size > 0 else 0
    anticausal_peak = np.max(cch[valid_anticausal]) if valid_anticausal.size > 0 else 0

    if causal_peak + anticausal_peak == 0:
        return 0.0
    # Add small epsilon to avoid division by zero if both peaks are zero
    return (causal_peak - anticausal_peak) / (causal_peak + anticausal_peak + 1e-9)

def subsample_by_presynaptic_isi(pre_spikes, isi_bins):
    """
    Subsamples presynaptic spikes based on the preceding ISI.

    Args:
        pre_spikes (np.array): Spike times of the presynaptic neuron.
        isi_bins (np.array): Edges of the ISI bins (in seconds).

    Returns:
        dict: Keys are bin centers (seconds), values are arrays of spike times
              that occurred *after* an ISI falling into that bin.
    """
    if len(pre_spikes) < 2: return {}
    isis = np.diff(pre_spikes) # ISIs in seconds
    # Indices of the spikes *following* each ISI
    spike_indices_following_isi = np.arange(1, len(pre_spikes)) 
    
    binned_spikes = {}
    for i in range(len(isi_bins) - 1):
        lower, upper = isi_bins[i], isi_bins[i+1];
        # Geometric mean for log bins
        bin_center = np.sqrt(lower * upper) 
        # Find ISIs within the current bin
        indices_in_bin = np.where((isis >= lower) & (isis < upper))[0]
        
        if len(indices_in_bin) > 0:
            # Get the spikes that *followed* these ISIs
            spikes_after_isi = pre_spikes[spike_indices_following_isi[indices_in_bin]]
            binned_spikes[bin_center] = spikes_after_isi
            
    return binned_spikes

# =============================================================================
# --- METHOD 1: DECONVOLUTION (Stark Lab 2022) ---
# =============================================================================
def hollowed_median_filter(data, window_size):
    if window_size % 2 == 0: raise ValueError("window_size must be odd.")
    n = len(data);
    predictor = np.zeros(n);
    half_window = window_size // 2
    padded_data = np.pad(data, (half_window, half_window), 'reflect')
    for i in range(n):
        window = padded_data[i : i + window_size]
        hollowed_window = np.delete(window, half_window)
        predictor[i] = np.median(hollowed_window)
    return predictor

def deconvolve_and_analyze(
    pre_spikes, post_spikes, window_size=0.1, bin_size=0.001,
    reg_factor=0.001, predictor_window=11,
    roi_ms=(0.5, 5), alpha=0.001
):
    cch_raw, bins = calculate_cch(pre_spikes, post_spikes, window_size, bin_size)
    acg_pre, _ = calculate_cch(pre_spikes, pre_spikes, window_size, bin_size)
    acg_post, _ = calculate_cch(post_spikes, post_spikes, window_size, bin_size)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    cch_fft = np.fft.fft(cch_raw)
    acg_fft = np.fft.fft(acg_pre)
    reg = np.mean(np.abs(acg_fft)) * reg_factor
    if reg == 0: reg = 1e-9
    kernel_fft = cch_fft / (acg_fft + reg)
    deconvolved_cch = np.fft.ifftshift(np.real(np.fft.ifft(kernel_fft)))
    deconvolved_cch[deconvolved_cch < 0] = 0
    predictor = hollowed_median_filter(deconvolved_cch, window_size=predictor_window)
    bin_centers_ms = bin_centers * 1000
    roi_indices = np.where((bin_centers_ms >= roi_ms[0]) & (bin_centers_ms <= roi_ms[1]))[0]
    significant_peaks = []
    if roi_indices.size > 0:
        n_bonf = max(len(roi_indices), 1) # Avoid division by zero
        corrected_alpha = alpha / n_bonf
        pred_roi = predictor[roi_indices]
        # Ensure lambda_upper is not zero for Poisson ppf calculation
        lambda_upper = max(np.max(pred_roi), 1e-9)
        lambda_lower = max(np.min(pred_roi), 1e-9)
        threshold_upper, threshold_lower = poisson.ppf(1 - corrected_alpha, lambda_upper), poisson.ppf(corrected_alpha, lambda_lower)
        dc_cch_roi = deconvolved_cch[roi_indices]
        excitatory_indices, inhibitory_indices = roi_indices[np.where(dc_cch_roi > threshold_upper)[0]], roi_indices[np.where(dc_cch_roi < threshold_lower)[0]]
        all_significant_indices = np.concatenate([excitatory_indices, inhibitory_indices])
        if all_significant_indices.size > 0:
            deviations = deconvolved_cch[all_significant_indices] - predictor[all_significant_indices]
            peak_index = all_significant_indices[np.argmax(np.abs(deviations))]
            spike_gain = deconvolved_cch[peak_index]
            z_score = (spike_gain - predictor[peak_index]) / (np.std(predictor) + 1e-9)
            significant_peaks.append({"lag_ms": bin_centers_ms[peak_index], "gain": spike_gain, "z_score": z_score})
    return {"cch_raw": cch_raw, "acg_pre": acg_pre, "acg_post": acg_post, "deconvolved_cch": deconvolved_cch, "predictor": predictor, "significant_peaks": significant_peaks, "bins": bins}

def plot_deconvolution_results(results, epoch_idx, pre_id, post_id, output_dir):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
    fig.suptitle(f'Deconvolution Analysis: Epoch {epoch_idx}\n{pre_id} (Pre) -> {post_id} (Post)', fontsize=14)
    bin_centers_ms = (results['bins'][:-1] + results['bins'][1:]) / 2 * 1000
    bin_width_ms = (results['bins'][1] - results['bins'][0]) * 1000

    ax1.bar(bin_centers_ms, results['cch_raw'], width=bin_width_ms, color='gray', label='Raw CCH (1ms bins)')
    ax1.set_ylabel('Spike Pair Count');
    ax1.set_title('Raw Cross-Correlogram (CCH)');
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax_inset_pre = ax1.inset_axes([0.02, 0.65, 0.25, 0.3]);
    ax_inset_pre.bar(bin_centers_ms, results['acg_pre'], width=bin_width_ms, color='darkred');
    ax_inset_pre.set_title('Pre-ACH', fontsize=9);
    ax_inset_pre.set_xlim(-15, 15);
    ax_inset_pre.tick_params(axis='both', which='major', labelsize=8)
    ax_inset_post = ax1.inset_axes([0.73, 0.65, 0.25, 0.3]);
    ax_inset_post.bar(bin_centers_ms, results['acg_post'], width=bin_width_ms, color='darkblue');
    ax_inset_post.set_title('Post-ACH', fontsize=9);
    ax_inset_post.set_xlim(-15, 15);
    ax_inset_post.tick_params(axis='both', which='major', labelsize=8)
    ax2.bar(bin_centers_ms, results['deconvolved_cch'], width=bin_width_ms, color='lightgray', label='Deconvolved CCH (dcCCH)')
    ax2.plot(bin_centers_ms, results['predictor'], color='red', linestyle='--', label='Predictor (Hollowed Median)')
    for peak in results["significant_peaks"]:
        ax2.plot(peak['lag_ms'], peak['gain'], 'bo', markersize=8, label=f"Significant Peak (Gain: {peak['gain']:.2f})")
    ax2.set_ylabel('Deconvolved Counts');
    ax2.set_title('Deconvolved CCH & Statistical Test');
    ax2.set_xlabel('Time Lag (ms)');
    ax2.legend(loc='upper right', fontsize=9);
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.set_xlim(-25, 25)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    filename = f"Epoch{epoch_idx}_Deconv_{pre_id}_vs_{post_id}.tiff";
    try:
        fig.savefig(output_dir / filename, dpi=150, format='tiff');
    except Exception as e:
        print(f"  Error saving plot {filename}: {e}")
    plt.close(fig)

# =============================================================================
# --- METHOD 2: POISSON (Stark & Abeles 2009) ---
# =============================================================================
def calculate_poisson_p_value(n, lambda_val):
    if lambda_val <= 0:
        return 1.0 if n > 0 else 0.0
    if n < 0: n = 0
    # Handle potential large lambda values leading to NaN in pmf
    if np.isinf(np.exp(-lambda_val)) or lambda_val > 700: # exp(-700) is near float limit
        # Use Normal approximation for large lambda
        from scipy.stats import norm
        sigma = np.sqrt(lambda_val)
        # P(N >= n) ~ 1 - Phi((n - 0.5 - lambda) / sigma) with continuity correction
        # Check for zero sigma to avoid division error
        if sigma <= 0: return 1.0 if n > lambda_val else 0.0
        return norm.sf(n - 0.5, loc=lambda_val, scale=sigma)

    # Standard calculation for smaller lambda
    try:
        p_val = 1.0 - poisson.cdf(n - 1, lambda_val) - (0.5 * poisson.pmf(n, lambda_val))
    except ValueError: # Handle potential issues if lambda_val is extreme but not caught above
         p_val = 1.0 # Default to non-significant if calculation fails
    return max(0, p_val)

def partially_hollow_gaussian(sigma_bins, hollow_fraction):
    kernel_width = int(np.ceil(sigma_bins * 8))
    if kernel_width % 2 == 0: kernel_width += 1
    kernel = gaussian(kernel_width, std=sigma_bins)
    hollow_kernel = kernel - (hollow_fraction * kernel)
    center_idx = kernel_width // 2
    hollow_kernel[center_idx] = 0
    # Avoid division by zero if sum is zero
    hollow_sum = np.sum(hollow_kernel)
    if hollow_sum > 0:
        hollow_kernel /= hollow_sum
    return hollow_kernel

def analyze_pair_poisson_method(
    pre_spikes, post_spikes, window_size=0.1, bin_size=0.0004, # 0.4ms bins
    kernel_sigma_ms=10.0, hollow_fraction=0.6
):
    cch_raw, bins = calculate_cch(pre_spikes, post_spikes, window_size, bin_size)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    n_pre_spikes = len(pre_spikes)

    sigma_bins = (kernel_sigma_ms / 1000) / bin_size
    kernel = partially_hollow_gaussian(sigma_bins, hollow_fraction)
    lambda_slow = np.convolve(cch_raw, kernel, 'same')
    # Ensure baseline is non-negative
    lambda_slow[lambda_slow < 0] = 0

    center_bin_idx = len(bin_centers) // 2
    causal_start_idx = center_bin_idx + int(np.ceil(0.0008 / bin_size))
    causal_end_idx = center_bin_idx + int(np.ceil(0.0028 / bin_size)) + 1
    causal_indices = np.arange(causal_start_idx, causal_end_idx)
    anticausal_start_idx = center_bin_idx - int(np.ceil(0.0020 / bin_size))
    anticausal_end_idx = center_bin_idx
    anticausal_indices = np.arange(anticausal_start_idx, anticausal_end_idx)

    p_fast, p_causal, stp, peak_lag_ms = 1.0, 1.0, 0.0, np.nan

    # Clip indices to be within array bounds BEFORE using them
    causal_indices = causal_indices[(causal_indices >= 0) & (causal_indices < len(cch_raw))]
    anticausal_indices = anticausal_indices[(anticausal_indices >= 0) & (anticausal_indices < len(cch_raw))]

    if causal_indices.size > 0: # Ensure causal window exists within bounds
        cch_causal_roi = cch_raw[causal_indices]
        peak_bin_local_idx = np.argmax(cch_causal_roi)
        peak_bin_global_idx = causal_indices[peak_bin_local_idx]
        peak_lag_ms = bin_centers[peak_bin_global_idx] * 1000 # <-- NEW: Save peak lag

        n_observed = cch_raw[peak_bin_global_idx]
        lambda_slow_at_peak = lambda_slow[peak_bin_global_idx]
        p_fast = calculate_poisson_p_value(n_observed, lambda_slow_at_peak)

        if anticausal_indices.size > 0: # Ensure anticausal window exists
            cch_anticausal_roi = cch_raw[anticausal_indices]
            lambda_anticausal = np.max(cch_anticausal_roi)
            p_causal = calculate_poisson_p_value(n_observed, lambda_anticausal)
        else:
             p_causal = 1.0 # Cannot calculate if anticausal window is empty/out of bounds

        excess_spikes = np.sum(cch_causal_roi - lambda_slow[causal_indices])
        if n_pre_spikes > 0:
            stp = excess_spikes / n_pre_spikes

    is_synapse = (p_fast < 0.01) and (p_causal < 0.01)
    is_nonconnected = (p_fast > 0.1) and (p_causal > 0.1)

    return {
        "cch_raw": cch_raw, "lambda_slow": lambda_slow, "bins": bins,
        "p_fast": p_fast, "p_causal": p_causal, "stp": stp,
        "is_synapse": is_synapse, "is_nonconnected": is_nonconnected,
        "causal_indices": causal_indices, "anticausal_indices": anticausal_indices,
        "peak_lag_ms": peak_lag_ms
    }

def plot_poisson_results(results, epoch_idx, pre_id, post_id, output_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    bin_centers_ms = (results['bins'][:-1] + results['bins'][1:]) / 2 * 1000
    bin_width_ms = (results['bins'][1] - results['bins'][0]) * 1000

    ax.bar(bin_centers_ms, results['cch_raw'], width=bin_width_ms, color='gray', label=f'Raw CCH ({bin_width_ms:.1f}ms bins)')
    ax.plot(bin_centers_ms, results['lambda_slow'], color='red', linestyle='--', label=r'$\lambda_{slow}$ (10ms hollow Gaussian)')

    # Ensure indices are valid before slicing
    valid_causal_indices = results['causal_indices'][(results['causal_indices'] >= 0) & (results['causal_indices'] < len(bin_centers_ms))]
    valid_anticausal_indices = results['anticausal_indices'][(results['anticausal_indices'] >= 0) & (results['anticausal_indices'] < len(bin_centers_ms))]

    causal_bins_ms = bin_centers_ms[valid_causal_indices] if valid_causal_indices.size > 0 else np.array([])
    anticausal_bins_ms = bin_centers_ms[valid_anticausal_indices] if valid_anticausal_indices.size > 0 else np.array([])

    if causal_bins_ms.size > 0:
        ax.axvspan(causal_bins_ms[0] - bin_width_ms/2, causal_bins_ms[-1] + bin_width_ms/2,
                   color='green', alpha=0.2, label='Causal Window (0.8-2.8ms)')
    if anticausal_bins_ms.size > 0:
        ax.axvspan(anticausal_bins_ms[0] - bin_width_ms/2, anticausal_bins_ms[-1] + bin_width_ms/2,
                   color='blue', alpha=0.2, label='Anticausal Window (-2.0-0ms)')

    status = "Unclassified"
    if results['is_synapse']: status = "Synapse"
    elif results['is_nonconnected']: status = "Non-connected"

    stats_text = (f"Classification: {status}\n"
                  f"$P_{{fast}}$: {results['p_fast']:.2e}\n"
                  f"$P_{{causal}}$: {results['p_causal']:.2e}\n"
                  f"STP: {results['stp']:.4f}")
    ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_title(f'Poisson Analysis: Epoch {epoch_idx}\n{pre_id} (Pre) -> {post_id} (Post)', fontsize=14)
    ax.set_ylabel('Spike Pair Count');
    ax.set_xlabel('Time Lag (ms)');
    ax.legend(loc='upper right', fontsize=9);
    ax.grid(True, linestyle=':', alpha=0.6); ax.set_xlim(-25, 25)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    filename = f"Epoch{epoch_idx}_Poisson_{pre_id}_vs_{post_id}.tiff";
    try:
        fig.savefig(output_dir / filename, dpi=150, format='tiff')
    except Exception as e:
        print(f"  Error saving plot {filename}: {e}")
    plt.close(fig)

# =============================================================================
# --- PLASTICITY MODELING (Shared) ---
# =============================================================================
@jit(nopython=True)
def model_full(isi, A, U, tau_syn, tau_fac, tau_dep):
    term_fac = (1 + (U - 1) * np.exp(-isi / tau_fac));
    term_syn = (1 - np.exp(-isi / tau_syn));
    term_dep = (1 - np.exp(-isi / tau_dep))
    return A * term_fac * term_syn * term_dep

@jit(nopython=True)
def model_depression(isi, A, tau_syn, tau_dep):
    term_syn = (1 - np.exp(-isi / tau_syn));
    term_dep = (1 - np.exp(-isi / tau_dep))
    return A * term_syn * term_dep

@jit(nopython=True)
def model_facilitation(isi, A, U, tau_syn, tau_fac):
    term_fac = (1 + (U - 1) * np.exp(-isi / tau_fac));
    term_syn = (1 - np.exp(-isi / tau_syn))
    return A * term_fac * term_syn

def calculate_adjusted_r2(y_true, y_pred, n_params):
    n_samples = len(y_true);
    ss_res = np.sum((y_true - y_pred)**2);
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    if ss_tot < 1e-9: return 0 # Avoid division by zero if all y_true are the same
    r2 = 1 - (ss_res / ss_tot)
    # Ensure denominator is positive
    denom = n_samples - n_params - 1
    if denom <= 0: return r2 # Cannot calculate adjusted R2 if not enough samples
    return 1 - (1 - r2) * (n_samples - 1) / denom

def fit_plasticity_models(isi_data, gain_data):
    if len(isi_data) < 6: # Need at least n_params + 2 data points for adj R2
        return {"model": "insufficient_data", "tau": np.nan, "adj_r2": -np.inf, "popt": None}

    sort_idx = np.argsort(isi_data);
    isi, gain = isi_data[sort_idx], gain_data[sort_idx]
    bounds_full = ([0, 0, 0, 0, 0], [np.inf, np.inf, 0.003, 10.0, 10.0]);
    bounds_dep = ([0, 0, 0], [np.inf, 0.003, 10.0]);
    bounds_fac = ([0, 0, 0, 0], [np.inf, np.inf, 0.003, 10.0])
    try: popt_full, _ = curve_fit(model_full, isi, gain, bounds=bounds_full, maxfev=5000)
    except (RuntimeError, ValueError): popt_full = None
    try: popt_dep, _ = curve_fit(model_depression, isi, gain, bounds=bounds_dep, maxfev=5000)
    except (RuntimeError, ValueError): popt_dep = None
    try: popt_fac, _ = curve_fit(model_facilitation, isi, gain, bounds=bounds_fac, maxfev=5000)
    except (RuntimeError, ValueError): popt_fac = None

    adj_r2_full = calculate_adjusted_r2(gain, model_full(isi, *popt_full), 5) if popt_full is not None else -np.inf
    adj_r2_dep = calculate_adjusted_r2(gain, model_depression(isi, *popt_dep), 3) if popt_dep is not None else -np.inf
    adj_r2_fac = calculate_adjusted_r2(gain, model_facilitation(isi, *popt_fac), 4) if popt_fac is not None else -np.inf

    shuffled_r2s = {'full': [], 'dep': [], 'fac': [], 'diff_dep': [], 'diff_fac': []}
    gain_shuffled = gain.copy()

    n_successful_shuffles = 0
    for _ in range(N_SHUFFLES_FOR_MODEL_FIT):
        np.random.shuffle(gain_shuffled)
        try: popt_s_full, _ = curve_fit(model_full, isi, gain_shuffled, bounds=bounds_full, maxfev=2000)
        except (RuntimeError, ValueError): popt_s_full = None
        try: popt_s_dep, _ = curve_fit(model_depression, isi, gain_shuffled, bounds=bounds_dep, maxfev=2000)
        except (RuntimeError, ValueError): popt_s_dep = None
        try: popt_s_fac, _ = curve_fit(model_facilitation, isi, gain_shuffled, bounds=bounds_fac, maxfev=2000)
        except (RuntimeError, ValueError): popt_s_fac = None

        # Only append if fits were successful
        r2_s_full = calculate_adjusted_r2(gain_shuffled, model_full(isi, *popt_s_full), 5) if popt_s_full is not None else -np.inf
        r2_s_dep = calculate_adjusted_r2(gain_shuffled, model_depression(isi, *popt_s_dep), 3) if popt_s_dep is not None else -np.inf
        r2_s_fac = calculate_adjusted_r2(gain_shuffled, model_facilitation(isi, *popt_s_fac), 4) if popt_s_fac is not None else -np.inf

        # Only consider shuffle if at least one model could be fit
        if popt_s_full is not None or popt_s_dep is not None or popt_s_fac is not None:
             shuffled_r2s['full'].append(r2_s_full);
             shuffled_r2s['dep'].append(r2_s_dep);
             shuffled_r2s['fac'].append(r2_s_fac);
             # Calculate diffs only if r2_s_full is valid
             if r2_s_full > -np.inf:
                 shuffled_r2s['diff_dep'].append(r2_s_full - r2_s_dep);
                 shuffled_r2s['diff_fac'].append(r2_s_full - r2_s_fac);
             else: # If full model failed, diffs are not meaningful in the same way
                 shuffled_r2s['diff_dep'].append(-np.inf);
                 shuffled_r2s['diff_fac'].append(-np.inf);

             n_successful_shuffles += 1

    if n_successful_shuffles < 0.5 * N_SHUFFLES_FOR_MODEL_FIT: # Check if enough shuffles worked
        print("Warning: Less than 50% of shuffles resulted in successful fits. Thresholds might be unreliable.")
        return {"model": "shuffle_failed", "tau": np.nan, "adj_r2": -np.inf, "popt": None}

    thresh_full = np.percentile(shuffled_r2s['full'], 95);
    thresh_dep = np.percentile(shuffled_r2s['dep'], 95);
    thresh_fac = np.percentile(shuffled_r2s['fac'], 95);
    # Ensure diff lists are not empty before calculating percentile
    thresh_diff_dep = np.percentile(shuffled_r2s['diff_dep'], 95) if shuffled_r2s['diff_dep'] else np.inf;
    thresh_diff_fac = np.percentile(shuffled_r2s['diff_fac'], 95) if shuffled_r2s['diff_fac'] else np.inf;

    best_fit = {"model": "none", "tau": np.nan, "adj_r2": -np.inf, "popt": None}

    # Check validity before comparison
    adj_r2_full_valid = adj_r2_full > -np.inf
    adj_r2_dep_valid = adj_r2_dep > -np.inf
    adj_r2_fac_valid = adj_r2_fac > -np.inf

    is_dep = adj_r2_dep_valid and (adj_r2_dep > thresh_dep) and (not adj_r2_full_valid or (adj_r2_full - adj_r2_dep) < thresh_diff_dep)
    is_fac = adj_r2_fac_valid and (adj_r2_fac > thresh_fac) and (not adj_r2_full_valid or (adj_r2_full - adj_r2_fac) < thresh_diff_fac)
    is_full = adj_r2_full_valid and (adj_r2_full > thresh_full)

    if is_dep: best_fit = {"model": "depression", "tau": popt_dep[2] * 1000, "adj_r2": adj_r2_dep, "popt": popt_dep}
    # Check if fac is better than current best (which might be dep)
    if is_fac and adj_r2_fac > best_fit['adj_r2']: best_fit = {"model": "facilitation", "tau": popt_fac[3] * 1000, "adj_r2": adj_r2_fac, "popt": popt_fac}
    # Check if full is better AND neither dep nor fac were chosen (or full explains significantly more)
    if is_full and adj_r2_full > best_fit['adj_r2'] and \
       (best_fit['model'] == 'none' or \
        (best_fit['model'] == 'depression' and (adj_r2_full - adj_r2_dep) >= thresh_diff_dep) or \
        (best_fit['model'] == 'facilitation' and (adj_r2_full - adj_r2_fac) >= thresh_diff_fac)):

        # Determine tau based on which component (fac or dep) is stronger/slower
        tau_f_ms = popt_full[3] * 1000
        tau_d_ms = popt_full[4] * 1000
        # Assign the dominant tau (often the slower one, but depends on context)
        # Here, let's just pick the larger one as a simple heuristic
        tau = max(tau_f_ms, tau_d_ms)
        best_fit = {"model": "full", "tau": tau, "adj_r2": adj_r2_full, "popt": popt_full}

    return best_fit

def plot_conditional_results(x_data, y_data, title, xlabel, output_path, metric_label, fit_results=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x_data, y_data, 'o', color='black', label='Data')
    if fit_results and fit_results['model'] != 'none' and fit_results['popt'] is not None:
        try:
            # Ensure min x > 0 for logspace and handle potential empty x_data
            min_x = x_data.min() if len(x_data) > 0 else 1e-1
            max_x = x_data.max() if len(x_data) > 0 else 1.0
            # Check if min_x and max_x are valid before creating logspace
            if min_x > 0 and max_x >= min_x:
                fine_x = np.logspace(np.log10(min_x), np.log10(max_x), 200)

                model_func = {'full': model_full, 'depression': model_depression, 'facilitation': model_facilitation}[fit_results['model']]
                y_fit = model_func(fine_x / 1000, *fit_results['popt'])
                ax.plot(fine_x, y_fit, color='blue', linewidth=2, label=f"Best Fit: {fit_results['model'].capitalize()} (τ = {fit_results['tau']:.1f} ms)")
                ax.legend(loc='best')
            else:
                 print(f"Warning: Invalid range for logspace ({min_x} - {max_x}) in {output_path.name}. Skipping fit line.")
                 if 'legend' not in locals() or ax.get_legend() is None: ax.legend(loc='best')

        except Exception as e:
            print(f"Warning: Could not plot fit line for {output_path.name}: {e}")
            if 'legend' not in locals() or ax.get_legend() is None: ax.legend(loc='best')

    ax.set_title(title, fontsize=14); ax.set_xlabel(xlabel, fontsize=12);
    ax.set_ylabel(metric_label, fontsize=12) # Use dynamic label
    ax.grid(True, linestyle=':', alpha=0.7)
    # Check for valid data before setting log scale
    if ('ISI' in xlabel or 'Interval' in xlabel) and len(x_data) > 0 and x_data.min() > 0:
        ax.set_xscale('log')
    plt.tight_layout();
    try:
        fig.savefig(output_path, dpi=150, format='tiff');
    except Exception as e:
        print(f"  Error saving plot {output_path.name}: {e}")
    plt.close(fig)

# =============================================================================
# --- POPULATION & ANATOMY PLOTTING (Parameterized) ---
# =============================================================================
def generate_population_plots(base_output_dir, method_name, metric_col_name):
    print(f"\n--- Generating Population Summary Plots for {method_name.upper()} ---")

    # Scan for method-specific CSVs
    csv_files = list(base_output_dir.glob(f"*/*/*_Cond_PreISI_{method_name}.csv"))
    if not csv_files: print("No conditional plasticity data found to generate population plots.");
    return
    all_plasticity_data = [pd.read_csv(f) for f in csv_files if os.path.getsize(f) > 0]
    if not all_plasticity_data: print("Conditional data was empty. Skipping population plots.");
    return
    pop_df = pd.concat(all_plasticity_data, ignore_index=True)

    # Filter out failed fits/shuffles
    pop_df = pop_df[~pop_df['best_model'].isin(['none', 'shuffle_failed', 'insufficient_data'])]
    if pop_df.empty: print("No successful plasticity fits found. Skipping population plots."); return

    # --- Plot 1: Overall Prevalence (Original) ---
    plt.figure(figsize=(6, 5));
    ax = sns.countplot(x='best_model', data=pop_df.drop_duplicates(subset=['recording', 'comparison', 'epoch']))
    ax.set_title(f'Prevalence of Plasticity Types ({method_name.capitalize()})'); ax.set_xlabel('Best Fit Model');
    ax.set_ylabel('Number of Connections')
    plt.tight_layout(); plt.savefig(base_output_dir / f"population_plasticity_prevalence_{method_name}.tiff", dpi=150);
    plt.close()

    # --- Plot 2: Interneuron Plasticity Percentage (New) ---
    print("Generating interneuron-specific plasticity prevalence plot...")
    interneuron_types_list = ['NS Interneuron', 'WS Interneuron']
    in_df = pop_df[pop_df['post_cell_type'].isin(interneuron_types_list)].copy()

    if not in_df.empty:
        in_df_unique = in_df.drop_duplicates(subset=['recording', 'comparison', 'epoch'])
        # Handle cases where a cell type might have only one model type
        pct_df = in_df_unique.groupby('post_cell_type')['best_model'].value_counts(normalize=True).unstack(fill_value=0).stack()
        pct_df = pct_df.mul(100).rename('percentage').reset_index()

        pct_df_filtered = pct_df[pct_df['best_model'].isin(['full', 'depression', 'facilitation'])]

        if not pct_df_filtered.empty:
            plt.figure(figsize=(8, 6))
            ax = sns.barplot(data=pct_df_filtered, x='post_cell_type', y='percentage', hue='best_model')
            ax.set_title(f'Plasticity Model Prevalence in Postsynaptic Interneurons ({method_name.capitalize()})')
            ax.set_xlabel('Postsynaptic Cell Type')
            ax.set_ylabel('Percent of Connections (%)')
            plt.legend(title='Best Fit Model')
            plt.tight_layout()
            plot_path = base_output_dir / f"population_plasticity_prevalence_interneurons_{method_name}.tiff"
            plt.savefig(plot_path, dpi=150)
            plt.close()
            print(f"  - Saved interneuron prevalence plot to {plot_path.name}")
        else:
            print("  - Skipping interneuron prevalence plot: No connections with 'full', 'depression', or 'facilitation' models found.")
    else:
        print("  - Skipping interneuron prevalence plot: No postsynaptic interneurons found in plasticity data.")

    # --- Plot 3: Tau Distributions by Cell Type (Original) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True);
    fig.suptitle(f'Time Constant Distributions by Postsynaptic Cell Type ({method_name.capitalize()})')
    dep_df_all = pop_df[pop_df['best_model'] == 'depression'].dropna(subset=['tau', 'post_cell_type']);
    fac_df_all = pop_df[pop_df['best_model'] == 'facilitation'].dropna(subset=['tau', 'post_cell_type'])
    if not dep_df_all.empty: sns.histplot(data=dep_df_all, x='tau', hue='post_cell_type', ax=ax1, log_scale=True, element='step', fill=False, bins=20);
    ax1.set_title('Depression (τ_d)'); ax1.set_xlabel('τ (ms)')
    if not fac_df_all.empty: sns.histplot(data=fac_df_all, x='tau', hue='post_cell_type', ax=ax2, log_scale=True, element='step', fill=False, bins=20); ax2.set_title('Facilitation (τ_f)');
    ax2.set_xlabel('τ (ms)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.savefig(base_output_dir / f"population_tau_distributions_by_type_{method_name}.tiff", dpi=150);
    plt.close()

    # --- New CSV and Plots: Interneuron-Specific Tau Distributions (New) ---
    print("Generating interneuron-specific tau distribution plots and data file...")
    in_type_map = {'NS Interneuron': 'FS (Narrow Spike)', 'WS Interneuron': 'non-FS (Wide Spike)'}
    interneuron_types = list(in_type_map.keys())
    in_tau_df = pop_df[pop_df['best_model'].isin(['depression', 'facilitation']) & pop_df['post_cell_type'].isin(interneuron_types)].copy()

    if not in_tau_df.empty:
        in_tau_df['in_type'] = in_tau_df['post_cell_type'].map(in_type_map)
        csv_path = base_output_dir / f"population_tau_data_interneurons_{method_name}.csv"
        cols_to_save = ['recording', 'comparison', 'epoch', 'post_cell_type', 'in_type', 'best_model', 'tau', 'adj_r2']
        in_tau_df[cols_to_save].to_csv(csv_path, index=False, float_format='%.4f')
        print(f"  - Saved interneuron tau data to {csv_path.name}")

        dep_tau_df = in_tau_df[in_tau_df['best_model'] == 'depression']
        fac_tau_df = in_tau_df[in_tau_df['best_model'] == 'facilitation']

        if not dep_tau_df.empty:
            plt.figure(figsize=(8, 6))
            ax = sns.kdeplot(data=dep_tau_df, x='tau', hue='in_type', style='in_type', log_scale=True, fill=True, alpha=0.1, linewidth=2.5)
            ax.set_title(f'Distribution of Depression Time Constants ($\tau_d$) ({method_name.capitalize()})')
            ax.set_xlabel('Depression τ (ms)'); ax.set_ylabel('Density')
            plt.tight_layout()
            plot_path = base_output_dir / f"population_tau_distribution_depression_{method_name}.tiff"
            plt.savefig(plot_path, dpi=150); plt.close()
            print(f"  - Saved depression tau distribution plot to {plot_path.name}")
        else: print("  - Skipping depression tau plot: No data found.")

        if not fac_tau_df.empty:
            plt.figure(figsize=(8, 6))
            ax = sns.kdeplot(data=fac_tau_df, x='tau', hue='in_type', style='in_type', log_scale=True, fill=True, alpha=0.1, linewidth=2.5)
            ax.set_title(f'Distribution of Facilitation Time Constants ($\tau_f$) ({method_name.capitalize()})')
            ax.set_xlabel('Facilitation τ (ms)'); ax.set_ylabel('Density')
            plt.tight_layout()
            plot_path = base_output_dir / f"population_tau_distribution_facilitation_{method_name}.tiff"
            plt.savefig(plot_path, dpi=150); plt.close()
            print(f"  - Saved facilitation tau distribution plot to {plot_path.name}")
        else: print("  - Skipping facilitation tau plot: No data found.")
    else: print("  - Skipping interneuron tau plots: No depressing/facilitating interneurons found.")

    # --- Final Plot: Heatmaps ---
    print("Generating PSTH-style heatmaps for depressing synapses...")
    dep_df = pop_df[pop_df['best_model'] == 'depression'].copy()
    if not dep_df.empty:
        # Use metric_col_name which should be 'metric_val' from conditional analysis
        metric_label = "Z-Scored Spike Gain" if metric_col_name == 'gain' else "Z-Scored Spike Transmission Prob." # Or adjust based on method_name if needed
        for post_type, group_df in dep_df.groupby('post_cell_type'):
            print(f"  - Processing connections onto {post_type}...")
            if len(group_df.drop_duplicates(subset=['recording', 'comparison', 'epoch'])) < 2: print(f"    ... skipping, not enough depressing connections found for {post_type}."); continue
            group_df['unique_id'] = group_df['recording'] + "_" + group_df['comparison'] + "_" + group_df['epoch'].astype(str)
            sorted_ids = group_df.drop_duplicates(subset=['unique_id']).sort_values('tau')['unique_id']
            isi_bins = np.logspace(np.log10(ISI_RANGE_MS[0]), np.log10(ISI_RANGE_MS[1]), 15)
            group_df['isi_bin'] = pd.cut(group_df['isi_ms'], bins=isi_bins)

            if metric_col_name not in group_df.columns:
                print(f"    ... skipping heatmap for {post_type}, '{metric_col_name}' column not found in data.")
                continue

            # Need numeric categories for pivot_table when observed=False
            group_df['isi_bin_cat'] = group_df['isi_bin'].cat.codes
            try:
                # Handle potential duplicate index/column issues during pivot
                pivot_data = group_df.drop_duplicates(subset=['unique_id', 'isi_bin_cat'])
                gain_matrix = pivot_data.pivot_table(index='unique_id', columns='isi_bin_cat', values=metric_col_name, observed=False)
            except Exception as e:
                print(f"    ... skipping heatmap for {post_type}, error during pivot: {e}")
                continue

            gain_matrix = gain_matrix.reindex(sorted_ids).dropna(axis=0, how='any')
            if gain_matrix.empty: continue

            # Calculate z-score, handle potential NaNs or constant rows
            zscore_matrix = gain_matrix.apply(lambda x: zscore(x, nan_policy='omit'), axis=1, result_type='broadcast')
            zscore_matrix.fillna(0, inplace=True) # Fill NaNs resulting from constant rows

            data_path = base_output_dir / f"heatmap_data_depression_onto_{post_type}_{method_name}.csv";
            gain_matrix.to_csv(data_path) # Save raw values
            print(f"    ... saved data matrix to {data_path.name}")
            plt.figure(figsize=(10, 8));
            ax = sns.heatmap(zscore_matrix, cmap='vlag', cbar_kws={'label': metric_label})
            ax.set_title(f'Depressing Synapses onto {post_type} Neurons ({method_name.capitalize()})\n(Sorted by τ_depression)');
            ax.set_xlabel('Presynaptic ISI (ms)'); ax.set_ylabel('Individual Synaptic Connections')
            ax.set_yticks([])
            # Correct tick labeling based on original categories if possible
            try:
                # Use the categories from the original 'isi_bin' if available and valid
                if pd.api.types.is_categorical_dtype(group_df['isi_bin']) and len(group_df['isi_bin'].cat.categories) == zscore_matrix.shape[1]:
                    tick_labels = [f"{b.left:.1f}" for b in group_df['isi_bin'].cat.categories];
                    ax.set_xticks(np.arange(len(tick_labels)) + 0.5); ax.set_xticklabels(tick_labels, rotation=45, ha='right')
                else:
                     ax.set_xticks([]) # Hide ticks if labels fail
                     print("    ... Warning: Could not generate x-tick labels for heatmap (category mismatch or invalid).")
            except Exception: # Fallback if categories aren't easily accessible
                 ax.set_xticks([]) # Hide ticks if labels fail
                 print("    ... Warning: Could not generate x-tick labels for heatmap.")

            plt.tight_layout();
            plot_path = base_output_dir / f"heatmap_plot_depression_onto_{post_type}_{method_name}.tiff"; plt.savefig(plot_path, dpi=150); plt.close()
            print(f"    ... saved heatmap plot to {plot_path.name}")
    print(f"Population plots for {method_name.upper()} generated successfully.")

def run_conditional_analysis(pre_neuron, post_neuron, epoch, output_dir, method_name):
    start_time, end_time = epoch['start_time_sec'], epoch['end_time_sec']
    pre_epoch_spikes = pre_neuron['spike_times_sec'][(pre_neuron['spike_times_sec'] >= start_time) & (pre_neuron['spike_times_sec'] < end_time)]
    post_epoch_spikes = post_neuron['spike_times_sec'][(post_neuron['spike_times_sec'] >= start_time) & (post_neuron['spike_times_sec'] < end_time)]
    pre_id = f"{pre_neuron['acronym']}_{pre_neuron['cell_type'].replace(' ', '')}_ID{pre_neuron['cluster_id']}";
    post_id = f"{post_neuron['acronym']}_{post_neuron['cell_type'].replace(' ', '')}_ID{post_neuron['cluster_id']}"
    
    isi_bins_sec = np.logspace(np.log10(ISI_RANGE_MS[0] / 1000), np.log10(ISI_RANGE_MS[1] / 1000), N_ISI_BINS + 1)
    binned_pre_spikes = subsample_by_presynaptic_isi(pre_epoch_spikes, isi_bins_sec)
    results_std = []
    
    for bin_center_sec, subsample in binned_pre_spikes.items():
        if len(subsample) >= MIN_SPIKES_FOR_CONDITIONAL_ANALYSIS:
            if method_name == 'deconvolution':
                res = deconvolve_and_analyze(subsample, post_epoch_spikes)
                if res['significant_peaks']: 
                    results_std.append({'isi_ms': bin_center_sec * 1000, 'metric_val': res['significant_peaks'][0]['gain']})
            elif method_name == 'poisson':
                res = analyze_pair_poisson_method(subsample, post_epoch_spikes)
                if res['is_synapse']: 
                    results_std.append({'isi_ms': bin_center_sec * 1000, 'metric_val': res['stp']})
    
    if len(results_std) > 5:
        df = pd.DataFrame(results_std)
        fit_results = fit_plasticity_models(df['isi_ms'].values / 1000, df['metric_val'].values)
        
        metric_name = 'gain' if method_name == 'deconvolution' else 'stp'
        metric_label = 'Spike Gain' if method_name == 'deconvolution' else 'Spike Transmission Prob.'
        
        df['best_model'] = fit_results['model'];
        df['tau'] = fit_results['tau']; df['adj_r2'] = fit_results['adj_r2']
        df['recording'] = output_dir.parts[-3]; # Go up 3 levels (comp/rec/method)
        df['comparison'] = output_dir.name;
        df['epoch'] = epoch['epoch_index']
        df['pre_cell_type'] = pre_neuron['cell_type'];
        df['post_cell_type'] = post_neuron['cell_type']
        df[metric_name] = df['metric_val'] # Add a clearly named column
        
        plot_path = output_dir / f"Epoch{epoch['epoch_index']}_Cond_PreISI_{method_name}.tiff"
        csv_path = output_dir / f"Epoch{epoch['epoch_index']}_Cond_PreISI_{method_name}.csv"
        
        plot_conditional_results(df['isi_ms'], df['metric_val'], f"Short-Term Plasticity ({method_name.capitalize()})\n{pre_id} -> {post_id}", 
                                 "Presynaptic ISI (ms)", plot_path, metric_label, fit_results=fit_results)
        df.to_csv(csv_path, index=False)

def generate_anatomical_plot(summary_df, all_units_df, all_locations_df, output_dir, method_name, metric_col, metric_label):
    print(f"Generating combined anatomical connectivity plot for {method_name.upper()} method...")

    # Define NEW column names based on user CSV
    loc_channel_col = 'global_channel_index' # From user CSV
    loc_x_col = 'xcoord_on_shank_um'       # From user CSV
    loc_y_col = 'ycoord_on_shank_um'       # From user CSV
    loc_shank_col = 'shank_index'          # From user CSV

    # Ensure necessary columns are present in unit/location data
    required_loc_cols = [loc_channel_col, 'recording', loc_x_col, loc_y_col, loc_shank_col]
    if not all(col in all_locations_df.columns for col in required_loc_cols):
        print(f"FATAL: Missing required columns in location data ({required_loc_cols}). Skipping anatomical plot.")
        return

    unit_channel_col = 'peak_channel_index_0based'
    if unit_channel_col not in all_units_df.columns: unit_channel_col = 'channels'
    if unit_channel_col not in all_units_df.columns:
        print(f"FATAL: Could not find a valid peak channel column in units data. Skipping anatomical plot.")
        return

    # Ensure data types are correct for merging
    try:
        all_units_df[unit_channel_col] = all_units_df[unit_channel_col].astype(int)
        all_locations_df[loc_channel_col] = all_locations_df[loc_channel_col].astype(int)
    except Exception as e:
        print(f"Error converting channel columns to int: {e}. Skipping anatomical plot.")
        return

    # Merge units with locations using NEW column names
    # Select only needed columns from locations to avoid duplicate merge keys if channel appears multiple times
    loc_subset = all_locations_df[required_loc_cols].drop_duplicates(subset=['recording', loc_channel_col])
    units_with_locations = pd.merge(all_units_df, loc_subset,
                                    left_on=['recording', unit_channel_col],
                                    right_on=['recording', loc_channel_col]) # Use global_channel_index here
    if units_with_locations.empty:
         print("Warning: Merging units and locations resulted in empty dataframe. Skipping plot.")
         return

    pre_merged = pd.merge(summary_df, units_with_locations, left_on=['recording', 'pre_cluster_id'], right_on=['recording', 'cluster_id'])
    # Need to handle potential duplicate columns after first merge if units_with_locations has 'cluster_id'
    pre_merged = pre_merged.drop(columns=['cluster_id'], errors='ignore')
    full_merged = pd.merge(pre_merged, units_with_locations, left_on=['recording', 'post_cluster_id'], right_on=['recording', 'cluster_id'], suffixes=('_pre', '_post'))
    full_merged = full_merged.drop(columns=['cluster_id'], errors='ignore')


    if full_merged.empty: print("No connections to plot after merging with location data."); return

    fig, ax = plt.subplots(figsize=(10, 15))
    type_markers = {'Pyramidal': 'o', 'NS Interneuron': '^', 'WS Interneuron': 's', 'Unclassified': 'x'}

    # Use NEW coordinate column names with _pre/_post suffixes
    all_plotted_neurons = pd.concat([
        full_merged[['pre_cluster_id', 'recording', 'pre_cell_type', f'{loc_x_col}_pre', f'{loc_y_col}_pre']].rename(columns={f'{loc_x_col}_pre': loc_x_col, f'{loc_y_col}_pre': loc_y_col, 'pre_cluster_id': 'cluster_id', 'pre_cell_type': 'cell_type'}),
        full_merged[['post_cluster_id', 'recording', 'post_cell_type', f'{loc_x_col}_post', f'{loc_y_col}_post']].rename(columns={f'{loc_x_col}_post': loc_x_col, f'{loc_y_col}_post': loc_y_col, 'post_cluster_id': 'cluster_id', 'post_cell_type': 'cell_type'})
    ]).drop_duplicates(subset=['cluster_id', 'recording'])

    for _, neuron in all_plotted_neurons.iterrows():
        marker = type_markers.get(neuron['cell_type'], '*')
        # Use NEW coordinate column names
        ax.scatter(neuron[loc_x_col], neuron[loc_y_col], marker=marker, s=50, ec='k', fc='w', zorder=10)

    if metric_col not in full_merged.columns:
        print(f"FATAL: Metric column '{metric_col}' not in summary. Skipping anatomical plot.")
        return

    # Handle cases where min/max might be the same or NaN
    min_val = full_merged[metric_col].min()
    max_val = full_merged[metric_col].max()
    if pd.isna(min_val) or pd.isna(max_val) or abs(min_val - max_val) < 1e-9:
        min_val = 0
        max_val = 1 if max_val > 0 else 0.1 # Default range
        print(f"Warning: Metric range invalid ({full_merged[metric_col].min()}-{full_merged[metric_col].max()}). Using default range {min_val}-{max_val} for color bar.")


    cmap = plt.get_cmap('viridis')
    norm = plt.Normalize(vmin=min_val, vmax=max_val)

    for _, conn in full_merged.iterrows():
        color = cmap(norm(conn[metric_col]))
        # Use NEW coordinate column names with suffixes
        ax.arrow(conn[f'{loc_x_col}_pre'], conn[f'{loc_y_col}_pre'],
                 conn[f'{loc_x_col}_post'] - conn[f'{loc_x_col}_pre'],
                 conn[f'{loc_y_col}_post'] - conn[f'{loc_y_col}_pre'],
                 color=color, alpha=0.6, width=1, head_width=10, length_includes_head=True, zorder=5) # Arrows below points

    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker=m, color='w', label=l, markerfacecolor='gray', markeredgecolor='k', markersize=10) for l, m in type_markers.items()]
    ax.legend(handles=legend_elements, title="Cell Types")

    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.6)
    cbar.set_label(metric_label)

    ax.set_title(f'Master Anatomical Connectivity Plot ({method_name.capitalize()})');
    ax.set_xlabel(f'{loc_x_col} (µm)'); ax.set_ylabel(f'{loc_y_col} (µm)') # Updated Labels
    ax.invert_yaxis(); ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plot_path = output_dir / f"master_anatomical_connectivity_{method_name}.tiff"
    plt.savefig(plot_path, dpi=300); plt.close()
    print(f"SUCCESS: Master anatomical plot for {method_name.upper()} saved to {plot_path.name}")

# =============================================================================
# --- NEW: ADVANCED PLOTS FUNCTION ---
# =============================================================================
def generate_advanced_plots(master_summary_df, master_units_with_locations_df, master_locations_df_orig, base_output_dir, method_name):
    print(f"\n--- Generating Advanced Plots for {method_name.upper()} ---")

    # Create a new subdirectory within the method's folder
    method_dir = base_output_dir / ("Deconvolution_Method" if method_name == 'deconvolution' else "Poisson_Method")
    output_dir = method_dir / "Advanced_Plots"
    output_dir.mkdir(exist_ok=True)

    # Define cell types and metric
    pyr_type = 'Pyramidal'
    int_types = ['NS Interneuron', 'WS Interneuron']
    metric_col = 'spike_gain' if method_name == 'deconvolution' else 'spike_transmission_prob'
    metric_label = 'Spike Gain' if method_name == 'deconvolution' else 'Spike Transmission Prob.'
    # Use internal consistent names after renaming in main
    loc_x_col = 'xcoord_on_shank_um'
    loc_y_col = 'ycoord_on_shank_um'
    loc_shank_col = 'shank_id'

    # --- Preprocessing ---
    df = master_summary_df.copy()
    if df.empty:
        print("  - Skipping advanced plots: No significant connections found.")
        return

    # Create pair type column
    def get_pair_type(row):
        is_pre_pyr = row['pre_cell_type'] == pyr_type
        is_pre_int = row['pre_cell_type'] in int_types
        is_post_pyr = row['post_cell_type'] == pyr_type
        is_post_int = row['post_cell_type'] in int_types
        if is_pre_pyr and is_post_int: return 'PYR-INT'
        if is_pre_int and is_post_pyr: return 'INT-PYR'
        if is_pre_pyr and is_post_pyr: return 'PYR-PYR'
        if is_pre_int and is_post_int: return 'INT-INT'
        return 'Other'
    df['pair_type'] = df.apply(get_pair_type, axis=1)

    # Calculate Y-distance using internal consistent names with suffixes added during merge
    y_col_pre = f'{loc_y_col}_pre'
    y_col_post = f'{loc_y_col}_post'

    # Check if coordinate columns are present (they might be missing if merge failed earlier)
    if y_col_pre in df.columns and y_col_post in df.columns:
        # Calculate distance only if both columns are numeric
         if pd.api.types.is_numeric_dtype(df[y_col_pre]) and pd.api.types.is_numeric_dtype(df[y_col_post]):
              df['y_distance_um'] = np.abs(df[y_col_pre] - df[y_col_post])
         else:
              print("  - Warning: Y-coordinate columns are not numeric in summary df. Skipping Y-distance calculation.")
              df['y_distance_um'] = np.nan
    else:
         print("  - Warning: Y-coordinate columns missing from summary df. Skipping Y-distance calculation.")
         df['y_distance_um'] = np.nan


    # --- Save Enhanced CSV ---
    csv_path = output_dir / f"advanced_metrics_all_pairs_{method_name}.csv"
    # Select columns to save, handle potential missing ones gracefully
    cols_to_save = [col for col in ['recording', 'comparison', 'epoch_index',
                    'pre_cluster_id', 'pre_region', 'pre_cell_type', 'pre_shank_id',
                    'post_cluster_id', 'post_region', 'post_cell_type', 'post_shank_id',
                    'peak_lag_ms', metric_col, 'radial_distance_um',
                    'pre_burst_index', 'asymmetry_index', 'pair_type', 'y_distance_um',
                    'p_fast', 'p_causal', 'z_score'] if col in df.columns]
    try:
        df[cols_to_save].to_csv(csv_path, index=False, float_format='%.4f')
        print(f"  - Saved advanced metrics CSV to {csv_path.name}")
    except KeyError as e:
        print(f"  - Error saving advanced CSV: Missing column {e}. Skipping save.")
    except Exception as e:
        print(f"  - Error saving advanced CSV: {e}. Skipping save.")


    # --- Plot G: Distribution of Strength (PYR-INT) ---
    pyr_int_df = df[df['pair_type'] == 'PYR-INT'].copy()
    if not pyr_int_df.empty and metric_col in pyr_int_df.columns and not pyr_int_df[metric_col].isna().all():
        plt.figure(figsize=(8, 6))
        sns.histplot(pyr_int_df[metric_col].dropna(), kde=True, bins=30) # dropna added
        plt.title(f'Distribution of PYR-INT Connection Strength ({method_name.capitalize()})')
        plt.xlabel(metric_label)
        plt.ylabel('Count')
        plot_path = output_dir / f"strength_distribution_pyr_int_{method_name}.tiff"
        plt.savefig(plot_path, dpi=150); plt.close()
        print(f"  - Saved PYR-INT Strength Distribution plot to {plot_path.name}")
    else:
        print("  - Skipping Plot G: No PYR-INT pairs or metric column missing/empty.")

    # --- Plot H: Strength vs Convergence (PYR-INT, within-shank) ---
    print("  - Calculating convergence for Plot H...")
    convergence_data = []
    # Use internal consistent shank name ('shank_id') with suffixes
    shank_col_pre = f'{loc_shank_col}_pre'
    shank_col_post = f'{loc_shank_col}_post'

    if shank_col_pre not in df.columns or shank_col_post not in df.columns:
         print("  - Skipping Plot H: Shank IDs missing from summary dataframe.")
    else:
        # Get all units with locations and types (use the pre-merged df from main with renamed cols)
        all_units_with_loc = master_units_with_locations_df # Use the df already created

        # Filter for within-shank PYR-INT pairs from the currently processed df
        within_shank_pyr_int = df[
            (df['pair_type'] == 'PYR-INT') &
            (df[shank_col_pre].notna()) &
            (df[shank_col_post].notna()) &
            (df[shank_col_pre] == df[shank_col_post])
        ].copy() # Ensure shanks are not NaN

        if not within_shank_pyr_int.empty:
            convergence_cache = {} # Cache potential counts per recording/shank

            for idx, row in tqdm(within_shank_pyr_int.iterrows(), total=len(within_shank_pyr_int), desc="Calculating convergence"):
                rec = row['recording']
                post_int_id = row['post_cluster_id']
                # Use the shank ID available in the row (pre and post are the same here)
                shank = row[shank_col_post]

                cache_key = (rec, shank)
                if cache_key not in convergence_cache:
                    # Find all potential presynaptic PYR on the same shank in this recording from the master list
                    potential_pre_pyr = all_units_with_loc[
                        (all_units_with_loc['recording'] == rec) &
                        (all_units_with_loc[loc_shank_col] == shank) & # Use internal shank col name
                        (all_units_with_loc['cell_type'] == pyr_type)
                    ]['cluster_id'].unique()
                    convergence_cache[cache_key] = len(potential_pre_pyr)

                total_potential = convergence_cache[cache_key]

                if total_potential > 0:
                    # Find actual connected presynaptic PYR *targeting this specific INT* from the summary df
                    actual_connected_pyr = df[
                        (df['recording'] == rec) &
                        (df['post_cluster_id'] == post_int_id) &
                        (df['pre_cell_type'] == pyr_type) &
                        (df[shank_col_pre] == shank) # Use internal shank col name
                    ]['pre_cluster_id'].unique()

                    num_connected = len(actual_connected_pyr)
                    convergence = num_connected / total_potential
                    convergence_data.append({'id': idx, 'convergence': convergence})
                else:
                    convergence_data.append({'id': idx, 'convergence': 0.0}) # No potential PYR on shank

            if convergence_data:
                conv_df = pd.DataFrame(convergence_data).set_index('id')
                within_shank_pyr_int = within_shank_pyr_int.join(conv_df)

                # Plotting (ensure metric_col exists and is not all NaN)
                if metric_col in within_shank_pyr_int.columns and not within_shank_pyr_int[metric_col].isna().all():
                    plt.figure(figsize=(8, 6))
                    plot_data = within_shank_pyr_int.dropna(subset=['convergence', metric_col]) # Drop NaNs before plotting/regression
                    if not plot_data.empty:
                        sns.scatterplot(data=plot_data, x='convergence', y=metric_col, alpha=0.7)
                        # Optional: Add regression line if desired and enough points exist
                        if len(plot_data) > 2:
                            try:
                                reg = linregress(plot_data['convergence'], plot_data[metric_col])
                                x_vals = np.array([0, plot_data['convergence'].max()])
                                y_vals = reg.intercept + reg.slope * x_vals
                                plt.plot(x_vals, y_vals, color='red', linestyle='--', label=f'R={reg.rvalue:.2f}, p={reg.pvalue:.2e}')
                                plt.legend()
                            except ValueError as e:
                                print(f"  - Could not calculate regression for convergence plot: {e}")

                        plt.title(f'PYR-INT Strength vs. Convergence (Within Shank, {method_name.capitalize()})')
                        plt.xlabel('Convergence (Fraction of Shank PYR Connected)')
                        plt.ylabel(metric_label)
                        plt.xlim(left=-0.05) # Start x-axis near 0
                        plot_path = output_dir / f"strength_vs_convergence_{method_name}.tiff"
                        plt.savefig(plot_path, dpi=150); plt.close()
                        print(f"  - Saved Strength vs. Convergence plot to {plot_path.name}")
                    else:
                        print("  - Skipping Plot H: No valid data points after dropping NaNs for convergence/metric.")

                else:
                     print(f"  - Skipping Plot H: Metric column '{metric_col}' missing or all NaN.")
        else:
            print("  - Skipping Plot H: No within-shank PYR-INT pairs found.")


    # --- Plot I: Strength vs Y-Distance (PYR-INT) ---
    if 'y_distance_um' in pyr_int_df.columns and not pyr_int_df['y_distance_um'].isna().all():
        # Ensure metric col exists and is not all NaN
        if metric_col in pyr_int_df.columns and not pyr_int_df[metric_col].isna().all():
             # Create bins and filter out NaN distances before plotting
            pyr_int_df_dist = pyr_int_df.dropna(subset=['y_distance_um', metric_col]).copy() # Use copy
            if not pyr_int_df_dist.empty:
                # Ensure distance bins cover the data range
                max_dist = pyr_int_df_dist['y_distance_um'].max()
                current_bins = [b for b in DISTANCE_BINS_UM if b != np.inf]
                if max_dist >= current_bins[-1]:
                     current_bins.append(np.ceil(max_dist / 100) * 100) # Extend last bin if needed
                else:
                     current_bins.append(np.inf) # Use inf if max_dist is covered

                pyr_int_df_dist['y_distance_bin'] = pd.cut(pyr_int_df_dist['y_distance_um'], bins=current_bins)

                plt.figure(figsize=(10, 7))
                # Ensure boxplot has data to plot for each category and handle empty bins
                order = sorted([cat for cat in pyr_int_df_dist['y_distance_bin'].unique() if pd.notna(cat)], key=lambda x: x.left)
                sns.boxplot(data=pyr_int_df_dist, x='y_distance_bin', y=metric_col, showfliers=False, order=order) # Boxplot shows distribution
                plt.title(f'PYR-INT Strength vs. Septo-Temporal Distance ({method_name.capitalize()})')
                plt.xlabel('Inter-Somatic Distance along Y-axis (µm)')
                plt.ylabel(metric_label)
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plot_path = output_dir / f"strength_vs_y_distance_{method_name}.tiff"
                plt.savefig(plot_path, dpi=150); plt.close()
                print(f"  - Saved Strength vs. Y-Distance plot to {plot_path.name}")
            else:
                 print("  - Skipping Plot I: No valid PYR-INT pairs after removing NaN distances or metrics.")
        else:
            print(f"  - Skipping Plot I: Metric column '{metric_col}' missing or all NaN.")

    else:
        print("  - Skipping Plot I: Y-distance information missing or all NaN.")
# =============================================================================
# --- GUI FUNCTIONS (All) ---
# =============================================================================
def get_method_selection_gui():
    """Presents a GUI to select which analysis methods to run."""
    selected_methods = []
    window = tk.Tk()
    window.title("Select Analysis Methods")
    window.attributes("-topmost", True)

    Label(window, text="Select which analysis method(s) to run:", font=('Helvetica', 10, 'bold')).pack(pady=10)

    vars = {}
    frame = Frame(window)
    frame.pack(padx=20, pady=10)

    methods = [
        ("deconvolution", "Deconvolution (Stark 2022)"),
        ("poisson", "Poisson (Stark & Abeles 2009)")
    ]

    for key, name in methods:
        var = BooleanVar(value=True) # Default to selected
        Checkbutton(frame, text=name, variable=var).pack(anchor='w')
        vars[key] = var

    def on_submit():
        selected_methods.clear()
        for key, var in vars.items():
            if var.get():
                selected_methods.append(key)
        if not selected_methods:
            print("Warning: No methods selected.")
        else:
            window.quit()
            window.destroy()

    Button(window, text="Confirm Methods", command=on_submit, font=('Helvetica', 10, 'bold')).pack(pady=20)
    window.mainloop()
    return selected_methods

def get_epoch_selection_gui(available_epochs):
    """
    Presents a GUI to select which epochs to analyze.

    Args:
        available_epochs (list): A list of dicts, each representing an epoch
                                 (must have 'epoch_index').

    Returns:
        list: A filtered list containing only the selected epoch dicts.
    """
    selected_epochs = []
    window = tk.Tk()
    window.title("Select Epochs to Analyze")
    window.attributes("-topmost", True)

    Label(window, text="Select which epoch(s) to process:", font=('Helvetica', 10, 'bold')).pack(pady=10)

    vars = {}
    frame = Frame(window)
    frame.pack(padx=20, pady=10)

    # Sort epochs by index for display
    sorted_epochs = sorted(available_epochs, key=lambda e: e.get('epoch_index', 0))

    for epoch in sorted_epochs:
        epoch_idx = epoch.get('epoch_index', 'Unknown')
        epoch_label = f"Epoch {epoch_idx}"
        # Default to selecting only the first epoch if available
        default_select = (epoch_idx == 1)
        var = BooleanVar(value=default_select)
        Checkbutton(frame, text=epoch_label, variable=var).pack(anchor='w')
        # Use epoch index as the key to handle potential duplicate labels if needed
        vars[epoch_idx] = (var, epoch)

    def on_submit():
        selected_epochs.clear()
        for idx, (var, epoch) in vars.items():
            if var.get():
                selected_epochs.append(epoch)
        if not selected_epochs:
            print("Warning: No epochs selected.")
        else:
            window.quit()
            window.destroy()

    Button(window, text="Confirm Epoch Selections", command=on_submit, font=('Helvetica', 10, 'bold')).pack(pady=20)
    window.mainloop()
    return selected_epochs

def get_recording_selection_gui(found_recordings):
    selected_recordings = []
    window = tk.Tk()
    window.title("Select Recordings to Analyze")
    window.attributes("-topmost", True)

    Label(window, text="Select which recordings to process:", font=('Helvetica', 10, 'bold')).pack(pady=10)
    vars = {}
    frame = Frame(window); frame.pack(padx=20, pady=10)
    for rec in found_recordings:
        rec_name = rec['name']
        var = BooleanVar(value=True)
        Checkbutton(frame, text=rec_name, variable=var).pack(anchor='w')
        vars[rec_name] = var
    def on_submit():
        selected_recordings.clear() # Clear list before appending
        for rec in found_recordings:
            # Check if key exists before accessing
            if rec['name'] in vars and vars[rec['name']].get():
                selected_recordings.append(rec)
        window.quit(); window.destroy()
    Button(window, text="Confirm Selections", command=on_submit, font=('Helvetica', 10, 'bold')).pack(pady=20)
    window.mainloop()
    return selected_recordings

def select_directory(title):
    root = tk.Tk();
    root.withdraw(); root.attributes("-topmost", True)
    dir_path = filedialog.askdirectory(title=title)
    root.destroy()
    return Path(dir_path) if dir_path else None

def get_multi_selection_gui(regions, types):
    result = {}
    window = tk.Tk();
    window.title("Select Groups for Pairwise Comparison"); window.attributes("-topmost", True)
    region_frame = Frame(window, relief='sunken', borderwidth=1);
    region_frame.pack(pady=10, padx=10, fill='x')
    type_frame = Frame(window, relief='sunken', borderwidth=1);
    type_frame.pack(pady=10, padx=10, fill='x')
    Label(region_frame, text="Select Brain Regions", font=('Helvetica', 10, 'bold')).pack()
    Label(type_frame, text="Select Neuron Types", font=('Helvetica', 10, 'bold')).pack()

    region_vars = {region: BooleanVar(value=False) for region in regions}
    all_regions_var = BooleanVar()
    def toggle_all_regions():
        new_state = all_regions_var.get()
        [var.set(new_state) for var in region_vars.values()]
    Checkbutton(region_frame, text="-- SELECT ALL --", variable=all_regions_var, command=toggle_all_regions).pack(anchor='w')
    for region, var in region_vars.items(): Checkbutton(region_frame, text=region, variable=var).pack(anchor='w')

    type_vars = {ntype: BooleanVar(value=False) for ntype in types}
    all_types_var = BooleanVar()
    def toggle_all_types():
        new_state = all_types_var.get()
        [var.set(new_state) for var in type_vars.values()]
    Checkbutton(type_frame, text="-- SELECT ALL --", variable=all_types_var, command=toggle_all_types).pack(anchor='w')
    for ntype, var in type_vars.items(): Checkbutton(type_frame, text=ntype, variable=var).pack(anchor='w')

    def on_submit():
        result['regions'] = [r for r, v in region_vars.items() if v.get()];
        result['types'] = [t for t, v in type_vars.items() if v.get()]
        if not result['regions'] or not result['types']:
            print("Warning: Select at least one region and type.")
        else:
            window.quit();
            window.destroy()

    Button(window, text="Confirm Selections for All Recordings", command=on_submit, font=('Helvetica', 10, 'bold')).pack(pady=10)
    window.mainloop()
    return result
# =============================================================================
# --- WORKER & MAIN FUNCTIONS ---
# =============================================================================
def analyze_pair_worker(args):
    pre_neuron, post_neuron, epoch, comparison_name, method_name = args
    # Ensure neurons are different (check both cluster_id and recording)
    if pre_neuron.get('cluster_id') == post_neuron.get('cluster_id') and pre_neuron.get('recording') == post_neuron.get('recording'):
        return None

    start_time, end_time = epoch['start_time_sec'], epoch['end_time_sec']
    pre_spikes_all, post_spikes_all = pre_neuron['spike_times_sec'], post_neuron['spike_times_sec']
    pre_epoch_spikes = pre_spikes_all[(pre_spikes_all >= start_time) & (pre_spikes_all < end_time)]
    post_epoch_spikes = post_spikes_all[(post_spikes_all >= start_time) & (post_spikes_all < end_time)]

    # Calculate distance and burst index (Ensure coordinates and shank are available)
    # Use internal consistent column names
    loc_x_col = 'xcoord_on_shank_um'
    loc_y_col = 'ycoord_on_shank_um'
    loc_shank_col = 'shank_id'

    radial_distance = np.nan
    pre_shank_id = pre_neuron.get(loc_shank_col, np.nan)
    post_shank_id = post_neuron.get(loc_shank_col, np.nan)
    # Check if coordinate keys exist before calculating distance
    if loc_x_col in pre_neuron and loc_x_col in post_neuron and \
       loc_y_col in pre_neuron and loc_y_col in post_neuron:
        try:
            dx = pre_neuron[loc_x_col] - post_neuron[loc_x_col]
            dy = pre_neuron[loc_y_col] - post_neuron[loc_y_col]
            # Ensure coordinates are numeric before calculation
            if pd.notna(dx) and pd.notna(dy):
                 radial_distance = np.sqrt(dx**2 + dy**2)
        except TypeError: # Handle potential None or other non-numeric types
            pass

    burst_index = calculate_burst_index(pre_epoch_spikes)

    result_dict = {
        "comparison": comparison_name, "epoch_index": epoch['epoch_index'],
        "pre_neuron": pre_neuron, "post_neuron": post_neuron, # Pass full dicts
        "pre_spike_count": len(pre_epoch_spikes), "post_spike_count": len(post_epoch_spikes),
        "status": "init", "analysis_results": None, "epoch": epoch, "method": method_name,
        "radial_distance": radial_distance, "pre_burst_index": burst_index,
        "asymmetry_index": np.nan, # Initialize
        "pre_shank_id": pre_shank_id, "post_shank_id": post_shank_id # Pass shank info
    }

    if len(pre_epoch_spikes) < MIN_SPIKES_FOR_ANALYSIS or len(post_epoch_spikes) < MIN_SPIKES_FOR_ANALYSIS:
        result_dict["status"] = "insufficient_spikes";
        return result_dict

    try:
        if method_name == 'deconvolution':
            analysis_results = deconvolve_and_analyze(pre_epoch_spikes, post_epoch_spikes)
            status = "significant_peak_found" if analysis_results["significant_peaks"] else "no_significant_peak"
            # Calculate asymmetry for deconv
            bin_centers_ms = (analysis_results['bins'][:-1] + analysis_results['bins'][1:]) / 2 * 1000
            causal_bins = np.where((bin_centers_ms >= 0.5) & (bin_centers_ms <= 5.0))[0]
            anticausal_bins = np.where((bin_centers_ms >= -5.0) & (bin_centers_ms <= -0.5))[0]
            asymm_index = calculate_asymmetry_index(analysis_results['deconvolved_cch'], causal_bins, anticausal_bins)

        elif method_name == 'poisson':
            analysis_results = analyze_pair_poisson_method(pre_epoch_spikes, post_epoch_spikes)
            if analysis_results["is_synapse"]: status = "significant_synapse"
            elif analysis_results["is_nonconnected"]: status = "non_connected"
            else: status = "unclassified"
            # Calculate asymmetry for poisson
            asymm_index = calculate_asymmetry_index(analysis_results['cch_raw'], analysis_results['causal_indices'], analysis_results['anticausal_indices'])

        else:
            status = "unknown_method"
            analysis_results = None
            asymm_index = np.nan

        result_dict["analysis_results"] = analysis_results
        result_dict["status"] = status
        result_dict["asymmetry_index"] = asymm_index

    except Exception as e:
        # Include cluster IDs in error message if available
        pre_id = pre_neuron.get('cluster_id', 'NA')
        post_id = post_neuron.get('cluster_id', 'NA')
        print(f"ERROR in worker for {pre_id}->{post_id} ({method_name}): {e}")
        # Optionally, include traceback for detailed debugging
        # import traceback
        # traceback.print_exc()
        result_dict["status"] = f"error: {e}"

    return result_dict

def main():
    print("--- Starting Dual Spike Gain Analysis Pipeline ---")
    data_dir = select_directory("Select the Folder Containing All Your Recordings")
    if not data_dir: print("No directory selected. Exiting."); return

    # --- 1. Find ALL available recordings ---
    all_found_recordings = []
    print("Scanning for recordings...")
    for unit_path in data_dir.glob('good_clusters_processed_*_CellExplorerACG.npy'):
        try:
            name_part = unit_path.name.replace('good_clusters_processed_', '').split('_imec0')[0]
            # Use NEW CSV naming convention
            anatomy_path = data_dir / f"channel_brain_regions_{name_part}.csv"
            timestamp_path = data_dir / f"{name_part}_tcat.nidq_timestamps.npy"
            if anatomy_path.exists() and timestamp_path.exists():
                all_found_recordings.append({"name": name_part, "units": unit_path, "timestamps": timestamp_path, "anatomy": anatomy_path})
                print(f"  - Found match: {name_part}")
            else:
                 print(f"  - Skipping {name_part}: Missing anatomy ({anatomy_path.name}) or timestamp ({timestamp_path.name}) file.")
        except IndexError: print(f"  - Warning: Could not parse name from: {unit_path.name}")
    if not all_found_recordings: print("No valid recording pairs found. Exiting."); return


    # --- 2. GUI Selection Cascade ---
    print("\nOpening recording selection GUI...")
    selected_recordings = get_recording_selection_gui(all_found_recordings)
    if not selected_recordings: print("No recordings selected for analysis. Exiting."); return
    print(f"Selected {len(selected_recordings)} recordings for processing.")

    print("\nAggregating all brain regions and cell types for GUI...")
    all_regions, all_types = set(), set()
    all_units_list = [] # Collect all units data here
    all_locations_list = [] # Collect all locations data here

    # --- Load ALL units/locations first ---
    print("Loading all unit and location data...")
    # Define columns expected directly from the CSV file IN THE CORRECT ORDER
    required_csv_cols = ['global_channel_index', 'shank_index', 'xcoord_on_shank_um', 'ycoord_on_shank_um', 'acronym'] # Adjusted to match user CSV
    # Define internal mapping names
    loc_channel_col = 'global_channel_index'
    loc_shank_col = 'shank_index'
    loc_x_col = 'xcoord_on_shank_um'
    loc_y_col = 'ycoord_on_shank_um'
    loc_region_col = 'acronym'


    for rec_info in tqdm(selected_recordings, desc="Loading data files"):
        rec_name = rec_info['name']
        try:
            # Load Units
            units_df = pd.DataFrame(list(np.load(rec_info["units"], allow_pickle=True)))
            if 'cluster_id' not in units_df.columns or 'cell_type' not in units_df.columns:
                 print(f"Warning: Unit file for {rec_name} missing 'cluster_id' or 'cell_type'. Skipping.")
                 continue
            units_df['recording'] = rec_name # Add recording here

            # Load Locations
            locations_df = pd.read_csv(rec_info["anatomy"])

            # Check for essential location columns FROM THE CSV using required_csv_cols
            missing_cols = [col for col in required_csv_cols if col not in locations_df.columns]
            if missing_cols:
                 print(f"Warning: Location file for {rec_name} missing required columns: {missing_cols}. Skipping aggregation for this file.")
                 continue

            # Add recording column AFTER check
            locations_df['recording'] = rec_name

            # --- Aggregation Logic ---
            all_types.update(units_df['cell_type'].dropna().unique())
            all_regions.update(locations_df[loc_region_col].dropna().unique()) # Aggregate regions directly from loaded locations

            all_units_list.append(units_df) # Append df with recording col
            all_locations_list.append(locations_df) # Append df with recording col

        except FileNotFoundError:
             print(f"Warning: File not found for {rec_name} (Units: {rec_info['units']}, Anatomy: {rec_info['anatomy']}). Skipping.")
        except Exception as e:
            print(f"Warning: Could not load data for {rec_info['name']}: {e}. Skipping this recording for aggregation.")

    if not all_units_list or not all_locations_list:
        print("FATAL: No valid unit or location data loaded after checks. Exiting.")
        return

    master_units_df = pd.concat(all_units_list, ignore_index=True)
    master_locations_df = pd.concat(all_locations_list, ignore_index=True)

    # --- Merge location data ONCE here using column names ---
    print("Merging unit and location data...")
    unit_channel_col = 'peak_channel_index_0based'
    if unit_channel_col not in master_units_df.columns: unit_channel_col = 'channels'
    if unit_channel_col not in master_units_df.columns: print("FATAL: Cannot find channel column in units data. Exiting."); return

    # Define the columns needed AFTER loading and adding 'recording' (using CSV names)
    required_loc_cols_for_merge = [loc_channel_col, 'recording', loc_x_col, loc_y_col, loc_shank_col, loc_region_col]

    # Ensure required cols exist before merge attempt in the concatenated df
    missing_in_master = [col for col in required_loc_cols_for_merge if col not in master_locations_df.columns]
    if missing_in_master:
        print(f"FATAL: Missing required columns in combined location data after loading: {missing_in_master}. Exiting.")
        return

    # Ensure data types before merge
    try:
        master_units_df[unit_channel_col] = master_units_df[unit_channel_col].astype(int)
        master_locations_df[loc_channel_col] = master_locations_df[loc_channel_col].astype(int)
    except Exception as e:
        print(f"Error converting channel columns to int before merge: {e}. Exiting.")
        return

    # Perform the merge using correct CSV column names from locations_df
    master_units_with_locations_df = pd.merge(master_units_df,
                                              master_locations_df[required_loc_cols_for_merge],
                                              left_on=['recording', unit_channel_col],
                                              right_on=['recording', loc_channel_col], # Use global_channel_index here
                                              how='left') # Use left merge to keep all units

    # Rename location columns in the merged df for internal consistency
    rename_dict = {
        loc_x_col: 'xcoord_on_shank_um', # Internal name
        loc_y_col: 'ycoord_on_shank_um', # Internal name
        loc_shank_col: 'shank_id',      # Internal name
        loc_region_col: 'acronym'      # Internal name
    }
    master_units_with_locations_df.rename(columns=rename_dict, inplace=True)


    if master_units_with_locations_df['xcoord_on_shank_um'].isna().all():
         print("Warning: Merging locations failed - coordinates are all NaN after merge. Distance calculations will fail.")

    # --- Continue GUI selections ---
    selections = get_multi_selection_gui(sorted(list(all_regions)), sorted(list(all_types)))
    if not selections or not selections.get('regions') or not selections.get('types'): print("No valid selections made. Exiting."); return

    methods_to_run = get_method_selection_gui()
    if not methods_to_run: print("No analysis methods selected. Exiting."); return
    print(f"Selected methods: {methods_to_run}")

    # --- 3. Setup Output Dirs & Data Structures ---
    base_output_dir = data_dir / "Batch_Analysis_Output"
    base_output_dir.mkdir(exist_ok=True)
    method_paths = {}
    for method in methods_to_run:
        method_dir_name = "Deconvolution_Method" if method == 'deconvolution' else "Poisson_Method"
        method_path = base_output_dir / method_dir_name
        method_path.mkdir(exist_ok=True)
        method_paths[method] = method_path
    significant_summary_data = {method: [] for method in methods_to_run}
    # We already loaded all units/locations: all_units_list, all_locations_list

    # --- 4. Main Processing Loop (Modified) ---
    for rec_info in selected_recordings:
        rec_name = rec_info['name']
        print(f"\n{'='*20}\nProcessing Recording: {rec_name}\n{'='*20}")

        # Get pre-merged units for this recording using the renamed columns
        # Use copy to avoid SettingWithCopyWarning if modifying later
        rec_units_df = master_units_with_locations_df[master_units_with_locations_df['recording'] == rec_name].copy()
        if rec_units_df.empty:
            print(f"Skipping {rec_name}: No units found after merging locations.")
            continue

        # --- Load epochs and ask for selection ---
        try:
            epochs_data = np.load(rec_info["timestamps"], allow_pickle=True).item()
            all_epochs_in_rec = epochs_data.get('EpochFrameData') # Safely get data
            if all_epochs_in_rec is None or not isinstance(all_epochs_in_rec, list) or not all_epochs_in_rec:
                raise ValueError("EpochFrameData not found or invalid format")
            selected_epochs_for_rec = get_epoch_selection_gui(all_epochs_in_rec)
            if not selected_epochs_for_rec:
                 print(f"No epochs selected for {rec_name}. Skipping recording.")
                 continue
            print(f"Selected epochs for {rec_name}: {[e.get('epoch_index', 'NA') for e in selected_epochs_for_rec]}")
        except Exception as e:
            print(f"Error loading or selecting epochs for {rec_name}: {e}. Skipping."); continue

        defined_groups = []
        # Ensure the 'acronym' and 'cell_type' columns (after potential renaming) exist
        region_col_internal = 'acronym' # This is the name AFTER renaming
        cell_type_col = 'cell_type'

        if region_col_internal not in rec_units_df.columns:
            print(f"Warning: Region column '{region_col_internal}' not found in merged data for {rec_name}. Cannot filter by region.")
            # Optionally skip or proceed without region filtering depending on desired behavior
            # continue # Skip this recording if region filtering is essential
        if cell_type_col not in rec_units_df.columns:
             print(f"Warning: Cell type column '{cell_type_col}' not found in merged data for {rec_name}. Cannot filter by type.")
             # continue # Skip this recording if cell type filtering is essential

        # Proceed with filtering only if columns exist
        if region_col_internal in rec_units_df.columns and cell_type_col in rec_units_df.columns:
            for region in selections['regions']:
                for ntype in selections['types']:
                    # Filter using the internal column name 'acronym'
                    group_df = rec_units_df[
                        (rec_units_df[region_col_internal].astype(str) == str(region)) & 
                        (rec_units_df[cell_type_col].astype(str) == str(ntype)) &
                        (rec_units_df[region_col_internal].notna()) & 
                        (rec_units_df[cell_type_col].notna())      
                    ]
                    if not group_df.empty:
                        defined_groups.append({
                            "name": f"{region}_{ntype.replace(' ', '')}",
                            "units_list": group_df.to_dict('records')
                        })

        if len(defined_groups) < 1: print(f"Fewer than one selected group exists in {rec_name}. Skipping analysis."); continue

        group_pairs_for_tasks = list(product(defined_groups, repeat=2))
        tasks = []
        for method_name in methods_to_run:
            for g1_info, g2_info in group_pairs_for_tasks:
                 for epoch in selected_epochs_for_rec:
                     for pre_n_dict in g1_info["units_list"]:
                         for post_n_dict in g2_info["units_list"]:
                             # Ensure spike times are present before adding task
                             if 'spike_times_sec' in pre_n_dict and 'spike_times_sec' in post_n_dict:                            
                                 if pre_n_dict.get('cluster_id') != post_n_dict.get('cluster_id'):
                                     tasks.append((pre_n_dict, post_n_dict, epoch,
                                                   f"{g1_info['name']}_vs_{g2_info['name']}", method_name))
                             else:
                                 print(f"Warning: Missing 'spike_times_sec' for unit {pre_n_dict.get('cluster_id')} or {post_n_dict.get('cluster_id')} in {rec_name}. Skipping pair.")


        if not tasks: print(f"No valid neuron pairs with spike times to analyze for {rec_name} in selected epochs."); continue
        print(f"Generated {len(tasks)} total tasks for {rec_name} across {len(methods_to_run)} method(s) and {len(selected_epochs_for_rec)} epoch(s).")

        num_workers = mp.cpu_count() - 2 if mp.cpu_count() > 2 else 1
        print(f"Using {num_workers} worker processes...")
        with mp.Pool(processes=num_workers) as pool:
            results = list(tqdm(pool.imap_unordered(analyze_pair_worker, tasks, chunksize=max(1, len(tasks)//(num_workers*4))), total=len(tasks), desc=f"Analyzing {rec_name}")) # Adjusted chunksize

        all_results = [res for res in results if res is not None]
        if not all_results: print(f"No valid pairs were analyzed for {rec_name}."); continue

        # --- Log and Summary Saving ---
        log_df_list = [{'method': res['method'], 'comparison': res['comparison'], 'epoch_index': res['epoch_index'],
                        'pre_cluster_id': res['pre_neuron']['cluster_id'], 'post_cluster_id': res['post_neuron']['cluster_id'],
                        'pre_spike_count': res['pre_spike_count'], 'post_spike_count': res['post_spike_count'], 'status': res['status']}
                       for res in all_results]
        log_df = pd.DataFrame(log_df_list);
        log_path = base_output_dir / f"full_analysis_log_{rec_name}.csv";
        try:
             log_df.to_csv(log_path, index=False)
             print(f"\nSUCCESS: Full analysis log for {rec_name} saved to {log_path.name}")
        except Exception as e:
             print(f"\nError saving analysis log for {rec_name}: {e}")


        significant_results = [res for res in all_results if res['status'] == 'significant_peak_found' or res['status'] == 'significant_synapse']
        if not significant_results: print(f"No significant interactions found for {rec_name}."); continue

        print(f"Found {len(significant_results)} total significant interactions for {rec_name}. Saving plots and summaries...")
        rec_summary_df = {method: [] for method in methods_to_run}

        for res in tqdm(significant_results, desc=f"Saving for {rec_name}"):
            method_name = res['method']
            method_dir = method_paths[method_name]
            rec_output_dir = method_dir / rec_name
            comp_dir = rec_output_dir / res['comparison'];
            comp_dir.mkdir(parents=True, exist_ok=True)

            pre_n, post_n = res['pre_neuron'], res['post_neuron']
            # Use renamed 'acronym' column for ID
            pre_id = f"{pre_n.get('acronym','NA')}_{pre_n.get('cell_type','NA').replace(' ', '')}_ID{pre_n.get('cluster_id','NA')}";
            post_id = f"{post_n.get('acronym','NA')}_{post_n.get('cell_type','NA').replace(' ', '')}_ID{post_n.get('cluster_id','NA')}"

            # Base summary row using internal consistent names (acronym, shank_id)
            summary_row_base = {
                'recording': rec_name, 'comparison': res['comparison'], 'epoch_index': res['epoch_index'],
                'pre_cluster_id': pre_n.get('cluster_id'), 'pre_region': pre_n.get('acronym'), 'pre_cell_type': pre_n.get('cell_type'), 'pre_shank_id': res['pre_shank_id'],
                'post_cluster_id': post_n.get('cluster_id'), 'post_region': post_n.get('acronym'), 'post_cell_type': post_n.get('cell_type'), 'post_shank_id': res['post_shank_id'],
                'radial_distance_um': res['radial_distance'], 'pre_burst_index': res['pre_burst_index'], 'asymmetry_index': res['asymmetry_index']
            }

            try: # Add error handling for plotting/summary saving
                if method_name == 'deconvolution':
                    if res['analysis_results'] and res['analysis_results']["significant_peaks"]:
                        plot_deconvolution_results(res['analysis_results'], res['epoch_index'], pre_id, post_id, comp_dir)
                        for peak in res['analysis_results']["significant_peaks"]:
                            summary_row = {**summary_row_base,
                                           'peak_lag_ms': peak['lag_ms'], 'spike_gain': peak['gain'], 'z_score': peak['z_score']}
                            significant_summary_data[method_name].append(summary_row)
                            rec_summary_df[method_name].append(summary_row)

                elif method_name == 'poisson':
                     if res['analysis_results']:
                        plot_poisson_results(res['analysis_results'], res['epoch_index'], pre_id, post_id, comp_dir)
                        summary_row = {**summary_row_base,
                                       'p_fast': res['analysis_results']['p_fast'], 'p_causal': res['analysis_results']['p_causal'],
                                       'spike_transmission_prob': res['analysis_results']['stp'],
                                       'peak_lag_ms': res['analysis_results']['peak_lag_ms']
                                      }
                        significant_summary_data[method_name].append(summary_row)
                        rec_summary_df[method_name].append(summary_row)

                if run_conditional_analysis:
                    # Check if analysis results exist before running conditional
                    if res.get('analysis_results'):
                        run_conditional_analysis(pre_n, post_n, res['epoch'], comp_dir, method_name)
                    else:
                        print(f"  Skipping conditional analysis for {pre_id}->{post_id}: No primary analysis results.")

            except Exception as e:
                print(f"  Error during saving/plotting for pair {pre_id}->{post_id}: {e}")


        for method_name in methods_to_run:
            summary_df_rec = pd.DataFrame(rec_summary_df[method_name])
            if not summary_df_rec.empty:
                rec_output_dir = method_paths[method_name] / rec_name
                summary_path = rec_output_dir / f"summary_interactions_{rec_name}_{method_name}.csv";
                try:
                    summary_df_rec.to_csv(summary_path, index=False, float_format='%.4f')
                    print(f"SUCCESS: Summary for {rec_name} ({method_name}) saved to {summary_path.name}")
                except Exception as e:
                    print(f"Error saving summary for {rec_name} ({method_name}): {e}")


    # --- 5. Final Master Summaries and Plots ---
    if not all_units_list or not all_locations_list: # Check if original lists have data
        print("\nOriginal unit or location lists empty, skipping master plots.")
        return

    # Use original DFs for anatomical plot which expects specific columns before potential rename
    master_units_df_orig = pd.concat(all_units_list, ignore_index=True)
    master_locations_df_orig = pd.concat(all_locations_list, ignore_index=True)

    for method_name in methods_to_run:
        print(f"\n--- Generating Master Files for {method_name.upper()} METHOD ---")
        method_path = method_paths[method_name]

        if significant_summary_data[method_name]:
            master_summary_df = pd.DataFrame(significant_summary_data[method_name])
            summary_path = base_output_dir / f"master_summary_of_all_interactions_{method_name}.csv";
            try:
                master_summary_df.to_csv(summary_path, index=False, float_format='%.4f')
                print(f"SUCCESS: Master summary for {method_name.upper()} saved to {summary_path.name}")
            except Exception as e:
                 print(f"Error saving master summary for {method_name}: {e}")
                 continue # Skip plotting if summary failed

            if method_name == 'deconvolution':
                metric_col = 'spike_gain'
                metric_label = 'Spike Gain (Deconv. Counts)'
            else: # poisson
                metric_col = 'spike_transmission_prob'
                metric_label = 'Spike Transmission Prob. (STP)'

            # Pass original DFs to anatomical plot
            generate_anatomical_plot(master_summary_df, master_units_df_orig, master_locations_df_orig,
                                     method_path, method_name, metric_col, metric_label)

            # Pass merged DF (with renamed cols) to advanced plots
            generate_advanced_plots(master_summary_df, master_units_with_locations_df, master_locations_df_orig, # Pass original locs too if needed
                                    base_output_dir, method_name)

        else:
            print(f"No significant interactions found for {method_name.upper()}, skipping master summary and advanced plots.")

        # Generate population plots for this method
        metric_col_name = 'metric_val' # This is the generic name used in the conditional CSV
        generate_population_plots(method_path, method_name, metric_col_name)

    print("\n--- All Analyses Processed ---")

if __name__ == "__main__":
    if os.name == 'nt':
        mp.freeze_support()
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module='seaborn') 
    warnings.filterwarnings("ignore", category=FutureWarning) 
    main()
    input("\nPress Enter to exit.")