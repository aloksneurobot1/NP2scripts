# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 14:00:00 2025

This script is a high-performance Python implementation of the Stark Lab's
cch_stg methodology. The CCH calculation is accelerated with Numba to
achieve C-like speeds, dramatically reducing runtime.

Significance estimated via different approach (poisson sim of baseline vs actual data)

It requires a directory with  two input files:
1. The output from the ACG analysis (good_clusters..._CellExplorerACG.npy)
2. The output from timestamp extraction (..._timestamps.npy)

This script is a direct Python implementation of the Stark Lab's cch_stg
methodology. This version includes a new module for "Conditional Analysis"
to test how spike transmission is modulated by prior neural activity.from English 2017 Neuron
including an fitting biophysical models of short-term synaptic
plasticity (Tsodyks-Markram type 1997) and performing a shuffle-based
statistical analysis to determine the best model fit.
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
from scipy.stats import poisson
from scipy.optimize import curve_fit
import numba
from numba import jit

# --- Configuration ---
MIN_SPIKES_FOR_ANALYSIS = 100 
MIN_SPIKES_FOR_CONDITIONAL_ANALYSIS = 300 
RUN_CONDITIONAL_ANALYSIS = True 
N_SHUFFLES_FOR_MODEL_FIT = 100 # Warning: High values are computationally expensive

# --- Conditional Analysis Parameters ---
N_ISI_BINS = 40
ISI_RANGE_MS = (0.5, 2000)
POST_RATE_WINDOW_MS = 200
POST_RATE_BINS = [(0, 5), (5, 10), (10, 20), (20, np.inf)]

# =============================================================================
# --- CORE ANALYSIS & HELPER FUNCTIONS ---
# =============================================================================
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

def hollowed_median_filter(data, window_size):
    if window_size % 2 == 0: raise ValueError("window_size must be odd.")
    n = len(data); predictor = np.zeros(n); half_window = window_size // 2
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
        n_bonf = len(roi_indices)
        corrected_alpha = alpha / n_bonf
        pred_roi = predictor[roi_indices]
        lambda_upper, lambda_lower = np.max(pred_roi), np.min(pred_roi)
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
    return {"cch_raw": cch_raw, "acg_pre": acg_pre, "acg_post": acg_post, "deconvolved_cch": deconvolved_cch, "predictor": predictor, "significant_peaks": significant_peaks}
    
# =============================================================================
# --- ADVANCED SYNAPTIC PLASTICITY MODELING ---
# =============================================================================
@jit(nopython=True)
def model_full(isi, A, U, tau_syn, tau_fac, tau_dep):
    """Full Tsodyks-Markram style model for synaptic plasticity."""
    term_fac = (1 + (U - 1) * np.exp(-isi / tau_fac))
    term_syn = (1 - np.exp(-isi / tau_syn))
    term_dep = (1 - np.exp(-isi / tau_dep))
    return A * term_fac * term_syn * term_dep

@jit(nopython=True)
def model_depression(isi, A, tau_syn, tau_dep):
    """Depression-only model."""
    term_syn = (1 - np.exp(-isi / tau_syn))
    term_dep = (1 - np.exp(-isi / tau_dep))
    return A * term_syn * term_dep

@jit(nopython=True)
def model_facilitation(isi, A, U, tau_syn, tau_fac):
    """Facilitation-only model."""
    term_fac = (1 + (U - 1) * np.exp(-isi / tau_fac))
    term_syn = (1 - np.exp(-isi / tau_syn))
    return A * term_fac * term_syn

def calculate_adjusted_r2(y_true, y_pred, n_params):
    """Calculates the adjusted R-squared value."""
    n_samples = len(y_true)
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    if ss_tot == 0: return 0
    r2 = 1 - (ss_res / ss_tot)
    adj_r2 = 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_params - 1)
    return adj_r2

def fit_plasticity_models(isi_data, gain_data):
    """Fits all three models and uses a shuffle analysis to find the best fit."""
    sort_idx = np.argsort(isi_data)
    isi, gain = isi_data[sort_idx], gain_data[sort_idx]
    
    bounds_full = ([0, 0, 0, 0, 0], [np.inf, np.inf, 0.003, 10.0, 10.0])
    bounds_dep = ([0, 0, 0], [np.inf, 0.003, 10.0])
    bounds_fac = ([0, 0, 0, 0], [np.inf, np.inf, 0.003, 10.0])
    
    try: popt_full, _ = curve_fit(model_full, isi, gain, bounds=bounds_full)
    except (RuntimeError, ValueError): popt_full = None
    try: popt_dep, _ = curve_fit(model_depression, isi, gain, bounds=bounds_dep)
    except (RuntimeError, ValueError): popt_dep = None
    try: popt_fac, _ = curve_fit(model_facilitation, isi, gain, bounds=bounds_fac)
    except (RuntimeError, ValueError): popt_fac = None
        
    adj_r2_full = calculate_adjusted_r2(gain, model_full(isi, *popt_full), 5) if popt_full is not None else -np.inf
    adj_r2_dep = calculate_adjusted_r2(gain, model_depression(isi, *popt_dep), 3) if popt_dep is not None else -np.inf
    adj_r2_fac = calculate_adjusted_r2(gain, model_facilitation(isi, *popt_fac), 4) if popt_fac is not None else -np.inf

    shuffled_r2s = {'full': [], 'dep': [], 'fac': [], 'diff_dep': [], 'diff_fac': []}
    gain_shuffled = gain.copy()
    for _ in range(N_SHUFFLES_FOR_MODEL_FIT):
        np.random.shuffle(gain_shuffled)
        try: popt_s_full, _ = curve_fit(model_full, isi, gain_shuffled, bounds=bounds_full)
        except (RuntimeError, ValueError): popt_s_full = None
        try: popt_s_dep, _ = curve_fit(model_depression, isi, gain_shuffled, bounds=bounds_dep)
        except (RuntimeError, ValueError): popt_s_dep = None
        try: popt_s_fac, _ = curve_fit(model_facilitation, isi, gain_shuffled, bounds=bounds_fac)
        except (RuntimeError, ValueError): popt_s_fac = None
        
        r2_s_full = calculate_adjusted_r2(gain_shuffled, model_full(isi, *popt_s_full), 5) if popt_s_full is not None else -np.inf
        r2_s_dep = calculate_adjusted_r2(gain_shuffled, model_depression(isi, *popt_s_dep), 3) if popt_s_dep is not None else -np.inf
        r2_s_fac = calculate_adjusted_r2(gain_shuffled, model_facilitation(isi, *popt_s_fac), 4) if popt_s_fac is not None else -np.inf
        
        shuffled_r2s['full'].append(r2_s_full); shuffled_r2s['dep'].append(r2_s_dep); shuffled_r2s['fac'].append(r2_s_fac)
        shuffled_r2s['diff_dep'].append(r2_s_full - r2_s_dep); shuffled_r2s['diff_fac'].append(r2_s_full - r2_s_fac)
    
    thresh_full = np.percentile(shuffled_r2s['full'], 95); thresh_dep = np.percentile(shuffled_r2s['dep'], 95)
    thresh_fac = np.percentile(shuffled_r2s['fac'], 95); thresh_diff_dep = np.percentile(shuffled_r2s['diff_dep'], 95)
    thresh_diff_fac = np.percentile(shuffled_r2s['diff_fac'], 95)
    
    best_fit = {"model": "none", "tau": np.nan, "adj_r2": -np.inf, "popt": None}
    is_dep = (adj_r2_dep > thresh_dep) and ((adj_r2_full - adj_r2_dep) < thresh_diff_dep)
    is_fac = (adj_r2_fac > thresh_fac) and ((adj_r2_full - adj_r2_fac) < thresh_diff_fac)
    is_full = (adj_r2_full > thresh_full)

    if is_dep:
        best_fit = {"model": "depression", "tau": popt_dep[2] * 1000, "adj_r2": adj_r2_dep, "popt": popt_dep}
    if is_fac and adj_r2_fac > best_fit['adj_r2']:
        best_fit = {"model": "facilitation", "tau": popt_fac[3] * 1000, "adj_r2": adj_r2_fac, "popt": popt_fac}
    if not is_dep and not is_fac and is_full:
        tau = popt_full[3] * 1000 if popt_full[3] > popt_full[4] else popt_full[4] * 1000
        best_fit = {"model": "full", "tau": tau, "adj_r2": adj_r2_full, "popt": popt_full}
    return best_fit

# =============================================================================
# --- PLOTTING AND CONDITIONAL ANALYSIS WORKFLOWS ---
# =============================================================================
def plot_deconvolution_results(results, epoch_idx, pre_id, post_id, output_dir):
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
    fig.suptitle(f'Interaction Analysis: Epoch {epoch_idx}\n{pre_id} (Pre) -> {post_id} (Post)', fontsize=14)
    bin_centers_ms = np.linspace(-100, 100, len(results['cch_raw']))
    ax1.bar(bin_centers_ms, results['cch_raw'], width=1.0, color='gray', label='Raw CCH')
    ax1.set_ylabel('Spike Pair Count'); ax1.set_title('Raw Cross-Correlogram (CCH)'); ax1.grid(True, linestyle=':', alpha=0.6)
    ax_inset_pre = ax1.inset_axes([0.02, 0.65, 0.25, 0.3]); ax_inset_pre.bar(bin_centers_ms, results['acg_pre'], width=1.0, color='darkred'); ax_inset_pre.set_title('Pre-ACH', fontsize=9); ax_inset_pre.set_xlim(-15, 15); ax_inset_pre.tick_params(axis='both', which='major', labelsize=8)
    ax_inset_post = ax1.inset_axes([0.73, 0.65, 0.25, 0.3]); ax_inset_post.bar(bin_centers_ms, results['acg_post'], width=1.0, color='darkblue'); ax_inset_post.set_title('Post-ACH', fontsize=9); ax_inset_post.set_xlim(-15, 15); ax_inset_post.tick_params(axis='both', which='major', labelsize=8)
    ax2.bar(bin_centers_ms, results['deconvolved_cch'], width=1.0, color='lightgray', label='Deconvolved CCH (dcCCH)')
    ax2.plot(bin_centers_ms, results['predictor'], color='red', linestyle='--', label='Predictor (Hollowed Median)')
    for peak in results["significant_peaks"]: ax2.plot(peak['lag_ms'], peak['gain'], 'bo', markersize=8, label=f"Significant Peak (Gain: {peak['gain']:.2f})")
    ax2.set_ylabel('Deconvolved Counts'); ax2.set_title('Deconvolved CCH & Statistical Test'); ax2.set_xlabel('Time Lag (ms)'); ax2.legend(fontsize=9); ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.set_xlim(-25, 25)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    filename = f"Epoch{epoch_idx}_Deconv_{pre_id}_vs_{post_id}.tiff"; fig.savefig(output_dir / filename, dpi=150, format='tiff'); plt.close(fig)

def plot_conditional_results(x_data, y_data, title, xlabel, output_path, fit_results=None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x_data, y_data, 'o', color='black', label='Data')
    if fit_results and fit_results['model'] != 'none':
        fine_x = np.logspace(np.log10(x_data.min()), np.log10(x_data.max()), 200)
        model_func = {'full': model_full, 'depression': model_depression, 'facilitation': model_facilitation}[fit_results['model']]
        y_fit = model_func(fine_x / 1000, *fit_results['popt'])
        ax.plot(fine_x, y_fit, color='blue', linewidth=2, label=f"Best Fit: {fit_results['model'].capitalize()} (τ = {fit_results['tau']:.1f} ms)")
        ax.legend()
    ax.set_title(title, fontsize=14); ax.set_xlabel(xlabel, fontsize=12); ax.set_ylabel('Spike Gain (Deconvolved Counts)', fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.7)
    if 'ISI' in xlabel or 'Interval' in xlabel: ax.set_xscale('log')
    plt.tight_layout(); fig.savefig(output_path, dpi=150, format='tiff'); plt.close(fig)

def generate_population_plots(base_output_dir):
    import matplotlib.pyplot as plt
    import seaborn as sns
    print("\n--- Generating Population Summary Plots ---")
    csv_files = list(base_output_dir.glob("*/*/*_Cond_PreISI.csv"))
    if not csv_files: print("No conditional plasticity data found to generate population plots."); return
    all_plasticity_data = [pd.read_csv(f) for f in csv_files]
    if not all_plasticity_data: print("Conditional data was empty. Skipping population plots."); return
    pop_df = pd.concat(all_plasticity_data, ignore_index=True)
    plt.figure(figsize=(6, 5)); ax = sns.countplot(x='best_model', data=pop_df.drop_duplicates(subset=['recording', 'comparison', 'epoch']))
    ax.set_title('Prevalence of Synaptic Plasticity Types'); ax.set_xlabel('Best Fit Model'); ax.set_ylabel('Number of Connections')
    plt.tight_layout(); plt.savefig(base_output_dir / "population_plasticity_prevalence.tiff", dpi=150); plt.close()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    dep_tau = pop_df[pop_df['best_model'] == 'depression']['tau'].dropna()
    fac_tau = pop_df[pop_df['best_model'] == 'facilitation']['tau'].dropna()
    if not dep_tau.empty: ax1.hist(dep_tau, bins=np.logspace(np.log10(1), np.log10(1000), 20), color='red', alpha=0.7); ax1.set_xscale('log'); ax1.set_title('Depression Time Constants (τ_d)'); ax1.set_xlabel('τ (ms)')
    if not fac_tau.empty: ax2.hist(fac_tau, bins=np.logspace(np.log10(1), np.log10(1000), 20), color='blue', alpha=0.7); ax2.set_xscale('log'); ax2.set_title('Facilitation Time Constants (τ_f)'); ax2.set_xlabel('τ (ms)')
    plt.tight_layout(); plt.savefig(base_output_dir / "population_tau_distributions.tiff", dpi=150); plt.close()
    print("Population plots generated successfully.")

def run_conditional_analysis(pre_neuron, post_neuron, epoch, output_dir):
    start_time, end_time = epoch['start_time_sec'], epoch['end_time_sec']
    pre_epoch_spikes = pre_neuron['spike_times_sec'][(pre_neuron['spike_times_sec'] >= start_time) & (pre_neuron['spike_times_sec'] < end_time)]
    post_epoch_spikes = post_neuron['spike_times_sec'][(post_neuron['spike_times_sec'] >= start_time) & (post_neuron['spike_times_sec'] < end_time)]
    pre_id = f"{pre_neuron['acronym']}_{pre_neuron['cell_type'].replace(' ', '')}_ID{pre_neuron['cluster_id']}"; post_id = f"{post_neuron['acronym']}_{post_neuron['cell_type'].replace(' ', '')}_ID{post_neuron['cluster_id']}"
    
    isi_bins_sec = np.logspace(np.log10(ISI_RANGE_MS[0] / 1000), np.log10(ISI_RANGE_MS[1] / 1000), N_ISI_BINS + 1)
    binned_pre_spikes = subsample_by_presynaptic_isi(pre_epoch_spikes, isi_bins_sec)
    results_std = []
    for bin_center_sec, subsample in binned_pre_spikes.items():
        if len(subsample) >= MIN_SPIKES_FOR_CONDITIONAL_ANALYSIS:
            res = deconvolve_and_analyze(subsample, post_epoch_spikes)
            if res['significant_peaks']: results_std.append({'isi_ms': bin_center_sec * 1000, 'gain': res['significant_peaks'][0]['gain']})
    
    if len(results_std) > 5:
        df = pd.DataFrame(results_std)
        fit_results = fit_plasticity_models(df['isi_ms'].values / 1000, df['gain'].values)
        df['best_model'] = fit_results['model']; df['tau'] = fit_results['tau']; df['adj_r2'] = fit_results['adj_r2']
        df['recording'] = output_dir.parts[-2]; df['comparison'] = output_dir.name; df['epoch'] = epoch['epoch_index']
        plot_conditional_results(df['isi_ms'], df['gain'], f"Short-Term Plasticity\n{pre_id} -> {post_id}", "Presynaptic ISI (ms)", output_dir / f"Epoch{epoch['epoch_index']}_Cond_PreISI.tiff", fit_results=fit_results)
        df.to_csv(output_dir / f"Epoch{epoch['epoch_index']}_Cond_PreISI.csv", index=False)

# =============================================================================
# --- NameError FIX: Restoring the missing GUI AND FILE HANDLING section ---
# =============================================================================
def select_directory(title):
    root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
    dir_path = filedialog.askdirectory(title=title)
    root.destroy()
    return Path(dir_path) if dir_path else None

def get_multi_selection_gui(regions, types):
    result = {}
    window = tk.Tk(); window.title("Select Groups for Pairwise Comparison"); window.attributes("-topmost", True)
    region_frame = Frame(window, relief='sunken', borderwidth=1); region_frame.pack(pady=10, padx=10, fill='x')
    type_frame = Frame(window, relief='sunken', borderwidth=1); type_frame.pack(pady=10, padx=10, fill='x')
    Label(region_frame, text="Select Brain Regions", font=('Helvetica', 10, 'bold')).pack()
    Label(type_frame, text="Select Neuron Types", font=('Helvetica', 10, 'bold')).pack()
    region_vars = {region: BooleanVar(value=False) for region in regions}
    def toggle_all_regions():
        new_state = all_regions_var.get()
        for var in region_vars.values(): var.set(new_state)
    all_regions_var = BooleanVar()
    Checkbutton(region_frame, text="-- SELECT ALL --", variable=all_regions_var, command=toggle_all_regions).pack(anchor='w')
    for region, var in region_vars.items(): Checkbutton(region_frame, text=region, variable=var).pack(anchor='w')
    type_vars = {ntype: BooleanVar(value=False) for ntype in types}
    def toggle_all_types():
        new_state = all_types_var.get()
        for var in type_vars.values(): var.set(new_state)
    all_types_var = BooleanVar()
    Checkbutton(type_frame, text="-- SELECT ALL --", variable=all_types_var, command=toggle_all_types).pack(anchor='w')
    for ntype, var in type_vars.items(): Checkbutton(type_frame, text=ntype, variable=var).pack(anchor='w')
    def on_submit():
        result['regions'] = [r for r, v in region_vars.items() if v.get()]
        result['types'] = [t for t, v in type_vars.items() if v.get()]
        if not result['regions'] or not result['types']: print("Warning: Select at least one region and type.")
        else: window.quit(); window.destroy()
    Button(window, text="Confirm Selections for All Recordings", command=on_submit, font=('Helvetica', 10, 'bold')).pack(pady=10)
    window.mainloop()
    return result

# =============================================================================
# --- PARALLEL WORKER AND MAIN WORKFLOW ---
# =============================================================================
def analyze_pair_worker(args):
    """Worker function that analyzes one pair and returns a detailed, consistent dictionary."""
    pre_neuron, post_neuron, epoch, comparison_name = args
    if pre_neuron['cluster_id'] == post_neuron['cluster_id']: return None
    start_time, end_time = epoch['start_time_sec'], epoch['end_time_sec']
    pre_spikes_all = pre_neuron['spike_times_sec']
    pre_epoch_spikes = pre_spikes_all[(pre_spikes_all >= start_time) & (pre_spikes_all < end_time)]
    post_spikes_all = post_neuron['spike_times_sec']
    post_epoch_spikes = post_spikes_all[(post_spikes_all >= start_time) & (post_spikes_all < end_time)]
    result_dict = {"comparison": comparison_name, "epoch_index": epoch['epoch_index'], "pre_neuron": pre_neuron, "post_neuron": post_neuron, "pre_spike_count": len(pre_epoch_spikes), "post_spike_count": len(post_epoch_spikes), "status": "", "analysis_results": None, "epoch": epoch}
    if len(pre_epoch_spikes) < MIN_SPIKES_FOR_ANALYSIS or len(post_epoch_spikes) < MIN_SPIKES_FOR_ANALYSIS:
        result_dict["status"] = "insufficient_spikes"
        return result_dict
    analysis_results = deconvolve_and_analyze(pre_epoch_spikes, post_epoch_spikes)
    if analysis_results["significant_peaks"]:
        result_dict["status"] = "significant_peak_found"; result_dict["analysis_results"] = analysis_results
    else:
        result_dict["status"] = "no_significant_peak"
    return result_dict

def main():
    print("--- Starting Numba-Accelerated Batch Deconvolution Analysis with Conditional Module ---")
    data_dir = select_directory("Select the Folder Containing All Your Recordings")
    if not data_dir: print("No directory selected. Exiting."); return
    recording_pairs = []
    print("Scanning for recordings...")
    for unit_path in data_dir.glob('good_clusters_processed_*_CellExplorerACG.npy'):
        try:
            name_part = unit_path.name.replace('good_clusters_processed_', '').split('_imec0')[0]
            ts_path = data_dir / f"{name_part}_tcat.nidq_timestamps.npy"
            if ts_path.exists():
                recording_pairs.append({"name": name_part, "units": unit_path, "timestamps": ts_path})
                print(f"  - Found match: {name_part}")
        except IndexError: print(f"  - Warning: Could not parse name from: {unit_path.name}")
    if not recording_pairs: print("No valid recording pairs found. Exiting."); return
    print("\nAggregating all brain regions and cell types for GUI...")
    all_regions, all_types = set(), set()
    for rec_info in tqdm(recording_pairs, desc="Scanning files"):
        try:
            df = pd.DataFrame(list(np.load(rec_info["units"], allow_pickle=True))); all_regions.update(df['acronym'].dropna().unique()); all_types.update(df['cell_type'].dropna().unique())
        except Exception as e: print(f"Warning: Could not read {rec_info['name']} during scan: {e}")
    selections = get_multi_selection_gui(sorted(list(all_regions)), sorted(list(all_types)))
    if not selections or not selections.get('regions') or not selections.get('types'):
        print("No valid selections made. Exiting."); return
    base_output_dir = data_dir / "Batch_Deconvolution_Output"
    base_output_dir.mkdir(exist_ok=True)
    
    significant_summary_data = [] # Move this outside the loop to collect all data
    
    for rec_info in recording_pairs:
        rec_name = rec_info['name']
        print(f"\n{'='*20}\nProcessing Recording: {rec_name}\n{'='*20}")
        rec_output_dir = base_output_dir / rec_name; rec_output_dir.mkdir(exist_ok=True)
        try:
            units_df = pd.DataFrame(list(np.load(rec_info["units"], allow_pickle=True))); epochs = np.load(rec_info["timestamps"], allow_pickle=True).item().get('EpochFrameData')
        except Exception as e: print(f"Error loading files for {rec_name}: {e}"); continue
        defined_groups = []
        print(f"Forming neuron groups for {rec_name}:")
        for region in selections['regions']:
            for ntype in selections['types']:
                group_df = units_df[(units_df['acronym'] == region) & (units_df['cell_type'] == ntype)]
                if not group_df.empty: print(f"  - Found {len(group_df)} neurons for group: {region}-{ntype}"); defined_groups.append({"name": f"{region}_{ntype.replace(' ', '')}", "dataframe": group_df})
        if len(defined_groups) < 1:
            print(f"Fewer than one selected group exists in {rec_name}. Skipping analysis."); continue
        
        group_pairs = list(product(defined_groups, repeat=2))
        tasks = [(pre_n, post_n, epoch, f"{g1['name']}_vs_{g2['name']}") for g1, g2 in group_pairs for epoch in epochs for _, pre_n in g1["dataframe"].iterrows() for _, post_n in g2["dataframe"].iterrows()]
        if not tasks: print(f"No valid neuron pairs to analyze for {rec_name}."); continue
        print(f"Generated {len(tasks)} tasks for {rec_name} from {len(group_pairs)} directed comparisons.")
        
        num_workers = 6
        with mp.Pool(processes=num_workers) as pool:
            results = list(tqdm(pool.imap_unordered(analyze_pair_worker, tasks), total=len(tasks), desc=f"Analyzing {rec_name}"))
        
        all_results = [res for res in results if res is not None]
        if not all_results:
            print(f"No valid pairs were analyzed for {rec_name}."); continue
        
        log_df_list = [{'comparison': res['comparison'], 'epoch_index': res['epoch_index'], 'pre_cluster_id': res['pre_neuron']['cluster_id'], 'post_cluster_id': res['post_neuron']['cluster_id'], 'pre_spike_count': res['pre_spike_count'], 'post_spike_count': res['post_spike_count'], 'status': res['status']} for res in all_results]
        log_df = pd.DataFrame(log_df_list)
        log_path = rec_output_dir / f"full_analysis_log_{rec_name}.csv"; log_df.to_csv(log_path, index=False)
        print(f"\nSUCCESS: Full analysis log for {rec_name} saved to {log_path.name}")
        
        significant_results = [res for res in all_results if res['status'] == 'significant_peak_found']
        if not significant_results:
            print(f"No significant interactions found for {rec_name}."); continue
            
        print(f"Found {len(significant_results)} interactions for {rec_name}. Saving plots and summary...")
        for res in tqdm(significant_results, desc=f"Saving for {rec_name}"):
            comp_dir = rec_output_dir / res['comparison']; comp_dir.mkdir(exist_ok=True)
            pre_n, post_n = res['pre_neuron'], res['post_neuron']
            pre_id = f"{pre_n['acronym']}_{pre_n['cell_type'].replace(' ', '')}_ID{pre_n['cluster_id']}"; post_id = f"{post_n['acronym']}_{post_n['cell_type'].replace(' ', '')}_ID{post_n['cluster_id']}"
            plot_deconvolution_results(res['analysis_results'], res['epoch_index'], pre_id, post_id, comp_dir)
            for peak in res['analysis_results']["significant_peaks"]:
                significant_summary_data.append({'recording': rec_name, 'comparison': res['comparison'], 'epoch_index': res['epoch_index'], 'pre_cluster_id': pre_n['cluster_id'], 'pre_region': pre_n['acronym'], 'pre_cell_type': pre_n['cell_type'], 'post_cluster_id': post_n['cluster_id'], 'post_region': post_n['acronym'], 'post_cell_type': post_n['cell_type'], 'peak_lag_ms': peak['lag_ms'], 'spike_gain': peak['gain'], 'z_score': peak['z_score']})
            if RUN_CONDITIONAL_ANALYSIS:
                run_conditional_analysis(pre_n, post_n, res['epoch'], comp_dir)
                
    summary_df = pd.DataFrame(significant_summary_data)
    if not summary_df.empty:
        summary_path = base_output_dir / "master_summary_of_all_interactions.csv"; summary_df.to_csv(summary_path, index=False, float_format='%.4f')
        print(f"\nSUCCESS: Master summary for all recordings saved to {summary_path.name}")

    generate_population_plots(base_output_dir)
    print("\n--- All Recordings Processed ---")

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    # You may need to install seaborn: pip install seaborn
    main()
    input("\nPress Enter to exit.")