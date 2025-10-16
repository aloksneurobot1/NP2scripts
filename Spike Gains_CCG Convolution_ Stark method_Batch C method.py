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
"""
# -*- coding: utf-8 -*-
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
import numba
from numba import jit

# --- Configuration ---
MIN_SPIKES_FOR_ANALYSIS = 100 

# =============================================================================
# --- DECONVOLUTION ANALYSIS CORE FUNCTIONS (Unchanged) ---
# =============================================================================
@jit(nopython=True)
def find_lags_fast(ref_spikes, target_spikes_sorted, window_size):
    """Numba-accelerated function to find all time lags between two spike trains."""
    all_lags = []
    for ref_time in ref_spikes:
        start_time, end_time = ref_time - window_size, ref_time + window_size
        start_idx = np.searchsorted(target_spikes_sorted, start_time, side='left')
        end_idx = np.searchsorted(target_spikes_sorted, end_time, side='right')
        if end_idx > start_idx:
            nearby_spikes = target_spikes_sorted[start_idx:end_idx]
            diffs = nearby_spikes - ref_time
            for d in diffs:
                all_lags.append(d)
    return np.array(all_lags)

def calculate_cch(ref_spikes, target_spikes, window_size, bin_size):
    """Calculates a CCH by calling a high-performance Numba worker function."""
    is_acg = np.array_equal(ref_spikes, target_spikes)
    target_spikes_sorted = np.sort(target_spikes)
    lags = find_lags_fast(ref_spikes, target_spikes_sorted, window_size)
    if is_acg:
        lags = lags[lags != 0]
    num_bins = int(2 * window_size / bin_size)
    bins = np.linspace(-window_size, window_size, num_bins + 1)
    counts, _ = np.histogram(lags, bins=bins)
    return counts, bins

def hollowed_median_filter(data, window_size):
    """Computes a hollowed median filter, as in cch_conv.m."""
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd.")
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
    """Performs CCH deconvolution using the hollowed median filter and Poisson test."""
    cch_raw, bins = calculate_cch(pre_spikes, post_spikes, window_size, bin_size)
    acg_raw, _ = calculate_cch(pre_spikes, pre_spikes, window_size, bin_size)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    cch_fft = np.fft.fft(cch_raw)
    acg_fft = np.fft.fft(acg_raw)
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
        lambda_upper = np.max(pred_roi)
        lambda_lower = np.min(pred_roi)
        threshold_upper = poisson.ppf(1 - corrected_alpha, lambda_upper)
        threshold_lower = poisson.ppf(corrected_alpha, lambda_lower)
        dc_cch_roi = deconvolved_cch[roi_indices]
        excitatory_indices = roi_indices[np.where(dc_cch_roi > threshold_upper)[0]]
        inhibitory_indices = roi_indices[np.where(dc_cch_roi < threshold_lower)[0]]
        all_significant_indices = np.concatenate([excitatory_indices, inhibitory_indices])
        if all_significant_indices.size > 0:
            deviations = deconvolved_cch[all_significant_indices] - predictor[all_significant_indices]
            peak_index = all_significant_indices[np.argmax(np.abs(deviations))]
            spike_gain = deconvolved_cch[peak_index] 
            z_score = (spike_gain - predictor[peak_index]) / (np.std(predictor) + 1e-9)
            significant_peaks.append({"lag_ms": bin_centers_ms[peak_index], "gain": spike_gain, "z_score": z_score})
    return {"deconvolved_cch": deconvolved_cch, "predictor": predictor, "significant_peaks": significant_peaks}

def plot_deconvolution_results(results, epoch_idx, pre_id, post_id, output_dir):
    """Creates and saves a plot of the deconvolved CCH and its predictor."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6))
    bin_centers_ms = np.linspace(-100, 100, len(results['deconvolved_cch']))
    ax.bar(bin_centers_ms, results['deconvolved_cch'], width=1.0, color='gray', label='Deconvolved CCH (dcCCH)')
    ax.plot(bin_centers_ms, results['predictor'], color='red', linestyle='--', label='Predictor (Hollowed Median)')
    for peak in results["significant_peaks"]:
        ax.plot(peak['lag_ms'], peak['gain'], 'bo', markersize=8, label=f"Significant Peak (Gain: {peak['gain']:.2f})")
    ax.set_title(f'Deconvolution: Epoch {epoch_idx}\n{pre_id} -> {post_id}', fontsize=14)
    ax.set_xlabel('Time Lag (ms)'); ax.set_ylabel('Deconvolved Counts'); ax.legend()
    plt.tight_layout()
    filename = f"Epoch{epoch_idx}_Deconv_{pre_id}_vs_{post_id}.tiff"
    fig.savefig(output_dir / filename, dpi=150, format='tiff'); plt.close(fig)

# =============================================================================
# --- GUI AND FILE HANDLING (UNCHANGED) ---
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
    result_dict = {"comparison": comparison_name, "epoch_index": epoch['epoch_index'], "pre_neuron": pre_neuron, "post_neuron": post_neuron, "pre_spike_count": len(pre_epoch_spikes), "post_spike_count": len(post_epoch_spikes), "status": "", "analysis_results": None}
    if len(pre_epoch_spikes) < MIN_SPIKES_FOR_ANALYSIS or len(post_epoch_spikes) < MIN_SPIKES_FOR_ANALYSIS:
        result_dict["status"] = "insufficient_spikes"
        return result_dict
    analysis_results = deconvolve_and_analyze(pre_epoch_spikes, post_epoch_spikes)
    if analysis_results["significant_peaks"]:
        result_dict["status"] = "significant_peak_found"
        result_dict["analysis_results"] = analysis_results
    else:
        result_dict["status"] = "no_significant_peak"
    return result_dict

def main():
    print("--- Starting Numba-Accelerated Batch Deconvolution Analysis ---")
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
        except IndexError:
            print(f"  - Warning: Could not parse name from: {unit_path.name}")
    if not recording_pairs: print("No valid recording pairs found. Exiting."); return
    print("\nAggregating all brain regions and cell types for GUI...")
    all_regions, all_types = set(), set()
    for rec_info in tqdm(recording_pairs, desc="Scanning files"):
        try:
            df = pd.DataFrame(list(np.load(rec_info["units"], allow_pickle=True)))
            all_regions.update(df['acronym'].dropna().unique())
            all_types.update(df['cell_type'].dropna().unique())
        except Exception as e:
            print(f"Warning: Could not read {rec_info['name']} during scan: {e}")
    selections = get_multi_selection_gui(sorted(list(all_regions)), sorted(list(all_types)))
    if not selections or not selections.get('regions') or not selections.get('types'):
        print("No valid selections made. Exiting."); return
    base_output_dir = data_dir / "Batch_Deconvolution_Output"
    base_output_dir.mkdir(exist_ok=True)
    
    for rec_info in recording_pairs:
        rec_name = rec_info['name']
        print(f"\n{'='*20}\nProcessing Recording: {rec_name}\n{'='*20}")
        rec_output_dir = base_output_dir / rec_name; rec_output_dir.mkdir(exist_ok=True)
        try:
            units_df = pd.DataFrame(list(np.load(rec_info["units"], allow_pickle=True)))
            epochs = np.load(rec_info["timestamps"], allow_pickle=True).item().get('EpochFrameData')
        except Exception as e:
            print(f"Error loading files for {rec_name}: {e}"); continue
        defined_groups = []
        print(f"Forming neuron groups for {rec_name}:")
        for region in selections['regions']:
            for ntype in selections['types']:
                group_df = units_df[(units_df['acronym'] == region) & (units_df['cell_type'] == ntype)]
                if not group_df.empty:
                    print(f"  - Found {len(group_df)} neurons for group: {region}-{ntype}")
                    defined_groups.append({"name": f"{region}_{ntype.replace(' ', '')}", "dataframe": group_df})
        if len(defined_groups) < 1:
            print(f"Fewer than one selected group exists in {rec_name}. Skipping analysis."); continue
        
        group_pairs = list(product(defined_groups, repeat=2))
        tasks = [(pre_n, post_n, epoch, f"{g1['name']}_vs_{g2['name']}")
                 for g1, g2 in group_pairs
                 for epoch in epochs
                 for _, pre_n in g1["dataframe"].iterrows()
                 for _, post_n in g2["dataframe"].iterrows()]
        if not tasks: print(f"No valid neuron pairs to analyze for {rec_name}."); continue
        print(f"Generated {len(tasks)} tasks for {rec_name} from {len(group_pairs)} directed comparisons.")
        
        num_workers = 6
        with mp.Pool(processes=num_workers) as pool:
            results = list(tqdm(pool.imap_unordered(analyze_pair_worker, tasks), total=len(tasks), desc=f"Analyzing {rec_name}"))
        
        all_results = [res for res in results if res is not None]
        if not all_results:
            print(f"No valid pairs were analyzed for {rec_name}."); continue
        
        log_df_list = []
        for res in all_results:
            log_df_list.append({
                'comparison': res['comparison'], 'epoch_index': res['epoch_index'],
                'pre_cluster_id': res['pre_neuron']['cluster_id'], 
                'post_cluster_id': res['post_neuron']['cluster_id'],
                'pre_spike_count': res['pre_spike_count'], 
                'post_spike_count': res['post_spike_count'],
                'status': res['status']
            })
        log_df = pd.DataFrame(log_df_list)
        log_path = rec_output_dir / f"full_analysis_log_{rec_name}.csv"; log_df.to_csv(log_path, index=False)
        print(f"\nSUCCESS: Full analysis log for {rec_name} saved to {log_path.name}")
        
        significant_summary_data = []
        significant_results = [res for res in all_results if res['status'] == 'significant_peak_found']
        if not significant_results:
            print(f"No significant interactions found for {rec_name}."); continue
            
        print(f"Found {len(significant_results)} interactions for {rec_name}. Saving plots and summary...")
        for res in tqdm(significant_results, desc=f"Saving for {rec_name}"):
            comp_dir = rec_output_dir / res['comparison']; comp_dir.mkdir(exist_ok=True)
            pre_n, post_n = res['pre_neuron'], res['post_neuron']
            pre_id = f"{pre_n['acronym']}_{pre_n['cell_type'].replace(' ', '')}_ID{pre_n['cluster_id']}"
            post_id = f"{post_n['acronym']}_{post_n['cell_type'].replace(' ', '')}_ID{post_n['cluster_id']}"
            # --- KeyError FIX: Changed res['epoch_idx'] to res['epoch_index'] ---
            plot_deconvolution_results(res['analysis_results'], res['epoch_index'], pre_id, post_id, comp_dir)
            for peak in res['analysis_results']["significant_peaks"]:
                significant_summary_data.append({'comparison': res['comparison'], 'epoch_index': res['epoch_index'], 'pre_cluster_id': pre_n['cluster_id'], 'pre_region': pre_n['acronym'], 'pre_cell_type': pre_n['cell_type'], 'post_cluster_id': post_n['cluster_id'], 'post_region': post_n['acronym'], 'post_cell_type': post_n['cell_type'], 'peak_lag_ms': peak['lag_ms'], 'spike_gain': peak['gain'], 'z_score': peak['z_score']})
        
        summary_df = pd.DataFrame(significant_summary_data)
        summary_path = rec_output_dir / f"summary_interactions_{rec_name}.csv"; summary_df.to_csv(summary_path, index=False, float_format='%.4f')
        print(f"SUCCESS: Summary for {rec_name} saved to {summary_path.name}")

    print("\n--- All Recordings Processed ---")

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()
    input("\nPress Enter to exit.")