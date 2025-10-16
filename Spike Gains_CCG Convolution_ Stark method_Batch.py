# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 16:30:00 2025

Batch analysis 

based on stark lab https://github.com/EranStarkLab/CCH-deconvolution
This script performs a comprehensive, epoch-aligned deconvolution analysis to find
spike transmission gain between all combinations of user-selected neuron groups.

It requires two input files:
1. The output from the ACG analysis (good_clusters..._CellExplorerACG.npy)
2. The output from timestamp extraction (..._timestamps.npy)

The script prompts the user with a GUI to select multiple brain regions and
neuron types. It then automatically performs a pairwise deconvolution analysis
for every possible combination of the resulting groups (e.g., comparing
CA1-Pyramidal vs. CA3-Interneurons, CA1-Pyramidal vs. CA1-Interneurons, etc.).

For each significant interaction found, it saves a diagnostic plot and compiles
all findings into a single summary CSV file.

"""

import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog, Toplevel, Label, Button, Checkbutton, Frame, BooleanVar
from pathlib import Path
import warnings
from itertools import product # --- DIRECTED COMPARISON CHANGE ---
import multiprocessing as mp
import os
from tqdm import tqdm

# --- Configuration ---
MIN_SPIKES_FOR_ANALYSIS = 100 

# =============================================================================
# --- DECONVOLUTION ANALYSIS CORE FUNCTIONS (Unchanged) ---
# =============================================================================

def calculate_cch(ref_spikes, target_spikes, window_size, bin_size):
    """Memory-safe CCH calculation."""
    lags = []
    is_acg = np.array_equal(ref_spikes, target_spikes)
    target_spikes_sorted = np.sort(target_spikes)
    for ref_time in ref_spikes:
        start_time, end_time = ref_time - window_size, ref_time + window_size
        start_idx = np.searchsorted(target_spikes_sorted, start_time, side='left')
        end_idx = np.searchsorted(target_spikes_sorted, end_time, side='right')
        nearby_spikes = target_spikes_sorted[start_idx:end_idx]
        if nearby_spikes.size > 0:
            diffs = nearby_spikes - ref_time
            lags.extend(diffs)
    if is_acg:
        lags = [l for l in lags if l != 0]
    num_bins = int(2 * window_size / bin_size)
    bins = np.linspace(-window_size, window_size, num_bins + 1)
    counts, _ = np.histogram(lags, bins=bins)
    return counts, bins

def jitter_spikes(spikes, jitter_window):
    """Applies a uniform jitter to spike times."""
    shifts = (np.random.rand(len(spikes)) - 0.5) * jitter_window
    return spikes + shifts

def deconvolve_and_analyze(
    pre_spikes, post_spikes, window_size=0.1, bin_size=0.001,
    num_jitter=500, jitter_window=0.02, reg_factor=0.001, z_threshold=3.0
):
    """Performs the full CCH deconvolution analysis."""
    cch_raw, bins = calculate_cch(pre_spikes, post_spikes, window_size, bin_size)
    acg_raw, _ = calculate_cch(pre_spikes, pre_spikes, window_size, bin_size)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    jittered_cchs = [calculate_cch(pre_spikes, jitter_spikes(post_spikes, jitter_window), window_size, bin_size)[0] for _ in range(num_jitter)]
    jittered_cchs = np.array(jittered_cchs)
    cch_jitter_mean = np.mean(jittered_cchs, axis=0)
    cch_jitter_std = np.std(jittered_cchs, axis=0)
    cch_fft = np.fft.fft(cch_raw)
    acg_fft = np.fft.fft(acg_raw)
    alpha = np.mean(np.abs(acg_fft)) * reg_factor
    if alpha == 0: alpha = 1e-9
    kernel_fft = cch_fft / (acg_fft + alpha)
    kernel = np.fft.ifftshift(np.real(np.fft.ifft(kernel_fft)))
    baseline_win_size = int(len(kernel) * 0.25)
    kernel_baseline_data = np.concatenate([kernel[:baseline_win_size], kernel[-baseline_win_size:]])
    kernel_mean = np.mean(kernel_baseline_data)
    kernel_std = np.std(kernel_baseline_data)
    if kernel_std == 0: kernel_std = 1e-9
    kernel_z = (kernel - kernel_mean) / kernel_std
    significant_peaks = []
    peak_indices = np.where(np.abs(kernel_z) > z_threshold)[0]
    for idx in peak_indices:
        if np.abs(kernel_z[idx]) == np.max(np.abs(kernel_z[max(0, idx-2):min(len(kernel_z), idx+3)])):
            peak_lag_ms = bin_centers[idx] * 1000
            peak_val_gain = kernel[idx]
            peak_z_score = kernel_z[idx]
            if not any(np.isclose(peak_lag_ms, p['lag_ms']) for p in significant_peaks):
                significant_peaks.append({"lag_ms": peak_lag_ms, "gain": peak_val_gain, "z_score": peak_z_score})
    return {
        "bin_centers_ms": bin_centers * 1000, "cch_raw": cch_raw, "cch_jitter_mean": cch_jitter_mean,
        "cch_jitter_std": cch_jitter_std, "kernel": kernel, "kernel_mean": kernel_mean,
        "kernel_std": kernel_std, "significant_peaks": significant_peaks, "z_threshold": z_threshold
    }

def plot_deconvolution_results(results, epoch_idx, pre_id, post_id, output_dir):
    """Creates and saves a two-panel deconvolution plot."""
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
    fig.suptitle(f'Deconvolution: Epoch {epoch_idx}\nPresynaptic: {pre_id} -> Postsynaptic: {post_id}', fontsize=14)
    ax1.bar(results["bin_centers_ms"], results["cch_raw"], width=1.0, color='gray', label='Raw CCH')
    ax1.plot(results["bin_centers_ms"], results["cch_jitter_mean"], color='red', linestyle='-', label='Jitter Mean')
    ax1.fill_between(
        results["bin_centers_ms"],
        results["cch_jitter_mean"] - results["cch_jitter_std"],
        results["cch_jitter_mean"] + results["cch_jitter_std"],
        color='red', alpha=0.3, label='Jitter ±1 STD'
    )
    ax1.set_title('Raw Cross-Correlogram (CCH)'); ax1.set_ylabel('Spike Pair Count'); ax1.legend()
    ax2.plot(results["bin_centers_ms"], results["kernel"], color='blue', label='Deconvolved Kernel')
    ax2.axhline(results["kernel_mean"], color='black', linestyle='--', label='Kernel Baseline')
    ax2.axhline(results["kernel_mean"] + results["z_threshold"] * results["kernel_std"], color='green', linestyle=':', label=f'±{results["z_threshold"]} Z-Score')
    ax2.axhline(results["kernel_mean"] - results["z_threshold"] * results["kernel_std"], color='green', linestyle=':')
    for peak in results["significant_peaks"]:
        ax2.plot(peak['lag_ms'], peak['gain'], 'ro', markersize=8)
    ax2.set_title('Deconvolved Synaptic Kernel'); ax2.set_xlabel('Time Lag (ms)'); ax2.set_ylabel('Spike Gain'); ax2.legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    filename = f"Epoch{epoch_idx}_Deconv_{pre_id}_vs_{post_id}.tiff"
    fig.savefig(output_dir / filename, dpi=150, format='tiff')
    plt.close(fig)

# =============================================================================
# --- GUI AND FILE HANDLING ---
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
    """A single-purpose "worker" function to be run in parallel."""
    pre_neuron, post_neuron, epoch, comparison_name = args
    # When comparing a group to itself, don't compare a neuron to itself.
    if pre_neuron['cluster_id'] == post_neuron['cluster_id']:
        return None
    start_time, end_time = epoch['start_time_sec'], epoch['end_time_sec']
    pre_spikes_all = pre_neuron['spike_times_sec']
    pre_epoch_spikes = pre_spikes_all[(pre_spikes_all >= start_time) & (pre_spikes_all < end_time)]
    post_spikes_all = post_neuron['spike_times_sec']
    post_epoch_spikes = post_spikes_all[(post_spikes_all >= start_time) & (post_spikes_all < end_time)]
    if len(pre_epoch_spikes) < MIN_SPIKES_FOR_ANALYSIS or len(post_epoch_spikes) < MIN_SPIKES_FOR_ANALYSIS:
        return None
    analysis_results = deconvolve_and_analyze(pre_epoch_spikes, post_epoch_spikes)
    if analysis_results["significant_peaks"]:
        return {"analysis_results": analysis_results, "epoch_idx": epoch['epoch_index'], "pre_neuron": pre_neuron, "post_neuron": post_neuron, "comparison_name": comparison_name}
    return None

def main():
    print("--- Starting Robust Batch Deconvolution Analysis ---")
    
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
        
        if len(defined_groups) < 1: # Need at least one group for self-comparison
            print(f"Fewer than one selected group exists in {rec_name}. Skipping analysis."); continue
            
        # --- DIRECTED COMPARISON CHANGE: Use product for all ordered pairs ---
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

        significant_results = [res for res in results if res is not None]
        if not significant_results:
            print(f"No significant interactions found for {rec_name}."); continue
            
        print(f"\nFound {len(significant_results)} interactions for {rec_name}. Saving...")
        all_summary_data = []
        for res in tqdm(significant_results, desc=f"Saving for {rec_name}"):
            comp_dir = rec_output_dir / res['comparison_name']; comp_dir.mkdir(exist_ok=True)
            pre_n, post_n = res['pre_neuron'], res['post_neuron']
            pre_id = f"{pre_n['acronym']}_{pre_n['cell_type'].replace(' ', '')}_ID{pre_n['cluster_id']}"
            post_id = f"{post_n['acronym']}_{post_n['cell_type'].replace(' ', '')}_ID{post_n['cluster_id']}"
            plot_deconvolution_results(res['analysis_results'], res['epoch_idx'], pre_id, post_id, comp_dir)
            for peak in res['analysis_results']["significant_peaks"]:
                all_summary_data.append({'comparison': res['comparison_name'], 'epoch_index': res['epoch_idx'], 'pre_cluster_id': pre_n['cluster_id'], 'pre_region': pre_n['acronym'], 'pre_cell_type': pre_n['cell_type'], 'post_cluster_id': post_n['cluster_id'], 'post_region': post_n['acronym'], 'post_cell_type': post_n['cell_type'], 'peak_lag_ms': peak['lag_ms'], 'spike_gain': peak['gain'], 'z_score': peak['z_score']})
        
        report_df = pd.DataFrame(all_summary_data)
        report_path = rec_output_dir / f"summary_interactions_{rec_name}.csv"
        report_df.to_csv(report_path, index=False, float_format='%.4f')
        print(f"SUCCESS: Summary for {rec_name} saved to {report_path.name}")

    print("\n--- All Recordings Processed ---")

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    # You may need to install tqdm: pip install tqdm
    main()
    input("\nPress Enter to exit.")