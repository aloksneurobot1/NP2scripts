# -*- coding: utf-8 -*-
"""
SWR analysis pipeline using the neurosuite package.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
import multiprocessing
from scipy import signal

from neurosuite.data import io
from neurosuite.analysis import swr
from neurosuite import config

def process_channel_swr(ch_idx, lfp_data, fs, non_rem_periods_samples):
    """
    Processes a single channel for SWR detection.
    """
    lfp_ch = lfp_data[:, ch_idx].astype(np.float64)

    # Ripple detection
    ripple_filtered = swr.apply_fir_filter(lfp_ch, config.RIPPLE_FILTER_LOWCUT, config.RIPPLE_FILTER_HIGHCUT, fs)
    ripple_power = swr.calculate_instantaneous_power(ripple_filtered)

    baseline_mean, baseline_sd = swr.get_baseline_stats(ripple_power, fs, non_rem_periods_samples, config.RIPPLE_POWER_LP_CUTOFF)

    detection_threshold = baseline_mean + config.RIPPLE_DETECTION_SD_THRESHOLD * baseline_sd
    expansion_threshold = baseline_mean + config.RIPPLE_EXPANSION_SD_THRESHOLD * baseline_sd

    ripple_starts, ripple_ends = swr.detect_events(
        ripple_power, fs,
        threshold_high=detection_threshold,
        threshold_low=expansion_threshold,
        min_duration_ms=config.RIPPLE_MIN_DURATION_MS,
        merge_gap_ms=config.RIPPLE_MERGE_GAP_MS
    )

    ripple_events = []
    for start, end in zip(ripple_starts, ripple_ends):
        lfp_segment = ripple_filtered[start:end+1]
        power_segment = ripple_power[start:end+1]
        peak_power_idx, trough_idx = swr.find_event_features(lfp_segment, power_segment)
        if peak_power_idx != -1:
            ripple_events.append({
                'start_sample': start,
                'end_sample': end,
                'peak_sample': start + peak_power_idx,
                'trough_sample': start + trough_idx
            })

    return ch_idx, ripple_events

def main(lfp_path, channel_info_path, output_dir):
    """
    Main SWR analysis pipeline.
    """
    lfp_path = Path(lfp_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lfp_data, fs, _ = io.load_sglx_data(lfp_path)
    channel_info = io.load_channel_info(channel_info_path)

    # For now, we'll process the whole recording without sleep state filtering
    non_rem_periods_samples = []

    n_cores = multiprocessing.cpu_count() - 2 if multiprocessing.cpu_count() > 2 else 1

    results = Parallel(n_jobs=n_cores)(
        delayed(process_channel_swr)(ch_idx, lfp_data, fs, non_rem_periods_samples)
        for ch_idx in channel_info['global_channel_index']
    )

    all_ripple_events = {ch_idx: events for ch_idx, events in results}

    output_path = output_dir / f"{lfp_path.stem}_swr_events.npy"
    np.save(output_path, all_ripple_events)
    print(f"SWR events saved to {output_path}")

if __name__ == '__main__':
    # This is where you would typically parse command-line arguments
    # For now, we'll hardcode the paths for demonstration purposes
    # In a real application, you would use argparse or similar

    # Create dummy data for testing
    if not Path("dummy_data").exists():
        Path("dummy_data").mkdir()

    lfp_path = "dummy_data/lfp.lf.bin"
    meta_path = "dummy_data/lfp.lf.meta"
    channel_info_path = "dummy_data/channel_info.csv"
    output_dir = "results/swr"

    # Create dummy meta file
    with open(meta_path, 'w') as f:
        f.write("nSavedChans=4\n")
        f.write("imSampRate=1250\n")

    # Create dummy lfp data
    dummy_lfp = np.random.randn(1250 * 10, 4).astype('int16')
    dummy_lfp.tofile(lfp_path)

    # Create dummy channel info
    dummy_channel_info = pd.DataFrame({
        'global_channel_index': np.arange(4),
        'shank_index': [0, 0, 1, 1],
        'acronym': ['CA1', 'CA1', 'CA3', 'CA3'],
        'name': ['ch0', 'ch1', 'ch2', 'ch3']
    })
    dummy_channel_info.to_csv(channel_info_path, index=False)

    print("Running SWR pipeline with dummy data...")
    main(lfp_path, channel_info_path, output_dir)
    print("SWR pipeline test run complete.")