
import yaml
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from utils.processing import bandpass_filter
from spike_analysis import load_sorting_and_lfp, calculate_spike_phase_locking

def detect_swr_events(lfp_signal, fs, ripple_band, detection_channel=0, threshold_sd=5, min_duration_ms=20, max_duration_ms=250):
    """Detects Sharp-Wave Ripple (SWR) events from an LFP signal."""
    ripple_filtered = bandpass_filter(lfp_signal[:, detection_channel], ripple_band[0], ripple_band[1], fs)
    window_size = int(0.01 * fs)
    rms = np.array([np.sqrt(np.mean(np.square(ripple_filtered[i:i+window_size]))) for i in range(len(ripple_filtered) - window_size)])
    threshold = np.mean(rms) + threshold_sd * np.std(rms)
    peak_indices, _ = find_peaks(rms, height=threshold)

    swr_events = []
    for peak in peak_indices:
        start_idx = peak
        while start_idx > 0 and rms[start_idx] > np.mean(rms):
            start_idx -= 1
        end_idx = peak
        while end_idx < len(rms) - 1 and rms[end_idx] > np.mean(rms):
            end_idx += 1
        duration = (end_idx - start_idx) / fs * 1000
        if min_duration_ms <= duration <= max_duration_ms:
            swr_events.append({
                'start_time': start_idx / fs, 'peak_time': peak / fs, 'end_time': end_idx / fs,
                'duration': duration, 'peak_amplitude': rms[peak]
            })
    return pd.DataFrame(swr_events)

def get_swr_triggered_csd(csd_data, fs, swr_events, window_ms=(-100, 100)):
    """Computes the average CSD map triggered by SWR peak times."""
    peak_indices = (swr_events['peak_time'] * fs).astype(int)
    window_samples = [int(t * fs / 1000) for t in window_ms]

    epochs = [csd_data[peak + window_samples[0] : peak + window_samples[1], :]
              for peak in peak_indices if peak + window_samples[0] > 0 and peak + window_samples[1] < len(csd_data)]

    return np.mean(epochs, axis=0) if epochs else np.zeros((window_samples[1] - window_samples[0], csd_data.shape[1]))

def analyze_ripple_locked_firing(sorting, lfp_data, fs, swr_events, ripple_band, epoch_start_time=0):
    """Calculates ripple-PLV for each neuron during detected SWRs within a specific epoch."""
    ripple_plvs = {}
    ripple_epochs = []
    spike_times_in_ripples = {unit_id: [] for unit_id in sorting.unit_ids}

    for _, event in swr_events.iterrows():
        start_idx = int(event['start_time'] * fs)
        end_idx = int(event['end_time'] * fs)
        ripple_epochs.append(lfp_data[start_idx:end_idx, :])

        for unit_id in sorting.unit_ids:
            st = sorting.get_unit_spike_train(unit_id, segment_index=0) / sorting.get_sampling_frequency()
            st_in_epoch = st[(st >= epoch_start_time + event['start_time']) & (st <= epoch_start_time + event['end_time'])]
            relative_start_time = len(np.concatenate(ripple_epochs[:-1])) / fs if ripple_epochs else 0
            spikes_in_event_relative = st_in_epoch - (epoch_start_time + event['start_time']) + relative_start_time
            spike_times_in_ripples[unit_id].extend(spikes_in_event_relative)

    concatenated_lfp = np.concatenate(ripple_epochs) if ripple_epochs else np.array([])
    if concatenated_lfp.size == 0:
        return pd.Series(np.nan, index=sorting.unit_ids, name='ripple_plv')

    for unit_id in sorting.unit_ids:
        spikes = np.array(spike_times_in_ripples[unit_id])
        if len(spikes) > 10:
            plv, _ = calculate_spike_phase_locking(spikes, concatenated_lfp[:, 0], fs, ripple_band)
            ripple_plvs[unit_id] = plv
        else:
            ripple_plvs[unit_id] = np.nan

    return pd.Series(ripple_plvs, name='ripple_plv')

def get_epochs(lfp_data, fs):
    """Defines pre- and post-interaction epochs. Placeholder: splits data in half."""
    midpoint_idx = int(lfp_data.shape[0] / 2)
    pre_epoch = {'start': 0, 'end': midpoint_idx / fs}
    post_epoch = {'start': midpoint_idx / fs, 'end': lfp_data.shape[0] / fs}
    return pre_epoch, post_epoch

if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    try:
        sorting, lfp_data, recording, fs = load_sorting_and_lfp(config)
        csd_data = np.fromfile(f"{config['processed_directory']}/csd.bin", dtype=lfp_data.dtype).reshape(-1, lfp_data.shape[1])

        pre_epoch, post_epoch = get_epochs(lfp_data, fs)
        pre_start_idx, pre_end_idx = int(pre_epoch['start'] * fs), int(pre_epoch['end'] * fs)
        post_start_idx, post_end_idx = int(post_epoch['start'] * fs), int(post_epoch['end'] * fs)

        lfp_pre, csd_pre = lfp_data[pre_start_idx:pre_end_idx], csd_data[pre_start_idx:pre_end_idx]
        lfp_post, csd_post = lfp_data[post_start_idx:post_end_idx], csd_data[post_start_idx:post_end_idx]

        swr_pre = detect_swr_events(lfp_pre, fs, config['ripple_band'])
        swr_post = detect_swr_events(lfp_post, fs, config['ripple_band'])

        avg_csd_pre = get_swr_triggered_csd(csd_pre, fs, swr_pre)
        avg_csd_post = get_swr_triggered_csd(csd_post, fs, swr_post)

        plv_pre = analyze_ripple_locked_firing(sorting, lfp_pre, fs, swr_pre, config['ripple_band'], epoch_start_time=pre_epoch['start'])
        plv_post = analyze_ripple_locked_firing(sorting, lfp_post, fs, swr_post, config['ripple_band'], epoch_start_time=post_epoch['start'])

        # --- Plotting ---
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        csd_diff = avg_csd_post - avg_csd_pre
        plt.imshow(csd_diff.T, aspect='auto', origin='lower', cmap='seismic', vmin=-np.max(np.abs(csd_diff)), vmax=np.max(np.abs(csd_diff)))
        plt.title('SWR-CSD Difference (Post - Pre)')
        plt.xlabel('Time from SWR Peak (ms)')
        plt.ylabel('Channel')

        plt.subplot(1, 2, 2)
        comparison_df = pd.DataFrame({'PLV_pre': plv_pre, 'PLV_post': plv_post}).dropna()
        plt.scatter(comparison_df['PLV_pre'], comparison_df['PLV_post'])
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlabel('Pre-interaction PLV')
        plt.ylabel('Post-interaction PLV')
        plt.title('Ripple-Locked Firing Plasticity')
        plt.axis('square')
        plt.tight_layout()
        plt.savefig('figures/plasticity_comparison.png')
        plt.show()

    except (FileNotFoundError, IndexError, KeyError) as e:
        print(f"\nCould not run analysis pipeline due to missing data or incorrect config: {e}")
