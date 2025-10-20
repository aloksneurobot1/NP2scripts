
import yaml
import json
import numpy as np
import pandas as pd
import spikeinterface.extractors as se
import spikeinterface as si
import spikeinterface.qualitymetrics as qm
from scipy.signal import hilbert
from sklearn.cluster import KMeans
from utils.processing import bandpass_filter

def load_sorting_and_lfp(config):
    """Loads Kilosort output and preprocessed LFP data."""
    sorting = se.read_kilosort(config['kilosort_output_directory'])

    processed_dir = config['processed_directory']
    with open(f"{processed_dir}/lfp_metadata.json", 'r') as f:
        metadata = json.load(f)
    lfp_data = np.fromfile(f"{processed_dir}/lfp.bin", dtype=metadata['dtype']).reshape(-1, metadata['num_channels'])

    # Load raw recording for waveform extraction
    recording = se.read_binary(config['raw_recording_file'], sampling_frequency=metadata['sampling_rate'],
                               num_channels=metadata['num_channels'], dtype=metadata['dtype'])

    return sorting, lfp_data, recording, metadata['sampling_rate']

def extract_waveform_features(sorting, recording):
    """
    Extracts trough-to-peak duration and spike half-width for each unit.
    """
    print("Extracting waveforms. This might take a while...")
    we = si.extract_waveforms(recording, sorting, folder='waveforms_temp',
                              overwrite=True, progress_bar=True, n_jobs=-1)

    print("Computing waveform features...")
    features = pd.DataFrame(index=sorting.unit_ids)
    features['trough_to_peak'] = qm.compute_trough_to_peak(we, ms_before=1., ms_after=1.5)
    features['half_width'] = qm.compute_half_width(we, ms_before=1., ms_after=1.5)

    return features

def classify_cell_types(features):
    """Classifies units into PYR and IN based on waveform features using K-Means."""
    kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(features)
    # Assuming the cluster with the larger trough-to-peak is pyramidal
    cluster_centers = kmeans.cluster_centers_
    if cluster_centers[0, 0] > cluster_centers[1, 0]:
        pyr_cluster = 0
    else:
        pyr_cluster = 1

    labels = pd.Series(['PYR' if label == pyr_cluster else 'IN' for label in kmeans.labels_], index=features.index, name='cell_type')
    return labels

def calculate_spike_phase_locking(spike_times, lfp_signal, fs, freq_band):
    """
    Calculates the Phase Locking Value (PLV) and preferred phase for a set of spikes.
    """
    lfp_filtered = bandpass_filter(lfp_signal, freq_band[0], freq_band[1], fs)
    lfp_phase = np.angle(hilbert(lfp_filtered))

    spike_indices = (spike_times * fs).astype(int)
    spike_indices = spike_indices[spike_indices < len(lfp_phase)]
    spike_phases = lfp_phase[spike_indices]

    if len(spike_phases) == 0:
        return 0.0, np.nan

    mean_vector = np.mean(np.exp(1j * spike_phases))
    plv = np.abs(mean_vector)
    preferred_phase = np.angle(mean_vector)

    return plv, preferred_phase

def load_behavioral_data(config, timestamps):
    """Loads and synchronizes behavioral data from a CSV file."""
    print(f"Loading behavioral data from {config['behavioral_data_file']}...")
    behavior_df = pd.read_csv(config['behavioral_data_file'])
    # Basic synchronization: assuming the behavior CSV has a 'timestamps' column
    # and we can align it to the LFP timestamps. This might need to be more sophisticated.
    behavior_df = behavior_df.set_index('timestamps').reindex(timestamps, method='nearest')
    return behavior_df

def segment_spikes_by_speed(sorting, behavior, speed_threshold=5):
    """Segments spike trains based on whether speed is above or below a threshold."""
    is_moving = behavior['speed'] > speed_threshold

    moving_spikes = {}
    still_spikes = {}

    for unit_id in sorting.unit_ids:
        spike_times = sorting.get_unit_spike_train(unit_id, segment_index=0) / sorting.get_sampling_frequency()
        spike_behavior_indices = np.searchsorted(behavior.index, spike_times)
        spike_behavior_indices[spike_behavior_indices >= len(is_moving)] = len(is_moving) - 1

        moving_mask = is_moving.iloc[spike_behavior_indices].values

        moving_spikes[unit_id] = spike_times[moving_mask]
        still_spikes[unit_id] = spike_times[~moving_mask]

    return moving_spikes, still_spikes

if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    try:
        sorting, lfp_data, recording, fs = load_sorting_and_lfp(config)

        print("\n--- Cell-Type Classification ---")
        waveform_features = extract_waveform_features(sorting, recording)
        cell_types = classify_cell_types(waveform_features)
        print("Classification results:")
        print(cell_types.value_counts())

        print("\n--- Spike-Phase Locking ---")
        example_unit_id = sorting.unit_ids[0]
        spike_times_sec = sorting.get_unit_spike_train(example_unit_id, segment_index=0) / fs

        plv, pref_phase = calculate_spike_phase_locking(spike_times_sec, lfp_data[:, 0], fs, config['theta_band'])
        print(f"Unit {example_unit_id}: Theta PLV = {plv:.3f}, Preferred Phase = {np.rad2deg(pref_phase):.1f} degrees")

        print("\n--- Behavioral Segmentation ---")
        lfp_timestamps = np.arange(lfp_data.shape[0]) / fs
        behavior_df = load_behavioral_data(config, lfp_timestamps)
        moving_spikes, still_spikes = segment_spikes_by_speed(sorting, behavior_df)
        print(f"Spikes for unit {example_unit_id} while moving: {len(moving_spikes[example_unit_id])}")
        print(f"Spikes for unit {example_unit_id} while still: {len(still_spikes[example_unit_id])}")

    except (FileNotFoundError, IndexError, KeyError) as e:
        print("\nCould not run analysis pipeline due to missing data or incorrect config.")
        print("Please ensure your 'config.yaml' points to valid data files.")
        print(f"Error: {e}")
