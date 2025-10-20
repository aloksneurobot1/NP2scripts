
import yaml
import json
import numpy as np
import pandas as pd
from scipy.signal import hilbert, find_peaks, butter, filtfilt
from tensorpac import Pac
import matplotlib.pyplot as plt
from utils.processing import bandpass_filter

def load_data_and_config():
    """Loads configuration, metadata, LFP, and CSD data."""
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    processed_dir = config['processed_directory']

    with open(f"{processed_dir}/lfp_metadata.json", 'r') as f:
        metadata = json.load(f)

    lfp_data = np.fromfile(f"{processed_dir}/lfp.bin", dtype=metadata['dtype']).reshape(-1, metadata['num_channels'])
    csd_data = np.fromfile(f"{processed_dir}/csd.bin", dtype=metadata['dtype']).reshape(-1, metadata['num_channels'])

    return config, metadata, lfp_data, csd_data

def get_theta_phase_locked_csd(lfp_signal, csd_data, fs, theta_band, ref_channel_idx=0, epoch_window_ms=(-200, 200)):
    """
    Calculates the average CSD locked to the troughs of the theta rhythm
    in a reference LFP channel.
    """
    # 1. Filter reference LFP for theta
    theta_lfp = bandpass_filter(lfp_signal[:, ref_channel_idx], theta_band[0], theta_band[1], fs)

    # 2. Find theta troughs
    # Troughs are peaks in the inverted signal
    trough_indices, _ = find_peaks(-theta_lfp, height=np.std(theta_lfp))

    # 3. Epoch CSD data around troughs
    epoch_window_samples = [int(t * fs / 1000) for t in epoch_window_ms]
    epochs = []
    for trough_idx in trough_indices:
        start = trough_idx + epoch_window_samples[0]
        end = trough_idx + epoch_window_samples[1]
        if start > 0 and end < csd_data.shape[0]:
            epochs.append(csd_data[start:end, :])

    # 4. Average the epochs
    if epochs:
        avg_csd_epoch = np.mean(epochs, axis=0)
    else:
        avg_csd_epoch = np.zeros((epoch_window_samples[1] - epoch_window_samples[0], csd_data.shape[1]))

    return avg_csd_epoch, epoch_window_ms

def get_gamma_amplitude_by_theta_phase(lfp_signal, csd_or_lfp_target, fs, theta_band, gamma_band, ref_channel_idx=0, target_channel_idx=0, n_bins=18):
    """
    Calculates the mean amplitude of a gamma band at different phases of the theta cycle.
    """
    # 1. Get theta phase from reference LFP
    theta_lfp = bandpass_filter(lfp_signal[:, ref_channel_idx], theta_band[0], theta_band[1], fs)
    theta_phase = np.angle(hilbert(theta_lfp))

    # 2. Get gamma amplitude from target signal
    gamma_target = bandpass_filter(csd_or_lfp_target[:, target_channel_idx], gamma_band[0], gamma_band[1], fs)
    gamma_amplitude = np.abs(hilbert(gamma_target))

    # 3. Bin gamma amplitude by theta phase
    phase_bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    binned_gamma_amplitude = []
    for i in range(n_bins):
        phase_start, phase_end = phase_bins[i], phase_bins[i+1]
        indices = (theta_phase >= phase_start) & (theta_phase < phase_end)
        binned_gamma_amplitude.append(np.mean(gamma_amplitude[indices]))

    return np.array(binned_gamma_amplitude), phase_bins[:-1]

def compute_pac(csd_data, fs, phase_freqs=(6, 12), amp_freqs=(25, 140), channels=[0, 1]):
    """
    Computes and visualizes a comodulogram for specified CSD channels using tensorpac.

    Args:
        csd_data (np.array): Time x Channels
        fs (int): Sampling frequency
        phase_freqs (tuple): Frequency band for phase
        amp_freqs (tuple): Frequency band for amplitude
        channels (list): A list of two channel indices to compare
    """
    # Tensorpac expects data in shape (n_epochs, n_channels, n_times)
    # Here we treat the whole recording as one epoch
    data_tp = csd_data.T[np.newaxis, :, :]

    p = Pac(idpac=(1, 2, 0), f_phase=phase_freqs, f_amp=amp_freqs)
    xpac = p.filterfit(fs, data_tp, n_jobs=-1)

    # Plotting
    plt.figure(figsize=(8, 6))
    p.comodulogram(xpac.mean(-1), title=f"PAC for CSD channels {channels[0]} and {channels[1]}", cmap='viridis')
    plt.xlabel("Phase Frequency (Hz)")
    plt.ylabel("Amplitude Frequency (Hz)")
    return p, xpac

if __name__ == '__main__':
    # This is a demonstration of how to use the functions.
    # The actual figure generation will be in a separate script.
    print("Loading data...")
    config, metadata, lfp_data, csd_data = load_data_and_config()

    fs = metadata['sampling_rate']
    theta_band = config['theta_band']

    print("\n1. Calculating Theta-Phase-Locked CSD...")
    # Using channel 0 as reference and a small window for demonstration
    avg_csd, window = get_theta_phase_locked_csd(lfp_data, csd_data, fs, theta_band, ref_channel_idx=0)
    print(f"  - Average CSD epoch shape: {avg_csd.shape}")

    print("\n2. Calculating Gamma Amplitude by Theta Phase...")
    slow_gamma_band = config['slow_gamma_band']
    gamma_amp, phase_bins = get_gamma_amplitude_by_theta_phase(lfp_data, csd_data, fs, theta_band, slow_gamma_band, ref_channel_idx=0, target_channel_idx=10)
    print(f"  - Calculated {len(gamma_amp)} amplitude points for {len(phase_bins)} phase bins.")

    print("\n3. Computing Phase-Amplitude Coupling (PAC)...")
    # Using placeholder channels 10 and 20 for CSD
    # This is computationally intensive and might take a while.
    # We will use a subset of data for demonstration
    subset_len = int(5 * fs) # 5 seconds of data
    p, xpac = compute_pac(csd_data[:subset_len, :], fs, channels=[10, 20])
    print("  - PAC computation complete. A figure should be generated.")

    plt.show() # Display the PAC figure
