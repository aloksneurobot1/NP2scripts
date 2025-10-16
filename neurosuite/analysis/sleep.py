# -*- coding: utf-8 -*-
"""
Functions for sleep state scoring.
"""
import numpy as np
from scipy import signal
from scipy.stats import skew, zscore
from sklearn.decomposition import PCA, FastICA

def compute_pca(data):
    """
    Performs PCA on z-scored spectrogram data.
    """
    if data is None or data.size == 0 or data.shape[0] < 2 or data.shape[1] < 2:
        return None

    zscored_data = zscore(data, axis=1)
    if np.any(np.isnan(zscored_data)):
        zscored_data = np.nan_to_num(zscored_data)

    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(zscored_data.T)
    return pc1.squeeze()

def compute_theta_dominance(spectrogram, frequencies):
    """
    Computes theta dominance from spectrogram.
    """
    if spectrogram is None or frequencies is None or spectrogram.size == 0 or frequencies.size == 0:
        return None

    theta_band = (frequencies >= 5) & (frequencies <= 10)
    total_band = (frequencies >= 2) & (frequencies <= 16)

    if not np.any(theta_band) or not np.any(total_band):
        return np.full(spectrogram.shape[1], np.nan)

    epsilon = 1e-12
    theta_power = np.nanmean(spectrogram[theta_band, :], axis=0)
    total_power = np.nanmean(spectrogram[total_band, :], axis=0)

    valid_mask = total_power > epsilon
    theta_dominance = np.full_like(theta_power, np.nan)
    theta_dominance[valid_mask] = theta_power[valid_mask] / total_power[valid_mask]

    return theta_dominance

def compute_emg_from_lfp(lfp_data, fs, window_size_sec, step_size_sec, n_ica_components=10, fit_duration_sec=240, n_channels_for_ica=48):
    """
    Estimates EMG from LFP data using ICA.
    """
    if lfp_data is None or lfp_data.ndim != 2 or lfp_data.shape[1] < n_ica_components:
        return None, None, None

    n_samples_fit = int(fit_duration_sec * fs)
    lfp_subset = lfp_data[:n_samples_fit, :n_channels_for_ica]

    ica = FastICA(n_components=n_ica_components, random_state=42, whiten='unit-variance')
    ica_sources = ica.fit_transform(lfp_subset)

    skewness = np.abs(skew(ica_sources, axis=0))
    emg_ic_index = np.argmax(skewness)

    full_source_signals = ica.transform(lfp_data[:, :n_channels_for_ica])
    emg_signal_raw = np.abs(full_source_signals[:, emg_ic_index])

    window_samples = int(window_size_sec * fs)
    step_samples = int(step_size_sec * fs)
    num_segments = (len(emg_signal_raw) - window_samples) // step_samples + 1

    emg_segmented = np.zeros(num_segments)
    for i in range(num_segments):
        start = i * step_samples
        end = start + window_samples
        emg_segmented[i] = np.mean(emg_signal_raw[start:end])

    return emg_segmented, ica, full_source_signals

def score_sleep_states(pc1, theta_dominance, emg, times, step_size, buffer_size=5, emg_thresh=None, speed_data=None, speed_thresh=1.5, use_emg=True):
    """
    Scores sleep states based on provided metrics.
    """
    n_points = len(pc1)
    sleep_states = np.zeros(n_points, dtype=int)

    nrem_thresh = np.nanpercentile(pc1, 75)
    rem_thresh = np.nanpercentile(theta_dominance, 75)
    if emg_thresh is None and use_emg:
        emg_thresh = np.nanpercentile(emg, 25)

    current_state = 0
    nrem_count, rem_count, awake_count = 0, 0, 0

    for i in range(n_points):
        is_moving = speed_data is not None and speed_data[i] > speed_thresh

        pc1_above = pc1[i] > nrem_thresh
        theta_above = theta_dominance[i] > rem_thresh
        emg_low = not use_emg or (emg[i] < emg_thresh)

        if current_state == 0:  # Awake
            if pc1_above and emg_low and not is_moving:
                nrem_count += 1
            else:
                nrem_count = 0
            if theta_above and emg_low and not is_moving:
                rem_count += 1
            else:
                rem_count = 0
            if nrem_count >= buffer_size:
                current_state = 1
            elif rem_count >= buffer_size:
                current_state = 2
        elif current_state == 1:  # NREM
            if not pc1_above or not emg_low or is_moving:
                awake_count += 1
            else:
                awake_count = 0
            if awake_count >= buffer_size:
                current_state = 0
        elif current_state == 2:  # REM
            if not theta_above or not emg_low or is_moving:
                awake_count += 1
            else:
                awake_count = 0
            if awake_count >= buffer_size:
                current_state = 0

        sleep_states[i] = current_state

    return sleep_states