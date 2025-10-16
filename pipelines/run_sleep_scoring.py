# -*- coding: utf-8 -*-
"""
Sleep scoring pipeline using the neurosuite package.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import signal

from neurosuite.data import io
from neurosuite.analysis import sleep
from neurosuite.utils import processing
from neurosuite import config

def main(lfp_path, channel_info_path, output_dir, speed_path=None):
    """
    Main sleep scoring pipeline.
    """
    lfp_path = Path(lfp_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lfp_data, fs, _ = io.load_sglx_data(lfp_path)

    # Downsample and filter
    downsampling_factor = int(round(fs / config.SLEEP_TARGET_FS))
    lfp_downsampled = signal.decimate(lfp_data, downsampling_factor, axis=0)
    fs_proc = fs / downsampling_factor

    lfp_filtered = processing.apply_fir_filter(lfp_downsampled, None, config.SLEEP_LFP_CUTOFF, fs_proc, pass_zero='lowpass')

    # Averaged LFP for spectrogram
    avg_lfp = np.mean(lfp_filtered[:, -config.SLEEP_NUM_CHANNELS:], axis=1)

    spectrogram, freqs, times = processing.compute_spectrogram(
        avg_lfp, fs_proc, config.SLEEP_SPECTROGRAM_WINDOW_SEC, config.SLEEP_SPECTROGRAM_STEP_SEC
    )

    pc1 = sleep.compute_pca(spectrogram)
    theta_dominance = sleep.compute_theta_dominance(spectrogram, freqs)

    emg = np.zeros(len(times))
    if config.SLEEP_CALCULATE_EMG:
        emg, _, _ = sleep.compute_emg_from_lfp(
            lfp_filtered, fs_proc,
            config.SLEEP_SPECTROGRAM_WINDOW_SEC, config.SLEEP_SPECTROGRAM_STEP_SEC,
            n_ica_components=config.SLEEP_ICA_N_COMPONENTS,
            fit_duration_sec=config.SLEEP_ICA_FIT_DURATION_SEC,
            n_channels_for_ica=config.SLEEP_ICA_NUM_CHANNELS
        )

    speed_data = None
    if speed_path:
        speed_df = pd.read_csv(speed_path)
        speed_values = speed_df['speed'].values
        speed_time = np.arange(len(speed_values)) / config.SPEED_VIDEO_FPS
        speed_data = np.interp(times, speed_time, speed_values, left=np.nan, right=np.nan)

    sleep_states = sleep.score_sleep_states(
        pc1, theta_dominance, emg, times,
        config.SLEEP_SPECTROGRAM_STEP_SEC,
        buffer_size=config.SLEEP_SCORING_BUFFER_STEPS,
        emg_thresh=config.SLEEP_EMG_THRESHOLD_FIXED,
        speed_data=speed_data,
        speed_thresh=config.SPEED_THRESHOLD_CM_S,
        use_emg=config.SLEEP_CALCULATE_EMG
    )

    # Save results
    output_states_path = output_dir / f"{lfp_path.stem}_sleep_states.npy"
    output_times_path = output_dir / f"{lfp_path.stem}_sleep_times.npy"
    np.save(output_states_path, sleep_states)
    np.save(output_times_path, times)
    print(f"Sleep states saved to {output_states_path}")
    print(f"Sleep times saved to {output_times_path}")

if __name__ == '__main__':
    # Create dummy data for testing
    if not Path("dummy_data").exists():
        Path("dummy_data").mkdir()

    lfp_path = "dummy_data/lfp_sleep.lf.bin"
    meta_path = "dummy_data/lfp_sleep.lf.meta"
    channel_info_path = "dummy_data/channel_info_sleep.csv"
    output_dir = "results/sleep"

    # Create dummy meta file
    with open(meta_path, 'w') as f:
        f.write("nSavedChans=48\n")
        f.write("imSampRate=2500\n")

    # Create dummy lfp data
    dummy_lfp = np.random.randn(2500 * 120, 48).astype('int16')
    dummy_lfp.tofile(lfp_path)

    # Create dummy channel info
    dummy_channel_info = pd.DataFrame({
        'global_channel_index': np.arange(48),
        'shank_index': [0] * 48,
        'acronym': ['CTX'] * 48,
        'name': [f'ch{i}' for i in range(48)]
    })
    dummy_channel_info.to_csv(channel_info_path, index=False)

    print("Running sleep scoring pipeline with dummy data...")
    main(lfp_path, channel_info_path, output_dir)
    print("Sleep scoring pipeline test run complete.")