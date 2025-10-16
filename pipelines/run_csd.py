# -*- coding: utf-8 -*-
"""
CSD analysis pipeline using the neurosuite package.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from neurosuite.data import io
from neurosuite.analysis import csd
from neurosuite import config

def main(lfp_path, channel_info_path, timestamps_path, output_dir):
    """
    Main CSD analysis pipeline.
    """
    lfp_path = Path(lfp_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lfp_data, fs_orig, meta = io.load_sglx_data(lfp_path)
    channel_info = io.load_channel_info(channel_info_path)
    timestamps = io.load_timestamps(timestamps_path)

    uv_scale_factor = csd.get_voltage_scaling_factor(meta)
    if uv_scale_factor is not None:
        lfp_data = lfp_data.astype(np.float32) * uv_scale_factor

    unique_shanks = sorted(channel_info['shank_index'].unique())

    for epoch_info in timestamps.item().get('EpochFrameData', []):
        start_sec = epoch_info['start_time_sec']
        end_sec = epoch_info['end_time_sec']
        epoch_idx = epoch_info['epoch_index']

        start_sample = int(start_sec * fs_orig)
        end_sample = int(end_sec * fs_orig)

        lfp_epoch = lfp_data[start_sample:end_sample, :]

        fig, axs = plt.subplots(len(unique_shanks), 1, figsize=(10, 5 * len(unique_shanks)), sharex=True)
        if len(unique_shanks) == 1:
            axs = [axs]

        for i, shank_id in enumerate(unique_shanks):
            shank_channels = channel_info[channel_info['shank_index'] == shank_id]
            shank_indices = shank_channels['global_channel_index'].values
            shank_coords = shank_channels['ycoord_on_shank_um'].values

            lfp_shank = lfp_epoch[:, shank_indices]

            lfp_processed, fs_proc = csd.preprocess_lfp_for_csd(
                lfp_shank, fs_orig,
                config.LFP_BAND_LOWCUT_CSD, config.LFP_BAND_HIGHCUT_CSD,
                config.NUMTAPS_CSD_FILTER, config.TARGET_FS_CSD
            )

            kcsd_params = {
                'sigma': config.CSD_SIGMA_CONDUCTIVITY,
                'lambdas': config.KCSD_LAMBDAS_CV,
                'Rs': config.KCSD_RS_CV_UM
            }

            csd_data, csd_pos, _ = csd.run_kcsd_analysis(lfp_processed, shank_coords, fs_proc, **kcsd_params)

            # Plotting
            im = axs[i].pcolormesh(
                np.arange(csd_data.shape[0]) / fs_proc,
                csd_pos,
                csd_data.T,
                cmap='RdBu_r',
                shading='gouraud'
            )
            axs[i].set_title(f"Shank {shank_id}")
            axs[i].set_ylabel("Depth (um)")
            fig.colorbar(im, ax=axs[i], label="CSD (uA/mm^3)")

        axs[-1].set_xlabel("Time (s)")
        fig.suptitle(f"CSD for Epoch {epoch_idx}")
        output_path = output_dir / f"{lfp_path.stem}_epoch_{epoch_idx}_csd.png"
        fig.savefig(output_path)
        plt.close(fig)
        print(f"CSD plot for epoch {epoch_idx} saved to {output_path}")

if __name__ == '__main__':
    # Create dummy data for testing
    if not Path("dummy_data").exists():
        Path("dummy_data").mkdir()

    lfp_path = "dummy_data/lfp_csd.lf.bin"
    meta_path = "dummy_data/lfp_csd.lf.meta"
    channel_info_path = "dummy_data/channel_info_csd.csv"
    timestamps_path = "dummy_data/timestamps_csd.npy"
    output_dir = "results/csd"

    # Create dummy meta file
    with open(meta_path, 'w') as f:
        f.write("nSavedChans=8\n")
        f.write("imSampRate=2500\n")
        f.write("imAiRangeMax=5.0\n")
        f.write("imMaxInt=32767\n")

    # Create dummy lfp data
    dummy_lfp = np.random.randn(2500 * 20, 8).astype('int16')
    dummy_lfp.tofile(lfp_path)

    # Create dummy channel info
    dummy_channel_info = pd.DataFrame({
        'global_channel_index': np.arange(8),
        'shank_index': [0, 0, 0, 0, 1, 1, 1, 1],
        'ycoord_on_shank_um': [0, 20, 40, 60, 0, 20, 40, 60],
        'acronym': ['CA1'] * 8,
        'name': [f'ch{i}' for i in range(8)]
    })
    dummy_channel_info.to_csv(channel_info_path, index=False)

    # Create dummy timestamps
    dummy_timestamps = {
        'EpochFrameData': [
            {'epoch_index': 0, 'start_time_sec': 1, 'end_time_sec': 5},
            {'epoch_index': 1, 'start_time_sec': 10, 'end_time_sec': 15}
        ]
    }
    np.save(timestamps_path, dummy_timestamps)

    print("Running CSD pipeline with dummy data...")
    main(lfp_path, channel_info_path, timestamps_path, output_dir)
    print("CSD pipeline test run complete.")