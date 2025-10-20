
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the scripts directory to the Python path to import the analysis script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts')))

from spectral_analysis import (
    load_data_and_config,
    get_theta_phase_locked_csd,
    get_gamma_amplitude_by_theta_phase,
    compute_pac
)

def generate_ruiz_figures():
    """
    Generates and saves figures replicating the analyses from Ruiz et al. (2021).
    """
    print("Loading data for figure generation...")
    config, metadata, lfp_data, csd_data = load_data_and_config()
    fs = metadata['sampling_rate']

    figures_dir = os.path.dirname(__file__)

    # --- Figure 1 & 2: Theta-Phase-Locked CSD ---
    print("Generating Figure 1/2: Theta-Phase-Locked CSD plot...")
    avg_csd, window_ms = get_theta_phase_locked_csd(
        lfp_data, csd_data, fs, config['theta_band'], ref_channel_idx=0
    )
    time_vec = np.linspace(window_ms[0], window_ms[1], avg_csd.shape[0])

    plt.figure(figsize=(8, 10))
    plt.imshow(avg_csd.T, aspect='auto', origin='lower', cmap='seismic',
               extent=[time_vec[0], time_vec[-1], 0, avg_csd.shape[1]])
    plt.colorbar(label='CSD Amplitude')
    plt.xlabel('Time from Theta Trough (ms)')
    plt.ylabel('Channel (sorted by depth)')
    plt.title('Average Theta-Locked CSD')
    plt.axvline(0, color='k', linestyle='--')
    fig1_path = os.path.join(figures_dir, 'fig1_theta_locked_csd.png')
    plt.savefig(fig1_path)
    plt.close()
    print(f"  - Saved to {fig1_path}")

    # --- Figure 3: Gamma Amplitude Modulation by Theta Phase ---
    print("Generating Figure 3: Gamma Amplitude Modulation...")
    gamma_amp, phase_bins = get_gamma_amplitude_by_theta_phase(
        lfp_data, csd_data, fs, config['theta_band'],
        config['slow_gamma_band'], ref_channel_idx=0, target_channel_idx=10
    )
    phase_degrees = np.rad2deg(phase_bins)

    plt.figure(figsize=(8, 5))
    plt.bar(phase_degrees, gamma_amp, width=np.diff(phase_degrees)[0])
    plt.xlabel('Theta Phase (degrees)')
    plt.ylabel(f"Slow Gamma ({config['slow_gamma_band'][0]}-{config['slow_gamma_band'][1]} Hz) Amplitude")
    plt.title('Gamma Amplitude Modulation by Theta Phase')
    plt.xticks([-180, -90, 0, 90, 180])
    fig3_path = os.path.join(figures_dir, 'fig3_gamma_amp_modulation.png')
    plt.savefig(fig3_path)
    plt.close()
    print(f"  - Saved to {fig3_path}")

    # --- Figure 4: Phase-Amplitude Coupling (PAC) Comodulogram ---
    print("Generating Figure 4: PAC Comodulogram...")
    # This is computationally intensive. Use a subset for speed.
    subset_len = int(5 * fs) # 5 seconds
    pac_computer, _ = compute_pac(
        csd_data[:subset_len, :], fs,
        channels=[10, 20] # Example channels
    )
    fig4_path = os.path.join(figures_dir, 'fig4_pac_comodulogram.png')
    pac_computer.savefig(fig4_path, dpi=300)
    plt.close()
    print(f"  - Saved to {fig4_path}")

    print("\nAll figures have been generated and saved.")

if __name__ == '__main__':
    # Ensure the script is run from the correct directory or paths are correct
    # This might require adjusting paths based on execution context
    generate_ruiz_figures()
