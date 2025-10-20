# src/ca3/analysis.py
#
# This module provides functions for analyzing and visualizing the results
# of the CA3 microcircuit simulation.

from brian2 import *
import numpy as np
from scipy.signal import welch

def calculate_lfp(state_monitor, populations):
    """
    Calculates the Local Field Potential (LFP) from the synaptic currents
    of the multi-compartmental CA3 pyramidal neuron.

    The LFP is approximated as the sum of the absolute values of the synaptic
    currents across all compartments of the neuron.
    """
    g_ampa = state_monitor.g_ampa
    g_gaba = state_monitor.g_gaba
    E_ampa = populations['CA3e'].E_ampa[0]
    E_gaba = populations['CA3e'].E_gaba[0]
    v = state_monitor.v

    # Calculate total synaptic current
    I_syn = g_ampa * (E_ampa - v) + g_gaba * (E_gaba - v)
    # Sum across all compartments to get a single LFP trace
    lfp = np.sum(I_syn, axis=0)

    return lfp

def plot_psd(lfp, dt, ax):
    """
    Plots the Power Spectral Density (PSD) of the LFP using Welch's method.
    """
    fs = 1 / dt
    # Use Welch's method for a smoother periodogram
    f, psd = welch(lfp, fs=fs, nperseg=min(len(lfp), 1024))

    ax.semilogy(f, psd)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power')
    ax.set_title('LFP Power Spectral Density')
    ax.set_xlim(0, 300)

def calculate_csd(ca3_monitor, ca3_neuron):
    """
    Calculates the Current Source Density (CSD).

    In this simulation, the CSD is approximated by the transmembrane current (Im)
    recorded from each compartment of the pyramidal neuron.
    """
    if 'Im' not in ca3_monitor.variables:
        raise ValueError("StateMonitor must record 'Im' to calculate CSD.")
    return ca3_monitor.Im

def plot_csd(csd, times, compartment_depths, ax):
    """
    Plots the Current Source Density (CSD) as a heatmap, showing current sinks
    and sources across the neuron's morphology over time.
    """
    im = ax.imshow(csd, extent=[times[0]/ms, times[-1]/ms,
                                compartment_depths[-1]/um,
                                compartment_depths[0]/um],
                   aspect='auto', origin='lower', cmap='viridis')
    ax.set_title('Current Source Density (CSD)')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Depth (um)')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Current (A/m^2)')

def detect_pre_swr_state(lfp_window, dt, ripple_band=(150*Hz, 250*Hz), threshold=1e-24):
    """
    Detects a pre-SWR state by checking if the power in the ripple band of
    the LFP exceeds a given threshold.
    """
    # Welch's method requires a minimum number of data points.
    if len(lfp_window) < 256:
        return False

    fs = 1 / dt
    f, psd = welch(lfp_window, fs=fs, nperseg=len(lfp_window))

    # Calculate the total power within the ripple frequency band.
    ripple_indices = np.where((f >= ripple_band[0]) & (f <= ripple_band[1]))
    ripple_power = np.sum(psd[ripple_indices])

    return ripple_power > threshold
