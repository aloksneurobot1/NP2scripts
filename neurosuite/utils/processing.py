# -*- coding: utf-8 -*-
"""
Signal processing utilities for electrophysiology data.
"""
import numpy as np
from scipy import signal

def apply_fir_filter(data, lowcut, highcut, fs, numtaps=101, pass_zero=False):
    """
    Applies a zero-lag linear phase FIR bandpass filter.
    """
    if data is None or data.size == 0:
        return None

    data_len = data.shape[0]
    if numtaps >= data_len:
        numtaps = data_len - 1 if data_len > 1 else 1
        if numtaps % 2 == 0: numtaps -= 1
        if numtaps < 3:
             return data

    try:
        if lowcut is None and highcut is None:
             return data
        elif lowcut is None: # Lowpass
            b = signal.firwin(numtaps, highcut, fs=fs, pass_zero=True, window='hamming')
        elif highcut is None: # Highpass
             b = signal.firwin(numtaps, lowcut, fs=fs, pass_zero=False, window='hamming')
        else: # Bandpass
            if lowcut >= highcut:
                 return None
            b = signal.firwin(numtaps, [lowcut, highcut], fs=fs, pass_zero=pass_zero, window='hamming')

        if not np.issubdtype(data.dtype, np.floating):
             data = data.astype(np.float64)
        filtered_data = signal.filtfilt(b, 1, data, axis=0)
        return filtered_data
    except Exception as e:
        print(f"Error during FIR filtering: {e}")
        return None

def calculate_instantaneous_power(data):
    """Calculates instantaneous power using Hilbert transform."""
    if data is None or data.size == 0:
        return None
    try:
        if not np.issubdtype(data.dtype, np.floating):
             data = data.astype(np.float64)
        analytic_signal = signal.hilbert(data, axis=-1)
        amplitude_envelope = np.abs(analytic_signal)
        power = amplitude_envelope**2
        return power
    except Exception as e:
        print(f"Error calculating instantaneous power: {e}")
        return None

def compute_spectrogram(data, fs, window_size_sec, step_size_sec):
    """
    Computes spectrogram using scipy.signal.spectrogram.
    """
    if data is None or data.ndim != 1 or data.size == 0:
        return None, None, None

    try:
        window_samples = int(round(window_size_sec * fs))
        step_samples = int(round(step_size_sec * fs))

        if window_samples <= 0 or step_samples <= 0 or window_samples > len(data):
             return None, None, None

        noverlap = max(0, window_samples - step_samples)
        win = signal.windows.hann(window_samples)

        frequencies, times, spectrogram = signal.spectrogram(
            data.astype(np.float64),
            fs=fs,
            window=win,
            nperseg=window_samples,
            noverlap=noverlap,
            scaling='density',
            mode='psd'
        )
        return spectrogram, frequencies, times
    except Exception as e:
        print(f"Error computing spectrogram: {e}")
        return None, None, None