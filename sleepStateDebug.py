# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 18:51:11 2025

@author: HT_bo
"""
# import scipy
# print(scipy.__version__)
import numpy as np
from scipy import signal
from scipy.signal import ShortTimeFFT
from sklearn.decomposition import PCA
from scipy.stats import zscore
from tkinter import Tk
from tkinter import filedialog
from pathlib import Path
from DemoReadSGLXData.readSGLX import readMeta
import os
import sys
def read_bin_data(file_path, meta_path, data_type='int16'):
    """
    Reads binary data from a file using memory-mapping.

    Args:
        file_path (str): Path to the binary file.
        meta_path (str): Path to the corresponding metadata file.
        data_type (str): Data type of the samples.

    Returns:
        numpy.ndarray: Data read from the file.
    """
    # Read metadata to get the number of channels
    meta = readMeta(meta_path)
    num_channels = int(meta['nSavedChans'])

    # Calculate the first dimension of the shape
    file_size = os.path.getsize(file_path)
    num_samples = file_size // (num_channels * np.dtype(data_type).itemsize)
    shape = (num_samples, num_channels)

    # Memory-map the binary file
    with open(file_path, 'rb') as f:
        data = np.memmap(f, dtype=data_type, mode='r', shape=shape)

    return data


def downsample_data(data, original_fs, target_fs):
    """
    Downsamples data to a target sampling rate.

    Args:
        data (numpy.ndarray): Data to downsample.
        original_fs (int): Original sampling rate.
        target_fs (int): Target sampling rate.

    Returns:
        numpy.ndarray: Downsampled data.
    """
    downsampling_factor = original_fs // target_fs
    downsampled_data = data[::downsampling_factor,:] #This is the fixed line
    return downsampled_data

def filter_data(data, fs, cutoff_freq):
    """
    Filters data using a sinc filter.

    Args:
        data (numpy.ndarray): Data to filter.
        fs (int): Sampling rate.
        cutoff_freq (int): Cutoff frequency for the filter.

    Returns:
        numpy.ndarray: Filtered data.
    """
    print("Shape of data entering filter_data:", data.shape)  # Debug print
    print("Data type of data entering filter_data:", data.dtype) # Debug print
    nyquist_freq = fs / 2
    normalized_cutoff_freq = cutoff_freq / nyquist_freq
    numtaps = 31  # Reduced filter order significantly
    taps = signal.firwin(numtaps=numtaps, cutoff=normalized_cutoff_freq)
    print("Shape of filter taps:", taps.shape) # Debug print
    filtered_data = signal.lfilter(taps, 1.0, data)
    print("Shape of filtered data exiting filter_data:", filtered_data.shape) # Debug print
    print("Data type of filtered data exiting filter_data:", filtered_data.dtype) # Debug print
    return filtered_data

def compute_spectrogram(data, fs, window_size, step_size):
    """
    Computes spectrogram of the data using ShortTimeFFT.

    Args:
        data (numpy.ndarray): Data to compute spectrogram from.
        fs (int): Sampling rate.
        window_size (int): Size of the window in seconds.
        step_size (int): Step size in seconds.

    Returns:
        numpy.ndarray, numpy.ndarray, numpy.ndarray: Spectrogram, frequencies, and times.
    """
    window_samples = int(window_size * fs)
    step_samples = int(step_size * fs)
    hop_samples = step_samples # Let hop_samples be step_samples, related to step_size in time
    win = signal.windows.hann(window_samples) # Use Hann window of window_samples length

    # Check if input data length is sufficient
    if len(data) < window_samples:
        print("Input data length is too small for spectrogram computation.")
        return None, None, None

    # Create ShortTimeFFT object - Initialize with fs, win, and hop
    stft = ShortTimeFFT(fs=fs, win=win, hop=hop_samples)

    # Compute spectrogram, now passing nperseg and noverlap to the spectrogram *method*
    # frequencies, times, spectrogram = stft.spectrogram(data,
    #                                                   nperseg=window_samples,
    #                                                   noverlap=window_samples - step_samples)
    frequencies, times, spectrogram = stft.spectrogram(data) # Remove nperseg and noverlap
    return spectrogram, frequencies, times

def compute_pca(data):
    """
    Performs Principal Component Analysis (PCA) on the z-scored data.

    Args:
        data (numpy.ndarray): Data to perform PCA on.

    Returns:
        numpy.ndarray: First principal component.
    """
    zscored_data = zscore(data)
    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(zscored_data)
    return pc1

def compute_theta_dominance(spectrogram, frequencies):
    """
    Computes theta dominance from spectrogram.

    Args:
        spectrogram (numpy.ndarray): Spectrogram data.
        frequencies (numpy.ndarray): Frequencies corresponding to the spectrogram.

    Returns:
        numpy.ndarray: Theta dominance.
    """
    theta_band = np.where((frequencies >= 5) & (frequencies <= 10))
    theta_power = np.mean(spectrogram[theta_band,:], axis=0)
    total_band = np.where((frequencies >= 2) & (frequencies <= 16))
    total_power = np.mean(spectrogram[total_band,:], axis=0)
    theta_dominance = theta_power / total_power
    return theta_dominance

def compute_emg(data, fs, frequencies): #Added frequencies as an argument
    """
    Estimates EMG from the data using zero-lag correlation.

    Args:
        data (numpy.ndarray): Data to estimate EMG from.
        fs (int): Sampling rate.

    Returns:
        numpy.ndarray: EMG estimate.
    """
    # Filter the data between 300 and 600 Hz
    emg_band = np.where((frequencies >= 300) & (frequencies <= 600))
    emg_data = data[emg_band,:]

    # Compute the zero-lag correlation across recording sites
    emg_corr = np.sum(emg_data[:,:-1] * emg_data[:, 1:], axis=1)

    return emg_corr

def score_sleep_states(pc1, theta_dominance, emg, buffer_size=10):
    """
    Scores sleep states based on the provided metrics with soft sticky
    thresholds.

    Args:
        pc1 (numpy.ndarray): First principal component.
        theta_dominance (numpy.ndarray): Theta dominance.
        emg (numpy.ndarray): EMG estimate.
        buffer_size (int): Size of the buffer zone in samples.

    Returns:
        numpy.ndarray: Sleep state labels.
    """
    nrem_threshold = np.percentile(pc1, 75)
    rem_threshold = np.percentile(theta_dominance, 75)
    emg_threshold = np.percentile(emg, 25)

    sleep_states = np.zeros_like(pc1, dtype=int)
    current_state = 0  # Initialize current state as Awake

    for i in range(len(pc1)):
        if current_state == 0:  # Awake
            if (pc1[i] > nrem_threshold) and (emg[i] < emg_threshold):
                nrem_count = 1
            else:
                nrem_count = 0
            if (theta_dominance[i] > rem_threshold) and (emg[i] < emg_threshold):
                rem_count = 1
            else:
                rem_count = 0
            if nrem_count >= buffer_size:
                current_state = 1  # Switch to NREM
            elif rem_count >= buffer_size:
                current_state = 2  # Switch to REM
        elif current_state == 1:  # NREM
            if (pc1[i] < nrem_threshold) or (emg[i] > emg_threshold):
                awake_count = 1
            else:
                awake_count = 0
            if awake_count >= buffer_size:
                current_state = 0  # Switch to Awake
        elif current_state == 2:  # REM
            if (theta_dominance[i] < rem_threshold) or (emg[i] > emg_threshold):
                awake_count = 1
            else:
                awake_count = 0
            if awake_count >= buffer_size:
                current_state = 0  # Switch to Awake
        sleep_states[i] = current_state

    return sleep_states


# Parameters
original_fs = 2500
target_fs = 1250
cutoff_freq = 450
window_size = 20
step_size = 2
num_channels = 10

# Get file path from user using a browser window
root = Tk()
root.withdraw()
root.attributes("-topmost", True)
file_path = Path(filedialog.askopenfilename(title="Select LFP binary file"))
root.destroy()

# Load data
data = read_bin_data(file_path, file_path.with_suffix('.meta'))

# Select last 10 channels
data = data[:, -num_channels:]

print("Shape of data before filtering:", data.shape) # Debug print
print("Data type of data before filtering:", data.dtype) # Debug print

# Filter and downsample data
filtered_data = filter_data(data, original_fs, cutoff_freq)
downsampled_data = downsample_data(filtered_data, original_fs, target_fs)

# Compute spectrogram
spectrogram, frequencies, times = compute_spectrogram(
    downsampled_data, target_fs, window_size, step_size
)
if spectrogram is None:  # Check if spectrogram is None
    print("Spectrogram computation failed due to insufficient data length. Exiting.")
    sys.exit()
# Compute PCA after z-scoring
pc1 = compute_pca(spectrogram)

# Compute theta dominance
theta_dominance = compute_theta_dominance(spectrogram, frequencies)

# Compute EMG
emg = compute_emg(downsampled_data, target_fs, frequencies) #Added frequencies as an argument

# Score sleep states
sleep_states = score_sleep_states(pc1, theta_dominance, emg)

# Save sleep states
np.save('sleep_states.npy', sleep_states)