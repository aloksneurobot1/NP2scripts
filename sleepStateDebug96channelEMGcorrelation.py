# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 18:51:11 2025

@author: HT_bo
"""
# import scipy
# print(scipy.__version__)
import numpy as np
from scipy import signal
# from scipy.signal import ShortTimeFFT # No longer needed
from sklearn.decomposition import PCA
from scipy.stats import zscore
from tkinter import Tk
from tkinter import filedialog
from pathlib import Path
from DemoReadSGLXData.readSGLX import readMeta
import os
import sys

# Redirect stdout and stderr to a file
output_file_path = "script_output.txt"  # You can change the filename if you like
original_stdout = sys.stdout
original_stderr = sys.stderr
sys.stdout = open(output_file_path, 'w')
sys.stderr = sys.stdout 

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
    Computes spectrogram of the data using scipy.signal.spectrogram.

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
    win = signal.windows.hann(window_samples) # Use Hann window
    noverlap = window_samples - step_samples

    # Compute spectrogram using scipy.signal.spectrogram
    frequencies, times, spectrogram = signal.spectrogram(
        data,
        fs=fs,
        window=win,
        nperseg=window_samples,
        noverlap=noverlap,
        scaling='density',
        mode='psd' # Power Spectral Density
    )
    return spectrogram, frequencies, times

def compute_pca(data):
    """
    Performs Principal Component Analysis (PCA) on the z-scored data.

    Args:
        data (numpy.ndarray): Data to perform PCA on.

    Returns:
        numpy.ndarray: First principal component (1D array).
    """
    zscored_data = zscore(data, axis=0) # z-score along frequency axis (axis=0)
    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(zscored_data.T) # Transpose for PCA on frequency axis (rows), shape will be (n_times, 1)
    return pc1.squeeze() # Remove the trailing dimension to get (n_times,) shape


def compute_theta_dominance(spectrogram, frequencies):
    """
    Computes theta dominance from spectrogram.

    Args:
        spectrogram (numpy.ndarray): Spectrogram data.
        frequencies (numpy.ndarray): Frequencies corresponding to the spectrogram.

    Returns:
        numpy.ndarray: Theta dominance.
    """
    print("Shape of spectrogram inside compute_theta_dominance:", spectrogram.shape) # Debug
    theta_band_indices = np.where((frequencies >= 5) & (frequencies <= 10))[0] # Get indices
    theta_power = np.mean(spectrogram[theta_band_indices,:], axis=0)
    print("Shape of theta_power:", theta_power.shape) # Debug
    total_band_indices = np.where((frequencies >= 2) & (frequencies <= 16))[0] # Get indices
    total_power = np.mean(spectrogram[total_band_indices,:], axis=0)
    print("Shape of total_power:", total_power.shape) # Debug
    theta_dominance = theta_power / total_power
    print("Shape of theta_dominance:", theta_dominance.shape) # Debug
    return theta_dominance

def compute_emg(data, fs, window_size, step_size): # Added window_size and step_size
    """
    Estimates EMG from the data using zero-lag correlation, averaged over spectrogram time segments.
    ... (rest of function docstring) ...
    """
    # Filter the data between 300 and 600 Hz
    emg_cutoff_low = 300
    emg_cutoff_high = 600
    nyquist_freq = fs / 2
    normalized_cutoff_low = emg_cutoff_low / nyquist_freq
    normalized_cutoff_high = emg_cutoff_high / nyquist_freq
    numtaps_emg = 31 # or adjust as needed, e.g., 127 for sharper filter
    emg_taps = signal.firwin(numtaps_emg, [normalized_cutoff_low, normalized_cutoff_high],
                             pass_zero=False) # bandpass filter

    emg_filtered_data = signal.lfilter(emg_taps, 1.0, data)
    print("Shape of emg_filtered_data:", emg_filtered_data.shape) # Debug
    print("Range of emg_filtered_data: Min=", np.min(emg_filtered_data), "Max=", np.max(emg_filtered_data)) # Debug

    emg_corr = np.mean(emg_filtered_data[:,:-1] * emg_filtered_data[:, 1:], axis=1) #Mean correlation
    print("Shape of emg_corr:", emg_corr.shape) # Debug
    print("Range of emg_corr: Min=", np.min(emg_corr), "Max=", np.max(emg_corr)) # Debug


    # Calculate EMG per spectrogram time segment by averaging
    window_samples = int(window_size * fs)
    step_samples = int(step_size * fs)
    num_segments = (len(data) - window_samples) // step_samples + 1 # Number of spectrogram time segments
    emg_segmented = np.zeros(num_segments) # Initialize segmented EMG array

    for i in range(num_segments):
        start_sample = i * step_samples
        end_sample = start_sample + window_samples
        if end_sample <= len(emg_corr): # Ensure we don't go out of bounds
            emg_segmented[i] = np.mean(emg_corr[start_sample:end_sample]) # Average EMG within the segment

    print("Shape of emg_segmented:", emg_segmented.shape) # Debug
    print("Range of emg_segmented: Min=", np.min(emg_segmented), "Max=", np.max(emg_segmented)) # Debug
    return emg_segmented


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
    # emg_threshold = -0.0001
    print("---Threshold Values---") # Debug print for thresholds
    print(f"NREM Threshold: {nrem_threshold:.4f}")
    print(f"REM Threshold: {rem_threshold:.4f}")
    print(f"EMG Threshold: {emg_threshold:.4f}")
    print("----------------------")

    sleep_states = np.zeros_like(pc1, dtype=int)
    current_state = 0  # Initialize current state as Awake
    nrem_count = 0
    rem_count = 0
    awake_count = 0 # Initialize counters

    # Ensure theta_dominance and emg are at least as long as pc1 to prevent IndexError
    min_len = min(len(pc1), len(theta_dominance), len(emg)) # Find the minimum length
    theta_dominance = theta_dominance[:min_len] # Truncate arrays to the minimum length
    emg = emg[:min_len]
    pc1 = pc1[:min_len] # Truncate pc1 as well to ensure consistent length


    for i in range(len(pc1)):
        print(f"\n---Time Point: {i}---") # Debug print for each time point
        print(f"Current State: {current_state}")
        print(f"PC1: {pc1[i]:.4f}, Theta Dominance: {theta_dominance[i]:.4f}, EMG: {emg[i]:.4f}")

        if current_state == 0:  # Awake
            print("State: Awake")
            if (pc1[i] > nrem_threshold) and (emg[i] < emg_threshold):
                nrem_count += 1
                print(f"  Condition NREM met: PC1 > NREM Thresh ({pc1[i]:.4f} > {nrem_threshold:.4f}) and EMG < EMG Thresh ({emg[i]:.4f} < {emg_threshold:.4f})")
            else:
                nrem_count = 0
                print("  Condition NREM not met")
            if (theta_dominance[i] > rem_threshold) and (emg[i] < emg_threshold):
                rem_count += 1
                print(f"  Condition REM met: Theta > REM Thresh ({theta_dominance[i]:.4f} > {rem_threshold:.4f}) and EMG < EMG Thresh ({emg[i]:.4f} < {emg_threshold:.4f})")
            else:
                rem_count = 0
                print("  Condition REM not met")

            print(f"  NREM Count: {nrem_count}, REM Count: {rem_count}")

            if nrem_count >= buffer_size:
                current_state = 1  # Switch to NREM
                print("  Transition to NREM (State 1)")
            elif rem_count >= buffer_size:
                current_state = 2  # Switch to REM
                print("  Transition to REM (State 2)")

        elif current_state == 1:  # NREM
            print("State: NREM")
            if (pc1[i] < nrem_threshold) or (emg[i] > emg_threshold):
                awake_count += 1
                print(f"  Condition Awake met (NREM exit): PC1 < NREM Thresh ({pc1[i]:.4f} < {nrem_threshold:.4f}) or EMG > EMG Thresh ({emg[i]:.4f} > {emg_threshold:.4f})")
            else:
                awake_count = 0
                print("  Condition Awake not met (NREM sustain)")

            print(f"  Awake Count (NREM exit): {awake_count}")
            if awake_count >= buffer_size:
                current_state = 0  # Switch to Awake
                print("  Transition to Awake (State 0) from NREM")

        elif current_state == 2:  # REM
            print("State: REM")
            if (theta_dominance[i] < rem_threshold) or (emg[i] > emg_threshold):
                awake_count += 1
                print(f"  Condition Awake met (REM exit): Theta < REM Thresh ({theta_dominance[i]:.4f} < {rem_threshold:.4f}) or EMG > EMG Thresh ({emg[i]:.4f} > {emg_threshold:.4f})")
            else:
                awake_count = 0
                print("  Condition Awake not met (REM sustain)")
            print(f"  Awake Count (REM exit): {awake_count}")
            if awake_count >= buffer_size:
                current_state = 0  # Switch to Awake
                print("  Transition to Awake (State 0) from REM")
        sleep_states[i] = current_state

    return sleep_states


# Parameters
original_fs = 2500
target_fs = 1250
cutoff_freq = 450
window_size = 10 # Changed to 10s window as per description
step_size = 1 # Changed to 1s step as per description
num_channels = 96

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

# Average across channels to get a 1D signal (time series)
averaged_lfp_data = np.mean(downsampled_data, axis=1)

print("Shape of averaged_lfp_data before spectrogram:", averaged_lfp_data.shape) # Debug print

# Compute spectrogram
spectrogram, frequencies, times = compute_spectrogram(
    averaged_lfp_data, target_fs, window_size, step_size
)
if spectrogram is None:
    print("Spectrogram computation failed...")
    sys.exit()
# Compute PCA after z-scoring
pc1 = compute_pca(spectrogram)

# Compute theta dominance
theta_dominance = compute_theta_dominance(spectrogram, frequencies)

# Compute EMG
emg = compute_emg(downsampled_data, target_fs, window_size, step_size) #Added window/step size

print("Shape of pc1 before score_sleep_states:", pc1.shape)
print("Shape of theta_dominance before score_sleep_states:", theta_dominance.shape)
print("Shape of emg before score_sleep_states:", emg.shape)

print("\n---Ranges of Metrics---") # Add this section
print(f"PC1 Range:     Min={np.min(pc1):.4f}, Max={np.max(pc1):.4f}")
print(f"Theta Range:   Min={np.min(theta_dominance):.4f}, Max={np.max(theta_dominance):.4f}")
print(f"EMG Range:     Min={np.min(emg):.4f}, Max={np.max(emg):.4f}")
print("-----------------------")

# Score sleep states
sleep_states = score_sleep_states(pc1, theta_dominance, emg)

# Save sleep states
np.save('sleep_states.npy', sleep_states)