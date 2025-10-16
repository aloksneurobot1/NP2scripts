# -*- coding: utf-8 -*-
"""
Central configuration file for the neurosuite package.
"""
import numpy as np
import multiprocessing

# --- Ripple Detection Parameters ---
RIPPLE_FILTER_LOWCUT = 80.0  # Hz
RIPPLE_FILTER_HIGHCUT = 250.0 # Hz
RIPPLE_POWER_LP_CUTOFF = 55.0 # Hz (for baseline calculation)
RIPPLE_DETECTION_SD_THRESHOLD = 4.0 # SD above mean
RIPPLE_EXPANSION_SD_THRESHOLD = 2.0 # SD above mean
RIPPLE_MIN_DURATION_MS = 15.0 # ms
RIPPLE_MERGE_GAP_MS = 15.0 # ms

# --- Sharp-Wave (SPW) Detection Parameters (for CA1) ---
SPW_FILTER_LOWCUT = 5.0   # Hz
SPW_FILTER_HIGHCUT = 40.0  # Hz
SPW_DETECTION_SD_THRESHOLD = 2.5 # SD above mean (applied to absolute filtered LFP)
SPW_MIN_DURATION_MS = 20.0  # ms
SPW_MAX_DURATION_MS = 400.0 # ms

# --- Co-occurrence Window ---
COOCCURRENCE_WINDOW_MS = 60.0 # ms (+/- around reference ripple peak)

# --- Spectrogram Parameters ---
SPECTROGRAM_WINDOW_MS = 200 # ms (+/- around event timestamp)
SPECTROGRAM_FREQS = np.arange(10, 300, 2) # Example frequency range for spectrogram

# --- Parallel Processing Configuration ---
NUM_CORES = max(1, multiprocessing.cpu_count() - 2)

# --- CSD Analysis Parameters ---
TARGET_FS_CSD = 1250.0  # Hz
LFP_BAND_LOWCUT_CSD = 1.0  # Hz
LFP_BAND_HIGHCUT_CSD = 300.0 # Hz
NUMTAPS_CSD_FILTER = 101 # Odd number

CSD_SIGMA_CONDUCTIVITY = 0.3  # Siemens per meter (S/m)
KCSD_LAMBDAS_CV = np.logspace(-7, -2, 9)
KCSD_RS_CV_UM = np.logspace(np.log10(20), np.log10(500), 9)

EPOCH_SUB_CHUNK_DURATION_SECONDS = 10

# --- Sleep State Scoring Parameters ---
SLEEP_TARGET_FS = 1250.0  # Desired target sampling rate
SLEEP_LFP_CUTOFF = 600.0  # Low-pass filter cutoff (Hz)
SLEEP_NUM_CHANNELS = 48  # Number of channels from END of recording
SLEEP_SPECTROGRAM_WINDOW_SEC = 10.0  # Spectrogram window size (seconds)
SLEEP_SPECTROGRAM_STEP_SEC = 1.0  # Spectrogram step size (seconds)
SLEEP_CALCULATE_EMG = True  # SET True TO ENABLE EMG CALCULATION
SLEEP_ICA_N_COMPONENTS = 10  # Number of ICA components (if using EMG)
SLEEP_ICA_FIT_DURATION_SEC = 240  # Duration (seconds) for ICA fitting (if using EMG)
SLEEP_ICA_NUM_CHANNELS = 48  # Number of channels for ICA (from selected LFP, if using EMG)
SLEEP_SCORING_BUFFER_STEPS = 5  # Steps to confirm state change
SLEEP_EMG_THRESHOLD_FIXED = None  # Use None for percentile, or set fixed value

# --- Speed/Motion Parameters (for sleep scoring) ---
SPEED_VIDEO_FPS = 30  # Video FPS (if using speed)
SPEED_VIDEO_START_OFFSET_SEC = 0  # Video vs Neural offset (seconds)
SPEED_THRESHOLD_CM_S = 1.5  # Speed threshold (cm/s)