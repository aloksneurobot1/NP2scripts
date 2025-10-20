
import numpy as np
from scipy.signal import butter, filtfilt

def bandpass_filter(signal, lowcut, highcut, fs, order=5):
    """
    Applies a Butterworth bandpass filter to a signal.
    Assumes signal is in shape (time, channels) or (time,).
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, signal, axis=0)
    return y
