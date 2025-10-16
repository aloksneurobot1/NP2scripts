# -*- coding: utf-8 -*-
"""
Functions for Sharp Wave Ripple (SWR) detection and analysis.
"""
import numpy as np
from scipy import signal
from neurosuite.utils.processing import apply_fir_filter, calculate_instantaneous_power

def get_baseline_stats(power_signal, fs, non_rem_periods_samples, ripple_power_lp_cutoff):
    """
    Calculates baseline mean and SD from power during specified periods (non-REM).
    """
    if power_signal is None or power_signal.size == 0:
        raise ValueError("Input power signal is invalid.")

    baseline_power_segments = []
    if not non_rem_periods_samples:
        baseline_power_segments = [power_signal]
    else:
        for start, end in non_rem_periods_samples:
             start = max(0, start)
             end = min(power_signal.shape[0], end)
             if start < end:
                 baseline_power_segments.append(power_signal[start:end])

    if not baseline_power_segments:
         baseline_power_segments = [power_signal]

    full_baseline_power = np.concatenate(baseline_power_segments)

    initial_mean = np.mean(full_baseline_power)
    initial_sd = np.std(full_baseline_power)

    if initial_sd == 0:
        clipped_power = full_baseline_power
    else:
        clip_threshold = initial_mean + 4 * initial_sd
        clipped_power = np.clip(full_baseline_power, a_min=None, a_max=clip_threshold)

    rectified_power = np.abs(clipped_power)

    numtaps_lp = 101
    if numtaps_lp >= rectified_power.size:
         processed_baseline_power = rectified_power
    else:
         b_lp = signal.firwin(numtaps_lp, ripple_power_lp_cutoff, fs=fs, pass_zero=True, window='hamming')
         processed_baseline_power = signal.filtfilt(b_lp, 1, rectified_power)

    baseline_mean = np.mean(processed_baseline_power)
    baseline_sd = np.std(processed_baseline_power)

    return baseline_mean, baseline_sd

def detect_events(signal_data, fs, threshold_high, threshold_low, min_duration_ms, merge_gap_ms):
    """
    Detects events based on signal thresholds, duration, and merging gaps.
    """
    if signal_data is None or signal_data.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    min_duration_samples = int(min_duration_ms * fs / 1000)
    merge_gap_samples = int(merge_gap_ms * fs / 1000)

    above_high_threshold = signal_data > threshold_high
    diff_high = np.diff(above_high_threshold.astype(np.int8))
    starts_high = np.where(diff_high == 1)[0] + 1
    ends_high = np.where(diff_high == -1)[0]

    if len(above_high_threshold) > 0:
        if above_high_threshold[0]: starts_high = np.insert(starts_high, 0, 0)
        if above_high_threshold[-1]: ends_high = np.append(ends_high, len(signal_data) - 1)
    else:
         return np.array([], dtype=int), np.array([], dtype=int)

    if len(starts_high) == 0 or len(ends_high) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    valid_pairs = []
    current_start_idx = 0
    current_end_idx = 0
    while current_start_idx < len(starts_high) and current_end_idx < len(ends_high):
        start_val = starts_high[current_start_idx]
        end_val = ends_high[current_end_idx]
        if start_val <= end_val:
            valid_pairs.append((start_val, end_val))
            current_start_idx += 1
            current_end_idx += 1
        else:
            current_end_idx += 1

    if not valid_pairs:
        return np.array([], dtype=int), np.array([], dtype=int)

    starts_high, ends_high = zip(*valid_pairs)
    starts_high = np.array(starts_high)
    ends_high = np.array(ends_high)

    expanded_starts = []
    expanded_ends = []
    below_low_mask = signal_data < threshold_low

    for start, end in zip(starts_high, ends_high):
        current_start = start
        while current_start > 0 and not below_low_mask[current_start - 1]: current_start -= 1
        current_end = end
        while current_end < len(signal_data) - 1 and not below_low_mask[current_end + 1]: current_end += 1
        expanded_starts.append(current_start)
        expanded_ends.append(current_end)

    if not expanded_starts:
        return np.array([], dtype=int), np.array([], dtype=int)

    merged_starts = [expanded_starts[0]]
    merged_ends = [expanded_ends[0]]
    for i in range(1, len(expanded_starts)):
        gap = expanded_starts[i] - merged_ends[-1] - 1
        if gap < merge_gap_samples:
            merged_ends[-1] = max(merged_ends[-1], expanded_ends[i])
        else:
            merged_starts.append(expanded_starts[i])
            merged_ends.append(expanded_ends[i])

    final_starts = []
    final_ends = []
    for start, end in zip(merged_starts, merged_ends):
        duration_samples = end - start + 1
        if duration_samples >= min_duration_samples:
            final_starts.append(start)
            final_ends.append(end)

    return np.array(final_starts, dtype=int), np.array(final_ends, dtype=int)

def find_event_features(lfp_segment, power_segment):
    """
    Finds peak power index and nearest trough index within an event segment.
    """
    if power_segment is None or lfp_segment is None or power_segment.size == 0 or lfp_segment.size == 0: return -1, -1
    if len(lfp_segment) != len(power_segment):
         return -1,-1

    peak_power_idx_rel = np.nanargmax(power_segment)
    lfp_segment_float = lfp_segment.astype(np.float64)
    troughs_rel, _ = signal.find_peaks(-lfp_segment_float)

    if len(troughs_rel) == 0:
        trough_idx_rel = np.nanargmin(lfp_segment_float)
    else:
        trough_idx_rel = troughs_rel[np.nanargmin(np.abs(troughs_rel - peak_power_idx_rel))]

    return peak_power_idx_rel, trough_idx_rel