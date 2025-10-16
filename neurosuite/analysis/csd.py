# -*- coding: utf-8 -*-
"""
Functions for Current Source Density (CSD) analysis.
"""
import numpy as np
import quantities as pq
import neo
from elephant.current_source_density_src.KCSD import KCSD1D
from scipy import signal

def get_voltage_scaling_factor(meta):
    """Calculates the factor to convert int16 ADC values to microvolts (uV)."""
    try:
        v_max = float(meta['imAiRangeMax'])
        i_max_adc_val = float(meta['imMaxInt'])
        probe_type = int(meta.get('imDatPrb_type', 0))
        lfp_gain = None

        if probe_type in [21, 24, 2013]:
            lfp_gain = 80.0
        else:
            general_lfp_gain_key_str = "~imChanLFGain"
            if general_lfp_gain_key_str in meta:
                 lfp_gain = float(meta[general_lfp_gain_key_str])
            else:
                first_lfp_gain_key_found = None
                sorted_keys = sorted([key for key in meta.keys() if key.endswith('lfGain')])
                if sorted_keys:
                    first_lfp_gain_key_found = sorted_keys[0]
                if first_lfp_gain_key_found:
                    lfp_gain = float(meta[first_lfp_gain_key_found])
                else:
                    lfp_gain = 250.0
        if lfp_gain is None: raise ValueError("LFP gain could not be determined.")
        if i_max_adc_val == 0 or lfp_gain == 0: raise ValueError("i_max_adc_val or LFP gain is zero.")
        scaling_factor_uv = (v_max / i_max_adc_val) * (1.0 / lfp_gain) * 1e6
        return scaling_factor_uv
    except Exception as e:
        print(f"Error calculating voltage scaling factor: {e}"); return None

def preprocess_lfp_for_csd(lfp_chunk, fs_in, lowcut, highcut, numtaps, target_fs):
    """Preprocesses a chunk of LFP data for CSD analysis."""
    if lfp_chunk.ndim == 1:
        lfp_chunk = lfp_chunk[:, np.newaxis]

    downsampling_factor = int(round(fs_in / target_fs)) if target_fs < fs_in else 1

    if downsampling_factor > 1:
        lfp_chunk_downsampled = signal.decimate(lfp_chunk, downsampling_factor, axis=0, ftype='fir', zero_phase=True)
        fs_out = fs_in / downsampling_factor
    else:
        lfp_chunk_downsampled = lfp_chunk
        fs_out = fs_in

    nyq = fs_out / 2.0
    actual_highcut = min(highcut, nyq * 0.99)
    actual_lowcut = max(lowcut, 0.01)

    if numtaps >= lfp_chunk_downsampled.shape[0]:
        numtaps = lfp_chunk_downsampled.shape[0] - 1
    if numtaps % 2 == 0 and numtaps > 0:
        numtaps -=1

    if numtaps < 3 or actual_lowcut >= actual_highcut:
        lfp_filtered = lfp_chunk_downsampled
    else:
        fir_taps = signal.firwin(numtaps, [actual_lowcut, actual_highcut], fs=fs_out, pass_zero='bandpass', window='hamming')
        lfp_filtered = signal.filtfilt(fir_taps, 1.0, lfp_chunk_downsampled, axis=0)

    return lfp_filtered, fs_out


def run_kcsd_analysis(lfp_data, coords_um, fs, **kcsd_params):
    """
    Runs kCSD analysis on a shank of LFP data.

    Args:
        lfp_data (np.ndarray): LFP data for a single shank (samples, channels).
        coords_um (np.ndarray): Electrode coordinates in micrometers.
        fs (float): Sampling rate of the LFP data.
        **kcsd_params: Keyword arguments for kCSD analysis.

    Returns:
        tuple: (csd_data, csd_positions, kcsd_estimator)
    """
    coords_quant_um = coords_um.reshape(-1, 1) * pq.um

    shank_analog_signal = neo.AnalogSignal(
        lfp_data.T,
        units='uV',
        sampling_rate=fs * pq.Hz
    )

    num_csd_estimation_points = max(32, int(len(coords_um) * 1.5))
    xmin_param = coords_um.min() * pq.um
    xmax_param = coords_um.max() * pq.um
    gdx_param = ((xmax_param - xmin_param) / (num_csd_estimation_points - 1)) if num_csd_estimation_points > 1 else 10.0 * pq.um

    kcsd_estimator_obj = KCSD1D(
        ele_pos=coords_quant_um.magnitude,
        pots=shank_analog_signal.magnitude,
        sigma=kcsd_params.get('sigma', 0.3),
        n_src_init=num_csd_estimation_points,
        xmin=xmin_param.rescale(pq.mm).magnitude,
        xmax=xmax_param.rescale(pq.mm).magnitude,
        gdx=gdx_param.rescale(pq.mm).magnitude,
        ext_x=0.0,
    )

    kcsd_estimator_obj.cross_validate(
        lambdas=kcsd_params.get('lambdas', np.logspace(-7, -2, 9)),
        Rs=kcsd_params.get('Rs', np.logspace(np.log10(20), np.log10(500), 9)) / 1000
    )

    csd_profile_values = kcsd_estimator_obj.values()

    csd_result_neo = neo.AnalogSignal(
        csd_profile_values.T,
        units = pq.uA / pq.mm**3,
        sampling_rate = fs * pq.Hz,
    )

    csd_data_matrix = np.asarray(csd_result_neo).astype(np.float32)
    csd_positions_plot_mm = kcsd_estimator_obj.estm_x.flatten()

    return csd_data_matrix, csd_positions_plot_mm, kcsd_estimator_obj