# -*- coding: utf-8 -*-
"""
I/O functions for loading electrophysiology data, including SpikeGLX formats.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import re
import traceback

def read_sglx_meta(meta_path):
    """
    Reads a SpikeGLX .meta file and returns a dictionary of parameters.
    """
    meta_path = Path(meta_path)
    with open(meta_path, 'r') as f:
        meta_content = f.read()

    meta_dict = {}
    for line in meta_content.splitlines():
        if '=' in line:
            key, value = line.split('=', 1)
            meta_dict[key.strip()] = value.strip()
    return meta_dict

def load_sglx_data(bin_path, meta_path=None, data_type='int16', memmap=True):
    """
    Loads SpikeGLX binary data.

    Args:
        bin_path (str or Path): Path to the binary LFP file (*.lf.bin or *.ap.bin).
        meta_path (str or Path, optional): Path to the metadata file (*.meta).
            If None, it's inferred from the bin_path. Defaults to None.
        data_type (str, optional): Data type of the samples. Defaults to 'int16'.
        memmap (bool, optional): If True, loads data using memory-mapping.
            If False, loads data into RAM. Defaults to True.

    Returns:
        tuple: (data, fs, meta)
            - data (np.ndarray or np.memmap): Loaded data (samples, channels).
            - fs (float): Sampling rate.
            - meta (dict): Metadata dictionary.
        Returns (None, None, None) on error.
    """
    bin_path = Path(bin_path)
    if meta_path is None:
        meta_path = bin_path.with_suffix('.meta')
    else:
        meta_path = Path(meta_path)

    if not bin_path.exists():
        print(f"Error: Binary file not found - {bin_path}")
        return None, None, None
    if not meta_path.exists():
        print(f"Error: Meta file not found - {meta_path}")
        return None, None, None

    try:
        meta = read_sglx_meta(meta_path)
        n_channels = int(meta['nSavedChans'])

        if 'imSampRate' in meta:
            fs = float(meta['imSampRate'])
        elif 'niSampRate' in meta:
            fs = float(meta['niSampRate'])
        else:
            raise ValueError("Sampling rate key not found in meta file.")

        file_size = bin_path.stat().st_size
        item_size = np.dtype(data_type).itemsize
        n_samples = file_size // (n_channels * item_size)

        shape = (n_samples, n_channels)

        mode = 'r' if memmap else 'c'
        data = np.memmap(bin_path, dtype=data_type, mode=mode, shape=shape)

        if not memmap:
            data = np.copy(data)

        return data, fs, meta

    except Exception as e:
        print(f"An unexpected error occurred in load_sglx_data: {e}")
        traceback.print_exc()
        return None, None, None

def load_channel_info(filepath):
    """Loads channel information from a CSV file."""
    try:
        channel_df = pd.read_csv(filepath)
        required_cols = ['global_channel_index', 'shank_index', 'acronym', 'name']
        if not all(col in channel_df.columns for col in required_cols):
            raise ValueError(f"Channel info CSV must contain columns: {required_cols}")
        return channel_df
    except FileNotFoundError:
        print(f"Error: Channel info file not found at {filepath}")
        raise
    except Exception as e:
        print(f"Error loading channel info: {e}")
        raise

def load_timestamps(filepath):
    """Loads timestamps from a .npy file."""
    try:
        return np.load(filepath, allow_pickle=True)
    except FileNotFoundError:
        print(f"Error: Timestamp file not found at {filepath}")
        raise
    except Exception as e:
        print(f"Error loading timestamps: {e}")
        raise