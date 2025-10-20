
import yaml
import json
import numpy as np
import spikeinterface.extractors as se
import spikeinterface.preprocessing as sp

def preprocess_lfp():
    """
    Loads raw SpikeGLX data, preprocesses it to extract the LFP, and saves the
    result as a memory-mappable binary file and a metadata JSON file.
    """
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # 1. Load SpikeGLX recording
    recording = se.read_spikeglx(config['data_directory'])

    # 2. Pre-processing
    # Apply local common average referencing (CAR) on a per-shank basis
    recording_car = sp.common_reference(recording, reference='local', operator='median')

    # Bandpass filter the data to the LFP range
    recording_filtered = sp.bandpass_filter(recording_car, freq_min=config['lfp_filter_band'][0], freq_max=config['lfp_filter_band'][1])

    # Downsample the data
    recording_downsampled = sp.resample(recording_filtered, resample_rate=config['lfp_sampling_rate'])

    # 4. Save the final LFP data
    lfp_data = recording_downsampled.get_traces(return_scaled=True)
    lfp_filepath = f"{config['processed_directory']}/lfp.bin"
    lfp_data.tofile(lfp_filepath)

    # Save metadata
    metadata = {
        'sampling_rate': recording_downsampled.get_sampling_frequency(),
        'num_channels': recording_downsampled.get_num_channels(),
        'dtype': str(lfp_data.dtype)
    }
    metadata_filepath = f"{config['processed_directory']}/lfp_metadata.json"
    with open(metadata_filepath, 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f"Processed LFP data saved to {lfp_filepath}")
    print(f"Metadata saved to {metadata_filepath}")

if __name__ == '__main__':
    preprocess_lfp()
