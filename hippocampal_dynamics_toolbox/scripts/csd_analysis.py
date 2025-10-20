
import yaml
import json
import numpy as np
import pandas as pd

def calculate_csd():
    """
    Loads the preprocessed LFP data and channel map, calculates the 1D Current
    Source Density (CSD) for each shank, and saves the result as a binary file.
    """
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    processed_dir = config['processed_directory']

    # 1. Load the LFP data and metadata
    with open(f"{processed_dir}/lfp_metadata.json", 'r') as f:
        metadata = json.load(f)

    lfp_data = np.fromfile(f"{processed_dir}/lfp.bin", dtype=metadata['dtype'])
    lfp_data = lfp_data.reshape(-1, metadata['num_channels'])

    # 2. Load the channel map
    channel_map = pd.read_csv(config['channel_map_file'])

    # Get CSD parameters from config
    conductivity = config['conductivity']
    inter_electrode_dist_um = config['inter_electrode_dist']
    inter_electrode_dist_m = inter_electrode_dist_um * 1e-6  # convert to meters

    all_csd = []

    # 3. For each shank, sort the channels by depth and calculate CSD
    for shank_id in sorted(channel_map['shank'].unique()):
        shank_channels = channel_map[channel_map['shank'] == shank_id].sort_values('y')
        shank_indices = shank_channels.index.values # Assuming channel index in file corresponds to row number

        # Get LFP data for the current shank, ordered by depth
        shank_lfp = lfp_data[:, shank_indices]

        # 4. Calculate CSD using the second-spatial-derivative formula
        # CSD = -sigma * (V(z+dz) - 2V(z) + V(z-dz)) / dz^2
        # We apply this along the depth axis (axis=1)
        csd = -conductivity * np.diff(shank_lfp, n=2, axis=1) / (inter_electrode_dist_m**2)

        # Pad the result to match the original number of channels for the shank
        # Here, we pad with zeros on both ends, as CSD is not defined for boundary electrodes
        csd_padded = np.pad(csd, ((0, 0), (1, 1)), mode='constant', constant_values=0)

        all_csd.append(csd_padded)

    # Combine CSD from all shanks
    # Note: This assumes the original channel ordering is preserved, we just replace LFP with CSD values
    final_csd = np.zeros_like(lfp_data)
    for shank_id, csd_shank_data in zip(sorted(channel_map['shank'].unique()), all_csd):
        shank_indices = channel_map[channel_map['shank'] == shank_id].sort_values('y').index.values
        final_csd[:, shank_indices] = csd_shank_data

    # 5. Save the resulting CSD data
    csd_filepath = f"{processed_dir}/csd.bin"
    final_csd.astype(metadata['dtype']).tofile(csd_filepath)

    print(f"CSD data saved to {csd_filepath}")

if __name__ == '__main__':
    calculate_csd()
