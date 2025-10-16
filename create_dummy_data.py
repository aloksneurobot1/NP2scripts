import numpy as np
import pandas as pd

# --- Configuration ---
N_NEURONS = 50
N_BINS = 2000
DATA_PATHS = {
    'control': 'control_data.csv',
    'overexpression': 'overexpression_data.csv',
    'haploinsufficient': 'haploinsufficient_data.csv'
}

# --- Generate and Save Dummy Data ---
def generate_dummy_data(n_neurons, n_bins):
    """Generates a dummy spike count matrix."""
    # Generate random data with some structure
    base_activity = np.random.rand(n_neurons, n_bins) * 0.1
    # Add some correlated activity
    for _ in range(5):
        start_neuron = np.random.randint(0, n_neurons - 5)
        start_bin = np.random.randint(0, n_bins - 100)
        base_activity[start_neuron:start_neuron+5, start_bin:start_bin+100] += np.random.rand() * 0.5
    return np.random.poisson(base_activity * 20)

for name, path in DATA_PATHS.items():
    print(f"Generating dummy data for {name}...")
    dummy_data = generate_dummy_data(N_NEURONS, N_BINS)
    pd.DataFrame(dummy_data).to_csv(path, header=False, index=False)
    print(f"Saved to {path}")

print("Dummy data generation complete.")