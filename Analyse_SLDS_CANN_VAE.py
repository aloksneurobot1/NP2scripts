# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 17:01:22 2025

@author: HT_bo
"""

# -*- coding: utf-8 -*-
"""
=======================================================================
 Main Pipeline for Analyzing Real Neural Data with the SLDS-CANN Model
=======================================================================
This script is designed to be run cell-by-cell in Spyder. It loads
real experimental data, preprocesses it, and fits the hierarchical
SLDS-CANN VAE model to discover underlying latent dynamics.
"""

# %% ===================================================================
#   1. IMPORTS AND CONFIGURATION
# =====================================================================
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from sklearn.decomposition import PCA

from cann_module import CANN
from hybrid_model import SLDS_CANN_VAE

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- Parameters to set based on your data ---
# OBS_DIM is now the number of principal components to keep.
# This captures the main modes of population activity. 50 is a reasonable start.
OBS_DIM = 50 
# You can adjust these based on hypotheses about the complexity of the dynamics
LATENT_DIM = 2
NUM_STATES = 3 # e.g., representing different sub-states within a behavior

# --- Model & Training Parameters ---
CANN_DIM = 100
RNN_HIDDEN_DIM = 128
TIME_STEPS_PER_TRIAL = 200 # Segment data into chunks of this many time steps (e.g., 20 seconds)
LEARNING_RATE = 1e-3
NUM_EPOCHS = 200 # Increase epochs for real data

# %% ===================================================================
#   2. DATA LOADING AND PREPROCESSING
# =====================================================================
print("Loading and preprocessing experimental data...")

# !!! ACTION REQUIRED: Replace with the actual paths to your data files !!!
SPIKE_DATA_PATH = "path/to/your/binned_spikes.npy"
TIMESTAMPS_PATH = "path/to/your/behavioral_timestamps.npy"

# Load the data. 
# We assume binned_spikes is a (Time, Neurons) array.
# We assume timestamps is a (Time, 1) array where each element is an integer
# label for the behavioral state (e.g., 0=NREM, 1=REM, 2=Task, 3=Awake-Quiet).
try:
    binned_spikes = np.load(SPIKE_DATA_PATH)
    timestamps = np.load(TIMESTAMPS_PATH)
except FileNotFoundError:
    print("="*60)
    print("!!! DATA FILES NOT FOUND !!!")
    print("Please update SPIKE_DATA_PATH and TIMESTAMPS_PATH in the script.")
    print("Using placeholder random data for now.")
    print("="*60)
    binned_spikes = np.random.randn(50000, 250) # (Time, Neurons)
    timestamps = np.zeros(50000)
    timestamps[10000:25000] = 1 # State 1
    timestamps[25000:40000] = 2 # State 2


# --- Step 2a: Dimensionality Reduction with PCA ---
# Fit PCA on the entire dataset to find a stable basis of population activity
print(f"Fitting PCA to reduce {binned_spikes.shape[1]} neurons to {OBS_DIM} dimensions...")
pca = PCA(n_components=OBS_DIM, whiten=True)
# It's good practice to standardize the data first
activity_for_pca = (binned_spikes - binned_spikes.mean(axis=0)) / (binned_spikes.std(axis=0) + 1e-6)
activity_pca = pca.fit_transform(activity_for_pca)
print(f"Explained variance by {OBS_DIM} components: {np.sum(pca.explained_variance_ratio_):.2f}")


# --- Step 2b: Segment Data by Behavioral State ---
# Create a dictionary to hold lists of trials for each state
segmented_data = {}
unique_states = np.unique(timestamps)

for state_label in unique_states:
    state_name = f"state_{int(state_label)}"
    print(f"Segmenting data for {state_name}...")
    
    # Find indices corresponding to the current state
    state_indices = np.where(timestamps == state_label)[0]
    state_activity = activity_pca[state_indices]
    
    # Chop the continuous data into trials of fixed length
    num_trials_in_state = len(state_activity) // TIME_STEPS_PER_TRIAL
    if num_trials_in_state == 0:
        print(f"  -> Warning: Not enough data to create any trials for {state_name}.")
        continue
        
    trials = state_activity[:num_trials_in_state * TIME_STEPS_PER_TRIAL].reshape(
        num_trials_in_state, TIME_STEPS_PER_TRIAL, OBS_DIM)
    
    segmented_data[state_name] = torch.tensor(trials, dtype=torch.float32).to(DEVICE)
    print(f"  -> Created {num_trials_in_state} trials.")

print("\nData loading and segmentation complete.")


# %% ===================================================================
#   3. MODEL INITIALIZATION AND TRAINING
# =====================================================================
# --- Select which behavioral state to train the model on ---
# You can train separate models for each state to compare their dynamics.
# For example, to train on the social interaction task:
# data_to_train = segmented_data.get('state_2') # Assuming '2' is the task label
# Or to train on NREM sleep:
# data_to_train = segmented_data.get('state_0') # Assuming '0' is NREM

# For this example, we'll pick the first available state
train_state_key = list(segmented_data.keys())[0]
data_to_train = segmented_data[train_state_key]

print(f"\nTraining the SLDS-CANN VAE model on: {train_state_key}...")
model = SLDS_CANN_VAE(OBS_DIM, LATENT_DIM, NUM_STATES, CANN_DIM, RNN_HIDDEN_DIM).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

pbar = tqdm(range(NUM_EPOCHS))
for epoch in pbar:
    optimizer.zero_grad()
    
    # Forward pass
    y_recon, q_z_mean, q_z_logvar, q_k_dist, z_sample, k_sample = model(data_to_train)
    
    # Compute loss
    loss = model.compute_loss(data_to_train, y_recon, q_z_mean, q_z_logvar, 
                              q_k_dist, z_sample, k_sample)
    
    # Backward pass and optimization
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        pbar.set_description(f"Epoch {epoch}, Loss: {loss.item():.2f}")

print("Training complete.")


# %% ===================================================================
#   4. ANALYSIS AND VISUALIZATION
# =====================================================================
print("\nAnalyzing and visualizing results from the trained model...")

# Select a trial to analyze from the trained dataset
trial_idx = 0
test_trial = data_to_train[trial_idx:trial_idx+1]

# Get model's inference for this trial
model.eval() # Set model to evaluation mode
with torch.no_grad():
    y_recon, q_z_mean, _, _, _, k_sample = model(test_trial)
    inferred_z = q_z_mean.cpu().numpy().squeeze()
    inferred_k = k_sample.cpu().numpy().squeeze()
    
# --- Plot 1: Inferred Latent States ---
# This plot shows the discovered dynamical structure.
plt.figure(figsize=(10, 8))
colors = sns.color_palette('hsv', n_colors=NUM_STATES)

for k in range(NUM_STATES):
    idx = inferred_k == k
    plt.plot(inferred_z[idx, 0], inferred_z[idx, 1], 'o', color=colors[k], 
                 markersize=4, label=f"Inferred State {k}")

plt.title(f"Discovered Latent Dynamics during '{train_state_key}' (Trial {trial_idx})")
plt.xlabel("Latent Dim 1")
plt.ylabel("Latent Dim 2")
plt.legend()
plt.show()


# --- Plot 2: Reconstructed CANN Activity ---
# This kymograph shows the model's internal representation of the
# continuous attractor state over time.
recon_cann_activity = []
model.cann.reset()
for t in range(TIME_STEPS_PER_TRIAL):
    I_ext = model.interface(q_z_mean[0, t, :])
    u = model.cann(I_ext.unsqueeze(0)).squeeze(0)
    recon_cann_activity.append(u.cpu().detach().numpy())

plt.figure(figsize=(12, 6))
plt.imshow(np.array(recon_cann_activity).T, aspect='auto', cmap='viridis')
plt.title("Model's Reconstructed CANN Activity (Kymograph)")
plt.ylabel("CANN Neuron #")
plt.xlabel("Time Step")
plt.colorbar(label="Activity")
plt.show()

