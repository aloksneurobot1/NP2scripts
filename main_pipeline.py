#%% [markdown]
# # Main Pipeline for SLDS-CANN VAE Model
# This script loads neural data, preprocesses it, trains the hybrid VAE model,
# and visualizes the inferred latent dynamics.

#%%
# =============================================================================
# CELL 1: IMPORTS AND CONFIGURATION
# =============================================================================
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm import tqdm
import os

# Custom modules
from hybrid_model import SLDS_CANN_VAE

# --- Parameters ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data Parameters
SPIKE_DATA_PATH = "path/to/your/spike_data.npy"  # UPDATE THIS
TIMESTAMPS_PATH = "path/to/your/timestamps.npy" # UPDATE THIS
TIME_STEPS_PER_TRIAL = 100 # Length of each trial sequence

# Model Hyperparameters
OBS_DIM = 20              # Dimensionality of PCA-reduced data
LATENT_DIM = 2            # Dimensionality of the continuous latent space (z)
NUM_STATES = 4            # Number of discrete latent states (k)
CANN_DIM = 128            # Number of neurons in the CANN
RNN_HIDDEN_DIM = 64       # Hidden dimension for the encoder RNN

# Training Parameters
LEARNING_RATE = 1e-3
NUM_EPOCHS = 50
BATCH_SIZE = 16

#%%
# =============================================================================
# CELL 2: DATA LOADING AND PREPROCESSING
# =============================================================================
print("Loading and preprocessing data...")

try:
    # Load real data
    spike_data = np.load(SPIKE_DATA_PATH) # Shape: (n_time_bins, n_neurons)
    timestamps = np.load(TIMESTAMPS_PATH) # Shape: (n_time_bins,) with state labels
    print(f"Successfully loaded data from {SPIKE_DATA_PATH}")

except FileNotFoundError:
    # Generate placeholder data if real data is not found
    print("Warning: Data files not found. Generating random placeholder data.")
    n_neurons_raw = 250
    n_total_time_bins = 5000
    spike_data = np.random.randn(n_total_time_bins, n_neurons_raw)
    timestamps = np.random.randint(0, NUM_STATES, size=n_total_time_bins)

# --- PCA for Dimensionality Reduction ---
# 1. Standardize the data (important for PCA)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(spike_data)

# 2. Fit and transform with PCA
pca = PCA(n_components=OBS_DIM)
data_pca = pca.fit_transform(scaled_data)
print(f"Data dimensionality reduced to {OBS_DIM} components.")
print(f"Explained variance by {OBS_DIM} components: {np.sum(pca.explained_variance_ratio_):.2f}")

# --- Segmentation into Trials by Behavioral State ---
trials_by_state = {}
unique_states = np.unique(timestamps)

for state in unique_states:
    # Find data corresponding to the current state
    state_data = data_pca[timestamps == state]

    # Reshape into non-overlapping trials
    num_trials = len(state_data) // TIME_STEPS_PER_TRIAL
    if num_trials > 0:
        trimmed_data = state_data[:num_trials * TIME_STEPS_PER_TRIAL]
        trials = trimmed_data.reshape(num_trials, TIME_STEPS_PER_TRIAL, OBS_DIM)
        trials_by_state[state] = torch.tensor(trials, dtype=torch.float32).to(DEVICE)
        print(f"State {state}: Created {num_trials} trials of length {TIME_STEPS_PER_TRIAL}")

# Ensure we have data to train on
if not trials_by_state:
    raise ValueError("No trials could be created. Check data paths and trial length.")

#%%
# =============================================================================
# CELL 3: MODEL INITIALIZATION AND TRAINING
# =============================================================================
# --- Select Data for Training ---
# Here, we'll just use the data from the first available state.
# You can modify this to select a specific state, e.g., `training_data = trials_by_state[desired_state]`
training_state = list(trials_by_state.keys())[0]
training_data = trials_by_state[training_state]
print(f"\nTraining model on data from state: {training_state}")

# --- Initialize Model and Optimizer ---
model = SLDS_CANN_VAE(
    obs_dim=OBS_DIM,
    latent_dim=LATENT_DIM,
    num_states=NUM_STATES,
    cann_dim=CANN_DIM,
    rnn_hidden_dim=RNN_HIDDEN_DIM
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- Training Loop ---
print("Starting training...")
losses = []
for epoch in tqdm(range(NUM_EPOCHS), desc="Epochs"):
    model.train()
    epoch_loss = 0

    # Create a DataLoader for batching
    train_loader = torch.utils.data.DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)

    for batch in train_loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()

        # Forward pass
        y_recon, z_sample, k_sample, posterior_params = model(batch)

        # Compute loss
        loss = model.compute_loss(batch, y_recon, posterior_params, z_sample, k_sample)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader.dataset)
    losses.append(avg_loss)
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Average Loss: {avg_loss:.4f}")

print("Training complete.")

# Plot training loss
plt.figure(figsize=(8, 4))
plt.plot(losses)
plt.title("Training Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Average Loss")
plt.show()

#%%
# =============================================================================
# CELL 4: ANALYSIS AND VISUALIZATION
# =============================================================================
print("\nPerforming analysis and visualization...")
model.eval()

# --- Select a single trial for analysis ---
test_trial = training_data[0:1].to(DEVICE) # Take the first trial, keep batch dim

# --- Run the model to get inferred states ---
with torch.no_grad():
    y_recon, z_sample, k_sample, _ = model(test_trial)

# Move results to CPU for plotting
z_inferred = z_sample.squeeze(0).cpu().numpy()
k_inferred = k_sample.squeeze(0).cpu().numpy()
y_recon_np = y_recon.squeeze(0).cpu().numpy()

# --- Plot 1: Scatter plot of inferred latent states (z) ---
plt.figure(figsize=(8, 6))
scatter = plt.scatter(z_inferred[:, 0], z_inferred[:, 1], c=k_inferred, cmap='viridis', s=15)
plt.title("Inferred Latent States (z) Colored by Discrete State (k)")
plt.xlabel("Latent Dimension 1")
plt.ylabel("Latent Dimension 2")
plt.legend(handles=scatter.legend_elements()[0], labels=range(NUM_STATES), title="States")
plt.grid(True)
plt.show()

# --- Plot 2: Kymograph of reconstructed CANN activity ---
# To get the CANN activity, we need to inspect the model's internal state.
# We'll re-run the generative part of the model step-by-step.
cann_activity = []
model.cann.reset(batch_size=1)
with torch.no_grad():
    for t in range(TIME_STEPS_PER_TRIAL):
        z_t = z_sample[:, t, :]
        I_ext = model.interface(z_t)
        cann_state = model.cann.forward(I_ext)
        cann_activity.append(cann_state.squeeze(0).cpu().numpy())

cann_activity = np.array(cann_activity) # Shape: (time_steps, cann_dim)

plt.figure(figsize=(10, 6))
plt.imshow(cann_activity.T, aspect='auto', cmap='viridis', origin='lower')
plt.colorbar(label="CANN Neuron Activity")
plt.title("Kymograph of Reconstructed CANN Activity")
plt.xlabel("Time Step in Trial")
plt.ylabel("CANN Neuron Index")
plt.show()