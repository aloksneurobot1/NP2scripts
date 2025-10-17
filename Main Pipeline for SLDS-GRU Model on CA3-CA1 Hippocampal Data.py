# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 09:27:19 2025

@author: Alok
"""

# -*- coding: utf-8 -*-
"""
=======================================================================
 Main Pipeline for SLDS-GRU Model on CA3-CA1 Hippocampal Data
=======================================================================
This script trains the SLDS-GRU VAE on CA3 data to learn its dynamics,
and then tests how well the learned CA3 representations can predict
the observed activity in CA1.
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
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

from hybrid_model_gru import SLDS_GRU_VAE

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- Parameters for dimensionality reduction ---
CA3_OBS_DIM = 40
CA1_OBS_DIM = 40

# --- Model & Training Parameters ---
SLDS_LATENT_DIM = 2
GRU_HIDDEN_DIM = 64
NUM_STATES = 3
RNN_HIDDEN_DIM = 128
TIME_STEPS_PER_TRIAL = 200
LEARNING_RATE = 1e-3
NUM_EPOCHS = 200

# %% ===================================================================
#   2. DATA LOADING AND PREPROCESSING
# =====================================================================
print("Loading and preprocessing experimental data for CA3 and CA1...")

# !!! ACTION REQUIRED: Replace with the actual paths to your data files !!!
CA3_SPIKE_PATH = "path/to/your/CA3_binned_spikes.npy"
CA1_SPIKE_PATH = "path/to/your/CA1_binned_spikes.npy"
TIMESTAMPS_PATH = "path/to/your/behavioral_timestamps.npy"

try:
    ca3_spikes = np.load(CA3_SPIKE_PATH)
    ca1_spikes = np.load(CA1_SPIKE_PATH)
    timestamps = np.load(TIMESTAMPS_PATH)
    # Ensure data is of the same length
    min_len = min(len(ca3_spikes), len(ca1_spikes), len(timestamps))
    ca3_spikes, ca1_spikes, timestamps = ca3_spikes[:min_len], ca1_spikes[:min_len], timestamps[:min_len]
except FileNotFoundError:
    print("="*60 + "\n!!! DATA FILES NOT FOUND !!!\nUsing placeholder random data.\n" + "="*60)
    ca3_spikes = np.random.randn(50000, 200)
    ca1_spikes = np.random.randn(50000, 180)
    timestamps = np.zeros(50000); timestamps[25000:] = 1

# --- Step 2a: PCA for CA3 and CA1 separately ---
pca_ca3 = PCA(n_components=CA3_OBS_DIM, whiten=True)
ca3_pca = pca_ca3.fit_transform(ca3_spikes)

pca_ca1 = PCA(n_components=CA1_OBS_DIM, whiten=True)
ca1_pca = pca_ca1.fit_transform(ca1_spikes)

# --- Step 2b: Segment Data by Behavioral State ---
segmented_ca3 = {}
segmented_ca1 = {}
for state_label in np.unique(timestamps):
    state_name = f"state_{int(state_label)}"
    print(f"Segmenting data for {state_name}...")
    
    state_indices = np.where(timestamps == state_label)[0]
    
    num_trials = len(state_indices) // TIME_STEPS_PER_TRIAL
    if num_trials == 0: continue
    
    # Segment CA3 data
    trials_ca3 = ca3_pca[state_indices][:num_trials * TIME_STEPS_PER_TRIAL].reshape(num_trials, TIME_STEPS_PER_TRIAL, CA3_OBS_DIM)
    segmented_ca3[state_name] = torch.tensor(trials_ca3, dtype=torch.float32).to(DEVICE)
    
    # Segment CA1 data
    trials_ca1 = ca1_pca[state_indices][:num_trials * TIME_STEPS_PER_TRIAL].reshape(num_trials, TIME_STEPS_PER_TRIAL, CA1_OBS_DIM)
    segmented_ca1[state_name] = torch.tensor(trials_ca1, dtype=torch.float32)

print("\nData loading and segmentation complete.")


# %% ===================================================================
#   3. MODEL INITIALIZATION AND TRAINING ON CA3 DATA
# =====================================================================
train_state_key = list(segmented_ca3.keys())[0]
data_to_train = segmented_ca3[train_state_key]

print(f"\nTraining the SLDS-GRU VAE model on CA3 data from: {train_state_key}...")
model = SLDS_GRU_VAE(CA3_OBS_DIM, SLDS_LATENT_DIM, GRU_HIDDEN_DIM, NUM_STATES, RNN_HIDDEN_DIM).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

pbar = tqdm(range(NUM_EPOCHS))
for epoch in pbar:
    optimizer.zero_grad()
    y_recon, q_z_mean, q_z_logvar, q_k_dist, z_sample, k_sample, _ = model(data_to_train)
    loss = model.compute_loss(data_to_train, y_recon, q_z_mean, q_z_logvar, q_k_dist, z_sample, k_sample)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0: pbar.set_description(f"Epoch {epoch}, Loss: {loss.item():.2f}")

print("Training complete.")


# %% ===================================================================
#   4. ANALYSIS 1: VISUALIZE INFERRED CA3 DYNAMICS
# =====================================================================
print("\nVisualizing inferred latent dynamics from CA3...")
model.eval()
with torch.no_grad():
    _, q_z_mean, _, _, _, k_sample, _ = model(data_to_train)

trial_idx = 0
inferred_z = q_z_mean[trial_idx].cpu().numpy()
inferred_k = k_sample[trial_idx].cpu().numpy()

plt.figure(figsize=(10, 8))
colors = sns.color_palette('hsv', n_colors=NUM_STATES)
for k in range(NUM_STATES):
    idx = inferred_k == k
    plt.plot(inferred_z[idx, 0], inferred_z[idx, 1], 'o', ms=4, label=f"State {k}")
plt.title(f"Discovered CA3 Latent Dynamics during '{train_state_key}'")
plt.xlabel("SLDS Latent Dim 1"); plt.ylabel("SLDS Latent Dim 2"); plt.legend()
plt.show()

# %% ===================================================================
#   5. ANALYSIS 2: DECODE CA1 ACTIVITY FROM INFERRED CA3 REPRESENTATIONS
# =====================================================================
print("\nTesting if inferred CA3 dynamics can predict CA1 activity...")

# Get the inferred CA3 working memory state (GRU hidden state)
with torch.no_grad():
    _, _, _, _, _, _, inferred_h_ca3 = model(data_to_train)

# Reshape data for scikit-learn
X = inferred_h_ca3.cpu().numpy().reshape(-1, GRU_HIDDEN_DIM)
y = segmented_ca1[train_state_key].numpy().reshape(-1, CA1_OBS_DIM)

# Use regularized linear regression (Ridge) with cross-validation
decoder = Ridge(alpha=1.0)
# cv=5 means 5-fold cross-validation. The score is R^2 (variance explained).
scores = cross_val_score(decoder, X, y, cv=5, scoring='r2')

print(f"\n--- CA3-to-CA1 Decoding Results ---")
print(f"Mean R^2 (variance explained) across 5 folds: {np.mean(scores):.3f}")
print(f"Std Dev of R^2 across 5 folds: {np.std(scores):.3f}")

# For comparison, let's decode from a time-shuffled control
X_shuffled = X.copy()
np.random.shuffle(X_shuffled)
shuffled_scores = cross_val_score(decoder, X_shuffled, y, cv=5, scoring='r2')
print(f"\nShuffled Control Mean R^2: {np.mean(shuffled_scores):.3f}")

plt.figure(figsize=(6, 5))
sns.barplot(x=['Actual', 'Shuffled'], y=[np.mean(scores), np.mean(shuffled_scores)])
plt.ylabel("R^2 (Variance Explained)")
plt.title("Decoding CA1 Activity from CA3 Latent State")
plt.ylim(min(0, np.mean(shuffled_scores) - 0.1), max(0.1, np.mean(scores) * 1.2))
plt.show()
