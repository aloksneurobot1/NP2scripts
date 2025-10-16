# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 13:49:29 2025

@author: HT_bo
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from matplotlib.colors import LinearSegmentedColormap

# Load the good clusters data
try:
    good_clusters_final = np.load('good_clusters_final.npy', allow_pickle=True)
except FileNotFoundError:
    print("Error: 'good_clusters_final.npy' not found in the current directory.")
    exit()

if not good_clusters_final.size:
    print("No good clusters found in 'good_clusters_final.npy'.")
    exit()

# Explicitly create a list of dictionaries
list_of_dictionaries = [dict(item) for item in good_clusters_final]

# Convert the list of dictionaries to a Pandas DataFrame
df_good_clusters = pd.DataFrame(list_of_dictionaries)

# --- Use the correct column names ---
if 'peak_channel_shank' not in df_good_clusters.columns or 'spike_times_seconds' not in df_good_clusters.columns or 'probe_area' not in df_good_clusters.columns:
    print("Error: DataFrame must contain columns 'peak_channel_shank', 'spike_times_seconds', and 'probe_area'.")
    print("Columns found:", df_good_clusters.columns)
    exit()

# Filter for shank 1
df_shank1 = df_good_clusters[df_good_clusters['peak_channel_shank'] == 1].copy()

if df_shank1.empty:
    print("No clusters found for shank 1.")
    exit()

# Filter for superficial units in probe area
df_superficial_shank1 = df_shank1[df_shank1['probe_area'] == 'Superficial'].copy()

if df_superficial_shank1.empty:
    print("No superficial clusters found in shank 1 based on 'probe_area'.")
    exit()

# Determine the overall time range for binning
all_spike_times_superficial = np.concatenate(df_superficial_shank1['spike_times_seconds'].values)
if all_spike_times_superficial.size == 0:
    print("No spike times found for superficial clusters in shank 1.")
    exit()
start_time = np.min(all_spike_times_superficial)
end_time = np.max(all_spike_times_superficial)

# Define bin parameters
bin_size_ms = 100
bin_size_s = bin_size_ms / 1000.0
sampling_rate = 1 / bin_size_s
window_size_ms = 1000
step_size_ms = 500
window_size_bins = int(window_size_ms / bin_size_ms)
step_size_bins = int(step_size_ms / bin_size_ms)

# Create time bins
bins = np.arange(start_time, end_time + bin_size_s, bin_size_s)
bin_centers = bins[:-1] + bin_size_s / 2

# Bin the spike data for each superficial cluster in shank 1
binned_data = {}
for index, row in df_superficial_shank1.iterrows():
    cluster_id = row['cluster_id']
    spike_times = np.array(row['spike_times_seconds'])
    spike_counts, _ = np.histogram(spike_times, bins=bins)
    binned_data[cluster_id] = spike_counts

# Convert binned data to a Pandas DataFrame for easier handling
df_binned = pd.DataFrame.from_dict(binned_data, orient='index')
df_binned.columns = bin_centers

# Smoothen the binned data using a sliding window
smoothed_data = {}
for cluster_id, spike_counts in df_binned.iterrows():
    smoothed_counts = np.convolve(spike_counts, np.ones(window_size_bins)/window_size_bins, mode='same')
    smoothed_data[cluster_id] = smoothed_counts

df_smoothed = pd.DataFrame.from_dict(smoothed_data, orient='index')
df_smoothed.columns = bin_centers

# Extract the smoothed data for superficial clusters in shank 1
superficial_cluster_ids = df_superficial_shank1['cluster_id'].tolist()
smoothed_data_superficial = df_smoothed.loc[superficial_cluster_ids]

# Perform PCA
pca = PCA(n_components=3)
principal_components = pca.fit_transform(smoothed_data_superficial.T) # Transpose so time is rows

# Create the 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Color code the trajectory based on time (light blue to dark blue)
num_time_points = principal_components.shape[0]
# Define a colormap from light blue to dark blue
colors_array = ["lightblue", "darkblue"]
cmap = LinearSegmentedColormap.from_list("my_blue", colors_array)
normalized_time = np.linspace(0, 1, num_time_points)
colors = cmap(normalized_time)

# Plot the trajectory points with color gradient using scatter
scatter = ax.scatter(principal_components[:, 0], principal_components[:, 1], principal_components[:, 2], c=colors, cmap=cmap)

# Plot lines connecting the points
ax.plot(principal_components[:, 0], principal_components[:, 1], principal_components[:, 2], color='gray', linewidth=0.5) # Use a neutral color for the lines

# Add a colorbar as a legend for time
cbar = fig.colorbar(scatter, label='Time (s)')

ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.set_title('PCA Trajectory of Superficial Units in Shank 1') # Updated title

# Enable interactive 3D plot (should be default in many backends)
plt.show()

print("\nNote:")
print("The 3D plot window should be interactive, allowing you to rotate, pan, and zoom.")
print("If you are not seeing this behavior, it might depend on your Matplotlib backend.")
print("For interactive plots in some environments (like Spyder), you might need to ensure you are not in a blocking state.")