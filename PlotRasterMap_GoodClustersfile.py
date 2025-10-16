# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 19:51:57 2025
Modified to include file Browse for .npy file.

@author: HT_bo
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

# --- Import tkinter for file dialog ---
import tkinter as tk
from tkinter import filedialog
# --------------------------------------

# --- Function to ask user for the .npy file ---
def select_npy_file():
    """Opens a file dialog for the user to select a .npy file."""
    root = tk.Tk()
    root.withdraw() # Hide the main tkinter window
    
    filepath = filedialog.askopenfilename(
        title="Select the good_clusters .npy file",
        filetypes=[("Numpy files", "*.npy"), ("All files", "*.*")] # Filter for .npy
    )
    
    if not filepath: # If the user cancels
        print("No file selected. Exiting script.")
        sys.exit()
        
    return filepath
# ---------------------------------------------

# --- Get the file path from the user ---
data_filepath = select_npy_file()
print(f"Selected file: {data_filepath}")
# ----------------------------------------


# --- Load the good clusters data ---
try:
    # allow_pickle=True can be a security risk if loading from untrusted sources.
    # Only use it if you trust the origin of the .npy file.
    good_clusters_final = np.load(data_filepath, allow_pickle=True) 
    print(f"Successfully loaded '{os.path.basename(data_filepath)}'.")
    
# Catch FileNotFoundError (less likely now, but good practice)
except FileNotFoundError: 
    print(f"Error: The selected file could not be found at '{data_filepath}'.")
    sys.exit()
# Catch other potential errors during loading (e.g., corrupt file, wrong format)
except Exception as e: 
    print(f"Error loading file '{os.path.basename(data_filepath)}': {e}")
    sys.exit()
# ----------------------------------


# --- Check if data was loaded successfully ---
if not good_clusters_final.size:
    print(f"No good clusters found in '{os.path.basename(data_filepath)}'.")
    sys.exit() 
# -----------------------------------------


# --- Process the loaded data ---
# Explicitly create a list of dictionaries
# This section assumes 'good_clusters_final' is a structured numpy array or similar.
# Adjust if your .npy file has a different structure (e.g., already a list of dicts).
try:
    if good_clusters_final.dtype.names: # Check if it's a structured array
        list_of_dictionaries = [dict(zip(good_clusters_final.dtype.names, row)) for row in good_clusters_final]
    elif isinstance(good_clusters_final[0], dict): # Check if it's an array of dicts
        list_of_dictionaries = list(good_clusters_final)
    else:
         # Fallback or specific handling if it's neither (might need adjustment)
         # This assumes items can be converted to dict; might fail for other types.
         print("Warning: Attempting conversion from unknown .npy structure.")
         list_of_dictionaries = [dict(item) for item in good_clusters_final] 

    # Convert the list of dictionaries to a Pandas DataFrame
    df_good_clusters = pd.DataFrame(list_of_dictionaries)

except Exception as e:
    print(f"Error processing data structure from '{os.path.basename(data_filepath)}': {e}")
    print("Please ensure the .npy file contains structured data (like records or dicts) readable into a DataFrame.")
    sys.exit()
# ---------------------------------


# --- Print the columns ---
print("Columns in the DataFrame:", df_good_clusters.columns)
# --------------------------


# --- Sort the DataFrame ---
required_sort_cols = ['cluster_id'] # Add others like 'channel_depth_um', 'peak_channel_shank' if needed
if not all(col in df_good_clusters.columns for col in required_sort_cols):
    print(f"Error: DataFrame is missing one or more required columns for sorting: {required_sort_cols}")
    print(f"Available columns: {df_good_clusters.columns}")
    sys.exit()

# Sort by cluster ID (Current implementation)
df_good_clusters = df_good_clusters.sort_values(by=['cluster_id'])
# Example: Sort by depth, shank, then ID (uncomment and adjust required_sort_cols if using)
# required_sort_cols = ['channel_depth_um', 'peak_channel_shank', 'cluster_id'] 
# df_good_clusters = df_good_clusters.sort_values(by=required_sort_cols)
# --------------------------


# --- Plotting ---
fig, ax = plt.subplots(figsize=(20, 10)) # Adjust figure size as needed

y_position = 0
ytick_positions = []
ytick_labels = []

for i, row in df_good_clusters.iterrows():
    try:
        cluster_id = row['cluster_id']
        # Ensure spike_times_seconds is a numpy array and not empty/scalar
        spike_times_seconds = np.asarray(row['spike_times_seconds']) 

        if spike_times_seconds.ndim == 0 or spike_times_seconds.size == 0:
            print(f"Warning: Cluster {cluster_id} has no valid spike times. Skipping.")
            continue
            
        # Plot spikes
        ax.eventplot(spike_times_seconds, lineoffsets=y_position, linelengths=0.2, color='k')
        ytick_positions.append(y_position)
        ytick_labels.append(str(cluster_id))
        y_position += 1 # Increment y-position only for clusters that are plotted

    except KeyError as e:
        print(f"Error accessing column {e} for plotting. Check DataFrame structure.")
        sys.exit()
    except Exception as e:
        print(f"Error plotting data for cluster (ID might be unavailable): {e}")
        # Decide if you want to skip the cluster or exit
        continue # Skips this cluster

# Set the y-axis ticks and labels if data was plotted
if ytick_positions: 
    ax.set_yticks(ytick_positions)
    ax.set_yticklabels(ytick_labels)
else:
    print("No valid cluster data was plotted.")
    # Optionally close the empty plot
    plt.close(fig) 
    sys.exit()

# Use the filename in the title
plot_title = f'Raster Plot from {os.path.basename(data_filepath)} (Ordered by Cluster ID)'
# Update title if using different sorting
# plot_title = f'Raster Plot from {os.path.basename(data_filepath)} (Ordered by Depth/Shank/ID)' 

ax.set_ylabel('Cluster ID') 
ax.set_xlabel('Time (s)')
ax.set_title(plot_title) 
ax.tick_params(axis='y', left=True) # Keep y-axis ticks

plt.tight_layout()
plt.show()
# ---------------