# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 12:00:45 2025

@author: HT_bo
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

# --- Print the columns again ---
print("Columns in the DataFrame:", df_good_clusters.columns)
# --------------------------------

# Sort by cluster ID
df_good_clusters = df_good_clusters.sort_values(by=['channel_depth_um'])

def plot_raster(df, ax, time_range=None, title_suffix=""):
    """
    Plots a raster plot of spike times for the given DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing cluster information, including 'cluster_id' and 'spike_times_seconds'.
        ax (matplotlib.axes.Axes): The axes object to plot on.
        time_range (tuple, optional): A tuple (start_time, end_time) to plot a specific time window. Defaults to None (plotting the whole duration).
        title_suffix (str, optional): Suffix to add to the plot title. Defaults to "".
    """
    ax.clear()  # Clear previous plot if any

    for i, row in df.iterrows():
        cluster_id = row['cluster_id']
        spike_times_seconds = np.array(row['spike_times_seconds'])

        if time_range:
            start_time, end_time = time_range
            # Filter spike times within the specified range
            valid_spikes = spike_times_seconds[(spike_times_seconds >= start_time) & (spike_times_seconds <= end_time)]
            ax.eventplot(valid_spikes, lineoffsets=i, linelengths=0.5, color='k')
        else:
            ax.eventplot(spike_times_seconds, lineoffsets=i, linelengths=0.5, color='k')

    # Set the y-axis ticks and labels to be the cluster IDs
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['cluster_id'].astype(str))
    ax.set_ylabel('Cluster ID')
    ax.set_xlabel('Time (s)')
    ax.set_title(f'Raster Plot of Good Clusters {title_suffix}')
    ax.tick_params(axis='y', left=False) # Remove y-axis ticks

    ax.set_xlim(time_range) if time_range else ax.set_xlim(df['spike_times_seconds'].apply(lambda x: np.min(x) if len(x) > 0 else 0).min(),
                                                          df['spike_times_seconds'].apply(lambda x: np.max(x) if len(x) > 0 else 1).max()) # Adjust x-axis limits

    plt.tight_layout()

# --- Option 1: Plot whole duration ---
def plot_whole_duration_raster(df):
    """Plots the raster plot for the entire duration of the recording."""
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_raster(df, ax, title_suffix="(Whole Duration)")
    plt.show()

# --- Option 2: Plot specific time ---
def plot_specific_time_raster(df, start_time, end_time):
    """Plots the raster plot for a specific time window."""
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_raster(df, ax, time_range=(start_time, end_time), title_suffix=f"(Time: {start_time:.2f}s to {end_time:.2f}s)")
    plt.show()

# --- User Interface for choosing the plotting option ---
while True:
    choice = input("Choose an option:\n1. Plot whole duration\n2. Plot specific time\n3. Exit\nEnter your choice (1, 2, or 3): ")

    if choice == '1':
        plot_whole_duration_raster(df_good_clusters)
        break
    elif choice == '2':
        try:
            start_time = float(input("Enter the start time (in seconds): "))
            end_time = float(input("Enter the end time (in seconds): "))
            if start_time >= end_time:
                print("Error: Start time must be less than end time.")
            else:
                plot_specific_time_raster(df_good_clusters, start_time, end_time)
                break
        except ValueError:
            print("Invalid input. Please enter numeric values for start and end times.")
    elif choice == '3':
        print("Exiting.")
        break
    else:
        print("Invalid choice. Please enter 1, 2, or 3.")