# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 19:51:57 2025
Plot median waveform, raster, and firing rate for a specific cluster,
including annotations and options for time duration.
Uses median_peak_waveforms.npy.

@author: HT_bo
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import math # Added for ceiling function in binning

# --- Import tkinter for file dialog ---
# Check if running in an environment where tkinter is available
try:
    import tkinter as tk
    from tkinter import filedialog
    TKINTER_AVAILABLE = True
except ImportError:
    print("Warning: tkinter module not found. File dialogs will not be available.")
    print("Please provide file paths directly if needed.")
    TKINTER_AVAILABLE = False
# --------------------------------------

# ==============================================================
# <<< CONFIGURATION >>>

# --- !!! ADJUST THESE VARIABLES IF NEEDED !!! ---

# Set this to the actual name of the column in your data
# that contains the brain region information.
REGION_COLUMN_NAME = 'brain_region_name' # Examples: 'acronym', 'location', 

# Set this to the actual name of the column for L-Ratio
L_RATIO_COLUMN_NAME = 'l_ratio' 

# Set this to the actual name of the column for number of spikes
NUM_SPIKES_COLUMN_NAME = 'num_spikes' 

# Set this to the actual name of the column for shank index/ID
# If you don't have shank info, set it to None
SHANK_COLUMN_NAME = 'shank_index' 

# Set the sampling rate of your recording (samples per second)
# Needed for waveform time axis.
SAMPLING_RATE_HZ = 30000.36276

# Name of the median waveforms file (expected in the same directory)
MEDIAN_WAVEFORMS_FILENAME = 'median_peak_waveforms.npy'

# Firing rate bin size in milliseconds
FIRING_RATE_BIN_MS = 100

# Set the cluster ID you want to plot
# This will be asked from the user when the script runs.

# --- END OF ADJUSTABLE VARIABLES ---

# ==============================================================


# --- Function to ask user for the .npy file ---
def select_npy_file(title="Select the good_clusters .npy file"):
    """Opens a file dialog for the user to select a .npy file."""
    if not TKINTER_AVAILABLE:
        filepath = input(f"Enter the full path to the '{title}' file: ")
        if not os.path.exists(filepath):
             print(f"Error: File not found at '{filepath}'. Exiting.")
             sys.exit()
        return filepath

    root = tk.Tk()
    root.withdraw() # Hide the main tkinter window

    filepath = filedialog.askopenfilename(
        title=title,
        filetypes=[("Numpy files", "*.npy"), ("All files", "*.*")] # Filter for .npy
    )

    if not filepath: # If the user cancels
        print("No file selected. Exiting script.")
        sys.exit()

    return filepath
# ---------------------------------------------

# --- Function to plot specific cluster data ---
def plot_cluster_data(cluster_id, df_clusters, median_waveforms, sampling_rate,
                      start_time=None, end_time=None, bin_width_s=0.1):
    """
    Plots the median waveform, raster plot, and firing rate for a specific cluster ID.

    Args:
        cluster_id (int): The ID of the cluster to plot.
        df_clusters (pd.DataFrame): DataFrame containing cluster metadata.
        median_waveforms (np.ndarray): Array containing median waveforms (indexed by cluster_id).
        sampling_rate (float): The sampling rate in Hz.
        start_time (float, optional): Start time for filtering raster and firing rate. Defaults to None.
        end_time (float, optional): End time for filtering raster and firing rate. Defaults to None.
        bin_width_s (float, optional): Bin width in seconds for firing rate calculation. Defaults to 0.1 (100ms).
    """
    print(f"\nAttempting to plot data for Cluster ID: {cluster_id}")
    if start_time is not None and end_time is not None:
        print(f"Filtering data between {start_time:.2f}s and {end_time:.2f}s")

    # --- Find the data for the specified cluster ---
    cluster_row = df_clusters[df_clusters['cluster_id'] == cluster_id]

    if cluster_row.empty:
        print(f"Error: Cluster ID {cluster_id} not found in the loaded data.")
        valid_ids = df_clusters['cluster_id'].unique()
        print(f"Available Cluster IDs: {valid_ids[:20]}..." if len(valid_ids)>20 else valid_ids) # Show some example IDs
        return # Exit the function if cluster not found

    # Extract data from the first (and only) row found
    cluster_data = cluster_row.iloc[0]

    # --- Safely extract metadata ---
    try:
        region = cluster_data.get(REGION_COLUMN_NAME, 'N/A') # Use .get for safety
        num_spikes_total = cluster_data.get(NUM_SPIKES_COLUMN_NAME, 'N/A') # Get total spikes before filtering
        l_ratio = cluster_data.get(L_RATIO_COLUMN_NAME, np.nan) # Default to NaN if missing
        spike_times_seconds_all = np.asarray(cluster_data['spike_times_seconds']) # Assume this exists

        if SHANK_COLUMN_NAME:
             shank = cluster_data.get(SHANK_COLUMN_NAME, 'N/A')
        else:
             shank = 'N/A' # If shank column is not specified

        # Format L-Ratio
        l_ratio_str = f"{l_ratio:.2f}" if pd.notna(l_ratio) else "N/A"

    except KeyError as e:
         print(f"Error: Missing expected column in DataFrame: {e}. Cannot retrieve all metadata.")
         print(f"Available columns: {list(df_clusters.columns)}")
         return
    except Exception as e:
        print(f"Error extracting metadata for cluster {cluster_id}: {e}")
        return

    # --- Extract the median waveform ---
    try:
        # IMPORTANT ASSUMPTION: Assumes median_waveforms array is indexed directly by cluster_id.
        # If cluster_ids are non-sequential or don't start at 0, this might fail
        # or require a different mapping strategy.
        median_waveform = median_waveforms[cluster_id] # Extracts the 82 uV values
    except IndexError:
        print(f"Error: Cannot access waveform for Cluster ID {cluster_id}.")
        print(f"The median_waveforms array has shape {median_waveforms.shape}.")
        print(f"Check if cluster IDs in metadata file match the indices (0 to {median_waveforms.shape[0]-1}) of '{MEDIAN_WAVEFORMS_FILENAME}'.")
        return
    except Exception as e:
        print(f"Error accessing waveform for cluster {cluster_id}: {e}")
        return

    # --- Prepare data for plotting ---
    # Time axis for waveform (in milliseconds) - Calculated based on sample count and rate
    num_samples = median_waveform.shape[0] # Should be 82 based on your info
    time_ms = np.arange(num_samples) / sampling_rate * 1000 # X-axis values

    # Check if spike times are valid and filter if needed
    if not isinstance(spike_times_seconds_all, np.ndarray) or spike_times_seconds_all.ndim == 0:
         print(f"Warning: Cluster {cluster_id} has no valid spike times in metadata. Skipping raster and firing rate plots.")
         plot_raster_fr = False
         spike_times_seconds = np.array([]) # Ensure it's an empty array
    else:
        # Further check if it became a 0-d array after loading (e.g., containing a single float)
        if spike_times_seconds_all.ndim == 0:
             spike_times_seconds_all = np.array([spike_times_seconds_all.item()]) # Convert to 1D array

        # Filter spike times based on provided duration
        if start_time is not None and end_time is not None:
            spike_times_seconds = spike_times_seconds_all[
                (spike_times_seconds_all >= start_time) & (spike_times_seconds_all <= end_time)
            ]
            # Determine plot limits based on user input
            plot_start_time = start_time
            plot_end_time = end_time
        else:
            spike_times_seconds = spike_times_seconds_all
            # Determine plot limits based on actual spike times if no duration given
            if spike_times_seconds.size > 0:
                plot_start_time = np.min(spike_times_seconds)
                plot_end_time = np.max(spike_times_seconds)
            else:
                plot_start_time = 0
                plot_end_time = 1 # Default if no spikes

        # Check if there are spikes left after filtering
        if spike_times_seconds.size == 0:
            print(f"Warning: No spikes found for Cluster {cluster_id} in the specified time range"
                  if (start_time is not None and end_time is not None) else
                  f"Warning: Cluster {cluster_id} has zero spikes.")
            plot_raster_fr = False
        else:
            plot_raster_fr = True

    # --- Create the plot ---
    # Now using 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), gridspec_kw={'width_ratios': [1, 2, 2]}) # Waveform, Raster, Firing Rate

    # 1. Plot Median Waveform (Uses raw uV values from file for Y-axis)
    ax_wave = axes[0]
    ax_wave.plot(time_ms, median_waveform, color='black', linewidth=1.5) # median_waveform provides Y values directly
    ax_wave.set_title('Median Spike Waveform')
    ax_wave.set_xlabel('Time (ms)')
    ax_wave.set_ylabel('Amplitude (uV)') # Unit confirmed from user info
    ax_wave.grid(True, linestyle='--', alpha=0.6)

    # 2. Plot Raster
    ax_rast = axes[1]
    if plot_raster_fr:
        # Plot spikes using eventplot
        ax_rast.eventplot(spike_times_seconds, color='black', linelengths=0.8)
        ax_rast.set_title('Raster Plot')
        ax_rast.set_xlabel('Time (s)')
        # Hide Y-axis ticks and labels for raster as it's a single cluster
        ax_rast.set_yticks([])
        ax_rast.set_ylabel('Spikes') # Indicate what the vertical lines are
        # Set x-limits based on specified duration or spike range
        ax_rast.set_xlim(left=plot_start_time, right=plot_end_time)
    else:
        # Display message if raster cannot be plotted
        no_spikes_message = "No spikes in selected range" if (start_time is not None and end_time is not None) else "No spikes available"
        ax_rast.text(0.5, 0.5, no_spikes_message, ha='center', va='center', transform=ax_rast.transAxes)
        ax_rast.set_title('Raster Plot')
        ax_rast.set_xlabel('Time (s)')
        ax_rast.set_yticks([])
        ax_rast.set_xlim(left=start_time if start_time is not None else 0,
                         right=end_time if end_time is not None else 1) # Set limits even if empty

    # 3. Plot Firing Rate
    ax_fr = axes[2]
    if plot_raster_fr:
        # Define bins for the histogram
        # Ensure bins cover the full range, even if slightly extending beyond start/end time
        # Add small epsilon to include endpoint if it falls exactly on a bin edge
        epsilon = 1e-9
        hist_start = plot_start_time
        hist_end = plot_end_time + epsilon
        num_bins = max(1, math.ceil((hist_end - hist_start) / bin_width_s)) # Ensure at least one bin
        bins = np.linspace(hist_start, hist_start + num_bins * bin_width_s, num_bins + 1)

        # Calculate histogram (spike counts per bin)
        spike_counts, bin_edges = np.histogram(spike_times_seconds, bins=bins)

        # Calculate firing rate (counts / bin width)
        firing_rate_hz = spike_counts / bin_width_s

        # Calculate bin centers for plotting
        bin_centers = bin_edges[:-1] + bin_width_s / 2

        # Plot firing rate
        ax_fr.plot(bin_centers, firing_rate_hz, marker='o', linestyle='-', color='crimson', markersize=4)
        ax_fr.set_title(f'Firing Rate ({int(bin_width_s*1000)} ms bins)')
        ax_fr.set_xlabel('Time (s)')
        ax_fr.set_ylabel('Firing Rate (Hz)')
        ax_fr.set_xlim(left=plot_start_time, right=plot_end_time) # Match raster x-axis
        ax_fr.grid(True, linestyle='--', alpha=0.6)
        # Optionally set y-limit starting from 0
        ax_fr.set_ylim(bottom=0)

    else:
        # Display message if firing rate cannot be plotted
        ax_fr.text(0.5, 0.5, "Cannot calculate firing rate", ha='center', va='center', transform=ax_fr.transAxes)
        ax_fr.set_title(f'Firing Rate ({int(bin_width_s*1000)} ms bins)')
        ax_fr.set_xlabel('Time (s)')
        ax_fr.set_ylabel('Firing Rate (Hz)')
        ax_fr.set_yticks([])
        ax_fr.set_xlim(left=start_time if start_time is not None else 0,
                       right=end_time if end_time is not None else 1) # Match raster x-axis

    # --- Add Annotations to the Figure ---
    # Use total number of spikes before filtering for annotation
    num_spikes_str = str(num_spikes_total) if num_spikes_total != 'N/A' else 'N/A'
    info_text = (
        f"Region: {region} | Total Spikes: {num_spikes_str}\n"
        f"L-Ratio: {l_ratio_str} | Shank: {shank}"
    )
    # Place text above the plots using figtext (coordinates relative to figure)
    plt.figtext(0.5, 0.98, info_text, ha='center', va='top', fontsize=10, wrap=True)

    # Add an overall title for the figure
    duration_str = f" ({start_time:.2f}s - {end_time:.2f}s)" if start_time is not None and end_time is not None else ""
    fig.suptitle(f'Analysis for Cluster ID: {cluster_id}{duration_str}', fontsize=14, y=1.04) # Adjust y to make space for figtext

    # Adjust layout to prevent titles/labels overlapping
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust rect to make space for suptitle and figtext

    print(f"Displaying plot for Cluster ID: {cluster_id}")
    plt.show()

# ==============================================================
# <<< MAIN SCRIPT EXECUTION >>>
# ==============================================================

if __name__ == "__main__":
    # --- Get the cluster metadata file path from the user ---
    data_filepath = select_npy_file("Select the Cluster Metadata (.npy) file")
    print(f"Selected metadata file: {data_filepath}")
    data_dir = os.path.dirname(data_filepath) # Get directory of the selected file
    # ----------------------------------------

    # --- Construct path for the median waveforms file ---
    waveforms_filepath = os.path.join(data_dir, MEDIAN_WAVEFORMS_FILENAME)
    print(f"Looking for waveforms file at: {waveforms_filepath}")
    # --------------------------------------------------

    # --- Load the good clusters metadata ---
    try:
        # Use allow_pickle=True cautiously, ensure the source is trusted
        good_clusters_struct = np.load(data_filepath, allow_pickle=True)
        print(f"Successfully loaded metadata '{os.path.basename(data_filepath)}'.")
    except FileNotFoundError:
        print(f"Error: The metadata file could not be found at '{data_filepath}'.")
        sys.exit()
    except Exception as e:
        print(f"Error loading metadata file '{os.path.basename(data_filepath)}': {e}")
        sys.exit()
    # ----------------------------------

    # --- Load the median waveforms ---
    try:
        median_waveforms = np.load(waveforms_filepath)
        print(f"Successfully loaded waveforms '{MEDIAN_WAVEFORMS_FILENAME}' with shape {median_waveforms.shape}.")
    except FileNotFoundError:
        print(f"Error: The waveforms file '{MEDIAN_WAVEFORMS_FILENAME}' was not found in the same directory as the metadata file.")
        print(f"Expected location: '{waveforms_filepath}'")
        sys.exit()
    except Exception as e:
        print(f"Error loading waveforms file '{MEDIAN_WAVEFORMS_FILENAME}': {e}")
        sys.exit()
    # -----------------------------

    # --- Check if metadata was loaded successfully ---
    if not good_clusters_struct.size:
        print(f"No cluster data found in '{os.path.basename(data_filepath)}'.")
        sys.exit()
    # -----------------------------------------

    # --- Process the loaded metadata into DataFrame ---
    try:
        # Check if it's a structured array
        if good_clusters_struct.dtype.names:
            list_of_dictionaries = [dict(zip(good_clusters_struct.dtype.names, row)) for row in good_clusters_struct]
        # Check if it's an array of dictionaries (object type)
        elif isinstance(good_clusters_struct[0], dict):
             list_of_dictionaries = list(good_clusters_struct)
        else:
             # Attempt conversion assuming it's an array of tuples/lists representing dicts
             print("Warning: Attempting conversion from unknown .npy structure to DataFrame.")
             # This part might need adjustment based on the actual unknown structure
             try:
                 # Example: If it's array of tuples and you know the column order
                 # column_names = ['cluster_id', 'spike_times_seconds', ...] # Define expected columns
                 # list_of_dictionaries = [dict(zip(column_names, item)) for item in good_clusters_struct]
                 # A more generic guess (might fail or produce unusable dicts):
                  list_of_dictionaries = [{f'col_{i}': val for i, val in enumerate(item)} for item in good_clusters_struct]
                  print("Warning: Used generic column names ('col_0', 'col_1', ...). Adjust CONFIGURATION section accordingly.")
             except Exception as convert_e:
                  print(f"Error: Could not automatically convert the structure from '{os.path.basename(data_filepath)}'. Please check the file format. Error: {convert_e}")
                  sys.exit()


        df_good_clusters = pd.DataFrame(list_of_dictionaries)
        print(f"Successfully processed metadata into DataFrame with {len(df_good_clusters)} entries.")

    except Exception as e:
        print(f"Error processing data structure from '{os.path.basename(data_filepath)}' into DataFrame: {e}")
        print("Please check the structure of your .npy file.")
        sys.exit()
    # ---------------------------------

    # --- Verify essential columns exist ---
    # Define columns essential FOR THIS SCRIPT's core functionality
    essential_cols = ['cluster_id', 'spike_times_seconds']
    annotation_cols = [] # Columns needed only for annotations
    if REGION_COLUMN_NAME: annotation_cols.append(REGION_COLUMN_NAME)
    if L_RATIO_COLUMN_NAME: annotation_cols.append(L_RATIO_COLUMN_NAME)
    if NUM_SPIKES_COLUMN_NAME: annotation_cols.append(NUM_SPIKES_COLUMN_NAME)
    if SHANK_COLUMN_NAME: annotation_cols.append(SHANK_COLUMN_NAME)

    missing_essential = [col for col in essential_cols if col not in df_good_clusters.columns]
    if missing_essential:
        print(f"--- CRITICAL ERROR ---")
        print(f"The DataFrame is missing essential columns required for plotting: {missing_essential}")
        print(f"Available columns: {list(df_good_clusters.columns)}")
        print("Cannot proceed. Please check the metadata file structure and CONFIGURATION section.")
        print(f"---")
        sys.exit()

    missing_annotations = [col for col in annotation_cols if col not in df_good_clusters.columns and col is not None]
    if missing_annotations:
        print(f"--- WARNING ---")
        print(f"The DataFrame is missing columns for some annotations: {missing_annotations}")
        print(f"Available columns: {list(df_good_clusters.columns)}")
        print("Annotations in the plot might be incomplete or marked 'N/A'.")
        print(f"---")

    # --- Fill missing region values if column exists ---
    if REGION_COLUMN_NAME in df_good_clusters.columns:
         df_good_clusters[REGION_COLUMN_NAME] = df_good_clusters[REGION_COLUMN_NAME].fillna('Unknown').astype(str)
    # ----------------------------------------------------

    # --- Get Cluster ID from User ---
    while True: # Loop until valid input is given
        cluster_id_input = input(f"Enter the Cluster ID you want to plot (e.g., {df_good_clusters['cluster_id'].iloc[0]}): ")
        try:
            cluster_id_to_plot = int(cluster_id_input)
            # Optional: Check if the ID actually exists in the dataframe
            if cluster_id_to_plot not in df_good_clusters['cluster_id'].values:
                 print(f"Warning: Cluster ID {cluster_id_to_plot} not found in the metadata file.")
                 # Ask if user wants to continue anyway or re-enter
                 cont = input("Continue anyway? (y/n): ").lower()
                 if cont == 'y':
                     break # Allow plotting attempt (will likely fail in plot_cluster_data)
                 else:
                     continue # Ask for input again
            else:
                break # Valid ID entered and found
        except ValueError:
            print("Invalid input. Please enter an integer Cluster ID.")
    # -----------------------------

    # --- Get Optional Time Duration from User ---
    start_time_to_plot = None
    end_time_to_plot = None
    while True:
        specify_duration = input("Do you want to specify a time duration for the plot? (y/n): ").lower()
        if specify_duration == 'y':
            try:
                start_str = input("Enter start time (in seconds): ")
                start_time_to_plot = float(start_str)
                end_str = input("Enter end time (in seconds): ")
                end_time_to_plot = float(end_str)
                if start_time_to_plot >= end_time_to_plot:
                    print("Error: Start time must be less than end time.")
                    start_time_to_plot = None # Reset on error
                    end_time_to_plot = None
                    continue # Ask again
                else:
                    break # Valid duration entered
            except ValueError:
                print("Invalid input. Please enter numeric values for start and end times.")
                start_time_to_plot = None # Reset on error
                end_time_to_plot = None
                # Optionally ask again or break loop
                cont = input("Try entering duration again? (y/n): ").lower()
                if cont != 'y':
                    break # Exit duration loop if user gives up
            except Exception as e:
                 print(f"An unexpected error occurred: {e}")
                 start_time_to_plot = None
                 end_time_to_plot = None
                 break # Exit loop on unexpected error
        elif specify_duration == 'n':
            print("Plotting entire duration.")
            break # Exit loop, keeping start/end times as None
        else:
            print("Invalid input. Please enter 'y' or 'n'.")
    # -----------------------------------------

    # --- Call the plotting function ---
    plot_cluster_data(
        cluster_id=cluster_id_to_plot,
        df_clusters=df_good_clusters,
        median_waveforms=median_waveforms,
        sampling_rate=SAMPLING_RATE_HZ,
        start_time=start_time_to_plot,
        end_time=end_time_to_plot,
        bin_width_s=FIRING_RATE_BIN_MS / 1000.0 # Convert ms to s
    )
    # ----------------------------------

    print("\nScript finished.")
