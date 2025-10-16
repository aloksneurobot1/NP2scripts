# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 19:51:57 2025
Modified to include file Browse, coloring by brain region,
option to plot subsets of data by cluster/region/acronym, and
option to plot specific time durations.

@author: HT_bo
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm # Import colormaps
import pandas as pd
import sys
import os

# --- Import tkinter for file dialog ---
# Make sure tkinter is installed (usually included with Python)
try:
    import tkinter as tk
    from tkinter import filedialog
    TKINTER_AVAILABLE = True
except ImportError:
    print("Warning: tkinter module not found. File dialog will not be available.")
    print("Please provide the full path to the .npy file directly in the script or ensure tkinter is installed.")
    TKINTER_AVAILABLE = False
# --------------------------------------

# ==============================================================
# <<< CONFIGURATION >>>
# --- !!! ADJUST THESE VARIABLES !!! ---
# Set this to the actual name of the column in your data
# that contains the brain region information (for coloring).
REGION_COLUMN_NAME = 'brain_region_name' # Examples: 'acronym', 'location', 'ecephys_structure_acronym'

# Set this to the actual name of the column containing region acronyms (for filtering)
# If you don't have acronyms or don't want to filter by them, set it to None or the same as REGION_COLUMN_NAME
ACRONYM_COLUMN_NAME = 'acronym' # Example: 'acronym', 'region_short_name'
# --- !!! ADJUST THESE VARIABLES !!! ---

# Choose a colormap for regions (e.g., 'tab10', 'tab20', 'viridis', 'plasma', 'Set3')
COLORMAP_NAME = 'tab20'

# Optional: Sort clusters primarily by region? (Uses REGION_COLUMN_NAME)
SORT_BY_REGION = True
# ==============================================================


# --- Function to ask user for the .npy file ---
def select_npy_file():
    """Opens a file dialog for the user to select a .npy file."""
    if not TKINTER_AVAILABLE:
        filepath = input("Enter the full path to the good_clusters .npy file: ")
        if not os.path.exists(filepath):
             print(f"Error: File not found at '{filepath}'. Exiting.")
             sys.exit()
        return filepath

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
    # allow_pickle=True is necessary if the .npy file contains objects (like lists/dicts)
    good_clusters_raw = np.load(data_filepath, allow_pickle=True)
    print(f"Successfully loaded '{os.path.basename(data_filepath)}'.")
except FileNotFoundError:
    print(f"Error: The selected file could not be found at '{data_filepath}'.")
    sys.exit()
except Exception as e:
    print(f"Error loading file '{os.path.basename(data_filepath)}': {e}")
    sys.exit()
# ----------------------------------

# --- Check if data was loaded successfully ---
if not good_clusters_raw.size:
    print(f"No good clusters found in '{os.path.basename(data_filepath)}'.")
    sys.exit()
# -----------------------------------------

# --- Process the loaded data into DataFrame ---
try:
    if hasattr(good_clusters_raw.dtype, 'names') and good_clusters_raw.dtype.names:
        list_of_dictionaries = [dict(zip(good_clusters_raw.dtype.names, row)) for row in good_clusters_raw]
    elif isinstance(good_clusters_raw[0], dict):
        list_of_dictionaries = list(good_clusters_raw)
    else:
        print("Warning: Attempting conversion from unknown .npy structure. Assuming iterable of dict-like items.")
        list_of_dictionaries = [dict(item) for item in good_clusters_raw]

    df_all_clusters = pd.DataFrame(list_of_dictionaries)

except Exception as e:
    print(f"Error processing data structure from '{os.path.basename(data_filepath)}': {e}")
    print("Please ensure the .npy file contains data convertible to a Pandas DataFrame.")
    sys.exit()
# ---------------------------------

# --- Verify Essential Columns ---
essential_columns = ['cluster_id', 'spike_times_seconds']
missing_essential = [col for col in essential_columns if col not in df_all_clusters.columns]
if missing_essential:
    print(f"Error: DataFrame is missing essential columns: {missing_essential}")
    print(f"Available columns: {list(df_all_clusters.columns)}")
    sys.exit()

# Convert cluster_id to a comparable type if needed (e.g., int)
try:
    df_all_clusters['cluster_id'] = df_all_clusters['cluster_id'].astype(int)
except ValueError:
    print("Warning: Could not convert 'cluster_id' column to integer. Sorting/filtering by ID might behave unexpectedly.")
except Exception as e:
     print(f"Warning: Error converting 'cluster_id' column: {e}")
# ----------------------------------

# --- Verify Region Column (for coloring) and Handle Missing Data ---
color_by_region_possible = False
unique_regions_available = []
if REGION_COLUMN_NAME not in df_all_clusters.columns:
    print(f"--- WARNING ---")
    print(f"Coloring region column '{REGION_COLUMN_NAME}' not found.")
    print("Cannot color by region. Plotting all clusters in default color.")
    print(f"---")
    SORT_BY_REGION = False # Disable sorting by region if column missing
else:
    color_by_region_possible = True
    df_all_clusters[REGION_COLUMN_NAME] = df_all_clusters[REGION_COLUMN_NAME].fillna('Unknown').astype(str)
    print(f"Found coloring region column: '{REGION_COLUMN_NAME}'.")
    unique_regions_available = sorted(df_all_clusters[REGION_COLUMN_NAME].unique())
    print(f"Available full region names: {unique_regions_available}")
# ----------------------------------------------------

# --- Verify Acronym Column (for filtering) and Handle Missing Data ---
filter_by_acronym_possible = False
unique_acronyms_available = []
if ACRONYM_COLUMN_NAME and ACRONYM_COLUMN_NAME in df_all_clusters.columns:
    # Ensure the acronym column is treated as string and fill NaNs
    df_all_clusters[ACRONYM_COLUMN_NAME] = df_all_clusters[ACRONYM_COLUMN_NAME].fillna('Unknown').astype(str)
    unique_acronyms_available = sorted(df_all_clusters[ACRONYM_COLUMN_NAME].unique())
    if len(unique_acronyms_available) > 0 and not (len(unique_acronyms_available) == 1 and unique_acronyms_available[0] == 'Unknown'):
        filter_by_acronym_possible = True
        print(f"Found acronym column for filtering: '{ACRONYM_COLUMN_NAME}'.")
        print(f"Available acronyms: {unique_acronyms_available}")
    else:
        print(f"Acronym column '{ACRONYM_COLUMN_NAME}' found, but contains no usable values for filtering.")
elif ACRONYM_COLUMN_NAME:
    print(f"Warning: Acronym column '{ACRONYM_COLUMN_NAME}' not found. Cannot filter by acronym.")
# ----------------------------------------------------


# --- Print the columns ---
print("Columns in the full DataFrame:", list(df_all_clusters.columns))
# --------------------------

# --- Ask User about Cluster Filtering ---
filter_applied = False
cluster_filter_description = "All Clusters"
df_good_clusters = df_all_clusters.copy() # Start with all data

while True: # Loop for valid input
    plot_choice = input("Filter clusters? (Enter 'yes' or 'no'): ").strip().lower()
    if plot_choice in ['yes', 'no']:
        break
    else:
        print("Invalid input. Please enter 'yes' or 'no'.")

if plot_choice == 'yes':
    # Build the prompt string based on available filter options
    filter_options = ["'cluster_id'"]
    if color_by_region_possible: # Check if the *coloring* region column exists for filtering by full name
        filter_options.append("'region'")
    if filter_by_acronym_possible:
        filter_options.append("'acronym'")
    filter_prompt = f"Filter clusters by {' or '.join(filter_options)}? "

    while True: # Loop for valid filter type
        filter_type = input(filter_prompt).strip().lower()
        # Validate against available options
        allowed_types = ['cluster_id']
        if color_by_region_possible: allowed_types.append('region')
        if filter_by_acronym_possible: allowed_types.append('acronym')

        if filter_type in allowed_types:
            break
        else:
            print(f"Invalid input. Please enter one of: {', '.join(allowed_types)}.")

    # --- Filter by Full Region Name ---
    if filter_type == 'region':
        print(f"Available full region names: {unique_regions_available}")
        while True:
            region_input = input(f"Enter the full region name(s) to plot (comma-separated, case-sensitive): ").strip()
            selected_regions = [r.strip() for r in region_input.split(',') if r.strip()]
            if not selected_regions:
                print("No regions entered. Please enter at least one region name.")
                continue

            # Validate selected regions
            valid_regions = [r for r in selected_regions if r in unique_regions_available]
            invalid_regions = [r for r in selected_regions if r not in unique_regions_available]

            if invalid_regions:
                print(f"Warning: The following full region names were not found and will be ignored: {invalid_regions}")

            if not valid_regions:
                print("No valid regions selected. Please try again.")
                continue
            else:
                # Filter using the REGION_COLUMN_NAME
                df_good_clusters = df_all_clusters[df_all_clusters[REGION_COLUMN_NAME].isin(valid_regions)]
                filter_applied = True
                cluster_filter_description = f"Region(s): {', '.join(sorted(valid_regions))}"
                print(f"Filtering data for region(s): {', '.join(sorted(valid_regions))}")
                break # Exit region input loop

    # --- Filter by Acronym ---
    elif filter_type == 'acronym':
        print(f"Available acronyms: {unique_acronyms_available}")
        while True:
            acronym_input = input(f"Enter the acronym(s) to plot (comma-separated, case-sensitive): ").strip()
            selected_acronyms = [a.strip() for a in acronym_input.split(',') if a.strip()]
            if not selected_acronyms:
                print("No acronyms entered. Please enter at least one acronym.")
                continue

            # Validate selected acronyms
            valid_acronyms = [a for a in selected_acronyms if a in unique_acronyms_available]
            invalid_acronyms = [a for a in selected_acronyms if a not in unique_acronyms_available]

            if invalid_acronyms:
                print(f"Warning: The following acronyms were not found and will be ignored: {invalid_acronyms}")

            if not valid_acronyms:
                print("No valid acronyms selected. Please try again.")
                continue
            else:
                 # Filter using the ACRONYM_COLUMN_NAME
                df_good_clusters = df_all_clusters[df_all_clusters[ACRONYM_COLUMN_NAME].isin(valid_acronyms)]
                filter_applied = True
                cluster_filter_description = f"Acronym(s): {', '.join(sorted(valid_acronyms))}"
                print(f"Filtering data for acronym(s): {', '.join(sorted(valid_acronyms))}")
                break # Exit acronym input loop

    # --- Filter by Cluster ID ---
    elif filter_type == 'cluster_id':
        while True:
            id_input = input("Enter the cluster ID(s) to plot (comma-separated integers): ").strip()
            selected_ids_str = [i.strip() for i in id_input.split(',') if i.strip()]
            selected_ids = []
            if not selected_ids_str:
                 print("No cluster IDs entered. Please enter at least one ID.")
                 continue

            try:
                for id_str in selected_ids_str:
                    selected_ids.append(int(id_str)) # Convert to int
            except ValueError:
                print(f"Error: Invalid input. Cluster IDs must be integers. Problematic input: '{id_str}'")
                selected_ids = [] # Reset
                continue # Restart the loop to get correct input

            # Validate selected IDs against available IDs
            all_cluster_ids = df_all_clusters['cluster_id'].unique()
            valid_ids = [i for i in selected_ids if i in all_cluster_ids]
            invalid_ids = [i for i in selected_ids if i not in all_cluster_ids]

            if invalid_ids:
                 print(f"Warning: The following cluster IDs were not found in the data and will be ignored: {invalid_ids}")

            if not valid_ids:
                print("No valid cluster IDs selected. Please try again.")
                continue
            else:
                df_good_clusters = df_all_clusters[df_all_clusters['cluster_id'].isin(valid_ids)]
                filter_applied = True
                cluster_filter_description = f"Cluster ID(s): {', '.join(map(str, sorted(valid_ids)))}"
                print(f"Filtering data for cluster ID(s): {', '.join(map(str, sorted(valid_ids)))}")
                break # Exit ID input loop

    # Check if filtering resulted in empty DataFrame
    if filter_applied and df_good_clusters.empty:
        print(f"Error: The selected cluster filter criteria resulted in no matching clusters.")
        print("Exiting script.")
        sys.exit()
    elif not filter_applied: # Should not happen if plot_choice == 'yes', but as safeguard
         print("Proceeding without cluster filtering.")

# --- Ask User about Time Filtering ---
time_filter_applied = False
time_filter_description = "Full Duration"
start_time = None
end_time = None

while True:
    time_choice = input("Filter by time duration? (Enter 'yes' or 'no'): ").strip().lower()
    if time_choice in ['yes', 'no']:
        break
    else:
        print("Invalid input. Please enter 'yes' or 'no'.")

if time_choice == 'yes':
    # Get start time
    while True:
        start_input = input("Enter start time in seconds (leave blank for beginning): ").strip()
        if not start_input:
            start_time = -np.inf # Effectively no lower bound
            break
        try:
            start_time = float(start_input)
            break
        except ValueError:
            print("Invalid input. Please enter a number or leave blank.")

    # Get end time
    while True:
        end_input = input("Enter end time in seconds (leave blank for end): ").strip()
        if not end_input:
            end_time = np.inf # Effectively no upper bound
            break
        try:
            end_time = float(end_input)
            # Use -np.inf for comparison as start_time could be exactly 0
            if start_time != -np.inf and end_time < start_time:
                print(f"Error: End time ({end_time}s) cannot be before start time ({start_time if start_time != -np.inf else 'Start'}s).")
                continue # Ask for end time again
            break
        except ValueError:
            print("Invalid input. Please enter a number or leave blank.")

    if start_time != -np.inf or end_time != np.inf:
        time_filter_applied = True
        start_desc = f"{start_time:.2f}s" if start_time != -np.inf else "Start"
        end_desc = f"{end_time:.2f}s" if end_time != np.inf else "End"
        time_filter_description = f"Time: {start_desc} to {end_desc}"
        print(f"Applying time filter: {time_filter_description}")
    else:
        print("No specific time bounds entered. Plotting full duration.")
        start_time = None # Reset to None if no bounds set
        end_time = None

# --- Sort the DataFrame (using the potentially filtered df_good_clusters) ---
sort_columns = ['cluster_id']
# Use REGION_COLUMN_NAME for sorting if requested and possible
if SORT_BY_REGION and color_by_region_possible and REGION_COLUMN_NAME in df_good_clusters.columns:
    sort_columns.insert(0, REGION_COLUMN_NAME)
    print(f"Sorting clusters primarily by: {sort_columns}")
else:
     print(f"Sorting clusters by: {sort_columns}")

# Check if required columns exist before sorting
if not all(col in df_good_clusters.columns for col in sort_columns):
    missing_cols = [col for col in sort_columns if col not in df_good_clusters.columns]
    print(f"Error: DataFrame is missing one or more required columns for sorting: {missing_cols}")
    sys.exit()

df_good_clusters = df_good_clusters.sort_values(by=sort_columns).reset_index(drop=True)
# --------------------------


# --- Create Color Map for Regions (based on REGION_COLUMN_NAME in filtered data) ---
region_color_map = {}
unique_regions_in_plot = []
# Coloring is always based on REGION_COLUMN_NAME if possible
if color_by_region_possible and REGION_COLUMN_NAME in df_good_clusters.columns:
    unique_regions_in_plot = sorted(df_good_clusters[REGION_COLUMN_NAME].unique())
    if not unique_regions_in_plot:
        print("Warning: No regions found in the filtered data to create a color map.")
        color_by_region_possible = False # Disable coloring if no regions left
    else:
        try:
            colors = plt.colormaps[COLORMAP_NAME]
            num_colors = len(unique_regions_in_plot)
            if hasattr(colors, 'N') and colors.N >= num_colors: # Discrete colormap check
                 color_cycle = [colors(i) for i in range(num_colors)]
            else: # Continuous colormap or not enough discrete colors
                 color_cycle = [colors(i / num_colors) for i in range(num_colors)]

            region_color_map = {region: color for region, color in zip(unique_regions_in_plot, color_cycle)}

            if 'Unknown' in region_color_map:
                 region_color_map['Unknown'] = (0.5, 0.5, 0.5, 1.0) # Gray

        except KeyError:
            print(f"Warning: Colormap '{COLORMAP_NAME}' not found. Using default 'tab10'.")
            COLORMAP_NAME = 'tab10' # Fallback
            colors = plt.colormaps[COLORMAP_NAME]
            num_colors = len(unique_regions_in_plot)
            color_cycle = [colors(i) for i in range(min(num_colors, colors.N))]
            region_color_map = {region: color_cycle[i % len(color_cycle)] for i, region in enumerate(unique_regions_in_plot)}
            if 'Unknown' in region_color_map:
                 region_color_map['Unknown'] = (0.5, 0.5, 0.5, 1.0) # Gray
        except Exception as e:
             print(f"Error creating color map: {e}")
             color_by_region_possible = False # Disable coloring on error

# -------------------------------------------------


# --- Plotting ---
print("Generating plot...")
fig, ax = plt.subplots(figsize=(12, 10))

ytick_positions = []
ytick_labels = []
plotted_regions_set = set() # Keep track of regions (full names) actually plotted for legend
actual_min_time = np.inf
actual_max_time = -np.inf
any_spikes_plotted = False

# Iterate using index for y-position after sorting
for y_position, (index, row) in enumerate(df_good_clusters.iterrows()):
    try:
        cluster_id = row['cluster_id']
        spike_times_seconds = np.asarray(row['spike_times_seconds'], dtype=float)

        if spike_times_seconds.ndim == 0 or spike_times_seconds.size == 0 or np.isnan(spike_times_seconds).all():
            continue

        spike_times_seconds = spike_times_seconds[np.isfinite(spike_times_seconds)]
        if spike_times_seconds.size == 0:
             continue

        # --- Apply Time Filter ---
        if time_filter_applied:
            time_mask = (spike_times_seconds >= start_time) & (spike_times_seconds <= end_time)
            spike_times_to_plot = spike_times_seconds[time_mask]
        else:
            spike_times_to_plot = spike_times_seconds

        if spike_times_to_plot.size == 0:
            continue
        # -------------------------

        any_spikes_plotted = True

        # Update overall time range observed in the plotted data
        min_t = np.min(spike_times_to_plot)
        max_t = np.max(spike_times_to_plot)
        if min_t < actual_min_time: actual_min_time = min_t
        if max_t > actual_max_time: actual_max_time = max_t

        # Determine color (always based on REGION_COLUMN_NAME)
        cluster_color = 'black' # Default color
        if color_by_region_possible and REGION_COLUMN_NAME in row and row[REGION_COLUMN_NAME] in region_color_map:
            region = row[REGION_COLUMN_NAME] # Use full name for coloring
            cluster_color = region_color_map.get(region, 'black')
            plotted_regions_set.add(region) # Add full name to legend set

        # Plot spikes
        ax.eventplot(spike_times_to_plot, lineoffsets=y_position, linelengths=0.7, color=cluster_color)

        # Store tick info
        ytick_positions.append(y_position)
        ytick_labels.append(str(cluster_id))

    except KeyError as e:
        print(f"Error accessing column {e} for plotting row index {index} (Cluster ID: {row.get('cluster_id', 'N/A')}). Check DataFrame structure.")
        continue
    except Exception as e:
        print(f"Error plotting data for cluster {row.get('cluster_id', 'ID Unknown')} at row index {index}: {e}")
        continue

# --- Finalize Plot ---
if not any_spikes_plotted:
    print("No spike data was plotted.")
    if filter_applied: print("This might be due to the cluster filter criteria.")
    if time_filter_applied: print("This might be due to the time filter criteria.")
    plt.close(fig)
    sys.exit()

# Set the y-axis ticks and labels
max_yticks = 60
num_clusters_plotted = len(ytick_positions)
tick_step = max(1, num_clusters_plotted // max_yticks)
if ytick_positions:
    ax.set_yticks(ytick_positions[::tick_step])
    ax.set_yticklabels(ytick_labels[::tick_step])
else:
    ax.set_yticks([])
    ax.set_yticklabels([])

ax.set_ylim(-1, num_clusters_plotted)

# Set x-axis limits
if time_filter_applied:
    plot_start_time = start_time if start_time != -np.inf else actual_min_time
    plot_end_time = end_time if end_time != np.inf else actual_max_time
    buffer = (plot_end_time - plot_start_time) * 0.02 if (plot_end_time > plot_start_time) else 1
    ax.set_xlim(plot_start_time - buffer, plot_end_time + buffer)
elif actual_min_time != np.inf and actual_max_time != -np.inf :
     buffer = (actual_max_time - actual_min_time) * 0.02 if (actual_max_time > actual_min_time) else 1
     ax.set_xlim(actual_min_time - buffer, actual_max_time + buffer)

ax.set_ylabel('Cluster ID' + (' (Sorted)' if sort_columns else ''))
ax.set_xlabel('Time (s)')

# Construct the title
title_line1 = f'Raster Plot: {cluster_filter_description}, {time_filter_description}'
title_line2 = f'(Source: {os.path.basename(data_filepath)}'
if sort_columns:
     sort_desc = ', '.join(sort_columns)
     title_line2 += f', Sorted by {sort_desc}'
title_line2 += ')'
plot_title = f"{title_line1}\n{title_line2}"
ax.set_title(plot_title, fontsize=10)
ax.tick_params(axis='y', left=True)

# --- Add Legend (using full region names from REGION_COLUMN_NAME) ---
legend_handles = []
if color_by_region_possible and region_color_map and plotted_regions_set:
    # Sort legend items based on the order in unique_regions_in_plot for consistency
    plotted_regions_sorted = sorted(list(plotted_regions_set), key=lambda x: unique_regions_in_plot.index(x) if x in unique_regions_in_plot else -1)
    for region in plotted_regions_sorted:
        if region in region_color_map:
             legend_handles.append(plt.Line2D([0], [0], color=region_color_map[region], lw=4, label=region)) # Label uses full name

    if legend_handles:
        ax.legend(handles=legend_handles, title="Brain Region", # Legend title
                  bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., fontsize='small')
        plt.subplots_adjust(right=0.80, top=0.92)
    else:
         plt.tight_layout()
else:
    plt.tight_layout()

print("Displaying plot...")
plt.show()
# ---------------

print("Script finished.")
