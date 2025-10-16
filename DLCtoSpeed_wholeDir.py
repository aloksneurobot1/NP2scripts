# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 09:41:49 2025

@author: HT_bo
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path # Use pathlib for path operations
from tkinter import Tk, filedialog # For directory selection
import sys # For exit

# --- Configuration ---
# Video dimensions (pixels) 
pixel_width = 1440
pixel_height = 1080

# Real-world dimensions corresponding to the video (cm) 
real_width_cm = 28.5
real_height_cm = 17

# Video frame rate (frames per second) 
frame_rate = 30

# Body parts to calculate speed for (use the base name from the header)
# Format: {"Display Name": "Column Prefix"}
body_parts_to_process = {
    "Headstage": "Headstage",
    "Trunk": "Trunk",
    "Tailbase": "Tailbase",
    "Nose": "Nose",
    "Neck": "Neck"
}

# CSV header rows (0-indexed, list of row numbers) - Adjust if needed
# For standard DLC CSVs, this is often [1, 2]
CSV_HEADER_ROWS = [1, 2]

# File matching pattern (glob syntax) - Use this to select only relevant CSVs
# Example: '*DLC*.csv' to only process files with 'DLC' in the name
# Example: '*.csv' to process all CSV files in the directory
# Important: Adjust this pattern if your DLC files have a different naming scheme!
FILE_PATTERN = '*DLC*.csv'
# --- End Configuration ---

# --- Helper Function ---
def calculate_speed_cm_per_s(df, x_col, y_col, px_to_cm_x, px_to_cm_y, fps):
    """Calculates speed in cm/s between consecutive frames."""
    # Check if required columns exist
    if x_col not in df.columns or y_col not in df.columns:
        print(f"    Warning: Columns '{x_col}' or '{y_col}' not found. Skipping speed calculation for this body part.")
        # Return an array of NaNs matching the DataFrame length
        return np.full(len(df), np.nan)

    # Convert coordinate columns to numeric, forcing errors to NaN (handles non-numeric data)
    x_coords = pd.to_numeric(df[x_col], errors='coerce')
    y_coords = pd.to_numeric(df[y_col], errors='coerce')

    # Calculate the difference in pixel coordinates between consecutive frames
    diff_x_pixels = x_coords.diff()
    diff_y_pixels = y_coords.diff()

    # Convert pixel differences to cm
    diff_x_cm = diff_x_pixels * px_to_cm_x
    diff_y_cm = diff_y_pixels * px_to_cm_y

    # Calculate the Euclidean distance (displacement) in cm
    distances_cm = np.sqrt(diff_x_cm**2 + diff_y_cm**2)

    # Time interval between frames in seconds
    time_interval = 1.0 / fps if fps > 0 else np.inf # Avoid division by zero

    # Calculate speed (distance / time) in cm/s
    speeds_cm_per_s = distances_cm / time_interval
    # The first value will naturally be NaN because .diff() has no previous row for the first row.
    return speeds_cm_per_s

# --- Core Processing Function for a Single File ---
def process_dlc_file(file_path, config):
    """Reads a DLC CSV, calculates speeds, and saves a new CSV."""
    print(f"---\nProcessing file: {file_path.name}")
    try:
        # Read the CSV file using configured header rows
        df = pd.read_csv(file_path, header=config['csv_header_rows'])
        # print(f"  Successfully read input file.") # Can be verbose
    except FileNotFoundError:
        # This check might be redundant if using Path.glob, but safe to keep
        print(f"  Error: Input file not found.")
        return False # Indicate failure
    except Exception as e:
        print(f"  Error reading CSV file: {e}")
        # Check if it might be because header rows are incorrect
        if isinstance(e, (ValueError, IndexError)) and 'header' in str(e).lower():
             print(f"  Hint: Check if the CSV_HEADER_ROWS={config['csv_header_rows']} config is correct for this file.")
        return False # Indicate failure

    # Flatten the multi-index columns
    try:
        # Check if columns are already flat (e.g., if header read failed gracefully)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(col).strip().rstrip('_') for col in df.columns]
        # else: columns are already flat, do nothing
    except TypeError as e:
         print(f"  Error flattening columns, potentially not a multi-index header as expected: {e}")
         print(f"  Columns found: {list(df.columns)}")
         return False # Indicate failure

    # --- Data Processing ---
    # Calculate pixel to cm conversion ratios
    pixel_to_cm_x = config['real_width_cm'] / config['pixel_width']
    pixel_to_cm_y = config['real_height_cm'] / config['pixel_height']
    frame_rate = config['frame_rate']

    # Calculate speed for each specified body part
    print(f"  Calculating speeds...")
    speed_cols_added = []
    for display_name, col_prefix in config['body_parts_to_process'].items():
        x_col_name = f"{col_prefix}_x"
        y_col_name = f"{col_prefix}_y"
        speed_col_name = f"{display_name} Speed (cm/s)"
        # Calculate speed
        speed_data = calculate_speed_cm_per_s(df, x_col_name, y_col_name, pixel_to_cm_x, pixel_to_cm_y, frame_rate)
        # Only add the column if speed calculation didn't just return all NaNs (because input columns were missing)
        if not np.isnan(speed_data).all():
             df[speed_col_name] = speed_data
             speed_cols_added.append(speed_col_name)
             # print(f"  - Calculated speed for {display_name}") # Can be verbose

    if not speed_cols_added:
         print("  Warning: No speed columns were successfully calculated (check input column names in CSV vs. config?).")
         # Optionally skip saving if no speeds calculated, but often useful to save anyway
         # return False

    # --- File Export ---
    try:
        # Generate the output file path
        input_directory = file_path.parent
        # Use Path.stem to get filename without the final extension
        base_name = file_path.stem
        output_filename = f"{base_name}_speed_data_cm_per_s.csv"
        output_file_path = input_directory / output_filename # Use pathlib division

        # Export the dataframe
        df.to_csv(output_file_path, index=False, float_format='%.4f')
        print(f"  Speed calculation complete. Data saved to: {output_filename}")
        return True # Indicate success
    except Exception as e:
        print(f"  Error saving the output file '{output_filename}': {e}")
        return False # Indicate failure

# --- Main Script ---
if __name__ == "__main__":
    # Prompt user to select the input DIRECTORY
    print("Please select the directory containing the DLC CSV files.")
    root = Tk()
    root.withdraw() # Hide the main Tkinter window
    root.attributes("-topmost", True) # Bring the dialog to the front
    input_dir_path_str = filedialog.askdirectory(title="Select Input Directory Containing DLC CSV Files")
    root.destroy()

    if not input_dir_path_str:
        print("No directory selected. Exiting.")
        sys.exit() # Use sys.exit

    input_directory = Path(input_dir_path_str)
    print(f"\nSelected directory: {input_directory}")

    # --- Prepare Configuration Dictionary ---
    # This makes it easier to pass parameters around
    config = {
        'pixel_width': pixel_width,
        'pixel_height': pixel_height,
        'real_width_cm': real_width_cm,
        'real_height_cm': real_height_cm,
        'frame_rate': frame_rate,
        'body_parts_to_process': body_parts_to_process,
        'csv_header_rows': CSV_HEADER_ROWS,
        'file_pattern': FILE_PATTERN
    }

    # Find files matching the pattern
    print(f"Searching for files matching '{config['file_pattern']}'...")
    # Use rglob for recursive search if needed: list(input_directory.rglob(config['file_pattern']))
    files_to_process = list(input_directory.glob(config['file_pattern']))

    if not files_to_process:
        print(f"No files found matching the pattern '{config['file_pattern']}' in the selected directory.")
        print("Please check the FILE_PATTERN setting in the script or the contents of the directory.")
        sys.exit() # Use sys.exit

    print(f"Found {len(files_to_process)} files to process.")

    # Process each file
    processed_count = 0
    failed_count = 0
    for file_path in files_to_process:
        # Basic check to avoid processing already processed files
        if '_speed_data_cm_per_s.csv' in file_path.name:
            print(f"\nSkipping already processed file: {file_path.name}")
            continue
        if process_dlc_file(file_path, config):
            processed_count += 1
        else:
            failed_count += 1
            print(f"Failed processing file: {file_path.name}")


    # Print summary
    print("\n--- Processing Summary ---")
    print(f"Successfully processed: {processed_count} files")
    print(f"Failed to process:     {failed_count} files")
    print("------------------------")