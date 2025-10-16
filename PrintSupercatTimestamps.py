# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 12:17:38 2025

@author: HT_bo
"""

import numpy as np
import pprint # Import pprint for potentially cleaner dictionary printing

# --- Configuration ---
# !! Replace this with the actual path to your file !!
filepath = r"H:\CatGT_Stitched_Output\supercat_M3_Prb2_Soc_g0\M3_Prb2_Soc_g0_tcat.nidq_timestamps.npy"

# --- Load Data ---
try:
    # Use allow_pickle=True as the dictionary is saved as a pickled object
    data = np.load(filepath, allow_pickle=True).item()
    print(f"Successfully loaded data from: {filepath}")
    print("\nTop-level keys in the loaded data:")
    print(list(data.keys())) 
except FileNotFoundError:
    print(f"Error: File not found at {filepath}")
    data = None
except Exception as e:
    print(f"An error occurred while loading the file: {e}")
    data = None

# --- Access and Print Epoch Data ---
if data and 'EpochFrameData' in data:
    epoch_info_list = data['EpochFrameData']
    print(f"\nFound {len(epoch_info_list)} epochs in 'EpochFrameData'.")

    if epoch_info_list: # Check if the list is not empty
        # Iterate through ALL epochs in the list
        for i, epoch_details in enumerate(epoch_info_list):
            print(f"\n--- Epoch {i} Details ---")

            # Print the dictionary contents for the current epoch
            # Using pprint for potentially better formatting of the dictionary
            pprint.pprint(epoch_details, indent=2)

            # Optionally, print specific fields like the frames array separately
            if 'all_frames_sec' in epoch_details:
                frames_in_epoch = epoch_details['all_frames_sec']
                print(f"\nFrames in Epoch {i} (Count: {len(frames_in_epoch)}):")
                # Only print first few and last few if array is large
                if len(frames_in_epoch) > 10:
                     print(f"  First 5: {frames_in_epoch[:5]}")
                     print(f"  Last 5:  {frames_in_epoch[-5:]}")
                else:
                     print(f"  {frames_in_epoch}")
            else:
                 print(f"\n'all_frames_sec' key not found in Epoch {i} details.")

    else:
        print("\nThe 'EpochFrameData' list is empty. No epoch details to display.")
elif data:
     print("\nError: The key 'EpochFrameData' was not found in the loaded data.")
else:
     print("\nCould not proceed because data loading failed.")