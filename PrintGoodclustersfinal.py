# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 19:14:14 2025

@author: HT_bo
"""

import numpy as np

# Specify the path to your .npy file
file_path = r'E:\spikeglx_datafile_tools-main\spikeglx_datafile_tools-main\python\good_clusters_final.npy'

try:
    # Load the data from the .npy file
    loaded_data = np.load(file_path, allow_pickle=True)

    # Print the contents
    print("Contents of good_clusters_final.npy:")
    print(loaded_data)

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found. Make sure the script that generates it has been run successfully.")
except Exception as e:
    print(f"An error occurred while loading the file: {e}")