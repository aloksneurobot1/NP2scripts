# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 17:03:20 2025

@author: Alok
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tkinter import Tk, filedialog

def plot_sleep_states(npy_file_path):
    """
    Plots sleep states from a .npy file.

    Args:
        npy_file_path (str or pathlib.Path): Path to the .npy file containing sleep states.
    """
    try:
        sleep_states = np.load(npy_file_path)

        plt.figure(figsize=(12, 4))
        plt.plot(sleep_states)
        plt.yticks([0, 1, 2], ['Awake', 'NREM', 'REM'])
        plt.xlabel("Time (1e6S)")  # Or adjust based on your time resolution
        plt.ylabel("Sleep State")
        plt.title("Sleep States")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print(f"Error: File not found at {npy_file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Get file path from user using a browser window
root = Tk()
root.withdraw()
root.attributes("-topmost", True)
npy_file_path_str = filedialog.askopenfilename(title="Select sleep_states.npy file")
npy_file_path = Path(npy_file_path_str) if npy_file_path_str else None # Handle case where user cancels file selection
root.destroy()

if npy_file_path:
    plot_sleep_states(npy_file_path)
else:
    print("No sleep_states.npy file selected. Exiting.")

# import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path 
# from tkinter import Tk, filedialog

# def plot_sleep_states(npy_file_path, start_minute= None , end_minute=None):
#     """
#     Plots sleep states from a .npy file, with optional time range selection.

#     Args:
#         npy_file_path (str or pathlib.Path): Path to the .npy file containing sleep states.
#         start_minute (int, optional): Start minute for plotting. Defaults to None (start from beginning).
#         end_minute (int, optional): End minute for plotting. Defaults to None (plot to end).
#     """
#     try:
#         sleep_states = np.load(npy_file_path)

#         # Assuming 1e6 samples correspond to 1 minute, adjust if needed
#         samples_per_minute = 1e6

#         start_sample = int(start_minute * samples_per_minute) if start_minute is not None else 0
#         end_sample = int(end_minute * samples_per_minute) if end_minute is not None else len(sleep_states)

#         if start_sample < 0:
#             start_sample = 0
#         if end_sample > len(sleep_states):
#             end_sample = len(sleep_states)

#         sleep_states_subset = sleep_states[start_sample:end_sample]

#         plt.figure(figsize=(12, 4))
#         plt.plot(sleep_states_subset)
#         plt.yticks([0, 1, 2], ['Awake', 'NREM', 'REM'])
#         plt.xlabel("Time (1e6S)")  # Or adjust based on your time resolution
#         plt.ylabel("Sleep State")
#         plt.title("Sleep States")
#         plt.grid(True)
#         plt.tight_layout()
#         plt.show()

#     except FileNotFoundError:
#         print(f"Error: File not found at {npy_file_path}")
#     except Exception as e:
#         print(f"An error occurred: {e}")

# # Get file path from user using a browser window
# root = Tk()
# root.withdraw()
# root.attributes("-topmost", True)
# npy_file_path_str = filedialog.askopenfilename(title="Select sleep_states.npy file")
# npy_file_path = Path(npy_file_path_str) if npy_file_path_str else None  # Handle case where user cancels file selection
# root.destroy()

# if npy_file_path:
#     start_minute_input = input("Enter start minute (or press Enter for beginning): ")
#     end_minute_input = input("Enter end minute (or press Enter for end): ")

#     start_minute = int(start_minute_input) if start_minute_input else None
#     end_minute = int(end_minute_input) if end_minute_input else None

#     plot_sleep_states(npy_file_path, start_minute, end_minute)
# else:
#     print("No sleep_states.npy file selected. Exiting.")