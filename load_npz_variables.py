# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 18:28:36 2025

@author: HT_bo
"""

import numpy as np
import tkinter as tk
from tkinter import filedialog
import os  # Import the 'os' module

def load_npz_variables():
    """
    Opens a file dialog to select an .npz file, loads it, and displays
    the loaded variables and their shapes in a text area.  It also makes
    the variables available in the global namespace (for Spyder's Variable Explorer).
    """

    root = tk.Tk()
    root.withdraw()  # Hide the main tkinter window

    file_path = filedialog.askopenfilename(
        title="Select an .npz file",
        filetypes=[("NPZ files", "*.npz"), ("All files", "*.*")]
    )

    if not file_path:  # User cancelled the dialog
        print("No file selected.")
        return

    if not file_path.lower().endswith(".npz"):
        print("Error: Selected file is not an .npz file.")
        return

    try:
        with np.load(file_path) as data:
            # Create a text area to display variable information
            info_window = tk.Toplevel(root)  # Use Toplevel for a separate window
            info_window.title("Loaded Variables")
            text_area = tk.Text(info_window)
            text_area.pack(expand=True, fill='both')  # Make the text area fill the window


            for var_name in data.files:
                var = data[var_name]
                # Add variables to the global namespace (Spyder's variable explorer)
                globals()[var_name] = var

                # Display variable name and shape in the text area
                info_text = f"Variable: {var_name}\nShape: {var.shape}\nType: {var.dtype}\n"
                # Add a preview of the array content, handling different shapes appropriately.
                if var.size == 0:  # Check for empty arrays
                    info_text += "Data: (empty array)\n\n"
                elif var.ndim == 0: #scalar
                    info_text += f"Data: {var}\n\n"
                elif var.size < 100 : # small arrays
                    info_text += f"Data: {var}\n\n"
                else: # Large arrays -- preview only a small part
                    if var.ndim == 1:
                        preview = var[:5]  # First 5 elements
                    elif var.ndim == 2:
                        preview = var[:5, :5]   # First 5 rows and columns
                    elif var.ndim == 3:
                        preview = var[:2, :2, :2] # First 2 elements in each dimension
                    else:
                        preview = " (Too large to display preview)"

                    info_text += f"Data (preview):\n{preview}\n...\n\n"
                text_area.insert(tk.END, info_text)

            print(f"Variables from '{os.path.basename(file_path)}' loaded successfully.") #use os.path.basename
            text_area.config(state=tk.DISABLED) #make text area read-only


    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
    except OSError as e:  # Catch more specific OSError
        print(f"Error loading file: {e}")
        print("This might be due to an invalid .npz file or a file corruption.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    load_npz_variables()

    # Keep the Tkinter event loop running (optional, for the info window)
    #  If you only want to see the variables in Spyder, you can comment this out.
    #  The variables will still be loaded into Spyder's namespace.
    tk.mainloop()