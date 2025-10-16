import numpy as np
import os 

npy_file_path = r'E:\Test\M4_Prb1_Soc1_g3\NeuropixelNIdaqFile_TimeStamps_M4_Prb1_Soc1_g3_t0.nidq_2025-02-25.npy' 

# --- Loading and Printing ---

print(f"Attempting to load: {npy_file_path}")

# Check if the file exists first (optional but good practice)
if not os.path.exists(npy_file_path):
    print(f"\nError: File not found at the specified path.")
    print("Please make sure the path is correct and the file exists.")
else:
    try:
        # Load the data from the .npy file
        # allow_pickle=True is often necessary if the .npy file contains
     
        loaded_data = np.load(npy_file_path, allow_pickle=True)

        print(f"\nFile successfully loaded!")

        # Print basic information about the loaded data
        print(f"Data type: {type(loaded_data)}")
        # Check if it has a 'shape' attribute (standard numpy arrays do)
        if hasattr(loaded_data, 'shape'):
            print(f"Data shape: {loaded_data.shape}")
        else:
            # If it's an object array (like a list of dicts), shape might be less informative
            try:
                print(f"Data length (if applicable): {len(loaded_data)}")
            except TypeError:
                 print("Data type does not have a standard shape or length.")


        # Print the actual contents
        print("\n--- File Contents ---")
        print(loaded_data)
        print("--- End of Contents ---")

        # If the data is large, you might want to print only a part of it, e.g.:
        if isinstance(loaded_data, np.ndarray) and loaded_data.ndim > 0:
             print("\n--- First 5 elements/rows ---")
             print(loaded_data[:5])
             print("--- End of Sample ---")
        elif isinstance(loaded_data, list):
             print("\n--- First 5 items ---")
             print(loaded_data[:5])
             print("--- End of Sample ---")


    except FileNotFoundError:
        # This might be redundant if using os.path.exists, but catches race conditions
        print(f"\nError: File not found at {npy_file_path}")
    except Exception as e:
        # Catch other potential errors during loading (e.g., corrupted file)
        print(f"\nAn error occurred while loading the file: {e}")