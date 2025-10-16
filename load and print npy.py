# import tkinter as tk
# from tkinter import filedialog
# import numpy as np

# def browse_and_print():
#     """Opens a file dialog, loads a .npy file, and prints its contents."""
#     filepath = filedialog.askopenfilename(filetypes=[("NumPy files", "*.npy")])
#     if filepath:
#         try:
#             # Allow pickling (ONLY if you trust the file source)
#             data = np.load(filepath, allow_pickle=True)
#             print("Data from", filepath + ":")
#             print(data)
#         except FileNotFoundError:
#             print("File not found.")
#         except Exception as e:
#             print(f"An error occurred: {e}")

# # Create the main window
# root = tk.Tk()
# root.title("Numpy File Viewer")

# # Create a button to trigger the browsing and printing
# browse_button = tk.Button(root, text="Browse and Print .npy", command=browse_and_print)
# browse_button.pack(pady=20)

# # Run the Tkinter event loop
# root.mainloop()

# import tkinter as tk
# from tkinter import filedialog
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# # Global variable to store the loaded data
# loaded_data = None
# plot_window = None

# def browse_and_load():
#     """Opens a file dialog, loads a .npy file, and stores its contents."""
#     global loaded_data
#     global plot_button
#     filepath = filedialog.askopenfilename(filetypes=[("NumPy files", "*.npy")])
#     if filepath:
#         try:
#             # Allow pickling (ONLY if you trust the file source)
#             loaded_data = np.load(filepath, allow_pickle=True)
#             print("Data loaded from", filepath + ".")
#             # Enable the plot button now that data is loaded
#             plot_button.config(state=tk.NORMAL)
#         except FileNotFoundError:
#             print("File not found.")
#             loaded_data = None
#             plot_button.config(state=tk.DISABLED)
#         except Exception as e:
#             print(f"An error occurred during loading: {e}")
#             loaded_data = None
#             plot_button.config(state=tk.DISABLED)
#     else:
#         loaded_data = None
#         plot_button.config(state=tk.DISABLED)

# def plot_data():
#     """Plots the loaded data in a new Tkinter window."""
#     global loaded_data
#     global plot_window

#     if loaded_data is None:
#         print("No data loaded yet. Please load a .npy file first.")
#         return

#     if plot_window is not None:
#         plot_window.destroy()  # Close the previous plot window

#     plot_window = tk.Toplevel(root)
#     plot_window.title("Data Plot")

#     try:
#         fig, ax = plt.subplots()

#         if loaded_data.ndim == 1:
#             ax.plot(loaded_data)
#             ax.set_xlabel("Index")
#             ax.set_ylabel("Value")
#             ax.set_title("1D Data Plot")
#         elif loaded_data.ndim == 2:
#             if loaded_data.shape[1] == 2:
#                 ax.scatter(loaded_data[:, 0], loaded_data[:, 1])
#                 ax.set_xlabel("Column 1")
#                 ax.set_ylabel("Column 2")
#                 ax.set_title("2D Scatter Plot")
#             else:
#                 ax.imshow(loaded_data, aspect='auto')
#                 ax.set_xlabel("Column Index")
#                 ax.set_ylabel("Row Index")
#                 ax.set_title("2D Data as Image")
#         else:
#             tk.messagebox.showinfo("Plotting Error", "Data has more than 2 dimensions and cannot be plotted directly.")
#             return

#         canvas = FigureCanvasTkAgg(fig, master=plot_window)
#         canvas_widget = canvas.get_tk_widget()
#         canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

#         toolbar = NavigationToolbar2Tk(canvas, plot_window)
#         toolbar.update()
#         canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

#         canvas.draw()

#     except Exception as e:
#         tk.messagebox.showerror("Plotting Error", f"An error occurred during plotting: {e}")

# # Create the main window
# root = tk.Tk()
# root.title("Numpy File Viewer and Plotter")

# # Create a button to trigger the Browse and loading
# browse_button = tk.Button(root, text="Browse and Load .npy", command=browse_and_load)
# browse_button.pack(pady=10)

# # Create a button to trigger the plotting (initially disabled)
# plot_button = tk.Button(root, text="Plot Data", command=plot_data, state=tk.DISABLED)
# plot_button.pack(pady=10)

# # Run the Tkinter event loop
# root.mainloop()

import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# Global variable to store the loaded data
loaded_data = None
plot_window = None

def browse_and_load():
    """Opens a file dialog, loads a .npy file, and stores its contents."""
    global loaded_data
    global plot_button
    filepath = filedialog.askopenfilename(filetypes=[("NumPy files", "*.npy")])
    if filepath:
        try:
            # Allow pickling (ONLY if you trust the file source)
            loaded_data = np.load(filepath, allow_pickle=True)
            print(f"Data loaded from {filepath} with shape: {loaded_data.shape}.")
            
            # Print the first and last five elements of the loaded data
            print("First 5 elements:")
            print(loaded_data[:5])
            print("\nLast 5 elements:")
            print(loaded_data[-5:])

            # Enable the plot button now that data is loaded
            plot_button.config(state=tk.NORMAL)
        except FileNotFoundError:
            print("File not found.")
            loaded_data = None
            plot_button.config(state=tk.DISABLED)
        except Exception as e:
            print(f"An error occurred during loading: {e}")
            loaded_data = None
            plot_button.config(state=tk.DISABLED)
    else:
        loaded_data = None
        plot_button.config(state=tk.DISABLED)

def plot_data():
    """Plots the loaded data in a new Tkinter window."""
    global loaded_data
    global plot_window

    if loaded_data is None:
        print("No data loaded yet. Please load a .npy file first.")
        return

    if plot_window is not None:
        plot_window.destroy()  # Close the previous plot window

    plot_window = tk.Toplevel(root)
    plot_window.title("Data Plot")

    try:
        fig, ax = plt.subplots()

        if loaded_data.ndim == 1:
            ax.plot(loaded_data)
            ax.set_xlabel("Index")
            ax.set_ylabel("Value")
            ax.set_title("1D Data Plot")
        elif loaded_data.ndim == 2:
            if loaded_data.shape[1] == 2:
                ax.scatter(loaded_data[:, 0], loaded_data[:, 1])
                ax.set_xlabel("Column 1")
                ax.set_ylabel("Column 2")
                ax.set_title("2D Scatter Plot")
            else:
                ax.imshow(loaded_data, aspect='auto')
                ax.set_xlabel("Column Index")
                ax.set_ylabel("Row Index")
                ax.set_title("2D Data as Image")
        else:
            tk.messagebox.showinfo("Plotting Error", "Data has more than 2 dimensions and cannot be plotted directly.")
            return

        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, plot_window)
        toolbar.update()
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        canvas.draw()

    except Exception as e:
        tk.messagebox.showerror("Plotting Error", f"An error occurred during plotting: {e}")

# Create the main window
root = tk.Tk()
root.title("Numpy File Viewer and Plotter")

# Create a button to trigger the Browse and loading
browse_button = tk.Button(root, text="Browse and Load .npy", command=browse_and_load)
browse_button.pack(pady=10)

# Create a button to trigger the plotting (initially disabled)
plot_button = tk.Button(root, text="Plot Data", command=plot_data, state=tk.DISABLED)
plot_button.pack(pady=10)

# Run the Tkinter event loop
root.mainloop()