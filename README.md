# Analyzing Real Neural Data with a Hierarchical SLDS-CANN Model

## Introduction
This project provides a complete Python pipeline for analyzing real neural recordings from the mouse hippocampus. The scientific goal is to discover the latent dynamical rules that govern neural population activity across different behavioral states.

We use a hierarchical generative model, structured as a Variational Autoencoder (VAE), to infer these dynamics. This approach allows us to uncover both continuous and discrete patterns hidden within complex, high-dimensional neural data.

## Model Architecture
The core of this project is a novel VAE that combines two powerful dynamical models:

1.  **Switching Linear Dynamical System (SLDS):** The top level of the hierarchy. The SLDS acts as a "context" or "rule-setter." It learns to identify discrete behavioral or cognitive states (e.g., "running," "resting," "decision-making") and models the smooth, linear dynamics of a continuous latent variable `z` within each state.

2.  **Continuous Attractor Neural Network (CANN):** The bottom level of the hierarchy. The CANN represents the "content" or internal representation. It is a recurrent neural network with "Mexican-hat" connectivity, allowing it to maintain a stable bump of activity that can represent a continuous variable, like position. The CANN's dynamics are driven by the latent variable `z` from the SLDS.

The VAE framework connects these two components:
*   An **encoder** (an LSTM) infers the likely latent states (`z` and discrete state `k`) from the observed neural data.
*   A **decoder** uses the inferred states to generate a reconstruction of the original data, first by having the SLDS latent state `z` drive the CANN, and then by mapping the CANN's activity back to the observed neural dimensions.

By training this model, we can simultaneously segment the data into meaningful states and understand the underlying dynamics that produce the observed neural firing patterns.

## File Structure
*   `cann_module.py`: A PyTorch module implementing the Continuous Attractor Neural Network (CANN) with 1D ring attractor dynamics.
*   `hybrid_model.py`: The main VAE model (`SLDS_CANN_VAE`) that integrates the SLDS prior from the `ssm-pytorch` library with our custom `CANN` module. It defines the full generative process and the inference network.
*   `main_pipeline.py`: The primary, user-facing script. It is structured into cells for easy interactive use in an IDE like Spyder. This file handles data loading, preprocessing (PCA), model training, and visualization of the results.

## Installation
You will need Python with pip installed. The main dependencies are PyTorch, scikit-learn, and the `ssm-pytorch` library.

1.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
    ```

2.  **Install standard packages:**
    ```bash
    pip install torch numpy matplotlib scikit-learn tqdm
    ```

3.  **Install `ssm-pytorch` from GitHub:**
    The library is not on PyPI, so it must be installed directly from its repository.
    ```bash
    pip install git+https://github.com/lindermanlab/ssm-pytorch.git
    ```

## How to Run
Follow these steps to run the analysis on your own data.

1.  **Format Your Data:**
    You need two `.npy` files:
    *   **Spike Data (`spike_data.npy`):** A 2D NumPy array of shape `(n_time_bins, n_neurons)`. This should contain your binned spike counts or firing rates.
    *   **Timestamps (`timestamps.npy`):** A 1D NumPy array of shape `(n_time_bins,)`. This array should contain an integer label for the behavioral state corresponding to each time bin.

2.  **Update File Paths:**
    Open `main_pipeline.py`. In the first cell ("IMPORTS AND CONFIGURATION"), update the placeholder paths for `SPIKE_DATA_PATH` and `TIMESTAMPS_PATH` to point to your data files.
    ```python
    # UPDATE THESE
    SPIKE_DATA_PATH = "path/to/your/spike_data.npy"
    TIMESTAMPS_PATH = "path/to/your/timestamps.npy"
    ```
    *Note: If the script cannot find these files, it will automatically generate random data so you can still run the pipeline and see how it works.*

3.  **Execute in Spyder:**
    The `main_pipeline.py` script is designed to be run cell-by-cell in Spyder (or a similar interactive environment like a Jupyter notebook). Use the `#%%` separators to navigate between cells.

    *   **Cell 1:** Run this to import all libraries and set the model/training parameters.
    *   **Cell 2:** This cell loads your data, performs PCA to reduce its dimensionality, and then segments the data into trials based on the behavioral state labels in your timestamps file.
    *   **Cell 3:** This cell initializes the `SLDS_CANN_VAE` model and runs the main training loop. You can monitor the progress and see the loss decrease over epochs. A plot of the training loss will be displayed upon completion.
    *   **Cell 4:** After training, this cell performs the analysis. It takes a sample trial, infers the latent dynamics, and generates two key plots:
        *   A scatter plot of the continuous latent states (`z`), colored by the inferred discrete state (`k`). This shows the learned dynamical structure.
        *   A kymograph of the reconstructed CANN activity, showing how the "bump" of activity evolves over time.