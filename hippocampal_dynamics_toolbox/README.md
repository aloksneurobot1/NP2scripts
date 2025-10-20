# Hippocampal Dynamics Toolbox

This repository provides a Python-based toolbox for analyzing neuropixels data from the hippocampus, focusing on LFP, CSD, spectral, and single-unit analyses. The pipeline is designed to be modular and is structured to replicate key analyses from studies like Ruiz et al. (2021) and Bieri et al. (2014), as well as to perform novel analyses on synaptic plasticity.

## Project Purpose

The primary goal of this toolbox is to provide a comprehensive pipeline for processing raw neuropixels data and extracting meaningful neurophysiological insights. This includes:
-   Preprocessing of LFP data and calculation of Current Source Density (CSD).
-   Spectral analysis of LFP and CSD, including theta-phase-locked events and phase-amplitude coupling.
-   Spike analysis, including cell-type classification and spike-phase locking.
-   Plasticity analysis, comparing neural dynamics across different behavioral epochs (e.g., pre- vs. post-social interaction).

## Environment Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd hippocampal_dynamics_toolbox
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

## Data Structure and Configuration

Before running the analysis, you must structure your data and configure the `config.yaml` file.

1.  **Data:**
    -   Place your raw SpikeGLX data in a directory.
    -   Place your Kilosort output in a separate directory.
    -   Create a channel map file (e.g., `channel_map.csv`) that contains information about the location of each electrode, including `shank` and `y` coordinates.

2.  **Configuration (`config.yaml`):**
    -   Open the `config.yaml` file and update the file paths to point to your data directories and channel map file.
    -   Adjust the analysis parameters (e.g., sampling rates, filter bands) as needed for your experiment.

## Workflow

The analysis pipeline is designed to be run sequentially. The scripts in the `/scripts` directory are numbered to indicate the order of execution.

1.  **`preprocess.py`**: Preprocesses the raw data to generate a clean LFP file (`lfp.bin`).
    ```bash
    python scripts/preprocess.py
    ```

2.  **`csd_analysis.py`**: Calculates the Current Source Density (CSD) from the LFP data (`csd.bin`).
    ```bash
    python scripts/csd_analysis.py
    ```

3.  **`spectral_analysis.py`**: Performs spectral analyses, such as theta-locked CSD and PAC. This script can be run to see example outputs, but its main purpose is to provide functions for the figure generation script.

4.  **`spike_analysis.py`**: Performs single-unit analyses, including cell-type classification and spike-phase locking.
    ```bash
    python scripts/spike_analysis.py
    ```

5.  **`plasticity_analysis.py`**: Analyzes changes in neural dynamics across different epochs.
    ```bash
    python scripts/plasticity_analysis.py
    ```

6.  **`figures/ruiz_figures.py`**: Generates figures based on the spectral analysis.
    ```bash
    python figures/ruiz_figures.py
    ```
