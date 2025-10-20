# In Silico CA3 Microcircuit

A closed-loop, biologically-constrained simulation of a dorsal hippocampal CA3 microcircuit. This model dynamically simulates and predicts network activity patterns corresponding to different arousal states in rodents (awake-moving, NREM, REM sleep).

## Model Architecture

The model is implemented in Python using the Brian2 simulator and consists of:

*   **A multi-compartmental CA3 Pyramidal Neuron:** This detailed model includes Hodgkin-Huxley dynamics for realistic spike generation and compartment-specific synaptic inputs.
*   **Four Interneuron Populations:** The model includes populations of PV+, SST+, CCK+, and Axo-axonic interneurons, each with distinct synaptic targets and dynamics.
*   **State-Dependent Afferent Inputs:** The model simulates inputs from the Dentate Gyrus (DG) and Entorhinal Cortex (EC), with distinct patterns for each arousal state (e.g., theta-rhythmic input during the awake state).

## Installation

To run the simulation, you will need Python 3.9+ and the following packages:

```bash
pip install -r requirements.txt
```

Brian2's C++ standalone mode also requires a C++ compiler (e.g., GCC) to be installed on your system.

## Usage

The main simulation script is `scripts/run_simulation.py`. You can run it from the command line with the following arguments:

*   `--state`: The arousal state to simulate. Options are `nrem`, `awake`, and `rem`. Defaults to `nrem`.
*   `--duration`: The duration of the simulation in milliseconds. Defaults to 500 ms.
*   `--closed_loop`: A flag to enable the closed-loop predictive functionality (only active in the `nrem` state).

### Examples

**Run a 1-second simulation of the NREM state:**
```bash
python3 scripts/run_simulation.py --state nrem --duration 1000
```

**Run a 500-ms simulation of the awake state:**
```bash
python3 scripts/run_simulation.py --state awake --duration 500
```

**Run a 1-second simulation of the NREM state with closed-loop feedback:**
```bash
python3 scripts/run_simulation.py --state nrem --duration 1000 --closed_loop
```

## Interpreting the Outputs

The simulation script generates a PNG image with a comprehensive set of plots:

*   **Soma Voltage:** The membrane potential of the CA3 pyramidal neuron's soma over time.
*   **Simulated LFP:** The Local Field Potential, calculated from the sum of synaptic currents.
*   **LFP Power Spectral Density:** The power spectrum of the LFP, showing the dominant frequencies in the network.
*   **Current Source Density (CSD):** A heatmap showing the location of current sinks and sources across the pyramidal neuron's morphology over time.
*   **Raster Plots:** Spike times for the different neuron populations.
