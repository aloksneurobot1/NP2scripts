from brian2 import *
import sys
import os
import argparse
import matplotlib.pyplot as plt

# --- Optimization ---
# Use C++ code generation for faster simulations
set_device('cpp_standalone')

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ca3.network import create_network_populations
from src.ca3.synapses import connect_populations
from src.ca3.inputs import create_afferent_inputs
from src.ca3.analysis import calculate_lfp, plot_psd, calculate_csd, plot_csd, detect_pre_swr_state

def run_simulation(state='nrem', duration=1*second, closed_loop=False):
    """
    Runs a simulation of the CA3 microcircuit with optional closed-loop feedback.

    Parameters
    ----------
    state : str, optional
        The arousal state to simulate ('nrem', 'awake', 'rem'), by default 'nrem'.
    duration : Quantity, optional
        The duration of the simulation.
    closed_loop : bool, optional
        Whether to enable the closed-loop feedback mechanism.

    Returns
    -------
    dict
        A dictionary containing the simulation results.
    """
    start_scope()

    # --- Network Setup ---
    populations = create_network_populations(N_PV=50, N_SST=50, N_CCK=50, N_AA=50)
    synapses = connect_populations(populations)
    inputs = create_afferent_inputs(populations, state=state)
    interneuron_drive = PoissonInput(target=populations['PV'], target_var='I_syn',
                                     N=100, rate=8*Hz, weight=5)

    spike_monitors = {'CA3e': SpikeMonitor(populations['CA3e']), 'PV': SpikeMonitor(populations['PV'])}
    ca3_monitor = StateMonitor(populations['CA3e'], ('v', 'g_ampa', 'g_gaba', 'Im'), record=True)

    net = Network(list(populations.values()) + list(synapses.values()) +
                  list(inputs.values()) + [interneuron_drive] +
                  list(spike_monitors.values()) + [ca3_monitor])

    # --- Simulation Loop ---
    if closed_loop and state == 'nrem':
        print("Running NREM simulation with closed-loop feedback...")
        window_duration = 50 * ms
        num_windows = int(duration / window_duration)
        for i in range(num_windows):
            net.run(window_duration, report='text')
            lfp = calculate_lfp(ca3_monitor, populations)
            if len(lfp) > 1024:
                lfp_window = lfp[-1024:]
                if detect_pre_swr_state(lfp_window, defaultclock.dt):
                    print(f"Pre-SWR state detected at t={net.t/ms:.2f} ms. Modulating inhibition.")
                    synapses['pve'].w_syn *= 1.1 # Increase PV inhibition
    else:
        print(f"Running {state.upper()} simulation...")
        net.run(duration, report='text')

    print("Simulation finished.")

    # --- Analysis ---
    lfp = calculate_lfp(ca3_monitor, populations)
    csd = calculate_csd(ca3_monitor, populations['CA3e'])

    return {'spike_monitors': spike_monitors, 'ca3_monitor': ca3_monitor,
            'lfp': lfp, 'csd': csd, 'populations': populations, 'dt': defaultclock.dt}

def plot_results(results, args):
    """
    Generates and saves plots of the simulation results.
    """
    plt.figure(figsize=(15, 15))

    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(results['ca3_monitor'].t/ms, results['ca3_monitor'].v[0]/mV)
    ax1.set_title(f'Soma Voltage ({args.state.upper()})')

    ax2 = plt.subplot(3, 2, 3)
    ax2.plot(results['ca3_monitor'].t/ms, results['lfp'])
    ax2.set_title('Simulated LFP')

    ax3 = plt.subplot(3, 2, 5)
    plot_psd(results['lfp'], results['dt'], ax3)

    ax4 = plt.subplot(1, 2, 2)
    compartment_depths = results['populations']['CA3e'].distance
    plot_csd(results['csd'], results['ca3_monitor'].t, compartment_depths, ax4)

    plt.tight_layout()
    output_filename = f'simulation_{args.state}{"_closed_loop" if args.closed_loop else ""}.png'
    plt.savefig(output_filename)
    print(f"Saved simulation results to {output_filename}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run a CA3 microcircuit simulation.")
    parser.add_argument('--state', type=str, default='nrem', help="Arousal state ('nrem', 'awake', 'rem').")
    parser.add_argument('--duration', type=float, default=500, help="Simulation duration in ms.")
    parser.add_argument('--closed_loop', action='store_true', help="Enable closed-loop feedback.")
    args = parser.parse_args()

    results = run_simulation(state=args.state, duration=args.duration*ms, closed_loop=args.closed_loop)
    plot_results(results, args)
