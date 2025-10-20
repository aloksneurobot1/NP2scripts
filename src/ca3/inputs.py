# src/ca3/inputs.py
#
# This module defines the afferent inputs to the CA3 microcircuit, which
# are designed to be state-dependent to simulate different arousal states.

from brian2 import *
import numpy as np

def create_afferent_inputs(populations, state='nrem', theta_freq=8*Hz):
    """
    Creates state-dependent afferent inputs from the Dentate Gyrus (DG) and
    Entorhinal Cortex (EC).

    Parameters
    ----------
    populations : dict
        A dictionary containing the Brian2 neuron objects.
    state : str, optional
        The arousal state ('nrem', 'awake', 'rem'), by default 'nrem'.
    theta_freq : Quantity, optional
        The frequency of the theta rhythm for the 'awake' and 'rem' states.

    Returns
    -------
    dict
        A dictionary containing the Brian2 input groups and their synapses.
    """

    ca3e_neuron = populations['CA3e']

    # --- State-specific input patterns ---

    if state == 'nrem':
        # NREM: Characterized by sparse, random firing.
        dg_rates = 0.5*Hz
        ec_rates = 1.0*Hz
        dg_input = PoissonGroup(100, rates=dg_rates, name='dg_input_nrem')
        ec_input = PoissonGroup(200, rates=ec_rates, name='ec_input_nrem')

    elif state in ['awake', 'rem']:
        # Awake & REM: Characterized by strong theta-rhythmic input from EC.
        if state == 'awake':
            dg_rates = 0.1*Hz  # Sparse, place-cell-like firing
            base_rate = 5*Hz
            mod_rate = 4*Hz
        else: # REM
            dg_rates = 0.2*Hz  # Different statistics for memory consolidation
            base_rate = 6*Hz
            mod_rate = 5*Hz

        # The EC firing rate is a sine wave modulated by the theta frequency.
        ec_rates_str = 'base_rate + mod_rate * sin(2 * pi * theta_freq * t)'

        dg_input = PoissonGroup(100, rates=dg_rates, name=f'dg_input_{state}')
        ec_input = PoissonGroup(200, rates=ec_rates_str,
                                namespace={'base_rate': base_rate,
                                           'mod_rate': mod_rate,
                                           'theta_freq': theta_freq},
                                name=f'ec_input_{state}')
    else:
        raise ValueError(f"Unknown state: {state}")

    # --- Synaptic connections to the pyramidal neuron ---

    # DG (mossy fibers) -> Proximal apical dendrites
    proximal_indices = ca3e_neuron.proximal_apical.indices[:]
    i_dg, j_dg = np.meshgrid(np.arange(len(dg_input)), proximal_indices)
    syn_dge = Synapses(dg_input, ca3e_neuron, on_pre='g_ampa += 15*nS', method='euler', name='syn_dge')
    syn_dge.connect(i=i_dg.flatten(), j=j_dg.flatten())

    # EC (perforant path) -> Distal apical dendrites
    distal_indices = ca3e_neuron.proximal_apical.mid_apical.distal_apical.indices[:]
    i_ec, j_ec = np.meshgrid(np.arange(len(ec_input)), distal_indices)
    syn_ece = Synapses(ec_input, ca3e_neuron, on_pre='g_ampa += 5*nS', method='euler', name='syn_ece')
    syn_ece.connect(i=i_ec.flatten(), j=j_ec.flatten())

    inputs = {
        'dg_input': dg_input,
        'syn_dge': syn_dge,
        'ec_input': ec_input,
        'syn_ece': syn_ece
    }

    return inputs
