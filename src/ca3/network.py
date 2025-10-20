# src/ca3/network.py
#
# This module is responsible for creating and assembling the neuron populations
# that form the CA3 microcircuit.

from brian2 import *
from src.ca3.neurons import (create_ca3_pyramidal_neuron,
                               create_pv_basket_cell,
                               create_sst_olm_cell,
                               create_cck_basket_cell,
                               create_axo_axonic_cell)

def create_network_populations(N_PV=50, N_SST=50, N_CCK=50, N_AA=50):
    """
    Creates the neuron populations for the CA3 microcircuit model.

    Note: This function currently creates a single, detailed multi-compartmental
    CA3 pyramidal neuron and populations of simpler single-compartment interneurons.
    This hybrid approach allows for detailed analysis of synaptic integration and
    LFP generation while keeping the simulation computationally tractable.

    Parameters
    ----------
    N_PV : int, optional
        The number of PV+ basket cells.
    N_SST : int, optional
        The number of SST+ O-LM cells.
    N_CCK : int, optional
        The number of CCK+ basket cells.
    N_AA : int, optional
        The number of Axo-axonic cells.

    Returns
    -------
    dict
        A dictionary containing the Brian2 neuron objects for each population.
    """

    # Create the single detailed CA3 pyramidal neuron
    ca3e_neuron = create_ca3_pyramidal_neuron()

    # Create the populations of different interneuron types
    populations = {
        'CA3e': ca3e_neuron,
        'PV': create_pv_basket_cell(N_PV),
        'SST': create_sst_olm_cell(N_SST),
        'CCK': create_cck_basket_cell(N_CCK),
        'AA': create_axo_axonic_cell(N_AA)
    }

    return populations

# Note: The Izhikevich helper functions are defined in src/ca3/neurons.py
# to keep this file focused on network assembly.
