# src/ca3/synapses.py
#
# This module defines the synaptic connections between the different neuron
# populations in the CA3 microcircuit model.

from brian2 import *
import numpy as np

def connect_populations(populations):
    """
    Connects the neuron populations with synapses, targeting specific
    compartments of the multi-compartmental CA3 neuron where appropriate.

    Parameters
    ----------
    populations : dict
        A dictionary containing the Brian2 neuron objects for each population.

    Returns
    -------
    dict
        A dictionary containing the Brian2 Synapses objects.
    """

    ca3e_neuron = populations['CA3e']

    # --- Synaptic Model Definitions ---
    # AMPA for excitatory synapses, GABA-A for inhibitory synapses.
    # The synaptic weight is named 'w_syn' to avoid conflicts with the
    # M-current state variable 'w' in the pyramidal neuron model.
    eqs_ampa = 'w_syn : siemens'
    on_pre_ampa = 'g_ampa += w_syn'

    eqs_gaba_a = 'w_syn : siemens'
    on_pre_gaba_a = 'g_gaba += w_syn'

    # --- Synaptic Connections ---

    synapses = {}

    # --- Compartment-specific targeting on the CA3 Pyramidal Neuron ---
    soma_indices = ca3e_neuron.main.indices[:]
    proximal_indices = ca3e_neuron.proximal_apical.indices[:]
    soma_and_proximal_indices = np.concatenate((soma_indices, proximal_indices))
    distal_indices = ca3e_neuron.proximal_apical.mid_apical.distal_apical.indices[:]

    # --- Inhibitory Connections to Pyramidal Neuron ---

    # 1. PV -> CA3e (Perisomatic inhibition)
    pv_pop = populations['PV']
    i, j = np.meshgrid(np.arange(len(pv_pop)), soma_and_proximal_indices)
    syn_pve = Synapses(pv_pop, ca3e_neuron, model=eqs_gaba_a, on_pre=on_pre_gaba_a, method='euler', name='syn_pve')
    syn_pve.connect(i=i.flatten(), j=j.flatten())
    syn_pve.w_syn = 5 * nS
    syn_pve.delay = 1*ms
    synapses['pve'] = syn_pve

    # 2. SST -> CA3e (Distal dendritic inhibition)
    sst_pop = populations['SST']
    i, j = np.meshgrid(np.arange(len(sst_pop)), distal_indices)
    syn_sste = Synapses(sst_pop, ca3e_neuron, model=eqs_gaba_a, on_pre=on_pre_gaba_a, method='euler', name='syn_sste')
    syn_sste.connect(i=i.flatten(), j=j.flatten())
    syn_sste.w_syn = 4 * nS
    syn_sste.delay = 3*ms
    synapses['sste'] = syn_sste

    # 3. CCK -> CA3e (Perisomatic inhibition)
    cck_pop = populations['CCK']
    i, j = np.meshgrid(np.arange(len(cck_pop)), soma_and_proximal_indices)
    syn_ccke = Synapses(cck_pop, ca3e_neuron, model=eqs_gaba_a, on_pre=on_pre_gaba_a, method='euler', name='syn_ccke')
    syn_ccke.connect(i=i.flatten(), j=j.flatten())
    syn_ccke.w_syn = 4.5 * nS
    syn_ccke.delay = 2*ms
    synapses['ccke'] = syn_ccke

    # 4. AA -> CA3e (Axon initial segment inhibition)
    aa_pop = populations['AA']
    i, j = np.meshgrid(np.arange(len(aa_pop)), soma_indices)
    syn_aae = Synapses(aa_pop, ca3e_neuron, model=eqs_gaba_a, on_pre=on_pre_gaba_a, method='euler', name='syn_aae')
    syn_aae.connect(i=i.flatten(), j=j.flatten())
    syn_aae.w_syn = 8 * nS # Powerful, targeted inhibition
    syn_aae.delay = 0.5*ms
    synapses['aae'] = syn_aae

    # Note: Connections between interneurons and from the pyramidal cell to
    # interneurons are omitted in this version for simplicity.

    return synapses
