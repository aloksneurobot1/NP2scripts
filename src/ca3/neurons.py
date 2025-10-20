# src/ca3/neurons.py
#
# This module defines the neuron models used in the CA3 microcircuit simulation.
# It includes the detailed multi-compartmental CA3 pyramidal neuron and the
# simpler Izhikevich models for the various interneuron populations.

from brian2 import *

def create_ca3_pyramidal_neuron():
    """
    Creates a multi-compartmental model of a CA3 Pyramidal Neuron.

    The model is based on Hodgkin-Huxley dynamics for sodium and potassium channels,
    includes a slow M-type potassium current for spike-frequency adaptation, and
    has compartments for a soma, basal dendrite, and a segmented apical dendrite.
    Synaptic inputs (AMPA and GABA-A) are modeled as point currents.
    """

    # --- Morphology Definition ---
    # A simplified 5-compartment morphology representing the key dendritic domains.
    morpho = Soma(diameter=30*um)
    morpho.basal = Cylinder(length=100*um, diameter=2*um, n=1)
    morpho.proximal_apical = Cylinder(length=150*um, diameter=3*um, n=1)
    morpho.proximal_apical.mid_apical = Cylinder(length=150*um, diameter=2*um, n=1)
    morpho.proximal_apical.mid_apical.distal_apical = Cylinder(length=200*um, diameter=1*um, n=1)

    # --- Passive Properties ---
    Cm = 1 * uF / cm**2  # Membrane capacitance
    Ri = 150 * ohm * cm  # Intracellular resistivity

    # --- Model Equations ---
    # The core of the model, defining the transmembrane and synaptic currents.
    eqs = """
    # Total transmembrane current density
    Im = gL*(EL - v) + gNa*m**3*h*(ENa - v) + gK*n**4*(EK - v) + gM*w*(EK - v) : amp/meter**2

    # Synaptic currents (defined as point currents)
    I_ampa = g_ampa * (E_ampa - v) : amp (point current)
    I_gaba = g_gaba * (E_gaba - v) : amp (point current)
    g_ampa : siemens  # Excitatory conductance
    g_gaba : siemens  # Inhibitory conductance

    # --- Hodgkin-Huxley Channel Gating Variables ---
    dm/dt = alpha_m*(1-m) - beta_m*m : 1  # Sodium channel activation
    dh/dt = alpha_h*(1-h) - beta_h*h : 1  # Sodium channel inactivation
    dn/dt = alpha_n*(1-n) - beta_n*n : 1  # Potassium channel activation
    dw/dt = (w_inf - w) / tau_w : 1      # M-current activation

    # Rate constants for gating variables (voltage-dependent)
    alpha_m = (0.32*(v - (-54*mV))/(4*mV)) / (1 - exp(-(v - (-54*mV))/(4*mV))) / ms : Hz
    beta_m = (0.28*(v - (-27*mV))/(5*mV)) / (exp((v - (-27*mV))/(5*mV)) - 1) / ms : Hz
    alpha_h = 0.128*exp(-(v - (-50*mV))/(18*mV)) / ms : Hz
    beta_h = 4 / (1 + exp(-(v - (-27*mV))/(5*mV))) / ms : Hz
    alpha_n = (0.032*(v - (-52*mV))/(5*mV)) / (1 - exp(-(v - (-52*mV))/(5*mV))) / ms : Hz
    beta_n = 0.5*exp(-(v - (-57*mV))/(40*mV)) / ms : Hz

    # M-current kinetics
    w_inf = 1 / (1 + exp(-(v + 35*mV)/(10*mV))) : 1
    tau_w = 400*ms : second

    # --- Parameters ---
    gL : siemens/meter**2
    gNa : siemens/meter**2
    gK : siemens/meter**2
    gM : siemens/meter**2
    EL : volt
    ENa : volt
    EK : volt
    E_ampa : volt
    E_gaba : volt
    """

    # --- Neuron Creation and Initialization ---
    neuron = SpatialNeuron(morphology=morpho, model=eqs, Cm=Cm, Ri=Ri,
                           method="exponential_euler",
                           threshold='v > -20*mV',
                           refractory='v > 0*mV',
                           threshold_location=0) # Spike detection at the soma

    # Set default values for all parameters
    neuron.gL = 5e-5 * siemens/cm**2
    neuron.EL = -65*mV
    neuron.gNa = 100 * msiemens/cm**2
    neuron.ENa = 50*mV
    neuron.gK = 30 * msiemens/cm**2
    neuron.EK = -90*mV
    neuron.gM = 4e-5 * siemens/cm**2
    neuron.E_ampa = 0*mV
    neuron.E_gaba = -70*mV

    # Initialize state variables
    neuron.v = -65*mV
    neuron.h = 1
    # Increase Na+ channel density at the soma to promote spike initiation
    neuron.main.gNa = 150 * msiemens/cm**2

    return neuron

# --- Interneuron Models ---
# For computational efficiency, interneurons are modeled using the simpler
# Izhikevich model, which can reproduce a wide variety of firing patterns.

def izhikevich_model(name, N, a, b, c, d):
    """A helper function to create Izhikevich neuron models."""
    eqs = '''
    dv/dt = (0.04*v**2 + 5*v + 140 - u + I_syn) / ms : 1
    du/dt = a_param * (b_param * v - u) / ms : 1
    I_syn : 1
    a_param : 1
    b_param : 1
    c_param : 1
    d_param : 1
    '''
    reset = '''
    v = c_param
    u = u + d_param
    '''
    neuron = NeuronGroup(N, model=eqs, reset=reset, threshold='v >= 30', method='euler', name=name)
    neuron.v = -65
    neuron.u = -13
    neuron.a_param = a
    neuron.b_param = b
    neuron.c_param = c
    neuron.d_param = d
    neuron.I_syn = 0
    return neuron

def create_pv_basket_cell(N=1):
    """Creates a population of fast-spiking PV+ basket cells."""
    return izhikevich_model('PV', N, a=0.1, b=0.2, c=-65, d=2)

def create_sst_olm_cell(N=1):
    """Creates a population of low-threshold spiking SST+ O-LM cells."""
    return izhikevich_model('SST', N, a=0.02, b=0.25, c=-65, d=2)

def create_cck_basket_cell(N=1):
    """Creates a population of regular-spiking CCK+ basket cells."""
    return izhikevich_model('CCK', N, a=0.02, b=0.2, c=-65, d=8)

def create_axo_axonic_cell(N=1):
    """Creates a population of fast-spiking axo-axonic (chandelier) cells."""
    return izhikevich_model('AA', N, a=0.1, b=0.2, c=-65, d=2)
