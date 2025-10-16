# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 16:59:43 2025

@author: HT_bo
"""

# -*- coding: utf-8 -*-
"""
=======================================================================
 Continuous Attractor Neural Network (CANN) PyTorch Module
=======================================================================
This module implements a 1D ring attractor network, a canonical model
for place cell activity and the representation of continuous variables.
"""

import torch
import torch.nn as nn
import numpy as np

class CANN(nn.Module):
    def __init__(self, n_neurons=100, tau=1.0, dt=0.1, k=0.1, a=1.0, J0=4.0):
        super(CANN, self).__init__()
        self.n_neurons = n_neurons
        self.tau = tau
        self.dt = dt
        self.k = k # Input scaling
        self.a = a # Activation gain

        # Pre-compute the recurrent weight matrix (local excitation, broad inhibition)
        self.W = self._create_weight_matrix(n_neurons, J0)
        self.u = None  # Internal state of the network
        self.reset()

    def _create_weight_matrix(self, n, J0):
        """Creates the circulant weight matrix for a ring attractor."""
        coords = np.arange(n)
        dist_matrix = np.minimum(np.abs(coords - coords[:, np.newaxis]), n - np.abs(coords - coords[:, np.newaxis]))
        W = J0 * (np.exp(-0.05 * dist_matrix**2) - 0.5)
        # Ensure matrix is balanced
        W -= W.mean(axis=1, keepdims=True)
        return torch.tensor(W, dtype=torch.float32)
        
    def reset(self):
        """Resets the internal state of the network."""
        # Initialize with a small bump of activity
        init_bump = torch.exp(-torch.linspace(-5, 5, self.n_neurons)**2)
        self.u = nn.Parameter(init_bump.unsqueeze(0), requires_grad=False)

    def forward(self, I_ext):
        """
        Performs one time-step of the CANN dynamics simulation.
        Equation: du/dt = (-u + W @ f(u) + k * I_ext) / tau
        """
        # Ensure W is on the same device as the input
        self.W = self.W.to(I_ext.device)
        self.u = self.u.to(I_ext.device)

        # Non-linearity (e.g., ReLU or Sigmoid)
        activity = torch.relu(self.a * self.u)
        
        # Euler integration for one step
        dudt = (-self.u + activity @ self.W.T + self.k * I_ext) / self.tau
        self.u = self.u + self.dt * dudt
        
        return self.u

