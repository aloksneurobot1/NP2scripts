import torch
import torch.nn as nn
import numpy as np

class CANN(nn.Module):
    """
    Continuous Attractor Neural Network (CANN) module.

    Implements a 1D ring attractor network with Mexican-hat connectivity.
    The dynamics are governed by the equation:
    du/dt = (-u + W @ f(u) + k * I_ext) / tau
    where f(u) is a non-linear activation function (ReLU).
    """
    def __init__(self, n_neurons, tau=1.0, dt=0.1, k=0.1, a=1.0, J0=1.0):
        """
        Initializes the CANN module.

        Args:
            n_neurons (int): Number of neurons in the network.
            tau (float): Time constant of the network dynamics.
            dt (float): Time step for Euler integration.
            k (float): Input scaling factor.
            a (float): Activation gain.
            J0 (float): Connectivity strength.
        """
        super(CANN, self).__init__()
        self.n_neurons = n_neurons
        self.tau = tau
        self.dt = dt
        self.k = k
        self.a = a
        self.J0 = J0

        # Create the recurrent weight matrix
        self.W = self._create_weight_matrix()

        # Initialize the internal state u
        self.u = torch.zeros(1, self.n_neurons)
        self.reset()

    def _create_weight_matrix(self):
        """
        Creates the circulant weight matrix with Mexican-hat connectivity.

        The connectivity is a difference of two Gaussians, creating local
        excitation and broader inhibition.

        Returns:
            torch.Tensor: The (n_neurons, n_neurons) weight matrix.
        """
        indices = torch.arange(self.n_neurons)
        distances = torch.min(torch.abs(indices[:, None] - indices[None, :]),
                              self.n_neurons - torch.abs(indices[:, None] - indices[None, :]))

        # Parameters for the two Gaussians
        sigma_exc = 2.0
        sigma_inh = 5.0
        J_exc = self.J0
        J_inh = self.J0 * 0.8

        # Difference of Gaussians
        w = J_exc * torch.exp(-0.5 * (distances / sigma_exc)**2) - \
            J_inh * torch.exp(-0.5 * (distances / sigma_inh)**2)

        # Normalize to maintain activity levels
        w -= w.mean(dim=1, keepdim=True)

        return nn.Parameter(w, requires_grad=False) # Keep weights fixed

    def reset(self, batch_size=1):
        """
        Resets the internal state `u` to a localized bump of activity.
        """
        # Create a small bump of activity in the center of the state
        center = self.n_neurons // 2
        width = self.n_neurons // 10
        indices = torch.arange(self.n_neurons).float()
        bump = torch.exp(-0.5 * ((indices - center) / width)**2)
        self.u = bump.unsqueeze(0).repeat(batch_size, 1)

    def forward(self, I_ext):
        """
        Performs a single step of the network's dynamics using Euler integration.

        Args:
            I_ext (torch.Tensor): External input to the network. Shape (batch_size, n_neurons).

        Returns:
            torch.Tensor: The updated network state `u`.
        """
        # Non-linear activation function
        f_u = torch.relu(self.a * self.u)

        # Euler integration step
        du = (-self.u + (f_u @ self.W.T) + self.k * I_ext) / self.tau
        self.u = self.u + self.dt * du

        return self.u