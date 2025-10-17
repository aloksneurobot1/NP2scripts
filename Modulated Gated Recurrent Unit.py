# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 09:26:15 2025

@author: Alok
"""

# -*- coding: utf-8 -*-
"""
=======================================================================
            Modulated Gated Recurrent Unit (GRU) Module
=======================================================================
This module implements a GRU whose input is contextually modulated by
an external latent variable (from an SLDS), allowing for top-down control.
"""
import torch
import torch.nn as nn

class ModulatedGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, slds_latent_dim):
        super(ModulatedGRU, self).__init__()
        self.hidden_dim = hidden_dim
        
        # The GRU's input will be the concatenation of the observed data
        # (or its embedding) and the modulating SLDS latent state z.
        gru_input_dim = input_dim + slds_latent_dim
        
        self.gru_cell = nn.GRU(gru_input_dim, hidden_dim, batch_first=True)

    def forward(self, x, z, h_init=None):
        """
        Performs one forward pass of the GRU over a sequence.
        
        Args:
            x (Tensor): The primary input sequence (batch, seq_len, input_dim).
            z (Tensor): The modulating SLDS latent sequence (batch, seq_len, slds_latent_dim).
            h_init (Tensor, optional): Initial hidden state. Defaults to None (zeros).
        
        Returns:
            Tensor: Output sequence from the GRU (batch, seq_len, hidden_dim).
            Tensor: Final hidden state of the GRU (1, batch, hidden_dim).
        """
        # Concatenate the primary input with the modulating SLDS state
        modulated_input = torch.cat((x, z), dim=-1)
        
        # Pass through the GRU
        output, h_final = self.gru_cell(modulated_input, h_init)
        
        return output, h_final
