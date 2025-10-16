# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 17:00:22 2025

@author: HT_bo
"""

# -*- coding: utf-8 -*-
"""
=======================================================================
 Hierarchical SLDS-CANN Variational Autoencoder (VAE)
=======================================================================
This file defines the main VAE model that integrates the SLDS and CANN.
"""

import torch
import torch.nn as nn
import ssm

from cann_module import CANN

class SLDS_CANN_VAE(nn.Module):
    def __init__(self, obs_dim, latent_dim, num_states, cann_dim, rnn_hidden_dim):
        super(SLDS_CANN_VAE, self).__init__()
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.num_states = num_states
        self.cann_dim = cann_dim

        # 1. The Generative Model (Decoder)
        # The SLDS prior is provided by the ssm library
        self.slds_prior = ssm.SLDS(obs_dim, num_states, latent_dim, 
                                   transitions="standard",
                                   dynamics="gaussian",
                                   emissions="gaussian",
                                   single_subspace=True)
        
        # Interface from SLDS latent state z -> CANN external input
        self.interface = nn.Linear(latent_dim, cann_dim)
        
        # The CANN module
        self.cann = CANN(n_neurons=cann_dim)

        # Emission from CANN state -> Observed data
        self.emission_layer = nn.Linear(cann_dim, obs_dim)

        # 2. The Inference Network (Encoder)
        # An LSTM processes the observed data sequence
        self.encoder_rnn = nn.LSTM(obs_dim, rnn_hidden_dim, num_layers=1, batch_first=True)
        
        # Output layers from RNN hidden state to latent posterior parameters
        self.z_mean_layer = nn.Linear(rnn_hidden_dim, latent_dim)
        self.z_logvar_layer = nn.Linear(rnn_hidden_dim, latent_dim)
        self.k_logits_layer = nn.Linear(rnn_hidden_dim, num_states)

    def forward(self, y):
        """
        Forward pass through the VAE.
        y: observed data (batch_size, time_steps, obs_dim)
        """
        T = y.shape[1]
        
        # --- INFERENCE (ENCODING) ---
        rnn_out, _ = self.encoder_rnn(y)
        q_z_mean = self.z_mean_layer(rnn_out)
        q_z_logvar = self.z_logvar_layer(rnn_out)
        q_k_logits = self.k_logits_layer(rnn_out)

        # Sample from the approximate posterior
        q_z_std = torch.exp(0.5 * q_z_logvar)
        z_sample = q_z_mean + torch.randn_like(q_z_std) * q_z_std # Reparameterization trick
        
        q_k_dist = torch.distributions.Categorical(logits=q_k_logits)
        k_sample = q_k_dist.sample()

        # --- GENERATION (DECODING) ---
        self.cann.reset()
        y_recon_list = []
        for t in range(T):
            zt = z_sample[:, t, :]
            
            # Interface translates z -> I_ext
            I_ext = self.interface(zt)
            
            # CANN evolves for one step
            cann_u = self.cann(I_ext)
            
            # Emission layer maps CANN state to reconstructed observation
            y_recon_t = self.emission_layer(cann_u)
            y_recon_list.append(y_recon_t)
            
        y_recon = torch.stack(y_recon_list, dim=1)
        
        # Return all necessary components for loss calculation
        return y_recon, q_z_mean, q_z_logvar, q_k_dist, z_sample, k_sample

    def compute_loss(self, y, y_recon, q_z_mean, q_z_logvar, q_k_dist, z_sample, k_sample):
        """Computes the ELBO loss."""
        # 1. Reconstruction Loss
        recon_loss = nn.functional.mse_loss(y_recon, y, reduction='sum')
        
        # 2. KL Divergence
        # The ssm library provides a convenient way to get the log prior probability
        # log p(z, k)
        log_prior = self.slds_prior.log_prior(k_sample, z_sample)
        
        # log q(z | y) - Gaussian
        log_q_z = -0.5 * torch.sum(q_z_logvar + (z_sample - q_z_mean)**2 / torch.exp(q_z_logvar))
        
        # log q(k | y) - Categorical
        log_q_k = torch.sum(q_k_dist.log_prob(k_sample))
        
        kl_div = log_q_z + log_q_k - log_prior
        
        # ELBO = Reconstruction - KL Divergence
        elbo = recon_loss + kl_div
        return elbo

