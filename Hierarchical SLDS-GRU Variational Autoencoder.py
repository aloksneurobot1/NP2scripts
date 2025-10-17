# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 09:26:51 2025

@author: Alok
"""

# -*- coding: utf-8 -*-
"""
=======================================================================
          Hierarchical SLDS-GRU Variational Autoencoder (VAE)
=======================================================================
This file defines the main VAE model that integrates the SLDS and the
modulated GRU to model behavior-timescale plasticity.
"""

import torch
import torch.nn as nn
import ssm

from modulated_gru import ModulatedGRU

class SLDS_GRU_VAE(nn.Module):
    def __init__(self, obs_dim, slds_latent_dim, gru_hidden_dim, num_states, rnn_hidden_dim):
        super(SLDS_GRU_VAE, self).__init__()
        
        # --- Model Dimensions ---
        self.obs_dim = obs_dim
        self.slds_latent_dim = slds_latent_dim
        self.gru_hidden_dim = gru_hidden_dim
        self.num_states = num_states

        # --- 1. Generative Model (Decoder) ---
        # The SLDS prior from the ssm library
        self.slds_prior = ssm.SLDS(obs_dim, num_states, slds_latent_dim, 
                                   transitions="standard", dynamics="gaussian",
                                   emissions="gaussian", single_subspace=True)
        
        # The modulated GRU represents the low-level working memory
        # Its input is the SLDS latent state z.
        self.decoder_gru = ModulatedGRU(input_dim=0, # No direct observation input to the generative GRU
                                        hidden_dim=gru_hidden_dim, 
                                        slds_latent_dim=slds_latent_dim)

        # Emission layer maps GRU hidden state -> Observed data
        self.emission_layer = nn.Linear(gru_hidden_dim, obs_dim)

        # --- 2. Inference Network (Encoder) ---
        self.encoder_rnn = nn.LSTM(obs_dim, rnn_hidden_dim, num_layers=1, batch_first=True)
        
        # Output layers from RNN hidden state to latent posterior parameters
        self.z_mean_layer = nn.Linear(rnn_hidden_dim, slds_latent_dim)
        self.z_logvar_layer = nn.Linear(rnn_hidden_dim, slds_latent_dim)
        self.k_logits_layer = nn.Linear(rnn_hidden_dim, num_states)

    def forward(self, y):
        """Forward pass through the VAE."""
        batch_size, T, _ = y.shape
        
        # --- INFERENCE (ENCODING) ---
        rnn_out, _ = self.encoder_rnn(y)
        q_z_mean = self.z_mean_layer(rnn_out)
        q_z_logvar = self.z_logvar_layer(rnn_out)
        q_k_logits = self.k_logits_layer(rnn_out)

        # Sample from the approximate posterior
        q_z_std = torch.exp(0.5 * q_z_logvar)
        z_sample = q_z_mean + torch.randn_like(q_z_std) * q_z_std
        
        q_k_dist = torch.distributions.Categorical(logits=q_k_logits)
        k_sample = q_k_dist.sample()

        # --- GENERATION (DECODING) ---
        # The decoder GRU is driven by the sampled latent state z
        # It gets an empty primary input, as all information comes from z.
        empty_input = torch.zeros(batch_size, T, 0).to(y.device)
        gru_output, _ = self.decoder_gru(empty_input, z_sample)
        
        # Reconstruct the observation from the GRU's hidden state
        y_recon = self.emission_layer(gru_output)
        
        return y_recon, q_z_mean, q_z_logvar, q_k_dist, z_sample, k_sample, gru_output

    def compute_loss(self, y, y_recon, q_z_mean, q_z_logvar, q_k_dist, z_sample, k_sample):
        """Computes the Evidence Lower Bound (ELBO) loss."""
        # 1. Reconstruction Loss (how well we reconstruct the data)
        recon_loss = nn.functional.mse_loss(y_recon, y, reduction='sum')
        
        # 2. KL Divergence (how much the inferred posterior deviates from the prior)
        log_prior = self.slds_prior.log_prior(k_sample, z_sample)
        log_q_z = -0.5 * torch.sum(q_z_logvar + (z_sample - q_z_mean)**2 / torch.exp(q_z_logvar))
        log_q_k = torch.sum(q_k_dist.log_prob(k_sample))
        
        kl_div = log_q_z + log_q_k - log_prior
        
        elbo = recon_loss + kl_div
        return elbo
