import torch
import torch.nn as nn
import ssm
from torch.distributions import Normal, Categorical
from cann_module import CANN

class SLDS_CANN_VAE(nn.Module):
    """
    Hierarchical Variational Autoencoder combining a Switching Linear
    Dynamical System (SLDS) and a Continuous Attractor Neural Network (CANN).
    """
    def __init__(self, obs_dim, latent_dim, num_states, cann_dim, rnn_hidden_dim):
        """
        Initializes the SLDS_CANN_VAE model.

        Args:
            obs_dim (int): Dimensionality of the observed data.
            latent_dim (int): Dimensionality of the continuous latent space (z).
            num_states (int): Number of discrete states (k).
            cann_dim (int): Number of neurons in the CANN.
            rnn_hidden_dim (int): Hidden dimension of the encoder RNN.
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.num_states = num_states
        self.cann_dim = cann_dim
        self.rnn_hidden_dim = rnn_hidden_dim

        # 1. Generative Model (Decoder)
        # The SLDS provides the prior p(z, k)
        self.slds_prior = ssm.SLDS(obs_dim, num_states, latent_dim, emissions="gaussian")

        # Interface from SLDS latent state z to CANN external input
        self.interface = nn.Linear(latent_dim, cann_dim)

        # CANN module
        self.cann = CANN(n_neurons=cann_dim)

        # Emission layer from CANN state to observed data
        self.emission_layer = nn.Linear(cann_dim, obs_dim)

        # 2. Inference Network (Encoder) q(z, k | y)
        # Encoder RNN processes the observed data y
        self.encoder_rnn = nn.LSTM(obs_dim, rnn_hidden_dim, num_layers=1, batch_first=True)

        # Layers to get parameters of the approximate posterior
        # For q(z_t | y_{1:T})
        self.z_mean_layer = nn.Linear(rnn_hidden_dim, latent_dim)
        self.z_logvar_layer = nn.Linear(rnn_hidden_dim, latent_dim)

        # For q(k_t | y_{1:T})
        self.k_logits_layer = nn.Linear(rnn_hidden_dim, num_states)

    def forward(self, y):
        """
        Performs the forward pass of the VAE.

        Args:
            y (torch.Tensor): The observed data. Shape (batch_size, time_steps, obs_dim).

        Returns:
            y_recon (torch.Tensor): Reconstructed data.
            z_sample (torch.Tensor): Sampled continuous latent variable.
            k_sample (torch.Tensor): Sampled discrete latent variable.
            posterior_params (dict): Dictionary of posterior parameters.
        """
        batch_size, time_steps, _ = y.shape

        # --- INFERENCE (ENCODER) ---
        # Get posterior parameters from the encoder RNN
        rnn_output, _ = self.encoder_rnn(y)
        z_mean = self.z_mean_layer(rnn_output)
        z_logvar = self.z_logvar_layer(rnn_output)
        k_logits = self.k_logits_layer(rnn_output)

        # Sample from the approximate posterior using reparameterization trick for z
        q_z = Normal(z_mean, torch.exp(0.5 * z_logvar))
        z_sample = q_z.rsample()

        # Sample from the approximate posterior for k
        q_k = Categorical(logits=k_logits)
        k_sample = q_k.sample()

        posterior_params = {
            'q_z': q_z, 'z_mean': z_mean, 'z_logvar': z_logvar,
            'q_k': q_k, 'k_logits': k_logits
        }

        # --- GENERATION (DECODER) ---
        y_recon_list = []
        self.cann.reset(batch_size=batch_size) # Reset CANN state for the batch

        for t in range(time_steps):
            # Get the latent state for this timestep
            z_t = z_sample[:, t, :]

            # Drive the CANN with the latent state
            I_ext = self.interface(z_t)
            cann_state = self.cann.forward(I_ext)

            # Reconstruct the observation from the CANN state
            y_recon_t = self.emission_layer(cann_state)
            y_recon_list.append(y_recon_t)

        y_recon = torch.stack(y_recon_list, dim=1)
        return y_recon, z_sample, k_sample, posterior_params

    def compute_loss(self, y, y_recon, posterior_params, z_sample, k_sample):
        """
        Computes the Evidence Lower Bound (ELBO) loss.

        ELBO = E_q[log p(y|z,k)] - KL(q(z,k|y) || p(z,k))

        Args:
            y (torch.Tensor): Original data.
            y_recon (torch.Tensor): Reconstructed data.
            posterior_params (dict): Dictionary from the forward pass.
            z_sample (torch.Tensor): Sampled continuous latent variables.
            k_sample (torch.Tensor): Sampled discrete latent variables.

        Returns:
            torch.Tensor: The total ELBO loss.
        """
        # 1. Reconstruction Loss: E_q[log p(y|z,k)]
        recon_loss = nn.MSELoss(reduction='sum')(y_recon, y) / y.shape[0]

        # 2. KL Divergence: KL(q || p)
        # We need log q(z|y), log q(k|y), and log p(z,k)

        # log q(z|y) and log q(k|y) from the posterior
        log_q_z = posterior_params['q_z'].log_prob(z_sample).sum(dim=[1, 2])
        log_q_k = posterior_params['q_k'].log_prob(k_sample).sum(dim=1)
        log_q = log_q_z + log_q_k

        # log p(z,k) from the SLDS prior
        # Note: ssm library expects (T, D) so we need to permute dimensions
        z_sample_permuted = z_sample.permute(1, 0, 2)
        k_sample_permuted = k_sample.permute(1, 0)
        log_p = self.slds_prior.log_prior(k_sample_permuted, z_sample_permuted)
        log_p = log_p.sum(dim=0) # Sum over time

        # KL divergence is E_q[log q - log p]
        kl_div = (log_q - log_p).mean()

        # Total loss (negative ELBO)
        loss = recon_loss + kl_div
        return loss