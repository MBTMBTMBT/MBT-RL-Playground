from typing import List, Any
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from abc import abstractmethod

from losses import FlexibleThresholdedLoss


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.InstanceNorm2d(out_channels, affine=True)
        self.relu = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.InstanceNorm2d(out_channels, affine=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class BaseVAE(nn.Module):

    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size: int, current_device: int, **kwargs) -> Tensor:
        raise NotImplementedError

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass


class VAE(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int, hidden_dims: list = None, input_size=(60, 80), ema_factor=0.01,):
        super(VAE, self).__init__()

        self.latent_dim = latent_dim
        self.input_height, self.input_width = input_size
        self.hidden_dims = hidden_dims if hidden_dims else [64, 128, 256, 512, 1024]

        # Encoder
        self.encoder = self.build_encoder(in_channels)

        # Dynamically compute flattened dimension
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, self.input_height, self.input_width)
            encoder_output = self.encoder(dummy_input)
            self.encoder_output_channels = encoder_output.shape[1]
            self.encoder_output_height = encoder_output.shape[2]
            self.encoder_output_width = encoder_output.shape[3]
            self.flattened_dim = encoder_output.view(-1).shape[0]

        self.fc_mu = nn.Linear(self.flattened_dim, latent_dim)
        self.fc_var = nn.Linear(self.flattened_dim, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, self.flattened_dim)
        self.decoder = self.build_decoder()

        # Final layer to match exact dimensions
        self.final_layer = nn.Sequential(
            nn.Conv2d(self.hidden_dims[0], in_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )

        # Apply Kaiming initialization
        # self._initialize_weights()

        # Use your custom loss function
        self.pixel_loss = FlexibleThresholdedLoss(
            use_mse_threshold=True,
            use_mae_threshold=True,
            reduction='mean',
            l1_weight=1.0,
            l2_weight=1.0,
            threshold_weight=1.0,
            non_threshold_weight=1.0,
            mse_clip_ratio=1.0,
            mae_clip_ratio=1.0,
        )

        self.recon_loss_ema = 0.0
        self.kl_loss_ema = 0.0
        self.ema_factor = ema_factor

    def build_encoder(self, in_channels):
        layers = []
        for h_dim in self.hidden_dims:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, h_dim, kernel_size=1, stride=2, bias=False),
                nn.InstanceNorm2d(h_dim, affine=True),
            )
            layers.append(ResidualBlock(in_channels, h_dim, stride=2, downsample=downsample))
            in_channels = h_dim
        return nn.Sequential(*layers)

    def build_decoder(self):
        layers = []
        hidden_dims = self.hidden_dims[::-1]
        for i in range(len(hidden_dims) - 1):
            layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1], kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.InstanceNorm2d(hidden_dims[i + 1], affine=True),
                    nn.LeakyReLU()
                )
            )
            layers.append(ResidualBlock(hidden_dims[i + 1], hidden_dims[i + 1]))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        # Initialize encoder and decoder weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')  # He initialization for Leaky ReLU
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)  # Xavier initialization for Linear layers
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d) and m in self.final_layer:
                nn.init.xavier_uniform_(m.weight)  # Xavier initialization for Tanh activation in final layer
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def encode(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        mu = F.tanh(self.fc_mu(x))
        log_var = self.fc_var(x)
        return mu, log_var

    def decode(self, z):
        x = self.decoder_input(z)
        batch_size = z.size(0)
        x = x.view(batch_size, self.encoder_output_channels, self.encoder_output_height, self.encoder_output_width)
        x = self.decoder(x)

        # Final adjustment layer to match input dimensions
        x = nn.functional.interpolate(x, size=(self.input_height, self.input_width), mode="bilinear", align_corners=False)
        x = self.final_layer(x)
        return x

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        decoded = self.decode(z)
        assert decoded.shape == x.shape, f"Mismatch: {decoded.shape} vs {x.shape}"
        return decoded, x, mu, log_var

    def loss_function(self, recons, input, mu, log_var, kld_weight=1.0, kld_threshold=1.0):
        # Use your custom pixel-wise loss
        recons_loss = self.pixel_loss(recons, input)
        self.recon_loss_ema = (1.0 - self.ema_factor) * self.recon_loss_ema + self.ema_factor * recons_loss.item()
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()
        self.kl_loss_ema = (1.0 - self.ema_factor) * self.kl_loss_ema + self.ema_factor * kld_loss.item()
        _kld_weight = kld_weight
        kld_weight = self.recon_loss_ema / (self.kl_loss_ema + 1e-10) * _kld_weight
        if kld_weight > _kld_weight:
            kld_weight = _kld_weight
        return {
            'loss': recons_loss + kld_weight * kld_loss if kld_loss.item() >= kld_threshold else recons_loss,
            'Reconstruction_Loss': recons_loss,
            'KLD': kld_loss
        }


class RSSM(nn.Module):
    def __init__(self, latent_dim, action_dim, rnn_hidden_dim, rnn_layers=1):
        super(RSSM, self).__init__()
        self.latent_dim = latent_dim
        self.rnn_hidden_dim = rnn_hidden_dim

        # RNN
        self.rnn = nn.GRU(latent_dim + action_dim, rnn_hidden_dim, rnn_layers, batch_first=True)

        # Prior and Posterior
        self.prior_fc = nn.Linear(rnn_hidden_dim, 2 * latent_dim)
        self.posterior_fc = nn.Linear(rnn_hidden_dim + latent_dim, 2 * latent_dim)

    def forward(self, latents, actions, rnn_hidden, observations=None):
        """
        Args:
            latents: (batch_size, seq_len, latent_dim)
            actions: (batch_size, seq_len, action_dim)
            rnn_hidden: (num_layers, batch_size, rnn_hidden_dim)
            observations: (batch_size, seq_len, latent_dim) or None

        Returns:
            prior_mean, prior_log_var: (batch_size, seq_len, latent_dim)
            post_mean, post_log_var: (batch_size, seq_len, latent_dim)
            rnn_hidden: (num_layers, batch_size, rnn_hidden_dim)
        """
        # Concatenate latent states and actions along the last dimension
        x = torch.cat([latents, actions], dim=-1)  # (batch_size, seq_len, latent_dim + action_dim)

        # Pass the concatenated sequence through the RNN
        rnn_output, rnn_hidden = self.rnn(x, rnn_hidden)  # rnn_output: (batch_size, seq_len, rnn_hidden_dim)

        # Compute Prior
        prior = self.prior_fc(rnn_output)  # (batch_size, seq_len, 2 * latent_dim)
        prior_mean, prior_log_var = torch.chunk(prior, 2, dim=-1)

        # Compute Posterior (if observations are provided)
        if observations is not None:
            obs_concat = torch.cat([rnn_output, observations], dim=-1)  # (batch_size, seq_len, rnn_hidden_dim + latent_dim)
            posterior = self.posterior_fc(obs_concat)  # (batch_size, seq_len, 2 * latent_dim)
            post_mean, post_log_var = torch.chunk(posterior, 2, dim=-1)
        else:
            post_mean, post_log_var = prior_mean, prior_log_var

        return prior_mean, prior_log_var, post_mean, post_log_var, rnn_hidden


# Multi-head Predictor
class MultiHeadPredictor(nn.Module):
    def __init__(self, rnn_hidden_dim):
        super(MultiHeadPredictor, self).__init__()
        self.reward_head = nn.Linear(rnn_hidden_dim, 1)
        self.termination_head = nn.Linear(rnn_hidden_dim, 1)

    def forward(self, rnn_hidden):
        reward = self.reward_head(rnn_hidden)
        termination = self.termination_head(rnn_hidden)
        return reward, termination


class WorldModel(nn.Module):
    def __init__(self, vae, rssm, predictor):
        super(WorldModel, self).__init__()
        self.vae = vae
        self.rssm = rssm
        self.predictor = predictor

    def train_batch(self, batch, optimizer, rnn_hidden_dim):
        true_obs = batch["state"]  # (batch_size, seq_len, obs_dim)
        true_actions = batch["action"]  # (batch_size, seq_len, action_dim)
        true_rewards = batch["reward"]  # (batch_size, seq_len, 1)
        true_terminations = batch["terminal"]  # (batch_size, seq_len, 1)
        masks = batch["mask"]  # (batch_size, seq_len)

        # Encode observations with VAE
        recon_obs, mean, log_var, latents = self.vae(true_obs)  # (batch_size, seq_len, latent_dim)

        # RSSM: Compute prior and posterior
        rnn_hidden = torch.zeros(1, true_obs.size(0), rnn_hidden_dim)  # Initialize hidden state
        prior_mean, prior_log_var, post_mean, post_log_var, rnn_hidden = self.rssm(
            latents, true_actions, rnn_hidden, observations=latents
        )

        # Multi-head Predictor
        rnn_output, _ = self.rssm.rnn(torch.cat([latents, true_actions], dim=-1), rnn_hidden)
        predicted_rewards, predicted_terminations = self.predictor(rnn_output)

        # Compute losses
        recon_loss = (F.mse_loss(recon_obs, true_obs, reduction="none") * masks.unsqueeze(-1)).mean()
        kl_vae_loss = -0.5 * torch.sum((1 + log_var - mean.pow(2) - log_var.exp()) * masks.unsqueeze(-1), dim=[1, 2]).mean()
        kl_dyn_loss = -0.5 * torch.sum((1 + prior_log_var - post_mean.pow(2) - prior_log_var.exp()) * masks.unsqueeze(-1), dim=[1, 2]).mean()
        kl_rep_loss = -0.5 * torch.sum((1 + post_log_var - prior_mean.pow(2) - post_log_var.exp()) * masks.unsqueeze(-1), dim=[1, 2]).mean()
        reward_loss = (F.mse_loss(predicted_rewards, true_rewards, reduction="none") * masks).mean()
        termination_loss = (F.binary_cross_entropy_with_logits(predicted_terminations, true_terminations, reduction="none") * masks).mean()

        # Total loss
        beta_vae, beta_dyn, beta_rep = 1.0, 1.0, 0.1
        total_loss = (
            recon_loss + beta_vae * kl_vae_loss + beta_dyn * kl_dyn_loss + beta_rep * kl_rep_loss + reward_loss + termination_loss
        )

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        return total_loss
