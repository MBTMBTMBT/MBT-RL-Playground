import torch
from torch import nn
from torch.nn import functional as F


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


class Encoder(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int, hidden_net_dims: list = None, input_size=(60, 80)):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.input_height, self.input_width = input_size
        self.hidden_net_dims = hidden_net_dims if hidden_net_dims else [64, 128, 256, 512, 1024]

        # Encoder layers
        self.encoder = self.build_encoder(in_channels)

        # Dynamically compute flattened dimension
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, self.input_height, self.input_width)
            encoder_output = self.encoder(dummy_input)
            self.encoder_output_channels = encoder_output.shape[1]
            self.encoder_output_height = encoder_output.shape[2]
            self.encoder_output_width = encoder_output.shape[3]
            self.flattened_dim = encoder_output.view(-1).shape[0]

        self.fc = nn.Linear(self.flattened_dim, latent_dim)

    def build_encoder(self, in_channels):
        layers = []
        for h_dim in self.hidden_net_dims:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, h_dim, kernel_size=1, stride=2, bias=False),
                nn.InstanceNorm2d(h_dim, affine=True),
                nn.LeakyReLU()
            )
            layers.append(ResidualBlock(in_channels, h_dim, stride=2, downsample=downsample))
            in_channels = h_dim
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        latent = torch.tanh(self.fc(x))  # Remove log_var, only return latent representation
        return latent


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, out_channels: int, hidden_net_dims: list, input_size=(60, 80)):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.output_height, self.output_width = input_size
        self.hidden_net_dims = hidden_net_dims[::-1]  # Reverse hidden dimensions for decoding

        # Input layer
        self.decoder_input = nn.Linear(latent_dim, hidden_net_dims[0] * self.output_height * self.output_width)

        # Decoder layers
        self.decoder = self.build_decoder()

        # Final layer to match input channels
        self.final_layer = nn.Sequential(
            nn.Conv2d(self.hidden_net_dims[-1], out_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def build_decoder(self):
        layers = []
        for i in range(len(self.hidden_net_dims) - 1):
            layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(self.hidden_net_dims[i], self.hidden_net_dims[i + 1], kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.InstanceNorm2d(self.hidden_net_dims[i + 1], affine=True),
                    nn.LeakyReLU()
                )
            )
            layers.append(ResidualBlock(self.hidden_net_dims[i + 1], self.hidden_net_dims[i + 1]))
        return nn.Sequential(*layers)

    def forward(self, z):
        batch_size = z.size(0)
        x = self.decoder_input(z)
        x = x.view(batch_size, self.hidden_net_dims[0], self.output_height, self.output_width)
        x = self.decoder(x)
        x = nn.functional.interpolate(x, size=(self.output_height, self.output_width), mode="bilinear", align_corners=False)
        x = self.final_layer(x)
        return x


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


class MultiHeadPredictor(nn.Module):
    def __init__(self, rnn_hidden_dim, hidden_dim=128):
        """
        Multi-Head Predictor with an additional hidden layer.
        Args:
            rnn_hidden_dim (int): Dimension of the RNN hidden state.
            hidden_dim (int): Fixed size of the intermediate hidden layer.
        """
        super(MultiHeadPredictor, self).__init__()

        # Shared intermediate layer
        self.shared_fc = nn.Linear(rnn_hidden_dim, hidden_dim)

        # Reward prediction head
        self.reward_head = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1)  # Output 1 scalar for reward
        )

        # Termination prediction head
        self.termination_head = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1)  # Output 1 scalar for termination flag
        )

    def forward(self, rnn_hidden):
        """
        Forward pass through the predictor.
        Args:
            rnn_hidden (Tensor): Input hidden state from RNN, shape (batch_size, rnn_hidden_dim).
        Returns:
            reward (Tensor): Predicted reward, shape (batch_size, 1).
            termination (Tensor): Predicted termination flag, shape (batch_size, 1).
        """
        # Pass through shared intermediate layer
        x = self.shared_fc(rnn_hidden)
        x = torch.relu(x)  # Activation function for shared layer

        # Predict reward and termination separately
        reward = self.reward_head(x)
        termination = self.termination_head(x)

        return reward, termination


class WorldModel(nn.Module):
    def __init__(
            self,
            encoder: Encoder,
            decoder: Decoder,
            rssm: RSSM,
            predictor: MultiHeadPredictor,
            lr: float = 1e-4,
    ):
        super(WorldModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.rssm = rssm
        self.predictor = predictor
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def train_batch(self, batch,):
        true_obs = batch["state"]  # (batch_size, seq_len, obs_dim)
        true_actions = batch["action"]  # (batch_size, seq_len, action_dim)
        true_rewards = batch["reward"]  # (batch_size, seq_len, 1)
        true_terminations = batch["terminal"]  # (batch_size, seq_len, 1)
        masks = batch["mask"]  # (batch_size, seq_len)

        # Initialize RNN hidden state
        rnn_hidden = torch.zeros(1, true_obs.size(0), self.rssm.rnn_hidden_dim)

        # Encode observations
        latents = self.encoder(true_obs)

        # RSSM: Compute prior and posterior
        prior_mean, prior_log_var, post_mean, post_log_var, rnn_hidden = self.rssm(
            latents, true_actions, rnn_hidden, observations=latents
        )

        # Decode latent states back to observations
        recon_obs = self.decoder(latents)

        # Multi-head Predictor
        rnn_output, _ = self.rssm.rnn(torch.cat([latents, true_actions], dim=-1), rnn_hidden)
        predicted_rewards, predicted_terminations = self.predictor(rnn_output)

        # Compute losses
        recon_loss = (F.mse_loss(recon_obs, true_obs, reduction="none") * masks.unsqueeze(-1)).mean()
        kl_dyn_loss = -0.5 * torch.sum((1 + prior_log_var - post_mean.pow(2) - prior_log_var.exp()) * masks.unsqueeze(-1), dim=[1, 2]).mean()
        reward_loss = (F.mse_loss(predicted_rewards, true_rewards, reduction="none") * masks).mean()
        termination_loss = (F.binary_cross_entropy_with_logits(predicted_terminations, true_terminations, reduction="none") * masks).mean()

        # Total loss
        total_loss = recon_loss + kl_dyn_loss + reward_loss + termination_loss

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss
