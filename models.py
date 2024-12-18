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
        """
        Decoder to reconstruct observations from latent representations.

        Args:
            latent_dim (int): Dimension of the latent representation.
            out_channels (int): Number of output channels (e.g., 3 for RGB).
            hidden_net_dims (list): Hidden dimensions for each decoder layer.
            input_size (tuple): Input size of the original image (height, width).
        """
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        self.hidden_net_dims = hidden_net_dims[::-1]  # Reverse the dimensions for decoding
        self.input_height, self.input_width = input_size

        # Compute the starting height and width of the decoder feature map
        self.start_height = self.input_height // (2 ** len(self.hidden_net_dims))
        self.start_width = self.input_width // (2 ** len(self.hidden_net_dims))

        # Input layer to map latent vector to initial feature map
        self.decoder_input = nn.Linear(latent_dim, self.hidden_net_dims[0] * self.start_height * self.start_width)

        # Build decoder layers
        self.decoder = self.build_decoder()

        # Final adjustment layer to match the output dimensions
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
        """
        Forward pass through the decoder.
        Args:
            z (Tensor): Latent vector (batch_size, latent_dim).
        Returns:
            Tensor: Reconstructed observation (batch_size, out_channels, height, width).
        """
        batch_size = z.size(0)

        # Map latent vector to initial feature map
        x = self.decoder_input(z)
        x = x.view(batch_size, self.hidden_net_dims[0], self.start_height, self.start_width)

        # Pass through decoder layers
        x = self.decoder(x)

        # Upsample to the original input size and apply final adjustment layer
        x = nn.functional.interpolate(x, size=(self.input_height, self.input_width), mode="bilinear", align_corners=False)
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
        reward = self.reward_head(x).squeeze()
        termination = self.termination_head(x).squeeze()

        return reward, termination


class WorldModel(nn.Module):
    def __init__(
            self,
            encoder: Encoder,
            decoder: Decoder,
            rssm: RSSM,
            predictor: MultiHeadPredictor,
            lr: float = 1e-4,
            device: torch.device = "cpu",
    ):
        super(WorldModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.rssm = rssm
        self.predictor = predictor
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.device = device
        self.to(device)

    def train_batch(self, batch, recon_weight=1.0, kl_dyn_weight=0.1, kl_rep_weight=0.1, reward_weight=1.0,
                    termination_weight=1.0, kl_min=1.0):
        """
        Train the world model on a single batch.

        Args:
            batch (dict): A batch of training data with keys:
                - "state": Real observations (batch_size, seq_len, channels, height, width)
                - "action": Actions taken (batch_size, seq_len, action_dim)
                - "reward": Rewards received (batch_size, seq_len, 1)
                - "terminal": Termination flags (batch_size, seq_len, 1)
                - "mask": Masks for valid steps (batch_size, seq_len)
            recon_weight (float): Weight for reconstruction loss.
            kl_dyn_weight (float): Weight for dynamic KL loss.
            kl_rep_weight (float): Weight for representation KL loss.
            reward_weight (float): Weight for reward prediction loss.
            termination_weight (float): Weight for termination prediction loss.
            kl_min (float): Minimum threshold for KL loss to be included.
        Returns:
            total_loss (float): The total loss for the batch.
        """
        # Convert batch data to PyTorch tensors
        true_obs = torch.tensor(batch["state"], dtype=torch.float32, device=self.device)
        true_actions = torch.tensor(batch["action"], dtype=torch.float32, device=self.device)
        true_rewards = torch.tensor(batch["reward"], dtype=torch.float32, device=self.device)
        true_terminations = torch.tensor(batch["terminal"], dtype=torch.float32, device=self.device)
        masks = torch.tensor(batch["mask"], dtype=torch.float32, device=self.device)

        # Initialize RNN hidden state
        rnn_hidden = torch.zeros(1, true_obs.size(0), self.rssm.rnn_hidden_dim, device=self.device)

        # Step-by-step encoding for observations
        batch_size, seq_len, channels, height, width = true_obs.size()
        latents = []
        for t in range(seq_len):
            frame = true_obs[:, t]  # Extract a single frame (batch_size, channels, height, width)
            latent = self.encoder(frame)  # Encode the frame to latent representation
            latents.append(latent)

        # Stack latents along the sequence dimension
        latents = torch.stack(latents, dim=1)

        # RSSM forward pass (compute prior and posterior)
        prior_mean, prior_log_var, post_mean, post_log_var, rnn_hidden = self.rssm(
            latents, true_actions, rnn_hidden, observations=latents
        )

        # Step-by-step decoding from latents
        recon_obs = []
        for t in range(seq_len):
            recon_frame = self.decoder(latents[:, t])  # Decode latent representation back to observation
            recon_obs.append(recon_frame)

        # Stack reconstructed frames
        recon_obs = torch.stack(recon_obs, dim=1)

        # Multi-head Predictor for reward and termination
        rnn_output, _ = self.rssm.rnn(torch.cat([latents, true_actions], dim=-1), rnn_hidden)
        predicted_rewards, predicted_terminations = self.predictor(rnn_output)

        # Compute losses
        recon_loss = (F.mse_loss(recon_obs, true_obs, reduction="none") * masks.unsqueeze(-1).unsqueeze(-1).unsqueeze(
            -1)).mean()

        kl_dyn_loss = -0.5 * torch.sum(
            (1 + prior_log_var - post_mean.pow(2) - prior_log_var.exp()) * masks.unsqueeze(-1), dim=[1, 2]
        ).mean()

        kl_rep_loss = -0.5 * torch.sum(
            (1 + post_log_var - prior_mean.pow(2) - post_log_var.exp()) * masks.unsqueeze(-1), dim=[1, 2]
        ).mean()

        reward_loss = (F.mse_loss(predicted_rewards, true_rewards, reduction="none") * masks).mean()
        termination_loss = (F.binary_cross_entropy_with_logits(predicted_terminations, true_terminations,
                                                               reduction="none") * masks).mean()

        # KL Loss lower bound
        if kl_dyn_loss.item() < kl_min:
            kl_dyn_loss = torch.tensor(0.0, device=self.device)
        if kl_rep_loss.item() < kl_min:
            kl_rep_loss = torch.tensor(0.0, device=self.device)

        # Compute time-step weights
        time_weights = torch.linspace(1.0, 0.1, steps=seq_len, device=self.device).unsqueeze(0).repeat(batch_size, 1)

        # Apply time-step weights
        weighted_recon_loss = (time_weights * recon_loss).mean()
        weighted_kl_dyn_loss = (time_weights * kl_dyn_loss).mean()
        weighted_kl_rep_loss = (time_weights * kl_rep_loss).mean()
        weighted_reward_loss = (time_weights * reward_loss).mean()
        weighted_termination_loss = (time_weights * termination_loss).mean()

        # Total loss
        total_loss = (
                recon_weight * weighted_recon_loss +
                kl_dyn_weight * weighted_kl_dyn_loss +
                kl_rep_weight * weighted_kl_rep_loss +
                reward_weight * weighted_reward_loss +
                termination_weight * weighted_termination_loss
        )

        # Backpropagation
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss
