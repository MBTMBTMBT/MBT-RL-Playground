import torch
from torch import nn
from torch.nn import functional as F

from losses import FlexibleThresholdedLoss


class SimpleTransitionModel(nn.Module):
    def __init__(
            self,
            obs_dim: int,
            action_dim: int,
            network_layers: list = None,
            dropout: float = 0.0,
            lr: float = 1e-4,
            device=torch.device("cpu"),
    ):
        """
        A simple fully connected neural network for transition modeling.

        Args:
            obs_dim (int): Dimension of the observation space.
            action_dim (int): Dimension of the action space.
            network_layers (list, optional): List defining the sizes of hidden layers. Defaults to [64, 64].
            dropout (float, optional): Dropout rate. Defaults to 0.0.
        """
        super(SimpleTransitionModel, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.network_layers = network_layers if network_layers else [64, 64]
        self.dropout = dropout
        self.device = device

        layers = []
        input_dim = obs_dim + action_dim  # Concatenate observation and action

        # Add hidden layers
        for i, layer_size in enumerate(self.network_layers):
            layers.append(nn.Linear(input_dim, layer_size))
            layers.append(nn.LayerNorm(layer_size))  # LayerNorm for normalization
            if i < len(self.network_layers) - 1:
                layers.append(nn.LeakyReLU(negative_slope=0.01))
            else:
                layers.append(nn.ReLU())  # ReLU for the last hidden layer
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            input_dim = layer_size

        # Shared feature extractor
        self.feature_extractor = nn.Sequential(*layers)

        # Separate heads for predicting next state, reward, and terminal
        self.next_state_head = nn.Linear(self.network_layers[-1], obs_dim)  # Next state prediction
        self.reward_head = nn.Linear(self.network_layers[-1], 1)  # Reward prediction
        self.terminal_head = nn.Sequential(
            nn.Linear(self.network_layers[-1], 1),  # Terminal prediction
            nn.Sigmoid()
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.to(self.device)

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        """
        Forward pass of the transition model.

        Args:
            obs (torch.Tensor): Input observations, shape (batch_size, obs_dim).
            action (torch.Tensor): Input actions, shape (batch_size, action_dim).

        Returns:
            tuple: (next_state, reward, terminal)
                - next_state: Predicted next state, shape (batch_size, obs_dim).
                - reward: Predicted reward, shape (batch_size, 1).
                - terminal: Predicted terminal state probability, shape (batch_size, 1).
        """
        x = torch.cat([obs, action], dim=-1)  # Concatenate observations and actions
        features = self.feature_extractor(x)
        next_state = self.next_state_head(features)
        reward = self.reward_head(features)
        terminal = self.terminal_head(features)
        return next_state, reward, terminal

    def train_batch(self, batch, recon_weight=1.0, reward_weight=1.0, termination_weight=1.0,):
        self.train()

        # Convert batch data to tensors
        true_obs = torch.tensor(batch["state"], dtype=torch.float32, device=self.device)
        next_obs = torch.tensor(batch["next_state"], dtype=torch.float32, device=self.device)
        true_actions = torch.tensor(batch["action"], dtype=torch.float32, device=self.device)
        true_rewards = torch.tensor(batch["reward"], dtype=torch.float32, device=self.device)
        true_terminations = torch.tensor(batch["terminal"], dtype=torch.float32, device=self.device)
        masks = torch.tensor(batch["mask"], dtype=torch.float32, device=self.device)

        batch_size, seq_len, _, = true_obs.size()
        time_weights = torch.linspace(1.0, 0.1, steps=seq_len, device=self.device)

        # Initialize loss accumulators as tensors
        recon_loss = torch.tensor(0.0, device=self.device)
        reward_loss = torch.tensor(0.0, device=self.device)
        termination_loss = torch.tensor(0.0, device=self.device)

        state = true_obs[:, 0]
        for t in range(seq_len):
            next_state, reward, terminal = self.forward(state, true_actions[:, t])

            # Reconstruction loss against next observation
            recon_loss += F.mse_loss(
                next_state * masks[:, t].unsqueeze(-1).unsqueeze(-1), next_obs[:, t] * masks[:, t].unsqueeze(-1).unsqueeze(-1)
            ) * time_weights[t]
            reward_loss += F.mse_loss(
                reward.squeeze() * masks[:, t].unsqueeze(-1), true_rewards[:, t] * masks[:, t].unsqueeze(-1), reduction="mean"
            ) * time_weights[t]
            termination_loss += F.binary_cross_entropy_with_logits(
                terminal.squeeze() * masks[:, t].unsqueeze(-1), true_terminations[:, t] * masks[:, t].unsqueeze(-1), reduction="mean"
            ) * time_weights[t]
            state = next_state

        # Total loss
        total_loss = (
            recon_weight * recon_loss
            + reward_weight * reward_loss
            + termination_weight * termination_loss
        )

        # Backpropagation
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {
            "total_loss": total_loss.item(),
            "recon_loss": recon_loss.item(),
            "reward_loss": reward_loss.item(),
            "termination_loss": termination_loss.item(),
        }


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
    def __init__(self, latent_dim, action_dim, rnn_hidden_dim, embedded_obs_dim):
        super(RSSM, self).__init__()
        self.latent_dim = latent_dim
        self.rnn_hidden_dim = rnn_hidden_dim

        # RNN
        self.rnn = nn.GRUCell(latent_dim, rnn_hidden_dim,)

        # State-Action MLP
        self.state_action_fc = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, latent_dim),
        )

        # Prior and Posterior
        self.prior_fc = nn.Sequential(
            nn.Linear(rnn_hidden_dim, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 2 * latent_dim),
        )
        self.posterior_fc = nn.Sequential(
            nn.Linear(rnn_hidden_dim + embedded_obs_dim, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 2 * latent_dim),
        )

    def reparameterize(self, mean, log_var):
        """
        Reparameterization trick to sample latent variables.
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, latents, actions, rnn_hidden, observations=None):
        """
        Forward pass through RSSM.
        Args:
            latents: (batch_size, seq_len, latent_dim)
            actions: (batch_size, seq_len, action_dim)
            rnn_hidden: (num_layers, batch_size, rnn_hidden_dim)
            observations: (batch_size, seq_len, latent_dim) or None

        Returns:
            prior_mean, prior_log_var, post_mean, post_log_var, rnn_hidden
        """
        x = torch.cat([latents, actions], dim=-1)  # (batch_size, seq_len, latent_dim + action_dim)
        x = self.state_action_fc(x)
        rnn_output = self.rnn(x, rnn_hidden)  # (batch_size, seq_len, rnn_hidden_dim)

        # Compute prior distribution
        prior = self.prior_fc(rnn_output)
        prior_mean, prior_log_var = torch.chunk(prior, 2, dim=-1)
        prior_mean = F.tanh(prior_mean)

        # Compute posterior distribution if observations are provided
        if observations is not None:
            obs_concat = torch.cat([rnn_output, observations], dim=-1)
            posterior = self.posterior_fc(obs_concat)
            post_mean, post_log_var = torch.chunk(posterior, 2, dim=-1)
            post_mean = F.tanh(post_mean)
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
        self.to(device)

    def train_batch(self, batch, start_t=3, recon_weight=1.0, kl_dyn_weight=1.0, kl_rep_weight=0.1, reward_weight=1.0,
                    termination_weight=1.0, kl_min=1.0):
        """
        Train the world model on a single batch.

        Args:
            batch (dict): A batch of training data with keys:
                - "state": Real observations (batch_size, seq_len, channels, height, width)
                - "action": Actions taken (batch_size, seq_len, action_dim)
                - "next_state": Real next observations (batch_size, seq_len, channels, height, width)
                - "reward": Rewards received (batch_size, seq_len, 1)
                - "terminal": Termination flags (batch_size, seq_len, 1)
                - "mask": Masks for valid steps (batch_size, seq_len)
            start_t (int): The starting timestep to compute RSSM prediction-related losses.
            recon_weight (float): Weight for reconstruction loss.
            kl_dyn_weight (float): Weight for dynamic KL loss.
            kl_rep_weight (float): Weight for representation KL loss.
            reward_weight (float): Weight for reward prediction loss.
            termination_weight (float): Weight for termination prediction loss.
            kl_min (float): Minimum threshold for KL loss to be included.

        Returns:
            dict: Losses computed during training.
        """
        # Convert batch data to tensors
        true_obs = torch.tensor(batch["state"], dtype=torch.float32, device=self.device)
        next_obs = torch.tensor(batch["next_state"], dtype=torch.float32, device=self.device)
        true_actions = torch.tensor(batch["action"], dtype=torch.float32, device=self.device)
        true_rewards = torch.tensor(batch["reward"], dtype=torch.float32, device=self.device)
        true_terminations = torch.tensor(batch["terminal"], dtype=torch.float32, device=self.device)
        masks = torch.tensor(batch["mask"], dtype=torch.float32, device=self.device)

        # Initialize RNN hidden state
        rnn_hidden = torch.zeros(
            true_obs.size(0),  # Batch size
            self.rssm.rnn_hidden_dim,
            device=self.device
        )
        sampled_latent = torch.zeros(
            true_obs.size(0),  # Batch size
            self.rssm.latent_dim,
            device=self.device
        )

        batch_size, seq_len, _, _, _ = true_obs.size()

        # Initialize loss accumulators as tensors
        recon_loss = torch.tensor(0.0, device=self.device)
        kl_dyn_loss = torch.tensor(0.0, device=self.device)
        kl_rep_loss = torch.tensor(0.0, device=self.device)
        reward_loss = torch.tensor(0.0, device=self.device)
        termination_loss = torch.tensor(0.0, device=self.device)

        # For returning unadjusted KL losses
        kl_dyn_loss_raw = torch.tensor(0.0, device=self.device)
        kl_rep_loss_raw = torch.tensor(0.0, device=self.device)

        # asset rssm is at least trained once
        assert seq_len > start_t, "seq_len is too short."

        # Define time decay weights
        time_weights = torch.ones(seq_len, device=self.device)
        if seq_len > start_t:
            time_weights[start_t:] = torch.linspace(1.0, 0.1, steps=seq_len - start_t, device=self.device)

        for t in range(seq_len):
            # Encode the current observation
            next_latent_obs = self.encoder(next_obs[:, t])

            # Compute RSSM outputs
            prior_mean, prior_log_var, post_mean, post_log_var, rnn_hidden = self.rssm(
                sampled_latent,
                true_actions[:, t],
                rnn_hidden,
                observations=next_latent_obs,  # if t < start_t else None,
            )

            # Reparameterize to sample latent
            sampled_latent = self.rssm.reparameterize(post_mean.squeeze(1), post_log_var.squeeze(1))

            # Decode reconstructed observation
            combined_latent = torch.cat([sampled_latent, rnn_hidden], dim=1)
            recon_obs = self.decoder(combined_latent)

            # Reconstruction loss against next observation
            recon_loss += self.pixel_loss(recon_obs * masks[:, t].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1), next_obs[:, t] * masks[:, t].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)) * time_weights[t]

            # Compute dynamic KL loss
            kl_dyn = -0.5 * torch.sum(
                (1 + prior_log_var.squeeze(1) - post_mean.pow(2) - prior_log_var.exp()) * masks[:, t].unsqueeze(-1),
                dim=-1,
            ).mean()
            kl_dyn_loss_raw += kl_dyn
            if kl_dyn.item() > kl_min:
                kl_dyn_loss += kl_dyn * time_weights[t]
            else:
                kl_dyn_loss += 0.0

            # Compute representation KL loss
            kl_rep = -0.5 * torch.sum(
                (1 + post_log_var.squeeze(1) - prior_mean.pow(2) - post_log_var.exp()) * masks[:, t].unsqueeze(-1),
                dim=-1,
            ).mean()
            kl_rep_loss_raw += kl_rep
            if kl_rep.item() > kl_min:
                kl_rep_loss += kl_rep * time_weights[t]
            else:
                kl_rep_loss += 0.0

            # Multi-head predictor losses
            predicted_reward, predicted_termination = self.predictor(combined_latent)

            reward_loss += F.mse_loss(
                predicted_reward * masks[:, t].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1), true_rewards[:, t].squeeze() * masks[:, t].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1), reduction="mean"
            ) * time_weights[t]
            termination_loss += F.binary_cross_entropy_with_logits(
                predicted_termination * masks[:, t].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1), true_terminations[:, t].squeeze() * masks[:, t].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1), reduction="mean"
            ) * time_weights[t]

        # Normalize losses by sequence length
        # recon_loss /= seq_len
        # kl_dyn_loss /= seq_len
        # kl_rep_loss /= seq_len
        # kl_dyn_loss_raw /= seq_len
        # kl_rep_loss_raw /= seq_len
        # reward_loss /= seq_len
        # termination_loss /= seq_len

        # Total loss
        total_loss = (
                recon_weight * recon_loss
                + kl_dyn_weight * kl_dyn_loss
                + kl_rep_weight * kl_rep_loss
                + reward_weight * reward_loss
                + termination_weight * termination_loss
        )

        # Backpropagation
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {
            "total_loss": total_loss.item(),
            "recon_loss": recon_loss.item(),
            "kl_dyn_loss": kl_dyn_loss.item(),
            "kl_rep_loss": kl_rep_loss.item(),
            "kl_dyn_loss_raw": kl_dyn_loss_raw.item(),  # Unadjusted dynamic KL loss
            "kl_rep_loss_raw": kl_rep_loss_raw.item(),  # Unadjusted representation KL loss
            "reward_loss": reward_loss.item(),
            "termination_loss": termination_loss.item(),
        }