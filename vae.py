from typing import List, Any
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from abc import abstractmethod


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


class VanillaVAE(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int, hidden_dims: list = None, input_size=(60, 80)):
        super(VanillaVAE, self).__init__()

        self.latent_dim = latent_dim
        self.input_height, self.input_width = input_size
        self.hidden_dims = hidden_dims if hidden_dims else [32, 64, 128, 256, 512]

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

    def build_encoder(self, in_channels):
        modules = []
        for h_dim in self.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()
                )
            )
            in_channels = h_dim
        return nn.Sequential(*modules)

    def build_decoder(self):
        modules = []
        hidden_dims = self.hidden_dims[::-1]
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1], kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU()
                )
            )
        return nn.Sequential(*modules)

    def encode(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
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

    def loss_function(self, recons, input, mu, log_var, kld_weight=1.0):
        recons_loss = nn.functional.mse_loss(recons, input, reduction="sum")
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()
        return {
            'loss': recons_loss + kld_weight * kld_loss,
            'Reconstruction_Loss': recons_loss,
            'KLD': kld_loss
        }
