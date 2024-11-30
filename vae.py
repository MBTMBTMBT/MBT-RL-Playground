import os

import torch
import torch.nn as nn
import torch.optim as optim


# Define the encoder class
class Encoder(nn.Module):
    def __init__(self, num_input_values: int, latent_dim: int, net_arch: list[int]):
        super(Encoder, self).__init__()
        layers = []
        input_dim = num_input_values
        # Construct hidden layers based on net_arch
        for hidden_units in net_arch:
            layers.append(nn.Linear(input_dim, hidden_units))
            layers.append(nn.LeakyReLU())  # Use leaky ReLU activation
            input_dim = hidden_units
        # Final layers for mean and log variance
        layers.append(nn.Linear(input_dim, latent_dim))
        self.fc2_mu = nn.Linear(input_dim, latent_dim)  # Mean layer
        self.fc2_logvar = nn.Linear(input_dim, latent_dim)  # Log variance layer
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Encoder forward pass
        h = self.net(x)
        mu = self.fc2_mu(h)  # Mean of latent distribution
        logvar = self.fc2_logvar(h)  # Log variance of latent distribution
        return mu, logvar


# Define the decoder class
class Decoder(nn.Module):
    def __init__(self, latent_dim: int, num_input_values: int, net_arch: list[int]):
        super(Decoder, self).__init__()
        layers = []
        input_dim = latent_dim
        # Construct hidden layers based on net_arch
        for hidden_units in reversed(net_arch):
            layers.append(nn.Linear(input_dim, hidden_units))
            layers.append(nn.LeakyReLU())  # Use leaky ReLU activation
            input_dim = hidden_units
        # Final layer to reconstruct the input
        layers.append(nn.Linear(input_dim, num_input_values))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Decoder forward pass
        x_recon = self.net(z)
        return x_recon


# Define the VAE class
class BetaVAE(nn.Module):
    def __init__(self, num_input_values: int, latent_dim: int, net_arch: list[int]):
        super(BetaVAE, self).__init__()
        self.encoder = Encoder(num_input_values, latent_dim, net_arch)
        self.decoder = Decoder(latent_dim, num_input_values, net_arch)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Encoder step
        mu, logvar = self.encoder(x)
        # Reparameterization step
        z = self.reparameterize(mu, logvar)
        # Decoder step
        x_recon = self.decoder(z)
        return x_recon, mu, logvar


# Define the loss function
def beta_vae_loss(recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float) -> tuple[torch.Tensor, float, float]:
    """
    Compute the Beta VAE loss, including reconstruction loss and KL divergence.

    Parameters:
    - recon_x (torch.Tensor): Reconstructed input.
    - x (torch.Tensor): Original input.
    - mu (torch.Tensor): Mean of latent distribution.
    - logvar (torch.Tensor): Log variance of latent distribution.
    - beta (float): Weight for KL divergence.

    Returns:
    - tuple[torch.Tensor, float, float]: Total loss, reconstruction loss, and KL divergence.
    """
    # Reconstruction loss (Mean Squared Error)
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    # KL divergence
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Total loss
    return recon_loss + beta * kl_divergence, recon_loss.item(), kl_divergence.item()


def main() -> None:
    # Hyperparameters
    num_input_values = 784  # Input dimension (e.g., 28x28 images for MNIST)
    latent_dim = 20  # Latent dimension
    net_arch = [400, 200]  # Network architecture for hidden layers
    beta = 4.0  # Beta value
    learning_rate = 1e-3  # Learning rate
    epochs = 10  # Number of epochs

    # Model, optimizer, and data
    model = BetaVAE(num_input_values, latent_dim, net_arch)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    data = torch.randn(64, num_input_values)  # Random data to simulate training (batch_size=64)

    # Training loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        recon_batch, mu, logvar = model(data)

        # Calculate loss
        loss, _, _ = beta_vae_loss(recon_batch, data, mu, logvar, beta)

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Print loss
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")


if __name__ == "__main__":
    main()
