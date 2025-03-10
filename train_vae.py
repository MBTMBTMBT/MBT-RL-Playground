import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os

# import torchvision.datasets as datasets

from gym_datasets import GymDataset
from gymnasium import make
from vae import VAE


def train_vae(
    model,
    dataloader,
    epochs,
    device,
    lr,
    log_dir,
    save_dir,
    is_color,
    beta_start=0.0,
    beta_end=1.0,
    kld_threshold=10.0,
):
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()

    beta = beta_start
    beta_increment = (beta_end - beta_start) / epochs

    for epoch in range(epochs):
        epoch_loss = 0
        pbar = tqdm(
            enumerate(dataloader),
            desc=f"Epoch {epoch + 1}/{epochs}",
            total=len(dataloader),
        )
        for batch_idx, batch in pbar:
            inputs = batch["state"].to(device) / 255.0  # Normalize to [0, 1]
            optimizer.zero_grad()

            # Forward pass
            recons, _, mu, log_var = model(inputs)
            loss_dict = model.loss_function(
                recons,
                inputs,
                mu,
                log_var,
                kld_weight=1.0 / dataloader.batch_size * beta,
                kld_threshold=kld_threshold,
            )
            loss = loss_dict["loss"]

            # Backward pass
            loss.backward()
            optimizer.step()

            # Accumulate loss
            epoch_loss += loss.item()

            # Update progress bar with detailed metrics
            pbar.set_postfix(
                {
                    "Batch Loss": f"{loss.item():.4f}",
                    "Reconstruction Loss": f"{loss_dict['Reconstruction_Loss']:.4f}",
                    "KLD Loss": f"{loss_dict['KLD']:.4f}",
                }
            )

            # Log to TensorBoard
            writer.add_scalar(
                "Loss/total", loss.item(), epoch * len(dataloader) + batch_idx
            )
            writer.add_scalar(
                "Loss/reconstruction",
                loss_dict["Reconstruction_Loss"],
                epoch * len(dataloader) + batch_idx,
            )
            writer.add_scalar(
                "Loss/KLD", loss_dict["KLD"], epoch * len(dataloader) + batch_idx
            )
            writer.add_scalar("beta", beta, epoch * len(dataloader) + batch_idx)

        # Epoch summary
        avg_loss = epoch_loss / len(dataloader)
        print(
            f"Epoch [{epoch + 1}/{epochs}], Beta: {beta}, Average Loss: {avg_loss:.4f}."
        )

        # Save the model after each epoch
        torch.save(
            model.state_dict(), os.path.join(save_dir, f"vae_epoch_{epoch + 1}.pth")
        )

        # Visualize reconstructions
        visualize_reconstruction(model, dataloader, epoch, save_dir, is_color)

        beta += beta_increment

    writer.close()


def visualize_reconstruction(model, dataloader, epoch, save_dir, is_color):
    """
    Visualizes reconstruction by comparing input and reconstructed images.

    Args:
        model (nn.Module): The trained VAE model.
        dataloader (DataLoader): DataLoader for the dataset.
        epoch (int): Current epoch number.
        save_dir (str): Directory to save the visualization.
        is_color (bool): Whether the images are in color (True) or grayscale (False).
    """
    model.eval()
    with torch.no_grad():
        batch = next(iter(dataloader))
        inputs = (
            batch["state"].to(next(model.parameters()).device) / 255.0
        )  # Normalize to [0, 1]
        recons, _, _, _ = model(inputs)

        # Convert grayscale to RGB if needed for visualization
        if not is_color:
            inputs = inputs.repeat(1, 3, 1, 1)
            recons = recons.repeat(1, 3, 1, 1)

        # Select a few samples (e.g., first 8) for visualization
        num_samples = min(8, inputs.shape[0])  # Ensure we don't exceed batch size
        inputs = inputs[:num_samples]
        recons = recons[:num_samples]

        # Concatenate inputs and reconstructions for comparison
        comparison = torch.cat(
            [inputs, recons], dim=0
        )  # Stack inputs and reconstructions vertically

        # Create a grid and save the visualization
        grid = make_grid(comparison, nrow=num_samples, normalize=True, scale_each=True)
        save_image(grid, os.path.join(save_dir, f"reconstruction_epoch_{epoch+1}.png"))

    model.train()


if __name__ == "__main__":
    # Setup
    env = make(
        "Acrobot-v1",
        render_mode="rgb_array",
    )
    dataset = GymDataset(
        env=env, num_samples=16384, frame_size=(96, 128), is_color=True, repeat=10
    )
    # mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # dataloader = DataLoader(mnist_trainset, batch_size=8, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = VAE(
        in_channels=3,
        latent_dim=64,
        input_size=(96, 128),
        hidden_dims=[64, 128],  # ema_factor=0.01,
    ).to(device)

    # vae = VAE(
    #     in_channels=1, latent_dim=64, input_size=(80, 80), hidden_dims=[256, 512, 1024], ema_factor=0.01
    # ).to(device)

    # Train the model
    train_vae(
        model=vae,
        dataloader=dataloader,
        epochs=20,
        device=device,
        lr=1e-4,
        log_dir="./experiments/vae/logs",
        save_dir="./experiments/vae/checkpoints",
        is_color=True,
        beta_start=1.0,
        beta_end=1.0,
        kld_threshold=1.0,
    )
