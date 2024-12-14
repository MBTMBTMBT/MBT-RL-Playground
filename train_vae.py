import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os

# Import your dataset and VAE model
from gym_dataset import GymDataset
from gymnasium import make
from vae import VanillaVAE

def train_vae(model, dataloader, epochs, device, lr, log_dir, save_dir, is_color):
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0
        pbar = tqdm(enumerate(dataloader), desc=f"Epoch {epoch + 1}/{epochs}", total=len(dataloader))
        for batch_idx, batch in pbar:
            inputs = batch['state'].to(device) / 255.  # Normalize to [0, 1]
            optimizer.zero_grad()

            # Forward pass
            recons, _, mu, log_var = model(inputs)
            loss_dict = model.loss_function(recons, inputs, mu, log_var, kld_weight=1.0 / dataloader.batch_size)
            loss = loss_dict['loss']

            # Backward pass
            loss.backward()
            optimizer.step()

            # Accumulate loss
            epoch_loss += loss.item()

            # Update progress bar with detailed metrics
            pbar.set_postfix({
                "Batch Loss": f"{loss.item():.4f}",
                "Reconstruction Loss": f"{loss_dict['Reconstruction_Loss']:.4f}",
                "KLD Loss": f"{loss_dict['KLD']:.4f}"
            })

            # Log to TensorBoard
            writer.add_scalar('Loss/total', loss.item(), epoch * len(dataloader) + batch_idx)
            writer.add_scalar('Loss/reconstruction', loss_dict['Reconstruction_Loss'],
                              epoch * len(dataloader) + batch_idx)
            writer.add_scalar('Loss/KLD', loss_dict['KLD'], epoch * len(dataloader) + batch_idx)

        # Epoch summary
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{epochs}], Average Loss: {avg_loss:.4f}")

        # Save the model after each epoch
        torch.save(model.state_dict(), os.path.join(save_dir, f"vae_epoch_{epoch + 1}.pth"))

        # Visualize reconstructions
        visualize_reconstruction(model, dataloader, epoch, save_dir, is_color)

    writer.close()

def visualize_reconstruction(model, dataloader, epoch, save_dir, is_color):
    model.eval()
    with torch.no_grad():
        batch = next(iter(dataloader))
        inputs = batch['state'].to(next(model.parameters()).device) / 255.0  # Normalize to [0, 1]
        recons, _, _, _ = model(inputs)

        # Select the first 3 channels for RGB visualization
        inputs = inputs[:, :3, :, :]
        recons = recons[:, :3, :, :]

        # Convert grayscale to RGB if needed
        if not is_color:
            inputs = inputs.repeat(1, 3, 1, 1)
            recons = recons.repeat(1, 3, 1, 1)

        # Concatenate inputs and reconstructions for visualization
        comparison = torch.cat([inputs[:8], recons[:8]])  # Show first 8 samples

        # Create a grid and save the image
        grid = make_grid(comparison, nrow=8, normalize=True, scale_each=True)
        save_image(grid, os.path.join(save_dir, f"reconstruction_epoch_{epoch+1}.png"))
    model.train()


if __name__ == '__main__':
    # Setup
    env = make("CartPole-v1", render_mode="rgb_array")
    dataset = GymDataset(env=env, num_samples=16384, frame_size=(60, 80), is_color=True, num_frames=3, repeat=10)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = VanillaVAE(in_channels=9, latent_dim=128, input_size=(60, 80)).to(device)  # in_channels = num_frames * 3 for RGB images

    # Train the model
    train_vae(
        model=vae,
        dataloader=dataloader,
        epochs=20,
        device=device,
        lr=1e-3,
        log_dir="./experiments/vae/logs",
        save_dir="./experiments/vae/checkpoints",
        is_color=True
    )
