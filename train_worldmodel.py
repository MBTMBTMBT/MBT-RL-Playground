import os

import imageio
import numpy as np
import torch
import gymnasium as gym
from gymnasium import make
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from gym_datasets import ReplayBuffer
from models import RSSM, MultiHeadPredictor, WorldModel, Encoder, Decoder


def generate_visualization_gif(world_model, test_batch, epoch, save_dir, history_len=3):
    """
    Generates and saves a visualization GIF showing actual, predicted, and difference frames.

    Args:
        world_model: The world model instance to generate predictions.
        test_batch (dict): A batch of test data.
        epoch (int): Current epoch number.
        save_dir (str): Directory to save the GIF.
        history_len (int): Number of frames used as history to initialize the model.

    Returns:
        str: Path to the saved GIF.
    """
    with torch.no_grad():
        # Extract test batch data
        true_obs = torch.tensor(test_batch["state"], dtype=torch.float32, device=world_model.device)
        true_actions = torch.tensor(test_batch["action"], dtype=torch.float32, device=world_model.device)

        batch_size, seq_len, channels, height, width = true_obs.size()

        # Initialize RNN hidden state
        rnn_hidden = torch.zeros(world_model.rssm.num_rnn_layers, batch_size, world_model.rssm.rnn_hidden_dim, device=world_model.device)

        # Decode the initial history frames
        recon_history = []
        for t in range(history_len):
            latent = world_model.encoder(true_obs[:, t])
            recon_frame = world_model.decoder(latent)
            recon_history.append(recon_frame.unsqueeze(1))

            # Pass through RSSM to initialize RNN hidden state
            _, _, _, _, rnn_hidden = world_model.rssm(
                latent.unsqueeze(1), true_actions[:, t].unsqueeze(1), rnn_hidden
            )

        # Generate predictions for remaining frames
        recon_obs = []
        current_latent = world_model.encoder(true_obs[:, history_len - 1])

        for t in range(history_len, seq_len):
            # Predict next latent state using RSSM prior
            prior_mean, prior_log_var, _, _, rnn_hidden = world_model.rssm(
                current_latent.unsqueeze(1), true_actions[:, t].unsqueeze(1), rnn_hidden
            )
            # Sample from the prior
            sampled_latent = world_model.rssm.reparameterize(prior_mean.squeeze(1), prior_log_var.squeeze(1))
            # Decode the predicted latent state
            predicted_frame = world_model.decoder(sampled_latent)
            recon_obs.append(predicted_frame.unsqueeze(1))

            # Update current latent for the next step
            current_latent = sampled_latent

        # Combine history and predictions
        recon_obs = torch.cat(recon_history + recon_obs, dim=1).cpu().numpy()
        true_obs = true_obs.cpu().numpy()
        diff_obs = abs(true_obs[:, :recon_obs.shape[1]] - recon_obs)

        # Combine frames for visualization
        combined_frames = []
        for t in range(recon_obs.shape[1]):
            actual_row = np.concatenate([true_obs[i, t].transpose(1, 2, 0) for i in range(batch_size)], axis=1)
            predicted_row = np.concatenate([recon_obs[i, t].transpose(1, 2, 0) for i in range(batch_size)], axis=1)
            difference_row = np.concatenate([diff_obs[i, t].transpose(1, 2, 0) for i in range(batch_size)], axis=1)

            combined_frame = np.concatenate((actual_row, predicted_row, difference_row), axis=0)
            combined_frames.append((combined_frame * 255).astype(np.uint8))

        # Save the GIF
        gif_path = f"{save_dir}/epoch_{epoch + 1}_visualization.gif"
        imageio.mimsave(gif_path, combined_frames, fps=1)

        return gif_path


def add_gif_to_tensorboard(writer, gif_path, tag, global_step):
    """
    Adds a GIF as a video to TensorBoard.

    Args:
        writer (SummaryWriter): TensorBoard SummaryWriter instance.
        gif_path (str): Path to the GIF file.
        tag (str): Tag name for the video.
        global_step (int): Global step value for TensorBoard.

    Returns:
        None
    """
    # Read the GIF file
    gif = imageio.mimread(gif_path)
    # Convert frames to a tensor with shape (T, H, W, C)
    frames = torch.stack([torch.tensor(frame) for frame in gif], dim=0)
    # Permute dimensions to match (T, C, H, W)
    frames = frames.permute(0, 3, 1, 2)
    # Add batch dimension, resulting in shape (1, T, C, H, W)
    frames = frames.unsqueeze(0)
    # Normalize pixel values to [0, 1]
    frames = frames.float() / 255.0
    # Add video to TensorBoard
    writer.add_video(tag, frames, global_step=global_step, fps=1)


if __name__ == '__main__':
    batch_size = 8
    test_batch_size = 8
    buffer_size = 8192
    data_repeat_times = 50
    traj_len_start = 6
    traj_len_end = 32
    frame_size = (60, 80)
    is_color = True
    input_channels = 3
    ae_latent_dim = 32
    encoder_hidden_net_dims = [32, 64, 128, 256]
    rnn_layers=3
    rnn_latent_dim = 64
    lr = 1e-4
    num_epochs = 20
    log_dir = "./experiments/worldmodel/logs"
    save_dir = "./experiments/worldmodel/checkpoints"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = make("Acrobot-v1", render_mode="rgb_array",)
    action_space = env.action_space
    if isinstance(action_space, gym.spaces.Discrete):
        action_dim = action_space.n
    elif isinstance(action_space, gym.spaces.Box):
        action_dim = action_space.shape[0]
    else:
        raise NotImplementedError("Unsupported action space type")
    print("Action dimension:", action_dim)

    dataset = ReplayBuffer(
        env=env,
        buffer_size=buffer_size,
        frame_size=frame_size,
        is_color=is_color,
    )

    encoder = Encoder(
        in_channels=input_channels,
        latent_dim=ae_latent_dim,
        hidden_net_dims=encoder_hidden_net_dims,
        input_size=frame_size,
    )

    decoder = Decoder(
        latent_dim=ae_latent_dim,
        out_channels=input_channels,
        hidden_net_dims=encoder_hidden_net_dims,
        input_size=frame_size,
    )

    rssm = RSSM(
        latent_dim=ae_latent_dim,
        action_dim=action_dim,
        rnn_hidden_dim=rnn_latent_dim,
        num_rnn_layers=rnn_layers,
    )

    predictor = MultiHeadPredictor(
        rnn_hidden_dim=rnn_latent_dim * rnn_layers,
    )

    world_model = WorldModel(
        encoder=encoder,
        decoder=decoder,
        rssm=rssm,
        predictor=predictor,
        lr=lr,
        device=device,
    )

    traj_lens = np.linspace(traj_len_start, traj_len_end, num=num_epochs, dtype=int)

    # step_count = 0
    # Training Loop
    for epoch, traj_len in zip(range(num_epochs), traj_lens):
        dataset.collect_samples(num_samples=buffer_size)
        total_loss = 0
        total_recon_loss = 0
        total_kl_dyn_loss = 0
        total_kl_rep_loss = 0
        total_reward_loss = 0
        total_termination_loss = 0

        progress_bar = tqdm(
            range((buffer_size // batch_size) * data_repeat_times),  #  * int(traj_len_end // traj_len)
            desc=f"Epoch {epoch + 1}/{num_epochs}, traj_len={int(traj_len)}",
        )

        for step in progress_bar:
            batch = dataset.sample(batch_size=batch_size, traj_len=int(traj_len),)
            losses = world_model.train_batch(batch)

            # Update total losses
            total_loss += losses["total_loss"]
            total_recon_loss += losses["recon_loss"]
            total_kl_dyn_loss += losses["kl_dyn_loss"]
            total_kl_rep_loss += losses["kl_rep_loss"]
            total_reward_loss += losses["reward_loss"]
            total_termination_loss += losses["termination_loss"]

            # Write batch losses to TensorBoard
            previous_steps = epoch * (buffer_size // batch_size) * data_repeat_times
            writer.add_scalar("Loss/Total", losses["total_loss"], previous_steps + step)
            writer.add_scalar("Loss/Reconstruction", losses["recon_loss"], previous_steps + step)
            writer.add_scalar("Loss/KL_Dynamic", losses["kl_dyn_loss"], previous_steps + step)
            writer.add_scalar("Loss/KL_Representation", losses["kl_rep_loss"],
                              previous_steps + step)
            writer.add_scalar("Loss/KL_Dynamic_Raw", losses["kl_dyn_loss_raw"],
                              previous_steps + step)
            writer.add_scalar("Loss/KL_Representation_Raw", losses["kl_rep_loss_raw"],
                              previous_steps + step)
            writer.add_scalar("Loss/Reward", losses["reward_loss"], previous_steps + step)
            writer.add_scalar("Loss/Termination", losses["termination_loss"],
                              previous_steps + step)

            # Update progress bar with detailed losses
            progress_bar.set_postfix({
                "Total Loss": f"{losses['total_loss']:.4f}",
                "Recon Loss": f"{losses['recon_loss']:.4f}",
                "KL Dyn Loss": f"{losses['kl_dyn_loss']:.2f}",
                "KL Rep Loss": f"{losses['kl_rep_loss']:.2f}",
                "KL Dyn Raw": f"{losses['kl_dyn_loss_raw']:.2f}",
                "KL Rep Raw": f"{losses['kl_rep_loss_raw']:.2f}",
                "Reward Loss": f"{losses['reward_loss']:.4f}",
                "Termination Loss": f"{losses['termination_loss']:.4f}",
            })

            # step_count += 1

        # Compute average losses for the epoch
        num_batches = buffer_size // batch_size
        avg_total_loss = total_loss / num_batches
        avg_recon_loss = total_recon_loss / num_batches
        avg_kl_dyn_loss = total_kl_dyn_loss / num_batches
        avg_kl_rep_loss = total_kl_rep_loss / num_batches
        avg_reward_loss = total_reward_loss / num_batches
        avg_termination_loss = total_termination_loss / num_batches

        # Write average epoch losses to TensorBoard
        writer.add_scalar("Epoch_Loss/Total", avg_total_loss, epoch)
        writer.add_scalar("Epoch_Loss/Reconstruction", avg_recon_loss, epoch)
        writer.add_scalar("Epoch_Loss/KL_Dynamic", avg_kl_dyn_loss, epoch)
        writer.add_scalar("Epoch_Loss/KL_Representation", avg_kl_rep_loss, epoch)
        writer.add_scalar("Epoch_Loss/Reward", avg_reward_loss, epoch)
        writer.add_scalar("Epoch_Loss/Termination", avg_termination_loss, epoch)

        # Print epoch summary
        print(f"Epoch [{epoch + 1}] Completed.")
        print(f"    Avg Total Loss: {avg_total_loss:.4f}")
        print(f"    Avg Recon Loss: {avg_recon_loss:.4f}")
        print(f"    Avg KL Dyn Loss: {avg_kl_dyn_loss:.4f}")
        print(f"    Avg KL Rep Loss: {avg_kl_rep_loss:.4f}")
        print(f"    Avg Reward Loss: {avg_reward_loss:.4f}")
        print(f"    Avg Termination Loss: {avg_termination_loss:.4f}")

        # Generate test batch and save visualization GIF
        test_batch = dataset.sample(batch_size=test_batch_size, traj_len=int(traj_len),)
        gif_path = generate_visualization_gif(world_model, test_batch, epoch, save_dir)

        # Add GIF as a video to TensorBoard
        add_gif_to_tensorboard(writer, gif_path, "Visualization/GIF", epoch)

        # Save model parameters
        torch.save(world_model.state_dict(), f"{save_dir}/world_model_epoch_{epoch + 1}.pth")
