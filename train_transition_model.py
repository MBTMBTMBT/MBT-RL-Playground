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


def generate_visualization_gif(env, transition_model, test_batch, epoch, save_dir):
    """
    Generate a GIF visualizing the prediction results of a transition model.

    Args:
        env: Gymnasium environment instance.
        transition_model: The trained transition model.
        test_batch: Dictionary containing test data with keys "state", "action", "next_state".
        epoch: Current training epoch (for naming the GIF).
        save_dir: Directory to save the GIF.
    """
    with torch.no_grad():
        # Extract test batch data
        true_obs = torch.tensor(test_batch["state"], dtype=torch.float32, device=transition_model.device)
        true_actions = torch.tensor(test_batch["action"], dtype=torch.float32, device=transition_model.device)
        next_obs = torch.tensor(test_batch["next_state"], dtype=torch.float32, device=transition_model.device)

        batch_size, seq_len, vec_dims = true_obs.size()

        # Generate predictions and corresponding visualizations
        recon_obs = []
        state = true_obs[:, 0]  # Initialize with the first frame

        for t in range(seq_len):
            # Predict next state using the transition model
            next_state, _, _ = transition_model(state, true_actions[:, t])
            state = next_state

            # Set the environment state and render the predicted frame
            for i in range(batch_size):
                env.unwrapped.state = state[i].cpu().numpy()  # Set the environment's internal state
                frame = env.render()  # Render the current environment frame
                if t == 0:
                    recon_obs.append([frame])  # Initialize list for each batch element
                else:
                    recon_obs[i].append(frame)

        # Convert true observations to rendered images
        true_frames = []
        for i in range(batch_size):
            env.unwrapped.state = true_obs[i, 0].cpu().numpy()
            true_frames.append([env.render()])  # Render initial state

            for t in range(1, seq_len):
                env.unwrapped.state = true_obs[i, t].cpu().numpy()
                true_frames[i].append(env.render())

        # Combine true, predicted, and difference images
        combined_frames = []
        for t in range(seq_len):
            actual_row = np.concatenate([true_frames[i][t] for i in range(batch_size)], axis=1)
            predicted_row = np.concatenate([recon_obs[i][t] for i in range(batch_size)], axis=1)
            diff_row = abs(actual_row.astype(np.float32) - predicted_row.astype(np.float32)).astype(np.uint8)

            combined_frame = np.concatenate((actual_row, predicted_row, diff_row), axis=0)
            combined_frames.append(combined_frame)

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
    data_repeat_times = 25
    traj_len_start = 32
    traj_len_end = 32
    frame_size = (60, 80)
    is_color = True
    input_channels = 3
    ae_latent_dim = 32
    latent_dim = 64
    rnn_latent_dim = 256 - 64
    encoder_hidden_net_dims = [16, 32, 64,]
    lr = 1e-4
    num_epochs = 25
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
        latent_dim=latent_dim + rnn_latent_dim,
        out_channels=input_channels,
        hidden_net_dims=encoder_hidden_net_dims,
        input_size=frame_size,
    )

    rssm = RSSM(
        latent_dim=latent_dim,
        action_dim=action_dim,
        rnn_hidden_dim=rnn_latent_dim,
        embedded_obs_dim=ae_latent_dim,
    )

    predictor = MultiHeadPredictor(
        rnn_hidden_dim=latent_dim + rnn_latent_dim,  # * rnn_layers,
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
            losses = world_model.train_batch(batch, kl_min=1.0)  # * rnn_latent_dim)

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
                "Total": f"{losses['total_loss']:.4f}",
                "Recon": f"{losses['recon_loss']:.4f}",
                "KLDyn": f"{losses['kl_dyn_loss']:.2f}",
                "KLRep": f"{losses['kl_rep_loss']:.2f}",
                "KLDynRaw": f"{losses['kl_dyn_loss_raw']:.2f}",
                "KLRepRaw": f"{losses['kl_rep_loss_raw']:.2f}",
                "Reward": f"{losses['reward_loss']:.4f}",
                "Termination": f"{losses['termination_loss']:.4f}",
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
        test_batch = dataset.sample(batch_size=test_batch_size, traj_len=int(traj_len_end),)
        gif_path = generate_visualization_gif(world_model, test_batch, epoch, save_dir)

        # Add GIF as a video to TensorBoard
        add_gif_to_tensorboard(writer, gif_path, "Visualization/GIF", epoch)

        # Save model parameters
        torch.save(world_model.state_dict(), f"{save_dir}/world_model_epoch_{epoch + 1}.pth")
