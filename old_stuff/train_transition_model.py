import os

import imageio
import numpy as np
import torch
import gymnasium as gym
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from old_stuff.gym_datasets import ReplayBuffer1D
from models import (
    SimpleTransitionModel,
)


def generate_visualization_gif(
    env,
    normalization_vec,
    transition_model,
    test_batch,
    epoch,
    save_dir,
    resize_shape=(80, 60),
):
    """
    Generate a GIF visualizing the prediction results of a transition model.

    Args:
        env: Gymnasium environment instance.
        transition_model: The trained transition model.
        test_batch: Dictionary containing test data with keys "state", "action", "next_state".
        epoch: Current training epoch (for naming the GIF).
        save_dir: Directory to save the GIF.
        resize_shape: Tuple (height, width) to resize each frame, or None to keep original size.
    """
    with torch.no_grad():
        # Extract test batch data
        true_obs = torch.tensor(
            test_batch["state"], dtype=torch.float32, device=transition_model.device
        )
        true_actions = torch.tensor(
            test_batch["action"], dtype=torch.float32, device=transition_model.device
        )
        normalization_vec = torch.tensor(
            normalization_vec, dtype=torch.float32, device=transition_model.device
        )
        true_obs = true_obs / normalization_vec

        batch_size, seq_len, vec_dims = true_obs.size()

        # Generate predictions and corresponding visualizations
        recon_obs = []
        state = true_obs[:, 0]  # Initialize with the first frame

        for t in range(seq_len):
            # Predict next state using the transition model
            next_state, _, _ = transition_model(state, true_actions[:, t])
            next_state = (next_state * normalization_vec).squeeze()

            # Set the environment state and render the predicted frame
            for i in range(batch_size):
                env.unwrapped.state = (
                    state[i].cpu().numpy()
                )  # Set the environment's internal state
                frame = env.render()  # Render the current environment frame
                if resize_shape:
                    frame = np.array(
                        Image.fromarray(frame).resize(resize_shape)
                    )  # Resize frame
                if t == 0:
                    recon_obs.append([frame])  # Initialize list for each batch element
                else:
                    recon_obs[i].append(frame)

            state = next_state

        # Convert true observations to rendered images
        true_frames = []
        for i in range(batch_size):
            env.unwrapped.state = true_obs[i, 0].cpu().numpy()
            frame = env.render()
            if resize_shape:
                frame = np.array(
                    Image.fromarray(frame).resize(resize_shape)
                )  # Resize frame
            true_frames.append([frame])  # Render initial state

            for t in range(1, seq_len):
                env.unwrapped.state = true_obs[i, t].cpu().numpy()
                frame = env.render()
                if resize_shape:
                    frame = np.array(
                        Image.fromarray(frame).resize(resize_shape)
                    )  # Resize frame
                true_frames[i].append(frame)

        # Combine true, predicted, and difference images
        combined_frames = []
        for t in range(seq_len):
            actual_row = np.concatenate(
                [true_frames[i][t] for i in range(batch_size)], axis=1
            )
            predicted_row = np.concatenate(
                [recon_obs[i][t] for i in range(batch_size)], axis=1
            )
            diff_row = abs(
                actual_row.astype(np.float32) - predicted_row.astype(np.float32)
            ).astype(np.uint8)

            combined_frame = np.concatenate(
                (actual_row, predicted_row, diff_row), axis=0
            )
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


if __name__ == "__main__":
    batch_size = 32
    test_batch_size = 8
    buffer_size = 16384
    data_repeat_times = 100
    traj_len_start = 2
    traj_len_end = 64
    obs_dim = 3
    lr = 1e-4
    normalization_vec = np.array(
        [
            1.01,
            1.01,
            8.01,
        ]
    ).reshape(1, 1, -1)
    reward_scale_down = 1e2
    num_epochs = 62
    log_dir = "./experiments/trans/logs"
    save_dir = "./experiments/trans/checkpoints"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make(
        "Pendulum-v1",
        render_mode="rgb_array",
    )
    action_space = env.action_space
    if isinstance(action_space, gym.spaces.Discrete):
        action_dim = action_space.n
    elif isinstance(action_space, gym.spaces.Box):
        action_dim = action_space.shape[0]
    else:
        raise NotImplementedError("Unsupported action space type")
    print("Action dimension:", action_dim)

    dataset = ReplayBuffer1D(
        obs_dim=obs_dim,
        action_dim=action_dim,
        buffer_size=buffer_size,
    )

    transition_model = SimpleTransitionModel(
        obs_dim=obs_dim,
        action_dim=action_dim,
        network_layers=[
            512,
            512,
            512,
        ],
        dropout=0.0,
        lr=lr,
        device=device,
    )

    traj_lens = np.linspace(traj_len_start, traj_len_end, num=num_epochs, dtype=int)

    # step_count = 0
    # Training Loop
    for epoch, traj_len in zip(range(num_epochs), traj_lens):
        dataset.collect_samples(env=env, num_samples=buffer_size)
        total_loss = 0
        total_recon_loss = 0
        total_reward_loss = 0
        total_termination_loss = 0

        progress_bar = tqdm(
            range(
                (buffer_size // batch_size) * data_repeat_times
            ),  #  * int(traj_len_end // traj_len)
            desc=f"Epoch {epoch + 1}/{num_epochs}, traj_len={int(traj_len)}",
        )

        for step in progress_bar:
            batch = dataset.sample(
                batch_size=batch_size,
                traj_len=int(traj_len),
            )
            batch["state"] = batch["state"] / normalization_vec
            batch["next_state"] = batch["next_state"] / normalization_vec
            batch["reward"] = batch["reward"] / reward_scale_down
            losses = transition_model.train_batch(
                batch,
            )  # * rnn_latent_dim)

            # Update total losses
            total_loss += losses["total_loss"]
            total_recon_loss += losses["recon_loss"]
            total_reward_loss += losses["reward_loss"]
            total_termination_loss += losses["termination_loss"]

            # Write batch losses to TensorBoard
            previous_steps = epoch * (buffer_size // batch_size) * data_repeat_times
            writer.add_scalar("Loss/Total", losses["total_loss"], previous_steps + step)
            writer.add_scalar(
                "Loss/Reconstruction", losses["recon_loss"], previous_steps + step
            )
            writer.add_scalar(
                "Loss/Reward", losses["reward_loss"], previous_steps + step
            )
            writer.add_scalar(
                "Loss/Termination", losses["termination_loss"], previous_steps + step
            )

            # Update progress bar with detailed losses
            progress_bar.set_postfix(
                {
                    "Total": f"{losses['total_loss']:.4f}",
                    "Recon": f"{losses['recon_loss']:.4f}",
                    "Reward": f"{losses['reward_loss']:.4f}",
                    "Termination": f"{losses['termination_loss']:.4f}",
                }
            )

            # step_count += 1

        # Compute average losses for the epoch
        num_batches = buffer_size // batch_size
        avg_total_loss = total_loss / num_batches
        avg_recon_loss = total_recon_loss / num_batches
        avg_reward_loss = total_reward_loss / num_batches
        avg_termination_loss = total_termination_loss / num_batches

        # Write average epoch losses to TensorBoard
        writer.add_scalar("Epoch_Loss/Total", avg_total_loss, epoch)
        writer.add_scalar("Epoch_Loss/Reconstruction", avg_recon_loss, epoch)
        writer.add_scalar("Epoch_Loss/Reward", avg_reward_loss, epoch)
        writer.add_scalar("Epoch_Loss/Termination", avg_termination_loss, epoch)

        # Print epoch summary
        print(f"Epoch [{epoch + 1}] Completed.")
        print(f"    Avg Total Loss: {avg_total_loss:.4f}")
        print(f"    Avg Recon Loss: {avg_recon_loss:.4f}")
        print(f"    Avg Reward Loss: {avg_reward_loss:.4f}")
        print(f"    Avg Termination Loss: {avg_termination_loss:.4f}")

        # Generate test batch and save visualization GIF
        test_batch = dataset.sample(
            batch_size=test_batch_size,
            traj_len=int(traj_len),
        )
        gif_path = generate_visualization_gif(
            env, normalization_vec, transition_model, test_batch, epoch, save_dir
        )

        # Add GIF as a video to TensorBoard
        add_gif_to_tensorboard(writer, gif_path, "Visualization/GIF", epoch)

        # Save model parameters
        torch.save(
            transition_model.state_dict(),
            f"{save_dir}/world_model_epoch_{epoch + 1}.pth",
        )
