import os

import torch
import gymnasium as gym
from gymnasium import make
from tqdm import tqdm

from gym_datasets import ReplayBuffer
from models import RSSM, MultiHeadPredictor, WorldModel, Encoder, Decoder

if __name__ == '__main__':
    batch_size = 8
    buffer_size = 16384
    traj_len = 48
    frame_size = (60, 80)
    is_color = True
    input_channels = 3
    ae_latent_dim = 64
    encoder_hidden_net_dims = [32, 64, 128,]
    rnn_latent_dim = 256
    rnn_layers=1
    lr = 1e-4
    num_epochs = 50
    log_dir = "./experiments/worldmodel/logs"
    save_dir = "./experiments/worldmodel/checkpoints"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)


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
        traj_len=traj_len,
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
        rnn_layers=rnn_layers,
    )

    predictor = MultiHeadPredictor(
        rnn_hidden_dim=rnn_latent_dim,
    )

    world_model = WorldModel(
        encoder=encoder,
        decoder=decoder,
        rssm=rssm,
        predictor=predictor,
        lr=lr,
        device=device,
    )

    # Training Loop
    for epoch in range(num_epochs):
        dataset.collect_samples(num_samples=buffer_size)
        total_loss = 0
        total_recon_loss = 0
        total_kl_dyn_loss = 0
        total_kl_rep_loss = 0
        total_reward_loss = 0
        total_termination_loss = 0

        progress_bar = tqdm(range(buffer_size // batch_size), desc=f"Epoch {epoch + 1}/{num_epochs}")

        for step in progress_bar:
            batch = dataset.sample(batch_size=batch_size)
            losses = world_model.train_batch(batch)

            # Update total losses
            total_loss += losses["total_loss"]
            total_recon_loss += losses["recon_loss"]
            total_kl_dyn_loss += losses["kl_dyn_loss"]
            total_kl_rep_loss += losses["kl_rep_loss"]
            total_reward_loss += losses["reward_loss"]
            total_termination_loss += losses["termination_loss"]

            # Update progress bar with detailed losses
            progress_bar.set_postfix({
                "Total Loss": f"{losses['total_loss']:.4f}",
                "Recon Loss": f"{losses['recon_loss']:.4f}",
                "KL Dyn Loss": f"{losses['kl_dyn_loss']:.4f}",
                "KL Rep Loss": f"{losses['kl_rep_loss']:.4f}",
                "Reward Loss": f"{losses['reward_loss']:.4f}",
                "Termination Loss": f"{losses['termination_loss']:.4f}",
            })

        # Compute average losses for the epoch
        num_batches = buffer_size // batch_size
        avg_total_loss = total_loss / num_batches
        avg_recon_loss = total_recon_loss / num_batches
        avg_kl_dyn_loss = total_kl_dyn_loss / num_batches
        avg_kl_rep_loss = total_kl_rep_loss / num_batches
        avg_reward_loss = total_reward_loss / num_batches
        avg_termination_loss = total_termination_loss / num_batches

        # Print epoch summary
        print(f"Epoch [{epoch + 1}] Completed.")
        print(f"    Avg Total Loss: {avg_total_loss:.4f}")
        print(f"    Avg Recon Loss: {avg_recon_loss:.4f}")
        print(f"    Avg KL Dyn Loss: {avg_kl_dyn_loss:.4f}")
        print(f"    Avg KL Rep Loss: {avg_kl_rep_loss:.4f}")
        print(f"    Avg Reward Loss: {avg_reward_loss:.4f}")
        print(f"    Avg Termination Loss: {avg_termination_loss:.4f}")
