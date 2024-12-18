import os

import torch
import gymnasium as gym
from gymnasium import make
from tqdm import tqdm

from gym_datasets import ReplayBuffer
from models import RSSM, MultiHeadPredictor, WorldModel, Encoder, Decoder

if __name__ == '__main__':
    batch_size = 8
    buffer_size = 1024
    traj_len = 16
    frame_size = (60, 80)
    is_color = True
    input_channels = 3
    ae_latent_dim = 64
    encoder_hidden_net_dims = [128, 256,]
    rnn_latent_dim = 256
    rnn_layers=1
    lr = 1e-4
    num_epochs = 10
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

    dataset.collect_samples(num_samples=buffer_size)
    batch = dataset.sample(batch_size=batch_size)

    # Training Loop
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(range(buffer_size // batch_size), desc=f"Epoch {epoch + 1}/{num_epochs}")

        for step in progress_bar:
            batch = dataset.sample(batch_size=batch_size)
            loss = world_model.train_batch(batch,)
            total_loss += loss.item()

            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / (buffer_size // batch_size)
        print(f"Epoch [{epoch + 1}] Completed. Avg Loss: {avg_loss:.4f}")
