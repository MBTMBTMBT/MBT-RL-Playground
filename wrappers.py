import os

import gymnasium as gym
import numpy as np
import torch
from torch import optim

from sb3_vec_dataset import GymDataset
from vae import BetaVAE, beta_vae_loss


class AddNoiseDimensionWrapper(gym.Wrapper):
    """
    A wrapper for gymnasium environments that adds an extra observation dimension
    containing uniformly distributed noise in the range [-1, 1].

    Attributes:
        env (gym.Env): The original environment.
    """
    def __init__(self, env):
        super().__init__(env)
        # Modify the observation space to include the new noise dimension
        original_obs_space = env.observation_space
        if not isinstance(original_obs_space, gym.spaces.Box):
            raise ValueError("This wrapper only supports environments with Box observation spaces.")

        low = np.append(original_obs_space.low, -1.0)
        high = np.append(original_obs_space.high, 1.0)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self, **kwargs):
        """
        Resets the environment and adds the noise dimension to the observation.

        Returns:
            np.ndarray: The modified observation with the noise dimension.
        """
        obs, info = self.env.reset(**kwargs)
        noise = np.random.uniform(-1.0, 1.0, size=1).astype(np.float32)
        obs = np.append(obs, noise)
        return obs, info

    def step(self, action):
        """
        Steps the environment and adds the noise dimension to the observation.

        Args:
            action: The action to take in the environment.

        Returns:
            tuple: The modified observation with noise, reward, done flag, and additional info.
        """
        obs, reward, done, truncated, info = self.env.step(action)
        noise = np.random.uniform(-1.0, 1.0, size=1).astype(np.float32)
        obs = np.append(obs, noise)
        return obs, reward, done, truncated, info


class VAEWrapper(gym.Wrapper):
    def __init__(
            self,
            env: gym.Env,
            num_hidden_values: int,
            net_arch: list[int],
            do_training: bool,
            buffer_size: int,
            iterations: int,
            batch_size: int,
            beta: float = 1.0,
            target_variance_scaling: float = 10,
            lr: float = 1e-4,
            device: torch.device = torch.device("cpu"),
    ):
        super().__init__(env)
        self.observation_space = env.observation_space
        self.env = env
        self.dataset = GymDataset(buffer_size)
        self.iterations = iterations
        self.batch_size = batch_size
        self.vae_model = BetaVAE(self.observation_space.shape[0], num_hidden_values, net_arch,)
        self.beta = beta
        self.target_variance_scaling = target_variance_scaling
        self.lr = lr
        self.optimizer = optim.Adam(self.vae_model.parameters(), lr=self.lr)
        self.device = device
        self.vae_model = self.vae_model.to(self.device)

        self.do_training = do_training

    def save_model(self, path):
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Save model and optimizer state dicts
        torch.save({
            'model_state_dict': self.vae_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)

    def load_model(self, path):
        # Load model and optimizer state dicts
        checkpoint = torch.load(path, map_location=self.device)
        self.vae_model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def _train(self):
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True,
                                                 drop_last=False)
        total_losses = []
        recon_losses = []
        kl_divergences = []

        for _ in range(self.iterations):
            for batch in dataloader:
                obss, actions, final_next_obss, rewards, dones = batch
                obss = obss.to(self.device)
                # actions = actions.to(self.device)
                # final_next_obss = final_next_obss.to(self.device)
                # rewards = rewards.to(self.device)
                # dones = dones.to(self.device)

                fake_obss, mu, logvar = self.vae_model(obss)
                loss, recon_loss_val, kl_divergence_val = beta_vae_loss(
                    fake_obss, obss, mu, logvar, self.beta, self.target_variance_scaling
                )

                # Record the loss value for each batch
                total_losses.append(loss.item())
                recon_losses.append(recon_loss_val)
                kl_divergences.append(kl_divergence_val)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # Print all recorded losses
        print("Training Complete. Final Losses:")
        print(f"Total Loss: {sum(total_losses) / len(total_losses):.4f}")
        print(f"Reconstruction Loss: {sum(recon_losses) / len(recon_losses):.4f}")
        print(f"KL Divergence: {sum(kl_divergences) / len(kl_divergences):.4f}")
