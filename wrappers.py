import os

import gymnasium as gym
import numpy as np
import torch
from torch import optim

from sb3_vec_dataset import GymDataset
from ae import DeterministicAE, ae_total_correlation_uniform_loss, contrastive_loss_v2


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


class AEWrapper(gym.Wrapper):
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
            gamma: float = 1.0,
            lr: float = 1e-4,
            state_min: np.ndarray = None,
            state_max: np.ndarray = None,
            device: torch.device = torch.device("cpu"),
    ):
        super().__init__(env)
        self.observation_space = env.observation_space
        self.env = env
        self.dataset = GymDataset(buffer_size)
        self.iterations = iterations
        self.batch_size = batch_size
        self.ae_model = DeterministicAE(self.observation_space.shape[0], num_hidden_values, net_arch, )
        self.beta = beta
        self.gamma = gamma
        self.lr = lr
        self.optimizer = optim.Adam(self.ae_model.parameters(), lr=self.lr)
        self.device = device
        self.ae_model = self.ae_model.to(self.device)
        if state_min is not None and state_max is not None:
            self.do_normalize = True
            self.state_min = torch.tensor(state_min, dtype=torch.float32)
            self.state_max = torch.tensor(state_max, dtype=torch.float32)
        else:
            self.do_normalize = False
        self.do_training = do_training
        self.step_counter = 0

        self.previous_obs = None
        self.total_loss = 0.0
        self.reconstruction_loss = 0.0
        self.kl_divergence = 0.0
        self.total_uniform_loss = 0.0
        self.contrastive_loss = 0.0

    def reset(self, **kwargs):
        """
        Resets the environment and adds the noise dimension to the observation.

        Returns:
            np.ndarray: The modified observation with the noise dimension.
        """
        obs, info = self.env.reset(**kwargs)
        self.previous_obs = obs
        return obs, info

    def step(self, action):
        """
        Steps the environment and adds the noise dimension to the observation.

        Args:
            action: The action to take in the environment.

        Returns:
            tuple: The modified observation with noise, reward, done flag, and additional info.
        """
        next_obs, reward, done, truncated, info = self.env.step(action)
        self.step_counter += 1
        if self.do_training:
            self.dataset.add_samples([
                {
                    'obs': self.previous_obs,
                    'action': action,
                    'next_obs': next_obs,
                    'reward': reward,
                    'done': done
                }
            ])
            if done:
                self.dataset.add_samples([
                    {
                        'obs': next_obs,
                        'action': action,
                        'next_obs': next_obs,
                        'reward': 0.0,
                        'done': done
                    }
                ])
            if self.dataset.full:
                # print(f"Step {self.step_counter}, start training...")
                self._train()
                self.dataset.clear()
        self.previous_obs = next_obs
        next_obs = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0)
        if self.do_normalize:
            next_obs = self.normalize_state(next_obs)
        next_obs = next_obs.to(self.device)
        with torch.no_grad():
            next_obs = self.ae_model.encoder(next_obs)
            # if self.do_normalize:
            #     _next_obs = self.denormalize_state(self.ae_model.decoder(next_obs)).cpu().squeeze().numpy()
            #     next_obs = next_obs.cpu().squeeze().numpy()
            #     print("Original Obs: ", self.previous_obs, "Reconstructed Obs: ", _next_obs)
            # else:
            #     next_obs = next_obs.cpu().squeeze().numpy()
            next_obs = next_obs.cpu().squeeze().numpy()
        return next_obs, reward, done, truncated, info

    def save_model(self, path):
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Save model and optimizer state dicts
        torch.save({
            'model_state_dict': self.ae_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)

    def load_model(self, path):
        # Load model and optimizer state dicts
        checkpoint = torch.load(path, map_location=self.device)
        self.ae_model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def normalize_state(self, state):
        normalized_state = (state - self.state_min) / (self.state_max - self.state_min)
        normalized_state = torch.clamp(normalized_state, 0, 1)
        return normalized_state

    def denormalize_state(self, normalized_state):
        normalized_state = torch.clamp(normalized_state, 0, 1)
        state = normalized_state * (self.state_max - self.state_min) + self.state_min
        return state

    def _train(self):
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True,
                                                 drop_last=False)
        total_losses = []
        recon_losses = []
        total_correlations = []
        total_uniform_losses = []
        contrastive_losses = []

        for _ in range(self.iterations):
            for batch in dataloader:
                obss, actions, next_obss, rewards, dones = batch
                if self.do_normalize:
                    obss = self.normalize_state(obss)
                obss = obss.to(self.device)
                actions = actions.to(self.device)
                next_obss = next_obss.to(self.device)
                rewards = rewards.to(self.device)
                dones = dones.to(self.device)

                # fake_obss, z = self.ae_model(obss)
                # loss, recon_loss_val, total_correlation, uniform_loss_val = ae_total_correlation_uniform_loss(
                #     fake_obss, obss, z, self.beta, self.gamma,
                # )

                # Forward pass for obss and next_obss
                fake_obss, z_obss = self.ae_model(obss)
                _, z_next_obss = self.ae_model(next_obss)

                # Calculate AE loss
                ae_loss, recon_loss_val, total_correlation, uniform_loss_val = ae_total_correlation_uniform_loss(
                    fake_obss, obss, z_obss, self.beta, self.gamma
                )

                # Contrastive loss
                negative_indices = torch.randperm(z_obss.size(0)).tolist()
                z_negative = z_obss[negative_indices]
                contrastive_loss = contrastive_loss_v2(z_obss, z_next_obss, z_negative)

                # Total loss
                total_loss = ae_loss + 0.1 * contrastive_loss

                # Record the loss value for each batch
                total_losses.append(total_loss.item())
                recon_losses.append(recon_loss_val)
                total_correlations.append(total_correlation)
                total_uniform_losses.append(uniform_loss_val)
                contrastive_losses.append(contrastive_loss.item())

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

        # Print all recorded losses
        # print("Training Complete. Final Losses:")
        # print(f"Total Loss: {sum(total_losses) / len(total_losses):.4f}")
        # print(f"Reconstruction Loss: {sum(recon_losses) / len(recon_losses):.4f}")
        # print(f"KL Divergence: {sum(total_correlations) / len(total_correlations):.4f}")
        self.total_loss = sum(total_losses) / len(total_losses)
        self.reconstruction_loss = sum(recon_losses) / len(recon_losses)
        self.kl_divergence = sum(total_correlations) / len(total_correlations)
        self.total_uniform_loss = sum(total_uniform_losses) / len(total_uniform_losses)
        self.contrastive_loss = sum(contrastive_losses) / len(contrastive_losses)
