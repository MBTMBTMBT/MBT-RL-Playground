import os
from collections import defaultdict
import random
from typing import Tuple, Union, List, Any, SupportsFloat

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from torch import optim

from discretizer import Discretizer
from sb3_vec_dataset import GymDataset
from ae import DeterministicAE, ae_total_correlation_uniform_loss, contrastive_loss_v2


# class SparseRewardWrapper(gym.Wrapper):
#     def __init__(self, env, mode="positive"):
#         """
#         A wrapper to transform the reward into a sparse reward.
#
#         Args:
#             env (gym.Env): The environment to wrap.
#             mode (str): The reward mode. Either "positive" or "negative".
#         """
#         super().__init__(env)
#         assert mode in {"positive", "negative"}, "Mode must be 'positive' or 'negative'."
#         self.mode = mode
#
#     def step(self, action):
#         """
#         Perform one step in the environment.
#
#         Args:
#             action: The action to take in the environment.
#
#         Returns:
#             tuple: A tuple of (obs, sparse_reward, done, info).
#         """
#         obs, reward, done, truncated, info = self.env.step(action)
#
#         if self.mode == "positive":
#             sparse_reward = 1.0 if reward > 0 else 0.0
#         elif self.mode == "negative":
#             sparse_reward = 0.0 if reward > 0 else -1.0
#
#         return obs, sparse_reward, done, truncated, info


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
            raise ValueError(
                "This wrapper only supports environments with Box observation spaces."
            )

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
        self.ae_model = DeterministicAE(
            self.observation_space.shape[0],
            num_hidden_values,
            net_arch,
        )
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
            self.dataset.add_samples(
                [
                    {
                        "obs": self.previous_obs,
                        "action": action,
                        "next_obs": next_obs,
                        "reward": reward,
                        "done": done,
                    }
                ]
            )
            if done:
                self.dataset.add_samples(
                    [
                        {
                            "obs": next_obs,
                            "action": action,
                            "next_obs": next_obs,
                            "reward": 0.0,
                            "done": done,
                        }
                    ]
                )
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
        torch.save(
            {
                "model_state_dict": self.ae_model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    def load_model(self, path):
        # Load model and optimizer state dicts
        checkpoint = torch.load(path, map_location=self.device)
        self.ae_model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    def normalize_state(self, state):
        normalized_state = (state - self.state_min) / (self.state_max - self.state_min)
        normalized_state = torch.clamp(normalized_state, 0, 1)
        return normalized_state

    def denormalize_state(self, normalized_state):
        normalized_state = torch.clamp(normalized_state, 0, 1)
        state = normalized_state * (self.state_max - self.state_min) + self.state_min
        return state

    def _train(self):
        dataloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True, drop_last=False
        )
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
                (
                    ae_loss,
                    recon_loss_val,
                    total_correlation,
                    uniform_loss_val,
                ) = ae_total_correlation_uniform_loss(
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


class DiscretizerWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        state_discretizer,
        action_discretizer,
        enable_counting: bool = True,
    ):
        """
        Wrapper to apply state and action discretization with counting functionality.

        :param env: The Gymnasium environment to wrap.
        :param state_discretizer: A Discretizer instance for states.
        :param action_discretizer: A Discretizer instance for actions.
        :param enable_counting: Whether to enable state-action counting.
        """
        super().__init__(env)
        self.state_discretizer = state_discretizer
        self.action_discretizer = action_discretizer
        self.enable_counting = enable_counting
        self.state_action_counts = defaultdict(int) if enable_counting else None

    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
        """
        Reset the environment and discretize the initial state.

        :param kwargs: Additional arguments for the environment's reset method.
        :return: A tuple of the discretized state and additional info.
        """
        state, info = self.env.reset(**kwargs)
        discretized_state, _ = self.state_discretizer.discretize(state)
        return discretized_state, info

    def step(
        self, action: Union[int, List[float]]
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Take a step in the environment using a discretized action.

        :param action: The action to take (discretized bucket index or original action).
        :return: A tuple containing the discretized next state, reward, done, truncated, and info.
        """
        if self.action_discretizer.num_buckets[0] == 0:  # Direct output for action
            action_to_take = action
        elif isinstance(action, int):  # Discrete action
            action_to_take = self.action_discretizer.bucket_midpoints[0][action]
        elif isinstance(action, list):  # Continuous action
            action_to_take, _ = self.action_discretizer.discretize(action)
        else:
            raise ValueError("Invalid action type provided.")

        state, reward, done, truncated, info = self.env.step(action_to_take)
        discretized_state, bucket_indices = self.state_discretizer.discretize(state)

        if self.enable_counting:
            key = {f"state_dim_{i}": val for i, val in enumerate(discretized_state)}
            if isinstance(action, list):
                key.update({f"action_dim_{j}": action[j] for j in range(len(action))})
            else:
                key.update({f"action_dim_0": action})
            # key["action_index"] = action if isinstance(action, int) else None
            self.state_action_counts[tuple(key.items())] += 1

        return discretized_state, reward, done, truncated, info

    def export_counts(self, path: str = None) -> Union[pd.DataFrame, None]:
        """
        Export state-action counts to a DataFrame or save as a CSV file.

        :param path: Optional file path to save the CSV.
        :return: A DataFrame of the counts if no path is provided.
        """
        if not self.enable_counting or not self.state_action_counts:
            return None

        data = []
        for key, count in self.state_action_counts.items():
            row = dict(key)
            row["count"] = count
            data.append(row)

        df = pd.DataFrame(data)

        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            df.to_csv(path, index=False)

        return df

    def import_counts(self, path: str):
        """
        Import state-action counts from a CSV file.

        :param path: Path to the CSV file.
        """
        if not self.enable_counting:
            raise ValueError("Counting is not enabled.")

        df = pd.read_csv(path)
        self.state_action_counts.clear()
        for _, row in df.iterrows():
            key = tuple((col, row[col]) for col in row.index if col != "count")
            self.state_action_counts[key] = row["count"]

    def clear_counts(self):
        """
        Clear all state-action counts.
        """
        if self.enable_counting:
            self.state_action_counts.clear()


# Example Usage
if __name__ == "__main__":
    # Define ranges and number of buckets for each dimension based on CartPole state space
    state_ranges = [
        (-4.8, 4.8),
        (-3.4, 3.4),
        (-0.418, 0.418),
        (-3.4, 3.4),
    ]  # CartPole observation ranges
    action_ranges = [(0, 1)]  # Two discrete actions: 0 and 1

    state_buckets = [5, 5, 5, 5]  # Discretize each state variable into 5 buckets
    action_buckets = [0]  # No discretization for actions

    state_discretizer = Discretizer(state_ranges, state_buckets)
    action_discretizer = Discretizer(action_ranges, action_buckets)

    env = gym.make("CartPole-v1")
    wrapped_env = DiscretizerWrapper(
        env, state_discretizer, action_discretizer, enable_counting=True
    )

    state, info = wrapped_env.reset()
    print("Initial Discretized State:", state)

    for step in range(1000):  # Perform 10 steps in the CartPole environment
        action = random.randint(0, 1)  # Randomly choose between action 0 or 1
        next_state, reward, done, truncated, info = wrapped_env.step(action)
        print(
            f"Step {step + 1}: Action: {action}, Next Discretized State: {next_state}, Reward: {reward}, Done: {done}"
        )
        if done:
            print("Episode finished. Resetting environment.")
            state, info = wrapped_env.reset()

    # Export and re-import counts to test functionality
    export_path = "./state_action_counts.csv"
    wrapped_env.export_counts(export_path)
    print(f"Counts exported to {export_path}")

    wrapped_env.clear_counts()
    print("Counts cleared.")

    wrapped_env.import_counts(export_path)
    print(f"Counts re-imported from {export_path}")

    # Compare exported and imported counts
    df = wrapped_env.export_counts()
    print("Re-imported State-Action Counts:")
    print(df)


class NoMovementTruncateWrapper(gym.Wrapper):
    """
    A wrapper that terminates the environment early if the observation
    does not change significantly for `n` consecutive steps.
    """

    def __init__(self, env, n: int = 10, mse_threshold: float = 0.0):
        """
        Initialize the NoMovementTruncateWrapper.

        Args:
            env (gym.Env): The environment to wrap.
            n (int): Number of consecutive steps to trigger truncation.
            mse_threshold (float): The MSE threshold to trigger truncation.
        """
        super().__init__(env)
        self.n = n
        self.mse_threshold = mse_threshold
        self.obs_buffer = []

    def reset(self, **kwargs):
        """Reset the environment and clear the observation buffer."""
        self.obs_buffer = []
        return self.env.reset(**kwargs)

    def step(self, action):
        """
        Perform a step in the environment and check for no-movement condition.

        Args:
            action: The action to take.

        Returns:
            observation, reward, done, truncated, info: The environment step output.
        """
        obs, reward, done, truncated, info = self.env.step(action)
        self.obs_buffer.append(obs)

        # Only check when the buffer is full
        if len(self.obs_buffer) == self.n:
            # Compute MSE between all consecutive observations in the buffer
            mse = np.sum(
                [
                    (np.array(self.obs_buffer[i]) - np.array(self.obs_buffer[i + 1]))
                    ** 2
                    for i in range(len(self.obs_buffer) - 1)
                ]
            )

            if mse <= self.mse_threshold:
                # Truncate if no significant movement
                truncated = True
                info["truncate_reason"] = "no_movement"
                self.obs_buffer = []  # Clear the buffer to avoid redundant checks
            else:
                # Clear the oldest observation to make room for the next step
                self.obs_buffer.pop(0)

        return obs, reward, done, truncated, info
