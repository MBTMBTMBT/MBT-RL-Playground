from torch.utils.data import Dataset
import torch
import gymnasium as gym
from tqdm import tqdm
import numpy as np


class GymDataset(Dataset):
    def __init__(self, data_size: int):
        """
        Args:
            data_size: The maximum number of samples the dataset can store.
        """
        self.data_size = data_size
        self.data: list[dict] = []
        self.current_size = 0
        self.full = False  # Flag to indicate if the dataset is full

    def add_samples(self, samples: list):
        """
        Add new samples to the dataset.

        Args:
            samples: A list of dictionaries, where each dictionary contains
                - 'obs': Tensor or ndarray, the observation.
                - 'action': Tensor or int, the action.
                - 'next_obs': Tensor or ndarray, the next observation.
                - 'reward': Tensor or float, the reward.
                - 'done': Tensor or bool, whether the episode ended.
        """
        for sample in samples:
            self.data.append(
                {
                    "obs": torch.tensor(sample["obs"], dtype=torch.float32),
                    "action": torch.tensor(sample["action"], dtype=torch.int64),
                    "next_obs": torch.tensor(sample["next_obs"], dtype=torch.float32),
                    "reward": torch.tensor(sample["reward"], dtype=torch.float32),
                    "done": torch.tensor(sample["done"], dtype=torch.bool),
                }
            )
            self.current_size += 1
            if self.current_size >= self.data_size:
                self.full = True
            else:
                self.full = False

    def clear(self):
        """
        Clear all data in the dataset and reset counters.
        """
        self.data.clear()
        self.current_size = 0
        self.full = False

    def __len__(self):
        """
        Return the number of stored samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieve a sample by index.

        Args:
            idx: The index of the desired sample.

        Returns:
            A tuple containing (obs, action, next_obs, reward, done).
        """
        item = self.data[idx]
        return (
            item["obs"],
            item["action"],
            item["next_obs"],
            item["reward"],
            item["done"],
        )


if __name__ == "__main__":
    # Create a single Gymnasium environment
    env_id = "CartPole-v1"
    env = gym.make(env_id)
    env.reset(seed=0)

    # Simulation parameters
    data_size = 100
    movement_augmentation = 1
    dataset = GymDataset(data_size=data_size)
    total_samples = data_size
    augmented = 0

    # Start sampling
    obs, _ = env.reset()
    with tqdm(total=total_samples, desc="Sampling Data", unit="sample") as pbar:
        while len(dataset.data) < total_samples:
            # Generate an action
            action = env.action_space.sample()
            next_obs, reward, done, _, info = env.step(action)

            final_next_obs = np.copy(next_obs)  # Copy next_obs to avoid overwriting
            samples = []

            # Ensure termination states are correctly handled
            # if done:
            #     final_next_obs = info["terminal_observation"]

            # Prepare data
            repeat = 0
            if movement_augmentation > 0:
                repeat = (
                    0
                    if np.allclose(obs, final_next_obs, rtol=1e-5, atol=1e-8)
                    else movement_augmentation
                )

            for _ in range(1 + repeat):
                samples.append(
                    {
                        "obs": obs,
                        "action": action,
                        "next_obs": final_next_obs,
                        "reward": reward,
                        "done": done,
                    }
                )
            augmented += repeat

            # Add the batch of samples to the dataset
            dataset.add_samples(samples)
            pbar.update(len(samples))

            # Update the observation for the next step
            if done:
                obs, _ = env.reset()
            else:
                obs = next_obs

    # Print summary
    print(f"{total_samples} samples collected, including {augmented} augmented.")
    print(f"Example sample from dataset: {dataset.data[0]}")
