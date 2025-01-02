import torch
from torch.utils.data import Dataset
import numpy as np
from collections import deque
from PIL import Image
from tqdm import tqdm


class GymDataset(Dataset):
    def __init__(self, env, num_samples, frame_size=(64, 128), is_color=True, repeat=1):
        """
        PyTorch Dataset to sample transitions from a Gymnasium environment.

        Args:
            env (gym.Env): Pre-created Gymnasium environment instance.
            num_samples (int): Number of samples to generate.
            frame_size (tuple): Tuple specifying the height and width of the resized frames.
            num_frames (int): Number of consecutive frames to stack.
        """
        self.env = env
        self.num_samples = num_samples
        self.frame_height, self.frame_width = frame_size
        self.is_color = is_color
        self.repeat = repeat
        self.data = self._collect_samples()

    def _collect_samples(self):
        """Collects samples from the environment with a progress bar."""
        self.env.reset()
        samples = []

        # Add a progress bar for the sampling process
        with tqdm(total=self.num_samples, desc="Collecting Samples") as pbar:
            while len(samples) < self.num_samples:  # Continue until we collect the required number of samples
                action = self.env.action_space.sample()
                next_obs, reward, terminated, truncated, info = self.env.step(action)

                # Preprocess rendered frame
                current_frame = self._preprocess_frame(self.env.render())
                next_frame = self._preprocess_frame(self.env.render()) if not (
                            terminated or truncated) else current_frame

                # Collect sample without stacking frames
                action_encoded = self._encode_action(action)
                sample = {
                    'state': current_frame,  # Single frame as state
                    'next_state': next_frame,  # Single frame as next state
                    'action': action_encoded,
                    'reward': reward,
                    'terminal': terminated
                }
                samples.append(sample)
                pbar.update(1)  # Update the progress bar

                if terminated or truncated:  # Reset the environment if done
                    self.env.reset()

        return samples

    def _preprocess_frame(self, frame):
        """Resizes and converts frame to grayscale if required."""
        frame = Image.fromarray(frame)
        if not self.is_color:
            frame = frame.convert('L')  # Grayscale
        frame = frame.resize((self.frame_width, self.frame_height))
        frame = np.array(frame)
        if self.is_color:
            if frame.ndim == 2:  # Handle grayscale image in a color setting
                frame = np.stack([frame] * 3, axis=-1)  # Convert to RGB-like format
        else:
            frame = frame[..., np.newaxis]  # Add channel dimension for grayscale
        return frame

    def _encode_action(self, action):
        """Encodes the action into a one-hot vector."""
        action_space_size = self.env.action_space.n
        action_encoded = np.zeros(action_space_size, dtype=np.float32)
        action_encoded[action] = 1.0
        return action_encoded

    def __len__(self):
        return len(self.data) * self.repeat

    def __getitem__(self, idx):
        """Retrieve a sample, accounting for repeat functionality."""
        actual_idx = idx % len(self.data)  # Map to original dataset indices
        sample = self.data[actual_idx]
        return {
            'state': torch.tensor(sample['state'], dtype=torch.float32).permute(2, 0, 1),  # Convert to [C, H, W]
            'next_state': torch.tensor(sample['next_state'], dtype=torch.float32).permute(2, 0, 1),  # Convert to [C, H, W]
            'action': torch.tensor(sample['action'], dtype=torch.float32),
            'reward': torch.tensor(sample['reward'], dtype=torch.float32),
            'terminal': torch.tensor(sample['terminal'], dtype=torch.bool)
        }


class ReplayBuffer:
    def __init__(self, env, buffer_size: int, frame_size=(64, 128), is_color=True):
        """
        Replay Buffer to store transitions and sample minibatches.

        Args:
            env (gym.Env): Pre-created Gymnasium environment instance.
            buffer_size (int): Maximum number of transitions to store.
            frame_size (tuple): Tuple specifying the height and width of the resized frames.
            is_color (bool): Whether the frames are in color.
        """
        self.env = env
        self.buffer_size = buffer_size
        self.frame_height, self.frame_width = frame_size
        self.is_color = is_color

        # Initialize buffer
        self.buffer = {
            "state": np.zeros((buffer_size, *frame_size, 3 if is_color else 1), dtype=np.float32),
            "next_state": np.zeros((buffer_size, *frame_size, 3 if is_color else 1), dtype=np.float32),
            "action": np.zeros((buffer_size, self.env.action_space.n), dtype=np.float32),
            "reward": np.zeros(buffer_size, dtype=np.float32),
            "terminal": np.zeros(buffer_size, dtype=np.bool_),
        }
        self.current_index = 0
        self.size = 0

    def add(self, state, next_state, action, reward, terminal):
        """Add a transition to the buffer."""
        idx = self.current_index % self.buffer_size
        self.buffer["state"][idx] = state
        self.buffer["next_state"][idx] = next_state
        self.buffer["action"][idx] = action
        self.buffer["reward"][idx] = reward
        self.buffer["terminal"][idx] = terminal

        self.current_index += 1
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size, traj_len: int):
        """Sample a minibatch of trajectory segments."""
        indices = np.random.randint(0, self.size - traj_len, size=batch_size)
        batch = {key: [] for key in self.buffer}

        masks = np.zeros((batch_size, traj_len), dtype=np.float32)
        for i, start_idx in enumerate(indices):
            for key in self.buffer:
                segment = self.buffer[key][start_idx : start_idx + traj_len]
                if key == "state" or key == "next_state":
                    segment = segment.transpose(0, 3, 1, 2) / 255.0
                batch[key].append(segment)

            # Generate mask for valid data (no padding)
            masks[i, :traj_len] = 1.0

        # Convert list of arrays to a single numpy array
        batch = {key: np.stack(batch[key]) for key in batch}
        batch["mask"] = masks

        return batch

    def collect_samples(self, num_samples):
        """Fill the buffer with data from the environment."""
        self.env.reset()
        count = 0
        with tqdm(total=num_samples, desc="Collecting Samples") as pbar:
            while count < min(num_samples, self.buffer_size):
                action = self.env.action_space.sample()
                next_obs, reward, terminated, truncated, info = self.env.step(action)

                # Preprocess rendered frame
                current_frame = self._preprocess_frame(self.env.render())
                next_frame = self._preprocess_frame(self.env.render()) if not (terminated or truncated) else current_frame

                # Collect and add sample
                action_encoded = self._encode_action(action)
                self.add(
                    state=current_frame,
                    next_state=next_frame,
                    action=action_encoded,
                    reward=reward,
                    terminal=terminated
                )
                pbar.update(1)
                count += 1

                if terminated or truncated:  # Reset the environment if done
                    self.env.reset()

    def _preprocess_frame(self, frame):
        """Resizes and converts frame to grayscale if required."""
        frame = Image.fromarray(frame)
        if not self.is_color:
            frame = frame.convert('L')  # Grayscale
        frame = frame.resize((self.frame_width, self.frame_height))
        frame = np.array(frame)
        if self.is_color:
            if frame.ndim == 2:  # Handle grayscale image in a color setting
                frame = np.stack([frame] * 3, axis=-1)  # Convert to RGB-like format
        else:
            frame = frame[..., np.newaxis]  # Add channel dimension for grayscale
        return frame

    def _encode_action(self, action):
        """Encodes the action into a one-hot vector."""
        action_space_size = self.env.action_space.n
        action_encoded = np.zeros(action_space_size, dtype=np.float32)
        action_encoded[action] = 1.0
        return action_encoded


class ReplayBuffer1D:
    def __init__(self, obs_dim: int, action_dim: int, buffer_size: int):
        """
        Replay Buffer to store and sample vector-based transitions.

        Args:
            obs_dim (int): Dimension of the observation space.
            action_dim (int): Dimension of the action space.
            buffer_size (int): Maximum number of transitions to store.
        """
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.buffer_size = buffer_size

        # Initialize buffer
        self.buffer = {
            "state": np.zeros((buffer_size, obs_dim), dtype=np.float32),
            "next_state": np.zeros((buffer_size, obs_dim), dtype=np.float32),
            "action": np.zeros((buffer_size, action_dim), dtype=np.float32),
            "reward": np.zeros(buffer_size, dtype=np.float32),
            "terminal": np.zeros(buffer_size, dtype=np.bool_),
        }
        self.current_index = 0
        self.size = 0

    def add(self, state: np.ndarray, next_state: np.ndarray, action: np.ndarray, reward: float, terminal: bool):
        """Add a transition to the buffer."""
        idx = self.current_index % self.buffer_size
        self.buffer["state"][idx] = state
        self.buffer["next_state"][idx] = next_state
        self.buffer["action"][idx] = action
        self.buffer["reward"][idx] = reward
        self.buffer["terminal"][idx] = terminal

        self.current_index += 1
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size: int):
        """Sample a minibatch of transitions."""
        indices = np.random.randint(0, self.size, size=batch_size)
        batch = {key: self.buffer[key][indices] for key in self.buffer}
        return batch

    def collect_samples(self, env, num_samples: int):
        """Fill the buffer with data from the environment."""
        env.reset()
        count = 0
        with tqdm(total=num_samples, desc="Collecting Samples") as pbar:
            while count < min(num_samples, self.buffer_size):
                action = env.action_space.sample()
                next_obs, reward, terminated, truncated, info = env.step(action)

                # Add sample to the buffer
                self.add(
                    state=env.state,  # Assuming `env.state` gives current state vector
                    next_state=next_obs,
                    action=self._encode_action(action),
                    reward=reward,
                    terminal=terminated
                )
                pbar.update(1)
                count += 1

                if terminated or truncated:  # Reset the environment if done
                    env.reset()

    def _encode_action(self, action):
        """Encodes the action into a one-hot vector."""
        action_space_size = self.action_dim
        action_encoded = np.zeros(action_space_size, dtype=np.float32)
        action_encoded[action] = 1.0
        return action_encoded


if __name__ == '__main__':
    # Example usage
    from gymnasium import make
    env = make("CartPole-v1")
    dataset = GymDataset(env=env, num_samples=1000, frame_size=(64, 128), is_color=True, num_frames=3)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    for batch in data_loader:
        print(batch['state'].shape, batch['next_state'].shape, batch['action'].shape, batch['reward'].shape, batch['terminal'].shape)
        break
