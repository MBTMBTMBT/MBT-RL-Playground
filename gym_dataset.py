import torch
from torch.utils.data import Dataset
import numpy as np
from collections import deque
from PIL import Image
from tqdm import tqdm


class GymDataset(Dataset):
    def __init__(self, env, num_samples, frame_size=(64, 128), is_color=True, num_frames=3, repeat=1):
        """
        PyTorch Dataset to sample transitions from a Gymnasium environment.

        Args:
            env (gym.Env): Pre-created Gymnasium environment instance.
            num_samples (int): Number of samples to generate.
            frame_size (tuple): Tuple specifying the height and width of the resized frames.
            is_color (bool): Whether to use color frames (True) or grayscale (False).
            num_frames (int): Number of consecutive frames to stack.
        """
        self.env = env
        self.num_samples = num_samples
        self.frame_height, self.frame_width = frame_size
        self.is_color = is_color
        self.num_frames = num_frames
        self.repeat = repeat
        self.data = self._collect_samples()

    def _collect_samples(self):
        """Collects samples from the environment with a progress bar."""
        self.env.reset()
        samples = []
        frame_queue = deque(maxlen=self.num_frames + 1)  # Fixed-length queue for frames + 1 for next_state

        # Fill initial frames with the first observation
        initial_obs, _ = self.env.reset()
        initial_frame = self._preprocess_frame(initial_obs['image'] if 'image' in initial_obs else initial_obs)
        for _ in range(self.num_frames):
            frame_queue.append(initial_frame)

        # Add a progress bar for the sampling process
        with tqdm(total=self.num_samples, desc="Collecting Samples") as pbar:
            while len(samples) < self.num_samples:  # Continue until we collect the required number of samples
                action = self.env.action_space.sample()
                next_obs, reward, terminated, truncated, info = self.env.step(action)

                # Preprocess frame and add to queue
                frame = self._preprocess_frame(next_obs['image'] if 'image' in next_obs else next_obs)
                frame_queue.append(frame)

                if len(frame_queue) == self.num_frames + 1:  # Only proceed if the queue has enough frames
                    action_encoded = self._encode_action(action)
                    state = np.concatenate(list(frame_queue)[:self.num_frames],
                                           axis=-1)  # Concatenate along channel axis
                    next_state = np.concatenate(list(frame_queue)[1:], axis=-1)  # Concatenate along channel axis
                    sample = {
                        'state': state,
                        'next_state': next_state,
                        'action': action_encoded,
                        'reward': reward,
                        'terminal': terminated
                    }
                    samples.append(sample)
                    pbar.update(1)  # Update the progress bar

                if terminated or truncated:  # Reset the environment if done
                    self.env.reset()
                    # Fill the queue with the last frame to maintain consistency
                    last_frame = frame_queue[-1]
                    for _ in range(self.num_frames):
                        frame_queue.append(last_frame)

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


if __name__ == '__main__':
    # Example usage
    from gymnasium import make
    env = make("CartPole-v1")
    dataset = GymDataset(env=env, num_samples=1000, frame_size=(64, 128), is_color=True, num_frames=3)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    for batch in data_loader:
        print(batch['state'].shape, batch['next_state'].shape, batch['action'].shape, batch['reward'].shape, batch['terminal'].shape)
        break
