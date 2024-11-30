import gymnasium as gym
import numpy as np


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
