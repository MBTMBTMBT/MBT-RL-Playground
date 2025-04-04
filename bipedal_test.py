import gymnasium as gym
import torch
import numpy as np
import imageio
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback

# Set random init_seed for reproducibility
SEED = 0
TOTAL_TIMESTEPS = 1_000_000  # Total training steps
EVAL_INTERVAL = 25_000  # Interval to save GIF
NUM_EVAL_EPISODES = 3  # Number of episodes to evaluate per GIF


# Define custom callback to save GIFs periodically
class GifRecorderCallback(BaseCallback):
    def __init__(
        self,
        eval_env,
        save_path="./",
        eval_freq=EVAL_INTERVAL,
        n_episodes=NUM_EVAL_EPISODES,
        verbose=1,
    ):
        super(GifRecorderCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.save_path = save_path
        self.eval_freq = eval_freq
        self.n_episodes = n_episodes

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            self.save_gif(self.n_calls)
        return True

    def save_gif(self, step):
        """Evaluate the model and save a GIF."""
        images = []
        for _ in range(self.n_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, _ = self.eval_env.step(action)
                done = terminated or truncated
                img = self.eval_env.render()
                images.append(img)

        gif_path = f"{self.save_path}bipedalwalker_step_{step}.gif"
        imageio.mimsave(gif_path, images, duration=0.03)
        if self.verbose:
            print(f"Saved GIF at step {step}: {gif_path}")


# Create training environment
env = make_vec_env("BipedalWalker-v3", n_envs=1, seed=SEED)

# Create evaluation environment (for GIF recording)
eval_env = gym.make("BipedalWalker-v3", render_mode="rgb_array")
eval_env.reset(seed=SEED)

# Define PPO model with appropriate policy
model = PPO(
    "MlpPolicy", env, seed=SEED, verbose=1, tensorboard_log="./ppo_bipedalwalker/"
)

# Train model with callback for GIF saving
callback = GifRecorderCallback(eval_env)
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback, progress_bar=True)

# Save the final model
model.save("ppo_bipedalwalker_hardcore")

# Close environments
env.close()
eval_env.close()
