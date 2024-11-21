import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import os
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import numpy as np

# Custom callback to evaluate the model periodically and save GIFs to TensorBoard
class TensorboardGifCallback(BaseCallback):
    def __init__(self, eval_env, log_dir, eval_freq=5000, n_eval_episodes=5):
        super().__init__(verbose=1)
        self.eval_env = eval_env
        self.log_dir = log_dir
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.gif_dir = os.path.join(log_dir, "gifs")
        os.makedirs(self.gif_dir, exist_ok=True)

    def evaluate_policy(self):
        """Evaluate the current policy and generate GIF."""
        frames = []
        total_rewards = []

        for _ in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset()
            terminated = False
            truncated = False
            episode_reward = 0

            while not (terminated or truncated):
                # Render the frame (ensure render_mode="rgb_array" is set)
                frame = self.eval_env.render()
                if frame is not None:
                    frames.append(frame)

                # Predict action and step environment
                action, _ = self.model.predict(obs, deterministic=False)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                episode_reward += reward

            total_rewards.append(episode_reward)

        # Save the generated GIF
        gif_path = os.path.join(self.gif_dir, f"eval_{self.num_timesteps}.gif")
        clip = ImageSequenceClip(frames, fps=30)
        clip.write_gif(gif_path, fps=30, logger=None)

        # Log GIF and evaluation results to TensorBoard
        self.logger.record(f"eval/mean_reward", np.mean(total_rewards))
        self.logger.record(f"eval/gif_path", gif_path)
        return gif_path

    def _on_step(self):
        # Periodically evaluate and save GIF
        if self.n_calls % self.eval_freq == 0:
            gif_path = self.evaluate_policy()
            self.logger.log(f"Saved GIF: {gif_path}")
        return True


# Create the training environment
env = gym.make("Ant-v5", render_mode="rgb_array")  # Ensure render_mode is set
vec_env = DummyVecEnv([lambda: env])  # Vectorized environment

# Set up logging directories for model and TensorBoard
log_dir = "./experiments/anttt/logs"
tensorboard_log = "./experiments/anttt/tensorboard"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(tensorboard_log, exist_ok=True)

# Initialize the PPO model
model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log=tensorboard_log)

# Create an evaluation environment
eval_env = gym.make("Ant-v5", render_mode="rgb_array")  # Ensure render_mode is set

# Initialize the custom callback
callback = TensorboardGifCallback(eval_env, log_dir=log_dir, eval_freq=10_000, n_eval_episodes=5)

# Start training the model
model.learn(total_timesteps=50_000_000, callback=callback, progress_bar=True)

# Save the trained model
model.save(os.path.join(log_dir, "ppo_ant_model"))

# # Load and test the trained model
# model = PPO.load(os.path.join(log_dir, "ppo_ant_model"))
# obs, _ = eval_env.reset()
# for _ in range(1000):
#     eval_env.render()
#     action, _ = model.predict(obs, deterministic=True)
#     obs, _, done, _, _ = eval_env.step(action)
#     if done:
#         obs, _ = eval_env.reset()
# eval_env.close()
