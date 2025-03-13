import gc
import random
from multiprocessing import freeze_support
import gymnasium as gym
import psutil
import torch
from PIL import Image
from gymnasium.wrappers import GrayScaleObservation, ResizeObservation
from matplotlib import pyplot as plt
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    DummyVecEnv,
    VecTransposeImage,
    VecFrameStack,
)
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import pandas as pd
import imageio
import os

from torch import nn
from torchvision import models
from tqdm import tqdm
import custom_envs

# Configuration
NUM_SEEDS = 10
N_ENVS = 24
TRAIN_STEPS = 2_500_000
EVAL_INTERVAL = 5_000 * N_ENVS
EVAL_EPISODES = 120
NEAR_OPTIMAL_SCORE = 9.00
GIF_LENGTH = 500
N_STACK = 5
FRAME_SKIP = 1
RESIZE_SHAPE = 64
SAVE_PATH = "./car_racing_results"

os.makedirs(SAVE_PATH, exist_ok=True)


class ResNet18FeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)

        # Load torchvision resnet18, pretrained=False for simplicity
        resnet = models.resnet18(pretrained=False)

        # Modify first conv layer to match CarRacing input shape (default is 3 channels)
        n_input_channels = observation_space.shape[0]
        resnet.conv1 = nn.Conv2d(
            n_input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Remove the classifier head (fc layer)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])

        # Final feature dimension from resnet18
        self._features_dim = resnet.fc.in_features

    def forward(self, observations):
        # Normalize observation to [0, 1]
        x = observations / 255.0
        x = self.resnet(x)
        # Flatten output
        return torch.flatten(x, 1)


# Custom FrameSkip wrapper
class FrameSkip(gym.Wrapper):
    def __init__(self, env, skip=2):
        super().__init__(env)
        self._skip = skip
        # self.counter = 0

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        for _ in range(self._skip):
            obs, reward, term, trunc, info = self.env.step(action)
            # if self.counter >= 10:
            #     plt.imshow(obs)
            #     plt.show()
            #     self.counter = 0
            total_reward += reward
            # self.counter += 1
            if term or trunc:
                terminated = term
                truncated = trunc
                break
        return obs, total_reward, terminated, truncated, info


# Environment wrapper pipeline
def wrap_carracing(env, frame_skip=2, resize_shape=64):
    env = FrameSkip(env, skip=frame_skip)
    env = GrayScaleObservation(env, keep_dim=True)
    env = ResizeObservation(env, shape=(resize_shape, resize_shape))
    return env


# Environment factory
def make_env(
    seed: int,
    render_mode="rgb_array",
    fixed_start=True,
):
    def _init():
        env = gym.make(
            "CarRacingFixedMap-v2",
            continuous=True,
            render_mode=render_mode,
            map_seed=seed,
            fixed_start=fixed_start,
        )
        env = wrap_carracing(env, frame_skip=FRAME_SKIP, resize_shape=RESIZE_SHAPE)
        # env.reset(seed=seed)
        # env.action_space.seed(seed)
        # env.observation_space.seed(seed)
        return env

    return _init


# Evaluation and GIF callback
class EvalAndGifCallback(BaseCallback):
    def __init__(self, seed, eval_interval, optimal_score, verbose=1):
        super().__init__(verbose)
        self.seed = seed
        self.eval_interval = eval_interval
        self.optimal_score = optimal_score
        self.best_score = -np.inf
        self.step_reached_optimal = None
        self.records = []
        self.last_eval_step = 0

        self.eval_episodes = EVAL_EPISODES
        self.n_eval_envs = N_ENVS

        self.eval_env = SubprocVecEnv([
            make_env(seed, fixed_start=False)
            for _ in range(self.n_eval_envs)
        ])
        self.eval_env = VecTransposeImage(self.eval_env)
        self.eval_env = VecFrameStack(self.eval_env, n_stack=N_STACK)

    def _on_step(self):
        if self.num_timesteps - self.last_eval_step >= self.eval_interval:
            self.last_eval_step = self.num_timesteps

            mean_reward, std_reward = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.eval_episodes,
                deterministic=True,
                render=False,
                warn=False,
            )
            self.records.append((self.num_timesteps, mean_reward, std_reward))

            if self.verbose:
                print(
                    f"[Seed {self.seed}] Step: {self.num_timesteps}, Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}"
                )

            if mean_reward > self.best_score:
                self.best_score = mean_reward
                self.save_gif()

            if mean_reward >= self.optimal_score and self.step_reached_optimal is None:
                self.step_reached_optimal = self.num_timesteps
                print(
                    f"[Seed {self.seed}] Near-optimal reached at step {self.num_timesteps}. Stopping training."
                )
                return False

        return True

    def _on_training_start(self):
        self.records = []
        self.step_reached_optimal = None

    def save_gif(self):
        frames = []

        single_env = DummyVecEnv([make_env(self.seed, fixed_start=True)])
        single_env = VecTransposeImage(single_env)
        single_env = VecFrameStack(single_env, n_stack=N_STACK)

        obs = single_env.reset()

        for _ in range(GIF_LENGTH):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = single_env.step(action)
            frame = single_env.render(mode="rgb_array")
            frames.append(frame)
            if dones[0]:
                break

        single_env.close()

        gif_path = os.path.join(SAVE_PATH, f"car_racing_seed_{self.seed}.gif")

        new_frames = []
        for frame in frames:
            img = Image.fromarray(frame)
            resized_img = img.resize((img.width // 2, img.height // 2))
            new_frames.append(np.array(resized_img))

        imageio.mimsave(gif_path, new_frames, duration=20, loop=0,)


# Progress bar callback
class ProgressBarCallback(BaseCallback):
    """
    Custom progress bar callback using tqdm.
    """

    def __init__(self, total_timesteps, verbose=1):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.pbar = None
        self._last_num_timesteps = 0

    def _on_training_start(self):
        self.pbar = tqdm(
            total=self.total_timesteps,
            desc="Training Progress",
            mininterval=1,
            maxinterval=25,
            smoothing=0.9,
        )
        self._last_num_timesteps = 0

    def _on_step(self):
        delta_steps = self.num_timesteps - self._last_num_timesteps
        self.pbar.update(delta_steps)
        self._last_num_timesteps = self.num_timesteps
        return True

    def _on_training_end(self):
        self.pbar.close()


if __name__ == "__main__":
    freeze_support()

    final_results = []
    seeds = list(range(NUM_SEEDS))
    random.shuffle(seeds)

    for seed in seeds:
        print(f"\n===== Training Seed {seed} =====")

        # Generate track image and save
        env = gym.make(
            "CarRacingFixedMap-v2",
            continuous=True,
            map_seed=seed,
            render_mode=None
        )
        env.reset()
        track_img = env.unwrapped.get_track_image(figsize=(10, 10))
        map_path = os.path.join(SAVE_PATH, f"car_racing_map_seed_{seed}.png")
        plt.imsave(map_path, track_img)
        env.close()
        del env
        gc.collect()

        # Create training environment
        train_env = SubprocVecEnv([
            make_env(
                seed,
                render_mode="rgb_array",
                fixed_start=False,
            )
            for _ in range(N_ENVS)
        ])
        train_env = VecTransposeImage(train_env)
        train_env = VecFrameStack(train_env, n_stack=N_STACK)

        # Create SAC model
        model = SAC(
            "CnnPolicy",
            train_env,
            verbose=1,
            batch_size=128,
            learning_rate=1e-4,
            buffer_size=200_000,
            train_freq=N_ENVS,
            gradient_steps=N_ENVS * 4,
            learning_starts=10_000,
            tau=0.005,
            use_sde=False,
            use_sde_at_warmup=False,
            ent_coef="auto",
            policy_kwargs=dict(
                net_arch=[256, 256],
                # features_extractor_class=ResNet18FeatureExtractor,
            ),
        )

        # Define callbacks
        callback_list = [
            EvalAndGifCallback(
                seed=seed,
                eval_interval=EVAL_INTERVAL,
                optimal_score=NEAR_OPTIMAL_SCORE,
                verbose=1,
            ),
            ProgressBarCallback(total_timesteps=TRAIN_STEPS, verbose=1),
        ]

        # Train the model
        model.learn(total_timesteps=TRAIN_STEPS, callback=callback_list)

        # Save model and logs
        model_path = os.path.join(SAVE_PATH, f"sac_carracing_seed_{seed}.zip")
        model.save(model_path)

        df_records = pd.DataFrame(
            callback_list[0].records,
            columns=["Steps", "MeanReward", "StdReward"]
        )
        df_records_path = os.path.join(SAVE_PATH, f"training_log_seed_{seed}.csv")
        df_records.to_csv(df_records_path, index=False)

        optimal_step = (
            callback_list[0].step_reached_optimal
            if callback_list[0].step_reached_optimal
            else TRAIN_STEPS
        )
        final_results.append({
            "Seed": seed,
            "OptimalStep": optimal_step,
            "BestScore": callback_list[0].best_score,
        })

        # Clean resources
        print(f"\n===== Cleaning up after seed {seed} =====")
        train_env.close()
        del train_env
        del model
        torch.cuda.empty_cache()
        gc.collect()

        print(f"Memory usage after seed {seed}: {psutil.virtual_memory().percent}%")

    # Save summary results
    df_final_results = pd.DataFrame(final_results)
    df_final_results.to_csv(os.path.join(SAVE_PATH, "summary_results.csv"), index=False)

    print("\n===== Experiment Completed =====")
