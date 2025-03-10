import random
from multiprocessing import freeze_support
import gymnasium as gym
from gymnasium.wrappers import GrayScaleObservation, ResizeObservation
from stable_baselines3 import PPO
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

# Configuration
NUM_SEEDS = 5
N_ENVS = 12
TRAIN_STEPS = 2_500_000
EVAL_INTERVAL = 2_500 * N_ENVS
EVAL_EPISODES = 1
NEAR_OPTIMAL_SCORE = 850
MIN_N_STEPS = 1024
GIF_LENGTH = 500
SAVE_PATH = "./car_racing_results"

os.makedirs(SAVE_PATH, exist_ok=True)


# Custom FrameSkip wrapper
class FrameSkip(gym.Wrapper):
    def __init__(self, env, skip=2):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        for _ in range(self._skip):
            obs, reward, term, trunc, info = self.env.step(action)
            total_reward += reward
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

        # Create evaluation environment
        self.eval_env = DummyVecEnv(
            [
                lambda: wrap_carracing(
                    gym.make("CarRacing-v2", continuous=True, render_mode="rgb_array")
                )
            ]
        )
        self.eval_env = VecTransposeImage(self.eval_env)
        self.eval_env = VecFrameStack(self.eval_env, n_stack=2)

    def _on_step(self):
        if self.num_timesteps - self.last_eval_step >= self.eval_interval:
            self.last_eval_step = self.num_timesteps

            mean_reward, _ = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=EVAL_EPISODES,
                deterministic=True,
            )
            self.records.append((self.num_timesteps, mean_reward))

            if self.verbose:
                print(
                    f"[Seed {self.seed}] Step: {self.num_timesteps}, Mean Reward: {mean_reward}"
                )

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
        single_env = wrap_carracing(
            gym.make("CarRacing-v2", continuous=True, render_mode="rgb_array")
        )
        obs, _ = single_env.reset(seed=self.seed)

        for _ in range(GIF_LENGTH):
            # Need to transpose manually because not using VecEnv here
            obs_input = np.transpose(obs, (2, 0, 1))
            action, _ = self.model.predict(obs_input, deterministic=True)
            obs, _, terminated, truncated, _ = single_env.step(action)
            frames.append(single_env.render())
            if terminated or truncated:
                break

        single_env.close()

        gif_path = os.path.join(SAVE_PATH, f"car_racing_seed_{self.seed}.gif")
        imageio.mimsave(gif_path, frames, duration=20, loop=0)


# Environment factory
def make_env(seed: int):
    def _init():
        env = gym.make("CarRacing-v2", continuous=True, render_mode="rgb_array")
        env = wrap_carracing(env)
        env.reset(seed=seed)
        return env

    return _init


# Main training loop
if __name__ == "__main__":
    freeze_support()

    final_results = []
    seeds = list(range(NUM_SEEDS))
    random.shuffle(seeds)

    for seed in seeds:
        print(f"\n===== Training Seed {seed} =====")

        train_env = SubprocVecEnv([make_env(seed) for _ in range(N_ENVS)])
        train_env = VecTransposeImage(train_env)
        train_env = VecFrameStack(train_env, n_stack=2)

        n_steps_value = max(2048 // N_ENVS, MIN_N_STEPS)

        model = PPO(
            "CnnPolicy",
            train_env,
            verbose=1,
            seed=seed,
            batch_size=32,
            n_steps=n_steps_value,
        )

        callback = EvalAndGifCallback(
            seed=seed,
            eval_interval=EVAL_INTERVAL,
            optimal_score=NEAR_OPTIMAL_SCORE,
            verbose=1,
        )

        model.learn(total_timesteps=TRAIN_STEPS, callback=callback, progress_bar=True)

        model_path = os.path.join(SAVE_PATH, f"ppo_carracing_seed_{seed}.zip")
        model.save(model_path)

        df_records = pd.DataFrame(callback.records, columns=["Steps", "MeanReward"])
        df_records_path = os.path.join(SAVE_PATH, f"training_log_seed_{seed}.csv")
        df_records.to_csv(df_records_path, index=False)

        optimal_step = (
            callback.step_reached_optimal
            if callback.step_reached_optimal
            else TRAIN_STEPS
        )
        final_results.append(
            {
                "Seed": seed,
                "OptimalStep": optimal_step,
                "BestScore": callback.best_score,
            }
        )

    df_final_results = pd.DataFrame(final_results)
    df_final_results.to_csv(os.path.join(SAVE_PATH, "summary_results.csv"), index=False)

    print("\n===== Experiment Completed =====")
