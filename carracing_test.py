import random
from multiprocessing import freeze_support
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import pandas as pd
import imageio
import os

NUM_SEEDS = 10
N_ENVS = 12
TRAIN_STEPS = 2_500_000
EVAL_INTERVAL = 2_500 * N_ENVS
EVAL_EPISODES = 1
NEAR_OPTIMAL_SCORE = 850
MIN_N_STEPS = 1024
GIF_LENGTH = 500
SAVE_PATH = "./car_racing_results"
os.makedirs(SAVE_PATH, exist_ok=True)


class EvalAndGifCallback(BaseCallback):
    def __init__(self, eval_env, seed, eval_interval, optimal_score, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.seed = seed
        self.eval_interval = eval_interval
        self.optimal_score = optimal_score
        self.best_score = -np.inf
        self.step_reached_optimal = None
        self.records = []
        self.last_eval_step = 0

    def _on_step(self):
        if self.num_timesteps - self.last_eval_step >= self.eval_interval:
            self.last_eval_step = self.num_timesteps

            mean_reward, _ = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=EVAL_EPISODES,
                deterministic=True
            )
            self.records.append((self.num_timesteps, mean_reward))

            if self.verbose:
                print(f"[Seed {self.seed}] Step: {self.num_timesteps}, Mean Reward: {mean_reward}")

            self.save_gif()

            if mean_reward >= self.optimal_score and self.step_reached_optimal is None:
                self.step_reached_optimal = self.num_timesteps
                print(f"[Seed {self.seed}] Near-optimal reached at step {self.num_timesteps}. Stopping training.")
                return False
        return True

    def _on_training_start(self):
        self.records = []
        self.step_reached_optimal = None

    def save_gif(self):
        frames = []
        single_env = gym.make("CarRacing-v3", continuous=True, render_mode="rgb_array")
        obs, _ = single_env.reset(seed=self.seed)
        for _ in range(GIF_LENGTH):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = single_env.step(action)
            frames.append(single_env.render())
            if terminated or truncated:
                break
        single_env.close()
        gif_path = os.path.join(SAVE_PATH, f"car_racing_seed_{self.seed}.gif")
        imageio.mimsave(gif_path, frames, duration=20, loop=0)


def make_env(seed: int):
    def _init():
        env = gym.make("CarRacing-v3", continuous=True, render_mode="rgb_array")
        env.reset(seed=seed)
        return env
    return _init


if __name__ == '__main__':
    freeze_support()

    final_results = []

    seeds = [s for s in range(NUM_SEEDS)]
    random.shuffle(seeds)
    for seed in seeds:
        print(f"\n===== Training Seed {seed} =====")

        train_env = SubprocVecEnv([make_env(seed) for _ in range(N_ENVS)])

        eval_env = gym.make("CarRacing-v3", continuous=True, render_mode="rgb_array")
        eval_env.reset(seed=seed)

        n_steps_value = max(2048 // N_ENVS, MIN_N_STEPS)

        model = PPO(
            "CnnPolicy",
            train_env,
            verbose=1,
            seed=seed,
            batch_size=32,
            n_steps=n_steps_value
        )

        callback = EvalAndGifCallback(
            eval_env=eval_env,
            seed=seed,
            eval_interval=EVAL_INTERVAL,
            optimal_score=NEAR_OPTIMAL_SCORE,
            verbose=1
        )

        model.learn(total_timesteps=TRAIN_STEPS, callback=callback, progress_bar=True)

        model_path = os.path.join(SAVE_PATH, f"ppo_carracing_seed_{seed}.zip")
        model.save(model_path)

        df_records = pd.DataFrame(callback.records, columns=["Steps", "MeanReward"])
        df_records_path = os.path.join(SAVE_PATH, f"training_log_seed_{seed}.csv")
        df_records.to_csv(df_records_path, index=False)

        optimal_step = callback.step_reached_optimal if callback.step_reached_optimal else TRAIN_STEPS
        final_results.append({
            "Seed": seed,
            "OptimalStep": optimal_step,
            "BestScore": callback.best_score
        })

    df_final_results = pd.DataFrame(final_results)
    df_final_results.to_csv(os.path.join(SAVE_PATH, "summary_results.csv"), index=False)

    print("\n===== Experiment Completed =====")
