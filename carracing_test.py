from multiprocessing import freeze_support
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import pandas as pd
import imageio
import os

# Configurations
NUM_SEEDS = 5
N_ENVS = 4
TRAIN_STEPS = 300_000  # Total timesteps across all environments
EVAL_INTERVAL = 10_000 * N_ENVS
EVAL_EPISODES = 50
NEAR_OPTIMAL_SCORE = 900
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

    def _on_step(self):
        if self.num_timesteps % self.eval_interval == 0:
            mean_reward, _ = evaluate_policy(
                self.model, self.eval_env, n_eval_episodes=EVAL_EPISODES, deterministic=True
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

    def save_gif(self):
        frames = []
        obs, _ = self.eval_env.reset(seed=self.seed)
        for _ in range(GIF_LENGTH):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = self.eval_env.step(action)
            frames.append(self.eval_env.render())
            if terminated or truncated:
                break
        gif_path = os.path.join(SAVE_PATH, f"car_racing_seed_{self.seed}.gif")
        imageio.mimsave(gif_path, frames, duration=20, loop=0)


def make_env(seed):
    def _init():
        env = gym.make("CarRacing-v3", continuous=True, render_mode="rgb_array")
        env.reset(seed=seed)
        return env
    return _init


if __name__ == '__main__':
    freeze_support()

    final_results = []

    for seed in range(NUM_SEEDS):
        print(f"\n===== Training Seed {seed} =====")

        train_env = SubprocVecEnv([make_env(seed) for _ in range(N_ENVS)])
        eval_env = Monitor(gym.make("CarRacing-v3", continuous=True, render_mode="rgb_array"))
        eval_env.reset(seed=seed)

        model = PPO("CnnPolicy", train_env, verbose=1, seed=seed)

        callback = EvalAndGifCallback(
            eval_env=eval_env,
            seed=seed,
            eval_interval=EVAL_INTERVAL,
            optimal_score=NEAR_OPTIMAL_SCORE,
            verbose=1
        )

        model.learn(total_timesteps=TRAIN_STEPS, callback=callback, progress_bar=True)

        df_records = pd.DataFrame(callback.records, columns=["Steps", "MeanReward"])
        df_records_path = f"{SAVE_PATH}/training_log_seed_{seed}.csv"
        df_records.to_csv(df_records_path, index=False)

        optimal_step = callback.step_reached_optimal if callback.step_reached_optimal else TRAIN_STEPS
        final_results.append({"Seed": seed, "OptimalStep": optimal_step, "BestScore": callback.best_score})

    df_final_results = pd.DataFrame(final_results)
    df_final_results.to_csv(f"{SAVE_PATH}/summary_results.csv", index=False)

    print("\n===== Experiment Completed =====")


if __name__ == "__main__":
    freeze_support()
