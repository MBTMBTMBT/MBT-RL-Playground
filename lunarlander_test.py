import gc
import os
import random

import psutil
import gymnasium as gym
import numpy as np
import pandas as pd
import torch

import imageio
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

from sbx import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
import custom_envs


# --------------- Configuration ---------------
NUM_GRAVITY_SETTINGS = 17  # number of different gravity values
N_REPEAT = 8               # number of repeated runs per gravity setting
N_ENVS = 20                # number of parallel environments
TRAIN_STEPS = 750_000      # total timesteps per training run
EVAL_INTERVAL = 1_250 * N_ENVS
EVAL_EPISODES = 256
NEAR_OPTIMAL_SCORE = 250

GIF_LENGTH = 500
SAVE_PATH = "./lunar_lander_results"
os.makedirs(SAVE_PATH, exist_ok=True)


# --------------- Environment Factory ---------------
def make_lander_env(gravity, render_mode=None, deterministic_init=False, seed=None):
    """
    Factory function for CustomLunarLander-v3 environment.
    """
    def _init():
        env = gym.make(
            "CustomLunarLander-v3",
            continuous=True,
            render_mode=render_mode,
            gravity=gravity,
            use_deterministic_initial_states=deterministic_init,
            custom_seed=seed if deterministic_init else None,
        )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return _init


# --------------- GIF and Evaluation Callback ---------------
class EvalAndGifCallback(BaseCallback):
    """
    Evaluation callback that runs periodically during training.
    Saves evaluation records and GIFs when improved.
    """
    def __init__(self, gravity, eval_interval, optimal_score, verbose=1):
        super().__init__(verbose)
        self.gravity = gravity
        self.eval_interval = eval_interval
        self.optimal_score = optimal_score

        self.best_mean_reward = -np.inf
        self.step_reached_optimal = None
        self.records = []
        self.last_eval_step = 0

        self.eval_episodes = EVAL_EPISODES
        self.n_eval_envs = N_ENVS

        self.eval_env = SubprocVecEnv([
            make_lander_env(
                gravity=self.gravity,
                render_mode=None,
                deterministic_init=True,
                seed=i,
            )
            for i in range(self.n_eval_envs)
        ])

    def _on_step(self) -> bool:
        """
        Evaluate the agent and save records if performance improves.
        """
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
                print(f"[EvalCallback] Gravity {self.gravity:.1f} | Steps {self.num_timesteps} | "
                      f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")

            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.save_gif()

            if mean_reward >= self.optimal_score and self.step_reached_optimal is None:
                self.step_reached_optimal = self.num_timesteps

        return True

    def _on_training_start(self):
        self.records = []
        self.step_reached_optimal = None

    def _on_training_end(self):
        df = pd.DataFrame(self.records, columns=["Timesteps", "MeanReward", "StdReward"])
        eval_log_path = os.path.join(SAVE_PATH, f"eval_log_gravity_{self.gravity:.1f}.csv")
        df.to_csv(eval_log_path, index=False)
        self.eval_env.close()

    def save_gif(self):
        """
        Record an evaluation episode as GIF if best reward is improved.
        """
        frames = []
        single_env = DummyVecEnv([
            make_lander_env(
                gravity=self.gravity,
                render_mode="rgb_array",
                deterministic_init=True
            )
        ])

        obs = single_env.reset()

        for _ in range(GIF_LENGTH):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = single_env.step(action)

            frame = single_env.render(mode="rgb_array")
            frames.append(frame)

            if dones[0]:
                break

        single_env.close()

        gif_path = os.path.join(SAVE_PATH, f"lander_gravity_{self.gravity:.1f}.gif")

        new_frames = []
        for frame in frames:
            img = Image.fromarray(frame)
            resized_img = img.resize((img.width // 2, img.height // 2))
            new_frames.append(np.array(resized_img))

        imageio.mimsave(gif_path, new_frames, duration=20, loop=0)


# --------------- Progress Bar Callback ---------------
class ProgressBarCallback(BaseCallback):
    """
    Progress bar using tqdm for visualization of training progress.
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
            mininterval=5,
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


# --------------- Plotting Utilities ---------------
def plot_results(gravity_results, save_dir):
    """
    Plot performance curves for each gravity and save the figures.
    """
    plt.figure(figsize=(12, 8))

    # Plot curves for each gravity
    for gravity, results in gravity_results.items():
        steps = results['Timesteps']
        means = results['MeanReward']
        stds = results['StdReward']

        plt.plot(steps, means, label=f"Gravity {gravity:.1f}")
        plt.fill_between(steps, np.array(means) - np.array(stds), np.array(means) + np.array(stds), alpha=0.2)

    plt.xlabel("Timesteps")
    plt.ylabel("Mean Reward")
    plt.title("Evaluation Reward over Timesteps (per Gravity)")
    plt.legend()
    plt.grid()

    plot_path = os.path.join(save_dir, "reward_curves_all_gravities.png")
    plt.savefig(plot_path)
    plt.close()


# --------------- Main Training Loop ---------------
if __name__ == "__main__":
    gravities = np.linspace(-5.0, -11.9999, NUM_GRAVITY_SETTINGS)
    random.shuffle(gravities)

    summary_results = []  # List of dicts for all runs
    gravity_results = {}  # Dict for storing reward curves

    for gravity in gravities:
        print(f"\n===== Gravity = {gravity:.1f} =====")

        repeat_results = []  # Results for N_REPEAT runs under same gravity
        reward_curves = []   # Store reward curves for plotting

        for repeat in range(N_REPEAT):
            print(f"\n--- Repeat {repeat + 1}/{N_REPEAT} for Gravity = {gravity:.1f} ---")

            # Create parallel training environments
            train_env = SubprocVecEnv([
                make_lander_env(gravity=gravity, render_mode=None, deterministic_init=False)
                for _ in range(N_ENVS)
            ])

            # SAC Model definition
            model = SAC(
                "MlpPolicy",
                train_env,
                verbose=0,
                learning_rate=2e-4,
                buffer_size=75_000,
                learning_starts=5_000,
                batch_size=256,
                tau=0.005,
                train_freq=N_ENVS,
                gradient_steps=N_ENVS * 8,
                ent_coef="auto",
                policy_kwargs=dict(net_arch=[256, 256]),
            )

            # Callbacks
            eval_callback = EvalAndGifCallback(
                gravity=gravity,
                eval_interval=EVAL_INTERVAL,
                optimal_score=NEAR_OPTIMAL_SCORE,
                verbose=1
            )
            progress_callback = ProgressBarCallback(total_timesteps=TRAIN_STEPS)

            # Train the model
            model.learn(
                total_timesteps=TRAIN_STEPS,
                callback=[eval_callback, progress_callback]
            )

            # Save final model
            model_path = os.path.join(SAVE_PATH, f"sac_lander_gravity_{gravity:.1f}_repeat_{repeat + 1}.zip")
            model.save(model_path)

            # Append repeat results
            repeat_results.append({
                "Gravity": gravity,
                "Repeat": repeat + 1,
                "OptimalStep": eval_callback.step_reached_optimal or TRAIN_STEPS,
                "BestScore": eval_callback.best_mean_reward
            })

            # Append reward curve data for this repeat
            df_repeat = pd.DataFrame(eval_callback.records, columns=["Timesteps", "MeanReward", "StdReward"])
            reward_curves.append(df_repeat)

            # Cleanup
            print(f"\n--- Cleanup after Repeat {repeat + 1} for Gravity {gravity:.1f} ---")
            train_env.close()
            del model
            gc.collect()
            torch.cuda.empty_cache()

        # Aggregate stats (mean/std across repeats)
        best_scores = [res["BestScore"] for res in repeat_results]
        optimal_steps = [res["OptimalStep"] for res in repeat_results]

        summary_results.append({
            "Gravity": gravity,
            "BestScoreMean": np.mean(best_scores),
            "BestScoreStd": np.std(best_scores),
            "OptimalStepMean": np.mean(optimal_steps),
            "OptimalStepStd": np.std(optimal_steps),
        })

        # Save the reward curves for plotting
        mean_rewards = np.mean([df["MeanReward"] for df in reward_curves], axis=0)
        std_rewards = np.std([df["MeanReward"] for df in reward_curves], axis=0)
        timesteps = reward_curves[0]["Timesteps"]

        gravity_results[gravity] = {
            "Timesteps": timesteps,
            "MeanReward": mean_rewards,
            "StdReward": std_rewards
        }

    # Save summary results
    df_summary = pd.DataFrame(summary_results)
    df_summary_path = os.path.join(SAVE_PATH, "summary_results_mean_std.csv")
    df_summary.to_csv(df_summary_path, index=False)

    print("\n===== All Training Completed =====")
    print(df_summary)

    # Plot all reward curves
    plot_results(gravity_results, SAVE_PATH)
