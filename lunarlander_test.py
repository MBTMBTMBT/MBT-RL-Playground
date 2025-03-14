import gc
import os
import random

import cv2
import gymnasium as gym
import numpy as np
import pandas as pd
import torch

import imageio
from tqdm import tqdm
import matplotlib.pyplot as plt

from sbx import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
import custom_envs


# --------------- Configuration ---------------
NUM_DENSITY_SETTINGS = 4  # number of different lander densities
N_REPEAT = 8
N_ENVS = 20
TRAIN_STEPS = 1_500_000
EVAL_INTERVAL = 1_250 * N_ENVS
NUM_INIT_STATES = 128
EVAL_EPISODES = NUM_INIT_STATES * 2
NEAR_OPTIMAL_SCORE = 275

GIF_LENGTH = 500
SAVE_PATH = "./lunar_lander_density_results"
os.makedirs(SAVE_PATH, exist_ok=True)


# --------------- Environment Factory ---------------
def make_lander_env(
    lander_density,
    render_mode=None,
    deterministic_init=False,
    seed=None,
    number_of_initial_states=NUM_INIT_STATES,
):
    def _init():
        env = gym.make(
            "CustomLunarLander-v3",
            continuous=True,
            render_mode=render_mode,
            gravity=-10.0,  # fixed gravity
            lander_density=lander_density,
            use_deterministic_initial_states=deterministic_init,
            custom_seed=seed if deterministic_init else None,
            number_of_initial_states=number_of_initial_states,
        )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return _init


# --------------- GIF and Evaluation Callback ---------------
class EvalAndGifCallback(BaseCallback):
    def __init__(self, lander_density, repeat, eval_interval, optimal_score, verbose=1):
        super().__init__(verbose)
        self.lander_density = lander_density
        self.repeat = repeat
        self.eval_interval = eval_interval
        self.optimal_score = optimal_score

        self.best_mean_reward = -np.inf
        self.step_reached_optimal = None
        self.records = []
        self.last_eval_step = 0

        self.eval_episodes = EVAL_EPISODES
        self.n_eval_envs = N_ENVS

        self.eval_env = SubprocVecEnv(
            [
                make_lander_env(
                    lander_density=self.lander_density,
                    render_mode=None,
                    deterministic_init=True,
                    seed=i,
                )
                for i in range(self.n_eval_envs)
            ]
        )

    def _on_step(self) -> bool:
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
                    f"[EvalCallback] Lander Density {self.lander_density:.1f} | Repeat {self.repeat} | "
                    f"Steps {self.num_timesteps} | Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}"
                )

            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                if TRAIN_STEPS - EVAL_INTERVAL * 2 < self.num_timesteps:
                    self.save_gif()

            if mean_reward >= self.optimal_score and self.step_reached_optimal is None:
                self.step_reached_optimal = self.num_timesteps

        return True

    def _on_training_start(self):
        self.records = []
        self.step_reached_optimal = None

    def _on_training_end(self):
        df = pd.DataFrame(
            self.records, columns=["Timesteps", "MeanReward", "StdReward"]
        )
        repeat_log_path = os.path.join(
            SAVE_PATH,
            f"eval_log_density_{self.lander_density:.1f}_repeat_{self.repeat}.csv",
        )
        df.to_csv(repeat_log_path, index=False)
        self.eval_env.close()

    def save_gif(self):
        frames = []
        initial_state_count = 16

        single_env = DummyVecEnv(
            [
                make_lander_env(
                    lander_density=self.lander_density,
                    render_mode="rgb_array",
                    deterministic_init=True,
                    seed=0,
                    number_of_initial_states=16,
                )
            ]
        )

        for idx in range(initial_state_count):
            obs = single_env.reset()
            episode_frames = []
            while True:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, rewards, dones, infos = single_env.step(action)

                frame = single_env.render(mode="rgb_array")
                episode_frames.append(frame)

                if dones[0]:
                    break

            separator_frame = np.zeros_like(episode_frames[0])
            for _ in range(5):
                frames.append(separator_frame)

            frames.extend(episode_frames)

        single_env.close()

        new_frames = []
        for frame in frames:
            resized = cv2.resize(
                frame,
                (frame.shape[1] // 2, frame.shape[0] // 2),
                interpolation=cv2.INTER_AREA,
            )
            new_frames.append(resized)

        gif_path = os.path.join(
            SAVE_PATH,
            f"lander_density_{self.lander_density:.1f}_repeat_{self.repeat}_all_initial_states.gif",
        )

        imageio.mimsave(gif_path, new_frames, duration=10, loop=0)
        print(f"[GIF Saved] {gif_path}")


# --------------- Progress Bar Callback ---------------
class ProgressBarCallback(BaseCallback):
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
def plot_results(density_results, save_dir):
    plt.figure(figsize=(12, 8))

    for density, results in density_results.items():
        steps = results["Timesteps"]
        means = results["MeanReward"]
        stds = results["StdReward"]

        plt.plot(steps, means, label=f"Lander Density {density:.1f}")
        plt.fill_between(
            steps,
            np.array(means) - np.array(stds),
            np.array(means) + np.array(stds),
            alpha=0.2,
        )

        plt_single = plt.figure(figsize=(10, 6))
        plt_single_ax = plt_single.add_subplot(111)
        plt_single_ax.plot(steps, means, label=f"Lander Density {density:.1f}")
        plt_single_ax.fill_between(steps, means - stds, means + stds, alpha=0.2)
        plt_single_ax.set_xlabel("Timesteps")
        plt_single_ax.set_ylabel("Mean Reward")
        plt_single_ax.set_title(f"Reward Curve for Lander Density {density:.1f}")
        plt_single_ax.legend()
        plt_single_ax.grid()

        plot_single_path = os.path.join(
            save_dir, f"reward_curve_density_{density:.1f}.png"
        )
        plt_single.savefig(plot_single_path)
        plt.close(plt_single)

    plt.xlabel("Timesteps")
    plt.ylabel("Mean Reward")
    plt.title("Evaluation Reward over Timesteps (All Lander Densities)")
    plt.legend()
    plt.grid()

    plot_path = os.path.join(save_dir, "reward_curves_all_densities.png")
    plt.savefig(plot_path)
    plt.close()


def plot_optimal_step_bar_chart(summary_results, save_dir):
    df = pd.DataFrame(summary_results)

    densities = df["LanderDensity"]
    means = df["OptimalStepMean"]
    stds = df["OptimalStepStd"]

    plt.figure(figsize=(12, 6))

    plt.bar(densities, means, yerr=stds, align="center", alpha=0.7, capsize=5)
    plt.xlabel("Lander Density")
    plt.ylabel("Average Steps to Reach Near-Optimal Score")
    plt.title("Average Optimal Step vs Lander Density")
    plt.xticks(densities, rotation=45)
    plt.grid(True, axis="y")

    plot_path = os.path.join(save_dir, "optimal_step_bar_chart.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"[Plot Saved] {plot_path}")


# --------------- Main Training Loop ---------------
if __name__ == "__main__":
    lander_densities = np.linspace(3.0, 9.0, NUM_DENSITY_SETTINGS)
    random.shuffle(lander_densities)

    summary_results = []
    density_results = {}

    for lander_density in lander_densities:
        print(f"\n===== Lander Density = {lander_density:.1f} =====")

        repeat_results = []
        reward_curves = []

        for repeat in range(N_REPEAT):
            print(
                f"\n--- Repeat {repeat + 1}/{N_REPEAT} for Lander Density = {lander_density:.1f} ---"
            )

            train_env = SubprocVecEnv(
                [
                    make_lander_env(
                        lander_density=lander_density,
                        render_mode=None,
                        deterministic_init=False,
                    )
                    for _ in range(N_ENVS)
                ]
            )

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

            eval_callback = EvalAndGifCallback(
                lander_density=lander_density,
                repeat=repeat + 1,
                eval_interval=EVAL_INTERVAL,
                optimal_score=NEAR_OPTIMAL_SCORE,
                verbose=1,
            )
            progress_callback = ProgressBarCallback(total_timesteps=TRAIN_STEPS)

            model.learn(
                total_timesteps=TRAIN_STEPS, callback=[eval_callback, progress_callback]
            )

            model_path = os.path.join(
                SAVE_PATH,
                f"sac_lander_density_{lander_density:.1f}_repeat_{repeat + 1}.zip",
            )
            model.save(model_path)

            repeat_results.append(
                {
                    "LanderDensity": lander_density,
                    "Repeat": repeat + 1,
                    "OptimalStep": eval_callback.step_reached_optimal or TRAIN_STEPS,
                    "BestScore": eval_callback.best_mean_reward,
                }
            )

            df_repeat = pd.DataFrame(
                eval_callback.records, columns=["Timesteps", "MeanReward", "StdReward"]
            )
            reward_curves.append(df_repeat)

            print(
                f"\n--- Cleanup after Repeat {repeat + 1} for Lander Density {lander_density:.1f} ---"
            )
            train_env.close()
            del model
            gc.collect()
            torch.cuda.empty_cache()

        best_scores = [res["BestScore"] for res in repeat_results]
        optimal_steps = [res["OptimalStep"] for res in repeat_results]

        summary_results.append(
            {
                "LanderDensity": lander_density,
                "BestScoreMean": np.mean(best_scores),
                "BestScoreStd": np.std(best_scores),
                "OptimalStepMean": np.mean(optimal_steps),
                "OptimalStepStd": np.std(optimal_steps),
            }
        )

        mean_rewards = np.mean([df["MeanReward"] for df in reward_curves], axis=0)
        std_rewards = np.std([df["MeanReward"] for df in reward_curves], axis=0)
        timesteps = reward_curves[0]["Timesteps"]

        density_results[lander_density] = {
            "Timesteps": timesteps,
            "MeanReward": mean_rewards,
            "StdReward": std_rewards,
        }

        df_density = pd.DataFrame(
            {
                "Timesteps": timesteps,
                "MeanReward": mean_rewards,
                "StdReward": std_rewards,
            }
        )
        density_csv_path = os.path.join(
            SAVE_PATH, f"mean_std_density_{lander_density:.1f}.csv"
        )
        df_density.to_csv(density_csv_path, index=False)

    df_summary = pd.DataFrame(summary_results)
    df_summary_path = os.path.join(SAVE_PATH, "summary_results_mean_std.csv")
    df_summary.to_csv(df_summary_path, index=False)

    print("\n===== All Training Completed =====")
    print(df_summary)

    plot_results(density_results, SAVE_PATH)
    plot_optimal_step_bar_chart(summary_results, SAVE_PATH)
