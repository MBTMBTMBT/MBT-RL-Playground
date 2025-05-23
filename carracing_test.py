import gc
import os

import gymnasium as gym
import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
import matplotlib.pyplot as plt

from sbx import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from utils import EvalAndGifCallback

# --------------- Configuration ---------------
NUM_SEEDS = 8  # Number of different map seeds
N_REPEAT = 3
N_ENVS = 16
TRAIN_STEPS = 1_500_000
EVAL_INTERVAL = 2_500 * N_ENVS
NUM_INIT_STATES = 16
EVAL_EPISODES = NUM_INIT_STATES
NEAR_OPTIMAL_SCORE = 8.5  # Adjusted for CarRacing reward scale

SAVE_PATH = "./carracing_mapseed_results"
os.makedirs(SAVE_PATH, exist_ok=True)


# --------------- Environment Factory ---------------
def make_carracing_env(
    map_seed,
    render_mode=None,
    deterministic_init=False,
    number_of_initial_states=NUM_INIT_STATES,
    init_seed=None,
):
    def _init():
        env = gym.make(
            "CarRacingFixedMap-v2",
            continuous=True,
            render_mode=render_mode,
            map_seed=map_seed,
            fixed_start=deterministic_init,
            backwards_tolerance=5,
            grass_tolerance=25,
            number_of_initial_states=number_of_initial_states,
            init_seed=init_seed,
            vector_obs=True,
        )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return _init


# --------------- GIF and Evaluation Callback ---------------


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


def plot_results(seed_results, save_dir):
    plt.figure(figsize=(12, 8))

    # Plot all map seeds on one figure
    for map_seed, results in seed_results.items():
        steps = results["Timesteps"]
        means = results["MeanReward"]
        stds = results["StdReward"]

        plt.plot(steps, means, label=f"Map Seed {map_seed}")
        plt.fill_between(
            steps,
            np.array(means) - np.array(stds),
            np.array(means) + np.array(stds),
            alpha=0.2,
        )

        # Plot individual per-map_seed reward curves
        plt_single = plt.figure(figsize=(10, 6))
        ax = plt_single.add_subplot(111)
        ax.plot(steps, means, label=f"Map Seed {map_seed}")
        ax.fill_between(steps, means - stds, means + stds, alpha=0.2)
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Mean Reward")
        ax.set_title(f"Reward Curve for Map Seed {map_seed}")
        ax.legend()
        ax.grid()

        single_plot_path = os.path.join(
            save_dir, f"reward_curve_mapseed_{map_seed}.png"
        )
        plt_single.savefig(single_plot_path)
        plt.close(plt_single)

    plt.xlabel("Timesteps")
    plt.ylabel("Mean Reward")
    plt.title("Evaluation Reward over Timesteps (All Map Seeds)")
    plt.legend()
    plt.grid()

    all_plot_path = os.path.join(save_dir, "reward_curves_all_mapseeds.png")
    plt.savefig(all_plot_path)
    plt.close()

    print(f"[Plot Saved] {all_plot_path}")


def plot_optimal_step_bar_chart(summary_results, save_dir):
    df = pd.DataFrame(summary_results)

    map_seeds = df["MapSeed"]
    means = df["OptimalStepMean"]
    stds = df["OptimalStepStd"]

    plt.figure(figsize=(12, 6))
    plt.bar(map_seeds, means, yerr=stds, align="center", alpha=0.7, capsize=5)

    plt.xlabel("Map Seed")
    plt.ylabel("Average Steps to Reach Near-Optimal Score")
    plt.title("Average Optimal Step vs Map Seed")
    plt.xticks(map_seeds)
    plt.grid(True, axis="y")

    plot_path = os.path.join(save_dir, "optimal_step_bar_chart.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"[Plot Saved] {plot_path}")


# --------------- Main Training Loop ---------------
if __name__ == "__main__":
    map_seeds = list(range(NUM_SEEDS))
    summary_results = []
    seed_results = {}

    for map_seed in map_seeds:
        print(f"\n===== CarRacing Map Seed = {map_seed} =====")

        env = gym.make(
            "CarRacingFixedMap-v2",
            continuous=True,
            render_mode=None,
            map_seed=map_seed,
            fixed_start=True,
            backwards_tolerance=5,
            grass_tolerance=15,
            number_of_initial_states=16,
            init_seed=None,
            vector_obs=True,
        )
        env.reset()
        track_img = env.unwrapped.get_track_image(
            figsize=(10, 10),
        )
        map_path = os.path.join(SAVE_PATH, f"car_racing_map_seed_{map_seed}.png")
        plt.imsave(map_path, track_img)
        env.close()
        gc.collect()

        repeat_results = []
        reward_curves = []

        for repeat in range(N_REPEAT):
            print(f"\n--- Repeat {repeat + 1}/{N_REPEAT} for Map Seed = {map_seed} ---")

            train_env = SubprocVecEnv(
                [
                    make_carracing_env(
                        map_seed=map_seed,
                        render_mode=None,
                        deterministic_init=False,
                        number_of_initial_states=NUM_INIT_STATES,
                        init_seed=None,
                    )
                    for _ in range(N_ENVS)
                ]
            )

            model = SAC(
                "MlpPolicy",  # Vector observation, so MlpPolicy
                train_env,
                verbose=0,
                learning_rate=2e-4,
                buffer_size=2_500_000,
                learning_starts=int(EVAL_INTERVAL * 2.5),
                batch_size=512,
                tau=0.005,
                train_freq=N_ENVS,
                gradient_steps=N_ENVS * 8,
                ent_coef="auto",
                policy_kwargs=dict(net_arch=[256, 256]),
            )

            eval_callback = EvalAndGifCallback(
                map_seed=map_seed,
                run_idx=repeat + 1,
                eval_interval=EVAL_INTERVAL,
                optimal_score=NEAR_OPTIMAL_SCORE,
                verbose=1,
            )
            progress_callback = ProgressBarCallback(total_timesteps=TRAIN_STEPS)

            model.learn(
                total_timesteps=TRAIN_STEPS, callback=[eval_callback, progress_callback]
            )

            repeat_results.append(
                {
                    "MapSeed": map_seed,
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
                f"\n--- Cleanup after Repeat {repeat + 1} for Map Seed {map_seed} ---"
            )
            train_env.close()
            del model
            gc.collect()
            torch.cuda.empty_cache()

        best_scores = [res["BestScore"] for res in repeat_results]
        optimal_steps = [res["OptimalStep"] for res in repeat_results]

        summary_results.append(
            {
                "MapSeed": map_seed,
                "BestScoreMean": np.mean(best_scores),
                "BestScoreStd": np.std(best_scores),
                "OptimalStepMean": np.mean(optimal_steps),
                "OptimalStepStd": np.std(optimal_steps),
            }
        )

        mean_rewards = np.mean([df["MeanReward"] for df in reward_curves], axis=0)
        std_rewards = np.std([df["MeanReward"] for df in reward_curves], axis=0)
        timesteps = reward_curves[0]["Timesteps"]

        seed_results[map_seed] = {
            "Timesteps": timesteps,
            "MeanReward": mean_rewards,
            "StdReward": std_rewards,
        }

        df_seed = pd.DataFrame(
            {
                "Timesteps": timesteps,
                "MeanReward": mean_rewards,
                "StdReward": std_rewards,
            }
        )
        seed_csv_path = os.path.join(SAVE_PATH, f"mean_std_mapseed_{map_seed}.csv")
        df_seed.to_csv(seed_csv_path, index=False)

    df_summary = pd.DataFrame(summary_results)
    df_summary_path = os.path.join(SAVE_PATH, "summary_results_mean_std.csv")
    df_summary.to_csv(df_summary_path, index=False)

    print("\n===== All Training Completed =====")
    print(df_summary)

    # Draw plots
    plot_results(seed_results, SAVE_PATH)
    plot_optimal_step_bar_chart(summary_results, SAVE_PATH)
