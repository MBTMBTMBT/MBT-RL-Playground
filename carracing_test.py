import gc
import os

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
NUM_SEEDS = 5  # Number of different map seeds
N_REPEAT = 8
N_ENVS = 16
TRAIN_STEPS = 1_200_000
EVAL_INTERVAL = 2_400 * N_ENVS
NUM_INIT_STATES = 32
EVAL_EPISODES = NUM_INIT_STATES
NEAR_OPTIMAL_SCORE = 9.0  # Adjusted for CarRacing reward scale

GIF_LENGTH = 500
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
            grass_tolerance=15,
            number_of_initial_states=number_of_initial_states,
            init_seed=init_seed,
            vector_obs=True,
        )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return _init


# --------------- GIF and Evaluation Callback ---------------
class EvalAndGifCallback(BaseCallback):
    def __init__(self, map_seed, repeat, eval_interval, optimal_score, verbose=1):
        super().__init__(verbose)
        self.map_seed = map_seed
        self.repeat = repeat
        self.eval_interval = eval_interval
        self.optimal_score = optimal_score

        self.best_mean_reward = -np.inf
        self.step_reached_optimal = None
        self.records = []
        self.last_eval_step = 0

        self.eval_episodes = EVAL_EPISODES
        self.n_eval_envs = N_ENVS

        self.eval_env = SubprocVecEnv([
            make_carracing_env(
                map_seed=self.map_seed,
                render_mode=None,
                deterministic_init=False,
                number_of_initial_states=NUM_INIT_STATES,
                init_seed=i,
            ) for i in range(self.n_eval_envs)
        ])

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
                    f"[EvalCallback] Map Seed {self.map_seed} | Repeat {self.repeat} | "
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
        df = pd.DataFrame(self.records, columns=["Timesteps", "MeanReward", "StdReward"])
        repeat_log_path = os.path.join(
            SAVE_PATH,
            f"eval_log_mapseed_{self.map_seed}_repeat_{self.repeat}.csv",
        )
        df.to_csv(repeat_log_path, index=False)
        self.eval_env.close()

    def save_gif(self):
        frames = []
        initial_state_count = 4

        single_env = DummyVecEnv([
            make_carracing_env(
                map_seed=self.map_seed,
                render_mode="rgb_array",
                deterministic_init=False,
                number_of_initial_states=initial_state_count,
                init_seed=0,
            )
        ])

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
            f"carracing_mapseed_{self.map_seed}_repeat_{self.repeat}_all_initial_states.gif",
        )

        imageio.mimsave(gif_path, new_frames, duration=20, loop=0)
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
            alpha=0.2
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

        single_plot_path = os.path.join(save_dir, f"reward_curve_mapseed_{map_seed}.png")
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

        repeat_results = []
        reward_curves = []

        for repeat in range(N_REPEAT):
            print(f"\n--- Repeat {repeat + 1}/{N_REPEAT} for Map Seed = {map_seed} ---")

            train_env = SubprocVecEnv([
                make_carracing_env(
                    map_seed=map_seed,
                    render_mode=None,
                    deterministic_init=False,
                    number_of_initial_states=NUM_INIT_STATES,
                    init_seed=None,
                ) for _ in range(N_ENVS)
            ])

            model = SAC(
                "MlpPolicy",  # Vector observation, so MlpPolicy
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
                map_seed=map_seed,
                repeat=repeat + 1,
                eval_interval=EVAL_INTERVAL,
                optimal_score=NEAR_OPTIMAL_SCORE,
                verbose=1,
            )
            progress_callback = ProgressBarCallback(total_timesteps=TRAIN_STEPS)

            model.learn(
                total_timesteps=TRAIN_STEPS,
                callback=[eval_callback, progress_callback]
            )

            model_path = os.path.join(
                SAVE_PATH,
                f"sac_carracing_mapseed_{map_seed}_repeat_{repeat + 1}.zip"
            )
            model.save(model_path)

            repeat_results.append({
                "MapSeed": map_seed,
                "Repeat": repeat + 1,
                "OptimalStep": eval_callback.step_reached_optimal or TRAIN_STEPS,
                "BestScore": eval_callback.best_mean_reward,
            })

            df_repeat = pd.DataFrame(
                eval_callback.records,
                columns=["Timesteps", "MeanReward", "StdReward"]
            )
            reward_curves.append(df_repeat)

            print(f"\n--- Cleanup after Repeat {repeat + 1} for Map Seed {map_seed} ---")
            train_env.close()
            del model
            gc.collect()
            torch.cuda.empty_cache()

        best_scores = [res["BestScore"] for res in repeat_results]
        optimal_steps = [res["OptimalStep"] for res in repeat_results]

        summary_results.append({
            "MapSeed": map_seed,
            "BestScoreMean": np.mean(best_scores),
            "BestScoreStd": np.std(best_scores),
            "OptimalStepMean": np.mean(optimal_steps),
            "OptimalStepStd": np.std(optimal_steps),
        })

        mean_rewards = np.mean([df["MeanReward"] for df in reward_curves], axis=0)
        std_rewards = np.std([df["MeanReward"] for df in reward_curves], axis=0)
        timesteps = reward_curves[0]["Timesteps"]

        seed_results[map_seed] = {
            "Timesteps": timesteps,
            "MeanReward": mean_rewards,
            "StdReward": std_rewards,
        }

        df_seed = pd.DataFrame({
            "Timesteps": timesteps,
            "MeanReward": mean_rewards,
            "StdReward": std_rewards,
        })
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
