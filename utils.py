import os
from typing import Union, Any

import cv2
import gymnasium as gym
import imageio
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from tqdm import tqdm

import custom_envs

def make_lunarlander_env(
    lander_density,
    render_mode=None,
    deterministic_init=False,
    number_of_initial_states=64,
    init_seed=None,
):
    def _init():
        env = gym.make(
            "CustomLunarLander-v3",
            continuous=True,
            render_mode=render_mode,
            gravity=-10.0,  # fixed gravity
            lander_density=lander_density,
            use_deterministic_initial_states=deterministic_init,
            custom_seed=init_seed if deterministic_init else None,
            number_of_initial_states=number_of_initial_states,
        )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return _init


def make_carracing_env(
    map_seed,
    render_mode=None,
    deterministic_init=False,
    number_of_initial_states=16,
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


class EvalAndGifCallback(BaseCallback):
    def __init__(
            self,
            config: dict,
            env_param: Union[int, float, Any],
            n_eval_envs: int,
            run_idx: int,
            eval_interval,
            optimal_score,
            verbose=1,
    ):
        super().__init__(verbose)
        self.config = config
        self.env_param = env_param
        self.run_idx = run_idx
        self.eval_interval = eval_interval
        self.optimal_score = optimal_score

        self.best_mean_reward = -np.inf
        self.step_reached_optimal = None
        self.records = []
        self.last_eval_step = 0

        self.eval_episodes = self.config["eval_episodes"]
        self.n_eval_envs = n_eval_envs

        # check the config to find the environment type
        if config["env_type"] == "lunarlander":
            self.eval_env = SubprocVecEnv(
                [
                    make_carracing_env(
                        map_seed=env_param,
                        render_mode=None,
                        deterministic_init=False,
                        number_of_initial_states=config["num_init_states"],
                        init_seed=i,
                    )
                    for i in range(self.n_eval_envs)
                ]
            )

        elif config["env_type"] == "carracing":
            self.eval_env = SubprocVecEnv(
                [
                    make_lunarlander_env(
                        lander_density=env_param,
                        render_mode=None,
                        deterministic_init=True,
                        number_of_initial_states=config["num_init_states"],
                        init_seed=i,
                    )
                    for i in range(self.n_eval_envs)
                ]
            )

        else:
            self.eval_env = None

        # Save path for best model
        self.best_model_path = os.path.join(
            config["save_path"],
            f"sac_env_param_{self.env_param}_run_{self.run_idx}_best.zip",
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
                    f"[EvalCallback] Env Param {self.env_param} | Repeat {self.run_idx} | "
                    f"Steps {self.num_timesteps} | Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}"
                )

            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                # Save current best model
                print(
                    f"[Best Model] Saving new best model at step {self.num_timesteps} "
                    f"with mean reward {mean_reward:.2f}"
                )
                self.model.save(self.best_model_path)
                if (
                    self.config["near_optimal_score"] > 0 and mean_reward >= (self.config["near_optimal_score"] / 2)
                ) or self.config["near_optimal_score"] <= 0:
                    self.save_gif()

            # if TRAIN_STEPS - EVAL_INTERVAL * 2 < self.num_timesteps:
            #     self.save_gif()

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
            self.config["save_path"],
            f"eval_log_mapseed_{self.env_param}_repeat_{self.run_idx}.csv",
        )
        df.to_csv(repeat_log_path, index=False)
        self.eval_env.close()

    def save_gif(self):
        frames = []
        initial_state_count = 8

        if self.config["env_type"] == "lunarlander":
            single_env = DummyVecEnv(
                [
                    make_carracing_env(
                        map_seed=self.env_param,
                        render_mode="rgb_array",
                        deterministic_init=False,
                        number_of_initial_states=initial_state_count,
                        init_seed=0,
                    )
                ]
            )

        elif self.config["env_type"] == "carracing":
            single_env = DummyVecEnv(
                [
                    make_lunarlander_env(
                        lander_density=self.env_param,
                        render_mode="rgb_array",
                        deterministic_init=True,
                        number_of_initial_states=initial_state_count,
                        init_seed=0,
                    )
                ]
            )

        else:
            single_env = None

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
            self.config["save_path"],
            f"sac_env_param_{self.env_param}_repeat_{self.run_idx}_all_initial_states.gif",
        )

        imageio.mimsave(gif_path, new_frames, duration=20, loop=0)
        print(f"[GIF Saved] {gif_path}")


def plot_lunarlander_results(density_results, save_dir):
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


def plot_lunarlander_optimal_step_bar_chart(summary_results, save_dir):
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


def plot_carracing_results(map_seed_results, save_dir):
    plt.figure(figsize=(12, 8))

    # Plot all map seeds on one figure
    for map_seed, results in map_seed_results.items():
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


def plot_carracing_optimal_step_bar_chart(summary_results, save_dir):
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


def compare_gaussian_distributions(
        mean1: np.ndarray,
        std1: np.ndarray,
        mean2: np.ndarray,
        std2: np.ndarray,
) -> dict:
    """
    Compute multiple distribution distance/divergence metrics between two diagonal Gaussians.

    Args:
        mean1 (np.ndarray): Mean of distribution 1, shape (batch_size, action_dim)
        std1 (np.ndarray): Std of distribution 1, same shape
        mean2 (np.ndarray): Mean of distribution 2, same shape
        std2 (np.ndarray): Std of distribution 2, same shape

    Returns:
        dict: {
            'kl_forward': [...],
            'kl_reverse': [...],
            'js': [...],
            'wasserstein2': [...],
            'hellinger': [...],
            'bhattacharyya': [...]
        }, each is np.ndarray of shape (batch_size,)
    """

    var1 = std1 ** 2
    var2 = std2 ** 2
    eps = 1e-8  # For numerical stability

    # KL(P||Q): forward KL
    kl_forward = 0.5 * np.sum(
        (var1 + (mean1 - mean2) ** 2) / (var2 + eps) - 1 + np.log((var2 + eps) / (var1 + eps)),
        axis=1
    )

    # KL(Q||P): reverse KL
    kl_reverse = 0.5 * np.sum(
        (var2 + (mean2 - mean1) ** 2) / (var1 + eps) - 1 + np.log((var1 + eps) / (var2 + eps)),
        axis=1
    )

    # JS Divergence: average of KL(P||M) and KL(Q||M)
    mean_m = 0.5 * (mean1 + mean2)
    var_m = 0.5 * (var1 + var2)
    kl1_m = 0.5 * np.sum(
        (var1 + (mean1 - mean_m) ** 2) / (var_m + eps) - 1 + np.log((var_m + eps) / (var1 + eps)),
        axis=1
    )
    kl2_m = 0.5 * np.sum(
        (var2 + (mean2 - mean_m) ** 2) / (var_m + eps) - 1 + np.log((var_m + eps) / (var2 + eps)),
        axis=1
    )
    js = 0.5 * (kl1_m + kl2_m)

    # Wasserstein-2 Distance (squared form)
    mean_diff2 = np.sum((mean1 - mean2) ** 2, axis=1)
    std_diff2 = np.sum((std1 - std2) ** 2, axis=1)
    wasserstein2 = mean_diff2 + std_diff2

    # Bhattacharyya distance (diagonal Gaussians)
    sigma_avg = 0.5 * (var1 + var2)
    term1 = 0.125 * np.sum((mean1 - mean2) ** 2 / (sigma_avg + eps), axis=1)
    term2 = 0.5 * np.sum(np.log((sigma_avg + eps) / np.sqrt(var1 * var2 + eps)), axis=1)
    bhattacharyya = term1 + term2

    # Hellinger distance (squared form)
    hellinger_squared = 1 - np.prod(
        (2 * std1 * std2 / (var1 + var2 + eps)) ** 0.25 *
        np.exp(-0.25 * (mean1 - mean2) ** 2 / (var1 + var2 + eps)),
        axis=1
    )
    hellinger = np.sqrt(np.clip(hellinger_squared, 0.0, 1.0))

    return {
        "kl_forward": kl_forward,
        "kl_reverse": kl_reverse,
        "js": js,
        "wasserstein2": wasserstein2,
        "bhattacharyya": bhattacharyya,
        "hellinger": hellinger
    }

