import os
import tempfile
import warnings
from typing import Union, Any, Optional, Callable, List

import cv2
import gymnasium as gym
import imageio
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy import floating
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    DummyVecEnv,
    VecEnv,
    is_vecenv_wrapped,
    VecMonitor,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from tabulate import tabulate
from tqdm import tqdm

import custom_envs
from gaussian_agent import SACJax as SAC, MixPolicySAC


MEASURES = [
    "kl_forward",
    "kl_reverse",
    "js",
    "wasserstein2",
    "bhattacharyya",
    "hellinger",
    "entropy1",
    "entropy2",
]


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
            number_of_initial_states=number_of_initial_states,
            use_deterministic_initial_states=deterministic_init,
            init_seed=init_seed if deterministic_init else None,
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


def hard_clone_agent(
    agent: Union[BaseAlgorithm, SAC],
    temp_dir: str = ".",
) -> Optional[Union[BaseAlgorithm, SAC]]:
    """Hard clone a SBX agent via save and load mechanism,
    including replay buffer if available.

    Args:
        agent (BaseAlgorithm | SAC): Agent to be cloned.
        temp_dir (str): Temporary directory to store checkpoint.

    Returns:
        Optional[BaseAlgorithm | SAC]: Cloned agent with same weights and replay buffer.
    """
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False, dir=temp_dir) as tmp_file:
        model_path = tmp_file.name
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False, dir=temp_dir) as tmp_file:
        buffer_path = tmp_file.name

    try:
        # Save model and replay buffer
        agent.save(model_path)
        if agent.replay_buffer is not None:
            agent.save_replay_buffer(buffer_path)

        # Load model and replay buffer
        new_agent = agent.__class__.load(model_path, env=agent.env)
        if new_agent.replay_buffer is not None:
            new_agent.load_replay_buffer(buffer_path)

    finally:
        os.remove(model_path)
        os.remove(buffer_path)

    return new_agent


class EvalAndGifCallback(BaseCallback):
    def __init__(
        self,
        config: dict,
        env_param: Union[int, float, Any],
        n_eval_envs: int,
        run_idx: int,
        eval_interval: int,
        optimal_score: Union[int, float],
        verbose=1,
        temp_dir=".",
        use_default_policy=True,
    ):
        super().__init__(verbose)
        self.config = config
        self.env_param = env_param
        self.run_idx = run_idx
        self.eval_interval = eval_interval
        self.optimal_score = optimal_score

        self.best_mean_reward = -np.inf
        self.step_reached_optimal = None
        self.records = {
            "reward": [],
            "prior_policy-kl_forward": [],
            "prior_policy-kl_reverse": [],
            "prior_policy-js": [],
            "prior_policy-wasserstein2": [],
            "prior_policy-bhattacharyya": [],
            "prior_policy-hellinger": [],
            "prior_policy-entropy1": [],
            "prior_policy-entropy2": [],
            "current_policy-kl_forward": [],
            "current_policy-kl_reverse": [],
            "current_policy-js": [],
            "current_policy-wasserstein2": [],
            "current_policy-bhattacharyya": [],
            "current_policy-hellinger": [],
            "current_policy-entropy1": [],
            "current_policy-entropy2": [],
        }
        self.last_eval_step = 0

        self.eval_episodes = self.config["eval_episodes"]
        self.n_eval_envs = n_eval_envs

        self.temp_dir = temp_dir
        self.original_model = None

        # check the config to find the environment type
        if config["env_type"] == "lunarlander":
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

        elif config["env_type"] == "carracing":
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

        else:
            self.eval_env = None

        # Save path for best model
        self.best_model_path = os.path.join(
            config["save_path"],
            f"sac_env_param_{self.env_param}_run_{self.run_idx}_best.zip",
        )

        self.use_default_policy = use_default_policy

    def _init_callback(self) -> None:
        self.original_model = hard_clone_agent(self.model, temp_dir=self.temp_dir)

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_eval_step >= self.eval_interval:
            self.last_eval_step = self.num_timesteps

            # Evaluate with prior policy sampling
            prior_result = evaluate_policy_with_distribution(
                self.original_model,
                self.model,
                use_default_policy=self.use_default_policy,
                use_default_policy_for_prior=False,
                env=self.eval_env,
                n_eval_episodes=self.eval_episodes,
                deterministic=True,
                render=False,
                warn=False,
            )

            # Evaluate with current policy sampling
            current_result = evaluate_policy_with_distribution(
                self.model,
                self.original_model,
                use_default_policy=False,
                use_default_policy_for_prior=self.use_default_policy,
                env=self.eval_env,
                n_eval_episodes=self.eval_episodes,
                deterministic=True,
                render=False,
                warn=False,
            )

            # Unpack reward results (only from current policy evaluation)
            mean_reward, std_reward = current_result["reward"]
            self.records["reward"].append((self.num_timesteps, mean_reward, std_reward))

            # Record prior policy sampling distribution metrics
            for key, value in prior_result.items():
                if key == "reward":
                    continue  # Skip prior reward
                self.records[f"prior_policy-{key}"].append(
                    (self.num_timesteps, value[0], value[1])
                )

            # Record current policy sampling distribution metrics
            for key, value in current_result.items():
                if key == "reward":
                    continue  # Already recorded
                self.records[f"current_policy-{key}"].append(
                    (self.num_timesteps, value[0], value[1])
                )

            if self.verbose:
                # Prepare table content
                table_data = [
                    ["Env Param", self.env_param],
                    ["Repeat", self.run_idx],
                    ["Steps", self.num_timesteps],
                    ["Mean Reward", f"{mean_reward:.2f} ± {std_reward:.2f}"],
                    ["-- Prior Policy Metrics --", ""],
                ]

                # Append prior_policy metrics
                for key in MEASURES:
                    mean, std = prior_result[key]
                    table_data.append(
                        [f"prior_policy-{key}", f"{mean:.4f} ± {std:.4f}"]
                    )

                # Append current_policy metrics
                table_data.append(["-- Current Policy Metrics --", ""])
                for key in MEASURES:
                    mean, std = current_result[key]
                    table_data.append(
                        [f"current_policy-{key}", f"{mean:.4f} ± {std:.4f}"]
                    )

                print("[EvalCallback] Evaluation Summary")
                print(tabulate(table_data, tablefmt="grid"))

            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                print(
                    f"[Best Model] Saving new best model at step {self.num_timesteps} "
                    f"with mean reward {mean_reward:.2f}"
                )
                self.model.save(self.best_model_path)

                if (
                    self.config["near_optimal_score"] > 0
                    and mean_reward >= (self.config["near_optimal_score"] / 2)
                ) or self.config["near_optimal_score"] <= 0:
                    pass
                    # self.save_gif()

            if mean_reward >= self.optimal_score and self.step_reached_optimal is None:
                self.step_reached_optimal = self.num_timesteps

        return True

    def _on_training_start(self):
        self.step_reached_optimal = None
        # Force evaluation at step 0
        self.last_eval_step = -self.eval_interval
        self._on_step()

    import numpy as np
    import pandas as pd
    import os

    def _on_training_end(self):
        """
        Save evaluation logs to a CSV file after training ends.
        Automatically pad shorter records with NaN at the front to align with the longest sequence.
        """
        print("[EvalCallback] Training ended. Saving evaluation logs.")
        self.save_gif()

        config = self.config
        env_type = config["env_type"]
        assert env_type in ["lunarlander", "carracing"], "Unsupported env_type."

        # Determine maximum length among all records
        max_len = max(len(v) for v in self.records.values())

        def pad_front(data, target_len):
            pad_len = target_len - len(data)
            if pad_len <= 0:
                return data
            return [np.nan] * pad_len + data

        # Initialize dataframe
        df = pd.DataFrame()

        # Handle Timesteps (assume reward always has timesteps)
        timesteps = [x[0] for x in self.records["reward"]]
        df["Timesteps"] = pad_front(timesteps, max_len)

        # Add reward columns
        df["reward_mean"] = pad_front([x[1] for x in self.records["reward"]], max_len)
        df["reward_std"] = pad_front([x[2] for x in self.records["reward"]], max_len)

        # Add other metrics
        for key in self.records.keys():
            if key == "reward":
                continue
            df[f"{key}_mean"] = pad_front([x[1] for x in self.records[key]], max_len)
            df[f"{key}_std"] = pad_front([x[2] for x in self.records[key]], max_len)

        # Generate save filename
        if env_type == "lunarlander":
            log_name = f"eval_log_density_{self.env_param}_repeat_{self.run_idx}.csv"
        elif env_type == "carracing":
            log_name = f"eval_log_mapseed_{self.env_param}_repeat_{self.run_idx}.csv"
        else:
            raise NotImplementedError

        # Generate full save path
        log_path = os.path.join(config["save_path"], log_name)

        # Save to CSV
        df.to_csv(log_path, index=False)

        print(f"[EvalCallback] Log saved to {log_path}")

        # Close evaluation environment
        self.eval_env.close()

    def save_gif(self):
        frames = []
        initial_state_count = 8

        if self.config["env_type"] == "lunarlander":
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

        elif self.config["env_type"] == "carracing":
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


class CurriculumCallBack(EvalAndGifCallback):
    def __init__(
            self,
            config: dict,
            env_param_at_start: Union[int, float, Any],
            env_param_target: Union[int, float, Any],
            n_eval_envs: int,
            run_idx: int,
            eval_interval: int,
            optimal_score: Union[int, float],
            verbose=1,
            temp_dir=".",
            use_default_policy_at_start=True,
    ):
        super().__init__(
            config=config,
            env_param=env_param_at_start,
            n_eval_envs=n_eval_envs,
            run_idx=run_idx,
            eval_interval=eval_interval,
            optimal_score=optimal_score,
            verbose=verbose,
            temp_dir=temp_dir,
            use_default_policy=use_default_policy_at_start,
        )

        self.env_param_target = env_param_target

        if config["env_type"] == "lunarlander":
            self.eval_env_target = SubprocVecEnv(
                [
                    make_lunarlander_env(
                        lander_density=env_param_target,
                        render_mode=None,
                        deterministic_init=True,
                        number_of_initial_states=config["num_init_states"],
                        init_seed=i,
                    )
                    for i in range(self.n_eval_envs)
                ]
            )

        elif config["env_type"] == "carracing":
            self.eval_env_target = SubprocVecEnv(
                [
                    make_carracing_env(
                        map_seed=env_param_target,
                        render_mode=None,
                        deterministic_init=False,
                        number_of_initial_states=config["num_init_states"],
                        init_seed=i,
                    )
                    for i in range(self.n_eval_envs)
                ]
            )

        else:
            self.eval_env_target = None

        self.records["reward_target"] = []

        self.change_env_flag = False

    def _on_step(self):
        if self.num_timesteps - self.last_eval_step >= self.eval_interval:
            self.last_eval_step = self.num_timesteps

            # Evaluate with prior policy sampling
            prior_result = evaluate_policy_with_distribution(
                self.original_model,
                self.model,
                use_default_policy=self.use_default_policy,
                use_default_policy_for_prior=False,
                env=self.eval_env if not self.change_env_flag else self.eval_env_target,
                n_eval_episodes=self.eval_episodes,
                deterministic=True,
                render=False,
                warn=False,
            )

            # Evaluate with current policy sampling
            current_result = evaluate_policy_with_distribution(
                self.model,
                self.original_model,
                use_default_policy=False,
                use_default_policy_for_prior=self.use_default_policy,
                env=self.eval_env if not self.change_env_flag else self.eval_env_target,
                n_eval_episodes=self.eval_episodes,
                deterministic=True,
                render=False,
                warn=False,
            )

            # Unpack reward results (only from current policy evaluation)
            mean_reward, std_reward = current_result["reward"]
            self.records["reward"].append((self.num_timesteps, mean_reward, std_reward))

            # test target env all the times.
            if not self.change_env_flag:
                mean_reward_, std_reward_ = evaluate_policy(
                    self.model,
                    env=self.eval_env_target,
                    n_eval_episodes=self.eval_episodes,
                    deterministic=True,
                    render=False,
                    warn=False,
                )
                self.records["reward_target"].append(
                    (self.num_timesteps, mean_reward_, std_reward_)
                )
            else:
                self.records["reward_target"].append(
                    (self.num_timesteps, mean_reward, std_reward)
                )

            if mean_reward > self.config["near_optimal_score"] and not self.change_env_flag:
                print("First stage training ends.")
                self.change_env_flag = True
                # clean the buffer, no reuse of the previous data
                self.model.replay_buffer = ReplayBuffer(
                    buffer_size=self.model.replay_buffer.buffer_size,
                    observation_space=self.model.observation_space,
                    action_space=self.model.action_space,
                    device="cpu",
                    n_envs=self.model.replay_buffer.n_envs,
                    optimize_memory_usage=self.model.replay_buffer.optimize_memory_usage,
                    handle_timeout_termination=self.model.replay_buffer.handle_timeout_termination,
                )
                if self.config["env_type"] == "lunarlander":
                    env_target = SubprocVecEnv(
                        [
                            make_lunarlander_env(
                                lander_density=self.env_param_target,
                                render_mode=None,
                                deterministic_init=True,
                                number_of_initial_states=self.config["num_init_states"],
                                init_seed=i,
                            )
                            for i in range(self.n_eval_envs)
                        ]
                    )

                elif self.config["env_type"] == "carracing":
                    env_target = SubprocVecEnv(
                        [
                            make_carracing_env(
                                map_seed=self.env_param_target,
                                render_mode=None,
                                deterministic_init=False,
                                number_of_initial_states=self.config["num_init_states"],
                                init_seed=i,
                            )
                            for i in range(self.n_eval_envs)
                        ]
                    )

                else:
                    env_target = None

                # Step 1: Set new environment
                self.model.set_env(env_target)

                # Step 2: Force reset
                obs = self.model.env.reset()

                # Step 3: Update model's internal state
                self.model._last_obs = obs
                self.model._last_episode_starts = np.ones((env_target.num_envs,), dtype=bool)

            # Record prior policy sampling distribution metrics
            for key, value in prior_result.items():
                if key == "reward":
                    continue  # Skip prior reward
                self.records[f"prior_policy-{key}"].append(
                    (self.num_timesteps, value[0], value[1])
                )

            # Record current policy sampling distribution metrics
            for key, value in current_result.items():
                if key == "reward" or (not self.change_env_flag):
                    continue  # Already recorded
                self.records[f"current_policy-{key}"].append(
                    (self.num_timesteps, value[0], value[1])
                )

            if self.verbose:
                # Prepare table content
                table_data = [
                    ["Env Param", self.env_param],
                    ["Repeat", self.run_idx],
                    ["Steps", self.num_timesteps],
                    ["Mean Reward", f"{mean_reward:.2f} ± {std_reward:.2f}"],
                    ["-- Prior Policy Metrics --", ""],
                ]

                # Append prior_policy metrics
                for key in MEASURES:
                    mean, std = prior_result[key]
                    table_data.append(
                        [f"prior_policy-{key}", f"{mean:.4f} ± {std:.4f}"]
                    )

                # Append current_policy metrics
                table_data.append(["-- Current Policy Metrics --", ""])
                for key in MEASURES:
                    mean, std = current_result[key]
                    table_data.append(
                        [f"current_policy-{key}", f"{mean:.4f} ± {std:.4f}"]
                    )

                print("[EvalCallback] Evaluation Summary")
                print(tabulate(table_data, tablefmt="grid"))

            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                print(
                    f"[Best Model] Saving new best model at step {self.num_timesteps} "
                    f"with mean reward {mean_reward:.2f}"
                )
                self.model.save(self.best_model_path)

                if (
                    self.config["near_optimal_score"] > 0
                    and mean_reward >= (self.config["near_optimal_score"] / 2)
                ) or self.config["near_optimal_score"] <= 0:
                    pass
                    # self.save_gif()

            # must be the target phase
            if self.change_env_flag and mean_reward >= self.optimal_score and self.step_reached_optimal is None:
                self.step_reached_optimal = self.num_timesteps

        return True


def plot_eval_results(config, results, save_dir, save_name=None):
    env_type = config["env_type"]
    assert env_type in ["lunarlander", "carracing"], "Unsupported env_type."

    metrics_keys = [key[:-5] for key in results[list(results.keys())[0]].keys()
                    if key.endswith("_mean")]

    # Get the longest step sequence (global timeline)
    all_timesteps = [res["Timesteps"] for res in results.values()]
    max_len = max(len(t) for t in all_timesteps)
    global_steps = max(all_timesteps, key=len)  # pick the longest one

    for metric in metrics_keys:
        plt.figure(figsize=(12, 8))

        for env_param, result in results.items():
            steps = result["Timesteps"]
            means = result[metric + "_mean"]
            stds = result[metric + "_std"]

            # Right-align the steps to match global_steps
            aligned_steps = global_steps[-len(steps):]

            # Plot on global axis
            plt.plot(aligned_steps, means, label=f"{env_type.capitalize()} Param {env_param}")
            plt.fill_between(aligned_steps, np.array(means) - np.array(stds),
                             np.array(means) + np.array(stds), alpha=0.2)

            # Single curve figure
            plt_single = plt.figure(figsize=(10, 6))
            ax = plt_single.add_subplot(111)
            ax.plot(aligned_steps, means, label=f"{env_type.capitalize()} Param {env_param}")
            ax.fill_between(aligned_steps, np.array(means) - np.array(stds),
                            np.array(means) + np.array(stds), alpha=0.2)
            ax.set_xlabel("Timesteps")
            ax.set_ylabel(f"Mean {metric.replace('_', ' ').title()}")
            ax.set_title(f"{metric.replace('_', ' ').title()} Curve for {env_type.capitalize()} Param {env_param}")
            ax.legend()
            ax.grid()

            single_name = save_name or f"{metric}_curve_{env_type}_param_{env_param}.png"
            if single_name == save_name:
                single_name = f"{metric}_param_{env_param}_" + single_name
            single_plot_path = os.path.join(save_dir, single_name)
            plt_single.savefig(single_plot_path)
            plt.close(plt_single)

        plt.xlabel("Timesteps")
        plt.ylabel(f"Mean {metric.replace('_', ' ').title()}")
        plt.title(f"{metric.replace('_', ' ').title()} over Timesteps (All {env_type.capitalize()} Params)")
        plt.legend()
        plt.grid()

        all_name = save_name or f"{metric}_curves_all_{env_type}.png"
        if all_name == save_name:
            all_name = f"{metric}_" + all_name
        all_plot_path = os.path.join(save_dir, all_name)
        plt.savefig(all_plot_path)
        plt.close()

        print(f"[Plot Saved] {all_plot_path}")


def plot_optimal_step_bar_chart_and_return_max(config, summary_results, save_dir, save_name=None):
    env_type = config["env_type"]
    assert env_type in ["lunarlander", "carracing"], "Unsupported env_type."

    df = pd.DataFrame(summary_results)

    if env_type == "lunarlander":
        x_labels = df["LanderDensity"]
        x_name = "Lander Density"
        param_column = "LanderDensity"
    else:
        x_labels = df["MapSeed"]
        x_name = "Map Seed"
        param_column = "MapSeed"

    means = df["OptimalStepMean"]
    stds = df["OptimalStepStd"]

    plt.figure(figsize=(12, 6))
    plt.bar(x_labels, means, yerr=stds, align="center", alpha=0.7, capsize=5)

    plt.xlabel(x_name)
    plt.ylabel("Average Steps to Reach Near-Optimal Score")
    plt.title(f"Average Optimal Step vs {x_name}")
    plt.xticks(x_labels, rotation=45 if env_type == "lunarlander" else 0)
    plt.grid(True, axis="y")

    bar_name = save_name or f"optimal_step_bar_chart_{env_type}.png"
    plot_path = os.path.join(save_dir, bar_name)
    plt.savefig(plot_path)
    plt.close()

    print(f"[Plot Saved] {plot_path}")

    max_idx = means.idxmax()
    max_param = df.loc[max_idx, param_column]

    return max_param


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
            dynamic_ncols=True
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
            'bhattacharyya': [...],
            'entropy1': [...],
            'entropy2': [...]
        }, each is np.ndarray of shape (batch_size,)
    """
    var1 = std1 ** 2
    var2 = std2 ** 2
    eps = 1e-8  # For numerical stability

    # KL(P||Q): forward KL
    kl_forward = 0.5 * np.sum(
        (var1 + (mean1 - mean2) ** 2) / (var2 + eps)
        - 1
        + np.log((var2 + eps) / (var1 + eps)),
        axis=1,
    )

    # KL(Q||P): reverse KL
    kl_reverse = 0.5 * np.sum(
        (var2 + (mean2 - mean1) ** 2) / (var1 + eps)
        - 1
        + np.log((var1 + eps) / (var2 + eps)),
        axis=1,
    )

    # JS Divergence
    mean_m = 0.5 * (mean1 + mean2)
    var_m = 0.5 * (var1 + var2)
    kl1_m = 0.5 * np.sum(
        (var1 + (mean1 - mean_m) ** 2) / (var_m + eps)
        - 1
        + np.log((var_m + eps) / (var1 + eps)),
        axis=1,
    )
    kl2_m = 0.5 * np.sum(
        (var2 + (mean2 - mean_m) ** 2) / (var_m + eps)
        - 1
        + np.log((var_m + eps) / (var2 + eps)),
        axis=1,
    )
    js = 0.5 * (kl1_m + kl2_m)

    # Wasserstein-2 distance (squared)
    mean_diff2 = np.sum((mean1 - mean2) ** 2, axis=1)
    std_diff2 = np.sum((std1 - std2) ** 2, axis=1)
    wasserstein2 = mean_diff2 + std_diff2

    # Bhattacharyya distance
    sigma_avg = 0.5 * (var1 + var2)
    term1 = 0.125 * np.sum((mean1 - mean2) ** 2 / (sigma_avg + eps), axis=1)
    term2 = 0.5 * np.sum(np.log((sigma_avg + eps) / np.sqrt(var1 * var2 + eps)), axis=1)
    bhattacharyya = term1 + term2

    # Hellinger distance (squared form)
    hellinger_squared = 1 - np.prod(
        (2 * std1 * std2 / (var1 + var2 + eps)) ** 0.25
        * np.exp(-0.25 * (mean1 - mean2) ** 2 / (var1 + var2 + eps)),
        axis=1,
    )
    hellinger = np.sqrt(np.clip(hellinger_squared, 0.0, 1.0))

    # Differential entropy of a diagonal Gaussian
    entropy1 = 0.5 * np.sum(np.log(2 * np.pi * np.e * var1 + eps), axis=1)
    entropy2 = 0.5 * np.sum(np.log(2 * np.pi * np.e * var2 + eps), axis=1)

    return {
        "kl_forward": kl_forward,
        "kl_reverse": kl_reverse,
        "js": js,
        "wasserstein2": wasserstein2,
        "bhattacharyya": bhattacharyya,
        "hellinger": hellinger,
        "entropy1": entropy1,
        "entropy2": entropy2,
    }


def evaluate_policy_with_distribution(
    model: Union[
        BaseAlgorithm,
        SAC,
    ],
    prior_model: Union[
        BaseAlgorithm,
        SAC,
    ],
    use_default_policy: bool,
    use_default_policy_for_prior: bool,
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[dict[str, Any], dict[str, Any]], None]] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
    warn: bool = True,
) -> dict[str, Any]:
    """
    Evaluate a policy and compare it to a prior policy in terms of both reward and Gaussian distribution metrics.

    Returns:
        A dictionary containing:
            - "reward": (mean, std)
            - divergence metrics: (mean, std) for each
            - (optional) episode_rewards and episode_lengths
    """

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])  # type: ignore

    is_monitor_wrapped = (
        is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]
    )
    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper.",
            UserWarning,
        )

    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []
    episode_counts = np.zeros(n_envs, dtype=int)
    episode_count_targets = np.array(
        [(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype=int
    )

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype=int)
    distribution_metrics = []

    observations = env.reset()
    states = None
    episode_starts = np.ones((n_envs,), dtype=bool)

    while (episode_counts < episode_count_targets).any():
        actions, states = model.predict(
            observations,
            state=states,
            episode_start=episode_starts,
            deterministic=deterministic,
        )

        if use_default_policy:
            mean1, std1 = model.get_default_action_distribution(observations)
        else:
            mean1, std1 = model.predict_action_distribution(observations)
        if use_default_policy_for_prior:
            mean2, std2 = prior_model.get_default_action_distribution(observations)
        else:
            mean2, std2 = prior_model.predict_action_distribution(observations)

        metrics = compare_gaussian_distributions(mean1, std1, mean2, std2)
        distribution_metrics.append(metrics)

        new_observations, rewards, dones, infos = env.step(actions)
        current_rewards += rewards
        current_lengths += 1

        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:
                reward = rewards[i]
                done = dones[i]
                info = infos[i]
                episode_starts[i] = done

                if callback is not None:
                    callback(locals(), globals())

                if done:
                    if is_monitor_wrapped and "episode" in info:
                        episode_rewards.append(info["episode"]["r"])
                        episode_lengths.append(info["episode"]["l"])
                        episode_counts[i] += 1
                    else:
                        episode_rewards.append(current_rewards[i])
                        episode_lengths.append(current_lengths[i])
                        episode_counts[i] += 1

                    current_rewards[i] = 0
                    current_lengths[i] = 0

        observations = new_observations

        if render:
            env.render()

    reward_mean = np.mean(episode_rewards)
    reward_std = np.std(episode_rewards)

    # Stack and aggregate metrics
    stacked_metrics = {
        key: np.concatenate([d[key] for d in distribution_metrics], axis=0)
        for key in distribution_metrics[0].keys()
    }
    aggregated_metrics = {
        key: (np.mean(val), np.std(val)) for key, val in stacked_metrics.items()
    }

    result = {"reward": (reward_mean, reward_std)}
    result.update(aggregated_metrics)

    if reward_threshold is not None:
        assert (
            reward_mean > reward_threshold
        ), f"Mean reward below threshold: {reward_mean:.2f} < {reward_threshold:.2f}"

    if return_episode_rewards:
        result["episode_rewards"] = episode_rewards
        result["episode_lengths"] = episode_lengths

    return result


def evaluate_mix_policy_agent(
    agent: MixPolicySAC,
    test_env,
    total_episodes: int,
    num_p_values: int = 20,
) -> list[floating[Any]]:
    """
    Evaluate MixPolicySAC agent with different p values from 0.0 to 1.0.

    Args:
        agent (MixPolicySAC): MixPolicySAC agent.
        test_env: SB3 VecEnv.
        total_episodes (int): Total episodes per p value.
        num_p_values (int): Number of p values to evaluate.

    Returns:
        List[float]: Mean episode rewards for each p value.
    """
    p_values = np.linspace(0.0, 1.0, num_p_values)
    rewards_per_p = []

    n_envs = test_env.num_envs
    n_eval_episodes = total_episodes
    assert n_eval_episodes >= n_envs, "Total episodes must >= number of envs."

    progress_bar = tqdm(p_values, desc="Eval MixPolicy", dynamic_ncols=True)

    for p in progress_bar:
        episode_rewards = []
        obs = test_env.reset()

        current_rewards = np.zeros(n_envs, dtype=np.float32)
        episode_counts = np.zeros(n_envs, dtype=np.int32)

        while len(episode_rewards) < n_eval_episodes:
            actions = agent.predict_with_weight(obs, p)
            obs, rewards, dones, infos = test_env.step(actions)
            current_rewards += rewards

            for i in range(n_envs):
                if dones[i]:
                    episode_rewards.append(current_rewards[i])
                    current_rewards[i] = 0.0
                    episode_counts[i] += 1

        mean_reward = np.mean(episode_rewards[:n_eval_episodes])
        rewards_per_p.append(mean_reward)

        # Dynamically update progress bar info
        progress_bar.set_postfix(
            p=f"{p:.2f}", mean_rwd=f"{mean_reward:.2f}"
        )

    return rewards_per_p


def compute_and_plot_mix_policy_results(config, results_dict, save_path):
    env_type = config["env_type"]
    param_name = "LanderDensity" if env_type == "lunarlander" else "MapSeed"

    # Initialize result dictionary
    final_results = {
        "mean_rewards": {},
        "std_rewards": {},
        "integrals": {},
        "integrals_std": {},
        "normalized_mean_rewards": {},
        "normalized_std_rewards": {},
        "integrals_norm": {},
        "integrals_norm_std": {},
    }

    raw_curve_data = []
    raw_integral_data = []
    norm_curve_data = []
    norm_integral_data = []

    print("[Compute] Start plotting raw reward curves...")
    plt.figure()
    for env_param in sorted(results_dict.keys()):
        result = results_dict[env_param]
        p_values = result["p_values"]
        mean_rewards_list = result["mean_rewards_list"]

        mean_rewards = np.mean(mean_rewards_list, axis=0)
        std_rewards = np.std(mean_rewards_list, axis=0)

        final_results["mean_rewards"][env_param] = mean_rewards
        final_results["std_rewards"][env_param] = std_rewards

        auc_list = [np.trapz(rewards, p_values) for rewards in mean_rewards_list]
        integral_mean = np.mean(auc_list)
        integral_std = np.std(auc_list)

        final_results["integrals"][env_param] = integral_mean
        final_results["integrals_std"][env_param] = integral_std

        raw_curve_data.append(pd.DataFrame({
            "p": p_values,
            "mean_reward": mean_rewards,
            "std_reward": std_rewards,
        }).assign(**{param_name: env_param}))

        raw_integral_data.append(pd.DataFrame({
            param_name: [env_param],
            "integral": [integral_mean],
            "std": [integral_std],
        }))

        plt.plot(p_values, mean_rewards, label=f"{param_name}={env_param}")
        plt.fill_between(
            p_values,
            mean_rewards - std_rewards,
            mean_rewards + std_rewards,
            alpha=0.2,
        )

        print(f"[Compute] Raw reward curve processed for {param_name}={env_param}")

    plt.xlabel("Mix Weight p")
    plt.ylabel("Mean Episode Reward")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    curve_path = os.path.join(save_path, "mix_policy_p_curve.png")
    plt.savefig(curve_path)
    plt.close()
    print(f"[Save Figure] Raw reward curve figure saved to {curve_path}")

    print("[Compute] Start plotting normalized reward curves...")
    plt.figure()
    for env_param in sorted(results_dict.keys()):
        result = results_dict[env_param]
        p_values = result["p_values"]
        mean_rewards_list = result["mean_rewards_list"]

        normalized_list = []
        for rewards in mean_rewards_list:
            min_r, max_r = rewards.min(), rewards.max()
            normalized = (rewards - min_r) / (max_r - min_r + 1e-8)
            normalized_list.append(normalized)

        normalized_mean = np.mean(normalized_list, axis=0)
        normalized_std = np.std(normalized_list, axis=0)

        final_results["normalized_mean_rewards"][env_param] = normalized_mean
        final_results["normalized_std_rewards"][env_param] = normalized_std

        auc_list_norm = [np.trapz(r, p_values) for r in normalized_list]
        integral_norm_mean = np.mean(auc_list_norm)
        integral_norm_std = np.std(auc_list_norm)

        final_results["integrals_norm"][env_param] = integral_norm_mean
        final_results["integrals_norm_std"][env_param] = integral_norm_std

        norm_curve_data.append(pd.DataFrame({
            "p": p_values,
            "mean_reward": normalized_mean,
            "std_reward": normalized_std,
        }).assign(**{param_name: env_param}))

        norm_integral_data.append(pd.DataFrame({
            param_name: [env_param],
            "integral": [integral_norm_mean],
            "std": [integral_norm_std],
        }))

        plt.plot(p_values, normalized_mean, label=f"{param_name}={env_param}")
        plt.fill_between(
            p_values,
            normalized_mean - normalized_std,
            normalized_mean + normalized_std,
            alpha=0.2,
        )

        print(f"[Compute] Normalized reward curve processed for {param_name}={env_param}")

    plt.xlabel("Mix Weight p")
    plt.ylabel("Normalized Mean Reward")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    norm_curve_path = os.path.join(save_path, "mix_policy_normalized_p_curve.png")
    plt.savefig(norm_curve_path)
    plt.close()
    print(f"[Save Figure] Normalized reward curve figure saved to {norm_curve_path}")

    # Plot raw integral bar chart
    plt.figure()
    env_params = list(sorted(results_dict.keys()))
    integrals = [final_results["integrals"][p] for p in env_params]
    integrals_std = [final_results["integrals_std"][p] for p in env_params]

    plt.bar([str(p) for p in env_params], integrals, yerr=integrals_std, capsize=5)
    plt.xlabel(param_name)
    plt.ylabel("Integral of Reward Curve")
    plt.grid(axis="y")
    plt.tight_layout()
    bar_path = os.path.join(save_path, "mix_policy_integral_bar_chart.png")
    plt.savefig(bar_path)
    plt.close()
    print(f"[Save Figure] Raw reward integral bar chart saved to {bar_path}")

    # Plot normalized integral bar chart
    plt.figure()
    integrals_norm = [final_results["integrals_norm"][p] for p in env_params]
    integrals_norm_std = [final_results["integrals_norm_std"][p] for p in env_params]

    plt.bar([str(p) for p in env_params], integrals_norm, yerr=integrals_norm_std, capsize=5)
    plt.xlabel(param_name)
    plt.ylabel("Integral of Normalized Reward Curve")
    plt.grid(axis="y")
    plt.tight_layout()
    norm_bar_path = os.path.join(save_path, "mix_policy_normalized_integral_bar_chart.png")
    plt.savefig(norm_bar_path)
    plt.close()
    print(f"[Save Figure] Normalized reward integral bar chart saved to {norm_bar_path}")

    # Save CSVs with log info
    raw_curve_csv = os.path.join(save_path, "mix_policy_curve_raw.csv")
    pd.concat(raw_curve_data).to_csv(raw_curve_csv, index=False)
    print(f"[Save CSV] Raw curve data saved to {raw_curve_csv}")

    raw_integral_csv = os.path.join(save_path, "mix_policy_integral_raw.csv")
    pd.concat(raw_integral_data).to_csv(raw_integral_csv, index=False)
    print(f"[Save CSV] Raw integral data saved to {raw_integral_csv}")

    norm_curve_csv = os.path.join(save_path, "mix_policy_curve_normalized.csv")
    pd.concat(norm_curve_data).to_csv(norm_curve_csv, index=False)
    print(f"[Save CSV] Normalized curve data saved to {norm_curve_csv}")

    norm_integral_csv = os.path.join(save_path, "mix_policy_integral_normalized.csv")
    pd.concat(norm_integral_data).to_csv(norm_integral_csv, index=False)
    print(f"[Save CSV] Normalized integral data saved to {norm_integral_csv}")

    print("[Compute] All results computation and visualization finished.")

    return final_results
