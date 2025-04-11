import gc
import os

import numpy as np
import pandas as pd
import torch

from gaussian_agent import SACJax as SAC
from stable_baselines3.common.vec_env import SubprocVecEnv

from utils import (
    ProgressBarCallback,
    make_carracing_env,
    make_lunarlander_env,
    EvalAndGifCallback,
    plot_eval_results,
    plot_optimal_step_bar_chart,
)

from configs import lunarlander_config, carracing_config


# --------------- Main Training Loop ---------------
if __name__ == "__main__":
    configs = [
        lunarlander_config,
        carracing_config,
    ]

    for config in configs:
        env_type = config["env_type"]
        summary_results = []
        curve_results = {}

        # Decide environment parameters
        if env_type == "lunarlander":
            env_params = np.linspace(3.0, 7.0, config["num_densities"]).tolist()
        elif env_type == "carracing":
            env_params = list(range(config["num_seeds"]))
        else:
            raise ValueError("Invalid environment type.")

        for env_param in env_params:
            repeat_results = []
            reward_curves = []

            for run in range(config["n_repeat"]):
                print(
                    f"\n--- Repeat {run + 1}/{config['n_repeat']} for Env Param = {env_param} ---"
                )

                if env_type == "lunarlander":
                    train_env = SubprocVecEnv(
                        [
                            make_lunarlander_env(
                                lander_density=env_param,
                                render_mode=None,
                                deterministic_init=False,
                                number_of_initial_states=config["num_init_states"],
                                init_seed=None,
                            )
                            for _ in range(config["n_envs"])
                        ]
                    )
                elif env_type == "carracing":
                    train_env = SubprocVecEnv(
                        [
                            make_carracing_env(
                                map_seed=env_param,
                                render_mode=None,
                                deterministic_init=False,
                                number_of_initial_states=config["num_init_states"],
                                init_seed=None,
                            )
                            for _ in range(config["n_envs"])
                        ]
                    )

                model = SAC(
                    "MlpPolicy",
                    train_env,
                    verbose=0,
                    learning_rate=2e-4,
                    buffer_size=config["train_steps"],
                    learning_starts=5_000,
                    batch_size=256,
                    tau=0.005,
                    train_freq=config["n_envs"],
                    gradient_steps=config["n_envs"] * 8,
                    ent_coef="auto",
                    policy_kwargs=dict(net_arch=[256, 256]),
                )

                eval_callback = EvalAndGifCallback(
                    config=config,
                    env_param=env_param,
                    n_eval_envs=config["n_envs"]
                    if env_type == "lunarlander"
                    else 1,
                    run_idx=run + 1,
                    eval_interval=config["eval_interval"],
                    optimal_score=config["near_optimal_score"],
                    verbose=1,
                    temp_dir=".",
                )

                progress_callback = ProgressBarCallback(
                    total_timesteps=config["train_steps"]
                )

                model.learn(
                    total_timesteps=config["train_steps"],
                    callback=[eval_callback, progress_callback],
                )

                repeat_results.append(
                    {
                        "EnvParam": env_param,
                        "Repeat": run + 1,
                        "OptimalStep": eval_callback.step_reached_optimal
                        or config["train_steps"],
                        "BestScore": eval_callback.best_mean_reward,
                    }
                )

                df_repeat = pd.DataFrame(
                    eval_callback.records["reward"],
                    columns=["Timesteps", "MeanReward", "StdReward"],
                )
                reward_curves.append(df_repeat)

                print(
                    f"\n--- Cleanup after Repeat {run + 1} for Env Param {env_param} ---"
                )
                train_env.close()
                del model
                gc.collect()
                torch.cuda.empty_cache()

            # Calculate summary
            best_scores = [res["BestScore"] for res in repeat_results]
            optimal_steps = [res["OptimalStep"] for res in repeat_results]

            summary_results.append(
                {
                    "LanderDensity"
                    if env_type == "lunarlander"
                    else "MapSeed": env_param,
                    "BestScoreMean": np.mean(best_scores),
                    "BestScoreStd": np.std(best_scores),
                    "OptimalStepMean": np.mean(optimal_steps),
                    "OptimalStepStd": np.std(optimal_steps),
                }
            )

            mean_rewards = np.mean([df["MeanReward"] for df in reward_curves], axis=0)
            std_rewards = np.std([df["MeanReward"] for df in reward_curves], axis=0)
            timesteps = reward_curves[0]["Timesteps"]

            curve_results[env_param] = {
                "Timesteps": timesteps,
                "MeanReward": mean_rewards,
                "StdReward": std_rewards,
            }

            # Save reward curves per env_param
            df_curve = pd.DataFrame(
                {
                    "Timesteps": timesteps,
                    "MeanReward": mean_rewards,
                    "StdReward": std_rewards,
                }
            )
            csv_curve_path = os.path.join(
                config["save_path"],
                f"mean_std_{'density' if env_type == 'lunarlander' else 'mapseed'}_{env_param}.csv",
            )
            df_curve.to_csv(csv_curve_path, index=False)

        # Save summary results
        df_summary = pd.DataFrame(summary_results)
        csv_summary_path = os.path.join(
            config["save_path"], "summary_results_mean_std.csv"
        )
        df_summary.to_csv(csv_summary_path, index=False)

        print("\n===== All Training Completed =====")
        print(df_summary)

        # Plot results
        plot_eval_results(config, curve_results, config["save_path"])
        plot_optimal_step_bar_chart(config, summary_results, config["save_path"])
