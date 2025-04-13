import gc
import os

import numpy as np
import pandas as pd
import torch

from gaussian_agent import SACJax as SAC, MixPolicySAC
from stable_baselines3.common.vec_env import SubprocVecEnv

from utils import (
    ProgressBarCallback,
    make_carracing_env,
    make_lunarlander_env,
    EvalAndGifCallback,
    plot_eval_results,
    plot_optimal_step_bar_chart,
    evaluate_mix_policy_agent,
    compute_and_plot_mix_policy_results,
)

from configs import lunarlander_config, carracing_config


# --------------- Main Training Loop ---------------
if __name__ == "__main__":
    configs = [
        lunarlander_config,
        carracing_config,
    ]

    # make model paths
    model_paths = {}
    for config in configs:
        env_type = config["env_type"]
        # Decide environment parameters
        if env_type == "lunarlander":
            env_params = np.linspace(config["density_start"], config["density_stop"], config["num_densities"]).tolist()
        elif env_type == "carracing":
            env_params = list(range(config["num_seeds"]))
        else:
            raise ValueError("Invalid environment type.")
        for env_param in env_params:
            for run in range(config["n_repeat"]):
                best_model_path = os.path.join(
                    config["save_path"],
                    f"sac_env_param_{env_param}_run_{run + 1}_best.zip",
                )
                model_paths[(env_type, env_param, run + 1)] = best_model_path
    print("All model paths:")
    print(model_paths)

    for config in configs:
        env_type = config["env_type"]
        summary_results = []
        curve_results = {}

        # Decide environment parameters
        if env_type == "lunarlander":
            env_params = np.linspace(config["density_start"], config["density_stop"], config["num_densities"]).tolist()
        elif env_type == "carracing":
            env_params = list(range(config["num_seeds"]))
        else:
            raise ValueError("Invalid environment type.")

        for env_param in env_params:
            repeat_results = []
            all_repeat_records = []

            for run in range(config["n_repeat"]):
                print(
                    f"\n--- Repeat {run + 1}/{config['n_repeat']} for Env Param = {env_param} ---"
                )

                # Prepare training environment
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
                    learning_rate=1e-4,
                    buffer_size=config["train_steps"],
                    learning_starts=5_000,
                    batch_size=256,
                    tau=0.005,
                    train_freq=config["n_envs"],
                    gradient_steps=config["n_envs"] * 8,
                    ent_coef="auto",
                    policy_kwargs=dict(net_arch=[256, 256, 256]),
                )

                eval_callback = EvalAndGifCallback(
                    config=config,
                    env_param=env_param,
                    n_eval_envs=config["n_envs"],
                    run_idx=run + 1,
                    eval_interval=config["eval_interval"],
                    optimal_score=config["near_optimal_score"],
                    verbose=1,
                    temp_dir=".",
                    use_default_policy=True,
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

                all_repeat_records.append(eval_callback.records)

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

            # Merge all repeat records for this env_param
            keys = all_repeat_records[0].keys()
            timesteps = [x[0] for x in all_repeat_records[0]["reward"]]

            curve_results[env_param] = {"Timesteps": timesteps}

            for key in keys:
                values_per_repeat = np.array(
                    [[v[1] for v in records[key]] for records in all_repeat_records]
                )

                mean_values = values_per_repeat.mean(axis=0)
                std_values = values_per_repeat.std(axis=0)

                curve_results[env_param][f"{key}_mean"] = mean_values
                curve_results[env_param][f"{key}_std"] = std_values

            df_curve = pd.DataFrame(curve_results[env_param])
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

        # Evaluate MixPolicy Performance
        mix_results = {}

        for env_param in env_params:
            all_repeat_rewards = []

            for run in range(config["n_repeat"]):
                print(f"Evaluating with param: {env_param} at run {run + 1}...")

                if config["env_type"] == "lunarlander":
                    test_env = SubprocVecEnv(
                        [
                            make_lunarlander_env(
                                lander_density=env_param,
                                render_mode=None,
                                deterministic_init=True,
                                number_of_initial_states=config["num_init_states"],
                                init_seed=i,
                            )
                            for i in range(config["n_envs"])
                        ]
                    )
                elif config["env_type"] == "carracing":
                    test_env = SubprocVecEnv(
                        [
                            make_carracing_env(
                                map_seed=env_param,
                                render_mode=None,
                                deterministic_init=False,
                                number_of_initial_states=config["num_init_states"],
                                init_seed=i,
                            )
                            for i in range(config["n_envs"])
                        ]
                    )
                else:
                    test_env = None
                model_path = model_paths[(env_type, env_param, run + 1)]
                model = SAC.load(model_path, env=test_env)
                mix_agent = MixPolicySAC(model)

                mean_rewards = evaluate_mix_policy_agent(
                    mix_agent,
                    test_env,
                    total_episodes=config["eval_episodes"],
                    num_p_values=20,
                )
                all_repeat_rewards.append(mean_rewards)

            all_repeat_rewards = np.array(all_repeat_rewards)  # (n_repeat, num_p_values)
            mean_rewards = all_repeat_rewards.mean(axis=0)
            std_rewards = all_repeat_rewards.std(axis=0)
            p_values = np.linspace(0.0, 1.0, 20)

            mix_results[env_param] = {
                "p_values": p_values,
                "mean_rewards": mean_rewards,
                "std_rewards": std_rewards,
            }

            df = pd.DataFrame({
                "p": p_values,
                "mean_reward": mean_rewards,
                "std_reward": std_rewards,
            })
            csv_path = os.path.join(
                config["save_path"],
                f"mix_policy_curve_{'density' if env_type == 'lunarlander' else 'mapseed'}_{env_param}.csv",
            )
            df.to_csv(csv_path, index=False)

        # Plot MixPolicy Evaluation Results
        rst = compute_and_plot_mix_policy_results(config, mix_results, config["save_path"])
