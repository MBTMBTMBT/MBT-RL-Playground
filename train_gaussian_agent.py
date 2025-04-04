import gc
import os

import numpy as np
import pandas as pd
import torch

from sbx import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv

from utils import ProgressBarCallback

from configs import lunarlander_configs, carracing_configs


# --------------- Main Training Loop ---------------
if __name__ == "__main__":
    lander_densities = np.linspace(3.0, 7.0, NUM_DENSITY_SETTINGS)
    # random.shuffle(lander_densities)

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
                buffer_size=1_200_000,
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
