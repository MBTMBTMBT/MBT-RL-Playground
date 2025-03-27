import os
import matplotlib
matplotlib.use("Agg")  # hope to stop the crazy error from X server
import matplotlib.pyplot as plt
import gymnasium as gym
import pandas as pd
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

import custom_envs
from awesome_tabular_q_agent import TabularQAgent, EvalCallback

print(custom_envs.WELCOME)  # so that IDE wouldn't say this is not used


if __name__ == '__main__':

    # ---- Frozen Lake Example ---- #

    N_ENVS = 4
    TOTAL_TIMESTEPS = 25_000
    EVAL_INTERVAL = 500 * N_ENVS
    EVAL_EPISODES = 100 * N_ENVS
    SAVE_DIR = "./tabular_playground/frozenlake"
    MODEL_NAME = "tabular_q_agent"
    LR = 0.025
    GAMMA = 0.99
    BUFFER_SIZE = 1_000_000
    LEARNING_STARTS = 1_000
    MAX_T = 0.1
    MIN_T = MAX_T / 4
    TEMPERATURE_SCALE = 0.5
    UPDATE_STEPS = N_ENVS * 512
    UPDATE_INTERVAL = N_ENVS

    MAP_DESC = [
        "SFFFFFFF",
        "FFFFFFFF",
        "HFFFHFFH",
        "FFFFFFFF",
        "FFFHHFFF",
        "FFFFHGFF",
        "FFFFFFFF",
        "FFFFFFFF",
    ]

    os.makedirs(SAVE_DIR, exist_ok=True)

    def make_frozenlake_env():
        def _init():
            env = gym.make(
                "CustomFrozenLake-v1", desc=MAP_DESC, is_slippery=True, slipperiness=0.25, render_mode="rgb_array",
            )
            return env
        return _init

    train_env = SubprocVecEnv([make_frozenlake_env() for _ in range(N_ENVS)])
    eval_env = SubprocVecEnv([make_frozenlake_env() for _ in range(N_ENVS)])
    gif_env = DummyVecEnv([make_frozenlake_env() for _ in range(1)])

    agent = TabularQAgent(
        env=train_env,
        learning_rate=LR,
        gamma=GAMMA,
        buffer_size=BUFFER_SIZE,
        learning_starts=LEARNING_STARTS,
        max_temperature=MAX_T,
        min_temperature=MIN_T,
        temperature_scale=TEMPERATURE_SCALE,
        update_steps=UPDATE_STEPS,
        update_interval=UPDATE_INTERVAL,
        print_info=True,
    )

    eval_callback = EvalCallback(
        eval_env=eval_env,
        eval_interval=EVAL_INTERVAL,
        eval_episodes=EVAL_EPISODES,
        save_dir=SAVE_DIR,
        model_name=MODEL_NAME,
        verbose=1,
        gif_env=gif_env,
        gif_fps=5,
    )

    agent.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=eval_callback,
        reset_num_timesteps=True,
        progress_bar=True,
    )

    eval_log_files = [
        (os.path.join(SAVE_DIR, f"{MODEL_NAME}_evaluation_determined_log.csv"), "determined"),
        (os.path.join(SAVE_DIR, f"{MODEL_NAME}_evaluation_non_determined_log.csv"), "not_determined"),
    ]
    for eval_log_file, descriptor in eval_log_files:
        if os.path.exists(eval_log_file):
            df = pd.read_csv(eval_log_file)
            plt.figure(figsize=(10, 6))
            plt.plot(df["Timesteps"], df["MeanReward"], label="Mean Reward")
            plt.fill_between(
                df["Timesteps"],
                df["MeanReward"] - df["StdReward"],
                df["MeanReward"] + df["StdReward"],
                alpha=0.2,
            )
            plt.title("Evaluation Mean Reward Over Time")
            plt.xlabel("Timesteps")
            plt.ylabel("Mean Reward")
            plt.legend()
            plt.grid()
            plt.tight_layout()
            plt.savefig(os.path.join(SAVE_DIR, f"{MODEL_NAME}_{descriptor}.png"))
            plt.close()

    print("\n=== Loading Best Model ===")
    loaded_agent = TabularQAgent.load(
        path=os.path.join(SAVE_DIR, f"{MODEL_NAME}.zip"), env=eval_env, print_system_info=True,
    )

    mean_reward, std_reward = evaluate_policy(
        loaded_agent,
        eval_env,
        n_eval_episodes=EVAL_EPISODES,
        deterministic=True,
        render=False,
        warn=False,
    )

    print(
        f"\n=== Loaded Model Mean Test Reward (greedy) over {EVAL_EPISODES} episodes: {mean_reward:.2f} ± {std_reward:.2f} ==="
    )

    mean_reward, std_reward = evaluate_policy(
        loaded_agent,
        eval_env,
        n_eval_episodes=EVAL_EPISODES,
        deterministic=False,
        render=False,
        warn=False,
    )

    print(
        f"\n=== Loaded Model Mean Test Reward (softmax) over {EVAL_EPISODES} episodes: {mean_reward:.2f} ± {std_reward:.2f} ==="
    )

    eval_env.close()
    train_env.close()
    gif_env.close()


    # ---- Taxi Example ---- #

    N_ENVS = 4
    TOTAL_TIMESTEPS = 25_000
    EVAL_INTERVAL = 500 * N_ENVS
    EVAL_EPISODES = 100 * N_ENVS
    SAVE_DIR = "./tabular_playground/taxi"
    MODEL_NAME = "tabular_q_agent"
    LR = 0.025
    GAMMA = 0.99
    BUFFER_SIZE = 1_000_000
    LEARNING_STARTS = 1_000
    MAX_T = 1.0
    MIN_T = MAX_T / 4
    TEMPERATURE_SCALE = 0.5
    UPDATE_STEPS = N_ENVS * 512
    UPDATE_INTERVAL = N_ENVS

    os.makedirs(SAVE_DIR, exist_ok=True)

    def make_taxi_env():
        def _init():
            env = gym.make(
                "Taxi-v3", render_mode="rgb_array",
            )
            return env
        return _init

    train_env = SubprocVecEnv([make_taxi_env() for _ in range(N_ENVS)])
    eval_env = SubprocVecEnv([make_taxi_env() for _ in range(N_ENVS)])
    gif_env = DummyVecEnv([make_taxi_env() for _ in range(1)])

    agent = TabularQAgent(
        env=train_env,
        learning_rate=LR,
        gamma=GAMMA,
        buffer_size=BUFFER_SIZE,
        learning_starts=LEARNING_STARTS,
        max_temperature=MAX_T,
        min_temperature=MIN_T,
        temperature_scale=TEMPERATURE_SCALE,
        update_steps=UPDATE_STEPS,
        update_interval=UPDATE_INTERVAL,
        print_info=True,
    )

    eval_callback = EvalCallback(
        eval_env=eval_env,
        eval_interval=EVAL_INTERVAL,
        eval_episodes=EVAL_EPISODES,
        save_dir=SAVE_DIR,
        model_name=MODEL_NAME,
        verbose=1,
        gif_env=gif_env,
        gif_fps=5,
    )

    agent.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=eval_callback,
        reset_num_timesteps=True,
        progress_bar=True,
    )

    eval_log_files = [
        (os.path.join(SAVE_DIR, f"{MODEL_NAME}_evaluation_determined_log.csv"), "determined"),
        (os.path.join(SAVE_DIR, f"{MODEL_NAME}_evaluation_non_determined_log.csv"), "not_determined"),
    ]
    for eval_log_file, descriptor in eval_log_files:
        if os.path.exists(eval_log_file):
            df = pd.read_csv(eval_log_file)
            plt.figure(figsize=(10, 6))
            plt.plot(df["Timesteps"], df["MeanReward"], label="Mean Reward")
            plt.fill_between(
                df["Timesteps"],
                df["MeanReward"] - df["StdReward"],
                df["MeanReward"] + df["StdReward"],
                alpha=0.2,
            )
            plt.title("Evaluation Mean Reward Over Time")
            plt.xlabel("Timesteps")
            plt.ylabel("Mean Reward")
            plt.legend()
            plt.grid()
            plt.tight_layout()
            plt.savefig(os.path.join(SAVE_DIR, f"{MODEL_NAME}_{descriptor}.png"))
            plt.close()

    print("\n=== Loading Best Model ===")
    loaded_agent = TabularQAgent.load(
        path=os.path.join(SAVE_DIR, f"{MODEL_NAME}.zip"), env=eval_env, print_system_info=True,
    )

    mean_reward, std_reward = evaluate_policy(
        loaded_agent,
        eval_env,
        n_eval_episodes=EVAL_EPISODES,
        deterministic=True,
        render=False,
        warn=False,
    )

    print(
        f"\n=== Loaded Model Mean Test Reward (greedy) over {EVAL_EPISODES} episodes: {mean_reward:.2f} ± {std_reward:.2f} ==="
    )

    mean_reward, std_reward = evaluate_policy(
        loaded_agent,
        eval_env,
        n_eval_episodes=EVAL_EPISODES,
        deterministic=False,
        render=False,
        warn=False,
    )

    print(
        f"\n=== Loaded Model Mean Test Reward (softmax) over {EVAL_EPISODES} episodes: {mean_reward:.2f} ± {std_reward:.2f} ==="
    )

    eval_env.close()
    train_env.close()
    gif_env.close()
