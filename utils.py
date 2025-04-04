import os

import cv2
import gymnasium as gym
import imageio
import numpy as np
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

import custom_envs
from configs import carracing_configs, lunarlander_configs
from carracing_test import EVAL_EPISODES, N_ENVS, NUM_INIT_STATES, SAVE_PATH, NEAR_OPTIMAL_SCORE
from train_gaussian_agent import NUM_INIT_STATES


def make_lunarlander_env(
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


class EvalAndGifCallback(BaseCallback):
    def __init__(self, config, repeat, eval_interval, optimal_score, verbose=1):
        super().__init__(verbose)
        self.config = config
        self.repeat = repeat
        self.eval_interval = eval_interval
        self.optimal_score = optimal_score

        self.best_mean_reward = -np.inf
        self.step_reached_optimal = None
        self.records = []
        self.last_eval_step = 0

        self.eval_episodes = self.config["eval_episodes"]
        self.n_eval_envs = 1

        self.eval_env = SubprocVecEnv(
            [
                make_carracing_env(
                    map_seed=self.map_seed,
                    render_mode=None,
                    deterministic_init=False,
                    number_of_initial_states=NUM_INIT_STATES,
                    init_seed=i,
                )
                for i in range(self.n_eval_envs)
            ]
        )

        # Save path for best model
        self.best_model_path = os.path.join(
            SAVE_PATH,
            f"sac_carracing_mapseed_{self.map_seed}_repeat_{self.repeat}_best.zip",
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
                    f"[EvalCallback] Map Seed {self.map_seed} | Repeat {self.repeat} | "
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
                    NEAR_OPTIMAL_SCORE > 0 and mean_reward >= (NEAR_OPTIMAL_SCORE / 2)
                ) or NEAR_OPTIMAL_SCORE <= 0:
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
            SAVE_PATH,
            f"eval_log_mapseed_{self.map_seed}_repeat_{self.repeat}.csv",
        )
        df.to_csv(repeat_log_path, index=False)
        self.eval_env.close()

    def save_gif(self):
        frames = []
        initial_state_count = 8

        for idx in range(initial_state_count):
            single_env = DummyVecEnv(
                [
                    make_carracing_env(
                        map_seed=self.map_seed,
                        render_mode="rgb_array",
                        deterministic_init=False,
                        number_of_initial_states=initial_state_count,
                        init_seed=idx,
                    )
                ]
            )
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
            SAVE_PATH,
            f"carracing_mapseed_{self.map_seed}_repeat_{self.repeat}_all_initial_states.gif",
        )

        imageio.mimsave(gif_path, new_frames, duration=20, loop=0)
        print(f"[GIF Saved] {gif_path}")
