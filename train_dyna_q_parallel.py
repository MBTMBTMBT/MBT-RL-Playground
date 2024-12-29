import random

import gymnasium as gym
import numpy as np
from tqdm import tqdm

from custom_mountain_car import CustomMountainCarEnv
from dyna_q import Discretizer, QCutTabularDynaQAgent

from dyna_q_task_configs import get_envs_discretizers_and_configs


def run_experiment(task_name: str, run_id: int):
    env, test_env, state_discretizer, action_discretizer, action_type, configs = get_envs_discretizers_and_configs(task_name)
    save_path = configs["save_path"] + f"-{run_id}"
    sample_strategies = ["explore_greedy", "greedy"]
    sample_strategy_distribution = configs["explore_policy_exploit_policy_ratio"]

    for init_group in configs["init_groups"].keys():
        group_save_path = save_path + f"-{init_group}"
        init_distribution = configs["init_groups"][init_group]

        agent = QCutTabularDynaQAgent(
            state_discretizer,
            action_discretizer,
            bonus_decay=configs["bonus_decay"],
            max_steps=configs["train_max_num_steps_per_episode"],
            rough_reward_resolution=configs["reward_resolution"],
        )

        with tqdm(total=configs["sample_steps"][-1], leave=False,) as pbar:
            sample_step_count = 0
            for sample_step in configs["sample_steps"]:
                num_steps_to_sample = sample_step - sample_step_count
                current_step = 0
                sample_strategy_step_count = {s: 0 for s in sample_strategies}

                state, _ = env.reset()
                if isinstance(state, int) or isinstance(state, float):
                    state = [state]
                encoded_state = agent.state_discretizer.encode_indices(
                    list(agent.state_discretizer.discretize(state)[1])
                )
                agent.transition_table_env.add_start_state(encoded_state)

                init_sample_strategy = random.choices(sample_strategies, weights=sample_strategy_distribution, k=1)[0]
                while current_step < num_steps_to_sample:
                    if random.random() < configs["explore_epsilon"]:
                        action_vec = agent.choose_action(state, strategy="random")
                    else:
                        action_vec = agent.choose_action(state, strategy=init_sample_strategy)
                    if action_type == "int":
                        action = action_vec.astype("int64")
                        action = action[0].item()
                    elif action_type == "float":
                        action = action_vec.astype("float32")
                    next_state, reward, done, truncated, _ = env.step(action)
                    if isinstance(next_state, int) or isinstance(next_state, float):
                        next_state = [next_state]

                    agent.update_from_env(
                        state, action_vec, reward, next_state, done, configs["explore_agent_lr"],
                        configs["explore_value_decay"], update_policy=False,
                    )
                    state = next_state

                    pbar.update(1)
                    current_step += 1
                    sample_strategy_step_count[init_sample_strategy] += 1

                    if done:
                        state, _ = env.reset()
                        if isinstance(state, int) or isinstance(state, float):
                            state = [state]
                        encoded_state = agent.state_discretizer.encode_indices(
                            list(agent.state_discretizer.discretize(state)[1])
                        )
                        agent.transition_table_env.add_start_state(encoded_state)
                        strategy_selection_dict = {}
                        for i, s in enumerate(sample_strategies):
                            if init_distribution[i] != 0:
                                strategy_selection_dict[s] = sample_strategy_step_count[s] / init_distribution[i]
                            else:
                                strategy_selection_dict[s] = np.inf
                        init_sample_strategy = min(strategy_selection_dict, key=strategy_selection_dict.get)
