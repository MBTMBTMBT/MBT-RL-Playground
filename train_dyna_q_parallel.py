import random

import gymnasium as gym
from tqdm import tqdm

from custom_mountain_car import CustomMountainCarEnv
from dyna_q import Discretizer

from dyna_q_task_configs import get_envs_discretizers_and_configs


def run_experiment(task_name: str, run_id: int):
    env, test_env, state_discretizer, action_discretizer, action_type, configs = get_envs_discretizers_and_configs(task_name)
    save_path = configs["save_path"] + f"-{run_id}"
    sample_strategies = ["explore_greedy", "greedy"]
    sample_strategy_distribution = configs["explore_policy_exploit_policy_ratio"]

    for init_group in configs["init_groups"].keys():
        group_save_path = save_path + f"-{init_group}"
        init_distribution = configs["init_groups"][init_group]

        with tqdm(total=configs["sample_steps"][-1], leave=False,) as pbar:
            sample_step_count = 0
            for sample_step in configs["sample_steps"]:
                num_steps_to_sample = sample_step - sample_step_count
                init_sample_strategy = random.choices(sample_strategies, weights=sample_strategy_distribution, k=1)[0]
                sample_strategy_step_count = {s: 0 for s in sample_strategies}
