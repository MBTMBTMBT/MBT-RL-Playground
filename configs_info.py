import gymnasium as gym
from dyna_q_modernized import Discretizer


def get_envs_discretizers_and_configs(name: str, env_idx: int, configs_only=False):
    if name == "frozen_lake-44":
        save_path = "./experiments/env-info/frozen_lake-44/frozen_lake-44"
        envs = [
            gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False, render_mode="rgb_array"),
            gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True, render_mode="rgb_array"),
        ]
        test_envs = [
            gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False, render_mode="rgb_array"),
            gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True, render_mode="rgb_array"),
        ]
        env_descs = [
            "44-not-slippery",
            "44-slippery",
        ]
        state_discretizer = Discretizer(
            ranges=[(0, envs[0].observation_space.n)],
            num_buckets=[0],
            normal_params=[None, ],
        )
        action_discretizer = Discretizer(
            ranges=[(0, 3), ],
            num_buckets=[0],
            normal_params=[None, ],
        )
        configs = {
            "num_envs": len(env_descs),
            "use_deep_agent": False,
            "save_path": save_path,
            "explore_agent_lr": 0.1,
            "train_max_num_steps_per_episode": 100,
            "exploit_agent_lr": 0.1,
            "exploit_softmax_temperature": 1.0,
            "exploit_policy_reward_rate": 1,
            "exploit_value_decay": 0.99,
            "exploit_policy_training_steps": int(25e3),
            "exploit_policy_test_per_num_steps": int(0.25e3),
            "exploit_policy_test_episodes": 25,
            "exploit_policy_eval_episodes": 200,
            "save_per_num_steps": int(5e3),
            "save_mdp_graph": False,
            "print_training_info": False,
            "initialization_distribution": (0.5, 0.5),
        }

    elif name == "frozen_lake-88":
        save_path = "./experiments/env-info/frozen_lake-88/frozen_lake-88"
        envs = [
            gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=False, render_mode="rgb_array"),
            gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=True, render_mode="rgb_array"),
        ]
        test_envs = [
            gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=False, render_mode="rgb_array"),
            gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=True, render_mode="rgb_array"),
        ]
        env_descs = [
            "88-not-slippery",
            "88-slippery",
        ]
        state_discretizer = Discretizer(
            ranges=[(0, envs[0].observation_space.n)],
            num_buckets=[0],
            normal_params=[None, ],
        )
        action_discretizer = Discretizer(
            ranges=[(0, 3), ],
            num_buckets=[0],
            normal_params=[None, ],
        )
        configs = {
            "num_envs": len(env_descs),
            "use_deep_agent": False,
            "save_path": save_path,
            "explore_agent_lr": 0.1,
            "train_max_num_steps_per_episode": 100,
            "exploit_agent_lr": 0.1,
            "exploit_softmax_temperature": 1.0,
            "exploit_policy_reward_rate": 1,
            "exploit_value_decay": 0.99,
            "exploit_policy_training_steps": int(250e3),
            "exploit_policy_test_per_num_steps": int(2.5e3),
            "exploit_policy_test_episodes": 25,
            "exploit_policy_eval_episodes": 200,
            "save_per_num_steps": int(50e3),
            "save_mdp_graph": False,
            "print_training_info": False,
            "initialization_distribution": (0.5, 0.5),
        }

    elif name == "frozen_lake-custom":
        save_path = "./experiments/env-info/frozen_lake-custom/frozen_lake-custom"
        envs = [
            gym.make(
                'FrozenLake-v1',
                desc=[
                    "SFFFFFFF",
                    "FFFFFFFF",
                    "FFFFFFFF",
                    "FFFFFFFF",
                    "FFFFFFFF",
                    "FFFFFGFF",
                    "FFFFFFFF",
                    "FFFFFFFF",
                ],
                map_name=None,
                is_slippery=False,
                render_mode="rgb_array",
            ),
            gym.make(
                'FrozenLake-v1',
                desc=[
                    "SFFFFFFF",
                    "FFHFFFFF",
                    "FFFFFFHF",
                    "FFFHFFFF",
                    "FFFFFFFF",
                    "FHFFFGFF",
                    "FFFFFFFF",
                    "FFFFFFFF",
                ],
                map_name=None,
                is_slippery=False,
                render_mode="rgb_array",
            ),
            gym.make(
                'FrozenLake-v1',
                desc=[
                    "SFFFFHFF",
                    "FFHFFFFF",
                    "FFFFFFHF",
                    "FFFHFFFF",
                    "FFFFFHFF",
                    "FHFHFGFF",
                    "FFFFFFHF",
                    "FFFFFFFF",
                ],
                map_name=None,
                is_slippery=False,
                render_mode="rgb_array",
            ),
            gym.make(
                'FrozenLake-v1',
                desc=[
                    "SFFFFHFF",
                    "FFHFFFFF",
                    "GFFFFFHF",
                    "FFFHFFFF",
                    "FFFFHHFF",
                    "FHFHHGFF",
                    "FFFFHFHF",
                    "FFFFFFFF",
                ],
                map_name=None,
                is_slippery=False,
                render_mode="rgb_array",
            ),
        ]
        test_envs = [
            gym.make(
                'FrozenLake-v1',
                desc=[
                    "SFFFFFFF",
                    "FFFFFFFF",
                    "FFFFFFFF",
                    "FFFFFFFF",
                    "FFFFFFFF",
                    "FFFFFGFF",
                    "FFFFFFFF",
                    "FFFFFFFF",
                ],
                map_name=None,
                is_slippery=False,
                render_mode="rgb_array",
            ),
            gym.make(
                'FrozenLake-v1',
                desc=[
                    "SFFFFFFF",
                    "FFHFFFFF",
                    "FFFFFFHF",
                    "FFFHFFFF",
                    "FFFFFFFF",
                    "FHFFFGFF",
                    "FFFFFFFF",
                    "FFFFFFFF",
                ],
                map_name=None,
                is_slippery=False,
                render_mode="rgb_array",
            ),
            gym.make(
                'FrozenLake-v1',
                desc=[
                    "SFFFFHFF",
                    "FFHFFFFF",
                    "FFFFFFHF",
                    "FFFHFFFF",
                    "FFFFFHFF",
                    "FHFHFGFF",
                    "FFFFFFHF",
                    "FFFFFFFF",
                ],
                map_name=None,
                is_slippery=False,
                render_mode="rgb_array",
            ),
            gym.make(
                'FrozenLake-v1',
                desc=[
                    "SFFFFHFF",
                    "FFHFFFFF",
                    "GFFFFFHF",
                    "FFFHFFFF",
                    "FFFFHHFF",
                    "FHFHHGFF",
                    "FFFFHFHF",
                    "FFFFFFFF",
                ],
                map_name=None,
                is_slippery=False,
                render_mode="rgb_array",
            ),
        ]
        env_descs = [
            "difficulty-1",
            "difficulty-2",
            "difficulty-3",
            "difficulty-4",
        ]
        state_discretizer = Discretizer(
            ranges=[(0, envs[0].observation_space.n)],
            num_buckets=[0],
            normal_params=[None, ],
        )
        action_discretizer = Discretizer(
            ranges=[(0, 3), ],
            num_buckets=[0],
            normal_params=[None, ],
        )
        configs = {
            "num_envs": len(env_descs),
            "use_deep_agent": False,
            "save_path": save_path,
            "explore_agent_lr": 0.1,
            "train_max_num_steps_per_episode": 100,
            "exploit_agent_lr": 0.1,
            "exploit_softmax_temperature": 1.0,
            "exploit_policy_reward_rate": 1,
            "exploit_value_decay": 0.99,
            "exploit_policy_training_steps": int(250e3),
            "exploit_policy_test_per_num_steps": int(2.5e3),
            "exploit_policy_test_episodes": 25,
            "exploit_policy_eval_episodes": 200,
            "save_per_num_steps": int(50e3),
            "save_mdp_graph": False,
            "print_training_info": False,
            "initialization_distribution": (0.5, 0.5),
        }

    else:
        raise ValueError(f"Invalid environment name: {name}.")

    if configs_only:
        return configs
    return envs[env_idx], test_envs[env_idx], env_descs[env_idx], state_discretizer, action_discretizer, configs
