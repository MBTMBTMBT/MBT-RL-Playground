import gymnasium as gym

from custom_mountain_car import CustomMountainCarEnv
from dyna_q import Discretizer


def get_envs_discretizers_and_configs(name: str):
    if name == "cartpole":
        save_path = "./experiments/DynaQ_Experiments/cartpole"
        env = gym.make("CartPole-v1", render_mode="rgb_array")
        test_env = gym.make("CartPole-v1", render_mode="rgb_array")
        state_discretizer = Discretizer(
            ranges=[(-2.4, 2.4), (-2, 2), (-0.25, 0.25), (-2, 2),],
            num_buckets=[13, 17, 17, 17],
            normal_params=[None, None, None, None,],
        )
        action_discretizer = Discretizer(
            ranges=[(0, 1),],
            num_buckets=[0],
            normal_params=[None, ],
        )
        action_type = "int"
        configs = {
            "save_path": save_path,
            "sample_steps": [int(0.5e6), int(1e6), int(1.5e6),],
            "explore_agent_lr": 0.1,
            "explore_bonus_decay": 0.9,
            "explore_epsilon": 0.25,
            "explore_strategy": "greedy",
            "reward_resolution": 0,
            "train_max_num_steps_per_episode": 500,
            "test_max_num_steps_per_episode": 500,
            "exploit_policy_reward_rate": 1e-3,
            "exploit_policy_training_per_num_steps": int(0.05e6),
            "exploit_policy_training_steps": int(0.25e6),
            "exploit_policy_test_per_num_steps": int(0.25e6),
            "exploit_policy_test_episodes": 64,
            "init_groups": {
                "real_start": (1.0, 0.0, 0.0),
                "random_init": (0.0, 1.0, 0.0),
                "real_start_random_init": (0.6, 0.4, 0.0),
                "q_cut": (0.6, 0.2, 0.2),
            },
            "q_cut_params": {
                "num_targets": 32,
                "min_cut_max_flow_search_space": 128,
                "q_cut_space": 16,
                "weighted_search": True,
                "init_state_reward_prob_below_threshold": 0.1,
                "quality_value_threshold": 1.0,
            },
            int(0.5e6): {
                "explore_policy_exploit_policy_ratio": (1.0, 0.0),
                "train_exploit_policy": True,
                "epsilon": 0.3,
                "train_exploit_strategy": "greedy",
                "test_exploit_policy": True,
                "test_exploit_strategy": "greedy",
            },
            int(1e6): {
                "explore_policy_exploit_policy_ratio": (0.5, 0.5),
                "train_exploit_policy": True,
                "epsilon": 0.2,
                "train_exploit_strategy": "greedy",
                "test_exploit_policy": True,
                "test_exploit_strategy": "greedy",
            },
            int(1.5e6): {
                "explore_policy_exploit_policy_ratio": (0.25, 0.75),
                "train_exploit_policy": True,
                "epsilon": 0.1,
                "train_exploit_strategy": "greedy",
                "test_exploit_policy": True,
                "test_exploit_strategy": "greedy",
            },
        }

    elif name == "mountain_car":
        save_path = "./experiments/DynaQ_Experiments/mountain_car"
        env = CustomMountainCarEnv(custom_gravity=0.005, render_mode="rgb_array")
        test_env = CustomMountainCarEnv(custom_gravity=0.005, render_mode="rgb_array")
        state_discretizer = Discretizer(
            ranges=[(-1.2, 0.6), (-0.07, 0.07), ],
            num_buckets=[65, 33],
            normal_params=[None, None],
        )
        action_discretizer = Discretizer(
            ranges=[(0, 2), ],
            num_buckets=[0],
            normal_params=[None, ],
        )
        action_type = "int"
        configs = {
            "save_path": save_path,
            "sample_steps": [int(7.5e6), int(10e6), int(15e6),],
            "explore_agent_lr": 0.1,
            "explore_bonus_decay": 0.9,
            "explore_epsilon": 0.25,
            "explore_strategy": "greedy",
            "reward_resolution": 0,
            "train_max_num_steps_per_episode": 200,
            "test_max_num_steps_per_episode": 200,
            "exploit_policy_reward_rate": 1e-3,
            "exploit_policy_training_per_num_steps": int(0.05e6),
            "exploit_policy_training_steps": int(0.25e6),
            "exploit_policy_test_per_num_steps": int(0.25e6),
            "exploit_policy_test_episodes": 64,
            "init_groups": {
                "real_start": (1.0, 0.0, 0.0),
                "random_init": (0.0, 1.0, 0.0),
                "real_start_random_init": (0.6, 0.4, 0.0),
                "q_cut": (0.6, 0.2, 0.2),
            },
            "q_cut_params": {
                "num_targets": 32,
                "min_cut_max_flow_search_space": 128,
                "q_cut_space": 16,
                "weighted_search": True,
                "init_state_reward_prob_below_threshold": 0.1,
                "quality_value_threshold": 1.0,
            },
            int(7.5e6): {
                "explore_policy_exploit_policy_ratio": (1.0, 0.0),
                "train_exploit_policy": False,
                "test_exploit_policy": False,
            },
            int(10e6): {
                "explore_policy_exploit_policy_ratio": (0.5, 0.5),
                "train_exploit_policy": True,
                "epsilon": 0.25,
                "train_exploit_strategy": "greedy",
                "test_exploit_policy": True,
                "test_exploit_strategy": "greedy",
            },
            int(15e6): {
                "explore_policy_exploit_policy_ratio": (0.25, 0.75),
                "train_exploit_policy": True,
                "epsilon": 0.1,
                "train_exploit_strategy": "greedy",
                "test_exploit_policy": True,
                "test_exploit_strategy": "greedy",
            },
        }

    elif name == "lunarlander":
        save_path = "./experiments/DynaQ_Experiments/lunarlander"
        env = gym.make("LunarLander-v3", render_mode="rgb_array", continuous=True, )
        test_env = gym.make("LunarLander-v3", render_mode="rgb_array", continuous=True, )
        state_discretizer = Discretizer(
            ranges=[
                (-1.5, 1.5), (-1.5, 1.5), (-5.0, 5.0), (-5.0, 5.0),
                (-3.14, 3.14), (-5.0, 5.0), (0, 1), (0, 1),
            ],
            num_buckets=[17, 17, 49, 49, 25, 49, 0, 0,],
            normal_params=[None, None, None, None, None, None, None, None,],
        )
        action_discretizer = Discretizer(
            ranges=[(-1, 1), (-1, 1)],
            num_buckets=[11, 11],
            normal_params=[None, None],
        )
        action_type = "float"

    elif name == "acrobot":
        save_path = "./experiments/DynaQ_Experiments/acrobot"
        env = gym.make("Acrobot-v1", render_mode="rgb_array")
        test_env = gym.make("Acrobot-v1", render_mode="rgb_array")
        state_discretizer = Discretizer(
            ranges=[
                (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0),
                (-6.0, 6.0), (-12.0, 12.0),
            ],
            num_buckets=[17, 17, 17, 17, 17, 17,],
            normal_params=[None, None, None, None, None, None,],
        )
        action_discretizer = Discretizer(
            ranges=[(0, 2), ],
            num_buckets=[0],
            normal_params=[None,],
        )
        action_type = "int"

    elif name =="pendulum":
        save_path = "./experiments/DynaQ_Experiments/pendulum"
        env = gym.make("Pendulum-v1", render_mode="rgb_array",)
        test_env = gym.make("Pendulum-v1", render_mode="rgb_array",)
        state_discretizer = Discretizer(
            ranges=[(-1.0, 1.0), (-1.0, 1.0), (-8.0, 8.0), ],
            num_buckets=[33, 33, 65],
            normal_params=[None, None, None],
        )
        action_discretizer = Discretizer(
            ranges=[(-2.0, 2.0),],
            num_buckets=[17],
            normal_params=[None,],
        )
        action_type = "float"

    elif name == "texi":
        save_path = "./experiments/DynaQ_Experiments/texi"
        env = gym.make("Taxi-v3", render_mode="rgb_array", )
        test_env = gym.make("Taxi-v3", render_mode="rgb_array", )
        state_discretizer = Discretizer(
            ranges=[(0, 499)],
            num_buckets=[0],
            normal_params=[None,],
        )
        action_discretizer = Discretizer(
            ranges=[(0, 5), ],
            num_buckets=[0],
            normal_params=[None, ],
        )
        action_type = "int"

    else:
        raise ValueError(f"Invalid environment name: {name}.")

    return env, test_env, state_discretizer, action_discretizer, action_type, configs
