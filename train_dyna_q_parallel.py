import gymnasium as gym

from custom_mountain_car import CustomMountainCarEnv
from dyna_q import Discretizer


def get_envs_discretizers_and_configs(name: str):
    if name == "cartpole":
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
            "sample_steps": [int(0.5e6), int(1e6), int(1.5e6),],
            int(0.5e6): {
                "explore_policy_exploit_policy_ratio": (1.0, 0.0),
                "train_exploit_policy": False,
            },
            int(1e6): {
                "explore_policy_exploit_policy_ratio": (0.5, 0.5),
                "train_exploit_policy": True,
                "epsilon": 0.25,
                "exploit_strategy": "greedy",
            },
            int(1.5e6): {
                "explore_policy_exploit_policy_ratio": (0.25, 0.75),
                "train_exploit_policy": True,
                "epsilon": 0.1,
                "exploit_strategy": "greedy",
            },
        }
        _configs = {
            "num_targets": 16,
            "min_cut_max_flow_search_space": 256,
            "q_cut_space": 32,
            "weighted_search": True,
            "init_state_reward_prob_below_threshold": 0.1,
            "quality_value_threshold": 1.0,
        }

    elif name == "mountain_car":
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

    elif name == "lunarlander":
        env = gym.make("LunarLander-v3", render_mode="rgb_array")
        test_env = gym.make("LunarLander-v3", render_mode="rgb_array")
        state_discretizer = Discretizer(
            ranges=[
                (-1.5, 1.5), (-1.5, 1.5), (-5.0, 5.0), (-5.0, 5.0),
                (-3.14, 3.14), (-5.0, 5.0), (0, 1), (0, 1),
            ],
            num_buckets=[17, 17, 49, 49, 25, 49, 0, 0,],
            normal_params=[None, None, None, None, None, None, None, None,],
        )
        action_discretizer = Discretizer(
            ranges=[(0, 3), ],
            num_buckets=[0],
            normal_params=[None, ],
        )
        action_type = "int"

    elif name == "acrobot":
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

    return env, test_env, state_discretizer, action_discretizer, action_type


