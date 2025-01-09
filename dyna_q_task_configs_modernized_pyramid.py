import gymnasium as gym

from custom_mountain_car import CustomMountainCarEnv
from dyna_q_modernized import Discretizer


def get_envs_discretizers_and_configs(name: str, configs_only=False):
    if name == "cartpole":
        save_path = "./experiments/DynaQ/pyramid-cartpole/cartpole"
        env = gym.make("CartPole-v1", render_mode="rgb_array")
        test_env = gym.make("CartPole-v1", render_mode="rgb_array")
        state_discretizers = [
            Discretizer(
                ranges=[(-2.4, 2.4), (-2, 2), (-0.25, 0.25), (-2, 2), ],
                num_buckets=[3, 7, 7, 7],
                normal_params=[None, None, None, None, ],
            ),
            Discretizer(
                ranges=[(-2.4, 2.4), (-2, 2), (-0.25, 0.25), (-2, 2), ],
                num_buckets=[9, 13, 13, 13],
                normal_params=[None, None, None, None, ],
            ),
            Discretizer(
                ranges=[(-2.4, 2.4), (-2, 2), (-0.25, 0.25), (-2, 2), ],
                num_buckets=[13, 17, 17, 17],
                normal_params=[None, None, None, None, ],
            ),
            Discretizer(
                ranges=[(-2.4, 2.4), (-2, 2), (-0.25, 0.25), (-2, 2), ],
                num_buckets=[17, 23, 23, 23],
                normal_params=[None, None, None, None, ],
            ),
        ]
        action_discretizers = [
            Discretizer(
                ranges=[(0, 1), ],
                num_buckets=[0],
                normal_params=[None, ],
            ),
            Discretizer(
                ranges=[(0, 1), ],
                num_buckets=[0],
                normal_params=[None, ],
            ),
            Discretizer(
                ranges=[(0, 1), ],
                num_buckets=[0],
                normal_params=[None, ],
            ),
            Discretizer(
                ranges=[(0, 1), ],
                num_buckets=[0],
                normal_params=[None, ],
            ),
        ]
        configs = {
            "save_path": save_path,
            "explore_agent_lr": 0.1,
            "explore_value_decay": 0.99,
            "explore_bonus_decay": 0.9,
            "explore_policy_training_per_num_steps": int(0.5e3),
            "explore_policy_training_steps": int(5e3),
            "explore_epsilon": 0.25,
            "explore_strategy": "greedy",
            "reward_resolution": 0,
            "train_max_num_steps_per_episode": 500,
            "exploit_agent_lr": 2.5e-4,
            "exploit_softmax_temperature": 0.5,
            "exploit_policy_reward_rate": 1e-1,
            "exploit_value_decay": 0.99,
            "exploit_policy_training_per_num_steps": int(2.5e3),
            "exploit_policy_training_steps": int(5e3),
            "exploit_policy_test_per_num_steps": int(2.5e3),
            "exploit_policy_test_episodes": 256,
            "save_per_num_steps": int(500e3),
            "save_mdp_graph": False,
            "print_training_info": False,

            int(10e3): {
                "explore_policy_exploit_policy_ratio": (1.0, 0.0),
                "train_exploit_policy": False,
                "test_exploit_policy": False,
            },
            int(15e3): {
                "explore_policy_exploit_policy_ratio": (0.75, 0.25),
                "train_exploit_policy": True,
                "test_exploit_policy": True,
                "test_exploit_strategy": "greedy",
                "pyramid_index": 0,
            },
            int(25e3): {
                "explore_policy_exploit_policy_ratio": (0.5, 0.5),
                "train_exploit_policy": True,
                "test_exploit_policy": True,
                "test_exploit_strategy": "greedy",
                "pyramid_index": 1,
            },
            int(35e3): {
                "explore_policy_exploit_policy_ratio": (0.25, 0.75),
                "train_exploit_policy": True,
                "test_exploit_policy": True,
                "test_exploit_strategy": "greedy",
                "pyramid_index": 2,
            },
            int(50e3): {
                "explore_policy_exploit_policy_ratio": (0.25, 0.75),
                "train_exploit_policy": True,
                "test_exploit_policy": True,
                "test_exploit_strategy": "greedy",
                "pyramid_index": 3,
            },
            int(70e3): {
                "explore_policy_exploit_policy_ratio": (0.25, 0.75),
                "train_exploit_policy": True,
                "test_exploit_policy": True,
                "test_exploit_strategy": "greedy",
                "pyramid_index": -1,
            },
        }
    else:
        raise ValueError(f"Invalid environment name: {name}.")

    if configs_only:
        return configs
    return env, test_env, state_discretizers, action_discretizers, configs
