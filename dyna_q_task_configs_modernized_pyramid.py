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
                num_buckets=[17, 25, 25, 25],
                normal_params=[None, None, None, None, ],
            ),
            Discretizer(
                ranges=[(-2.4, 2.4), (-2, 2), (-0.25, 0.25), (-2, 2), ],
                num_buckets=[25, 65, 65, 65],
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
            int(20e3): {
                "explore_policy_exploit_policy_ratio": (0.75, 0.25),
                "train_exploit_policy": True,
                "test_exploit_policy": True,
                "test_exploit_strategy": "greedy",
                "pyramid_index": 0,
            },
            int(30e3): {
                "explore_policy_exploit_policy_ratio": (0.5, 0.5),
                "train_exploit_policy": True,
                "test_exploit_policy": True,
                "test_exploit_strategy": "greedy",
                "pyramid_index": 1,
            },
            int(50e3): {
                "explore_policy_exploit_policy_ratio": (0.25, 0.75),
                "train_exploit_policy": True,
                "test_exploit_policy": True,
                "test_exploit_strategy": "greedy",
                "pyramid_index": 2,
            },
            int(75e3): {
                "explore_policy_exploit_policy_ratio": (0.25, 0.75),
                "train_exploit_policy": True,
                "test_exploit_policy": True,
                "test_exploit_strategy": "greedy",
                "pyramid_index": 3,
            },
            int(100e3): {
                "explore_policy_exploit_policy_ratio": (0.25, 0.75),
                "train_exploit_policy": True,
                "test_exploit_policy": True,
                "test_exploit_strategy": "greedy",
                "pyramid_index": -1,
            },
        }

    elif name == "mountain_car":
        save_path = "./experiments/DynaQ/pyramid-mountain_car/mountain_car"
        env = CustomMountainCarEnv(custom_gravity=0.005, render_mode="rgb_array")
        test_env = CustomMountainCarEnv(custom_gravity=0.005, render_mode="rgb_array")
        state_discretizers = [
            # Discretizer(
            #     ranges=[(-1.2, 0.6), (-0.07, 0.07), ],
            #     num_buckets=[9, 5],
            #     normal_params=[None, None],
            # ),
            Discretizer(
                ranges=[(-1.2, 0.6), (-0.07, 0.07), ],
                num_buckets=[17, 9],
                normal_params=[None, None],
            ),
            Discretizer(
                ranges=[(-1.2, 0.6), (-0.07, 0.07), ],
                num_buckets=[33, 17],
                normal_params=[None, None],
            ),
            Discretizer(
                ranges=[(-1.2, 0.6), (-0.07, 0.07), ],
                num_buckets=[65, 25],
                normal_params=[None, None],
            ),
            Discretizer(
                ranges=[(-1.2, 0.6), (-0.07, 0.07), ],
                num_buckets=[129, 65],
                normal_params=[None, None],
            ),
        ]
        action_discretizers = [
            # Discretizer(
            #     ranges=[(0, 2),],
            #     num_buckets=[0],
            #     normal_params=[None,],
            # ),
            Discretizer(
                ranges=[(0, 2),],
                num_buckets=[0],
                normal_params=[None,],
            ),
            Discretizer(
                ranges=[(0, 2),],
                num_buckets=[0],
                normal_params=[None,],
            ),
            Discretizer(
                ranges=[(0, 2),],
                num_buckets=[0],
                normal_params=[None,],
            ),
            Discretizer(
                ranges=[(0, 2),],
                num_buckets=[0],
                normal_params=[None,],
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
            "train_max_num_steps_per_episode": 200,
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

            int(250e3): {
                "explore_policy_exploit_policy_ratio": (1.0, 0.0),
                "train_exploit_policy": False,
                "test_exploit_policy": False,
            },
            # int(5_050e3): {
            #     "explore_policy_exploit_policy_ratio": (0.75, 0.25),
            #     "train_exploit_policy": True,
            #     "test_exploit_policy": True,
            #     "test_exploit_strategy": "greedy",
            #     "pyramid_index": 0,
            # },
            int(500e3): {
                "explore_policy_exploit_policy_ratio": (0.75, 0.25),
                "train_exploit_policy": True,
                "test_exploit_policy": True,
                "test_exploit_strategy": "greedy",
                "pyramid_index": 1,
            },
            int(750e3): {
                "explore_policy_exploit_policy_ratio": (0.5, 0.5),
                "train_exploit_policy": True,
                "test_exploit_policy": True,
                "test_exploit_strategy": "greedy",
                "pyramid_index": 2,
            },
            int(900e3): {
                "explore_policy_exploit_policy_ratio": (0.25, 0.75),
                "train_exploit_policy": True,
                "test_exploit_policy": True,
                "test_exploit_strategy": "greedy",
                "pyramid_index": 3,
            },
            int(1_200e3): {
                "explore_policy_exploit_policy_ratio": (0.25, 0.75),
                "train_exploit_policy": True,
                "test_exploit_policy": True,
                "test_exploit_strategy": "greedy",
                "pyramid_index": -4,
            },
            int(1_500e3): {
                "explore_policy_exploit_policy_ratio": (0.25, 0.75),
                "train_exploit_policy": True,
                "test_exploit_policy": True,
                "test_exploit_strategy": "greedy",
                "pyramid_index": -1,
            },
        }

    elif name == "acrobot":
        save_path = "./experiments/DynaQ/pyramid-acrobot/acrobot"
        env = gym.make("Acrobot-v1", render_mode="rgb_array")
        test_env = gym.make("Acrobot-v1", render_mode="rgb_array")
        state_discretizers = [
            Discretizer(
                ranges=[
                    (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0),
                    (-6.0, 6.0), (-12.0, 12.0),
                ],
                num_buckets=[9, 9, 9, 9, 9, 9, ],
                normal_params=[None, None, None, None, None, None, ],
            ),
            Discretizer(
                ranges=[
                    (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0),
                    (-6.0, 6.0), (-12.0, 12.0),
                ],
                num_buckets=[13, 13, 13, 13, 13, 13, ],
                normal_params=[None, None, None, None, None, None, ],
            ),
        ]
        action_discretizers = [
            Discretizer(
                ranges=[(0, 2),],
                num_buckets=[0],
                normal_params=[None,],
            ),
            Discretizer(
                ranges=[(0, 2),],
                num_buckets=[0],
                normal_params=[None,],
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

            int(250e3): {
                "explore_policy_exploit_policy_ratio": (1.0, 0.0),
                "train_exploit_policy": False,
                "test_exploit_policy": False,
            },
            int(350e3): {
                "explore_policy_exploit_policy_ratio": (0.75, 0.25),
                "train_exploit_policy": True,
                "test_exploit_policy": True,
                "test_exploit_strategy": "greedy",
                "pyramid_index": 0,
            },
            int(450e3): {
                "explore_policy_exploit_policy_ratio": (0.5, 0.5),
                "train_exploit_policy": True,
                "test_exploit_policy": True,
                "test_exploit_strategy": "greedy",
                "pyramid_index": 1,
            },
            int(750e3): {
                "explore_policy_exploit_policy_ratio": (0.25, 0.75),
                "train_exploit_policy": True,
                "test_exploit_policy": True,
                "test_exploit_strategy": "greedy",
                "pyramid_index": -1,
            },
        }
    elif name == "pendulum":
        save_path = "./experiments/DynaQ/pyramid-pendulum/pendulum"
        env = gym.make("Pendulum-v1", render_mode="rgb_array", )
        test_env = gym.make("Pendulum-v1", render_mode="rgb_array", )
        state_discretizers = [
            # Discretizer(
            #     ranges=[(-1.0, 1.0), (-1.0, 1.0), (-8.0, 8.0), ],
            #     num_buckets=[9, 9, 13],
            #     normal_params=[None, None, None],
            # ),
            Discretizer(
                ranges=[(-1.0, 1.0), (-1.0, 1.0), (-8.0, 8.0), ],
                num_buckets=[15, 15, 25],
                normal_params=[None, None, None],
            ),
            Discretizer(
                ranges=[(-1.0, 1.0), (-1.0, 1.0), (-8.0, 8.0), ],
                num_buckets=[25, 25, 33],
                normal_params=[None, None, None],
            )
        ]
        action_discretizers = [
            # Discretizer(
            #     ranges=[(-2.0, 2.0), ],
            #     num_buckets=[13],
            #     normal_params=[None, ],
            # ),
            Discretizer(
                ranges=[(-2.0, 2.0), ],
                num_buckets=[17],
                normal_params=[None, ],
            ),
            Discretizer(
                ranges=[(-2.0, 2.0), ],
                num_buckets=[17],
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
            "reward_resolution": 1,
            "train_max_num_steps_per_episode": 200,
            "exploit_agent_lr": 2.5e-4,
            "exploit_softmax_temperature": 0.5,
            "exploit_policy_reward_rate": 1e-2,
            "exploit_value_decay": 0.99,
            "exploit_policy_training_per_num_steps": int(10e3),
            "exploit_policy_training_steps": int(20e3),
            "exploit_policy_test_per_num_steps": int(10e3),
            "exploit_policy_test_episodes": 200,
            "save_per_num_steps": int(2.5e6),
            "save_mdp_graph": False,
            "print_training_info": False,
            int(100e3): {
                "explore_policy_exploit_policy_ratio": (1.0, 0.0),
                "train_exploit_policy": False,
                "test_exploit_policy": False,
            },
            int(150e3): {
                "explore_policy_exploit_policy_ratio": (0.5, 0.5),
                "train_exploit_policy": True,
                "test_exploit_policy": True,
                "test_exploit_strategy": "greedy",
                "pyramid_index": 0,
            },
            int(300e3): {
                "explore_policy_exploit_policy_ratio": (0.25, 0.75),
                "train_exploit_policy": True,
                "test_exploit_policy": True,
                "test_exploit_strategy": "greedy",
                "pyramid_index": 1,
            },
            # int(500e3): {
            #     "explore_policy_exploit_policy_ratio": (0.25, 0.75),
            #     "train_exploit_policy": True,
            #     "test_exploit_policy": True,
            #     "test_exploit_strategy": "greedy",
            #     "pyramid_index": 2,
            # },
            int(500e3): {
                "explore_policy_exploit_policy_ratio": (0.25, 0.75),
                "train_exploit_policy": True,
                "test_exploit_policy": True,
                "test_exploit_strategy": "greedy",
                "pyramid_index": -1,
            },
        }

    elif name == "lunarlander":
        save_path = "./experiments/DynaQ/pyramid-lunarlander/lunarlander"
        env = gym.make("LunarLander-v3", render_mode="rgb_array", continuous=True, )
        test_env = gym.make("LunarLander-v3", render_mode="rgb_array", continuous=True, )
        state_discretizers = [
            Discretizer(
                ranges=[
                    (-1.5, 1.5), (-1.5, 1.5), (-5.0, 5.0), (-5.0, 5.0),
                    (-3.14, 3.14), (-5.0, 5.0), (0, 1), (0, 1),
                ],
                num_buckets=[9, 9, 7, 7, 9, 9, 0, 0, ],
                normal_params=[None, None, None, None, None, None, None, None, ],
            ),
            Discretizer(
                ranges=[
                    (-1.5, 1.5), (-1.5, 1.5), (-5.0, 5.0), (-5.0, 5.0),
                    (-3.14, 3.14), (-5.0, 5.0), (0, 1), (0, 1),
                ],
                num_buckets=[9, 9, 9, 9, 13, 9, 0, 0, ],
                normal_params=[None, None, None, None, None, None, None, None, ],
            ),
            Discretizer(
                ranges=[
                    (-1.5, 1.5), (-1.5, 1.5), (-5.0, 5.0), (-5.0, 5.0),
                    (-3.14, 3.14), (-5.0, 5.0), (0, 1), (0, 1),
                ],
                num_buckets=[13, 13, 13, 13, 17, 13, 0, 0, ],
                normal_params=[None, None, None, None, None, None, None, None, ],
            ),
        ]
        action_discretizers = [
            Discretizer(
                ranges=[(-1, 1), (-1, 1)],
                num_buckets=[13, 13],
                normal_params=[None, None],
            ),
            Discretizer(
                ranges=[(-1, 1), (-1, 1)],
                num_buckets=[13, 13],
                normal_params=[None, None],
            ),
            Discretizer(
                ranges=[(-1, 1), (-1, 1)],
                num_buckets=[17, 17],
                normal_params=[None, None],
            )
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
            "reward_resolution": 10,
            "train_max_num_steps_per_episode": 500,
            "exploit_agent_lr": 2.5e-4,
            "exploit_softmax_temperature": 0.5,
            "exploit_policy_reward_rate": 1e-2,
            "exploit_value_decay": 0.99,
            "exploit_policy_training_per_num_steps": int(2.5e3),
            "exploit_policy_training_steps": int(5e3),
            "exploit_policy_test_per_num_steps": int(2.5e3),
            "exploit_policy_test_episodes": 200,
            "save_per_num_steps": int(2.5e6),
            "save_mdp_graph": False,
            "print_training_info": True,
            int(100e3): {
                "explore_policy_exploit_policy_ratio": (1.0, 0.0),
                "train_exploit_policy": False,
                "test_exploit_policy": False,
            },
            int(150e3): {
                "explore_policy_exploit_policy_ratio": (0.75, 0.25),
                "train_exploit_policy": True,
                "test_exploit_policy": True,
                "test_exploit_strategy": "greedy",
                "pyramid_index": 0,
            },
            int(300e3): {
                "explore_policy_exploit_policy_ratio": (0.5, 0.5),
                "train_exploit_policy": True,
                "test_exploit_policy": True,
                "test_exploit_strategy": "greedy",
                "pyramid_index": 1,
            },
            int(500e3): {
                "explore_policy_exploit_policy_ratio": (0.25, 0.75),
                "train_exploit_policy": True,
                "test_exploit_policy": True,
                "test_exploit_strategy": "greedy",
                "pyramid_index": 2,
            },
            int(750e3): {
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
