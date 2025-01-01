import gymnasium as gym

from custom_mountain_car import CustomMountainCarEnv
from dyna_q import Discretizer


def get_envs_discretizers_and_configs(name: str, configs_only=False):
    if name == "cartpole":
        save_path = "./experiments/DynaQ/cartpole/cartpole"
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
            "explore_agent_lr": 0.1,
            "explore_value_decay": 0.99,
            "explore_bonus_decay": 0.9,
            "explore_epsilon": 0.25,
            "explore_strategy": "greedy",
            "reward_resolution": 0,
            "train_max_num_steps_per_episode": 500,
            "exploit_policy_reward_rate": 1e-3,
            "exploit_value_decay": 0.99,
            "exploit_policy_training_per_num_steps": int(0.001e6),
            "exploit_policy_training_steps": int(0.005e6),
            "exploit_policy_test_per_num_steps": int(0.05e6),
            "exploit_policy_test_episodes": 64,
            "save_per_num_steps": int(0.25e6),
            "save_mdp_graph": False,
            "print_training_info": False,
            "init_groups": {
                "realstart": (1.0, 0.0, 0.0),
                "randstart": (0.0, 1.0, 0.0),
                "rand_real": (0.5, 0.5, 0.0),
                "qcutstart": (0.5, 0.25, 0.25),
            },
            "q_cut_params": {
                "num_targets": 32,
                "min_cut_max_flow_search_space": 128,
                "q_cut_space": 16,
                "weighted_search": True,
                "init_state_reward_prob_below_threshold": 0.1,
                "quality_value_threshold": 1.0,
                "take_done_states_as_targets": True,
            },
            int(1e6): {
                "explore_policy_exploit_policy_ratio": (1.0, 0.0),
                "train_exploit_policy": True,
                "epsilon": 0.3,
                "train_exploit_strategy": "greedy",
                "train_exploit_lr": 0.1,
                "test_exploit_policy": True,
                "test_exploit_strategy": "greedy",
            },
            int(3e6): {
                "explore_policy_exploit_policy_ratio": (0.5, 0.5),
                "train_exploit_policy": True,
                "epsilon": 0.2,
                "train_exploit_strategy": "greedy",
                "train_exploit_lr": 0.1,
                "test_exploit_policy": True,
                "test_exploit_strategy": "greedy",
            },
            int(5e6): {
                "explore_policy_exploit_policy_ratio": (0.25, 0.75),
                "train_exploit_policy": True,
                "epsilon": 0.1,
                "train_exploit_strategy": "greedy",
                "train_exploit_lr": 0.1,
                "test_exploit_policy": True,
                "test_exploit_strategy": "greedy",
            },
        }

    elif name == "mountain_car":
        save_path = "./experiments/DynaQ/mountain_car/mountain_car"
        env = CustomMountainCarEnv(custom_gravity=0.005, render_mode="rgb_array")
        test_env = CustomMountainCarEnv(custom_gravity=0.005, render_mode="rgb_array")
        state_discretizer = Discretizer(
            ranges=[(-1.2, 0.6), (-0.07, 0.07), ],
            num_buckets=[65, 25],
            normal_params=[None, None],
        )
        action_discretizer = Discretizer(
            ranges=[(0, 2),],
            num_buckets=[0],
            normal_params=[None, ],
        )
        action_type = "int"
        configs = {
            "save_path": save_path,
            "explore_agent_lr": 0.1,
            "explore_value_decay": 0.99,
            "explore_bonus_decay": 0.9,
            "explore_epsilon": 0.25,
            "explore_strategy": "greedy",
            "reward_resolution": 0,
            "train_max_num_steps_per_episode": 200,
            "exploit_policy_reward_rate": 1e-3,
            "exploit_value_decay": 0.99,
            "exploit_policy_training_per_num_steps": int(0.001e6),
            "exploit_policy_training_steps": int(0.005e6),
            "exploit_policy_test_per_num_steps": int(0.25e6),
            "exploit_policy_test_episodes": 64,
            "save_per_num_steps": int(2.5e6),
            "save_mdp_graph": False,
            "print_training_info": False,
            "init_groups": {
                "realstart": (1.0, 0.0, 0.0),
                "randstart": (0.0, 1.0, 0.0),
                "rand_real": (0.5, 0.5, 0.0),
                "qcutstart": (0.5, 0.25, 0.25),
            },
            "q_cut_params": {
                "num_targets": 32,
                "min_cut_max_flow_search_space": 128,
                "q_cut_space": 32,
                "weighted_search": True,
                "init_state_reward_prob_below_threshold": 0.1,
                "quality_value_threshold": 1.0,
                "take_done_states_as_targets": True,
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
                "train_exploit_lr": 0.1,
                "test_exploit_policy": True,
                "test_exploit_strategy": "greedy",
            },
            int(15e6): {
                "explore_policy_exploit_policy_ratio": (0.25, 0.75),
                "train_exploit_policy": True,
                "epsilon": 0.1,
                "train_exploit_strategy": "greedy",
                "train_exploit_lr": 0.1,
                "test_exploit_policy": True,
                "test_exploit_strategy": "greedy",
            },
        }

    elif name == "lunarlander":
        save_path = "./experiments/DynaQ/lunarlander/lunarlander"
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
        configs = {
            "save_path": save_path,
            "explore_agent_lr": 0.1,
            "explore_value_decay": 0.99,
            "explore_bonus_decay": 0.9,
            "explore_epsilon": 0.25,
            "explore_strategy": "greedy",
            "reward_resolution": 10,
            "train_max_num_steps_per_episode": 500,
            "exploit_policy_reward_rate": 1e-3,
            "exploit_value_decay": 0.99,
            "exploit_policy_training_per_num_steps": int(0.025e6),
            "exploit_policy_training_steps": int(0.05e6),
            "exploit_policy_test_per_num_steps": int(0.1e6),
            "exploit_policy_test_episodes": 200,
            "save_per_num_steps": int(2.5e6),
            "save_mdp_graph": False,
            "print_training_info": True,
            "init_groups": {
                "realstart": (1.0, 0.0, 0.0),
                "randstart": (0.0, 1.0, 0.0),
                "rand_real": (0.5, 0.5, 0.0),
                "qcutstart": (0.33, 0.33, 0.33),
            },
            "q_cut_params": {
                "num_targets": 256,
                "min_cut_max_flow_search_space": 512,
                "q_cut_space": 16,
                "weighted_search": True,
                "init_state_reward_prob_below_threshold": 0.1,
                "quality_value_threshold": 1.0,
                "take_done_states_as_targets": False,
            },
            int(10e6): {
                "explore_policy_exploit_policy_ratio": (0.75, 0.25),
                "train_exploit_policy": True,
                "epsilon": 0.3,
                "train_exploit_strategy": "greedy",
                "train_exploit_lr": 0.1,
                "test_exploit_policy": True,
                "test_exploit_strategy": "greedy",
            },
            int(20e6): {
                "explore_policy_exploit_policy_ratio": (0.5, 0.5),
                "train_exploit_policy": True,
                "epsilon": 0.2,
                "train_exploit_strategy": "greedy",
                "train_exploit_lr": 0.1,
                "test_exploit_policy": True,
                "test_exploit_strategy": "greedy",
            },
            int(25e6): {
                "explore_policy_exploit_policy_ratio": (0.25, 0.75),
                "train_exploit_policy": True,
                "epsilon": 0.1,
                "train_exploit_strategy": "greedy",
                "train_exploit_lr": 0.1,
                "test_exploit_policy": True,
                "test_exploit_strategy": "greedy",
            },
        }

    elif name == "acrobot":
        save_path = "./experiments/DynaQ/acrobot/acrobot"
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
        configs = {
            "save_path": save_path,
            "explore_agent_lr": 0.1,
            "explore_value_decay": 0.99,
            "explore_bonus_decay": 0.9,
            "explore_epsilon": 0.25,
            "explore_strategy": "greedy",
            "reward_resolution": 1,
            "train_max_num_steps_per_episode": 500,
            "exploit_policy_reward_rate": 1e-3,
            "exploit_value_decay": 0.99,
            "exploit_policy_training_per_num_steps": int(0.005e6),
            "exploit_policy_training_steps": int(0.025e6),
            "exploit_policy_test_per_num_steps": int(0.1e6),
            "exploit_policy_test_episodes": 200,
            "save_per_num_steps": int(2.5e6),
            "save_mdp_graph": False,
            "print_training_info": False,
            "init_groups": {
                "realstart": (1.0, 0.0, 0.0),
                "randstart": (0.0, 1.0, 0.0),
                "rand_real": (0.5, 0.5, 0.0),
                "qcutstart": (0.5, 0.25, 0.25),
            },
            "q_cut_params": {
                "num_targets": 32,
                "min_cut_max_flow_search_space": 128,
                "q_cut_space": 8,
                "weighted_search": True,
                "init_state_reward_prob_below_threshold": 0.05,
                "quality_value_threshold": 1.0,
                "take_done_states_as_targets": True,
            },
            int(1e6): {
                "explore_policy_exploit_policy_ratio": (1.0, 0.0),
                "train_exploit_policy": False,
                "test_exploit_policy": False,
            },
            int(7.5e6): {
                "explore_policy_exploit_policy_ratio": (0.5, 0.5),
                "train_exploit_policy": True,
                "epsilon": 0.25,
                "train_exploit_strategy": "greedy",
                "train_exploit_lr": 0.1,
                "test_exploit_policy": True,
                "test_exploit_strategy": "greedy",
            },
            int(10e6): {
                "explore_policy_exploit_policy_ratio": (0.25, 0.75),
                "train_exploit_policy": True,
                "epsilon": 0.1,
                "train_exploit_strategy": "greedy",
                "train_exploit_lr": 0.1,
                "test_exploit_policy": True,
                "test_exploit_strategy": "greedy",
            },
        }

    elif name =="pendulum":
        save_path = "./experiments/DynaQ/pendulum/pendulum"
        env = gym.make("Pendulum-v1", render_mode="rgb_array",)
        test_env = gym.make("Pendulum-v1", render_mode="rgb_array",)
        state_discretizer = Discretizer(
            ranges=[(-1.0, 1.0), (-1.0, 1.0), (-8.0, 8.0), ],
            num_buckets=[17, 17, 33],
            normal_params=[None, None, None],
        )
        action_discretizer = Discretizer(
            ranges=[(-2.0, 2.0),],
            num_buckets=[17],
            normal_params=[None,],
        )
        action_type = "float"
        configs = {
            "save_path": save_path,
            "explore_agent_lr": 0.1,
            "explore_value_decay": 0.99,
            "explore_bonus_decay": 0.9,
            "explore_epsilon": 0.25,
            "explore_strategy": "greedy",
            "reward_resolution": 1,
            "train_max_num_steps_per_episode": 200,
            "exploit_policy_reward_rate": 1e-3,
            "exploit_value_decay": 0.99,
            "exploit_policy_training_per_num_steps": int(0.005e6),
            "exploit_policy_training_steps": int(0.025e6),
            "exploit_policy_test_per_num_steps": int(0.1e6),
            "exploit_policy_test_episodes": 200,
            "save_per_num_steps": int(2.5e6),
            "save_mdp_graph": False,
            "print_training_info": False,
            "init_groups": {
                "realstart": (1.0, 0.0, 0.0),
                "randstart": (0.0, 1.0, 0.0),
                "rand_real": (0.5, 0.5, 0.0),
                "qcutstart": (0.5, 0.25, 0.25),
            },
            "q_cut_params": {
                "num_targets": 32,
                "min_cut_max_flow_search_space": 128,
                "q_cut_space": 16,
                "weighted_search": True,
                "init_state_reward_prob_below_threshold": 0.05,
                "quality_value_threshold": 1.0,
                "take_done_states_as_targets": True,
            },
            int(1e6): {
                "explore_policy_exploit_policy_ratio": (1.0, 0.0),
                "train_exploit_policy": False,
                "test_exploit_policy": False,
            },
            int(3e6): {
                "explore_policy_exploit_policy_ratio": (0.5, 0.5),
                "train_exploit_policy": True,
                "epsilon": 0.25,
                "train_exploit_strategy": "greedy",
                "train_exploit_lr": 0.1,
                "test_exploit_policy": True,
                "test_exploit_strategy": "greedy",
            },
            int(5e6): {
                "explore_policy_exploit_policy_ratio": (0.25, 0.75),
                "train_exploit_policy": True,
                "epsilon": 0.1,
                "train_exploit_strategy": "greedy",
                "train_exploit_lr": 0.1,
                "test_exploit_policy": True,
                "test_exploit_strategy": "greedy",
            },
        }

    elif name == "texi":
        save_path = "./experiments/DynaQ/texi/texi"
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
        configs = {
            "save_path": save_path,
            "explore_agent_lr": 0.1,
            "explore_value_decay": 0.99,
            "explore_bonus_decay": 0.9,
            "explore_epsilon": 0.25,
            "explore_strategy": "greedy",
            "reward_resolution": 0,
            "train_max_num_steps_per_episode": 200,
            "exploit_policy_reward_rate": 1e-3,
            "exploit_value_decay": 0.99,
            "exploit_policy_training_per_num_steps": int(0.001e6),
            "exploit_policy_training_steps": int(0.0025e6),
            "exploit_policy_test_per_num_steps": int(0.001e6),
            "exploit_policy_test_episodes": 200,
            "save_per_num_steps": int(0.2e6),
            "save_mdp_graph": True,
            "print_training_info": False,
            "init_groups": {
                "realstart": (1.0, 0.0, 0.0),
                "randstart": (0.0, 1.0, 0.0),
                "rand_real": (0.5, 0.5, 0.0),
                "qcutstart": (0.5, 0.25, 0.25),
            },
            "q_cut_params": {
                "num_targets": 32,
                "min_cut_max_flow_search_space": 64,
                "q_cut_space": 16,
                "weighted_search": True,
                "init_state_reward_prob_below_threshold": 0.1,
                "quality_value_threshold": 1.0,
                "take_done_states_as_targets": True,
            },
            int(0.05e6): {
                "explore_policy_exploit_policy_ratio": (1.0, 0.0),
                "train_exploit_policy": False,
                "test_exploit_policy": False,
            },
            int(0.15e6): {
                "explore_policy_exploit_policy_ratio": (0.5, 0.5),
                "train_exploit_policy": True,
                "epsilon": 0.25,
                "train_exploit_strategy": "greedy",
                "train_exploit_lr": 0.1,
                "test_exploit_policy": True,
                "test_exploit_strategy": "greedy",
            },
            int(0.2e6): {
                "explore_policy_exploit_policy_ratio": (0.25, 0.75),
                "train_exploit_policy": True,
                "epsilon": 0.25,
                "train_exploit_strategy": "greedy",
                "train_exploit_lr": 0.1,
                "test_exploit_policy": True,
                "test_exploit_strategy": "greedy",
            },
        }

    else:
        raise ValueError(f"Invalid environment name: {name}.")

    if configs_only:
        return configs
    return env, test_env, state_discretizer, action_discretizer, action_type, configs
