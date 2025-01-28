import gymnasium as gym
import custom_envs  # import to register custom envs
from dyna_q_modernized import Discretizer


def get_envs_discretizers_and_configs(name: str, env_idx: int, configs_only=False):
    if name == "frozen_lake-44":
        save_path = "./experiments/env-info/frozen_lake-44/frozen_lake-44"
        envs = [
            dict(id='FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False, render_mode="rgb_array"),
            dict(id='FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True, render_mode="rgb_array"),
        ]
        test_envs = [
            dict(id='FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False, render_mode="rgb_array"),
            dict(id='FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True, render_mode="rgb_array"),
        ]
        env_descs = [
            "44-not-slippery",
            "44-slippery",
        ]
        state_discretizer = Discretizer(
            ranges=[(0, gym.make(**envs[0]).observation_space.n)],
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
            "train_max_num_steps_per_episode": 100,
            "exploit_agent_lr": 0.1,
            "exploit_softmax_temperature": 1.0,
            "exploit_policy_reward_rate": 1,
            "exploit_value_decay": 0.99,
            "exploit_policy_training_steps": int(25e3),
            "exploit_policy_test_per_num_steps": int(0.25e3),
            "exploit_policy_test_episodes": 25,
            "exploit_policy_eval_episodes": 500,
            "save_per_num_steps": int(5e3),
            "save_mdp_graph": False,
            "print_training_info": False,
            "initialization_distribution": (0.5, 0.5),
            "use_balanced_random_init": True,
        }

    elif name == "frozen_lake-88":
        save_path = "./experiments/env-info/frozen_lake-88/frozen_lake-88"
        envs = [
            dict(id='FrozenLake-v1', desc=None, map_name="8x8", is_slippery=False, render_mode="rgb_array"),
            dict(id='FrozenLake-v1', desc=None, map_name="8x8", is_slippery=True, render_mode="rgb_array"),
        ]
        test_envs = [
            dict(id='FrozenLake-v1', desc=None, map_name="8x8", is_slippery=False, render_mode="rgb_array"),
            dict(id='FrozenLake-v1', desc=None, map_name="8x8", is_slippery=True, render_mode="rgb_array"),
        ]
        env_descs = [
            "88-not-slippery",
            "88-slippery",
        ]
        state_discretizer = Discretizer(
            ranges=[(0, gym.make(**envs[0]).observation_space.n)],
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
            "train_max_num_steps_per_episode": 100,
            "exploit_agent_lr": 0.1,
            "exploit_softmax_temperature": 1.0,
            "exploit_policy_reward_rate": 1,
            "exploit_value_decay": 0.99,
            "exploit_policy_training_steps": int(250e3),
            "exploit_policy_test_per_num_steps": int(2.5e3),
            "exploit_policy_test_episodes": 25,
            "exploit_policy_eval_episodes": 500,
            "save_per_num_steps": int(50e3),
            "save_mdp_graph": False,
            "print_training_info": False,
            "initialization_distribution": (0.5, 0.5),
            "use_balanced_random_init": True,
        }

    elif name == "frozen_lake-custom":
        save_path = "./experiments/env-info/frozen_lake-custom/frozen_lake-custom"
        descs = [
            [
                "SFFFFFFF",
                "FFFFFFFF",
                "FFFFFFFF",
                "FFFFFFFF",
                "FFFFFFFF",
                "FFFFFGFF",
                "FFFFFFFF",
                "FFFFFFFF",
            ],
            [
                "SFFFFFFF",
                "FFFFFFFF",
                "FFFFFFFF",
                "FFFFFFFF",
                "FFFFHHFF",
                "FFFFHGFF",
                "FFFFFFFF",
                "FFFFFFFF",
            ],
            [
                "SFFFFFFF",
                "FFFFFFFF",
                "FFHFFFFF",
                "FFFFFFFF",
                "FFFFHHFF",
                "FFFFHGFF",
                "FFHFFFFF",
                "FFFFHFFF",
            ],
            [
                "SFFFFFFF",
                "FFFFFFFF",
                "FFHFFHHH",
                "FFFFFFFF",
                "FFFFHHFF",
                "FFFFHGFF",
                "FFHFFFFF",
                "FFFFHFFF",
            ],
        ]
        envs = [
            dict(
                id='CustomFrozenLake-v1',
                desc=descs[0],
                map_name=None,
                is_slippery=True,
                slipperiness=0.10,
                render_mode="rgb_array",
            ),
            dict(
                id='CustomFrozenLake-v1',
                desc=descs[1],
                map_name=None,
                is_slippery=True,
                slipperiness=0.10,
                render_mode="rgb_array",
            ),
            dict(
                id='CustomFrozenLake-v1',
                desc=descs[2],
                map_name=None,
                is_slippery=True,
                slipperiness=0.10,
                render_mode="rgb_array",
            ),
            dict(
                id='CustomFrozenLake-v1',
                desc=descs[3],
                map_name=None,
                is_slippery=True,
                slipperiness=0.10,
                render_mode="rgb_array",
            ),
            dict(
                id='CustomFrozenLake-v1',
                desc=descs[0],
                map_name=None,
                is_slippery=True,
                slipperiness=0.90,
                render_mode="rgb_array",
            ),
            dict(
                id='CustomFrozenLake-v1',
                desc=descs[1],
                map_name=None,
                is_slippery=True,
                slipperiness=0.90,
                render_mode="rgb_array",
            ),
            dict(
                id='CustomFrozenLake-v1',
                desc=descs[2],
                map_name=None,
                is_slippery=True,
                slipperiness=0.90,
                render_mode="rgb_array",
            ),
            dict(
                id='CustomFrozenLake-v1',
                desc=descs[3],
                map_name=None,
                is_slippery=True,
                slipperiness=0.90,
                render_mode="rgb_array",
            ),
        ]
        test_envs = envs
        env_descs = [
            "difficulty-1-sl10",
            "difficulty-2-sl10",
            "difficulty-3-sl10",
            "difficulty-4-sl10",
            "difficulty-1-sl90",
            "difficulty-2-sl90",
            "difficulty-3-sl90",
            "difficulty-4-sl90",
        ]
        state_discretizer = Discretizer(
            ranges=[(0, gym.make(**envs[0]).observation_space.n)],
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
            "train_max_num_steps_per_episode": 100,
            "exploit_agent_lr": 0.1,
            "exploit_softmax_temperature": 1.0,
            "exploit_policy_reward_rate": 1,
            "exploit_value_decay": 0.99,
            "exploit_policy_training_steps": int(200e3),
            "exploit_policy_test_per_num_steps": int(2e3),
            "exploit_policy_test_episodes": 100,
            "exploit_policy_eval_episodes": 500,
            "save_per_num_steps": int(50e3),
            "save_mdp_graph": False,
            "print_training_info": False,
            "initialization_distribution": (0.5, 0.5),
            "use_balanced_random_init": True,
        }

    elif name == "mountaincar-custom":
        save_path = "./experiments/env-info/mountaincar-custom/mountaincar-custom"
        envs = [
            dict(
                id='CustomMountainCar-v0',
                goal_velocity=0,
                custom_gravity=0.0025,
                custom_force=0.001,
                goal_position=0.5,
                reward_type="sparse",
                render_mode="rgb_array",
            ),
            dict(
                id='CustomMountainCar-v0',
                goal_velocity=0,
                custom_gravity=0.0015,
                custom_force=0.001,
                goal_position=0.5,
                reward_type="sparse",
                render_mode="rgb_array",
            ),
            dict(
                id='CustomMountainCar-v0',
                goal_velocity=0,
                custom_gravity=0.0025,
                custom_force=0.002,
                goal_position=0.5,
                reward_type="sparse",
                render_mode="rgb_array",
            ),
            dict(
                id='CustomMountainCar-v0',
                goal_velocity=0,
                custom_gravity=0.0025,
                custom_force=0.003,
                goal_position=0.5,
                reward_type="sparse",
                render_mode="rgb_array",
            ),
            dict(
                id='CustomMountainCar-v0',
                goal_velocity=0,
                custom_gravity=0.0025,
                custom_force=0.001,
                goal_position=0.125,
                reward_type="sparse",
                render_mode="rgb_array",
            ),
            dict(
                id='CustomMountainCar-v0',
                goal_velocity=0,
                custom_gravity=0.0025,
                custom_force=0.001,
                goal_position=-0.25,
                reward_type="sparse",
                render_mode="rgb_array",
            ),
        ]
        test_envs = envs
        env_descs = [
            "default",
            "low-grav",
            "high-force",
            "xhigh-force",
            "low-pos",
            "xlow-pos",
        ]
        state_discretizer = Discretizer(
            ranges=[(-1.2, 0.6), (-0.07, 0.07), ],
            num_buckets=[25, 13],
            normal_params=[None, None],
        )
        action_discretizer = Discretizer(
            ranges=[(0, 2), ],
            num_buckets=[0],
            normal_params=[None, ],
        )
        configs = {
            "num_envs": len(env_descs),
            "use_deep_agent": True,
            "save_path": save_path,
            "train_max_num_steps_per_episode": 200,
            "exploit_agent_lr": 2.5e-4,
            "exploit_softmax_temperature": 1.0,
            "exploit_policy_reward_rate": 1,
            "exploit_value_decay": 0.99,
            "exploit_policy_training_steps": int(250e3),
            "exploit_policy_test_per_num_steps": int(25e3),
            "exploit_policy_test_episodes": 25,
            "exploit_policy_eval_episodes": 500,
            "save_per_num_steps": int(500e3),
            "save_mdp_graph": False,
            "print_training_info": False,
            "initialization_distribution": (0.5, 0.5),
            "use_balanced_random_init": True,
        }

    else:
        raise ValueError(f"Invalid environment name: {name}.")

    if configs_only:
        return configs
    return gym.make(**envs[env_idx]), gym.make(**test_envs[env_idx]), env_descs[env_idx], state_discretizer, action_discretizer, configs
