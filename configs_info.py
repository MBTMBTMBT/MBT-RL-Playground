import gymnasium as gym
import custom_envs  # import to register custom envs
from dyna_q_modernized import Discretizer


def get_envs_discretizers_and_configs(name: str, env_idx: int, configs_only=False):
    if name == "frozen_lake-4-times-4":
        save_path = "./experiments/env-info/frozen_lake-44/frozen_lake-4-times-4"
        envs = [
            dict(
                id="FrozenLake-v1",
                desc=None,
                map_name="4x4",
                is_slippery=False,
                render_mode="rgb_array",
            ),
            dict(
                id="FrozenLake-v1",
                desc=None,
                map_name="4x4",
                is_slippery=True,
                render_mode="rgb_array",
            ),
        ]
        test_envs = [
            dict(
                id="FrozenLake-v1",
                desc=None,
                map_name="4x4",
                is_slippery=False,
                render_mode="rgb_array",
            ),
            dict(
                id="FrozenLake-v1",
                desc=None,
                map_name="4x4",
                is_slippery=True,
                render_mode="rgb_array",
            ),
        ]
        env_descs = [
            "not-slippery",
            "slippery",
        ]
        state_discretizer = Discretizer(
            ranges=[(0, gym.make(**envs[0]).observation_space.n)],
            num_buckets=[0],
            normal_params=[
                None,
            ],
        )
        action_discretizer = Discretizer(
            ranges=[
                (0, 3),
            ],
            num_buckets=[0],
            normal_params=[
                None,
            ],
        )
        configs = {
            "num_envs": len(env_descs),
            "use_deep_agent": False,
            "save_path": save_path,
            "train_max_num_steps_per_episode": 100,
            "exploit_agent_lr": 0.05,
            "exploit_softmax_temperature": 1.0,
            "exploit_policy_reward_rate": 1,
            "exploit_value_decay": 0.99,
            "fast_exploit_policy_training_steps": int(50e3),
            "exploit_policy_training_steps": int(50e3),
            "exploit_policy_test_per_num_steps": int(0.25e3),
            "exploit_policy_test_episodes": 250,
            "exploit_policy_eval_episodes": 2500,
            "save_per_num_steps": int(5e3),
            "save_mdp_graph": False,
            "print_training_info": False,
            "initialization_distribution": (0.5, 0.5),
            "use_balanced_random_init": False,
            "quick_test_threshold": 0.1,
            "quick_test_num_episodes": 25,
            "early_stop_counts": 5,
            "success_threshold": 0.70,
        }

    elif name == "frozen_lake-88":
        save_path = "./experiments/env-info/frozen_lake-88/frozen_lake-88"
        envs = [
            dict(
                id="FrozenLake-v1",
                desc=None,
                map_name="8x8",
                is_slippery=False,
                render_mode="rgb_array",
            ),
            dict(
                id="FrozenLake-v1",
                desc=None,
                map_name="8x8",
                is_slippery=True,
                render_mode="rgb_array",
            ),
        ]
        test_envs = [
            dict(
                id="FrozenLake-v1",
                desc=None,
                map_name="8x8",
                is_slippery=False,
                render_mode="rgb_array",
            ),
            dict(
                id="FrozenLake-v1",
                desc=None,
                map_name="8x8",
                is_slippery=True,
                render_mode="rgb_array",
            ),
        ]
        env_descs = [
            "88-not-slippery",
            "88-slippery",
        ]
        state_discretizer = Discretizer(
            ranges=[(0, gym.make(**envs[0]).observation_space.n)],
            num_buckets=[0],
            normal_params=[
                None,
            ],
        )
        action_discretizer = Discretizer(
            ranges=[
                (0, 3),
            ],
            num_buckets=[0],
            normal_params=[
                None,
            ],
        )
        configs = {
            "num_envs": len(env_descs),
            "use_deep_agent": False,
            "save_path": save_path,
            "train_max_num_steps_per_episode": 100,
            "exploit_agent_lr": 0.05,
            "exploit_softmax_temperature": 1.0,
            "exploit_policy_reward_rate": 1,
            "exploit_value_decay": 0.99,
            "fast_exploit_policy_training_steps": int(200e3),
            "exploit_policy_training_steps": int(300e3),
            "exploit_policy_test_per_num_steps": int(2.5e3),
            "exploit_policy_test_episodes": 25,
            "exploit_policy_eval_episodes": 250,
            "save_per_num_steps": int(50e3),
            "save_mdp_graph": False,
            "print_training_info": False,
            "initialization_distribution": (0.5, 0.5),
            "use_balanced_random_init": False,
            "quick_test_threshold": 0.1,
            "quick_test_num_episodes": 25,
            "early_stop_counts": 5,
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
                "FFFFHFFF",
                "FFFFHGFF",
                "FFFFFFFF",
                "FFFFFFFF",
            ],
            [
                "SFFFFFFF",
                "FFFFFFFF",
                "FFFFFFFF",
                "FFFFFFFF",
                "FFFFHFFF",
                "FFFFHGFF",
                "FFFFHFFF",
                "FFFFHFFF",
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
                "FFHFFHHF",
                "FFFFFFFF",
                "FFFFHHFF",
                "FFFFHGFF",
                "FFHFFFFF",
                "FFFFHFFF",
            ],
        ]
        envs = [
            # dict(
            #     id='CustomFrozenLake-v1',
            #     desc=descs[0],
            #     map_name=None,
            #     is_slippery=True,
            #     slipperiness=0.33,
            #     render_mode="rgb_array",
            # ),
            # dict(
            #     id='CustomFrozenLake-v1',
            #     desc=descs[1],
            #     map_name=None,
            #     is_slippery=True,
            #     slipperiness=0.33,
            #     render_mode="rgb_array",
            # ),
            # dict(
            #     id='CustomFrozenLake-v1',
            #     desc=descs[2],
            #     map_name=None,
            #     is_slippery=True,
            #     slipperiness=0.33,
            #     render_mode="rgb_array",
            # ),
            # dict(
            #     id='CustomFrozenLake-v1',
            #     desc=descs[3],
            #     map_name=None,
            #     is_slippery=True,
            #     slipperiness=0.33,
            #     render_mode="rgb_array",
            # ),
            # dict(
            #     id='CustomFrozenLake-v1',
            #     desc=descs[4],
            #     map_name=None,
            #     is_slippery=True,
            #     slipperiness=0.33,
            #     render_mode="rgb_array",
            # ),
            # dict(
            #     id='CustomFrozenLake-v1',
            #     desc=descs[0],
            #     map_name=None,
            #     is_slippery=True,
            #     slipperiness=0.1,
            #     render_mode="rgb_array",
            # ),
            # dict(
            #     id='CustomFrozenLake-v1',
            #     desc=descs[1],
            #     map_name=None,
            #     is_slippery=True,
            #     slipperiness=0.1,
            #     render_mode="rgb_array",
            # ),
            # dict(
            #     id='CustomFrozenLake-v1',
            #     desc=descs[2],
            #     map_name=None,
            #     is_slippery=True,
            #     slipperiness=0.1,
            #     render_mode="rgb_array",
            # ),
            # dict(
            #     id='CustomFrozenLake-v1',
            #     desc=descs[3],
            #     map_name=None,
            #     is_slippery=True,
            #     slipperiness=0.1,
            #     render_mode="rgb_array",
            # ),
            # dict(
            #     id='CustomFrozenLake-v1',
            #     desc=descs[4],
            #     map_name=None,
            #     is_slippery=True,
            #     slipperiness=0.2,
            #     render_mode="rgb_array",
            # ),
            dict(
                id="CustomFrozenLake-v1",
                desc=descs[0],
                map_name=None,
                is_slippery=True,
                slipperiness=0.2,
                render_mode="rgb_array",
            ),
            dict(
                id="CustomFrozenLake-v1",
                desc=descs[1],
                map_name=None,
                is_slippery=True,
                slipperiness=0.2,
                render_mode="rgb_array",
            ),
            dict(
                id="CustomFrozenLake-v1",
                desc=descs[2],
                map_name=None,
                is_slippery=True,
                slipperiness=0.2,
                render_mode="rgb_array",
            ),
            dict(
                id="CustomFrozenLake-v1",
                desc=descs[3],
                map_name=None,
                is_slippery=True,
                slipperiness=0.2,
                render_mode="rgb_array",
            ),
            dict(
                id="CustomFrozenLake-v1",
                desc=descs[4],
                map_name=None,
                is_slippery=True,
                slipperiness=0.2,
                render_mode="rgb_array",
            ),
        ]
        test_envs = envs
        env_descs = [
            # "env-1-lowslipperiness",
            # "env-2-lowslipperiness",
            # "env-3-lowslipperiness",
            # "env-4-lowslipperiness",
            # "env-5-lowslipperiness",
            # "env-1-lowslipperiness",
            # "env-2-lowslipperiness",
            # "env-3-lowslipperiness",
            # "env-4-lowslipperiness",
            # "env-5-lowslipperiness",
            "env-1",
            "env-2",
            "env-3",
            "env-4",
            "env-5",
        ]
        state_discretizer = Discretizer(
            ranges=[(0, gym.make(**envs[0]).observation_space.n)],
            num_buckets=[0],
            normal_params=[
                None,
            ],
        )
        action_discretizer = Discretizer(
            ranges=[
                (0, 3),
            ],
            num_buckets=[0],
            normal_params=[
                None,
            ],
        )
        configs = {
            "num_envs": len(env_descs),
            "use_deep_agent": False,
            "save_path": save_path,
            "train_max_num_steps_per_episode": 200,
            "exploit_agent_lr": 0.025,
            "exploit_softmax_temperature": 1.0,
            "exploit_policy_reward_rate": 1,
            "exploit_value_decay": 0.99,
            "fast_exploit_policy_training_steps": int(500e3),
            "exploit_policy_training_steps": int(750e3),
            "exploit_policy_test_per_num_steps": int(5e3),
            "exploit_policy_test_episodes": 500,
            "exploit_policy_eval_episodes": 5000,
            "save_per_num_steps": int(50e3),
            "save_mdp_graph": False,
            "print_training_info": False,
            "initialization_distribution": (0.5, 0.5),
            "use_balanced_random_init": True,
            "quick_test_threshold": 0.1,
            "quick_test_num_episodes": 25,
            "early_stop_counts": 5,
            "success_threshold": 0.95,
        }

    elif name == "mountaincar-custom":
        save_path = "./experiments/env-info/mountaincar-custom/mountaincar-custom"
        envs = [
            dict(
                id="CustomMountainCar-v0",
                goal_velocity=0,
                custom_gravity=0.0025,
                custom_force=0.001,
                goal_position=0.5,
                reward_type="sparse",
                render_mode="rgb_array",
            ),
            dict(
                id="CustomMountainCar-v0",
                goal_velocity=0,
                custom_gravity=0.002,
                custom_force=0.001,
                goal_position=0.5,
                reward_type="sparse",
                render_mode="rgb_array",
            ),
            dict(
                id="CustomMountainCar-v0",
                goal_velocity=0,
                custom_gravity=0.005,
                custom_force=0.001,
                goal_position=0.5,
                reward_type="sparse",
                render_mode="rgb_array",
            ),
            dict(
                id="CustomMountainCar-v0",
                goal_velocity=0,
                custom_gravity=0.0025,
                custom_force=0.0015,
                goal_position=0.5,
                reward_type="sparse",
                render_mode="rgb_array",
            ),
            dict(
                id="CustomMountainCar-v0",
                goal_velocity=0,
                custom_gravity=0.0025,
                custom_force=0.001,
                goal_position=0.125,
                reward_type="sparse",
                render_mode="rgb_array",
            ),
            dict(
                id="CustomMountainCar-v0",
                goal_velocity=0,
                custom_gravity=0.0025,
                custom_force=0.001,
                goal_position=-0.1,
                reward_type="sparse",
                render_mode="rgb_array",
            ),
        ]
        test_envs = envs
        env_descs = [
            "default",
            "low-grav",
            "high-grav",
            "high-force",
            "low-pos",
            "xlow-pos",
        ]
        state_discretizer = Discretizer(
            ranges=[
                (-1.2, 0.6),
                (-0.07, 0.07),
            ],
            num_buckets=[25, 13],
            normal_params=[None, None],
        )
        action_discretizer = Discretizer(
            ranges=[
                (0, 2),
            ],
            num_buckets=[0],
            normal_params=[
                None,
            ],
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
            "fast_exploit_policy_training_steps": int(2500e3),
            "exploit_policy_training_steps": int(3000e3),
            "exploit_policy_test_per_num_steps": int(300e3),
            "exploit_policy_test_episodes": 25,
            "exploit_policy_eval_episodes": 250,
            "save_per_num_steps": int(500e3),
            "save_mdp_graph": False,
            "print_training_info": False,
            "initialization_distribution": (0.2, 0.8),
            "use_balanced_random_init": True,
            "quick_test_threshold": 0.33,
            "quick_test_num_episodes": 25,
        }

    elif name == "acrobot-custom":
        save_path = "./experiments/env-info/acrobot-custom/acrobot-custom"
        envs = [
            dict(
                id="CustomAcrobot-v1",
                termination_height=1.8,
                friction=0.0,
                torque_scaling=1.0,
                gravity=9.8,
                reward_type="scaled",
                render_mode="rgb_array",
            ),
            dict(
                id="CustomAcrobot-v1",
                termination_height=1.6,
                friction=0.0,
                torque_scaling=1.0,
                gravity=9.8,
                reward_type="scaled",
                render_mode="rgb_array",
            ),
            dict(
                id="CustomAcrobot-v1",
                termination_height=1.4,
                friction=0.0,
                torque_scaling=1.0,
                gravity=9.8,
                reward_type="scaled",
                render_mode="rgb_array",
            ),
            dict(
                id="CustomAcrobot-v1",
                termination_height=1.2,
                friction=0.0,
                torque_scaling=1.0,
                gravity=9.8,
                reward_type="scaled",
                render_mode="rgb_array",
            ),
            dict(
                id="CustomAcrobot-v1",
                termination_height=1.8,
                friction=0.0,
                torque_scaling=1.0,
                gravity=6.8,
                reward_type="scaled",
                render_mode="rgb_array",
            ),
            dict(
                id="CustomAcrobot-v1",
                termination_height=1.8,
                friction=0.0,
                torque_scaling=1.0,
                gravity=4.8,
                reward_type="scaled",
                render_mode="rgb_array",
            ),
            dict(
                id="CustomAcrobot-v1",
                termination_height=1.8,
                friction=0.0,
                torque_scaling=1.0,
                gravity=2.8,
                reward_type="scaled",
                render_mode="rgb_array",
            ),
        ]
        test_envs = envs
        env_descs = [
            "target",
            "mid-termination",
            "low-termination",
            "verylow-termination",
            "mid-gravity",
            "low-gravity",
            "verylow-gravity",
        ]
        state_discretizer = Discretizer(
            ranges=[
                (-1.0, 1.0),
                (-1.0, 1.0),
                (-1.0, 1.0),
                (-1.0, 1.0),
                (-6.0, 6.0),
                (-12.0, 12.0),
            ],
            num_buckets=[
                9,
                9,
                9,
                9,
                9,
                9,
            ],
            normal_params=[
                None,
                None,
                None,
                None,
                None,
                None,
            ],
        )
        action_discretizer = Discretizer(
            ranges=[
                (0, 2),
            ],
            num_buckets=[0],
            normal_params=[
                None,
            ],
        )
        configs = {
            "num_envs": len(env_descs),
            "use_deep_agent": True,
            "save_path": save_path,
            "train_max_num_steps_per_episode": 250,
            "exploit_agent_lr": 1e-4,
            "exploit_softmax_temperature": 1.0,
            "exploit_policy_reward_rate": 1,
            "exploit_value_decay": 0.99,
            "fast_exploit_policy_training_steps": int(150e3),
            "exploit_policy_training_steps": int(300e3),
            "exploit_policy_test_per_num_steps": int(5e3),
            "exploit_policy_test_episodes": 500,
            "exploit_policy_eval_episodes": 2000,
            "save_per_num_steps": int(50e3),
            "save_mdp_graph": False,
            "print_training_info": False,
            "initialization_distribution": (0.75, 0.25),
            "use_balanced_random_init": False,
            "quick_test_threshold": -0.75,
            "quick_test_num_episodes": 100,
            "early_stop_counts": 5,
            "success_threshold": -0.525,
        }

    else:
        raise ValueError(f"Invalid environment name: {name}.")

    if configs_only:
        return configs
    return (
        gym.make(**envs[env_idx]),
        gym.make(**test_envs[env_idx]),
        env_descs[env_idx],
        state_discretizer,
        action_discretizer,
        configs,
    )
