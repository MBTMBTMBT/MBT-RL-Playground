if __name__ == '__main__':
    import numpy as np
    from matplotlib import pyplot as plt
    import os
    from parallel_training import run_all_experiments

    # General experiment parameters
    experiment_name = "MountainCar_Experiments"
    save_dir = f"./experiments/{experiment_name}/"
    os.makedirs(save_dir, exist_ok=True)

    # Define experiment groups
    experiment_groups = [
        {
            "group_name": "MC-25-5",
            "env_id": "Custom-MountainCar",
            "train_env_params": [
                {
                    "render_mode": "rgb_array",
                    "goal_velocity": 0,
                    "custom_gravity": 0.0005,
                    "max_episode_steps": 200,
                    "reward_type": 'default',
                },
                {
                    "render_mode": "rgb_array",
                    "goal_velocity": 0,
                    "custom_gravity": 0.0010,
                    "max_episode_steps": 200,
                    "reward_type": 'default',
                },
                {
                    "render_mode": "rgb_array",
                    "goal_velocity": 0,
                    "custom_gravity": 0.0015,
                    "max_episode_steps": 200,
                    "reward_type": 'default',
                },
                {
                    "render_mode": "rgb_array",
                    "goal_velocity": 0,
                    "custom_gravity": 0.0020,
                    "max_episode_steps": 200,
                    "reward_type": 'default',
                },
                {
                    "render_mode": "rgb_array",
                    "goal_velocity": 0,
                    "custom_gravity": 0.0025,
                    "max_episode_steps": 200,
                    "reward_type": 'default',
                },
            ],
            "test_per_num_steps": int(0.1e6),
            "test_runs": 10,
            "test_env_params": {
                "render_mode": "rgb_array",
                "goal_velocity": 0,
                "custom_gravity": 0.0015,
                "max_episode_steps": 200,
                "reward_type": 'progress',
            },
            "state_space": [
                {'type': 'continuous', 'range': (-1.2, 0.6), 'bins': 16},  # Position
                {'type': 'continuous', 'range': (-0.07, 0.07), 'bins': 16}  # Velocity
            ],
            "action_space": [{'type': 'discrete', 'bins': 3}],
            "alpha": 0.1,
            "gamma": 0.99,
            "epsilon_start": 0.25,
            "epsilon_end": 0.05,
            "total_steps": int(25e6),
            "runs": 3,
        },
        {
            "group_name": "MC-25-4",
            "env_id": "Custom-MountainCar",
            "train_env_params": [
                {
                    "render_mode": "rgb_array",
                    "goal_velocity": 0,
                    "custom_gravity": 0.0005,
                    "max_episode_steps": 200,
                    "reward_type": 'default',
                },
                {
                    "render_mode": "rgb_array",
                    "goal_velocity": 0,
                    "custom_gravity": 0.0012,
                    "max_episode_steps": 200,
                    "reward_type": 'default',
                },
                {
                    "render_mode": "rgb_array",
                    "goal_velocity": 0,
                    "custom_gravity": 0.0018,
                    "max_episode_steps": 200,
                    "reward_type": 'default',
                },
                {
                    "render_mode": "rgb_array",
                    "goal_velocity": 0,
                    "custom_gravity": 0.0025,
                    "max_episode_steps": 200,
                    "reward_type": 'default',
                },
            ],
            "test_per_num_steps": int(0.1e6),
            "test_runs": 10,
            "test_env_params": {
                "render_mode": "rgb_array",
                "goal_velocity": 0,
                "custom_gravity": 0.0015,
                "max_episode_steps": 200,
                "reward_type": 'progress',
            },
            "state_space": [
                {'type': 'continuous', 'range': (-1.2, 0.6), 'bins': 16},  # Position
                {'type': 'continuous', 'range': (-0.07, 0.07), 'bins': 16}  # Velocity
            ],
            "action_space": [{'type': 'discrete', 'bins': 3}],
            "alpha": 0.1,
            "gamma": 0.99,
            "epsilon_start": 0.25,
            "epsilon_end": 0.05,
            "total_steps": int(25e6),
            "runs": 3,
        },
        {
            "group_name": "MC-25-3",
            "env_id": "Custom-MountainCar",
            "train_env_params": [
                {
                    "render_mode": "rgb_array",
                    "goal_velocity": 0,
                    "custom_gravity": 0.0005,
                    "max_episode_steps": 200,
                    "reward_type": 'default',
                },
                {
                    "render_mode": "rgb_array",
                    "goal_velocity": 0,
                    "custom_gravity": 0.0015,
                    "max_episode_steps": 200,
                    "reward_type": 'default',
                },
                {
                    "render_mode": "rgb_array",
                    "goal_velocity": 0,
                    "custom_gravity": 0.0025,
                    "max_episode_steps": 200,
                    "reward_type": 'default',
                },
            ],
            "test_per_num_steps": int(0.1e6),
            "test_runs": 10,
            "test_env_params": {
                "render_mode": "rgb_array",
                "goal_velocity": 0,
                "custom_gravity": 0.0015,
                "max_episode_steps": 200,
                "reward_type": 'progress',
            },
            "state_space": [
                {'type': 'continuous', 'range': (-1.2, 0.6), 'bins': 16},  # Position
                {'type': 'continuous', 'range': (-0.07, 0.07), 'bins': 16}  # Velocity
            ],
            "action_space": [{'type': 'discrete', 'bins': 3}],
            "alpha": 0.1,
            "gamma": 0.99,
            "epsilon_start": 0.25,
            "epsilon_end": 0.05,
            "total_steps": int(25e6),
            "runs": 3,
        },
    ]

    # Run all experiments
    max_workers = 12  # Number of parallel processes
    aggregated_results = run_all_experiments(experiment_groups, save_dir, max_workers)

    plt.figure(figsize=(12, 8))
    linestyles = [
        '-', '--', '-.', ':', (0, (1, 1)), (0, (5, 1)),
        (0, (3, 1, 1, 1)), (0, (3, 5, 1, 5)), (0, (5, 10)),
        (0, (1, 10)), (0, (5, 5, 1, 5)), (0, (2, 2, 1, 2))
    ]

    # Target number of points for the plot
    target_points = 8192

    for i, (group_name, (avg_rewards, std_rewards, avg_test_reward)) in enumerate(aggregated_results.items()):
        total_steps = experiment_groups[i]["total_steps"]
        total_points = len(avg_rewards)
        downsample_rate = max(1, total_points // target_points)

        # Combine smoothing and downsampling using a rolling average
        avg_rewards_array = np.array(avg_rewards)
        std_rewards_array = np.array(std_rewards)

        kernel_size = downsample_rate
        kernel = np.ones(kernel_size) / kernel_size

        smoothed_rewards = np.convolve(avg_rewards_array, kernel, mode='valid')[::downsample_rate]
        smoothed_std = np.convolve(std_rewards_array, kernel, mode='valid')[::downsample_rate]
        steps = np.linspace(1, total_steps, len(smoothed_rewards))

        color = plt.cm.tab10(i % 10)

        plt.plot(steps, smoothed_rewards, color=color, linestyle='-', alpha=0.8,
                 label=f'{group_name} Smoothed Training Avg')
        plt.fill_between(steps,
                         smoothed_rewards - smoothed_std,
                         smoothed_rewards + smoothed_std,
                         color=color, alpha=0.25, label=f'{group_name} Training Std Dev')
        plt.axhline(avg_test_reward, color="black", linestyle=linestyles[i % len(linestyles)],
                    label=f'{group_name} Test Avg', alpha=0.9)
        plt.text(total_steps * 0.98, avg_test_reward + 2,
                 f'{avg_test_reward:.2f}', color="black", fontsize=10,
                 horizontalalignment='right', verticalalignment='bottom')

    plt.title("Training Results Across Experiment Groups")
    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.grid()
    plot_path = os.path.join(save_dir, "aggregated_training_results.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Aggregated training results saved to {plot_path}")
