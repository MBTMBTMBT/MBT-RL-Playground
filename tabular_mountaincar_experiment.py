from scipy.ndimage import gaussian_filter1d

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
            "group_name": "MC-25-05",
            "env_id": "Custom-MountainCar",
            "train_env_params": [
                {
                    "render_mode": "rgb_array",
                    "goal_velocity": 0,
                    "custom_gravity": g,
                    "max_episode_steps": 200,
                    "reward_type": 'default',
                } for g in np.linspace(0.0005, 0.0025, 3)
            ],
            "test_per_num_steps": int(0.1e6),
            "test_runs": 10,
            "test_env_params": {
                "render_mode": "rgb_array",
                "goal_velocity": 0,
                "custom_gravity": 0.0025,
                "max_episode_steps": 200,
                "reward_type": 'default',
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
            "total_steps": int(50e6),
            "runs": 3,
        },
        {
            "group_name": "MC-25-10",
            "env_id": "Custom-MountainCar",
            "train_env_params": [
                {
                    "render_mode": "rgb_array",
                    "goal_velocity": 0,
                    "custom_gravity": g,
                    "max_episode_steps": 200,
                    "reward_type": 'default',
                } for g in np.linspace(0.0010, 0.0025, 3)
            ],
            "test_per_num_steps": int(0.1e6),
            "test_runs": 10,
            "test_env_params": {
                "render_mode": "rgb_array",
                "goal_velocity": 0,
                "custom_gravity": 0.0025,
                "max_episode_steps": 200,
                "reward_type": 'default',
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
            "total_steps": int(50e6),
            "runs": 3,
        },
        {
            "group_name": "MC-25-15",
            "env_id": "Custom-MountainCar",
            "train_env_params": [
                {
                    "render_mode": "rgb_array",
                    "goal_velocity": 0,
                    "custom_gravity": g,
                    "max_episode_steps": 200,
                    "reward_type": 'default',
                } for g in np.linspace(0.0015, 0.0025, 3)
            ],
            "test_per_num_steps": int(0.1e6),
            "test_runs": 10,
            "test_env_params": {
                "render_mode": "rgb_array",
                "goal_velocity": 0,
                "custom_gravity": 0.0025,
                "max_episode_steps": 200,
                "reward_type": 'default',
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
            "total_steps": int(50e6),
            "runs": 3,
        },
        {
            "group_name": "MC-25-nc",
            "env_id": "Custom-MountainCar",
            "train_env_params": [
                {
                    "render_mode": "rgb_array",
                    "goal_velocity": 0,
                    "custom_gravity": g,
                    "max_episode_steps": 200,
                    "reward_type": 'default',
                } for g in np.linspace(0.0025, 0.0025, 1)
            ],
            "test_per_num_steps": int(0.1e6),
            "test_runs": 10,
            "test_env_params": {
                "render_mode": "rgb_array",
                "goal_velocity": 0,
                "custom_gravity": 0.0025,
                "max_episode_steps": 200,
                "reward_type": 'default',
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
            "total_steps": int(50e6),
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
    target_points = 2048

    for i, (group_name, (avg_rewards, std_rewards, avg_test_reward)) in enumerate(aggregated_results.items()):
        total_steps = experiment_groups[i]["total_steps"]
        total_points = len(avg_rewards)
        downsample_rate = max(1, total_points // target_points)

        # Step 1: Apply Gaussian smoothing for anti-aliasing
        avg_rewards_array = np.array(avg_rewards)
        std_rewards_array = np.array(std_rewards)
        sigma = downsample_rate / 2  # Adjust sigma to reduce high-frequency noise
        smoothed_rewards = gaussian_filter1d(avg_rewards_array, sigma=sigma)
        smoothed_std = gaussian_filter1d(std_rewards_array, sigma=sigma)

        # Step 2: Downsample the smoothed data with averaging
        avg_rewards_downsampled = [np.mean(smoothed_rewards[i:i + downsample_rate]) for i in
                                   range(0, len(smoothed_rewards), downsample_rate)]
        std_rewards_downsampled = [np.mean(smoothed_std[i:i + downsample_rate]) for i in
                                   range(0, len(smoothed_std), downsample_rate)]
        steps = np.linspace(1, total_steps, len(avg_rewards_downsampled))

        # Step 3: Apply Gaussian smoothing after downsampling
        final_sigma = 5  # Separate sigma for post-downsampling smoothing
        final_smoothed_rewards = gaussian_filter1d(avg_rewards_downsampled, sigma=final_sigma)
        final_smoothed_std = gaussian_filter1d(std_rewards_downsampled, sigma=final_sigma)

        # Plotting
        color = plt.cm.tab10(i % 10)

        plt.plot(steps, final_smoothed_rewards, color=color, linestyle='-', alpha=0.8,
                 label=f'{group_name} Smoothed Training Avg')
        plt.fill_between(steps,
                         np.array(final_smoothed_rewards) - np.array(final_smoothed_std),
                         np.array(final_smoothed_rewards) + np.array(final_smoothed_std),
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
