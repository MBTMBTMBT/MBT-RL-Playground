from scipy.ndimage import gaussian_filter1d

if __name__ == '__main__':
    import numpy as np
    import plotly.graph_objs as go
    import plotly.subplots as sp
    import plotly.io as pio
    import os
    from parallel_training import run_all_experiments

    # General experiment parameters
    experiment_name = "MountainCar_Experiments"
    save_dir = f"./experiments/{experiment_name}/"
    os.makedirs(save_dir, exist_ok=True)

    # Define experiment groups
    experiment_groups = [
        {
            "group_name": "MC-50-10",
            "env_id": "Custom-MountainCar",
            "train_env_params": [
                {
                    "render_mode": "rgb_array",
                    "goal_velocity": 0,
                    "custom_gravity": g,
                    "max_episode_steps": 500,
                    "reward_type": 'default',
                } for g in np.linspace(0.0010, 0.0050, 5)
            ],
            "test_per_num_steps": int(0.1e6),
            "test_runs": 10,
            "test_env_params": {
                "render_mode": "rgb_array",
                "goal_velocity": 0,
                "custom_gravity": 0.0050,
                "max_episode_steps": 500,
                "reward_type": 'default',
            },
            "state_space": [
                {'type': 'continuous', 'range': (-1.2, 0.6), 'bins': 64},  # Position
                {'type': 'continuous', 'range': (-0.07, 0.07), 'bins': 64}  # Velocity
            ],
            "action_space": [{'type': 'discrete', 'bins': 3}],
            "alpha": 0.1,
            "gamma": 0.99,
            "epsilon_start": 0.25,
            "epsilon_end": 0.05,
            "total_steps": int(25e6),
            "runs": 5,
        },
        {
            "group_name": "MC-50-15",
            "env_id": "Custom-MountainCar",
            "train_env_params": [
                {
                    "render_mode": "rgb_array",
                    "goal_velocity": 0,
                    "custom_gravity": g,
                    "max_episode_steps": 500,
                    "reward_type": 'default',
                } for g in np.linspace(0.0015, 0.0050, 5)
            ],
            "test_per_num_steps": int(0.1e6),
            "test_runs": 10,
            "test_env_params": {
                "render_mode": "rgb_array",
                "goal_velocity": 0,
                "custom_gravity": 0.0050,
                "max_episode_steps": 500,
                "reward_type": 'default',
            },
            "state_space": [
                {'type': 'continuous', 'range': (-1.2, 0.6), 'bins': 64},  # Position
                {'type': 'continuous', 'range': (-0.07, 0.07), 'bins': 64}  # Velocity
            ],
            "action_space": [{'type': 'discrete', 'bins': 3}],
            "alpha": 0.1,
            "gamma": 0.99,
            "epsilon_start": 0.25,
            "epsilon_end": 0.05,
            "total_steps": int(25e6),
            "runs": 5,
        },
        {
            "group_name": "MC-50-25",
            "env_id": "Custom-MountainCar",
            "train_env_params": [
                {
                    "render_mode": "rgb_array",
                    "goal_velocity": 0,
                    "custom_gravity": g,
                    "max_episode_steps": 500,
                    "reward_type": 'default',
                } for g in np.linspace(0.0025, 0.0050, 5)
            ],
            "test_per_num_steps": int(0.1e6),
            "test_runs": 10,
            "test_env_params": {
                "render_mode": "rgb_array",
                "goal_velocity": 0,
                "custom_gravity": 0.0050,
                "max_episode_steps": 500,
                "reward_type": 'default',
            },
            "state_space": [
                {'type': 'continuous', 'range': (-1.2, 0.6), 'bins': 64},  # Position
                {'type': 'continuous', 'range': (-0.07, 0.07), 'bins': 64}  # Velocity
            ],
            "action_space": [{'type': 'discrete', 'bins': 3}],
            "alpha": 0.1,
            "gamma": 0.99,
            "epsilon_start": 0.25,
            "epsilon_end": 0.05,
            "total_steps": int(25e6),
            "runs": 5,
        },
        {
            "group_name": "MC-50-nc",
            "env_id": "Custom-MountainCar",
            "train_env_params": [
                {
                    "render_mode": "rgb_array",
                    "goal_velocity": 0,
                    "custom_gravity": g,
                    "max_episode_steps": 500,
                    "reward_type": 'default',
                } for g in np.linspace(0.0050, 0.0050, 1)
            ],
            "test_per_num_steps": int(0.1e6),
            "test_runs": 10,
            "test_env_params": {
                "render_mode": "rgb_array",
                "goal_velocity": 0,
                "custom_gravity": 0.0050,
                "max_episode_steps": 500,
                "reward_type": 'default',
            },
            "state_space": [
                {'type': 'continuous', 'range': (-1.2, 0.6), 'bins': 64},  # Position
                {'type': 'continuous', 'range': (-0.07, 0.07), 'bins': 64}  # Velocity
            ],
            "action_space": [{'type': 'discrete', 'bins': 3}],
            "alpha": 0.1,
            "gamma": 0.99,
            "epsilon_start": 0.25,
            "epsilon_end": 0.05,
            "total_steps": int(25e6),
            "runs": 5,
        },
    ]

    # Run all experiments
    max_workers = 20  # Number of parallel processes
    aggregated_results = run_all_experiments(experiment_groups, save_dir, max_workers)

    # Create a figure object
    fig = sp.make_subplots(rows=1, cols=1, subplot_titles=["Training Results Across Experiment Groups"])

    for i, (group_name, (avg_rewards, std_rewards, steps, avg_test_reward)) in enumerate(aggregated_results.items()):
        # Plot training curve
        sigma = 3  # Standard deviation for Gaussian kernel
        avg_rewards = gaussian_filter1d(avg_rewards, sigma=sigma)
        std_rewards = gaussian_filter1d(std_rewards, sigma=sigma)

        trace = go.Scatter(
            x=steps,
            y=avg_rewards,
            mode='lines+markers',
            name=f'{group_name} Smoothed Training Avg',
            line_shape='spline'  # Smooth curve
        )

        # Plot standard deviation area
        trace_std = go.Scatter(
            x=list(steps) + list(steps)[::-1],
            y=[v + s for v, s in zip(avg_rewards, std_rewards)] + [v - s for v, s in zip(avg_rewards, std_rewards)][
                                                                  ::-1],
            fill='toself',
            fillcolor='rgba(0,100,80,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name=f'{group_name} Training Std Dev',
            showlegend=False
        )

        # Plot test average line
        trace_test = go.Scatter(
            x=[min(steps), max(steps)],
            y=[avg_test_reward, avg_test_reward],
            mode='lines',
            name=f'{group_name} Test Avg',
            line=dict(dash=['dash', 'dot', 'dashdot', 'longdash'][i % 4], color='black')
        )

        # Add traces to the figure
        fig.add_trace(trace, row=1, col=1)
        fig.add_trace(trace_std, row=1, col=1)
        fig.add_trace(trace_test, row=1, col=1)

        # Add annotation for test average
        fig.add_annotation(
            x=max(steps) * 0.98,
            y=avg_test_reward,
            text=f'{avg_test_reward:.2f}',
            showarrow=False,
            xanchor='right',
            yanchor='bottom'
        )

    # Update figure layout
    fig.update_layout(
        # title="Training Results Across Experiment Groups",
        xaxis_title="Steps",
        yaxis_title="Average Reward",
        legend_title="Groups",
        template="plotly_white"
    )

    print("Saving training curve...")
    # Display figure and save as PNG
    plotly_png_path = os.path.join(save_dir, "aggregated_training_results_plotly.png")
    pio.write_image(fig, plotly_png_path, format='png', scale=5, width=1200, height=675)
    print(f"Aggregated training results saved to {plotly_png_path}")
