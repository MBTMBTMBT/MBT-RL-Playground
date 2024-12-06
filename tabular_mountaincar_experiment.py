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
            "group_name": "MC-50-nc",
            "env_id": "Custom-MountainCar",
            "train_env_params": [
                {
                    "render_mode": "rgb_array",
                    "goal_velocity": 0,
                    "custom_gravity": 0.0050,
                    "max_episode_steps": 200,
                    "goal_position": 0.5,
                    "reward_type": 'default',
                }
            ],
            "test_per_num_steps": int(0.1e6),
            "test_runs": 10,
            "test_env_params": {
                "render_mode": "rgb_array",
                "goal_velocity": 0,
                "custom_gravity": 0.0050,
                "max_episode_steps": 200,
                "goal_position": 0.5,
                "reward_type": 'default',
            },
            "state_space": [
                {'type': 'continuous', 'range': (-1.2, 0.6), 'bins': 256},  # Position
                {'type': 'continuous', 'range': (-0.14, 0.14), 'bins': 128}  # Velocity
            ],
            "action_space": [{'type': 'discrete', 'bins': 3}],
            "alpha": 0.1,
            "gamma": 0.99,
            "epsilon_start": 0.25,
            "epsilon_end": 0.25,
            "total_steps": int(10e6),
            "runs": 4,
        },
        {
            "group_name": "MC-25-nc",
            "env_id": "Custom-MountainCar",
            "train_env_params": [
                {
                    "render_mode": "rgb_array",
                    "goal_velocity": 0,
                    "custom_gravity": 0.0050,
                    "max_episode_steps": 200,
                    "goal_position": 0.25,
                    "reward_type": 'default',
                }
            ],
            "test_per_num_steps": int(0.1e6),
            "test_runs": 10,
            "test_env_params": {
                "render_mode": "rgb_array",
                "goal_velocity": 0,
                "custom_gravity": 0.0050,
                "max_episode_steps": 200,
                "goal_position": 0.25,
                "reward_type": 'default',
            },
            "state_space": [
                {'type': 'continuous', 'range': (-1.2, 0.6), 'bins': 256},  # Position
                {'type': 'continuous', 'range': (-0.14, 0.14), 'bins': 128}  # Velocity
            ],
            "action_space": [{'type': 'discrete', 'bins': 3}],
            "alpha": 0.1,
            "gamma": 0.99,
            "epsilon_start": 0.25,
            "epsilon_end": 0.25,
            "total_steps": int(10e6),
            "runs": 4,
        },
        {
            "group_name": "MC-00-nc",
            "env_id": "Custom-MountainCar",
            "train_env_params": [
                {
                    "render_mode": "rgb_array",
                    "goal_velocity": 0,
                    "custom_gravity": 0.0050,
                    "max_episode_steps": 200,
                    "goal_position": -0.0,
                    "reward_type": 'default',
                }
            ],
            "test_per_num_steps": int(0.1e6),
            "test_runs": 10,
            "test_env_params": {
                "render_mode": "rgb_array",
                "goal_velocity": 0,
                "custom_gravity": 0.0050,
                "max_episode_steps": 200,
                "goal_position": -0.0,
                "reward_type": 'default',
            },
            "state_space": [
                {'type': 'continuous', 'range': (-1.2, 0.6), 'bins': 256},  # Position
                {'type': 'continuous', 'range': (-0.14, 0.14), 'bins': 128}  # Velocity
            ],
            "action_space": [{'type': 'discrete', 'bins': 3}],
            "alpha": 0.1,
            "gamma": 0.99,
            "epsilon_start": 0.25,
            "epsilon_end": 0.25,
            "total_steps": int(10e6),
            "runs": 4,
        },
    ]

    # Run all experiments
    max_workers = 12  # Number of parallel processes
    aggregated_results = run_all_experiments(experiment_groups, save_dir, max_workers)

    # Define color map to ensure consistent colors across figures
    color_map = {}
    colors = ['rgba(31, 119, 180, 1)', 'rgba(255, 127, 14, 1)', 'rgba(44, 160, 44, 1)', 'rgba(214, 39, 40, 1)',
              'rgba(148, 103, 189, 1)', 'rgba(140, 86, 75, 1)', 'rgba(227, 119, 194, 1)', 'rgba(127, 127, 127, 1)',
              'rgba(188, 189, 34, 1)', 'rgba(23, 190, 207, 1)']

    # Create a figure object for average rewards
    fig = sp.make_subplots(rows=1, cols=1, subplot_titles=["Training Results Across Experiment Groups"])

    for i, (group_name, (avg_rewards, std_rewards, avg_kls, std_kls, steps, avg_test_reward)) in enumerate(
            aggregated_results.items()):
        # Assign color for each group
        if group_name not in color_map:
            color_map[group_name] = colors[i % len(colors)]
        color = color_map[group_name]

        # Apply Gaussian smoothing
        sigma = 1  # Standard deviation for Gaussian kernel
        avg_rewards_smoothed = gaussian_filter1d(avg_rewards, sigma=sigma)
        std_rewards_smoothed = gaussian_filter1d(std_rewards, sigma=sigma)

        # Plot training curve
        trace = go.Scatter(
            x=steps,
            y=avg_rewards_smoothed,
            mode='lines+markers',
            name=f'{group_name} Smoothed Training Avg',
            line_shape='spline',  # Smooth curve
            line=dict(color=color)
        )

        # Plot standard deviation area
        trace_std = go.Scatter(
            x=list(steps) + list(steps)[::-1],
            y=[v + s for v, s in zip(avg_rewards_smoothed, std_rewards_smoothed)] + [v - s for v, s in
                                                                                     zip(avg_rewards_smoothed,
                                                                                         std_rewards_smoothed)][::-1],
            fill='toself',
            fillcolor=color.replace('1)', '0.2)'),  # Adjust alpha for fill color
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

    # Create a figure object for KL divergence
    fig_kl = sp.make_subplots(rows=1, cols=1, subplot_titles=["KL Divergence Across Experiment Groups"])

    for i, (group_name, (avg_rewards, std_rewards, avg_kls, std_kls, steps, avg_test_reward)) in enumerate(
            aggregated_results.items()):
        # Use the same color as the rewards plot
        color = color_map[group_name]

        # Apply Gaussian smoothing
        # sigma = 1  # Standard deviation for Gaussian kernel
        # avg_kls_smoothed = gaussian_filter1d(avg_kls, sigma=sigma)
        # std_kls_smoothed = gaussian_filter1d(std_kls, sigma=sigma)
        avg_kls_smoothed = avg_kls
        std_kls_smoothed = std_kls

        # Plot KL divergence curve
        trace_kl = go.Scatter(
            x=steps,
            y=avg_kls_smoothed,
            mode='lines+markers',
            name=f'{group_name} Smoothed KL Avg',
            line_shape='spline',  # Smooth curve
            line=dict(color=color)
        )

        # Plot standard deviation area for KL
        trace_kl_std = go.Scatter(
            x=list(steps) + list(steps)[::-1],
            y=[v + s for v, s in zip(avg_kls_smoothed, std_kls_smoothed)] + [v - s for v, s in
                                                                             zip(avg_kls_smoothed, std_kls_smoothed)][
                                                                            ::-1],
            fill='toself',
            fillcolor=color.replace('1)', '0.2)'),  # Adjust alpha for fill color
            line=dict(color='rgba(255,255,255,0)'),
            name=f'{group_name} KL Std Dev',
            showlegend=False
        )

        # Add traces to the figure
        fig_kl.add_trace(trace_kl, row=1, col=1)
        fig_kl.add_trace(trace_kl_std, row=1, col=1)

    # Update KL figure layout
    fig_kl.update_layout(
        xaxis_title="Steps",
        yaxis_title="KL Divergence",
        legend_title="Groups",
        template="plotly_white"
    )

    print("Saving KL divergence curve...")
    # Display figure and save as PNG
    plotly_kl_png_path = os.path.join(save_dir, "aggregated_kl_results_plotly.png")
    pio.write_image(fig_kl, plotly_kl_png_path, format='png', scale=5, width=1200, height=675)
    print(f"Aggregated KL divergence results saved to {plotly_kl_png_path}")
