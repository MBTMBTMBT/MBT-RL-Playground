import numpy as np
import gymnasium as gym
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import plotly.graph_objs as go
import plotly.subplots as sp
import plotly.io as pio
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import os
import pandas as pd

from custom_mountain_car import CustomMountainCarEnv
from q_table_agent import QTableAgent


CUSTOM_ENVS = {
    "Custom-MountainCar": CustomMountainCarEnv,
}


# Generate GIF for the final test episode
def generate_test_gif(frames, gif_path):
    fig, ax = plt.subplots()
    ax.axis('off')  # Turn off axis
    img = ax.imshow(frames[0])

    def update(frame):
        img.set_data(frame)
        return [img]

    ani = FuncAnimation(fig, update, frames=frames, interval=50, blit=True)
    ani.save(gif_path, dpi=90, writer="pillow")
    plt.close(fig)
    print(f"GIF saved to {gif_path}")


# Single experiment runner
def run_experiment(args):
    group, run_id, save_dir = args
    env_id = group['env_id']
    total_steps = group["total_steps"]
    alpha = group["alpha"]
    gamma = group["gamma"]
    epsilon_start = group["epsilon_start"]
    epsilon_end = group["epsilon_end"]
    state_space = group["state_space"]
    action_space = group["action_space"]
    group_name = group["group_name"]
    training_rewards = []
    test_rewards = []
    kl_divergences = []
    current_steps = 0
    epsilon_decay = (epsilon_start - epsilon_end) / total_steps
    epsilon = epsilon_start

    # Create QTableAgent
    agent = QTableAgent(state_space, action_space)
    old_agent = None

    # Initialize CartPole environment
    if env_id in CUSTOM_ENVS:
        envs = [CUSTOM_ENVS[env_id](**(group["train_env_params"][i])) for i in range(len(group["train_env_params"]))]
        test_env = CUSTOM_ENVS[env_id](**group["test_env_params"])
    else:
        envs = [gym.make(env_id, render_mode="rgb_array")]
        test_env = gym.make(env_id, render_mode="rgb_array")
    test_per_num_steps = group["test_per_num_steps"]
    test_runs = group["test_runs"]

    curriculum_steps = total_steps // len(envs)
    # Training
    with tqdm(total=total_steps, desc=f"[{group_name}] Run {run_id}", leave=False) as pbar:
        avg_test_reward = 0.0  # Initialize avg_test_reward to avoid UnboundLocalError
        average_kl = 0.0
        env = envs[0]
        while current_steps < total_steps:
            state, _ = env.reset()
            total_reward = 0
            done = False

            while not done:
                if current_steps % curriculum_steps == 0:
                    epsilon_decay = (epsilon_start - epsilon_end) / curriculum_steps
                    epsilon = epsilon_start
                if current_steps % curriculum_steps == 0 and current_steps > 0 and len(envs) > 1:
                    if current_steps // curriculum_steps <= len(envs) - 1:
                        env = envs[current_steps // curriculum_steps]
                    state, _ = env.reset()
                    q_table_path = os.path.join(
                        save_dir,
                        f"{group_name}_run_{run_id}_q_table_{current_steps // curriculum_steps}.csv",
                    )
                    agent.save_q_table(q_table_path)
                    old_agent = agent.clone()

                if np.random.random() < epsilon:
                    action = [np.random.choice([0, 1])]
                else:
                    probabilities = agent.get_action_probabilities(state, strategy="softmax")
                    action = [np.argmax(probabilities)]

                next_state, reward, done, truncated, _ = env.step(action[0])
                agent.update(state, action, reward, next_state, alpha, gamma)

                state = next_state
                total_reward += reward
                current_steps += 1
                epsilon = max(epsilon_end, epsilon - epsilon_decay)
                pbar.update(1)

                if old_agent is None and current_steps >= 100:
                    old_agent = agent.clone()

                # Periodic testing
                if current_steps % test_per_num_steps == 0:
                    periodic_test_rewards = []
                    for _ in range(test_runs):
                        test_state, _ = test_env.reset()
                        test_total_reward = 0
                        test_done = False
                        while not test_done:
                            test_action = [np.argmax(agent.get_action_probabilities(test_state, strategy="softmax"))]
                            test_next_state, test_reward, test_done, test_truncated, _ = test_env.step(test_action[0])
                            test_state = test_next_state
                            test_total_reward += test_reward
                            if test_done or test_truncated:
                                break
                        periodic_test_rewards.append(test_total_reward)
                    avg_test_reward = np.mean(periodic_test_rewards)
                    recent_avg = np.mean([r for _, r in training_rewards[-10:]])
                    try:
                        policy = QTableAgent.compute_action_probabilities(agent.query_q_table([]), strategy="softmax")
                        old_policy = QTableAgent.compute_action_probabilities(old_agent.query_q_table([]), strategy="softmax")
                        average_kl = QTableAgent.compute_average_kl_divergence_between_dfs(policy, old_policy, visit_threshold=0)
                    except ValueError:
                        average_kl = 0.0
                    pbar.set_description(f"[{group_name}] Run {run_id} | "
                                         f"Epsilon: {epsilon:.4f} | "
                                         f"Recent Avg Reward: {recent_avg:.2f} | "
                                         f"Stage KL Divergence: {average_kl:.2f} | "
                                         f"Avg Test Reward: {avg_test_reward:.2f}")
                    test_rewards.append((current_steps, avg_test_reward))
                    kl_divergences.append((current_steps, average_kl))

                if done or truncated:
                    training_rewards.append((current_steps, total_reward))
                    recent_avg = np.mean([r for _, r in training_rewards[-10:]])
                    pbar.set_description(f"[{group_name}] Run {run_id} | "
                                         f"Epsilon: {epsilon:.4f} | "
                                         f"Recent Avg Reward: {recent_avg:.2f} | "
                                         f"Stage KL Divergence: {average_kl:.2f} | "
                                         f"Avg Test Reward: {avg_test_reward:.2f}")
                    break

    # Save Q-Table and training data
    q_table_path = os.path.join(save_dir, f"{group_name}_run_{run_id}_q_table_final.csv")
    agent.save_q_table(q_table_path)

    # Final Testing and GIF generation
    frames = []
    final_test_rewards = []  # Initialize final_test_rewards to store final test results
    for episode in range(test_runs):
        state, _ = test_env.reset()
        total_reward = 0
        done = False

        while not done:
            probabilities = agent.get_action_probabilities(state, strategy="greedy")
            action = [np.argmax(probabilities)]
            state, reward, done, truncated, _ = test_env.step(action[0])
            total_reward += reward

            # Collect frames for the final test episode
            if episode == 0:
                frames.append(test_env.render())

            if done or truncated:
                break
        final_test_rewards.append(total_reward)
    if current_steps % test_per_num_steps >= test_per_num_steps // 2:
        test_rewards.append((current_steps, np.mean(final_test_rewards)))  # Append the final test result to test_rewards
        try:
            policy = QTableAgent.compute_action_probabilities(agent.query_q_table([]), strategy="softmax")
            old_policy = QTableAgent.compute_action_probabilities(old_agent.query_q_table([]), strategy="softmax")
            average_kl = QTableAgent.compute_average_kl_divergence_between_dfs(policy, old_policy, visit_threshold=0)
        except ValueError:
            average_kl = 0.0
        kl_divergences.append((current_steps, average_kl))

    # Save GIF for the first test episode
    gif_path = os.path.join(save_dir, f"{group_name}_run_{run_id}_test.gif")
    generate_test_gif(frames, gif_path)

    # Save updated training data with final test results
    training_data_path = os.path.join(save_dir, f"{group_name}_run_{run_id}_training_data.csv")
    pd.DataFrame(test_rewards, columns=["Step", "Avg Test Reward"]).to_csv(training_data_path, index=False)

    return test_rewards, kl_divergences, np.mean(final_test_rewards)


# Aggregating results for consistent step-based plotting
def run_all_experiments(experiment_groups, save_dir, max_workers):
    tasks = []
    for group in experiment_groups:
        print(group)
        for run_id in range(group["runs"]):
            tasks.append((
                group, run_id, save_dir,
            ))

    # Execute tasks in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(run_experiment, tasks))

    # Group results by experiment group
    group_results = {group["group_name"]: [] for group in experiment_groups}
    for task, result in zip(tasks, results):
        group, run_id, save_dir, = task
        group_name = group["group_name"]
        group_results[group_name].append(result)

    # Aggregate results for each group
    aggregated_results = {}
    for group in experiment_groups:
        group_name = group["group_name"]
        total_steps = group["total_steps"]
        all_training_results, all_kl_divergences, all_testing_results = zip(*group_results[group_name])

        steps, rewards, kls = [], [], []
        for step_rewards in all_training_results:
            step, reward = zip(*step_rewards)
            steps.append(step)
            rewards.append(reward)
        avg_rewards = np.mean(rewards, axis=0)
        std_rewards = np.std(rewards, axis=0)
        steps = np.array(steps[0])
        for step_rewards in all_kl_divergences:
            step, reward = zip(*step_rewards)
            kls.append(reward)
        avg_kls = np.mean(kls, axis=0)
        std_kls = np.std(kls, axis=0)

        avg_testing_rewards = np.mean(all_testing_results)

        aggregated_results[group_name] = (avg_rewards, std_rewards, avg_kls, std_kls, steps, avg_testing_rewards)

    return aggregated_results


if __name__ == '__main__':
    # General experiment parameters
    experiment_name = "_MountainCar_Experiments"
    save_dir = f"./experiments/{experiment_name}/"
    os.makedirs(save_dir, exist_ok=True)

    # Define experiment groups
    experiment_groups = [
        {
            "group_name": "MC-test-1",
            "env_id": "Custom-MountainCar",
            "train_env_params": [
                {
                    "render_mode": "rgb_array",
                    "goal_velocity": 0,
                    "custom_gravity": 0.0005,
                    "max_episode_steps": 200,
                    "reward_type": 'progress',
                },
                {
                    "render_mode": "rgb_array",
                    "goal_velocity": 0,
                    "custom_gravity": 0.0010,
                    "max_episode_steps": 200,
                    "reward_type": 'progress',
                },
                {
                    "render_mode": "rgb_array",
                    "goal_velocity": 0,
                    "custom_gravity": 0.0015,
                    "max_episode_steps": 200,
                    "reward_type": 'progress',
                },
            ],
            "test_per_num_steps": int(0.001e6),
            "test_runs": 1,
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
            "total_steps": int(1e6),
            "runs": 3,
        },
        {
            "group_name": "MC-test-2",
            "env_id": "Custom-MountainCar",
            "train_env_params": [
                {
                    "render_mode": "rgb_array",
                    "goal_velocity": 0,
                    "custom_gravity": 0.0015,
                    "max_episode_steps": 200,
                    "reward_type": 'progress',
                },
            ],
            "test_per_num_steps": int(0.001e6),
            "test_runs": 1,
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
            "total_steps": int(1e6),
            "runs": 3,
        },
    ]

    # Run all experiments
    max_workers = 12  # Number of parallel processes
    aggregated_results = run_all_experiments(experiment_groups, save_dir, max_workers)

    # Create a figure object
    fig = sp.make_subplots(rows=1, cols=1, subplot_titles=["Training Results Across Experiment Groups"])

    for i, (group_name, (avg_rewards, std_rewards, avg_kls, std_kls, steps, avg_test_reward)) in enumerate(aggregated_results.items()):
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
