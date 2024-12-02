import numpy as np
import gymnasium as gym
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
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


# Helper function to align training rewards using step
def align_training_rewards_with_steps(all_training_results, total_steps):
    """
    Align training rewards to a fixed number of steps for consistent plotting.

    :param all_training_results: List of step-reward tuples for all runs.
    :param total_steps: The total number of steps in the experiment.
    :return: A numpy array of aligned rewards averaged across all runs.
    """
    aligned_rewards = []
    for step_rewards in all_training_results:
        steps, rewards = zip(*step_rewards)
        # Interpolate rewards to align with total_steps
        interpolated_rewards = np.interp(
            np.linspace(1, total_steps, total_steps),
            steps,
            rewards
        )
        aligned_rewards.append(interpolated_rewards)
    return np.mean(aligned_rewards, axis=0), np.std(aligned_rewards, axis=0)  # Return mean and std


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
    current_steps = 0
    epsilon_decay = (epsilon_start - epsilon_end) / total_steps
    epsilon = epsilon_start

    # Create QTableAgent
    agent = QTableAgent(state_space, action_space)

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
                    pbar.set_description(f"[{group_name}] Run {run_id} | "
                                         f"Epsilon: {epsilon:.4f} | "
                                         f"Recent Avg Reward: {recent_avg:.2f} | "
                                         f"Avg Test Reward: {avg_test_reward:.2f}")
                    test_rewards.append((current_steps, avg_test_reward))

                if done or truncated:
                    training_rewards.append((current_steps, total_reward))
                    recent_avg = np.mean([r for _, r in training_rewards[-10:]])
                    pbar.set_description(f"[{group_name}] Run {run_id} | "
                                         f"Epsilon: {epsilon:.4f} | "
                                         f"Recent Avg Reward: {recent_avg:.2f} | "
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
    test_rewards.append((current_steps, np.mean(final_test_rewards)))  # Append the final test result to test_rewards

    # Save GIF for the first test episode
    gif_path = os.path.join(save_dir, f"{group_name}_run_{run_id}_test.gif")
    generate_test_gif(frames, gif_path)

    # Save updated training data with final test results
    training_data_path = os.path.join(save_dir, f"{group_name}_run_{run_id}_training_data.csv")
    pd.DataFrame(test_rewards, columns=["Step", "Avg Test Reward"]).to_csv(training_data_path, index=False)

    return test_rewards, np.mean(final_test_rewards)


# Aggregating results for consistent step-based plotting
def run_all_experiments(experiment_groups, save_dir, max_workers):
    tasks = []
    for group in experiment_groups:
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
        all_training_results, all_testing_results = zip(*group_results[group_name])

        # Align rewards with steps and compute mean and std
        avg_training_rewards, std_training_rewards = align_training_rewards_with_steps(
            all_training_results, total_steps
        )
        avg_testing_rewards = np.mean(all_testing_results)

        aggregated_results[group_name] = (avg_training_rewards, std_training_rewards, avg_testing_rewards)

    return aggregated_results


if __name__ == '__main__':
    # General experiment parameters
    experiment_name = "_MountainCar_Experiments"
    save_dir = f"./experiments/{experiment_name}/"
    os.makedirs(save_dir, exist_ok=True)

    # Define experiment groups
    experiment_groups = [
        {
            "group_name": "MC-15-3",
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
            "total_steps": int(1e6),
            "runs": 3,
        },
        {
            "group_name": "MC-15-2",
            "env_id": "Custom-MountainCar",
            "train_env_params": [
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
            "total_steps": int(1e6),
            "runs": 3,
        },
        {
            "group_name": "MC-15-1",
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
            "total_steps": int(1e6),
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
