import numpy as np
import gymnasium as gym
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import os
import pandas as pd
from q_table_agent import QTableAgent


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
    total_steps, alpha, gamma, epsilon_start, epsilon_end, state_space, action_space, group_name, run_id, save_dir = args
    epsilon_decay = (epsilon_start - epsilon_end) / total_steps
    step_rewards = []
    test_rewards = []
    current_steps = 0
    epsilon = epsilon_start

    # Create QTableAgent
    agent = QTableAgent(state_space, action_space)

    # Initialize CartPole environment
    env = gym.make('Acrobot-v1')

    # Training
    with tqdm(total=total_steps, desc=f"[{group_name}] Run {run_id + 1}", leave=False) as pbar:
        while current_steps < total_steps:
            state, _ = env.reset()
            total_reward = 0
            done = False

            while not done:
                if np.random.random() < epsilon:
                    action = [np.random.choice([0, 1])]
                else:
                    probabilities = agent.get_action_probabilities(state, strategy="greedy")
                    action = [np.argmax(probabilities)]

                next_state, reward, done, truncated, _ = env.step(action[0])
                agent.update(state, action, reward, next_state, alpha, gamma)

                state = next_state
                total_reward += reward
                current_steps += 1
                epsilon = max(epsilon_end, epsilon - epsilon_decay)
                pbar.update(1)

                if done or truncated:
                    step_rewards.append((current_steps, total_reward))
                    recent_avg = np.mean([r for _, r in step_rewards[-10:]])
                    pbar.set_description(f"[{group_name}] Run {run_id + 1} | "
                                         f"Epsilon: {epsilon:.4f} | "
                                         f"Recent Avg Reward: {recent_avg:.2f}")
                    break

    # Save Q-Table and training data
    q_table_path = os.path.join(save_dir, f"{group_name}_run_{run_id + 1}_q_table.csv")
    agent.save_q_table(q_table_path)
    training_data_path = os.path.join(save_dir, f"{group_name}_run_{run_id + 1}_training_data.csv")
    pd.DataFrame(step_rewards, columns=["Step", "Reward"]).to_csv(training_data_path, index=False)

    # Testing and GIF generation
    env = gym.make('Acrobot-v1', render_mode="rgb_array")
    frames = []
    for episode in range(20):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            probabilities = agent.get_action_probabilities(state, strategy="greedy")
            action = [np.argmax(probabilities)]
            state, reward, done, truncated, _ = env.step(action[0])
            total_reward += reward

            # Collect frames for the final test episode
            if episode == 0:
                frames.append(env.render())

            if done or truncated:
                break
        test_rewards.append(total_reward)

    # Save GIF for the first test episode
    gif_path = os.path.join(save_dir, f"{group_name}_run_{run_id + 1}_test.gif")
    generate_test_gif(frames, gif_path)

    return step_rewards, np.mean(test_rewards)


# Aggregating results for consistent step-based plotting
def run_all_experiments(experiment_groups, save_dir, max_workers):
    tasks = []
    for group in experiment_groups:
        for run_id in range(group["runs"]):
            tasks.append((
                group["total_steps"], group["alpha"], group["gamma"],
                group["epsilon_start"], group["epsilon_end"],
                group["state_space"], group["action_space"],
                group["group_name"], run_id, save_dir
            ))

    # Execute tasks in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(run_experiment, tasks))

    # Group results by experiment group
    group_results = {group["group_name"]: [] for group in experiment_groups}
    for task, result in zip(tasks, results):
        _, _, _, _, _, _, _, group_name, run_id, _ = task
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
    experiment_name = "Acrobot_Experiments"
    save_dir = f"./experiments/{experiment_name}/"
    os.makedirs(save_dir, exist_ok=True)

    # Define experiment groups
    experiment_groups = [
        {
            "group_name": "16_bins",
            "state_space": [
                {'type': 'continuous', 'range': (-1.0, 1.0), 'bins': 16},  # Cosine of theta1
                {'type': 'continuous', 'range': (-1.0, 1.0), 'bins': 16},  # Sine of theta1
                {'type': 'continuous', 'range': (-1.0, 1.0), 'bins': 16},  # Cosine of theta2
                {'type': 'continuous', 'range': (-1.0, 1.0), 'bins': 16},  # Sine of theta2
                {'type': 'continuous', 'range': (-6.0, 6.0), 'bins': 16},  # Angular velocity of link 1
                {'type': 'continuous', 'range': (-12.0, 12.0), 'bins': 16}  # Angular velocity of link 2
            ],
            "action_space": [{'type': 'discrete', 'bins': 3}],
            "alpha": 0.1,
            "gamma": 0.99,
            "epsilon_start": 0.25,
            "epsilon_end": 0.05,
            "total_steps": int(75e6),
            "runs": 3,
        },
        {
            "group_name": "12_bins",
            "state_space": [
                {'type': 'continuous', 'range': (-1.0, 1.0), 'bins': 12},  # Cosine of theta1
                {'type': 'continuous', 'range': (-1.0, 1.0), 'bins': 12},  # Sine of theta1
                {'type': 'continuous', 'range': (-1.0, 1.0), 'bins': 12},  # Cosine of theta2
                {'type': 'continuous', 'range': (-1.0, 1.0), 'bins': 12},  # Sine of theta2
                {'type': 'continuous', 'range': (-6.0, 6.0), 'bins': 12},  # Angular velocity of link 1
                {'type': 'continuous', 'range': (-12.0, 12.0), 'bins': 12}  # Angular velocity of link 2
            ],
            "action_space": [{'type': 'discrete', 'bins': 3}],
            "alpha": 0.1,
            "gamma": 0.99,
            "epsilon_start": 0.25,
            "epsilon_end": 0.05,
            "total_steps": int(75e6),
            "runs": 3,
        },
        {
            "group_name": "8_bins",
            "state_space": [
                {'type': 'continuous', 'range': (-1.0, 1.0), 'bins': 8},  # Cosine of theta1
                {'type': 'continuous', 'range': (-1.0, 1.0), 'bins': 8},  # Sine of theta1
                {'type': 'continuous', 'range': (-1.0, 1.0), 'bins': 8},  # Cosine of theta2
                {'type': 'continuous', 'range': (-1.0, 1.0), 'bins': 8},  # Sine of theta2
                {'type': 'continuous', 'range': (-6.0, 6.0), 'bins': 8},  # Angular velocity of link 1
                {'type': 'continuous', 'range': (-12.0, 12.0), 'bins': 8}  # Angular velocity of link 2
            ],
            "action_space": [{'type': 'discrete', 'bins': 3}],
            "alpha": 0.1,
            "gamma": 0.99,
            "epsilon_start": 0.25,
            "epsilon_end": 0.05,
            "total_steps": int(75e6),
            "runs": 3,
        },
        {
            "group_name": "4_bins",
            "state_space": [
                {'type': 'continuous', 'range': (-1.0, 1.0), 'bins': 4},  # Cosine of theta1
                {'type': 'continuous', 'range': (-1.0, 1.0), 'bins': 4},  # Sine of theta1
                {'type': 'continuous', 'range': (-1.0, 1.0), 'bins': 4},  # Cosine of theta2
                {'type': 'continuous', 'range': (-1.0, 1.0), 'bins': 4},  # Sine of theta2
                {'type': 'continuous', 'range': (-6.0, 6.0), 'bins': 4},  # Angular velocity of link 1
                {'type': 'continuous', 'range': (-12.0, 12.0), 'bins': 4}  # Angular velocity of link 2
            ],
            "action_space": [{'type': 'discrete', 'bins': 3}],
            "alpha": 0.1,
            "gamma": 0.99,
            "epsilon_start": 0.25,
            "epsilon_end": 0.05,
            "total_steps": int(75e6),
            "runs": 3,
        },
    ]

    # Run all experiments
    max_workers = 12  # Number of parallel processes
    aggregated_results = run_all_experiments(experiment_groups, save_dir, max_workers)

    # Run all experiment groups
    plt.figure(figsize=(12, 8))
    linestyles = [
        '-',  # Solid line
        '--',  # Dashed line
        '-.',  # Dash-dot line
        ':',  # Dotted line
        (0, (1, 1)),  # Dotted line with tighter dots
        (0, (5, 1)),  # Long dash with short gap
        (0, (3, 1, 1, 1)),  # Dash-dot-dot pattern
        (0, (3, 5, 1, 5)),  # Dash-dot with longer gaps
        (0, (5, 10)),  # Long dash with longer gap
        (0, (1, 10)),  # Very tight dots with longer gap
        (0, (5, 5, 1, 5)),  # Dash-dot pattern with shorter gaps
        (0, (2, 2, 1, 2)),  # Short dash-dot pattern
    ]

    for i, (group_name, (avg_rewards, std_rewards, avg_test_reward)) in enumerate(aggregated_results.items()):
        total_steps = experiment_groups[i]["total_steps"]
        steps = np.linspace(1, total_steps, len(avg_rewards))

        color = plt.cm.tab10(i % 10)

        # Set smoothing factor (adjust this value to control smoothing level)
        smooth_factor = 0.001  # Smaller values result in less smoothing, larger values result in more smoothing

        # Calculate the window size for rolling operations based on the smoothing factor
        window_size = int(len(avg_rewards) * smooth_factor)
        window_size = max(1, window_size)  # Ensure the window size is at least 1

        smoothed_rewards = pd.Series(avg_rewards).rolling(window=window_size, min_periods=1).mean()
        smoothed_std = pd.Series(std_rewards).rolling(window=window_size, min_periods=1).mean()

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
