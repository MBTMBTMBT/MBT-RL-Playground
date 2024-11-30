import numpy as np
import gymnasium as gym
import torch
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import os
import pandas as pd
from q_table_agent import QTableAgent
from wrappers import VAEWrapper


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
    total_steps = group["total_steps"]
    alpha = group["alpha"]
    gamma = group["gamma"]
    epsilon_start = group["epsilon_start"]
    epsilon_end = group["epsilon_end"]
    state_space = group["state_space"]
    action_space = group["action_space"]
    group_name = group["group_name"]
    use_normal_partition_state = group["normal_partition_state"]
    epsilon_decay = (epsilon_start - epsilon_end) / total_steps
    step_rewards = []
    test_rewards = []
    current_steps = 0
    epsilon = epsilon_start

    # Create QTableAgent
    agent = QTableAgent(state_space, action_space, normal_partition_state=use_normal_partition_state)

    # Initialize CartPole environment
    env = gym.make('CartPole-v1')

    if "wrapper_args" in group.keys():
        wrapper_args = group["wrapper_args"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # device = torch.device("cpu")
        print(f"Using {device} device")
        env = VAEWrapper(
            env,
            num_hidden_values=wrapper_args["num_hidden_values"],
            net_arch=wrapper_args["net_arch"],
            do_training=True,
            buffer_size=wrapper_args["buffer_size"],
            iterations=wrapper_args["iterations"],
            batch_size=wrapper_args["batch_size"],
            beta=wrapper_args["beta"],
            lr=wrapper_args["lr"],
            device=device,
        )

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
                    probabilities = agent.get_action_probabilities(state, strategy="softmax")
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
    env = gym.make('CartPole-v1', render_mode="rgb_array")
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
                group, run_id, save_dir,
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
    experiment_name = "CartPole_VAE_Experiments"
    save_dir = f"./experiments/{experiment_name}/"
    os.makedirs(save_dir, exist_ok=True)

    # Define experiment groups
    experiment_groups = [
        {
            "group_name": "direct_input_8",
            "state_space": [
                {'type': 'continuous', 'range': (-2.4, 2.4), 'bins': 8},
                {'type': 'continuous', 'range': (-2, 2), 'bins': 8},
                {'type': 'continuous', 'range': (-0.25, 0.25), 'bins': 8},
                {'type': 'continuous', 'range': (-2, 2), 'bins': 8},
            ],
            "action_space": [{'type': 'discrete', 'bins': 2}],
            "normal_partition_state": False,
            "alpha": 0.25,
            "gamma": 0.99,
            "epsilon_start": 0.25,
            "epsilon_end": 0.05,
            "total_steps": int(15e6),
            "runs": 4,
        },
        {
            "group_name": "normal_distribution_8",
            "state_space": [
                {'type': 'continuous', 'bins': 8},
                {'type': 'continuous', 'bins': 8},
                {'type': 'continuous', 'bins': 8},
                {'type': 'continuous', 'bins': 8},
            ],
            "action_space": [{'type': 'discrete', 'bins': 2}],
            "normal_partition_state": True,
            "alpha": 0.25,
            "gamma": 0.99,
            "epsilon_start": 0.25,
            "epsilon_end": 0.05,
            "total_steps": int(15e6),
            "runs": 4,
            "wrapper_args": {
                "num_hidden_values": 4,
                "net_arch": [32, 32,],
                "buffer_size": int(1e5),
                "iterations": 10,
                "batch_size": 32,
                "beta": 1.0,
                "lr": 1e-3,
            }
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
