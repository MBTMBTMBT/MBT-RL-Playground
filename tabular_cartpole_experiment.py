import numpy as np
import gymnasium as gym
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
from q_table_agent import QTableAgent
import os
import pandas as pd


# Helper function to align training rewards with truncation
def align_training_rewards(all_training_results):
    # Determine the shortest step length across all experiments
    min_steps = min(len(res) for res in all_training_results)
    aligned_rewards = []
    for step_rewards in all_training_results:
        steps, rewards = zip(*step_rewards)
        aligned_rewards.append(np.array(rewards[:min_steps]))  # Truncate rewards to min_steps
    return np.mean(aligned_rewards, axis=0)


# Main experiment function
def run_experiment(total_steps, alpha, gamma, epsilon_start, epsilon_end, state_space, action_space, group_name, runs, save_dir):
    epsilon_decay = (epsilon_start - epsilon_end) / total_steps
    all_training_results = []
    all_testing_results = []

    for run_id in range(runs):
        # Create QTableAgent
        agent = QTableAgent(state_space, action_space)

        # Initialize CartPole environment
        env = gym.make('CartPole-v1')

        # Metrics
        step_rewards = []
        current_steps = 0
        epsilon = epsilon_start

        # Define custom state range
        custom_state_range = {
            "position": (-3.0, 3.0),  # Cart position
            "velocity": (-2.5, 2.5),  # Cart velocity
            "angle": (-0.3, 0.3),  # Pole angle
            "angular_velocity": (-2.5, 2.5)  # Pole angular velocity
        }

        # Training loop
        with tqdm(total=total_steps, desc=f"Training {group_name} Run {run_id + 1}/{runs}") as pbar:
            while current_steps < total_steps:
                state, _ = env.reset()
                low = np.array([v[0] for v in custom_state_range.values()])
                high = np.array([v[1] for v in custom_state_range.values()])
                random_state = np.random.uniform(low, high)
                env.state = random_state

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

                        # Update progress bar description
                        recent_rewards = [r for _, r in step_rewards[-10:]]
                        pbar.set_description(
                            f"Steps: {current_steps}/{total_steps}, "
                            f"Recent Avg: {np.mean(recent_rewards):.2f}, Max: {max(recent_rewards):.2f}, "
                            f"Epsilon: {epsilon:.4f}"
                        )
                        break

        # Save Q-Table
        q_table_path = os.path.join(save_dir, f"{group_name}_run_{run_id + 1}_q_table.csv")
        agent.save_q_table(q_table_path)

        # Save training data
        training_data_path = os.path.join(save_dir, f"{group_name}_run_{run_id + 1}_training_data.csv")
        pd.DataFrame(step_rewards, columns=["Step", "Reward"]).to_csv(training_data_path, index=False)

        all_training_results.append(step_rewards)

        # Testing
        env = gym.make('CartPole-v1', render_mode="rgb_array")
        test_rewards = []
        for _ in range(20):
            state, _ = env.reset()
            total_reward = 0
            done = False

            while not done:
                probabilities = agent.get_action_probabilities(state, strategy="greedy")
                action = [np.argmax(probabilities)]
                state, reward, done, truncated, _ = env.step(action[0])
                total_reward += reward

                if done or truncated:
                    break

            test_rewards.append(total_reward)

        all_testing_results.append(np.mean(test_rewards))

    # Align and average training rewards
    avg_training_rewards = align_training_rewards(all_training_results)
    avg_testing_rewards = np.mean(all_testing_results)

    return avg_training_rewards, avg_testing_rewards


if __name__ == '__main__':
    # General experiment parameters
    experiment_name = "CartPole_Experiments"
    save_dir = f"./experiments/{experiment_name}/"
    os.makedirs(save_dir, exist_ok=True)

    # Define experiment groups
    experiment_groups = [
        {
            "group_name": "16_positions",
            "state_space": [
                {'type': 'continuous', 'range': (-2.4, 2.4), 'bins': 16},
                {'type': 'continuous', 'range': (-2, 2), 'bins': 16},
                {'type': 'continuous', 'range': (-0.25, 0.25), 'bins': 16},
                {'type': 'continuous', 'range': (-2, 2), 'bins': 16}
            ],
            "action_space": [{'type': 'discrete', 'bins': 2}],
            "alpha": 0.025,
            "gamma": 0.99,
            "epsilon_start": 0.25,
            "epsilon_end": 0.001,
            "total_steps": int(2.5e6),
            "runs": 3
        },
        {
            "group_name": "8_positions",
            "state_space": [
                {'type': 'continuous', 'range': (-2.4, 2.4), 'bins': 8},
                {'type': 'continuous', 'range': (-2, 2), 'bins': 16},
                {'type': 'continuous', 'range': (-0.25, 0.25), 'bins': 16},
                {'type': 'continuous', 'range': (-2, 2), 'bins': 16}
            ],
            "action_space": [{'type': 'discrete', 'bins': 2}],
            "alpha": 0.05,
            "gamma": 0.99,
            "epsilon_start": 0.25,
            "epsilon_end": 0.001,
            "total_steps": int(2.5e6),
            "runs": 3
        }
    ]

    # Run all experiment groups
    plt.figure(figsize=(12, 8))
    linestyles = ['--', '-.', ':']  # Different line styles for horizontal lines
    for i, group in enumerate(experiment_groups):
        avg_training_rewards, avg_testing_rewards = run_experiment(
            total_steps=group["total_steps"],
            alpha=group["alpha"],
            gamma=group["gamma"],
            epsilon_start=group["epsilon_start"],
            epsilon_end=group["epsilon_end"],
            state_space=group["state_space"],
            action_space=group["action_space"],
            group_name=group["group_name"],
            runs=group["runs"],
            save_dir=save_dir
        )

        # Plot training results
        steps = np.linspace(1, group["total_steps"], len(avg_training_rewards))  # Align steps to total_steps
        plt.plot(steps, avg_training_rewards, label=f'{group["group_name"]} (Test Avg: {avg_testing_rewards:.2f})')

        # Add horizontal line for average test reward
        linestyle = linestyles[i % len(linestyles)]  # Cycle through line styles
        plt.axhline(avg_testing_rewards, color="black", linestyle=linestyle, label=f'{group["group_name"]} Test Avg',
                    alpha=0.7)

    # Finalize and save plot
    plt.title("Training Results Across Experiment Groups")
    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.grid()
    plot_path = os.path.join(save_dir, "aggregated_training_results.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Aggregated training results saved to {plot_path}")
