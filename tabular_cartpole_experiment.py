import numpy as np
import gymnasium as gym
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map  # For parallel progress bar
from q_table_agent import QTableAgent
import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor


# Helper function to align training rewards with truncation
def align_training_rewards(all_training_results):
    min_steps = min(len(res) for res in all_training_results)
    aligned_rewards = [np.array([r for _, r in res[:min_steps]]) for res in all_training_results]
    return np.mean(aligned_rewards, axis=0)


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
def run_experiment_for_group(args):
    total_steps, alpha, gamma, epsilon_start, epsilon_end, state_space, action_space, group_name, run_id, save_dir = args
    epsilon_decay = (epsilon_start - epsilon_end) / total_steps
    step_rewards = []
    test_rewards = []
    current_steps = 0
    epsilon = epsilon_start

    # Create QTableAgent
    agent = QTableAgent(state_space, action_space)

    # Initialize CartPole environment
    env = gym.make('CartPole-v1')

    # Training
    with tqdm(total=total_steps, desc=f"[{group_name}] Run {run_id + 1}", leave=True) as pbar:
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


# Parallel experiment runner for each group
def run_experiment_group(group, save_dir):
    args_list = [
        (
            group["total_steps"], group["alpha"], group["gamma"],
            group["epsilon_start"], group["epsilon_end"],
            group["state_space"], group["action_space"],
            group["group_name"], run_id, save_dir
        )
        for run_id in range(group["runs"])
    ]

    with ProcessPoolExecutor() as executor:
        results = process_map(run_experiment_for_group, args_list, chunksize=1)

    all_training_results, all_testing_results = zip(*results)
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
            "alpha": 0.1,
            "gamma": 0.99,
            "epsilon_start": 0.25,
            "epsilon_end": 0.001,
            "total_steps": int(5e6),
            "runs": 8
        },
        {
            "group_name": "12_positions",
            "state_space": [
                {'type': 'continuous', 'range': (-2.4, 2.4), 'bins': 12},
                {'type': 'continuous', 'range': (-2, 2), 'bins': 16},
                {'type': 'continuous', 'range': (-0.25, 0.25), 'bins': 16},
                {'type': 'continuous', 'range': (-2, 2), 'bins': 16}
            ],
            "action_space": [{'type': 'discrete', 'bins': 2}],
            "alpha": 0.1,
            "gamma": 0.99,
            "epsilon_start": 0.25,
            "epsilon_end": 0.001,
            "total_steps": int(5e6),
            "runs": 8
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
            "alpha": 0.1,
            "gamma": 0.99,
            "epsilon_start": 0.25,
            "epsilon_end": 0.001,
            "total_steps": int(5e6),
            "runs": 8
        },
        {
            "group_name": "4_positions",
            "state_space": [
                {'type': 'continuous', 'range': (-2.4, 2.4), 'bins': 4},
                {'type': 'continuous', 'range': (-2, 2), 'bins': 16},
                {'type': 'continuous', 'range': (-0.25, 0.25), 'bins': 16},
                {'type': 'continuous', 'range': (-2, 2), 'bins': 16}
            ],
            "action_space": [{'type': 'discrete', 'bins': 2}],
            "alpha": 0.1,
            "gamma": 0.99,
            "epsilon_start": 0.25,
            "epsilon_end": 0.001,
            "total_steps": int(5e6),
            "runs": 8
        },
    ]

    # Run all experiment groups
    plt.figure(figsize=(12, 8))
    linestyles = ['--', '-.', ':']
    for i, group in enumerate(experiment_groups):
        avg_training_rewards, avg_testing_rewards = run_experiment_group(group, save_dir)

        steps = np.linspace(1, group["total_steps"], len(avg_training_rewards))
        window_size = 25
        training_rewards_smoothed = pd.Series(avg_training_rewards).rolling(window=window_size, min_periods=1).mean()
        training_rewards_std = pd.Series(avg_training_rewards).rolling(window=window_size, min_periods=1).std()

        color = plt.cm.tab10(i % 10)
        plt.plot(steps, training_rewards_smoothed, color=color, linestyle='-', alpha=0.8,
                 label=f'{group["group_name"]} Smoothed Training Avg')
        plt.fill_between(steps,
                         training_rewards_smoothed - training_rewards_std,
                         training_rewards_smoothed + training_rewards_std,
                         color=color, alpha=0.25, label=f'{group["group_name"]} Training Std Dev')
        plt.axhline(avg_testing_rewards, color="black", linestyle=linestyles[i % len(linestyles)],
                    label=f'{group["group_name"]} Test Avg', alpha=0.9)
        plt.text(group["total_steps"] * 0.98, avg_testing_rewards + 2,
                 f'{avg_testing_rewards:.2f}', color="black", fontsize=10,
                 horizontalalignment='right', verticalalignment='bottom')

    plt.title("Training Results Across Experiment Groups")
    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.grid()
    plot_path = os.path.join(save_dir, "aggregated_training_results.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Aggregated training results saved to {plot_path}")
