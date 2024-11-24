if __name__ == '__main__':
    import numpy as np
    import gymnasium as gym
    from matplotlib import pyplot as plt
    from matplotlib.animation import FuncAnimation
    from tqdm import tqdm
    from q_table_agent import QTableAgent
    import os

    # Create save directory
    save_dir = "./experiments/acrobot/"
    os.makedirs(save_dir, exist_ok=True)

    # Define Acrobot state and action spaces
    state_space = [
        {'type': 'continuous', 'range': (-1.0, 1.0), 'bins': 12},  # Cosine of theta1
        {'type': 'continuous', 'range': (-1.0, 1.0), 'bins': 12},  # Sine of theta1
        {'type': 'continuous', 'range': (-1.0, 1.0), 'bins': 12},  # Cosine of theta2
        {'type': 'continuous', 'range': (-1.0, 1.0), 'bins': 12},  # Sine of theta2
        {'type': 'continuous', 'range': (-6.0, 6.0), 'bins': 14},  # Angular velocity of link 1
        {'type': 'continuous', 'range': (-12.0, 12.0), 'bins': 14}  # Angular velocity of link 2
    ]

    action_space = [
        {'type': 'discrete', 'bins': 3}  # Three discrete actions: -1, 0, 1 (torque on the joint)
    ]

    # Create QTableAgent instance
    agent = QTableAgent(state_space, action_space)

    # Initialize Acrobot environment
    env = gym.make('Acrobot-v1')

    # Training parameters
    total_steps = int(10e6)       # Total steps
    alpha = 0.05                # Learning rate
    gamma = 0.99                # Discount factor
    epsilon_start = 0.25        # Starting exploration rate
    epsilon_end = 0.001         # Minimum exploration rate
    epsilon_decay = (epsilon_start - epsilon_end) / total_steps  # Linear decay rate
    epsilon = epsilon_start     # Initial exploration rate

    # Metrics
    train_rewards = []           # Store rewards for each episode
    episode_rewards = []         # Store recent episode rewards for progress bar updates
    current_steps = 0            # Track total steps so far
    episode_steps = 0            # Steps in the current episode

    # Define custom initialization ranges for each state variable
    custom_state_range = {
        "Cos_theta1": (-1.0, 1.0),  # Cosine of theta1
        "Sin_theta1": (-1.0, 1.0),  # Sine of theta1
        "Cos_theta2": (-1.0, 1.0),  # Cosine of theta2
        "Sin_theta2": (-1.0, 1.0),  # Sine of theta2
        "Angular_velocity_1": (-4.0, 4.0),  # Angular velocity of link 1
        "Angular_velocity_2": (-9.0, 9.0)  # Angular velocity of link 2
    }

    with tqdm(total=total_steps, desc="Training Progress") as pbar:
        while current_steps < total_steps:
            # Reset environment and sample a random initial state
            state, _ = env.reset()
            low = np.array([v[0] for v in custom_state_range.values()])
            high = np.array([v[1] for v in custom_state_range.values()])
            random_state = np.random.uniform(low, high)  # Randomly sample initial state

            # Manually set the environment's state to the sampled random state
            env.env.state = random_state

            total_reward = 0
            done = False

            while not done:
                # Epsilon-greedy policy
                if np.random.random() < epsilon:
                    action = [np.random.choice([0, 1, 2])]  # Random action
                else:
                    probabilities = agent.get_action_probabilities(state, strategy="greedy")
                    action = [np.argmax(probabilities)]  # Exploit the best action

                # Perform action in the environment
                next_state, reward, done, truncated, _ = env.step(action[0])

                # Update the Q-Table
                agent.update(state, action, reward, next_state, alpha, gamma)

                # Update state and reward
                state = next_state
                total_reward += reward
                episode_steps += 1
                current_steps += 1

                # Update exploration rate
                epsilon = max(epsilon_end, epsilon - epsilon_decay)

                # Update progress bar
                pbar.update(1)

                # Exit if total steps exceed limit
                if current_steps >= total_steps:
                    break

                if done or truncated:
                    # Track episode reward and reset episode steps
                    train_rewards.append(total_reward)
                    episode_rewards.append(total_reward)
                    episode_steps = 0

                    # Update progress bar description
                    recent_avg = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)
                    recent_max = max(episode_rewards[-10:]) if len(episode_rewards) >= 10 else max(episode_rewards)
                    pbar.set_description(
                        f"Steps: {current_steps}/{total_steps}, "
                        f"Recent Avg: {recent_avg:.2f}, Max: {recent_max:.2f}, Epsilon: {epsilon:.4f}"
                    )
                    break

    # Save agent
    agent.save_q_table(os.path.join(save_dir, "q_table_agent.csv"))

    # Test loading the agent
    agent = QTableAgent.load_q_table(os.path.join(save_dir, "q_table_agent.csv"))

    # Initialize Acrobot environment for testing and rendering
    env = gym.make('Acrobot-v1', render_mode="rgb_array")

    # Select one episode to save as a video
    state, _ = env.reset()
    frames = []  # List to store frames for the animation
    total_reward = 0
    done = False

    while not done:
        probabilities = agent.get_action_probabilities(state, strategy="greedy")
        action = [np.argmax(probabilities)]
        state, reward, done, truncated, _ = env.step(action[0])
        total_reward += reward

        # Append the rendered frame
        frames.append(env.render())

        if done or truncated:
            break

    # Save the frames as a video using matplotlib.animation
    fig, ax = plt.subplots()
    ax.axis('off')  # Turn off axes for a cleaner output
    img = ax.imshow(frames[0])  # Display the first frame

    def update(frame):
        img.set_data(frame)
        return [img]

    # Create an animation object
    ani = FuncAnimation(fig, update, frames=frames, interval=50, blit=True)

    # Save the animation as a GIF
    gif_path = os.path.join(save_dir, "acrobot_test.gif")
    ani.save(gif_path, dpi=300, writer="pillow")
    print(f"Animation saved to {gif_path}")

    # Test the trained agent for evaluation
    test_rewards = []
    for _ in range(20):  # Perform 20 test episodes
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

    # Plot training and testing results
    plt.figure(figsize=(10, 6))
    plt.plot(train_rewards, label='Training Rewards')
    plt.axhline(np.mean(test_rewards), color='r', linestyle='--', label='Mean Test Reward')
    plt.title("Acrobot Training and Testing Results")
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid()
    save_path = os.path.join(save_dir, "acrobot_training_results.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')  # Save the figure to the specified path
    print(f"Plot saved to {save_path}")
