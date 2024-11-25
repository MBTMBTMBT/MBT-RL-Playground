if __name__ == '__main__':
    import numpy as np
    import gymnasium as gym
    from matplotlib import pyplot as plt
    from matplotlib.animation import FuncAnimation
    from tqdm import tqdm
    from q_table_agent import QTableAgent
    import os

    # Create save directory
    save_dir = "./experiments/cartpole/"
    os.makedirs(save_dir, exist_ok=True)

    # Define CartPole state and action spaces
    state_space = [
        {'type': 'continuous', 'range': (-2.4, 2.4), 'bins': 16},  # Cart position
        {'type': 'continuous', 'range': (-2, 2), 'bins': 16},  # Cart velocity
        {'type': 'continuous', 'range': (-0.25, 0.25), 'bins': 16},  # Pole angle
        {'type': 'continuous', 'range': (-2, 2), 'bins': 16}  # Pole angular velocity
    ]

    action_space = [
        {'type': 'discrete', 'bins': 2}  # Two discrete actions: left (0), right (1)
    ]

    # Create QTableAgent instance
    agent = QTableAgent(state_space, action_space)

    # Initialize CartPole environment
    env = gym.make('CartPole-v1')

    # Training parameters
    total_steps = int(50e6)       # Total steps
    alpha = 0.025                # Learning rate
    gamma = 0.99                # Discount factor
    epsilon_start = 0.25        # Starting exploration rate
    epsilon_end = 0.001         # Minimum exploration rate
    epsilon_decay = (epsilon_start - epsilon_end) / total_steps  # Linear decay rate
    epsilon = epsilon_start     # Initial exploration rate

    # Metrics
    train_rewards = []           # Store rewards for each episode
    step_rewards = []            # Store rewards with step as x-axis
    current_steps = 0            # Track total steps so far
    episode_steps = 0            # Steps in the current episode

    # Define the custom state range
    custom_state_range = {
        "position": (-3.0, 3.0),  # Cart position
        "velocity": (-2.5, 2.5),  # Cart velocity
        "angle": (-0.3, 0.3),  # Pole angle
        "angular_velocity": (-2.5, 2.5)  # Pole angular velocity
    }

    # Training loop with progress bar
    with tqdm(total=total_steps, desc="Training Progress") as pbar:
        while current_steps < total_steps:
            state, _ = env.reset()

            # Randomly sample the initial state from the specified range
            low = np.array([v[0] for v in custom_state_range.values()])
            high = np.array([v[1] for v in custom_state_range.values()])
            random_state = np.random.uniform(low, high)

            # Manually set the environment's internal state
            env.state = random_state

            total_reward = 0
            done = False

            while not done:
                # Epsilon-greedy policy
                if np.random.random() < epsilon:
                    action = [np.random.choice([0, 1])]  # Random action
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

                if done or truncated:
                    # Track episode reward
                    train_rewards.append(total_reward)
                    step_rewards.append((current_steps, total_reward))

                    # Update progress bar description
                    recent_rewards = [r for _, r in step_rewards[-10:]]
                    pbar.set_description(
                        f"Steps: {current_steps}/{total_steps}, "
                        f"Recent Avg: {np.mean(recent_rewards):.2f}, Max: {max(recent_rewards):.2f}, "
                        f"Epsilon: {epsilon:.4f}"
                    )
                    break

    # Save agent
    agent.save_q_table(os.path.join(save_dir, "q_table_agent.csv"))

    # Test loading the agent
    agent = QTableAgent.load_q_table(os.path.join(save_dir, "q_table_agent.csv"))

    # Initialize CartPole environment for testing
    env = gym.make('CartPole-v1', render_mode="rgb_array")

    # Save test episode as a video
    state, _ = env.reset()
    frames = []
    total_reward = 0
    done = False

    while not done:
        probabilities = agent.get_action_probabilities(state, strategy="greedy")
        action = [np.argmax(probabilities)]
        state, reward, done, truncated, _ = env.step(action[0])
        total_reward += reward
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

    ani = FuncAnimation(fig, update, frames=frames, interval=50, blit=True)
    gif_path = os.path.join(save_dir, "cartpole_test.gif")
    ani.save(gif_path, dpi=300, writer="pillow")
    print(f"Animation saved to {gif_path}")

    # Test the trained agent
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
    steps, rewards = zip(*step_rewards)
    plt.figure(figsize=(10, 6))
    plt.plot(steps, rewards, label='Training Rewards')
    plt.axhline(np.mean(test_rewards), color='r', linestyle='--', label='Mean Test Reward')
    plt.title("CartPole Training and Testing Results")
    plt.xlabel("Steps")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid()
    save_path = os.path.join(save_dir, "cartpole_training_results.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.show()
    print(f"Training and testing results saved to {save_path}")
