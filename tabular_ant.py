if __name__ == '__main__':
    import numpy as np
    import gymnasium as gym
    from matplotlib import pyplot as plt
    from matplotlib.animation import FuncAnimation
    from tqdm import tqdm
    from q_table_agent import QTableAgent
    import os

    # Create save directory
    save_dir = "./experiments/ant/"
    os.makedirs(save_dir, exist_ok=True)

    # Define Ant state and action spaces
    state_space = [
        {'type': 'continuous', 'range': (-5.0, 5.0), 'bins': 3} for _ in range(27)  # 27 state dimensions
    ]
    action_space = [
        {'type': 'continuous', 'range': (-1.0, 1.0), 'bins': 3} for _ in range(8)  # 8 action dimensions
    ]

    # Create QTableAgent instance
    agent = QTableAgent(state_space, action_space, action_combination=True)

    # Initialize Ant environment
    env = gym.make('Ant-v4')

    # Training parameters
    total_steps = int(10e6)       # Total steps
    alpha = 0.1                 # Learning rate
    gamma = 0.99                # Discount factor
    epsilon_start = 0.3         # Starting exploration rate
    epsilon_end = 0.01          # Minimum exploration rate
    epsilon_decay = (epsilon_start - epsilon_end) / total_steps  # Linear decay rate
    epsilon = epsilon_start     # Initial exploration rate

    # Metrics
    train_rewards = []           # Store rewards for each episode
    episode_rewards = []         # Store recent episode rewards for progress bar updates
    current_steps = 0            # Track total steps so far
    episode_steps = 0            # Steps in the current episode

    # Define custom state and action range for initialization
    custom_state_range = {
        f"state_dim_{i}": (-5.0, 5.0) for i in range(27)  # State dimensions
    }
    custom_action_range = {
        f"action_dim_{i}": (-1.0, 1.0) for i in range(8)  # Action dimensions
    }

    # Training loop with progress bar
    with tqdm(total=total_steps, desc="Training Progress") as pbar:
        while current_steps < total_steps:
            state, _ = env.reset()

            # Randomly sample the initial state
            state_low = np.array([v[0] for v in custom_state_range.values()])
            state_high = np.array([v[1] for v in custom_state_range.values()])
            random_state = np.random.uniform(state_low, state_high)
            env.env.state = random_state

            total_reward = 0
            done = False

            while not done:
                # Epsilon-greedy policy
                if np.random.random() < epsilon:
                    action = np.random.uniform(
                        low=[v[0] for v in custom_action_range.values()],
                        high=[v[1] for v in custom_action_range.values()]
                    )  # Random continuous action
                else:
                    probabilities = agent.get_action_probabilities(state, strategy="greedy")
                    action = [np.argmax(probabilities)]  # Exploit the best action

                # Perform action in the environment
                next_state, reward, done, truncated, _ = env.step(action)

                # Update the Q-Table
                agent.update(state, action, reward, next_state, alpha, gamma)

                # Update state and reward
                state = next_state
                total_reward += reward
                current_steps += 1

                # Update exploration rate
                epsilon = max(epsilon_end, epsilon - epsilon_decay)

                # Update progress bar
                pbar.update(1)

                # Exit if total steps exceed limit
                if current_steps >= total_steps:
                    break

            # Record episode reward
            train_rewards.append(total_reward)
            if len(train_rewards) >= 10:
                avg_reward = np.mean(train_rewards[-10:])
                max_reward = np.max(train_rewards[-10:])
            else:
                avg_reward = np.mean(train_rewards)
                max_reward = np.max(train_rewards)
            pbar.set_description(
                f"Steps: {current_steps}/{total_steps}, Avg Reward: {avg_reward:.2f}, Max Reward: {max_reward:.2f}, Epsilon: {epsilon:.4f}"
            )

    # Save agent
    agent.save_q_table(os.path.join(save_dir, "q_table_agent.csv"))

    # Test loading the agent
    agent = QTableAgent.load_q_table(os.path.join(save_dir, "q_table_agent.csv"))

    # Initialize Ant environment for testing and rendering
    env = gym.make('Ant-v4', render_mode="rgb_array")

    # Select one episode to save as a video
    state, _ = env.reset()
    frames = []  # List to store frames for the animation
    total_reward = 0
    done = False

    while not done:
        probabilities = agent.get_action_probabilities(state, strategy="greedy")
        action = [np.argmax(probabilities)]
        state, reward, done, truncated, _ = env.step(action)
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
    gif_path = os.path.join(save_dir, "ant_test.gif")
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
            state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            if done or truncated:
                break
        test_rewards.append(total_reward)

    # Plot training and testing results
    plt.figure(figsize=(10, 6))
    plt.plot(train_rewards, label='Training Rewards')
    plt.axhline(np.mean(test_rewards), color='r', linestyle='--', label='Mean Test Reward')
    plt.title("Ant Training and Testing Results")
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid()
    save_path = os.path.join(save_dir, "ant_training_results.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')  # Save the figure to the specified path
    print(f"Plot saved to {save_path}")
