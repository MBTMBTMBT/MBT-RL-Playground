if __name__ == '__main__':
    import numpy as np
    import gymnasium as gym
    from matplotlib import pyplot as plt
    from tqdm import tqdm
    from q_table_agent import QTableAgent

    # Define CartPole state and action spaces
    state_space = [
        {'type': 'continuous', 'range': (-2.4, 2.4), 'bins': 8},  # Cart position
        {'type': 'continuous', 'range': (-2, 2), 'bins': 16},  # Cart velocity
        {'type': 'continuous', 'range': (-0.25, 0.25), 'bins': 8},  # Pole angle
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
    total_steps = int(10*1e6)       # Total steps
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

    # Define the custom state range
    custom_state_range = {
        "position": (-2.4, 2.4),  # Cart position
        "velocity": (-2.0, 2.0),  # Cart velocity
        "angle": (-0.2, 0.2),  # Pole angle
        "angular_velocity": (-2.0, 2.0)  # Pole angular velocity
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
    agent.save_q_table("./experiments/cartpole/q_table_agent.csv")

    # Test loading the agent
    agent = QTableAgent.load_q_table("./experiments/cartpole/q_table_agent.csv")

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
    plt.figure(figsize=(10, 6))
    plt.plot(train_rewards, label='Training Rewards')
    plt.axhline(np.mean(test_rewards), color='r', linestyle='--', label='Mean Test Reward')
    plt.title("CartPole Training and Testing Results")
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid()
    plt.show()
