from typing import List, Dict, Union, Tuple, Any
import numpy as np
import gymnasium as gym
from matplotlib import pyplot as plt
from tqdm import tqdm


class QTableAgent:
    def __init__(self,
                 state_space: List[Dict[str, Union[str, Tuple[float, float], int]]],
                 action_space: List[Dict[str, Union[str, Tuple[float, float], int]]],
                 action_combination: bool = False):
        """
        Initialize the Q-Table Agent.

        # Example usage scenarios for state_space and action_space definitions.

        # Example 1: For CartPole (Default Setup)
        # The environment is simple with 4 continuous state dimensions and 1 discrete action dimension.
        state_space = [
            {'type': 'continuous', 'range': (-4.8, 4.8), 'bins': 10},  # Cart position
            {'type': 'continuous', 'range': (-10, 10), 'bins': 10},    # Cart velocity
            {'type': 'continuous', 'range': (-0.418, 0.418), 'bins': 10},  # Pole angle
            {'type': 'continuous', 'range': (-10, 10), 'bins': 10}     # Pole angular velocity
        ]
        action_space = [
            {'type': 'discrete', 'bins': 2}  # Two discrete actions: move left (0), move right (1)
        ]

        # Example 2: For a custom robot arm with joint control
        # Each joint angle is continuous, and there are multiple continuous actions to control torque on each joint.
        state_space = [
            {'type': 'continuous', 'range': (-180, 180), 'bins': 20},  # Joint 1 angle
            {'type': 'continuous', 'range': (-180, 180), 'bins': 20},  # Joint 2 angle
            {'type': 'continuous', 'range': (-180, 180), 'bins': 20}   # Joint 3 angle
        ]
        action_space = [
            {'type': 'continuous', 'range': (-10, 10), 'bins': 5},  # Torque for Joint 1
            {'type': 'continuous', 'range': (-10, 10), 'bins': 5},  # Torque for Joint 2
            {'type': 'continuous', 'range': (-10, 10), 'bins': 5}   # Torque for Joint 3
        ]

        # Example 3: For GridWorld (Tabular Setup)
        # States are discrete grid cells, and actions are discrete directions.
        state_space = [
            {'type': 'discrete', 'bins': 10},  # Grid rows (10 rows)
            {'type': 'discrete', 'bins': 10}   # Grid columns (10 columns)
        ]
        action_space = [
            {'type': 'discrete', 'bins': 4}  # Four actions: up (0), down (1), left (2), right (3)
        ]

        # Example 4: For a simulated car with velocity and angle control
        # The car's state includes continuous velocity and heading angle, with continuous actions for acceleration and steering.
        state_space = [
            {'type': 'continuous', 'range': (0, 100), 'bins': 10},  # Speed (0 to 100 km/h)
            {'type': 'continuous', 'range': (-180, 180), 'bins': 18}  # Heading angle (-180 to 180 degrees)
        ]
        action_space = [
            {'type': 'continuous', 'range': (-5, 5), 'bins': 5},  # Acceleration (-5 to 5 m/s^2)
            {'type': 'continuous', 'range': (-30, 30), 'bins': 5}  # Steering angle (-30 to 30 degrees)
        ]

        # Example 5: For a complex multi-agent environment (Action Combinations Enabled)
        # Each agent has a discrete state, and actions are combinations of multiple discrete commands.
        state_space = [
            {'type': 'discrete', 'bins': 5},  # State of Agent 1
            {'type': 'discrete', 'bins': 5}   # State of Agent 2
        ]
        action_space = [
            {'type': 'discrete', 'bins': 3},  # Action set for Agent 1
            {'type': 'discrete', 'bins': 3}   # Action set for Agent 2
        ]
        # Enable action combinations (both agents act simultaneously)
        agent = QTableAgent(state_space, action_space, action_combination=True)

        :param state_space: A list of dictionaries defining the state space.
                            Each dictionary specifies:
                            - 'type': 'discrete' or 'continuous'
                            - 'bins': Number of bins for discrete space
                            - 'range': (low, high) for continuous space
        :param action_space: A list of dictionaries defining the action space.
                             Format is similar to `state_space`.
        :param action_combination: Whether actions can be combined into groups (True or False).
        """
        self.state_space: List[Dict[str, Any]] = state_space
        self.action_space: List[Dict[str, Any]] = action_space
        self.action_combination: bool = action_combination

        # Discretize the state space
        self.state_bins: List[np.ndarray] = [self._discretize_space(dim) for dim in state_space]

        # Discretize or combine the action space
        if action_combination:
            self.action_bins: np.ndarray = self._generate_action_combinations(action_space)
        else:
            self.action_bins: List[np.ndarray] = [self._discretize_space(dim) for dim in action_space]

        # Initialize Q-Table
        self.q_table: np.ndarray = self._initialize_q_table()

    def _discretize_space(self, space: Dict[str, Any]) -> np.ndarray:
        """
        Discretize a single state or action dimension.

        :param space: A dictionary defining the dimension type and properties.
        :return: A numpy array of discretized bins.
        """
        if space['type'] == 'discrete':
            return np.arange(space['bins'])  # Direct discrete indices
        elif space['type'] == 'continuous':
            low, high = space['range']
            return np.linspace(low, high, space['bins'])  # Linearly spaced bins
        else:
            raise ValueError("Invalid space type. Use 'discrete' or 'continuous'.")

    def _generate_action_combinations(self, action_space: List[Dict[str, Any]]) -> np.ndarray:
        """
        Generate all possible combinations of actions when action combination is enabled.

        :param action_space: A list of dictionaries defining the action space.
        :return: A numpy array of all possible action combinations.
        """
        action_bins: List[np.ndarray] = [self._discretize_space(dim) for dim in action_space]
        return np.array(np.meshgrid(*action_bins)).T.reshape(-1, len(action_space))

    def _initialize_q_table(self) -> np.ndarray:
        """
        Initialize the Q-Table with zeros based on state and action space sizes.

        :return: A multi-dimensional numpy array representing the Q-Table.
        """
        state_sizes: List[int] = [len(bins) for bins in self.state_bins]
        action_size: int = len(self.action_bins) if self.action_combination else np.prod(
            [len(bins) for bins in self.action_bins])
        return np.zeros(state_sizes + [action_size])

    def get_state_index(self, state: List[float]) -> Tuple[int, ...]:
        """
        Get the index of a given state in the Q-Table.

        :param state: A list of state values.
        :return: A tuple of indices corresponding to the discretized state.
        """
        state_index: List[int] = []
        for i, value in enumerate(state):
            bins = self.state_bins[i]
            state_index.append(np.digitize(value, bins) - 1)
        return tuple(state_index)

    def get_action_index(self, action: List[float]) -> int:
        """
        Get the index of a given action in the Q-Table.

        :param action: A list of action values.
        :return: An integer index corresponding to the discretized action.
        """
        if self.action_combination:
            action = np.array(action).reshape(1, -1)
            distances = np.linalg.norm(self.action_bins - action, axis=1)
            return np.argmin(distances)
        else:
            action_index: List[int] = []
            for i, value in enumerate(action):
                bins = self.action_bins[i]
                action_index.append(np.digitize(value, bins) - 1)
            return np.ravel_multi_index(action_index, [len(bins) for bins in self.action_bins])

    def get_q_value(self, state: List[float], action: List[float]) -> float:
        """
        Get the Q-value for a given state and action.

        :param state: A list of state values.
        :param action: A list of action values.
        :return: The Q-value corresponding to the state and action.
        """
        state_idx: Tuple[int, ...] = self.get_state_index(state)
        action_idx: int = self.get_action_index(action)
        return self.q_table[state_idx + (action_idx,)]

    def get_action_probabilities(self, state: List[float], strategy: str = "greedy",
                                 temperature: float = 1.0) -> np.ndarray:
        """
        Get the action probabilities for a given state based on the strategy.

        :param state: A list of state values.
        :param strategy: Strategy type, either 'greedy' or 'softmax'.
        :param temperature: Temperature parameter for the softmax strategy.
        :return: A numpy array of action probabilities.
        """
        state_idx: Tuple[int, ...] = self.get_state_index(state)
        q_values: np.ndarray = self.q_table[state_idx]

        if strategy == "greedy":
            probabilities: np.ndarray = np.zeros_like(q_values)
            probabilities[np.argmax(q_values)] = 1.0
        elif strategy == "softmax":
            exp_values: np.ndarray = np.exp(q_values / temperature)
            probabilities = exp_values / np.sum(exp_values)
        else:
            raise ValueError("Invalid strategy. Use 'greedy' or 'softmax'.")

        return probabilities

    def update_q_value(self, state: List[float], action: List[float], value: float) -> None:
        """
        Update the Q-value for a given state and action.

        :param state: A list of state values.
        :param action: A list of action values.
        :param value: The new Q-value to be updated.
        """
        state_idx: Tuple[int, ...] = self.get_state_index(state)
        action_idx: int = self.get_action_index(action)
        self.q_table[state_idx + (action_idx,)] = value

    def update(self, state: List[float], action: List[float], reward: float, next_state: List[float],
               alpha: float = 0.1, gamma: float = 0.99) -> None:
        """
        Update the Q-Table using the standard Q-Learning update rule.

        :param state: Current state (list of floats for continuous dimensions).
        :param action: Current action (list of floats for continuous dimensions).
        :param reward: Reward received after taking the action.
        :param next_state: Next state after taking the action.
        :param alpha: Learning rate.
        :param gamma: Discount factor.
        """
        # Get indices for the current state and action
        state_idx = self.get_state_index(state)
        action_idx = self.get_action_index(action)

        # Compute the target value
        next_q = np.max(self.q_table[self.get_state_index(next_state)])
        target = reward + gamma * next_q

        # Update current state-action pair
        self.q_table[state_idx + (action_idx,)] += alpha * (target - self.q_table[state_idx + (action_idx,)])


if __name__ == '__main__':
    # Define CartPole state and action spaces
    state_space = [
        {'type': 'continuous', 'range': (-4.8, 4.8), 'bins': 16},  # Cart position
        {'type': 'continuous', 'range': (-10, 10), 'bins': 64},    # Cart velocity
        {'type': 'continuous', 'range': (-0.418, 0.418), 'bins': 64},  # Pole angle
        {'type': 'continuous', 'range': (-10, 10), 'bins': 64}     # Pole angular velocity
    ]

    action_space = [
        {'type': 'discrete', 'bins': 2}  # Two discrete actions: left (0), right (1)
    ]

    # Create QTableAgent instance
    agent = QTableAgent(state_space, action_space)

    # Initialize CartPole environment
    env = gym.make('CartPole-v1')

    # Training parameters
    num_episodes = int(50e3)
    alpha = 0.1  # Learning rate
    gamma = 0.99  # Discount factor
    epsilon = 0.25  # Exploration rate

    # Store rewards for visualization
    train_rewards = []

    # Training loop with progress bar
    with tqdm(total=num_episodes) as pbar:
        for episode in range(num_episodes):
            state, _ = env.reset()
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

                if done or truncated:
                    break

            train_rewards.append(total_reward)

            # Adjust exploration rate (epsilon decay)
            epsilon = max(0.01, epsilon * 0.995)

            # Update progress bar and statistics
            avg_reward = np.mean(train_rewards[-100:]) if len(train_rewards) >= 100 else np.mean(train_rewards)
            max_reward = np.max(train_rewards)
            pbar.set_description(
                f"Episode {episode + 1}/{num_episodes}, Reward: {total_reward}, Avg: {avg_reward:.2f}, Max: {max_reward}"
            )
            pbar.update(1)

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
