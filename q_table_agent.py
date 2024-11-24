import ast
import base64
from collections import defaultdict
import os
from typing import List, Dict, Union, Tuple, Any
import numpy as np
import pandas as pd


"""
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
"""


class QTableAgent:
    def __init__(self,
                 state_space: List[Dict[str, Union[str, Tuple[float, float], int]]],
                 action_space: List[Dict[str, Union[str, Tuple[float, float], int]]],
                 action_combination: bool = False):
        """
        Initialize the Q-Table Agent.

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

        # Discretize the state space and create a mapping to their original values
        self.state_bins: List[np.ndarray] = [self._discretize_space(dim) for dim in state_space]
        self.state_value_map: List[np.ndarray] = self.state_bins  # Save the bin edges as the mapping

        # Discretize the action space and create a mapping
        if action_combination:
            self.action_bins: np.ndarray = self._generate_action_combinations(action_space)
            self.action_value_map: np.ndarray = self.action_bins
        else:
            self.action_bins: List[np.ndarray] = [self._discretize_space(dim) for dim in action_space]
            self.action_value_map: List[np.ndarray] = self.action_bins

        # Initialize Q-Table and visit counts as defaultdicts
        self.q_table = defaultdict(lambda: 0.0)  # Default Q-value is 0.0
        self.visit_counts = defaultdict(lambda: 0)  # Default visit count is 0

        # Print Q-table size and dimension details
        self.print_q_table_info()

    def _discretize_space(self, space: Dict[str, Any]) -> np.ndarray:
        """Discretize a single state or action dimension."""
        if space['type'] == 'discrete':
            return np.arange(space['bins'])
        elif space['type'] == 'continuous':
            low, high = space['range']
            return np.linspace(low, high, space['bins'])
        else:
            raise ValueError("Invalid space type. Use 'discrete' or 'continuous'.")

    def _generate_action_combinations(self, action_space: List[Dict[str, Any]]) -> np.ndarray:
        """Generate all possible combinations of actions when action combination is enabled."""
        action_bins: List[np.ndarray] = [self._discretize_space(dim) for dim in action_space]
        return np.array(np.meshgrid(*action_bins)).T.reshape(-1, len(action_space))

    def _initialize_q_table(self) -> np.ndarray:
        """Initialize the Q-Table with zeros based on state and action space sizes."""
        state_sizes: List[int] = [len(bins) for bins in self.state_bins]
        action_size: int = len(self.action_bins) if self.action_combination else np.prod(
            [len(bins) for bins in self.action_bins])
        return np.zeros(state_sizes + [action_size])

    def print_q_table_info(self) -> None:
        """Print detailed information about the sparse Q-Table."""
        state_sizes = [len(bins) for bins in self.state_bins]
        action_size = len(self.action_bins) if self.action_combination else np.prod(
            [len(bins) for bins in self.action_bins])

        # Calculate sparsity information
        total_entries = np.prod(state_sizes) * action_size  # Full dense size
        non_zero_entries = len(self.q_table)  # Only entries in the sparse table

        print("Sparse Q-Table Details:")
        print(f" - State Space Dimensions: {len(state_sizes)}")
        print(f"   Sizes: {state_sizes}")
        print(f" - Action Space Dimensions: {1 if self.action_combination else len(self.action_bins)}")
        print(f"   Sizes: {action_size}")
        print(f" - Non-Zero Entries in Sparse Q-Table: {non_zero_entries}")
        print(f" - Total Possible Entries (Dense): {total_entries}")
        print(f" - Sparsity: {100 * (1 - non_zero_entries / total_entries):.2f}%")

    def save_q_table(self, file_path: str) -> None:
        """
        Save the Q-Table, visit counts, and agent configuration to a CSV file.
        Only rows with `Visit_Count` >= 1 are saved in the file.
        """
        print(f"Saving Q-Table and configuration to {file_path}...")

        # Calculate decimal places for states and actions (only for continuous)
        state_decimals = [
            max(0, int(-np.floor(np.log10(np.abs(bins[1] - bins[0]))))) + 1 if dim['type'] == 'continuous' else 0
            for bins, dim in zip(self.state_bins, self.state_space)
        ]
        action_decimals = [
            max(0, int(-np.floor(np.log10(np.abs(bins[1] - bins[0]))))) + 1 if dim['type'] == 'continuous' else 0
            for bins, dim in zip(self.action_bins, self.action_space)
        ]

        # Prepare data for saving
        data = []
        for key, q_value in self.q_table.items():
            state_indices = key[:len(self.state_bins)]
            action_indices = key[len(self.state_bins):]

            # Map state indices to values (only for continuous states)
            state_values = [
                round(self.state_bins[dim][state_idx], state_decimals[dim])
                if self.state_space[dim]['type'] == 'continuous' else None
                for dim, state_idx in enumerate(state_indices)
            ]
            # Map action indices to values (only for continuous actions)
            action_values = [
                round(self.action_bins[dim][action_idx], action_decimals[dim])
                if self.action_space[dim]['type'] == 'continuous' else None
                for dim, action_idx in enumerate(action_indices)
            ]

            # Collect visit count
            visit_count = self.visit_counts.get(key, 0)

            # Append row to data
            data.append(
                list(state_indices) + list(filter(lambda x: x is not None, state_values)) +
                list(action_indices) + list(filter(lambda x: x is not None, action_values)) +
                [q_value, visit_count]
            )

        # Define column names
        column_names = (
                [f"State_{i}_Index" for i, dim in enumerate(self.state_space)] +
                [f"State_{i}_Value" for i, dim in enumerate(self.state_space) if dim['type'] == 'continuous'] +
                [f"Action_{i}_Index" for i, dim in enumerate(self.action_space)] +
                [f"Action_{i}_Value" for i, dim in enumerate(self.action_space) if dim['type'] == 'continuous'] +
                ["Q_Value", "Visit_Count"]
        )

        # Convert to DataFrame
        df = pd.DataFrame(data, columns=column_names)

        # Filter rows where `Visit_Count` >= 1
        filtered_df = df[df["Visit_Count"] >= 1]

        # Add metadata about the agent configuration
        metadata = {
            "state_space": self.state_space,
            "action_space": self.action_space,
            "action_combination": self.action_combination
        }
        metadata_str = base64.b64encode(repr(metadata).encode("utf-8")).decode("utf-8")
        metadata_df = pd.DataFrame({"Metadata": [metadata_str]})

        # Save metadata and filtered Q-Table
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            metadata_df.to_csv(f, index=False, header=False)
            filtered_df.to_csv(f, index=False)
        print(f"Filtered Q-Table and configuration successfully saved to {file_path}.")

    @classmethod
    def load_q_table(cls, file_path: str) -> "QTableAgent":
        """
        Load the Q-Table, visit counts, and agent configuration from a CSV file,
        restoring both bin indices and actual values.
        """
        print(f"Loading sparse Q-Table and configuration from {file_path}...")

        with open(file_path, "r") as f:
            # Read metadata
            metadata_line = f.readline().strip()
            metadata_encoded = metadata_line.split(",", 1)[-1].strip()
            metadata_str = base64.b64decode(metadata_encoded).decode("utf-8")
            metadata = ast.literal_eval(metadata_str)

            # Read Q-Table data
            data = pd.read_csv(f)

        # Extract metadata for initialization
        state_space = metadata["state_space"]
        action_space = metadata["action_space"]
        action_combination = metadata["action_combination"]

        # Initialize a new agent
        agent = cls(state_space, action_space, action_combination)

        # Restore the sparse Q-Table and visit counts
        for _, row in data.iterrows():
            # Use saved indices directly for restoration
            state_indices = tuple(
                int(row[f"State_{dim}_Index"]) for dim in range(len(agent.state_bins))
            )
            if agent.action_combination:
                # If action_combination is True, handle as a single index
                action_index = int(row[f"Action_Index"])
            else:
                # Otherwise, handle as multi-dimensional action indices
                action_index = tuple(
                    int(row[f"Action_{dim}_Index"]) for dim in range(len(agent.action_bins))
                )

            # Retrieve Q-value and visit count
            q_value = row["Q_Value"]
            visit_count = row["Visit_Count"]

            # Update the sparse Q-Table
            key = state_indices + (action_index if agent.action_combination else action_index)
            agent.q_table[key] = q_value
            agent.visit_counts[key] = visit_count

        print("Sparse Q-Table successfully loaded.")
        agent.print_q_table_info()
        return agent

    def get_state_index(self, state: List[float]) -> Tuple[int, ...]:
        """
        Get the index of a given state in the Q-Table.
        If the state is out of range, clip it to the nearest boundary.

        :param state: A list of state values.
        :return: A tuple of indices corresponding to the discretized state.
        """
        state_index: List[int] = []
        for i, value in enumerate(state):
            bins = self.state_bins[i]
            # Clip the value to stay within the bin range
            clipped_value = np.clip(value, bins[0], bins[-1])
            state_index.append(np.digitize(clipped_value, bins) - 1)
        return tuple(state_index)

    def get_action_index(self, action: List[float]) -> int:
        """
        Get the index of a given action in the Q-Table.
        If the action is out of range, clip it to the nearest boundary.

        :param action: A list of action values.
        :return: An integer index corresponding to the discretized action.
        """
        if self.action_combination:
            # If combining actions, match action with the closest in the precomputed action_bins
            action = np.array(action).reshape(1, -1)
            distances = np.linalg.norm(self.action_bins - action, axis=1)
            return np.argmin(distances)
        else:
            # For non-combined actions, calculate the multi-index for each dimension
            action_index = []
            for i, value in enumerate(action):
                bins = self.action_bins[i]
                clipped_value = np.clip(value, bins[0], bins[-1])  # Clip to bin range
                action_index.append(np.digitize(clipped_value, bins) - 1)
            return np.ravel_multi_index(action_index, [len(bins) for bins in self.action_bins])

    def get_q_value(self, state: List[float], action: List[float]) -> float:
        """
        Get the Q-value for a given state and action.

        :param state: A list of state values.
        :param action: A list of action values.
        :return: The Q-value corresponding to the state and action.
        """
        state_idx = self.get_state_index(state)
        action_idx = self.get_action_index(action)
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

        # Calculate total actions based on the action_combination flag
        if self.action_combination:
            total_actions = len(self.action_bins)  # Combined actions
        else:
            total_actions = np.prod([len(bins) for bins in self.action_bins])  # Individual dimensions

        # Initialize Q-values for all actions
        q_values: np.ndarray = np.zeros(total_actions)
        for action_idx in range(total_actions):
            key = state_idx + (action_idx,)
            if key in self.q_table:
                q_values[action_idx] = self.q_table[key]

        # Calculate probabilities based on the specified strategy
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
        state_idx = self.get_state_index(state)
        action_idx = self.get_action_index(action)
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
        key = state_idx + (action_idx,)

        # Calculate total actions
        if self.action_combination:
            total_actions = len(self.action_bins)
        else:
            total_actions = np.prod([len(bins) for bins in self.action_bins])

        # Get next Q-values for all possible actions in the next state
        next_state_idx = self.get_state_index(next_state)
        next_q_values = [self.q_table.get(next_state_idx + (a,), 0) for a in range(total_actions)]
        next_q = max(next_q_values)

        # Compute the Q-Learning target
        target = reward + gamma * next_q

        # Update the Q-Table
        current_q = self.q_table.get(key, 0)
        self.q_table[key] = current_q + alpha * (target - current_q)

        # Update visit counts
        self.visit_counts[key] = self.visit_counts.get(key, 0) + 1
