import ast
import base64
from collections import defaultdict
import os
from itertools import product, combinations
from typing import List, Dict, Union, Tuple, Any, Optional
import numpy as np
import pandas as pd
from scipy.stats import norm

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

# Example 6: For using normal distribution to partition state and action spaces
# Continuous state or action spaces are divided using a standard normal distribution N(0, 1).
state_space = [
    {'type': 'continuous', 'bins': 5}  # Partition state space using normal distribution into 5 bins
]
action_space = [
    {'type': 'continuous', 'bins': 3}  # Partition action space using normal distribution into 3 bins
]
agent = QTableAgent(state_space, action_space, normal_partition_state=True, normal_partition_action=True)
"""


class QTableAgent:
    def __init__(self,
                 state_space: List[Dict[str, Union[str, Tuple[float, float], int]]],
                 action_space: List[Dict[str, Union[str, Tuple[float, float], int]]],
                 action_combination: bool = False,
                 normal_partition_state: bool = False,
                 normal_partition_action: bool = False):
        """
        Initialize the Q-Table Agent.

        :param state_space: A list of dictionaries defining the state space.
                            Each dictionary specifies:
                            - 'type': 'discrete' or 'continuous'
                            - 'bins': Number of bins for discrete space
                            - 'range': (low, high) for continuous space
                            - If normal_partition_state is True, 'range' is not required for continuous space.
        :param action_space: A list of dictionaries defining the action space.
                             Format is similar to `state_space`.
                             - If normal_partition_action is True, 'range' is not required for continuous space.
        :param action_combination: Whether actions can be combined into groups (True or False).
        :param normal_partition_state: Whether to use normal distribution to partition continuous state spaces.
        :param normal_partition_action: Whether to use normal distribution to partition continuous action spaces.
        """
        self.state_space: List[Dict[str, Any]] = state_space
        self.action_space: List[Dict[str, Any]] = action_space
        self.action_combination: bool = action_combination
        self.normal_partition_state: bool = normal_partition_state
        self.normal_partition_action: bool = normal_partition_action

        # Discretize the state space and create a mapping to their original values
        self.state_bins: List[np.ndarray] = [self._discretize_space(dim, normal_partition_state) for dim in state_space]
        self.state_value_map: List[np.ndarray] = self.state_bins  # Save the bin edges as the mapping

        # Discretize the action space and create a mapping
        if action_combination:
            self.action_bins: np.ndarray = self._generate_action_combinations(action_space)
            self.action_value_map: np.ndarray = self.action_bins
        else:
            self.action_bins: List[np.ndarray] = [self._discretize_space(dim, normal_partition_action) for dim in action_space]
            self.action_value_map: List[np.ndarray] = self.action_bins

        # Initialize Q-Table and visit counts as defaultdicts
        self.q_table = defaultdict(lambda: 0.0)  # Default Q-value is 0.0
        self.visit_counts = defaultdict(lambda: 0)  # Default visit count is 0

        # Print Q-table size and dimension details
        self.print_q_table_info()

    def clone(self, retain_visit_counts: bool = True) -> 'QTableAgent':
        """
        Create a new QTableAgent object that is an identical but independent copy of the current agent.
        Optionally clear visit counts in the cloned agent.

        :param retain_visit_counts: Whether to retain the visit counts in the cloned agent. Default is True.
        :return: A new QTableAgent object.
        """
        # Create a new instance with the same initialization parameters
        print("Cloning agent...")
        new_agent = QTableAgent(
            state_space=[dim.copy() for dim in self.state_space],  # Copy each state space definition
            action_space=[dim.copy() for dim in self.action_space],  # Copy each action space definition
            action_combination=self.action_combination,
            normal_partition_state=self.normal_partition_state,
            normal_partition_action=self.normal_partition_action
        )

        # Manually copy over attributes that are generated during initialization
        new_agent.state_bins = [np.copy(bin_edges) for bin_edges in self.state_bins]
        new_agent.state_value_map = [np.copy(bin_edges) for bin_edges in self.state_value_map]

        if self.action_combination:
            new_agent.action_bins = np.copy(self.action_bins)
            new_agent.action_value_map = np.copy(self.action_value_map)
        else:
            new_agent.action_bins = [np.copy(bin_edges) for bin_edges in self.action_bins]
            new_agent.action_value_map = [np.copy(bin_edges) for bin_edges in self.action_value_map]

        # Copy the Q-table
        new_agent.q_table = defaultdict(
            lambda: 0.0, {key: value for key, value in self.q_table.items()}
        )

        # Copy or clear visit counts based on the retain_visit_counts flag
        if retain_visit_counts:
            new_agent.visit_counts = defaultdict(
                lambda: 0, {key: value for key, value in self.visit_counts.items()}
            )
        else:
            new_agent.visit_counts = defaultdict(lambda: 0)  # Initialize all visit counts to 0

        return new_agent

    def _discretize_space(self, space: Dict[str, Any], normal_partition: bool) -> np.ndarray:
        """Discretize a single state or action dimension."""
        if space['type'] == 'discrete':
            return np.arange(space['bins'])
        elif space['type'] == 'continuous':
            if normal_partition:
                return self._discretize_normal(space['bins'])
            else:
                low, high = space['range']
                return np.linspace(low, high, space['bins'])
        else:
            raise ValueError("Invalid space type. Use 'discrete' or 'continuous'.")

    def _discretize_normal(self, bins: int) -> np.ndarray:
        """Discretize a continuous space using normal distribution N(0, 1)."""
        # Use percent point function (inverse of CDF) to get equal probability bins
        bin_edges = [norm.ppf((i + 1) / (bins + 1)) for i in range(bins)]
        return np.array(bin_edges)

    def _generate_action_combinations(self, action_space: List[Dict[str, Any]]) -> np.ndarray:
        """Generate all possible combinations of actions when action combination is enabled."""
        action_bins: List[np.ndarray] = [self._discretize_space(dim) for dim in action_space]
        return np.array(np.meshgrid(*action_bins)).T.reshape(-1, len(action_space))

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

    def query_q_table(
            self,
            filters: List[Dict[str, Union[str, Any]]] = None,
            logic: str = "AND"
    ) -> pd.DataFrame:
        """
        Query the Q-Table based on specified filters and return a filtered sub-table.

        :param filters: A list of dictionaries specifying the query conditions.
                        Each dictionary must have the following keys:
                        - 'field': The name of the field to filter (e.g., "State_0_Index", "Action_0_Value").
                        - 'operator': The comparison operator, such as '=', '>', '<', '>=', '<=', 'IN', or 'BETWEEN'.
                        - 'value': The value(s) to compare against (for 'BETWEEN', provide a tuple (low, high)).
        :param logic: The logic operator for combining filters. Options are "AND" or "OR". Default is "AND".
        :return: A Pandas DataFrame containing the filtered Q-Table.
        """
        # Prepare data for query
        data = []
        for key, q_value in self.q_table.items():
            state_indices = key[:len(self.state_bins)]
            action_indices = key[len(self.state_bins):]

            # Map state indices to values (only for continuous states)
            state_values = [
                self.state_bins[dim][idx] if self.state_space[dim]["type"] == "continuous" else None
                for dim, idx in enumerate(state_indices)
            ]
            # Map action indices to values (only for continuous actions)
            action_values = [
                self.action_bins[dim][idx] if self.action_space[dim]["type"] == "continuous" else None
                for dim, idx in enumerate(action_indices)
            ]

            visit_count = self.visit_counts.get(key, 0)

            # Append row to data
            data.append(
                list(state_indices) +
                list(filter(lambda x: x is not None, state_values)) +  # Only continuous state values
                list(action_indices) +
                list(filter(lambda x: x is not None, action_values)) +  # Only continuous action values
                [q_value, visit_count]
            )

        # Define column names
        column_names = (
                [f"State_{i}_Index" for i in range(len(self.state_bins))] +
                [f"State_{i}_Value" for i, dim in enumerate(self.state_space) if dim["type"] == "continuous"] +
                [f"Action_{i}_Index" for i, dim in enumerate(self.action_space)] +
                [f"Action_{i}_Value" for i, dim in enumerate(self.action_space) if dim["type"] == "continuous"] +
                ["Q_Value", "Visit_Count"]
        )

        # Convert to DataFrame
        df = pd.DataFrame(data, columns=column_names)

        # Filter rows where `Visit_Count` > 0
        filtered_df = df[df["Visit_Count"] > 0]

        # If no filters are provided, return the full table with Visit_Count > 0
        if not filters:
            return filtered_df

        # Apply filters based on logic
        if logic not in ["AND", "OR"]:
            raise ValueError(f"Unsupported logic: {logic}")

        filter_masks = []
        for condition in filters:
            field = condition["field"]
            operator = condition["operator"]
            value = condition["value"]

            if operator == "=":
                mask = filtered_df[field] == value
            elif operator == ">":
                mask = filtered_df[field] > value
            elif operator == ">=":
                mask = filtered_df[field] >= value
            elif operator == "<":
                mask = filtered_df[field] < value
            elif operator == "<=":
                mask = filtered_df[field] <= value
            elif operator == "IN":
                mask = filtered_df[field].isin(value)
            elif operator == "BETWEEN":
                if not isinstance(value, tuple) or len(value) != 2:
                    raise ValueError(f"BETWEEN operator requires a tuple of two values, got: {value}")
                low, high = value
                mask = (filtered_df[field] >= low) & (filtered_df[field] <= high)
            else:
                raise ValueError(f"Unsupported operator: {operator}")

            filter_masks.append(mask)

        # Combine masks with specified logic
        if logic == "AND":
            final_mask = filter_masks[0]
            for mask in filter_masks[1:]:
                final_mask &= mask
        elif logic == "OR":
            final_mask = filter_masks[0]
            for mask in filter_masks[1:]:
                final_mask |= mask

        return filtered_df[final_mask]

    @classmethod
    def compute_action_probabilities(
            cls,
            df: pd.DataFrame,
            strategy: str = "greedy",
            epsilon: float = 0.0,
            temperature: float = 1.0
    ) -> pd.DataFrame:
        """
        Compute action probabilities for each state using the specified strategy, organize actions horizontally,
        and retain both the action index and `Visit_Count` columns.

        :param df: Input DataFrame containing Q-values and visit counts.
        :param strategy: The strategy to use for computing action probabilities. Options are "greedy" or "softmax".
        :param epsilon: Epsilon for epsilon-greedy strategy. Only used if strategy is "greedy".
        :param temperature: Temperature for softmax strategy. Only used if strategy is "softmax".
        :return: A new DataFrame with states and actions represented horizontally.
        """
        # Verify input strategy
        if strategy not in ["greedy", "softmax"]:
            raise ValueError("Invalid strategy. Use 'greedy' or 'softmax'.")

        # Extract state and action columns dynamically
        state_index_columns = [col for col in df.columns if col.startswith("State_") and "Index" in col]
        state_value_columns = [col for col in df.columns if col.startswith("State_") and "Value" in col]
        action_index_columns = [col for col in df.columns if col.startswith("Action_") and "Index" in col]

        if not action_index_columns:
            raise ValueError("No valid action index columns detected in the input DataFrame.")

        # Group by unique state and calculate probabilities for each action
        grouped = df.groupby(state_index_columns + state_value_columns)

        # Dynamically generate all possible action combinations
        unique_action_combinations = np.array(
            np.meshgrid(*[df[col].unique() for col in action_index_columns])
        ).T.reshape(-1, len(action_index_columns))
        all_action_indices = [tuple(row) for row in unique_action_combinations]

        action_column_names = [
            f"{'_'.join(f'{col}_{value}' for col, value in zip(action_index_columns, action))}_Probability"
            for action in all_action_indices
        ]
        new_columns = state_index_columns + state_value_columns + action_column_names + ["Visit_Count"]

        new_data = []

        for group_keys, group in grouped:
            # Dynamically unpack state indices and values
            state_indices = group_keys[:len(state_index_columns)]
            state_values = group_keys[len(state_index_columns):]

            q_values = group["Q_Value"].values
            actions = group[action_index_columns].to_numpy()
            visit_count = group["Visit_Count"].sum()  # Sum visit counts for this state

            # Compute probabilities based on strategy
            if strategy == "greedy":
                probabilities = np.zeros(len(all_action_indices), dtype=float)
                best_action_idx = np.argmax(q_values)
                for i, action in enumerate(actions):
                    action_tuple = tuple(action)
                    action_idx = all_action_indices.index(action_tuple)
                    probabilities[action_idx] = epsilon / len(all_action_indices)
                probabilities[all_action_indices.index(tuple(actions[best_action_idx]))] += 1.0 - epsilon
            elif strategy == "softmax":
                exp_q_values = np.exp(q_values / temperature)
                softmax_probs = exp_q_values / np.sum(exp_q_values)
                probabilities = np.zeros(len(all_action_indices), dtype=float)
                for i, action in enumerate(actions):
                    action_tuple = tuple(action)
                    action_idx = all_action_indices.index(action_tuple)
                    probabilities[action_idx] = softmax_probs[i]

            # Append state, action probabilities, and visit count as one row
            row = list(state_indices) + list(state_values) + list(probabilities) + [visit_count]
            new_data.append(row)

        # Create and return the new DataFrame
        new_df = pd.DataFrame(new_data, columns=new_columns)
        return new_df

    @staticmethod
    def compute_mutual_information(
            df: pd.DataFrame,
            group1_columns: Union[str, List[str]],
            group2_columns: Optional[Union[str, List[str]]] = None,
            use_visit_count: bool = False
    ) -> float:
        """
        Compute mutual information (MI) between two groups of features or between features and actions.

        :param df: A DataFrame containing the precomputed probability distributions and optionally visit counts.
        :param group1_columns: A single column or a list of columns representing the first group (e.g., features).
        :param group2_columns: A single column, a list of columns, or None. If None, computes self-MI for group1.
                               If the group represents actions, provide the base names, e.g., "Action_0_Index".
        :param use_visit_count: Whether to weight probabilities by visit counts. Default is False.
        :return: The mutual information (MI) value between the two groups.
        """
        # Make a copy of the DataFrame to avoid modifying the original
        df = df.copy()

        # Ensure group1_columns is a list
        if isinstance(group1_columns, str):
            group1_columns = [group1_columns]

        # Process the second group (group2_columns)
        if group2_columns is None:
            group2_columns = group1_columns  # Compute self-MI within group1
        elif isinstance(group2_columns, str):
            group2_columns = [group2_columns]

        # Check if the second group contains actions
        is_action_group = all(col.startswith("action_dim_") for col in group2_columns)

        if is_action_group:
            # If the group is actions, find corresponding probability columns
            full_action_columns = []
            for base_col in group2_columns:
                matching_columns = [
                    col for col in df.columns if col.startswith(base_col) and "_Probability" in col
                ]
                if not matching_columns:
                    raise ValueError(f"Invalid or missing action columns for base: {base_col}")
                full_action_columns.extend(matching_columns)
            group2_columns = full_action_columns

        # Check for visit counts if `use_visit_count` is enabled
        if use_visit_count and "count" not in df.columns:
            raise ValueError("Visit count column ('count') is required when 'use_visit_count=True'.")

        # Normalize probabilities, optionally weighted by visit counts
        if use_visit_count:
            df["Weighted_Visit_Count"] = df["count"] / df["count"].sum()
            if is_action_group:
                for col in group2_columns:
                    df[col] *= df["Weighted_Visit_Count"]

        # Normalize the probabilities to ensure valid distribution
        if is_action_group:
            total_prob = df[group2_columns].sum().sum()
            df[group2_columns] /= total_prob
        else:
            # For feature groups, normalize visit counts
            df["Visit_Prob"] = df["count"] / df["count"].sum()

        # Compute joint probability P(Group1, Group2)
        if is_action_group:
            joint_prob = df.groupby(group1_columns, as_index=False)[group2_columns].sum()
        else:
            # For feature groups, compute joint probabilities using Visit_Prob
            joint_prob = df.groupby(group1_columns + group2_columns)["Visit_Prob"].sum().reset_index()
            joint_prob.rename(columns={"Visit_Prob": "P(Group1,Group2)"}, inplace=True)

        # Compute marginal probabilities P(Group1) and P(Group2)
        if is_action_group:
            joint_prob["P(Group1)"] = joint_prob[group2_columns].sum(axis=1)
            marginal_group2_prob = df[group2_columns].sum()
        else:
            # For feature groups, compute marginals
            marginal_group1_prob = joint_prob.groupby(group1_columns)["P(Group1,Group2)"].sum().reset_index()
            marginal_group1_prob.rename(columns={"P(Group1,Group2)": "P(Group1)"}, inplace=True)

            marginal_group2_prob = joint_prob.groupby(group2_columns)["P(Group1,Group2)"].sum().reset_index()
            marginal_group2_prob.rename(columns={"P(Group1,Group2)": "P(Group2)"}, inplace=True)

        # Merge joint and marginal probabilities
        if not is_action_group:
            joint_prob = joint_prob.merge(marginal_group1_prob, on=group1_columns)
            joint_prob = joint_prob.merge(marginal_group2_prob, on=group2_columns)

        # Calculate mutual information
        mi = 0.0
        if is_action_group:
            for _, row in joint_prob.iterrows():
                px = row["P(Group1)"]  # P(Group1)
                for group2_col in group2_columns:
                    pa = marginal_group2_prob[group2_col]  # P(Group2)
                    p_group1_group2 = row[group2_col]  # P(Group1, Group2)
                    if p_group1_group2 > 0:  # Avoid log(0)
                        mi += p_group1_group2 * np.log2(p_group1_group2 / (px * pa))
        else:
            for _, row in joint_prob.iterrows():
                p_group1_group2 = row["P(Group1,Group2)"]
                px = row["P(Group1)"]
                py = row["P(Group2)"]
                if p_group1_group2 > 0:  # Avoid log(0)
                    mi += p_group1_group2 * np.log2(p_group1_group2 / (px * py))

        return mi

    # @staticmethod
    # def compute_average_kl_divergence_between_dfs(
    #         df1: pd.DataFrame,
    #         df2: pd.DataFrame,
    #         visit_threshold: int = 0
    # ) -> float:
    #     """
    #     Compute the average KL divergence between action distributions in two DataFrames for shared states.
    #
    #     :param df1: The first DataFrame, generated using `compute_action_probabilities`.
    #     :param df2: The second DataFrame, generated using `compute_action_probabilities`.
    #     :param visit_threshold: Minimum visit count required for states to be considered. Default is 0.
    #     :return: The average KL divergence for shared states based on action distributions.
    #     """
    #     # Identify the state columns
    #     state_columns = [col for col in df1.columns if col.startswith("State_") and "Index" in col]
    #
    #     # Identify action probability columns in both DataFrames
    #     action_prob_columns_df1 = [col for col in df1.columns if col.endswith("_Probability")]
    #     action_prob_columns_df2 = [col for col in df2.columns if col.endswith("_Probability")]
    #
    #     # Ensure the action columns match between the two DataFrames
    #     if set(action_prob_columns_df1) != set(action_prob_columns_df2):
    #         raise ValueError("Action probability columns do not match between the two DataFrames.")
    #
    #     # Merge the two DataFrames on state columns
    #     merged_df = pd.merge(
    #         df1[state_columns + action_prob_columns_df1 + ["Visit_Count"]],
    #         df2[state_columns + action_prob_columns_df2 + ["Visit_Count"]],
    #         on=state_columns,
    #         suffixes=("_df1", "_df2")
    #     )
    #
    #     # Apply visit threshold filter if specified
    #     if visit_threshold > 0:
    #         merged_df = merged_df[
    #             (merged_df["Visit_Count_df1"] > visit_threshold) &
    #             (merged_df["Visit_Count_df2"] > visit_threshold)
    #             ]
    #
    #     # If no states meet the threshold, return 0 to avoid division by zero
    #     if merged_df.empty:
    #         return 0.0
    #
    #     # Compute KL divergence for each shared state
    #     kl_divergences = []
    #     for _, row in merged_df.iterrows():
    #         # Adjust the column names to include suffixes
    #         p = row[[f"{col}_df1" for col in action_prob_columns_df1]].values
    #         q = row[[f"{col}_df2" for col in action_prob_columns_df2]].values
    #
    #         # Avoid division by zero and log(0) by adding a small epsilon
    #         epsilon = 1e-20
    #         p = np.clip(p, epsilon, 1.0)
    #         q = np.clip(q, epsilon, 1.0)
    #
    #         # Normalize the distributions to ensure they sum to 1
    #         p /= p.sum()
    #         q /= q.sum()
    #
    #         # Compute the KL divergence for this state
    #         kl_divergence = np.sum(p * np.log(p / q))
    #         kl_divergences.append(kl_divergence)
    #
    #     # Compute and return the average KL divergence
    #     average_kl_divergence = np.mean(kl_divergences)
    #     return average_kl_divergence

    @staticmethod
    def compute_average_kl_divergence_between_dfs(
            df1: pd.DataFrame,
            df2: pd.DataFrame,
            visit_threshold: int = 0,
            weighted_by_visitation: bool = False
    ) -> float:
        """
        Compute the average KL divergence between action distributions in two DataFrames for shared states.
        The filtering and optional weighting are based on the first DataFrame's visitation count.

        :param df1: The first DataFrame, generated using `compute_action_probabilities`.
        :param df2: The second DataFrame, generated using `compute_action_probabilities`.
        :param visit_threshold: Minimum visit count in df1 required for states to be considered. Default is 0.
        :param weighted_by_visitation: Whether to weight KL divergence by df1's visitation distribution. Default is False.
        :return: The average KL divergence for shared states based on action distributions.
        """
        # Identify the state columns
        state_columns = [col for col in df1.columns if col.startswith("State_") and "Index" in col]

        # Identify action probability columns in both DataFrames
        action_prob_columns_df1 = [col for col in df1.columns if col.endswith("_Probability")]
        action_prob_columns_df2 = [col for col in df2.columns if col.endswith("_Probability")]

        # Ensure the action columns match between the two DataFrames
        if set(action_prob_columns_df1) != set(action_prob_columns_df2):
            raise ValueError("Action probability columns do not match between the two DataFrames.")

        # Merge the two DataFrames on state columns
        merged_df = pd.merge(
            df1[state_columns + action_prob_columns_df1 + ["Visit_Count"]],
            df2[state_columns + action_prob_columns_df2],
            on=state_columns,
            suffixes=("_df1", "_df2")
        )

        # Apply visit threshold filter based on df1
        if visit_threshold > 0:
            merged_df = merged_df[merged_df["Visit_Count"] > visit_threshold]

        # If no states meet the threshold, return 0 to avoid division by zero
        if merged_df.empty:
            return 0.0

        # Compute KL divergence for each shared state
        kl_divergences = []
        visitation_weights = []
        for _, row in merged_df.iterrows():
            # Extract the probabilities from both DataFrames
            p = row[[f"{col}_df1" for col in action_prob_columns_df1]].values
            q = row[[f"{col}_df2" for col in action_prob_columns_df2]].values

            # Avoid division by zero and log(0) by adding a small epsilon
            epsilon = 1e-20
            p = np.clip(p, epsilon, 1.0)
            q = np.clip(q, epsilon, 1.0)

            # Normalize the distributions to ensure they sum to 1
            p /= p.sum()
            q /= q.sum()

            # Compute the KL divergence for this state
            kl_divergence = np.sum(p * np.log(p / q))
            kl_divergences.append(kl_divergence)

            # Append the visit count from df1 if weighting is enabled
            if weighted_by_visitation:
                visitation_weights.append(row["Visit_Count"])

        # Compute and return the average KL divergence
        if weighted_by_visitation:
            visitation_weights = np.array(visitation_weights) / np.sum(visitation_weights)  # Normalize weights
            average_kl_divergence = np.sum(np.array(kl_divergences) * visitation_weights)
        else:
            average_kl_divergence = np.mean(kl_divergences)

        return average_kl_divergence


class _QTableAgent:
    def __init__(self, action_space: List[int]):
        """
        Initialize the Q-Table agent.

        :param action_space: List of possible values for each action dimension.
        """
        self.q_table = defaultdict(lambda: 0.0)  # Flattened Q-Table with state-action tuple keys
        self.action_space = action_space
        self.print_q_table_info()

    def clone(self) -> '_QTableAgent':
        """
        Create a deep copy of the Q-Table agent.

        :return: A new QTableAgent instance with the same Q-Table.
        """
        new_agent = _QTableAgent(self.action_space)
        new_agent.q_table = self.q_table.copy()
        return new_agent

    def print_q_table_info(self) -> None:
        """
        Print information about the Q-Table size and its structure.
        """
        print(f"Q-Table Size: {len(self.q_table)} state-action pairs")
        for key, value in list(self.q_table.items())[:5]:  # Print first 5 entries for preview
            print(f"State-Action: {key}, Q-Value: {value}")

    def save_q_table(self, file_path: str = None) -> pd.DataFrame:
        """
        Save the Q-Table to a CSV file and/or return as a DataFrame.

        :param file_path: Path to save the file.
        :return: DataFrame representation of the Q-Table.
        """
        data = []
        for (state, action), q_value in self.q_table.items():
            row = {f"state_dim_{i}": state[i] for i in range(len(state))}
            row.update({f"action_dim_{j}": action[j] for j in range(len(action))})
            row.update({"q_value": q_value})
            data.append(row)
        df = pd.DataFrame(data)
        if file_path:
            df.to_csv(file_path, index=False)
            print(f"Q-Table saved to {file_path}.")
        return df

    @classmethod
    def load_q_table(cls, file_path: str = None, df: pd.DataFrame = None) -> "_QTableAgent":
        """
        Load a Q-Table from a CSV file or a DataFrame.

        :param file_path: Path to the saved file.
        :param df: DataFrame representation of the Q-Table.
        :return: An instance of QTableAgent.
        """
        if file_path:
            df = pd.read_csv(file_path)
        elif df is None:
            raise ValueError("Either file_path or df must be provided.")

        state_columns = sorted([col for col in df.columns if col.startswith("state_dim_")], key=lambda x: int(x.split("_")[-1]))
        action_columns = sorted([col for col in df.columns if col.startswith("action_dim_")], key=lambda x: int(x.split("_")[-1]))

        # Determine action space sizes from unique values in each action column
        action_space = [df[col].nunique() for col in action_columns]

        agent = cls(action_space=action_space)
        for _, row in df.iterrows():
            state = tuple(row[col] for col in state_columns)
            action = tuple(row[col] for col in action_columns)
            q_value = float(row["q_value"])
            agent.q_table[(state, action)] = q_value
        print(f"Q-Table loaded from {'file' if file_path else 'DataFrame'}.")
        return agent

    def get_action_probabilities(self, state: np.ndarray, strategy: str = "greedy", temperature: float = 1.0) -> np.ndarray:
        """
        Calculate action probabilities based on the specified strategy.

        :param state: The current state.
        :param strategy: Strategy to calculate probabilities ("greedy" or "softmax").
        :param temperature: Temperature parameter for softmax.
        :return: An array of action probabilities.
        """
        state_key = tuple(state)

        # Generate all possible actions
        possible_actions = self._generate_all_possible_actions()

        # Retrieve Q-values for all actions
        q_values = np.array([self.q_table.get((state_key, tuple(action)), 0.0) for action in possible_actions])

        if strategy == "greedy":
            probabilities = np.zeros_like(q_values, dtype=float)
            if len(q_values) > 0:
                probabilities[np.argmax(q_values)] = 1.0
        elif strategy == "softmax":
            if np.all(q_values == 0):  # Handle all-zero Q-values
                probabilities = np.ones_like(q_values, dtype=float) / len(q_values)
            else:
                exp_values = np.exp(q_values / temperature)
                probabilities = exp_values / np.sum(exp_values)
        else:
            raise ValueError("Invalid strategy. Use 'greedy' or 'softmax'.")

        return probabilities

    def _generate_all_possible_actions(self) -> List[List[int]]:
        """
        Generate all possible combinations of actions given the action space.

        :return: List of all possible actions.
        """
        if not self.action_space:
            raise ValueError("Action space is not defined.")
        return [list(action) for action in product(*[range(dim) for dim in self.action_space])]

    def update(self, state: np.ndarray, action: List[int], reward: float, next_state: np.ndarray, alpha: float = 0.1, gamma: float = 0.99) -> None:
        """
        Update the Q-Table using the Q-learning update rule.

        :param state: The current state.
        :param action: The action taken (multi-dimensional).
        :param reward: The received reward.
        :param next_state: The next state.
        :param alpha: Learning rate.
        :param gamma: Discount factor.
        """
        state_key = tuple(state)
        next_state_key = tuple(next_state)

        # Generate all possible next actions
        possible_actions = self._generate_all_possible_actions()

        # Compute the best next action's Q-value
        best_next_action_value = max(
            [self.q_table.get((next_state_key, tuple(a)), 0.0) for a in possible_actions],
            default=0.0
        )

        # Update Q-value for the current state-action pair
        td_target = reward + gamma * best_next_action_value
        td_error = td_target - self.q_table[(state_key, tuple(action))]
        self.q_table[(state_key, tuple(action))] += alpha * td_error

        print(f"Updated Q-value for state {state_key}, action {action}: {self.q_table[(state_key, tuple(action))]}")

# Example Usage
if __name__ == "__main__":
    import random
    import gymnasium as gym
    from utils import merge_q_table_with_counts, compute_action_probabilities
    from discretizer import Discretizer
    from wrappers import DiscretizerWrapper

    # Define ranges and number of buckets for each dimension based on CartPole state space
    state_ranges = [(-4.8, 4.8), (-3.4, 3.4), (-0.418, 0.418), (-3.4, 3.4)]  # CartPole observation ranges
    action_ranges = [(0, 1)]  # Two discrete actions: 0 and 1

    state_buckets = [5, 5, 5, 5]  # Discretize each state variable into 5 buckets
    action_buckets = [0]  # No discretization for actions

    state_discretizer = Discretizer(state_ranges, state_buckets)
    action_discretizer = Discretizer(action_ranges, action_buckets)

    env = gym.make("CartPole-v1")
    wrapped_env = DiscretizerWrapper(env, state_discretizer, action_discretizer, enable_counting=True)

    agent = _QTableAgent(action_space=[len(action_ranges[i]) for i in range(len(action_ranges))])

    # Train the agent
    state, info = wrapped_env.reset()
    for episode in range(10):
        done = False
        while not done:
            action = random.choice(agent._generate_all_possible_actions())  # Choose a random action
            next_state, reward, done, truncated, _ = wrapped_env.step(action[0])  # Use the first action dimension for CartPole
            agent.update(state, action, reward, next_state)
            state = next_state if not done else wrapped_env.reset()[0]

    # Save Q-table and simulate wrapper counts
    q_table_df = agent.save_q_table("./q_table.csv")
    print("Saved Q-Table DataFrame:")
    print(q_table_df.head())

    counts_df = wrapped_env.export_counts("./counts.csv")
    print("Exported Counts DataFrame:")
    print(counts_df.head())

    # Merge Q-table with counts
    merged_df = merge_q_table_with_counts(q_table_df, counts_df)
    merged_df.to_csv("merged_test.csv", index=False)
    print("Merged Q-Table with Counts:")
    print(merged_df.head())

    merged_df = compute_action_probabilities(merged_df, "softmax")
    print("Action Probabilities:")
    print(merged_df.head())

    # List of all state features
    state_features = ["state_dim_0", "state_dim_1", "state_dim_2", "state_dim_3"]

    # Iterate over all possible combinations of features
    for r in range(1, len(state_features) + 1):  # r is the number of features in each combination
        for feature_combination in combinations(state_features, r):
            # Compute mutual information for the current feature combination
            mi = QTableAgent.compute_mutual_information(
                merged_df, list(feature_combination), "action_dim_0", use_visit_count=True
            )
            # Print the results with detailed explanation
            print(f"Mutual Information for Features {feature_combination} with action_dim_0: {mi}")

    # Compute mutual information for all features as values
    mi = QTableAgent.compute_mutual_information(
        merged_df, state_features, "action_dim_0", use_visit_count=True
    )
    print(f"Mutual Information for All Features as Values with action_dim_0: {mi}")

    # Load agent from merged DataFrame
    loaded_agent = _QTableAgent.load_q_table(df=q_table_df)

    # Print loaded Q-table for verification
    loaded_agent.print_q_table_info()

    # Test action probabilities
    test_state = np.array([0.1, 0.0, -0.1, 0.0])
    print("Action probabilities (greedy):", loaded_agent.get_action_probabilities(test_state, strategy="greedy"))
    print("Action probabilities (softmax):", loaded_agent.get_action_probabilities(test_state, strategy="softmax", temperature=1.0))
