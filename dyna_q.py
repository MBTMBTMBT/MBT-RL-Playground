from collections import defaultdict
from itertools import product

import numpy as np
import pandas as pd
import scipy.stats
from typing import List, Tuple, Optional, Dict


class Discretizer:
    def __init__(self, ranges: List[Tuple[float, float]], num_buckets: List[int],
                 normal_params: List[Optional[Tuple[float, float]]] = None):
        """
        Initialize the Discretizer.

        :param ranges: List of tuples specifying the min and max value for each dimension. [(min1, max1), (min2, max2), ...]
        :param num_buckets: List of integers specifying the number of buckets for each dimension. [buckets1, buckets2, ...]
                            A value of -1 means no discretization (output the original number),
                            a value of 0 means discretize into integers within the range,
                            and a value of 1 means all values map to the single bucket midpoint.
        :param normal_params: List of tuples specifying the mean and std for normal distribution for each dimension.
                              If None, use uniform distribution. [(mean1, std1), None, (mean3, std3), ...]
        """
        assert len(ranges) == len(num_buckets), "Ranges and num_buckets must have the same length."
        if normal_params:
            assert len(normal_params) == len(num_buckets), "normal_params must match the length of num_buckets."

        self.ranges: List[Tuple[float, float]] = ranges
        self.num_buckets: List[int] = [
            int(np.floor(max_val) - np.ceil(min_val) + 1) if buckets == 0 else buckets
            for (min_val, max_val), buckets in zip(ranges, num_buckets)
        ]
        self.normal_params: List[Optional[Tuple[float, float]]] = normal_params if normal_params else [None] * len(
            num_buckets)
        self.bucket_midpoints: List[List[float]] = []

        for i, ((min_val, max_val), buckets, normal_param) in enumerate(zip(ranges, num_buckets, self.normal_params)):
            if buckets == -1:
                self.bucket_midpoints.append([])
            elif buckets == 0:
                # Discretize into integers within range
                midpoints = list(range(int(np.ceil(min_val)), int(np.floor(max_val)) + 1))
                self.bucket_midpoints.append(midpoints)
            elif buckets == 1:
                midpoint = [(min_val + max_val) / 2]
                self.bucket_midpoints.append(midpoint)
            else:
                if normal_param:
                    mean, std = normal_param
                    # Restrict edges to a finite range if necessary
                    edges = [scipy.stats.norm.ppf(min(max((j / buckets), 1e-6), 1 - 1e-6), loc=mean, scale=std) for j in range(buckets + 1)]
                    midpoints = [round((edges[j] + edges[j + 1]) / 2, 6) for j in range(buckets)]
                else:
                    step = (max_val - min_val) / buckets
                    midpoints = [round(min_val + (i + 0.5) * step, 6) for i in range(buckets)]
                self.bucket_midpoints.append(midpoints)

    def discretize(self, vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Discretize a vector.

        :param vector: Input vector to discretize. Must have the same length as ranges and num_buckets.
        :return: A tuple of two vectors:
                 - The first vector contains the bucket midpoints (or original value if no discretization).
                 - The second vector contains the bucket indices (or -1 if no discretization).
        """
        assert len(vector) == len(self.ranges), "Input vector must have the same length as ranges."

        midpoints: List[float] = []
        bucket_indices: List[int] = []

        for i, (value, (min_val, max_val), buckets, normal_param) in enumerate(zip(vector, self.ranges, self.num_buckets, self.normal_params)):
            if buckets == -1:
                # No discretization
                midpoints.append(value)
                bucket_indices.append(-1)
            elif buckets == 0:
                # Discretize into integers within range
                int_range = list(range(int(np.ceil(min_val)), int(np.floor(max_val)) + 1))
                closest = min(int_range, key=lambda x: abs(x - value))
                midpoints.append(closest)
                bucket_indices.append(int_range.index(closest))
            elif buckets == 1:
                # Single bucket, always map to midpoint
                midpoint = round((min_val + max_val) / 2, 6)
                midpoints.append(midpoint)
                bucket_indices.append(0)
            else:
                if normal_param:
                    mean, std = normal_param
                    bucket_edges = [scipy.stats.norm.ppf(min(max((j / buckets), 1e-6), 1 - 1e-6), loc=mean, scale=std) for j in range(buckets + 1)]
                    for idx in range(buckets):
                        if bucket_edges[idx] <= value < bucket_edges[idx + 1]:
                            midpoints.append(round((bucket_edges[idx] + bucket_edges[idx + 1]) / 2, 6))
                            bucket_indices.append(idx)
                            break
                    else:
                        midpoints.append(round((bucket_edges[0] + bucket_edges[-1]) / 2, 6))  # Fallback to average if out of range
                        bucket_indices.append(-1)
                else:
                    step = (max_val - min_val) / buckets
                    bucket = int((value - min_val) / step)
                    bucket = min(max(bucket, 0), buckets - 1)  # Ensure bucket index is within bounds
                    midpoints.append(self.bucket_midpoints[i][bucket])
                    bucket_indices.append(bucket)

        return np.array(midpoints), np.array(bucket_indices)

    def encode_indices(self, indices: List[int]) -> int:
        """
        Encode bucket indices into a unique integer.

        :param indices: List of bucket indices.
        :return: Encoded integer.
        """
        assert len(indices) == len(self.num_buckets), "Indices must match the number of dimensions."
        encoded = 0
        multiplier = 1

        for index, buckets in zip(reversed(indices), reversed(self.num_buckets)):
            if buckets != -1:
                encoded += index * multiplier
                multiplier *= buckets

        return encoded

    def decode_indices(self, code: int) -> List[int]:
        """
        Decode a unique integer back into bucket indices.

        :param code: Encoded integer.
        :return: List of bucket indices.
        """
        indices = []

        for buckets in self.num_buckets:
            if buckets == -1:
                indices.append(-1)
            else:
                indices.append(code % buckets)
                code //= buckets

        return list(reversed(indices))

    def indices_to_midpoints(self, indices: List[int]) -> List[float]:
        """
        Convert bucket indices to bucket midpoints.

        :param indices: List of bucket indices.
        :return: List of bucket midpoints.
        """
        midpoints = []

        for index, midpoints_list in zip(indices, self.bucket_midpoints):
            if index == -1:
                midpoints.append(None)
            else:
                midpoints.append(midpoints_list[index])

        return midpoints

    def list_all_possible_combinations(self) -> Tuple[List[Tuple[float, ...]], List[Tuple[int, ...]]]:
        """
        List all possible combinations of bucket midpoints and their indices.

        :return: A tuple of two lists:
                 - The first list contains tuples of all possible bucket midpoints.
                 - The second list contains tuples of the corresponding bucket indices.
        """
        all_midpoints = []
        all_indices = []

        for midpoints, buckets in zip(self.bucket_midpoints, self.num_buckets):
            if buckets == -1:
                all_midpoints.append([None])
                all_indices.append([-1])
            else:
                all_midpoints.append(midpoints)
                all_indices.append(list(range(len(midpoints))))

        midpoints_product = list(product(*all_midpoints))
        indices_product = list(product(*all_indices))

        return midpoints_product, indices_product

    def count_possible_combinations(self) -> int:
        """
        Count the total number of possible combinations of bucket midpoints.

        :return: The total number of combinations.
        """
        total_combinations = 1
        for midpoints, buckets in zip(self.bucket_midpoints, self.num_buckets):
            if buckets != -1:
                total_combinations *= len(midpoints)
        return total_combinations

    def print_buckets(self) -> None:
        """
        Print all buckets and their corresponding ranges.
        """
        for i, ((min_val, max_val), buckets, normal_param) in enumerate(zip(self.ranges, self.num_buckets, self.normal_params)):
            if buckets == -1:
                print(f"Dimension {i}: No discretization")
            elif buckets == 0:
                int_range = list(range(int(np.ceil(min_val)), int(np.floor(max_val)) + 1))
                print(f"Dimension {i}: Integer buckets {int_range}")
            elif buckets == 1:
                midpoint = round((min_val + max_val) / 2, 6)
                print(f"Dimension {i}: Single bucket at midpoint {midpoint}")
            else:
                if normal_param:
                    mean, std = normal_param
                    edges = [scipy.stats.norm.ppf(min(max((j / buckets), 1e-6), 1 - 1e-6), loc=mean, scale=std) for j in range(buckets + 1)]
                    for j in range(buckets):
                        bucket_min = round(edges[j], 6)
                        bucket_max = round(edges[j + 1], 6)
                        print(f"Dimension {i}, Bucket {j}: Range [{bucket_min}, {bucket_max})")
                else:
                    step = (max_val - min_val) / buckets
                    for j in range(buckets):
                        bucket_min = round(min_val + j * step, 6)
                        bucket_max = round(bucket_min + step, 6)
                        print(f"Dimension {i}, Bucket {j}: Range [{bucket_min}, {bucket_max})")


class TabularQAgent:
    def __init__(self, state_discretizer: Discretizer, action_discretizer: Discretizer, print_info: bool = True):
        self.state_discretizer = state_discretizer
        self.action_discretizer = action_discretizer
        self.q_table = defaultdict(lambda: 0.0)  # Flattened Q-Table with state-action tuple keys
        self.visit_table = defaultdict(lambda: 0)  # Uses the same keys of the Q-Table to do visit count.
        if print_info:
            self.print_q_table_info()
        self.all_actions_encoded = [
            self.action_discretizer.encode_indices([*indices])
            for indices in self.action_discretizer.list_all_possible_combinations()[1]
        ].sort()

    def clone(self) -> 'TabularQAgent':
        """
        Create a deep copy of the Q-Table agent.

        :return: A new QTableAgent instance with the same Q-Table.
        """
        new_agent = TabularQAgent(self.state_discretizer, self.action_discretizer, print_info=False)
        new_agent.q_table = self.q_table.copy()
        new_agent.visit_table = self.visit_table.copy()
        new_agent.print_q_table_info()
        return new_agent

    def print_q_table_info(self) -> None:
        """
        Print information about the Q-Table size and its structure.
        """
        print("Q-Table Information:")
        print(f"State Discretizer:")
        self.state_discretizer.print_buckets()
        print(f"Action Discretizer:")
        self.action_discretizer.print_buckets()
        total_combinations = (self.state_discretizer.count_possible_combinations()
                              * self.action_discretizer.count_possible_combinations())
        print(f"Q-Table Size: {len(self.q_table)} state-action pairs / total combinations: {total_combinations}.")
        print(f"State-Action usage: {len(self.q_table) / total_combinations * 100:.2f}%.")

    def save_q_table(self, file_path: str = None) -> pd.DataFrame:
        """
        Save the Q-Table to a CSV file and/or return as a DataFrame.

        :param file_path: Path to save the file.
        :return: DataFrame representation of the Q-Table.
        """
        # list all possible actions (but not states, cause states are just too many)
        checked_states = set()
        data = []
        for (encoded_state, encoded_action), q_value in self.q_table.items():
            if encoded_state in checked_states:
                continue
            row = {f"state": encoded_state}
            row.update({f"action_{a}_q_value": self.q_table[(encoded_state, a)] for a in self.all_actions_encoded})
            row.update({f"action_{a}_visit_count": self.visit_table[(encoded_state, a)] for a in self.all_actions_encoded})
            row.update({"total_visit_count": sum([self.visit_table[(encoded_state, a)] for a in self.all_actions_encoded])})
            data.append(row)
            checked_states.add(encoded_state)
        df = pd.DataFrame(data)
        if file_path:
            df.to_csv(file_path, index=False)
            print(f"Q-Table saved to {file_path}.")
        return df

    def load_q_table(self, file_path: str = None, df: pd.DataFrame = None):
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

        if len(self.q_table) > 0 or len(self.visit_table) > 0:
            print("Warning: Loading a Q-Table that already has data. Part of them might be overwritten.")

        for _, row in df.iterrows():
            encoded_state = int(row["state"])
            for a in self.all_actions_encoded:
                self.q_table[(encoded_state, a)] = row[f"action_{a}_q_value"]
                self.visit_table[(encoded_state, a)] = row[f"action_{a}_visit_count"]
        print(f"Q-Table loaded from {f'{file_path}' if file_path else 'DataFrame'}.")
        self.print_q_table_info()

    def get_action_probabilities(self, state: np.ndarray, strategy: str = "greedy",
                                 temperature: float = 1.0) -> np.ndarray:
        """
        Calculate action probabilities based on the specified strategy.

        :param state: The current state.
        :param strategy: Strategy to calculate probabilities ("greedy" or "softmax").
        :param temperature: Temperature parameter for softmax.
        :return: An array of action probabilities.
        """
        encoded_state = self.state_discretizer.encode_indices([*self.state_discretizer.discretize(state)[1]])

        # Retrieve Q-values for all actions
        q_values = np.array([self.q_table.get((encoded_state, a), 0.0) for a in self.all_actions_encoded])

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

    def update(self, state: np.ndarray, action: List[int], reward: float, next_state: np.ndarray, done: bool,
               alpha: float = 0.1, gamma: float = 0.99) -> None:
        """
        Update the Q-Table using the Q-learning update rule.

        :param state: The current state.
        :param action: The action taken (multi-dimensional).
        :param reward: The received reward.
        :param next_state: The next state.
        :param done: Whether the episode is finished.
        :param alpha: Learning rate.
        :param gamma: Discount factor.
        """
        state_key = tuple(state)
        next_state_key = tuple(next_state)

        if done:
            td_target = reward  # No future reward if the episode is done
        else:
            # Compute the best next action's Q-value
            best_next_action_value = max(
                [self.q_table.get((next_state_key, tuple(a)), 0.0) for a in self.all_actions_encoded],
                default=0.0
            )
            td_target = reward + gamma * best_next_action_value

        # Update Q-value for the current state-action pair
        td_error = td_target - self.q_table[(state_key, tuple(action))]
        self.q_table[(state_key, tuple(action))] += alpha * td_error


class TransitionTable:
    def __init__(self, state_discretizer: Discretizer, action_discretizer: Discretizer,):
        self.state_discretizer = state_discretizer
        self.action_discretizer = action_discretizer
        self.transition_table: Dict[int, Dict[int, Dict[str, int]]] = {}  # {state: {action: {next_state: count}}}
        self.neighbour_dict = defaultdict(lambda: set())
        self.forward_dict = defaultdict(lambda: set())
        self.inverse_dict = defaultdict(lambda: set())


if __name__ == "__main__":
    # Define test parameters
    ranges = [(0, 10), (5, 15), (-10, 10), (-np.inf, np.inf)]
    num_buckets = [5, 0, 3, 4]
    normal_params = [None, None, (0, 5), (0, 1)]  # Use normal distribution for the last two dimensions

    # Create Discretizer instance
    discretizer = Discretizer(ranges, num_buckets, normal_params)

    # Print bucket information
    print("Bucket Information:")
    discretizer.print_buckets()

    # Test vectors
    test_vectors = [
        [2, 7, -8, -2],
        [10, 12, 0, 0],
        [5, 15, 10, 2],
        [-1, 5, -5, 1],
        [0, 10, 5, -3]
    ]

    # Apply discretization and print results
    for vector in test_vectors:
        midpoints, indices = discretizer.discretize(vector)
        print(f"\nInput vector: {vector}")
        print(f"Midpoints: {midpoints}")
        print(f"Bucket indices: {indices}")

    # Define test parameters
    ranges = [(0, 4), (5, 6)]  # Reduced range for easier testing
    num_buckets = [2, 0]
    normal_params = [None, None]  # Uniform distribution

    # Create Discretizer instance
    discretizer = Discretizer(ranges, num_buckets, normal_params)

    # Print bucket information
    print("Bucket Information:")
    discretizer.print_buckets()

    # Test all possible combinations
    midpoints_product, indices_product = discretizer.list_all_possible_combinations()
    print("\nAll possible combinations of bucket midpoints:")
    for combo in midpoints_product:
        print(combo)

    print("\nAll possible combinations of bucket indices:")
    for combo in indices_product:
        print(combo)

    # Test vectors
    test_vectors = [
        [1, 5.2],
        [3.5, 5.8],
    ]

    for vector in test_vectors:
        midpoints, indices = discretizer.discretize(vector)
        print(f"\nInput vector: {vector}")
        print(f"Midpoints: {midpoints}")
        print(f"Bucket indices: {indices}")

    # Define test parameters
    ranges = [(0, 4), (-1, 1), (5, 6)]  # Added an integer range example
    num_buckets = [2, 0, 0]  # Integer range for second dimension, no discretization for the third
    normal_params = [None, None, None]  # Uniform distribution

    # Create Discretizer instance
    discretizer = Discretizer(ranges, num_buckets, normal_params)

    # Print bucket information
    print("Bucket Information:")
    discretizer.print_buckets()

    # Test all possible combinations
    midpoints_product, indices_product = discretizer.list_all_possible_combinations()
    print("\nAll possible combinations of bucket midpoints:")
    for combo in midpoints_product:
        print(combo)

    print("\nAll possible combinations of bucket indices:")
    for combo in indices_product:
        print(combo)

    # Count possible combinations
    total_combinations = discretizer.count_possible_combinations()
    print(f"\nTotal number of possible combinations: {total_combinations}")

    # Test encoding and decoding
    test_indices = [1, 2, 0]
    encoded = discretizer.encode_indices(test_indices)
    decoded = discretizer.decode_indices(encoded)
    midpoints = discretizer.indices_to_midpoints(test_indices)
    print(f"\nTest indices: {test_indices}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    print(f"Midpoints: {midpoints}")

    # Test vectors
    test_vectors = [
        [1, 0, 5.2],
        [3.5, -1, 6.0],
    ]

    for vector in test_vectors:
        midpoints, indices = discretizer.discretize(vector)
        print(f"\nInput vector: {vector}")
        print(f"Midpoints: {midpoints}")
        print(f"Bucket indices: {indices}")
