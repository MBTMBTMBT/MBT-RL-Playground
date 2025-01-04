import heapq
import math
import random
from collections import defaultdict
from itertools import product
import gymnasium as gym
import networkx as nx
import numpy as np
import pandas as pd
import scipy.stats
from typing import List, Tuple, Optional, Dict

from gymnasium import spaces
from networkx.classes import DiGraph
from pandas import DataFrame
import tqdm
from pyvis.network import Network

import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap


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
        self.input_num_buckets: List[int] = num_buckets
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
        if (
                isinstance(vector, int)
                or isinstance(vector, np.int64)
                or isinstance(vector, float)
                or isinstance(vector, np.float32)
        ):
            vector = [vector]
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
        remaining_code = code

        for buckets in reversed(self.num_buckets):
            if buckets == -1:
                indices.append(-1)  # No discretization
            else:
                indices.append(remaining_code % buckets)  # Extract the current dimension index
                remaining_code //= buckets  # Update the remaining code

        # Reverse the indices to match the original order
        return indices[::-1]

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

    def get_gym_space(self) -> spaces.Space:
        """
        Create a Gymnasium space representing the discretized ranges.

        :return: A `spaces.MultiDiscrete`, `spaces.Discrete`, or `spaces.Box` space depending on the discretization.
        """
        if all(buckets == 0 for buckets in self.input_num_buckets):
            if len(self.num_buckets) == 1:
                # Use Discrete if only one dimension
                return spaces.Discrete(self.num_buckets[0])
            else:
                # Use MultiDiscrete for multiple dimensions
                return spaces.MultiDiscrete(self.num_buckets)
        else:
            # Use Box if some dimensions are not discretized
            low = []
            high = []
            for (min_val, max_val), buckets in zip(self.ranges, self.num_buckets):
                if buckets == -1:
                    low.append(min_val)
                    high.append(max_val)
                elif buckets == 0:
                    low.append(np.ceil(min_val))
                    high.append(np.floor(max_val))
                else:
                    low.append(0)
                    high.append(buckets - 1)
            return spaces.Box(low=np.array(low, dtype=np.float32), high=np.array(high, dtype=np.float32), dtype=np.float32)


class TabularQAgent:
    def __init__(
            self,
            env: gym.Env,
            state_discretizer: Discretizer,
            action_discretizer: Discretizer,
            n_steps: int = 2048,
            lr: float = 0.1,
            gamma: float = 0.99,
            print_info: bool = True,
    ):
        self.state_discretizer = state_discretizer
        self.action_discretizer = action_discretizer
        self.lr = lr
        self.gamma = gamma
        self.env = env
        self.n_steps = n_steps
        self.q_table = defaultdict(lambda: 0.0)  # Flattened Q-Table with state-action tuple keys
        self.visit_table = defaultdict(lambda: 0)  # Uses the same keys of the Q-Table to do visit count.
        self.all_actions_encoded = sorted([
            self.action_discretizer.encode_indices([*indices])
            for indices in self.action_discretizer.list_all_possible_combinations()[1]
        ])
        if print_info:
            self.print_q_table_info()

    def set_env(self, env: gym.Env):
        self.env = env

    def reset_q_table(self) -> None:
        self.q_table = defaultdict(lambda: 0.0)  # Flattened Q-Table with state-action tuple keys

    def reset_visit_table(self) -> None:
        self.visit_table = defaultdict(lambda: 0)  # Uses the same keys of the Q-Table to do visit count.

    def clone(self) -> 'TabularQAgent':
        """
        Create a deep copy of the Q-Table agent.

        :return: A new QTableAgent instance with the same Q-Table.
        """
        new_agent = TabularQAgent(
            self.env,
            self.state_discretizer,
            self.action_discretizer,
            lr=self.lr,
            gamma=self.gamma,
            n_steps=self.n_steps,
            print_info=False,
        )
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
        for (encoded_state, encoded_action), q_value in tuple(self.q_table.items()):
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

    def get_action_probabilities(self, state: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """
        Calculate action probabilities based on the specified strategy.

        :param state: The current state.
        :param temperature: Temperature parameter for softmax.
        :return: An array of action probabilities.
        """
        encoded_state = self.state_discretizer.encode_indices([*self.state_discretizer.discretize(state)[1]])

        # Retrieve Q-values for all actions
        q_values = np.array([self.q_table[(encoded_state, a)] for a in self.all_actions_encoded])
        visitation = sum([self.visit_table[(encoded_state, a)] for a in self.all_actions_encoded])

        if np.all(q_values == 0):  # Handle all-zero Q-values
            probabilities = np.ones_like(q_values, dtype=float) / len(q_values)
        else:
            # Subtract the maximum value for numerical stability
            q_values_stable = q_values - np.max(q_values)
            exp_values = np.exp(q_values_stable / temperature)
            probabilities = exp_values / (np.sum(exp_values) + 1e-10)  # Add small value to prevent division by zero

        if visitation == 0:
            probabilities = np.ones_like(q_values, dtype=float) / len(q_values)

        return probabilities

    def choose_action(self, state: np.ndarray, temperature: float = 1.0, greedy: bool = False) -> np.ndarray:
        action_probabilities = self.get_action_probabilities(state, temperature)
        if greedy:
            action_encoded = np.argmax(action_probabilities)
        else:
            action_encoded = np.random.choice(self.all_actions_encoded, p=action_probabilities)
        action = np.array(self.action_discretizer.indices_to_midpoints(self.action_discretizer.decode_indices(action_encoded)))
        if isinstance(self.env.action_space, spaces.Discrete):
            action = action.squeeze().item()
        return action

    def update(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool,
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
        state_encoded = self.state_discretizer.encode_indices([*self.state_discretizer.discretize(state)[1]])
        next_state_encoded = self.state_discretizer.encode_indices([*self.state_discretizer.discretize(next_state)[1]])
        action_encoded = self.action_discretizer.encode_indices([*self.action_discretizer.discretize(action)[1]])

        reward /= 10.

        if done:
            td_target = reward  # No future reward if the episode is done
        else:
            # Compute the best next action's Q-value
            best_next_action_value = max(
                [self.q_table.get((next_state_encoded, a), 0.0) for a in self.all_actions_encoded],
                default=0.0
            )
            td_target = reward + gamma * best_next_action_value

        # Update Q-value for the current state-action pair
        td_error = td_target - self.q_table[(state_encoded, action_encoded)]
        self.q_table[(state_encoded, action_encoded)] += alpha * td_error
        self.visit_table[(state_encoded, action_encoded)] += 1

    def learn(
            self,
            total_timesteps: int,
            reset_num_timesteps: bool = True,
            progress_bar: bool = False,
            temperature: float = 1.0,
    ):
        state, info = self.env.reset()
        episode_step_count = 0
        num_episodes = 0
        num_truncated = 0
        num_terminated = 0
        sum_episode_rewards = 0

        # Initialize the progress bar
        pbar = None
        if progress_bar:
            pbar = tqdm.tqdm(total=total_timesteps, desc="Inner Training", unit="step")

        for timestep in range(total_timesteps):
            if reset_num_timesteps:
                if episode_step_count >= self.n_steps:
                    episode_step_count = 0
                    num_episodes += 1
                    state, info = self.env.reset()
            action = self.choose_action(state, temperature=temperature)
            next_state, reward, terminated, truncated, info = self.env.step(action)
            self.update(state, action, reward, next_state, terminated, alpha=self.lr, gamma=self.gamma)
            state = next_state
            episode_step_count += 1
            sum_episode_rewards += reward
            if terminated or truncated:
                episode_step_count = 0
                num_episodes += 1
                num_terminated += 1 if terminated else 0
                num_truncated += 1 if truncated else 0
                state, info = self.env.reset()

            if progress_bar:
                # Update the progress bar
                pbar.set_postfix({
                    "Episodes": num_episodes,
                    "Terminated": num_terminated,
                    "Truncated": num_truncated,
                    "Reward (last)": reward,
                    "Avg Episode Reward": sum_episode_rewards / num_episodes if num_episodes > 0 else 0.0,
                })
                pbar.update(1)
        if progress_bar:
            pbar.close()


class TransitionTable:
    def __init__(self, state_discretizer: Discretizer, action_discretizer: Discretizer, reward_resolution: int = -1):
        self.state_discretizer = state_discretizer
        self.action_discretizer = action_discretizer
        self.transition_table: Dict[int, Dict[int, Dict[int, Dict[float, int]]]] \
            = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0))))  # {state: {action: {next_state: {reward: count}}}
        self.neighbour_dict = defaultdict(lambda: set())
        self.forward_dict = defaultdict(lambda: defaultdict(lambda: set()))
        self.inverse_dict = defaultdict(lambda: defaultdict(lambda: set()))

        # They will not be saved!
        self.state_count = defaultdict(lambda: 0)
        self.state_action_count = defaultdict(lambda: defaultdict(lambda: 0))
        self.transition_prob_table: Dict[int, Dict[int, Dict[int, float]]] \
            = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))  # {state: {action: {next_state: rate}}
        self.done_set = set()
        self.start_set = set()
        self.reward_set_dict = defaultdict(lambda: set())
        self.reward_resolution = reward_resolution

        self.mdp_graph: DiGraph = None

    def print_transition_table_info(self):
        print("Transition Table Information:")
        print(f"Total num transition pairs: {len(self.forward_dict)}.")
        print(f"Collected initial states: {len(self.start_set)}.")
        print(f"Collected termination states: {len(self.done_set)}.")
        print(f"Collected rewards:")
        total_reward_count = 0
        for reward, reward_set in self.reward_set_dict.items():
            total_reward_count += len(reward_set)
        for reward, reward_set in sorted(self.reward_set_dict.items(), key=lambda x: x[0]):
            print(f"{reward}: {len(reward_set)} - {len(reward_set) / total_reward_count * 100:.2f}%")

    def update(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool):
        encoded_state = self.state_discretizer.encode_indices(list(self.state_discretizer.discretize(state)[1]))
        encoded_next_state = self.state_discretizer.encode_indices(list(self.state_discretizer.discretize(next_state)[1]))
        encoded_action = self.action_discretizer.encode_indices(list(self.action_discretizer.discretize(action)[1]))

        if done:
            self.done_set.add(encoded_next_state)

        self.transition_table[encoded_state][encoded_action][encoded_next_state][round(reward, 1)] += 1
        self.neighbour_dict[encoded_state].add(encoded_action)
        self.neighbour_dict[encoded_next_state].add(encoded_action)
        self.forward_dict[encoded_state][encoded_next_state].add(encoded_action)
        self.inverse_dict[encoded_next_state][encoded_state].add(encoded_action)
        self.state_count[encoded_state] += 1
        self.state_action_count[encoded_state][encoded_action] += 1

        transition_state_avg_reward_and_prob \
            = self.get_transition_state_avg_reward_and_prob(encoded_state, encoded_action)
        for encoded_next_state, (avg_reward, prob) in transition_state_avg_reward_and_prob.items():
            self.transition_prob_table[encoded_state][encoded_action][encoded_next_state] = prob

        if self.reward_resolution > 0:
            self.reward_set_dict[round(reward / self.reward_resolution) * self.reward_resolution].add(encoded_next_state)
        elif self.reward_resolution < 0:
            self.reward_set_dict[round(reward, abs(self.reward_resolution))].add(encoded_next_state)
        else:
            self.reward_set_dict[reward].add(encoded_next_state)

    def save_transition_table(self, file_path: str = None) -> pd.DataFrame:
        transition_table_data = []
        for encoded_state in self.transition_table.keys():
            for encoded_action in self.transition_table[encoded_state].keys():
                for encoded_next_state in self.transition_table[encoded_state][encoded_action].keys():
                    for reward in self.transition_table[encoded_state][encoded_action][encoded_next_state].keys():
                        count = self.transition_table[encoded_state][encoded_action][encoded_next_state][reward]
                        row = {
                            "state": encoded_state,
                            "action": encoded_action,
                            "next_state": encoded_next_state,
                            "reward": reward,
                            "count": count,
                        }
                        transition_table_data.append(row)
        transition_table_df = pd.DataFrame(transition_table_data)

        if file_path:
            transition_table_df.to_csv(file_path, index=False)
            print(f"Transition Table saved to {file_path}.")
        return transition_table_df

    def make_mdp_graph(self, use_encoded_states=False):
        # Create a directed graph
        G = nx.DiGraph()

        # Traverse the transition table and construct the graph
        for encoded_state in self.transition_table.keys():
            for encoded_action in self.transition_table[encoded_state].keys():
                total_count = 0
                for encoded_next_state in self.transition_table[encoded_state][encoded_action].keys():
                    for reward in self.transition_table[encoded_state][encoded_action][encoded_next_state].keys():
                        total_count += self.transition_table[encoded_state][encoded_action][encoded_next_state][reward]
                for encoded_next_state in self.transition_table[encoded_state][encoded_action].keys():
                    for reward in self.transition_table[encoded_state][encoded_action][encoded_next_state].keys():
                        count = self.transition_table[encoded_state][encoded_action][encoded_next_state][reward]
                        # Add edges and attributes
                        state_str = str(self.state_discretizer.indices_to_midpoints(
                            self.state_discretizer.decode_indices(encoded_state)))
                        next_state_str = str(self.state_discretizer.indices_to_midpoints(
                            self.state_discretizer.decode_indices(encoded_next_state)))
                        if use_encoded_states:
                            state = int(encoded_state)
                            next_state = int(encoded_next_state)
                        else:
                            state = state_str
                            next_state = next_state_str
                        G.add_edge(
                            state,
                            next_state,
                            label=f"{encoded_action}\nR={reward}\nCount={count}",
                            count=count,
                            prob=count / total_count,
                        )
                        G.nodes[state]['count'] = self.state_count[encoded_state]
                        G.nodes[next_state]['count'] = self.state_count[encoded_next_state]
                        G.nodes[state]['code'] = int(encoded_state)
                        G.nodes[next_state]['code'] = int(encoded_next_state)
                        G.nodes[state]['str'] = state_str
                        G.nodes[next_state]['str'] = next_state_str

        self.mdp_graph = G
        return G

    def save_mdp_graph(self, output_file='mdp_visualization.html'):
        # Create a directed graph
        G = self.make_mdp_graph()

        # Use Pyvis for visualization
        net = Network(height='1000px', width='100%', directed=True)
        net.from_nx(G)

        # Normalize counts for coloring
        all_node_counts = [data['count'] for _, data in G.nodes(data=True)]
        all_edge_counts = [data['count'] for _, _, data in G.edges(data=True)]

        node_norm = mcolors.Normalize(vmin=min(all_node_counts), vmax=max(all_node_counts))
        edge_norm = mcolors.Normalize(vmin=min(all_edge_counts), vmax=max(all_edge_counts))

        cmap = LinearSegmentedColormap.from_list("custom_blues", ['#ADD8E6', '#00008B'])  # LightBlue to DarkBlue

        # Set edge colors based on counts
        for edge in net.edges:
            edge_count = G.edges[edge['from'], edge['to']]['count']
            edge_color = mcolors.to_hex(cmap(edge_norm(edge_count)))
            edge['color'] = edge_color

        # Set node colors based on counts
        for node in G.nodes():
            node_count = G.nodes[node]['count']
            node_color = mcolors.to_hex(cmap(node_norm(node_count)))
            net.get_node(node)['color'] = node_color
            net.get_node(node)['title'] = f"State: {node}, Count: {node_count}"

        # # Disable physics for faster rendering
        # net.toggle_physics(False)

        # Save and display
        net.write_html(output_file, notebook=False, open_browser=False)
        print(f"Saved tranisiton graph at {output_file}.")

    def load_transition_table(self, file_path: str = None, transition_table_df: pd.DataFrame = None):
        if file_path:
            transition_table_df = pd.read_csv(file_path)
        elif transition_table_df is None:
            raise ValueError("Either file_path or df must be provided.")

        for _, row in transition_table_df.iterrows():
            encoded_state = row["state"]
            encoded_action = row["action"]
            encoded_next_state = row["next_state"]
            reward = row["reward"]
            count = row["count"]
            self.transition_table[encoded_state][encoded_action][encoded_next_state][reward] = count
            self.neighbour_dict[encoded_state].add(encoded_action)
            self.neighbour_dict[encoded_next_state].add(encoded_action)
            self.forward_dict[encoded_state][encoded_next_state].add(encoded_action)
            self.inverse_dict[encoded_next_state][encoded_state].add(encoded_action)
        print(f"Transition Table loaded from {f'{file_path}' if file_path else 'DataFrame'}.")

    def get_transition_state_avg_reward_and_prob(self, encoded_state: int, encoded_action: int) -> Dict[int, Tuple[float, float]]:
        # Transition to state probs: from given state, with given action, probs of getting into next states
        # Avg Reward: from given state, with given action, ending up in certain state, the average reward it gets
        transition_state_reward_and_prob = {}
        _transition_state_reward_and_prob = {}
        encoded_next_states = []
        encoded_next_state_counts = []
        avg_rewards = []
        for encoded_next_state in self.transition_table[encoded_state][encoded_action].keys():
            encoded_next_state_count = 0
            _transition_state_reward_and_prob[encoded_next_state] = {}
            rewards = []
            reward_counts = []
            for reward in self.transition_table[encoded_state][encoded_action][encoded_next_state].keys():
                reward_count = self.transition_table[encoded_state][encoded_action][encoded_next_state][reward]
                _transition_state_reward_and_prob[encoded_next_state][reward] = reward_count
                encoded_next_state_count += reward_count
                rewards.append(reward)
                reward_counts.append(reward_count)
            avg_rewards.append(np.average(rewards, weights=reward_counts))
            encoded_next_states.append(encoded_next_state)
            encoded_next_state_counts.append(encoded_next_state_count)
        encoded_next_state_probs = np.array(encoded_next_state_counts) / np.sum(encoded_next_state_counts)
        for encoded_next_state, avg_reward, prob in zip(encoded_next_states, avg_rewards, encoded_next_state_probs):
            transition_state_reward_and_prob[encoded_next_state] = (float(avg_reward), float(prob))
        return transition_state_reward_and_prob

    def get_neighbours(self, encoded_state: int) -> set[int]:
        return self.neighbour_dict[encoded_state]

    def get_forward_neighbours(self, encoded_state: int) -> Dict[int, set[int]]:
        return self.forward_dict[encoded_state]

    def get_inverse_neighbours(self, encoded_state: int) -> Dict[int, set[int]]:
        return self.inverse_dict[encoded_state]

    def get_done_set(self) -> set[int]:
        return self.done_set

    def add_start_state(self, state):
        if isinstance(state, int) or isinstance(state, float):
            state = [state]
        encoded_state = self.state_discretizer.encode_indices(list(self.state_discretizer.discretize(state)[1]))
        self.start_set.add(encoded_state)

    def get_start_set(self) -> set[int]:
        return self.start_set


class TransitionalTableEnv(TransitionTable, gym.Env):
    INIT_STRATEGIES = ["real_start_states", "random"]
    def __init__(
            self,
            state_discretizer: Discretizer,
            action_discretizer: Discretizer,
            max_steps: int = 500,
            reward_resolution: int = -1,
            init_strategy_distribution: Tuple[float] = (0.5, 0.5),
            unknown_reward: float = None
    ):
        TransitionTable.__init__(self, state_discretizer, action_discretizer, reward_resolution=reward_resolution)
        gym.Env.__init__(self)

        assert len(init_strategy_distribution) == len(TransitionalTableEnv.INIT_STRATEGIES), "init_strategy_distribution must have the same length as INIT_STRATEGIES."
        self.init_strategy_distribution = init_strategy_distribution
        self.init_strategy = None

        # State space
        self.state_discretizer = state_discretizer
        self.observation_space = state_discretizer.get_gym_space()

        # Action space
        self.action_discretizer = action_discretizer
        self.action_space = action_discretizer.get_gym_space()

        self.max_steps = max_steps
        self.unknown_reward = unknown_reward

        self.step_count = 0
        self.current_state = None
        self.strategy_counts = {s: 1 for s in TransitionalTableEnv.INIT_STRATEGIES}
        self.strategy_step_counts = {s: 1 for s in TransitionalTableEnv.INIT_STRATEGIES}

    def reset(self, seed=None, options=None, init_state: np.ndarray = None, reset_all: bool = False):
        if reset_all:
            self.step_count = 0
            self.current_state = None
            self.strategy_counts = {s: 1 for s in TransitionalTableEnv.INIT_STRATEGIES}
            self.strategy_step_counts = {s: 1 for s in TransitionalTableEnv.INIT_STRATEGIES}

        init_state_encode = None if init_state is None else self.state_discretizer.encode_indices(
            list(self.state_discretizer.discretize(init_state)[1])
        )
        strategy_selection_dict = {}
        for i, s in enumerate(TransitionalTableEnv.INIT_STRATEGIES):
            if self.init_strategy_distribution[i] != 0:
                strategy_selection_dict[s] = self.strategy_step_counts[s] / self.init_strategy_distribution[i]
            else:
                strategy_selection_dict[s] = np.inf
        self.init_strategy = min(strategy_selection_dict, key=strategy_selection_dict.get)
        self.step_count = 0
        if init_state_encode is None or init_state_encode in self.done_set:
            if init_state_encode in self.done_set:
                # print("Warning: Starting from a done state, reset to a random state.")
                self.init_strategy = "random"
            if self.init_strategy == "random":
                init_state_encode = random.choice(tuple(self.forward_dict.keys()))
            elif self.init_strategy == "real_start_states":
                init_state_encode = random.choice(tuple(self.start_set))
            else:
                raise ValueError(f"Init strategy not supported: {self.init_strategy}.")
        self.current_state = init_state_encode
        current_state = self.state_discretizer.indices_to_midpoints(
            self.state_discretizer.decode_indices(self.current_state)
        )
        return current_state, {}

    def step(self, action: int or np.ndarray,):
        if self.unknown_reward is None:
            r_sum = 0.0
            total_rewards = 0
            for r in self.reward_set_dict.keys():
                r_sum += r * len(self.reward_set_dict[r])
                total_rewards += len(self.reward_set_dict[r])
            unknown_reward = r_sum / total_rewards
        else:
            unknown_reward = self.unknown_reward
        encoded_state = self.current_state
        if isinstance(action, int):
            encoded_action = action
        else:
            encoded_action = self.action_discretizer.encode_indices(list(self.action_discretizer.discretize(action)[1]))
        transition_state_avg_reward_and_prob \
            = self.get_transition_state_avg_reward_and_prob(encoded_state, encoded_action)
        if len(transition_state_avg_reward_and_prob) == 0:
            state = self.state_discretizer.indices_to_midpoints(
                self.state_discretizer.decode_indices(encoded_state)
            )
            return state, unknown_reward, True, False, {"current_step": self.step_count}
        encoded_next_state = random.choices(
            tuple(transition_state_avg_reward_and_prob.keys()),
            weights=[v[1] for v in transition_state_avg_reward_and_prob.values()],
            k=1,
        )[0]
        reward = transition_state_avg_reward_and_prob[encoded_next_state][0]
        self.step_count += 1

        terminated = encoded_next_state in self.done_set
        truncated = self.step_count >= self.max_steps
        self.current_state = encoded_next_state

        self.strategy_step_counts[self.init_strategy] += 1

        info = {"current_step": self.step_count}

        next_state = self.state_discretizer.indices_to_midpoints(
            self.state_discretizer.decode_indices(encoded_next_state)
        )
        return next_state, reward, terminated, truncated, info

    def strategy_step_plus_1(self):
        self.strategy_step_counts[self.init_strategy] += 1


class LandmarksTransitionalTableEnv(TransitionalTableEnv):
    INIT_STRATEGIES = TransitionalTableEnv.INIT_STRATEGIES + ["landmarks"]
    def __init__(
            self,
            state_discretizer: Discretizer,
            action_discretizer: Discretizer,
            num_targets: int,
            min_cut_max_flow_search_space: int,
            q_cut_space: int,
            weighted_search: bool = True,
            init_state_reward_prob_below_threshold: float = 0.2,
            quality_value_threshold: float = 1.0,
            take_done_states_as_targets: bool = False,
            max_steps: int = 500,
            reward_resolution: int = -1,
            init_strategy_distribution: Tuple[float] = (0.33, 0.33, 0.33),
            unknown_reward: float = None
    ):
        TransitionalTableEnv.__init__(
            self, state_discretizer, action_discretizer, max_steps, reward_resolution, unknown_reward=unknown_reward,
        )
        assert len(init_strategy_distribution) == len(LandmarksTransitionalTableEnv.INIT_STRATEGIES), \
            "init_strategy_distribution must have the same length as INIT_STRATEGIES."
        self.init_strategy_distribution = init_strategy_distribution

        self.num_targets = num_targets
        self.min_cut_max_flow_search_space = min_cut_max_flow_search_space
        self.q_cut_space = q_cut_space
        self.weighted_search = weighted_search
        self.init_state_reward_prob_below_threshold = init_state_reward_prob_below_threshold
        self.quality_value_threshold = quality_value_threshold
        self.take_done_states_as_targets = take_done_states_as_targets

        self.landmark_states, self.landmark_start_states, self.targets = None, None, None
        self.landmarks_inited = False

    def find_nearest_nodes_and_subgraph(self, start_node, n, weighted=True, direction='both') -> Tuple[List[Tuple[int, float]], DiGraph]:
        """
        Find the nearest n nodes from the starting node, searching in the specified direction.

        :param start_node: Starting node for the search
        :param n: Number of nearest nodes to find
        :param weighted: Whether to consider weights (True: use 'prob' as weights; False: unweighted search)
        :param direction: Direction of search ('forward', 'backward', or 'both')
        :return: List of the nearest n nodes (sorted by distance) and the subgraph containing these nodes
        """
        G = self.mdp_graph if self.mdp_graph else self.make_mdp_graph(use_encoded_states=True)

        visited = set()  # Set to track visited nodes
        heap = []  # Min-heap to prioritize nodes by accumulated probability (log space)
        result = []  # List to store the nearest nodes, sorted by log probability
        accumulated_prob = {start_node: 0.0}  # Cumulative log-probabilities for each node

        # Initialize the heap based on the specified direction
        if direction in ['forward', 'both']:
            heapq.heappush(heap, (0.0, start_node, 'forward'))  # Forward direction
        if direction in ['backward', 'both']:
            heapq.heappush(heap, (0.0, start_node, 'backward'))  # Backward direction

        while heap and len(result) < n:
            # Pop the node with the smallest log-probability from the heap
            log_prob, current_node, search_direction = heapq.heappop(heap)

            # Skip if the node has already been visited
            if current_node in visited:
                continue

            # Mark the current node as visited
            visited.add(current_node)
            result.append((current_node, log_prob))  # Add the node to the result list with its log-probability

            # Get neighbors based on the current search direction
            if search_direction == 'forward':
                neighbors = G.successors(current_node)
            else:  # Backward direction
                neighbors = G.predecessors(current_node)

            for neighbor in neighbors:
                if neighbor not in visited:
                    if weighted:
                        # Weighted mode: Use -log(prob) for cumulative probability
                        if search_direction == 'forward':
                            prob = max(G.edges[current_node, neighbor].get('prob', 0.0), 1e-8)
                        else:
                            prob = max(G.edges[neighbor, current_node].get('prob', 0.0), 1e-8)
                        new_log_prob = log_prob - math.log(prob)  # Accumulate in log space
                    else:
                        # Unweighted mode: Fixed step distance
                        new_log_prob = log_prob + 1

                    # Update the heap and cumulative probabilities if this path is better
                    if neighbor not in accumulated_prob or new_log_prob < accumulated_prob[neighbor]:
                        accumulated_prob[neighbor] = new_log_prob
                        heapq.heappush(heap, (new_log_prob, neighbor, search_direction))

        # Sort result by distance (log probability)
        result = sorted(result, key=lambda x: x[1])  # Sort by log probability

        # Create a subgraph based on the nodes in the result
        nodes_in_result = [node for node, _ in result]
        subgraph = self.mdp_graph.subgraph(nodes_in_result).copy()

        # Return the list of nearest nodes and the subgraph
        return result, subgraph

    @staticmethod
    def find_min_cut_max_flow(G, source, target, invert_weights=False):
        """
        Find the Minimum Cut - Max Flow edges and the partitioned sets of nodes,
        and calculate the ratio-cut quality factor.

        Parameters:
            G (nx.DiGraph): The directed graph.
            source (str): The source node.
            target (str): The target node.
            invert_weights (bool): Whether to use -log(prob) as the weight (True for maximizing prob).

        Returns:
            cut_value (float): The total weight of the minimum cut.
            reachable (set): Nodes in the source partition.
            non_reachable (set): Nodes in the sink partition.
            edges_in_cut (list): The edges included in the minimum cut.
            quality_factor (float): The ratio-cut quality factor.
        """
        # Create a copy of the graph to modify edge weights
        H = G.copy()

        # Update edge weights
        for u, v, data in H.edges(data=True):
            prob = max(data.get('prob', 1e-8), 1e-8)  # Ensure probabilities are > 0
            if invert_weights:
                # Use -log(prob) for maximizing overall probabilities
                weight = -math.log(prob)
            else:
                weight = prob
            H[u][v]['capacity'] = weight  # Set the capacity for the edge

        # Compute the minimum cut
        cut_value, partition = nx.minimum_cut(H, source, target, capacity='capacity')
        reachable, non_reachable = partition

        # Find the edges in the cut
        edges_in_cut = []
        for u in reachable:
            for v in G.successors(u):
                if v in non_reachable:
                    edges_in_cut.append((u, v))

        # Calculate the ratio-cut quality factor
        size_reachable = len(reachable)
        size_non_reachable = len(non_reachable)
        num_cut_edges = len(edges_in_cut)

        if num_cut_edges > 0:  # Avoid division by zero
            quality_factor = (size_reachable * size_non_reachable) / num_cut_edges
        else:
            quality_factor = float('inf')  # Perfect separation

        return cut_value, reachable, non_reachable, edges_in_cut, quality_factor

    def get_landmark_states(self,):
        landmark_states = set()
        landmark_start_states = set()

        targets = set()
        total_reward_count = 0
        for reward, reward_set in self.reward_set_dict.items():
            total_reward_count += len(reward_set)
        for reward in self.reward_set_dict.keys():
            if len(self.reward_set_dict[reward]) / total_reward_count < self.init_state_reward_prob_below_threshold:
                for state in self.reward_set_dict[reward]:
                    targets.add(state)

        if self.take_done_states_as_targets:
            for state in self.done_set:
                targets.add(state)

        if len(targets) == 0:
            return [], [], []

        selected_targets = targets if len(targets) <= self.num_targets else random.sample(targets, self.num_targets)

        for target in selected_targets:
            nearest_nodes, subgraph = self.find_nearest_nodes_and_subgraph(
                target, self.min_cut_max_flow_search_space, weighted=self.weighted_search, direction='backward'
            )
            start_node = random.choice(nearest_nodes[-len(nearest_nodes)//10:])[0]
            cut_value, reachable, non_reachable, edges_in_cut, quality_factor = self.find_min_cut_max_flow(
                subgraph, start_node, target, invert_weights=False,
            )
            if quality_factor > self.quality_value_threshold:
                for edge in edges_in_cut:
                    for node in edge:
                        landmark_states.add(node)

        for node in landmark_states:
            q_cut_nodes, q_cut_subgraph = self.find_nearest_nodes_and_subgraph(
                node, self.q_cut_space, weighted=False, direction='both'
            )
            for n in q_cut_nodes:
                landmark_start_states.add(n[0])

        return landmark_states, landmark_start_states, targets

    def reset(
            self,
            seed=None,
            options=None,
            init_state: np.ndarray = None,
            reset_all: bool = False,
            do_print: bool = False,
    ):
        if reset_all:
            self.step_count = 0
            self.current_state = None
            self.strategy_counts = {s: 1 for s in LandmarksTransitionalTableEnv.INIT_STRATEGIES}
            self.strategy_step_counts = {s: 1 for s in LandmarksTransitionalTableEnv.INIT_STRATEGIES}

            self.landmark_states, self.landmark_start_states, self.targets = None, None, None
            self.landmarks_inited = False

        init_state_encode = None if init_state is None else self.state_discretizer.encode_indices(
            list(self.state_discretizer.discretize(init_state)[1])
        )
        strategy_selection_dict = {}
        for i, s in enumerate(LandmarksTransitionalTableEnv.INIT_STRATEGIES):
            if self.init_strategy_distribution[i] != 0:
                strategy_selection_dict[s] = self.strategy_step_counts[s] / self.init_strategy_distribution[i]
            else:
                strategy_selection_dict[s] = np.inf
        self.init_strategy = min(strategy_selection_dict, key=strategy_selection_dict.get)
        self.step_count = 0

        if init_state_encode is None or init_state_encode in self.done_set:
            if init_state_encode in self.done_set:
                # print("Warning: Starting from a done state, reset to a random state.")
                self.init_strategy = "random"
            if self.init_strategy == "random":
                init_state_encode = random.choice(tuple(self.forward_dict.keys()))
            elif self.init_strategy == "real_start_states":
                init_state_encode = random.choice(tuple(self.start_set))
            elif self.init_strategy == "landmarks":
                if (not self.landmarks_inited) or self.landmark_states is None or self.landmark_start_states is None or self.targets is None:
                    self.make_mdp_graph(use_encoded_states=True)
                    self.landmark_states, self.landmark_start_states, self.targets = self.get_landmark_states()
                    self.landmarks_inited = True
                    if do_print:
                        print(f"Initialized: {len(self.landmark_states)} landmark states; {len(self.landmark_start_states)} start states.")
                        if self.take_done_states_as_targets:
                            print("Done states are also used as targets for landmark generation.")
                if len(self.landmark_states) == 0:
                    init_state_encode = random.choice(tuple(self.forward_dict.keys()))
                    self.init_strategy = "random"
                else:
                    init_state_encode = random.choice(tuple(self.landmark_start_states))
            else:
                raise ValueError(f"Init strategy not supported: {self.init_strategy}.")
        self.current_state = init_state_encode
        current_state = self.state_discretizer.indices_to_midpoints(
            self.state_discretizer.decode_indices(self.current_state)
        )
        return current_state, {}

    def save_mdp_graph(self, output_file='mdp_visualization.html', use_encoded_states=True):
        # Create a directed graph
        G = self.make_mdp_graph(use_encoded_states=use_encoded_states)

        # Use Pyvis for visualization
        net = Network(height='1000px', width='100%', directed=True)
        net.from_nx(G)

        # Normalize counts for coloring
        all_node_counts = [data['count'] for _, data in G.nodes(data=True)]
        all_edge_counts = [data['count'] for _, _, data in G.edges(data=True)]

        node_norm = mcolors.Normalize(vmin=min(all_node_counts), vmax=max(all_node_counts))
        edge_norm = mcolors.Normalize(vmin=min(all_edge_counts), vmax=max(all_edge_counts))

        cmap = LinearSegmentedColormap.from_list("custom_blues", ['#ADD8E6', '#00008B'])  # LightBlue to DarkBlue

        # Set edge colors based on counts
        for edge in net.edges:
            edge_count = G.edges[edge['from'], edge['to']]['count']
            edge_color = mcolors.to_hex(cmap(edge_norm(edge_count)))
            edge['color'] = edge_color

        # Set node colors based on counts
        for node in G.nodes():
            node_count = G.nodes[node]['count']
            node_color = mcolors.to_hex(cmap(node_norm(node_count)))
            net.get_node(node)['color'] = node_color
            net.get_node(node)['title'] = f"State: {net.get_node(node)['str']}, Count: {node_count}"

        if self.targets is not None:
            for node in self.targets:
                net.get_node(node)['color'] = '#FF0000'

        if self.landmark_states is not None:
            for node in self.landmark_states:
                if node in self.targets:
                    continue
                net.get_node(node)['color'] = '#FFA500'

        if self.landmark_start_states is not None:
            for node in self.landmark_start_states:
                if node in self.landmark_states or node in self.targets:
                    continue
                net.get_node(node)['color'] = '#00FF00'

        # Save and display
        net.write_html(output_file, notebook=False, open_browser=False)
        print(f"Saved transition graph at {output_file}.")


class DoubleTransitionalTableEnv(gym.Env):
    def __init__(
            self,
            transition_table_env_t: TransitionalTableEnv or LandmarksTransitionalTableEnv,
            transition_table_env_b: TransitionalTableEnv or LandmarksTransitionalTableEnv,
    ):
        self.transition_table_env_t = transition_table_env_t
        self.transition_table_env_b = transition_table_env_b
        self.observation_space = self.transition_table_env_b.observation_space
        self.action_space = self.transition_table_env_b.action_space

    def reset(self, seed=None, options=None, init_state: np.ndarray = None, reset_all: bool = False):
        t_state, _ = self.transition_table_env_t.reset(seed=seed, options=options, init_state=init_state, reset_all=reset_all)
        b_state, b_info = self.transition_table_env_b.reset(seed=seed, options=options, init_state=t_state, reset_all=reset_all)
        return b_state, b_info

    def step(self, action):
        self.transition_table_env_t.strategy_step_plus_1()
        return self.transition_table_env_b.step(action)


class TabularDynaQAgent:
    def __init__(
            self,
            state_discretizer_t: Discretizer,
            action_discretizer_t: Discretizer,
            state_discretizer_b: Discretizer,
            action_discretizer_b: Discretizer,
            num_targets: int,
            min_cut_max_flow_search_space: int,
            q_cut_space: int,
            weighted_search: bool = True,
            init_state_reward_prob_below_threshold: float = 0.2,
            quality_value_threshold: float = 1.0,
            take_done_states_as_targets: bool = False,
            max_steps: int = 500,
            reward_resolution: int = -1,
            init_strategy_distribution: Tuple[float] = (0.33, 0.33, 0.33),
            exploit_lr: float = 0.1,
            explore_lr: float = 0.1,
            gamma: float = 0.99,
            bonus_decay: float = 0.9,
    ):
        self.state_discretizer_t = state_discretizer_t
        self.action_discretizer_t = action_discretizer_t
        self.state_discretizer_b = state_discretizer_b
        self.action_discretizer_b = action_discretizer_b
        self.transition_table_env_t = LandmarksTransitionalTableEnv(
            state_discretizer_t,
            action_discretizer_t,
            num_targets,
            min_cut_max_flow_search_space,
            q_cut_space,
            weighted_search,
            init_state_reward_prob_below_threshold,
            quality_value_threshold,
            take_done_states_as_targets,
            max_steps,
            reward_resolution,
            init_strategy_distribution,
            unknown_reward=None,
        )
        self.transition_table_env_b = TransitionalTableEnv(
            state_discretizer_b,
            action_discretizer_b,
            max_steps=max_steps,
            reward_resolution=reward_resolution,
            unknown_reward=None,
        )
        self.transition_table_env_e = TransitionalTableEnv(
            state_discretizer_b,
            action_discretizer_b,
            max_steps=max_steps,
            reward_resolution=reward_resolution,
            unknown_reward=1.0,
        )
        self.double_env = DoubleTransitionalTableEnv(self.transition_table_env_t, self.transition_table_env_b)
        self.q_table_agent = TabularQAgent(
            self.double_env,
            self.state_discretizer_b,
            self.action_discretizer_b,
            n_steps=max_steps,
            lr=exploit_lr,
            gamma=gamma,
            print_info=False,
        )
        self.exploration_agent = TabularQAgent(
            self.transition_table_env_e,
            self.state_discretizer_b,
            self.action_discretizer_b,
            n_steps=max_steps,
            lr=explore_lr,
            gamma=gamma,
            print_info=False,
        )
        self.exploration_agent.q_table = defaultdict(lambda: 1.0)
        self.bonus_states = defaultdict(lambda: 1.0)
        self.bonus_decay = bonus_decay

    def print_agent_info(self):
        self.q_table_agent.print_q_table_info()
        self.exploration_agent.print_q_table_info()
        self.transition_table_env_t.print_transition_table_info()
        self.transition_table_env_b.print_transition_table_info()

    def save_agent(self, file_path: str = None) -> tuple[DataFrame, DataFrame, DataFrame, DataFrame, DataFrame]:
        if file_path:
            q_table_file_path = file_path.split(".csv")[0] + "_q_table.csv"
            exploration_agent_q_table_file_path = file_path.split(".csv")[0] + "_exploration_agent_q_table.csv"
            transition_table_t_file_path = file_path.split(".csv")[0] + "_transition_table_t.csv"
            transition_table_b_file_path = file_path.split(".csv")[0] + "_transition_table_b.csv"
            transition_table_e_file_path = file_path.split(".csv")[0] + "_transition_table_e.csv"
        else:
            q_table_file_path = None
            exploration_agent_q_table_file_path = None
            transition_table_t_file_path = None
            transition_table_b_file_path = None
            transition_table_e_file_path = None
        q_table_df = self.q_table_agent.save_q_table(file_path=q_table_file_path)
        exploration_agent_q_table_df = self.exploration_agent.save_q_table(file_path=exploration_agent_q_table_file_path)
        transition_table_t_df = self.transition_table_env_t.save_transition_table(file_path=transition_table_t_file_path)
        transition_table_b_df = self.transition_table_env_b.save_transition_table(file_path=transition_table_b_file_path)
        transition_table_e_df = self.transition_table_env_e.save_transition_table(file_path=transition_table_e_file_path)
        return q_table_df, transition_table_t_df, transition_table_b_df, exploration_agent_q_table_df, transition_table_e_df

    def load_agent(
            self,
            file_path: str = None,
            dataframes: tuple[DataFrame, DataFrame, DataFrame, DataFrame, DataFrame] = (None, None, None, None, None),
    ):
        if file_path:
            q_table_file_path = file_path.split(".csv")[0] + "_q_table.csv"
            exploration_agent_q_table_file_path = file_path.split(".csv")[0] + "_exploration_agent_q_table.csv"
            transition_table_t_file_path = file_path.split(".csv")[0] + "_transition_table_t.csv"
            transition_table_b_file_path = file_path.split(".csv")[0] + "_transition_table_b.csv"
            transition_table_e_file_path = file_path.split(".csv")[0] + "_transition_table_e.csv"
        else:
            q_table_file_path = None
            exploration_agent_q_table_file_path = None
            transition_table_t_file_path = None
            transition_table_b_file_path = None
            transition_table_e_file_path = None
        self.q_table_agent.load_q_table(file_path=q_table_file_path, df=dataframes[0])
        self.exploration_agent.load_q_table(file_path=exploration_agent_q_table_file_path, df=dataframes[0])
        self.transition_table_env_t.load_transition_table(
            file_path=transition_table_t_file_path, transition_table_df=dataframes[1]
        )
        self.transition_table_env_b.load_transition_table(
            file_path=transition_table_b_file_path, transition_table_df=dataframes[2]
        )
        self.transition_table_env_e.load_transition_table(
            file_path=transition_table_e_file_path, transition_table_df=dataframes[3]
        )

    def choose_action(
            self, state: np.ndarray, explore_action: bool = False, temperature: float = 1.0, greedy: bool = False,
    ) -> np.ndarray:
        if explore_action:
            action = self.exploration_agent.choose_action(state, temperature=temperature, greedy=greedy)
        else:
            action = self.q_table_agent.choose_action(state, temperature=temperature, greedy=greedy)
        return action

    def update_from_env(
            self,
            state: np.ndarray,
            action: np.ndarray,
            reward: float,
            next_state: np.ndarray,
            done: bool,
    ):
        encoded_next_state = self.state_discretizer_b.encode_indices(
            list(self.state_discretizer_b.discretize(next_state)[1]))
        reward_bonus = self.bonus_states[encoded_next_state]
        self.bonus_states[encoded_next_state] *= self.bonus_decay
        self.transition_table_env_t.update(state, action, reward, next_state, done)
        self.transition_table_env_b.update(state, action, reward, next_state, done)
        self.transition_table_env_e.update(state, action, reward_bonus, next_state, done)

    def update_from_transition_table(
            self,
            total_timesteps: int,
            train_exploration_agent: bool = False,
            progress_bar: bool = False,
            temperature: float = 1.0,
    ):
        if train_exploration_agent:
            self.transition_table_env_e.reset(reset_all=True)
            self.exploration_agent.learn(total_timesteps=total_timesteps, progress_bar=progress_bar)
        else:
            self.double_env.reset(reset_all=True)
            self.q_table_agent.learn(total_timesteps=total_timesteps, progress_bar=progress_bar, temperature=temperature)
