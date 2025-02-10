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
import warnings

import torch
from gymnasium import spaces
from gymnasium.spaces import Box
from networkx.classes import DiGraph
from pandas import DataFrame
import tqdm
from pyvis.network import Network

import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from stable_baselines3 import PPO, SAC


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
        elif isinstance(vector, np.ndarray) and vector.size == 1:
            vector = vector.item()
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
                    low.append(min_val)
                    high.append(max_val)
            return spaces.Box(low=np.array(low, dtype=np.float32), high=np.array(high, dtype=np.float32), dtype=np.float32)

    def get_space_length(self,):
        """
        Return the flattened vector length of a given Gymnasium space.
        """
        space = self.get_gym_space()
        if isinstance(space, spaces.Discrete):
            return space.n
        elif isinstance(space, spaces.MultiDiscrete):
            return sum(space.nvec)
        elif isinstance(space, spaces.Box):
            return int(np.prod(space.shape))
        elif isinstance(space, spaces.MultiBinary):
            return space.n
        else:
            raise NotImplementedError(f"Space type {type(space)} is not supported.")

    def add_noise(self, vector: np.ndarray) -> np.ndarray:
        """
        Add noise to the input vector. The noise is uniformly sampled within the current bucket's range
        for dimensions with buckets > 1.

        :param vector: Input vector to add noise. Must have the same length as ranges.
        :return: A new vector with added noise.
        """
        assert len(vector) == len(self.ranges), "Input vector must have the same length as ranges."

        noisy_vector = np.copy(vector)

        for i, (value, (min_val, max_val), buckets) in enumerate(zip(vector, self.ranges, self.num_buckets)):
            if buckets > 1:
                # Calculate bucket size
                bucket_size = (max_val - min_val) / buckets

                # Find the current bucket index
                bucket_index = int((value - min_val) / bucket_size)
                bucket_index = min(max(bucket_index, 0), buckets - 1)  # Ensure index is within bounds

                # Calculate the current bucket's range
                bucket_start = min_val + bucket_index * bucket_size
                bucket_end = bucket_start + bucket_size

                # Add noise within the bucket's range
                noisy_vector[i] = np.random.uniform(bucket_start, bucket_end)

        return noisy_vector


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

        # if np.isnan(probabilities[0]):
        #     print()

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

        # reward /= 10.

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

    def add_done_state(self, state):
        if isinstance(state, int) or isinstance(state, float):
            state = [state]
        encoded_state = self.state_discretizer.encode_indices(list(self.state_discretizer.discretize(state)[1]))
        self.done_set.add(encoded_state)

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
            unknown_reward: float = None,
            use_redistribution: bool = False,
            max_delta_reward: float = 10,
            max_self_loop_prob: float = 0.25,
            use_balanced_random_init: bool = False,
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

        self.use_redistribution = use_redistribution
        self.max_delta_reward = max_delta_reward
        self.max_self_loop_prob = max_self_loop_prob

        self.use_balanced_random_init = use_balanced_random_init

    def reset(self, seed=None, options=None, init_state: np.ndarray = None, reset_all: bool = False):
        if len(self.reward_set_dict) == 0:
            warnings.warn("Resetting empty environment, this is invalid, will return None.")
            self.init_strategy = "real_start_states"  # it doesn't really matter in this case.
            return None, {}

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
                if not self.use_balanced_random_init:
                    init_state_encode = random.choice(tuple(self.forward_dict.keys()))
                else:
                    weights = np.array([value + 1 for value in self.state_count.values()])
                    inverse_weights = 1 / weights
                    softmax_probs = np.exp(inverse_weights) / np.sum(np.exp(inverse_weights))
                    np.random.seed(random.randint(0, 10000000))
                    init_state_encode = np.random.choice(list(self.state_count.keys()), p=softmax_probs)

            elif self.init_strategy == "real_start_states":
                init_state_encode = random.choice(tuple(self.start_set))
            else:
                raise ValueError(f"Init strategy not supported: {self.init_strategy}.")
        self.current_state = int(init_state_encode)
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

        ######################################################################################
        # Step 1: Extract transition probabilities
        transition_probabilities = {
            encoded_next_state: prob
            for encoded_next_state, (_, prob) in transition_state_avg_reward_and_prob.items()
        }

        if self.use_redistribution:
            self_loop_prob = transition_probabilities.get(encoded_state, 0.0)

            if self_loop_prob > 0.0:
                # Step 2: Compare rewards for adjustment
                current_reward = transition_state_avg_reward_and_prob[encoded_state][0]
                should_adjust = True  # Flag to determine if adjustment is needed

                # Check reward differences
                for state, (reward, prob) in transition_state_avg_reward_and_prob.items():
                    if state != encoded_state:  # Compare only with other states
                        reward_diff = abs(current_reward - reward)
                        if reward_diff >= self.max_delta_reward:
                            should_adjust = False
                            break

                # Step 3: Adjust probabilities if necessary
                if should_adjust and self_loop_prob > self.max_self_loop_prob:
                    total_prob = sum(transition_probabilities.values())
                    adjustment = self_loop_prob - self.max_self_loop_prob
                    transition_probabilities[encoded_state] = self.max_self_loop_prob

                    # Redistribute the remaining probability to other states
                    for state in transition_probabilities:
                        if state != encoded_state:
                            transition_probabilities[state] += adjustment * (
                                    transition_probabilities[state] / (total_prob - self_loop_prob)
                            )

                    # Normalize probabilities to ensure they sum to 1
                    total_prob = sum(transition_probabilities.values())
                    transition_probabilities = {
                        state: prob / total_prob for state, prob in transition_probabilities.items()
                    }

                    # Step 4: Adjust rewards to maintain expected reward consistency
                    adjusted_rewards = {}
                    for state, prob in transition_probabilities.items():
                        original_prob = transition_state_avg_reward_and_prob[state][1]
                        original_reward = transition_state_avg_reward_and_prob[state][0]
                        if prob > 0:
                            # Adjust reward based on new probability
                            adjusted_rewards[state] = (original_prob * original_reward) / prob
                        else:
                            adjusted_rewards[state] = 0  # For zero-probability states

                    # Update the original dictionary with adjusted rewards
                    for state in transition_state_avg_reward_and_prob:
                        transition_state_avg_reward_and_prob[state] = (
                            adjusted_rewards[state],
                            transition_probabilities[state],
                        )

                    # Step 5: Compute KL divergence between original and new distributions
                    original_probs = [
                        v[1] for v in transition_state_avg_reward_and_prob.values()
                    ]
                    new_probs = [
                        transition_probabilities[state]
                        for state in transition_state_avg_reward_and_prob.keys()
                    ]

                    kl_divergence = sum(
                        p * math.log(p / q) for p, q in zip(original_probs, new_probs) if p > 0 and q > 0
                    )

        # Step 6: Sample next state from adjusted probabilities
        encoded_next_state = random.choices(
            population=list(transition_probabilities.keys()),
            weights=list(transition_probabilities.values()),
            k=1
        )[0]

        # Step 6: Get reward for the sampled state
        reward = transition_state_avg_reward_and_prob[encoded_next_state][0]
        ######################################################################################

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

    def activate_redistribution(self):
        self.use_redistribution = True

    def deactivate_redistribution(self):
        self.use_redistribution = False


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
            if start_node == target:
                continue
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
        if len(self.reward_set_dict) == 0:
            warnings.warn("Resetting empty environment, this is invalid, will return None.")
            return None, {}

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
        self.current_state = int(init_state_encode)
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
            exploit_policy_reward_rate=0.1,
    ):
        self.transition_table_env_t = transition_table_env_t
        self.transition_table_env_b = transition_table_env_b
        self.observation_space = self.transition_table_env_b.observation_space
        self.action_space = self.transition_table_env_b.action_space
        self.exploit_policy_reward_rate = exploit_policy_reward_rate

    def reset(self, seed=None, options=None, init_state: np.ndarray = None, reset_all: bool = False):
        t_state, _ = self.transition_table_env_t.reset(seed=seed, options=options, init_state=init_state, reset_all=reset_all)
        b_state, b_info = self.transition_table_env_b.reset(seed=seed, options=options, init_state=t_state, reset_all=reset_all)
        return b_state, b_info

    def step(self, action):
        self.transition_table_env_t.strategy_step_plus_1()
        next_state, reward, terminated, truncated, info = self.transition_table_env_b.step(action)
        return next_state, reward * self.exploit_policy_reward_rate, terminated, truncated, info


class HybridEnv(gym.Env):
    def __init__(
            self,
            transition_table_env: TransitionalTableEnv or LandmarksTransitionalTableEnv,
            real_env: gym.Env,
            exploit_policy_reward_rate=0.1,
    ):
        self.transition_table_env = transition_table_env
        self.real_env = real_env
        self.observation_space = self.real_env.observation_space
        self.action_space = self.real_env.action_space
        self.exploit_policy_reward_rate = exploit_policy_reward_rate
        self.current_state = None

    def reset(self, seed=None, options=None, init_state: np.ndarray = None, reset_all: bool = False):
        t_state, _ = self.transition_table_env.reset(seed=seed, options=options, init_state=init_state, reset_all=reset_all)
        b_state, b_info = self.real_env.reset(seed=seed, options=options,)
        if t_state is None:
            t_state = b_state
        if "Pendulum" in self.real_env.spec.id:
            t_state = self.transition_table_env.state_discretizer.add_noise(t_state)
            t_state_ = np.arccos(t_state[0]), t_state[1]
            self.real_env.unwrapped.state = t_state_
        elif "Acrobot" in self.real_env.spec.id:
            t_state = self.transition_table_env.state_discretizer.add_noise(t_state)
            t_state_ = np.arccos(t_state[0]), np.arccos(t_state[1]), t_state[2], t_state[3]
            self.real_env.unwrapped.state = t_state_
        elif "HalfCheetah" in self.real_env.spec.id:
            t_state = self.transition_table_env.state_discretizer.add_noise(t_state)
            # Ensure the state vector has the correct observation dimensions (17)
            assert len(t_state) == 17, \
                f"State vector length must be 17 for HalfCheetah observation space."
            # Retrieve the model dimensions
            nq = self.real_env.unwrapped.model.nq  # Number of qpos (9)
            nv = self.real_env.unwrapped.model.nv  # Number of qvel (9)
            # Initialize qpos and qvel
            target_qpos = np.zeros(nq)
            target_qvel = np.zeros(nv)
            # Fill qpos and qvel based on t_state
            target_qpos[1:] = t_state[:nq - 1]  # Skip rootx for qpos
            target_qvel[:] = t_state[nq - 1:]  # Full qvel from t_state
            # Set the state using the environment's set_state method
            self.real_env.unwrapped.set_state(target_qpos, target_qvel)
            # Update the t_state to reflect the new state
            t_state = np.concatenate([
                self.real_env.unwrapped.data.qpos[1:],  # Exclude rootx
                self.real_env.unwrapped.data.qvel
            ])
        elif "Hopper" in self.real_env.spec.id:
            t_state = self.transition_table_env.state_discretizer.add_noise(t_state)
            # Ensure the observation vector length is 11
            assert len(t_state) == 11, \
                f"Observation vector length must be 11 for Hopper, but got {len(t_state)}."

            # Retrieve the model's qpos and qvel dimensions
            nq = self.real_env.unwrapped.model.nq  # qpos has 6 elements
            nv = self.real_env.unwrapped.model.nv  # qvel has 6 elements

            # Initialize target qpos and qvel
            target_qpos = np.zeros(nq)
            target_qvel = np.zeros(nv)

            # Set a default value for rootx (e.g., 0.0)
            target_qpos[0] = 0.0  # Horizontal position of the torso

            # Map the observation vector to qpos and qvel
            target_qpos[1:] = t_state[:nq - 1]  # Skip rootx for qpos
            target_qvel[:] = t_state[nq - 1:]  # Set qvel from observation

            # Set the state using the environment's set_state method
            self.real_env.unwrapped.set_state(target_qpos, target_qvel)

            # Update t_state to reflect the new state (exclude rootx from qpos)
            t_state = np.concatenate([
                self.real_env.unwrapped.data.qpos[1:],  # Exclude rootx
                self.real_env.unwrapped.data.qvel
            ])
        elif "Reacher" in self.real_env.spec.id:
            # Ensure the observation vector length is 10
            assert len(t_state) == 10, \
                f"Observation vector length must be 10 for Reacher, but got {len(t_state)}."

            # Retrieve the model's qpos and qvel dimensions
            nq = self.real_env.unwrapped.model.nq  # qpos has 4 elements
            nv = self.real_env.unwrapped.model.nv  # qvel has 4 elements

            # Initialize target qpos and qvel
            target_qpos = np.zeros(nq)
            target_qvel = np.zeros(nv)

            # Map the observation vector to qpos and qvel
            # qpos: [joint0_angle, joint1_angle, target_x, target_y]
            # qvel: [joint0_velocity, joint1_velocity, 0, 0] (last two are not used)

            # Calculate joint angles from sin and cos values
            target_qpos[0] = np.arctan2(t_state[2], t_state[0])  # joint0_angle
            target_qpos[1] = np.arctan2(t_state[3], t_state[1])  # joint1_angle

            # Set target positions
            target_qpos[2] = t_state[4]  # target_x
            target_qpos[3] = t_state[5]  # target_y

            # Set joint velocities
            target_qvel[0] = t_state[6]  # joint0_velocity
            target_qvel[1] = t_state[7]  # joint1_velocity

            # Set the state using the environment's set_state method
            self.real_env.unwrapped.set_state(target_qpos, target_qvel)

            # Update t_state to reflect the new state
            t_state = np.concatenate([
                [np.cos(target_qpos[0]), np.cos(target_qpos[1])],  # cos(joint0), cos(joint1)
                [np.sin(target_qpos[0]), np.sin(target_qpos[1])],  # sin(joint0), sin(joint1)
                target_qpos[2:],  # target_x, target_y
                target_qvel[:2],  # joint0_velocity, joint1_velocity
                t_state[8:]  # fingertip-target vector
            ])
        elif "FrozenLake" in self.real_env.spec.id:
            if isinstance(t_state, list):
                t_state = t_state[0]
            self.real_env.unwrapped.s = t_state
        elif "MountainCar" in self.real_env.spec.id:
            self.real_env.unwrapped.state = t_state
        self.current_state = t_state
        self.transition_table_env.add_start_state(t_state)
        return t_state, b_info

    def step(self, action):
        self.transition_table_env.strategy_step_plus_1()
        next_state, reward, terminated, truncated, info = self.real_env.step(action)
        if terminated:
            self.transition_table_env.add_done_state(next_state)
        self.transition_table_env.update(self.current_state, action, reward, next_state, terminated)
        self.current_state = next_state
        return next_state, reward * self.exploit_policy_reward_rate, terminated, truncated, info


class PyramidTransitionalTableEnv(gym.Env):
    def __init__(
            self,
            transition_table_envs: List[TransitionalTableEnv] or List[LandmarksTransitionalTableEnv],
            exploit_policy_reward_rate: float = 0.1,
            add_noise: bool = False,
            leader_env_index: int = 0,
    ):
        self.transition_table_envs = transition_table_envs
        self.exploit_policy_reward_rate = exploit_policy_reward_rate
        self.add_noise = add_noise
        self.observation_space = self.transition_table_envs[0].observation_space
        self.action_space = self.transition_table_envs[0].action_space
        self.leader_env = self.transition_table_envs[leader_env_index]
        self.slave_env: Optional[TransitionalTableEnv] = None

    def reset(
            self,
            seed=None,
            options=None,
            init_state: np.ndarray = None,
            reset_all: bool = False,
            slave_env_index: int = 0,
            add_noise: bool = None,
            use_redistribution: bool = None,
            add_noise_from_leader: bool = False,
    ):
        if reset_all:
            if add_noise is not None:
                self.add_noise = add_noise
            self.slave_env = self.transition_table_envs[slave_env_index]
            if use_redistribution is not None:
                if use_redistribution:
                    self.slave_env.activate_redistribution()
                else:
                    self.slave_env.deactivate_redistribution()
        leader_state, _ = self.leader_env.reset(seed=seed, options=options, init_state=init_state, reset_all=reset_all)
        if add_noise_from_leader:
            leader_state = self.leader_env.state_discretizer.add_noise(leader_state)
        state, info = self.slave_env.reset(seed=seed, options=options, init_state=leader_state, reset_all=reset_all)
        if self.add_noise:
            state = self.slave_env.state_discretizer.add_noise(state)
        return state, info

    def step(self, action):
        self.leader_env.strategy_step_plus_1()
        next_state, reward, terminated, truncated, info = self.slave_env.step(action)
        if self.add_noise:
            next_state = self.slave_env.state_discretizer.add_noise(next_state)
        return next_state, reward * self.exploit_policy_reward_rate, terminated, truncated, info

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
            init_strategy_distribution: Tuple[float] = None,
            exploit_lr: float = 0.1,
            explore_lr: float = 0.1,
            gamma: float = 0.99,
            bonus_decay: float = 0.9,
            use_real_env: bool = False,
            real_env: gym.Env = None,
    ):
        self.state_discretizer_t = state_discretizer_t
        self.action_discretizer_t = action_discretizer_t
        self.state_discretizer_b = state_discretizer_b
        self.action_discretizer_b = action_discretizer_b
        if init_strategy_distribution is None:
            init_strategy_distribution = (0.33, 0.33, 0.33)
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
        if use_real_env:
            assert real_env is not None, "env must be provided if use_real_env is True."
            self.double_env = HybridEnv(
                self.transition_table_env_t,
                real_env,
            )
        else:
            self.double_env = DoubleTransitionalTableEnv(
                self.transition_table_env_t,
                self.transition_table_env_b,
            )
        self.exploit_agent = TabularQAgent(
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
        self.exploit_agent.print_q_table_info()
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
        q_table_df = self.exploit_agent.save_q_table(file_path=q_table_file_path)
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
        self.exploit_agent.load_q_table(file_path=q_table_file_path, df=dataframes[0])
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
            action = self.exploit_agent.choose_action(state, temperature=temperature, greedy=greedy)
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
            self.exploit_agent.learn(total_timesteps=total_timesteps, progress_bar=progress_bar, temperature=temperature)

    def update_from_real_env(self, total_timesteps: int, real_env: gym.Env, progress_bar: bool = False,):
        current_env = self.exploit_agent.env
        self.exploit_agent.env = real_env
        self.exploit_agent.learn(total_timesteps=total_timesteps, progress_bar=progress_bar)
        self.exploit_agent.env = current_env


class DeepDynaQAgent:
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
            init_strategy_distribution: Tuple[float] = None,
            exploit_lr: float = 0.1,
            explore_lr: float = 0.1,
            gamma: float = 0.99,
            bonus_decay: float = 0.9,
            exploit_policy_reward_rate: float = 1e-1,
            use_real_env: bool = False,
            real_env: gym.Env = None,
    ):
        self.state_discretizer_t = state_discretizer_t
        self.action_discretizer_t = action_discretizer_t
        self.state_discretizer_b = state_discretizer_b
        self.action_discretizer_b = action_discretizer_b
        if init_strategy_distribution is None:
            init_strategy_distribution = (0.33, 0.33, 0.33)
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
            # n_steps=max_steps,
            reward_resolution=reward_resolution,
            unknown_reward=1.0,
        )
        if use_real_env:
            assert real_env is not None, "env must be provided if use_real_env is True."
            self.double_env = HybridEnv(
                self.transition_table_env_t,
                real_env,
                exploit_policy_reward_rate=exploit_policy_reward_rate,
            )
        else:
            self.double_env = DoubleTransitionalTableEnv(
                self.transition_table_env_t,
                self.transition_table_env_b,
                exploit_policy_reward_rate=exploit_policy_reward_rate,
            )
        # print(self.double_env.action_space)
        self.exploit_agent = PPO(
            "MlpPolicy",
            self.double_env,
            learning_rate=exploit_lr,
            gamma=gamma,
            verbose=0,
            n_epochs=5,
            device='auto',
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
        self.transition_table_env_t.print_transition_table_info()
        self.transition_table_env_b.print_transition_table_info()

    def save_agent(self, file_path: str = None) -> tuple[DataFrame, DataFrame, DataFrame,]:
        if file_path:
            q_table_file_path = file_path.split(".csv")[0] + "_exploit_agent"
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
        self.exploit_agent.save(q_table_file_path)
        exploration_agent_q_table_df = self.exploration_agent.save_q_table(file_path=exploration_agent_q_table_file_path)
        transition_table_t_df = self.transition_table_env_t.save_transition_table(file_path=transition_table_t_file_path)
        transition_table_b_df = self.transition_table_env_b.save_transition_table(file_path=transition_table_b_file_path)
        transition_table_e_df = self.transition_table_env_e.save_transition_table(file_path=transition_table_e_file_path)
        return transition_table_t_df, transition_table_b_df, transition_table_e_df

    def load_agent(
            self,
            file_path: str = None,
    ):
        if file_path:
            q_table_file_path = file_path.split(".csv")[0] + "_exploit_agent"
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
        self.exploit_agent.load(q_table_file_path)
        self.exploration_agent.load_q_table(file_path=exploration_agent_q_table_file_path,)
        self.transition_table_env_t.load_transition_table(
            file_path=transition_table_t_file_path,
        )
        self.transition_table_env_b.load_transition_table(
            file_path=transition_table_b_file_path,
        )
        self.transition_table_env_e.load_transition_table(
            file_path=transition_table_e_file_path,
        )

    def choose_action(
            self, state: np.ndarray, explore_action: bool = False, temperature: float = 1.0, greedy: bool = False,
    ) -> np.ndarray:
        if explore_action:
            action = self.exploration_agent.choose_action(state, temperature=temperature, greedy=greedy)
        else:
            action, _ = self.exploit_agent.predict(state, deterministic=greedy)
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
    ):
        if train_exploration_agent:
            self.transition_table_env_e.reset(reset_all=True)
            self.exploration_agent.learn(total_timesteps=total_timesteps, progress_bar=progress_bar)
        else:
            self.double_env.reset(reset_all=True)
            self.exploit_agent.learn(total_timesteps=total_timesteps, progress_bar=progress_bar)

    def update_from_real_env(self, total_timesteps: int, real_env: gym.Env, progress_bar: bool = False,):
        current_env = self.exploit_agent.env
        real_env = PPO._wrap_env(real_env, verbose=self.exploit_agent.verbose, monitor_wrapper=True)
        self.exploit_agent.env = real_env
        self.exploit_agent.learn(total_timesteps=total_timesteps, progress_bar=progress_bar)
        self.exploit_agent.env = current_env


class DeepPyramidDynaQAgent:
    def __init__(
            self,
            state_discretizers: List[Discretizer],
            action_discretizers: List[Discretizer],
            max_steps: int = 500,
            reward_resolution: int = -1,
            init_strategy_distribution: Tuple[float] = (0.5, 0.5,),
            exploit_lr: float = 2.5e-4,
            explore_lr: float = 0.1,
            gamma: float = 0.99,
            bonus_decay: float = 0.9,
            exploit_policy_reward_rate: float = 1e-1,
            add_noise: bool = False,
    ):
        self.transition_table_envs = []
        for state_discretizer, action_discretizer in zip(state_discretizers, action_discretizers):
            self.transition_table_envs.append(
                TransitionalTableEnv(
                    state_discretizer,
                    action_discretizer,
                    max_steps=max_steps,
                    reward_resolution=reward_resolution,
                    unknown_reward=None,
                    init_strategy_distribution=init_strategy_distribution,
                )
            )
        self.pyramid_env = PyramidTransitionalTableEnv(
            self.transition_table_envs,
            exploit_policy_reward_rate=exploit_policy_reward_rate,
            add_noise=add_noise,
            leader_env_index=0,
        )
        self.transition_table_env_e = TransitionalTableEnv(
            state_discretizers[-1],
            action_discretizers[-1],
            # n_steps=max_steps,
            reward_resolution=reward_resolution,
            unknown_reward=1.0,
        )
        # print(self.double_env.action_space)
        self.exploit_agent = PPO(
            "MlpPolicy",
            self.pyramid_env,
            learning_rate=exploit_lr,
            gamma=gamma,
            verbose=0,
            n_epochs=5,
            device='auto',
        )
        self.exploration_agent = TabularQAgent(
            self.transition_table_env_e,
            state_discretizers[-1],
            action_discretizers[-1],
            n_steps=max_steps,
            lr=explore_lr,
            gamma=gamma,
            print_info=False,
        )
        self.exploration_agent.q_table = defaultdict(lambda: 1.0)
        self.bonus_states = defaultdict(lambda: 1.0)
        self.bonus_decay = bonus_decay

    def print_agent_info(self):
        for transition_table_env in self.transition_table_envs:
            transition_table_env.print_transition_table_info()

    def choose_action(
            self, state: np.ndarray, explore_action: bool = False, temperature: float = 1.0, greedy: bool = False,
    ) -> np.ndarray:
        if explore_action:
            action = self.exploration_agent.choose_action(state, temperature=temperature, greedy=greedy)
        else:
            action, _ = self.exploit_agent.predict(state, deterministic=greedy)
        return action

    def update_from_env(
            self,
            state: np.ndarray,
            action: np.ndarray,
            reward: float,
            next_state: np.ndarray,
            done: bool,
    ):
        encoded_next_state = self.transition_table_env_e.state_discretizer.encode_indices(
            list(self.transition_table_env_e.state_discretizer.discretize(next_state)[1]))
        reward_bonus = self.bonus_states[encoded_next_state]
        self.bonus_states[encoded_next_state] *= self.bonus_decay
        for transition_table_env in self.transition_table_envs:
            transition_table_env.update(state, action, reward, next_state, done)
        self.transition_table_env_e.update(state, action, reward_bonus, next_state, done)

    def update_from_transition_table(
            self,
            total_timesteps: int,
            train_exploration_agent: bool = False,
            progress_bar: bool = False,
            slave_env_index: int = 0,
            add_noise: bool = False,
            use_redistribution: bool = False,
    ):
        if train_exploration_agent:
            self.transition_table_env_e.reset(reset_all=True)
            self.exploration_agent.learn(total_timesteps=total_timesteps, progress_bar=progress_bar)
        else:
            self.pyramid_env.reset(
                reset_all=True,
                slave_env_index=slave_env_index,
                add_noise=add_noise,
                use_redistribution=use_redistribution,
                add_noise_from_leader=False,
            )
            self.exploit_agent.learn(total_timesteps=total_timesteps, progress_bar=progress_bar)

    def update_from_real_env(self, total_timesteps: int, real_env: gym.Env, progress_bar: bool = False,):
        current_env = self.exploit_agent.env
        real_env = PPO._wrap_env(real_env, verbose=self.exploit_agent.verbose, monitor_wrapper=True)
        self.exploit_agent.env = real_env
        self.exploit_agent.learn(total_timesteps=total_timesteps, progress_bar=progress_bar)
        self.exploit_agent.env = current_env


class Agent:
    def __init__(
            self,
            state_discretizer: Discretizer,
            action_discretizer: Discretizer,
            env: gym.Env,
            use_deep_agent: bool,
            max_steps: int = 500,
            init_strategy_distribution: Tuple[float, float] = None,
            exploit_lr: float = 0.1,
            gamma: float = 0.99,
            exploit_policy_reward_rate: float = 1.0,
            use_balanced_random_init: bool = False,
    ):
        self.state_discretizer = state_discretizer
        self.action_discretizer = action_discretizer
        if init_strategy_distribution is None:
            init_strategy_distribution = (0.5, 0.5,)
        self.transition_table_env = TransitionalTableEnv(
            state_discretizer,
            action_discretizer,
            max_steps,
            reward_resolution=0,
            init_strategy_distribution=init_strategy_distribution,
            unknown_reward=None,
            use_balanced_random_init=use_balanced_random_init,
        )
        self.double_env = HybridEnv(
            self.transition_table_env,
            env,
            exploit_policy_reward_rate=exploit_policy_reward_rate,
        )
        # print(self.double_env.action_space)
        self.use_deep_agent = use_deep_agent
        if use_deep_agent:
            self.exploit_agent = PPO(
                "MlpPolicy",
                self.double_env,
                learning_rate=exploit_lr,
                gamma=gamma,
                verbose=0,
                n_epochs=5,
                device='auto',
            )
        else:
            self.exploit_agent = TabularQAgent(
                self.double_env,
                self.state_discretizer,
                self.action_discretizer,
                n_steps=max_steps,
                lr=exploit_lr,
                gamma=gamma,
                print_info=False,
            )

    def print_agent_info(self):
        self.transition_table_env.print_transition_table_info()

    def save_agent(self, file_path: str):
        q_table_file_path = file_path + "_exploit_agent.csv"
        deep_model_file_path = file_path + "_deep_model.zip"
        transition_table_file_path = file_path + "_transition_table.csv"
        if self.use_deep_agent:
            self.exploit_agent.save(deep_model_file_path)
        else:
            self.exploit_agent.save_q_table(file_path=q_table_file_path)
        self.transition_table_env.save_transition_table(file_path=transition_table_file_path)

    def load_agent(self, file_path: str, load_transition_table: bool = True):
        q_table_file_path = file_path + "_exploit_agent.csv"
        deep_model_file_path = file_path + "_deep_model.zip"
        transition_table_file_path = file_path + "_transition_table.csv"
        if self.use_deep_agent:
            self.exploit_agent = PPO.load(deep_model_file_path, env=self.double_env, print_system_info=True)
        else:
            self.exploit_agent.load_q_table(file_path=q_table_file_path)
        if load_transition_table:
            self.transition_table_env.load_transition_table(file_path=transition_table_file_path)

    def choose_action(
            self, state: np.ndarray, temperature: float = 1.0, greedy: bool = False,
    ) -> np.ndarray:
        if not self.use_deep_agent:
            action = self.exploit_agent.choose_action(state, temperature=temperature, greedy=greedy)
        else:
            action, _ = self.exploit_agent.predict(state, deterministic=greedy)
        if isinstance(self.action_discretizer.get_gym_space(), spaces.Discrete):
            action = int(action)
        return action

    def get_action_probabilities(self, state: np.ndarray, temperature: float = 1.0, greedy: bool = False):
        if not self.use_deep_agent:
            action_probabilities = self.exploit_agent.get_action_probabilities(state, temperature)
            if greedy:
                # Only consider the case that it is discrete action space for q table agent
                # Make the optimal action probability 1, others 0
                optimal_action = np.argmax(action_probabilities)
                action_probabilities = np.zeros_like(action_probabilities)
                action_probabilities[optimal_action] = 1.0
            return action_probabilities
        else:
            if isinstance(self.double_env.action_space, spaces.Discrete):
                # Discrete action space
                action_distribution = self.exploit_agent.policy.get_distribution(state)
                logits = action_distribution.distribution.logits
                if greedy:
                    # Make the optimal action probability 1, others 0
                    optimal_action = np.argmax(logits)
                    action_probabilities = np.zeros_like(logits)
                    action_probabilities[optimal_action] = 1.0
                else:
                    # Apply temperature to logits and compute probabilities
                    logits = logits / temperature
                    exp_logits = np.exp(logits - np.max(logits))  # For numerical stability
                    action_probabilities = exp_logits / np.sum(exp_logits)
                return action_probabilities
            else:
                # Continuous action space
                action_distribution = self.exploit_agent.policy.get_distribution(state)
                mean = action_distribution.distribution.mean.detach().cpu().numpy()
                std = action_distribution.distribution.stddev.detach().cpu().numpy()
                if greedy:
                    # Set standard deviation to 0 for greedy policy
                    std = np.zeros_like(mean)
                return mean, std

    def get_default_policy_distribution(self, state: np.ndarray, p=1.0):
        """
        Get the default policy distribution for actions.

        :param state: The current state as a NumPy array.
        :return: The default distribution:
                 - For discrete actions: Uniform distribution.
                 - For continuous actions: Mean and standard deviation based on the action space.
        """
        if isinstance(self.action_discretizer.get_gym_space(), spaces.Discrete):
            # Discrete action space: Uniform distribution
            action_space_size = self.double_env.action_space.n
            default_distribution = np.ones(action_space_size) / action_space_size
            sum_except_last = np.sum(default_distribution[:-1])
            default_distribution[-1] = 1 - sum_except_last
            return default_distribution
        else:
            # Continuous action space: Uniform mean and std
            low, high = self.double_env.action_space.low, self.double_env.action_space.high
            uniform_mean = (low + high) / 2
            uniform_std = (high - low) / 2
            return uniform_mean, uniform_std

    def get_greedy_weighted_action_distribution(
            self,
            state: np.ndarray,
            p: float = 1.0,
            default_policy_func=None,
    ):
        """
        Get a greedy-weighted distribution for actions.

        :param state: The current state as a NumPy array.
        :param p: Weight for the greedy action distribution. Must be in [0, 1].
        :return: A distribution for the actions:
                 - For discrete actions: A single probability array.
                 - For continuous actions: Two arrays, mean and std of the distribution.

        Args:
            default_policy_func:
        """
        assert 0 <= p <= 1, "p must be between 0 and 1 (inclusive)."

        if not self.use_deep_agent:
            action_probabilities = self.exploit_agent.get_action_probabilities(state)
            if isinstance(self.action_discretizer.get_gym_space(), spaces.Discrete):
                # Greedy discrete action
                # Find the maximum probability action
                greedy_action = np.argmax(action_probabilities)
                max_prob = action_probabilities[greedy_action]

                # Find all actions whose probability is at least 99% of the maximum probability
                optimal_actions = np.where(action_probabilities >= 0.99 * max_prob)[0]

                # Create a new greedy distribution where all optimal actions share the probability equally
                greedy_distribution = np.zeros_like(action_probabilities)
                greedy_distribution[optimal_actions] = 1.0 / len(optimal_actions)  # Equally distribute probability

                # Normalize to ensure sum = 1
                greedy_distribution /= np.sum(greedy_distribution)

                # Default distribution
                if default_policy_func is None:
                    default_distribution = self.get_default_policy_distribution(state)
                else:
                    default_distribution = default_policy_func(state, p=1.0)

                # Weighted combination
                weighted_distribution = p * greedy_distribution + (1 - p) * default_distribution
                weighted_distribution /= np.sum(weighted_distribution)
                return weighted_distribution
            else:
                raise NotImplementedError("Non-deep agents for continuous actions are not supported.")
        else:
            if isinstance(self.double_env.action_space, spaces.Discrete):
                # Deep agent with discrete action space
                with torch.no_grad():
                    action_distribution = self.exploit_agent.policy.get_distribution(torch.tensor(state).unsqueeze(0).to(self.exploit_agent.policy.device))
                    logits = action_distribution.distribution.logits.cpu().squeeze().numpy()

                # Compute softmax probabilities with numerical stability
                probabilities = np.exp(logits) / (np.sum(np.exp(logits)) + 1e-10)

                # Find the maximum probability action
                greedy_action = np.argmax(probabilities)
                max_prob = probabilities[greedy_action]

                # Find all actions whose probability is at least 99% of the maximum probability
                optimal_actions = np.where(probabilities >= 0.99 * max_prob)[0]

                # Create a new greedy distribution
                greedy_distribution = np.zeros_like(probabilities)

                # Assign equal probability to all optimal actions
                greedy_distribution[optimal_actions] = 1.0 / len(optimal_actions)

                # Normalize to ensure sum = 1
                greedy_distribution /= np.sum(greedy_distribution)

                # Default distribution
                if default_policy_func is None:
                    default_distribution = self.get_default_policy_distribution(state)
                else:
                    default_distribution = default_policy_func(state, p=1.0)

                # Weighted combination
                weighted_distribution = p * greedy_distribution + (1 - p) * default_distribution
                weighted_distribution /= np.sum(weighted_distribution)
                return weighted_distribution
            else:
                # Deep agent with continuous action space
                with torch.no_grad():
                    action_distribution = self.exploit_agent.policy.get_distribution(torch.tensor(state).unsqueeze(0).to(self.exploit_agent.policy.device))
                    greedy_mean = action_distribution.distribution.mean.cpu().squeeze().numpy()

                # Default distribution
                if default_policy_func is None:
                    uniform_mean, uniform_std = self.get_default_policy_distribution(state)
                else:
                    uniform_mean, uniform_std = default_policy_func(state, p=1.0)

                # Weighted combination
                weighted_mean = p * greedy_mean + (1 - p) * uniform_mean
                weighted_std = uniform_std
                return weighted_mean, weighted_std

    def choose_action_by_weight(
            self, state: np.ndarray, p: float = 0.5, default_policy_func=None,
    ) -> np.ndarray:
        """
        Choose an action based on the greedy-weighted distribution.

        :param state: The current state as a NumPy array.
        :param p: Weight for the greedy action distribution. Must be in [0, 1].
        :return: The chosen action as a NumPy array (or scalar for discrete actions).

        Args:
            default_policy_func:
        """
        assert 0 <= p <= 1, "p must be between 0 and 1 (inclusive)."

        if isinstance(self.double_env.action_space, spaces.Discrete):
            # Get the weighted distribution
            action_probabilities = self.get_greedy_weighted_action_distribution(state, p=p, default_policy_func=default_policy_func)

            # Sample an action based on the distribution
            np.random.seed(random.randint(0, 100000000))
            action = np.random.choice(len(action_probabilities), p=action_probabilities)
            return int(action)
        else:
            # Continuous action space
            mean, std = self.get_greedy_weighted_action_distribution(state, p=p, default_policy_func=default_policy_func)

            # Sample an action from the Gaussian distribution
            np.random.seed(random.randint(0, 100000000))
            action = np.random.normal(mean, std)

            # Clip action to ensure it remains within the valid range
            action = np.clip(action, self.double_env.action_space.low, self.double_env.action_space.high)
            return action

    def learn(
            self,
            total_timesteps: int,
            progress_bar: bool = False,
    ):
        self.double_env.reset(reset_all=True)
        self.exploit_agent.learn(total_timesteps=total_timesteps, progress_bar=progress_bar)
