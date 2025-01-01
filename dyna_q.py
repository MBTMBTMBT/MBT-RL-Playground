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

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class TabularQAgent:
    def __init__(self, state_discretizer: Discretizer, action_discretizer: Discretizer, print_info: bool = True):
        self.state_discretizer = state_discretizer
        self.action_discretizer = action_discretizer
        self.q_table = defaultdict(lambda: 0.0)  # Flattened Q-Table with state-action tuple keys
        self.visit_table = defaultdict(lambda: 0)  # Uses the same keys of the Q-Table to do visit count.
        if print_info:
            self.print_q_table_info()
        self.all_actions_encoded = sorted([
            self.action_discretizer.encode_indices([*indices])
            for indices in self.action_discretizer.list_all_possible_combinations()[1]
        ])

    def reset_q_table(self) -> None:
        self.q_table = defaultdict(lambda: 0.0)  # Flattened Q-Table with state-action tuple keys

    def reset_visit_table(self) -> None:
        self.visit_table = defaultdict(lambda: 0)  # Uses the same keys of the Q-Table to do visit count.

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
        q_values = np.array([self.q_table[(encoded_state, a)] for a in self.all_actions_encoded])
        visitation = sum([self.visit_table[(encoded_state, a)] for a in self.all_actions_encoded])

        if strategy == "greedy":
            probabilities = np.zeros_like(q_values, dtype=float)
            if np.any(q_values) > 0:
                probabilities[np.argmax(q_values)] = 1.0
            else:
                probabilities = np.ones_like(q_values, dtype=float) / len(q_values)
        elif strategy == "softmax":
            if np.all(q_values == 0):  # Handle all-zero Q-values
                probabilities = np.ones_like(q_values, dtype=float) / len(q_values)
            else:
                # Subtract the maximum value for numerical stability
                q_values_stable = q_values - np.max(q_values)
                exp_values = np.exp(q_values_stable / temperature)
                probabilities = exp_values / (np.sum(exp_values) + 1e-10)  # Add small value to prevent division by zero
        else:
            raise ValueError("Invalid strategy. Use 'greedy' or 'softmax'.")

        if visitation == 0:
            probabilities = np.ones_like(q_values, dtype=float) / len(q_values)

        return probabilities

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


# # hyper-parameters
# BATCH_SIZE = 128
# LR = 0.01
# GAMMA = 0.90
# EPISILO = 0.9
# MEMORY_CAPACITY = 2000
# Q_NETWORK_ITERATION = 100
#
# env = gym.make("CartPole-v0")
# env = env.unwrapped
# NUM_ACTIONS = env.action_space.n
# NUM_STATES = env.observation_space.shape[0]
# ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample.shape


# https://github.com/sweetice/Deep-reinforcement-learning-with-pytorch/blob/master/Char01%20DQN/DQN.py
# class Net(nn.Module):
#     """docstring for Net"""
#
#     def __init__(self, num_inputs: int, num_actions: int, hidden_size=None):
#         super(Net, self).__init__()
#         if hidden_size is None:
#             hidden_size = [32, 32]
#         self.net = nn.Sequential(
#             nn.Linear(num_inputs, hidden_size[0]),
#             nn.InstanceNorm1d(hidden_size[0], affine=True),
#             nn.LeakyReLU(),
#         )
#         for i in range(1, len(hidden_size)):
#             self.net.append(nn.Linear(hidden_size[i - 1], hidden_size[i])),
#             self.net.append(nn.InstanceNorm1d(hidden_size[i], affine=True)),
#             self.net.append(nn.LeakyReLU())
#         self.net.append(nn.Linear(hidden_size[-1], num_actions))
#
#     def forward(self, x):
#         action_prob = self.net(x)
#         return action_prob
#
#
# class DQN(nn.Module):
#     """docstring for DQN"""
#
#     def __init__(
#             self,
#             input_dims: int,
#             num_actions: int,
#             hidden_size=None,
#             memory_size: int = 16384,
#             lr: float = 1e-4,
#     ):
#         super(DQN, self).__init__()
#         self.eval_net, self.target_net = Net(input_dims, num_actions, hidden_size), Net(input_dims, num_actions, hidden_size)
#
#         self.learn_step_counter = 0
#         self.memory_counter = 0
#         self.memory = np.zeros((memory_size, input_dims * 2 + 2))
#         # why the NUM_STATE*2 +2
#         # When we store the memory, we put the state, action, reward and next_state in the memory
#         # here reward and action is a number, state is a ndarray
#         self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr)
#         self.loss_func = nn.MSELoss()
#
#     def choose_action(self, state):
#         state = torch.unsqueeze(torch.FloatTensor(state), 0)  # get a 1D array
#         if np.random.randn() <= EPISILO:  # greedy policy
#             action_value = self.eval_net.forward(state)
#             action = torch.max(action_value, 1)[1].data.numpy()
#             action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
#         else:  # random policy
#             action = np.random.randint(0, NUM_ACTIONS)
#             action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
#         return action
#
#     def store_transition(self, state, action, reward, next_state):
#         transition = np.hstack((state, [action, reward], next_state))
#         index = self.memory_counter % MEMORY_CAPACITY
#         self.memory[index, :] = transition
#         self.memory_counter += 1
#
#     def learn(self):
#
#         # update the parameters
#         if self.learn_step_counter % Q_NETWORK_ITERATION == 0:
#             self.target_net.load_state_dict(self.eval_net.state_dict())
#         self.learn_step_counter += 1
#
#         # sample batch from memory
#         sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
#         batch_memory = self.memory[sample_index, :]
#         batch_state = torch.FloatTensor(batch_memory[:, :NUM_STATES])
#         batch_action = torch.LongTensor(batch_memory[:, NUM_STATES:NUM_STATES + 1].astype(int))
#         batch_reward = torch.FloatTensor(batch_memory[:, NUM_STATES + 1:NUM_STATES + 2])
#         batch_next_state = torch.FloatTensor(batch_memory[:, -NUM_STATES:])
#
#         # q_eval
#         q_eval = self.eval_net(batch_state).gather(1, batch_action)
#         q_next = self.target_net(batch_next_state).detach()
#         q_target = batch_reward + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
#         loss = self.loss_func(q_eval, q_target)
#
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()
#
#
# def reward_func(env, x, x_dot, theta, theta_dot):
#     r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.5
#     r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
#     reward = r1 + r2
#     return reward
#
#
# def main():
#     dqn = DQN()
#     episodes = 400
#     print("Collecting Experience....")
#     reward_list = []
#     plt.ion()
#     fig, ax = plt.subplots()
#     for i in range(episodes):
#         state = env.reset()
#         ep_reward = 0
#         while True:
#             env.render()
#             action = dqn.choose_action(state)
#             next_state, _, done, info = env.step(action)
#             x, x_dot, theta, theta_dot = next_state
#             reward = reward_func(env, x, x_dot, theta, theta_dot)
#
#             dqn.store_transition(state, action, reward, next_state)
#             ep_reward += reward
#
#             if dqn.memory_counter >= MEMORY_CAPACITY:
#                 dqn.learn()
#                 if done:
#                     print("episode: {} , the episode reward is {}".format(i, round(ep_reward, 3)))
#             if done:
#                 break
#             state = next_state
#         r = copy.copy(reward)
#         reward_list.append(r)
#         ax.set_xlim(0, 300)
#         # ax.cla()
#         ax.plot(reward_list, 'g-', label='total_loss')
#         plt.pause(0.001)


class TransitionTable:
    def __init__(self, state_discretizer: Discretizer, action_discretizer: Discretizer, rough_reward_resolution: int = -1):
        self.state_discretizer = state_discretizer
        self.action_discretizer = action_discretizer
        self.transition_table: Dict[int, Dict[int, Dict[int, Dict[float, int]]]] \
            = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0))))  # {state: {action: {next_state: {reward: count}}}
        self.neighbour_dict = defaultdict(lambda: set())
        self.forward_dict = defaultdict(lambda: defaultdict(lambda: set()))
        self.inverse_dict = defaultdict(lambda: defaultdict(lambda: set()))

        # todo: currently will not be saved!
        self.state_count = defaultdict(lambda: 0)
        self.state_action_count = defaultdict(lambda: defaultdict(lambda: 0))
        self.transition_prob_table: Dict[int, Dict[int, Dict[int, float]]] \
            = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))  # {state: {action: {next_state: rate}}
        self.done_set = set()
        self.start_set = set()
        self.reward_set_dict = defaultdict(lambda: set())
        self.rough_reward_set_dict = defaultdict(lambda: set())
        self.rough_reward_resolution = rough_reward_resolution

        self.mdp_graph: DiGraph = self.make_mdp_graph()

    def print_transition_table_info(self):
        print("Transition Table Information:")
        print(f"Total num transition pairs: {len(self.forward_dict)}.")
        print(f"Collected initial states: {len(self.start_set)}.")
        print(f"Collected termination states: {len(self.done_set)}.")
        print(f"Collected rewards:")
        total_reward_count = 0
        for reward, reward_set in self.rough_reward_set_dict.items():
            total_reward_count += len(reward_set)
        for reward, reward_set in sorted(self.rough_reward_set_dict.items(), key=lambda x: x[0]):
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

        self.reward_set_dict[reward].add(encoded_next_state)
        if self.rough_reward_resolution > 0:
            self.rough_reward_set_dict[round(reward / self.rough_reward_resolution) * self.rough_reward_resolution].add(encoded_next_state)
        elif self.rough_reward_resolution < 0:
            self.rough_reward_set_dict[round(reward, abs(self.rough_reward_resolution))].add(encoded_next_state)
        else:
            self.rough_reward_set_dict[reward].add(encoded_next_state)

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
        a = self.transition_table[encoded_state][encoded_action].keys()
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

    def add_start_state(self, encoded_state: int):
        self.start_set.add(encoded_state)

    def get_start_set(self) -> set[int]:
        return self.start_set


class TransitionalTableEnv(TransitionTable, gym.Env):
    def __init__(
            self,
            state_discretizer: Discretizer,
            action_discretizer: Discretizer,
            max_steps: int = 500,
            rough_reward_resolution: int = -1,
    ):
        TransitionTable.__init__(self, state_discretizer, action_discretizer, rough_reward_resolution=rough_reward_resolution)
        gym.Env.__init__(self)

        # State space
        self.state_discretizer = state_discretizer
        state_size = state_discretizer.count_possible_combinations()
        self.observation_space = spaces.Discrete(state_size)

        # Action space
        self.action_discretizer = action_discretizer
        action_size = action_discretizer.count_possible_combinations()
        self.action_space = spaces.Discrete(action_size)

        self.max_steps = max_steps
        self.step_count = 0
        self.current_state = None

    def reset(self, seed=None, options=None, init_state_encode: int = None, init_strategy: str = "real_start_states"):
        super().reset(seed=seed)
        self.step_count = 0
        if init_state_encode is None or init_state_encode in self.done_set:
            if init_state_encode in self.done_set:
                print("Warning: Starting from a done state, reset to a random state.")
            if init_strategy == "random":
                init_state_encode = random.choice(tuple(self.forward_dict.keys()))
            elif init_strategy == "real_start_states":
                init_state_encode = random.choice(tuple(self.start_set))
            else:
                raise ValueError(f"Init strategy not supported: {init_strategy}.")
            self.current_state = init_state_encode
        return self.current_state, {}

    def step(self, action: int, transition_strategy: str = "weighted", unknown_reward: float = None):
        if unknown_reward is None:
            r_sum = 0.0
            total_rewards = 0
            for r in self.reward_set_dict.keys():
                r_sum += r * len(self.reward_set_dict[r])
                total_rewards += len(self.reward_set_dict[r])
            unknown_reward = r_sum / total_rewards
        encoded_state = self.current_state
        encoded_action = action
        transition_state_avg_reward_and_prob \
            = self.get_transition_state_avg_reward_and_prob(encoded_state, encoded_action)
        if len(transition_state_avg_reward_and_prob) == 0:
            return encoded_state, unknown_reward, True, False, {"current_step": self.step_count}
        if transition_strategy == "weighted":
            a = tuple(transition_state_avg_reward_and_prob.keys())
            b = [v[1] for v in transition_state_avg_reward_and_prob.values()]
            encoded_next_state = random.choices(
                tuple(transition_state_avg_reward_and_prob.keys()),
                weights=[v[1] for v in transition_state_avg_reward_and_prob.values()],
                k=1,
            )[0]
        elif transition_strategy == "random":
            encoded_next_state = random.choice(tuple(transition_state_avg_reward_and_prob.keys()))
        elif transition_strategy == "inverse_weighted":
            probabilities = [v[1] for v in transition_state_avg_reward_and_prob.values()]
            total_weight = sum(probabilities)
            inverse_weights = [total_weight - p for p in probabilities]
            encoded_next_state = random.choices(
                tuple(transition_state_avg_reward_and_prob.keys()),
                weights=inverse_weights,
                k=1,
            )[0]
        else:
            raise ValueError(f"Transition strategy not supported: {transition_strategy}.")
        reward = transition_state_avg_reward_and_prob[encoded_next_state][0]
        self.step_count += 1

        terminated = encoded_next_state in self.done_set
        truncated = self.step_count >= self.max_steps
        self.current_state = encoded_next_state

        info = {"current_step": self.step_count}
        return encoded_next_state, reward, terminated, truncated, info


class QCutTransitionalTableEnv(TransitionalTableEnv):
    def __init__(
            self,
            state_discretizer: Discretizer,
            action_discretizer: Discretizer,
            max_steps: int = 500,
            rough_reward_resolution: int = -1,
    ):
        TransitionalTableEnv.__init__(self, state_discretizer, action_discretizer, max_steps, rough_reward_resolution)
        self.landmark_states, self.landmark_start_states, self.targets = None, None, None
        self.no_target_error_printed_times = 10

    def find_nearest_nodes_and_subgraph(self, start_node, n, weighted=True, direction='both') -> Tuple[List[Tuple[int, float]], DiGraph]:
        """
        Find the nearest n nodes from the starting node, searching in the specified direction.

        :param start_node: Starting node for the search
        :param n: Number of nearest nodes to find
        :param weighted: Whether to consider weights (True: use 'prob' as weights; False: unweighted search)
        :param direction: Direction of search ('forward', 'backward', or 'both')
        :return: List of the nearest n nodes (sorted by distance) and the subgraph containing these nodes
        """
        G = self.mdp_graph

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

    # def get_landmark_states(
    #         self,
    #         num_targets: int,
    #         min_cut_max_flow_search_space: int,
    #         nums_in_layers: List[int],
    #         init_state_reward_prob_below_threshold: float = 0.2,
    # ):
    #     beginners = set()
    #     total_reward_count = 0
    #     for reward, reward_set in self.reward_set_dict.items():
    #         total_reward_count += len(reward_set)
    #     for reward in self.reward_set_dict.keys():
    #         if len(self.reward_set_dict[reward]) / total_reward_count < init_state_reward_prob_below_threshold:
    #             for state in self.reward_set_dict[reward]:
    #                 beginners.add(state)
    #
    #     if len(beginners) == 0:
    #         print("No beginner states for Q-Cut search found.")
    #         return
    #
    #     for level, num_in_layer in enumerate(nums_in_layers):
    #         if level == 0:
    #             pass

    def get_landmark_states(
            self,
            num_targets: int,
            min_cut_max_flow_search_space: int,
            q_cut_space: int,
            weighted_search: bool = True,
            init_state_reward_prob_below_threshold: float = 0.2,
            quality_value_threshold: float = 1.0,
            take_done_states_as_targets: bool = False,
    ):
        landmark_states = set()
        landmark_start_states = set()

        targets = set()
        total_reward_count = 0
        for reward, reward_set in self.rough_reward_set_dict.items():
            total_reward_count += len(reward_set)
        for reward in self.rough_reward_set_dict.keys():
            if len(self.rough_reward_set_dict[reward]) / total_reward_count < init_state_reward_prob_below_threshold:
                for state in self.rough_reward_set_dict[reward]:
                    targets.add(state)

        if take_done_states_as_targets:
            for state in self.done_set:
                targets.add(state)

        if len(targets) == 0:
            if self.no_target_error_printed_times <= 0:
                print("No target states for Q-Cut search found.")
                self.no_target_error_printed_times = 10
            else:
                self.no_target_error_printed_times -= 1
            return [], [], []

        selected_targets = targets if len(targets) <= num_targets else random.sample(targets, num_targets)

        for target in selected_targets:
            nearest_nodes, subgraph = self.find_nearest_nodes_and_subgraph(
                target, min_cut_max_flow_search_space, weighted=weighted_search, direction='backward'
            )
            start_node = random.choice(nearest_nodes[-len(nearest_nodes)//10:])[0]
            cut_value, reachable, non_reachable, edges_in_cut, quality_factor = self.find_min_cut_max_flow(
                subgraph, start_node, target, invert_weights=False,
            )
            if quality_factor > quality_value_threshold:
                for edge in edges_in_cut:
                    for node in edge:
                        landmark_states.add(node)

        for node in landmark_states:
            q_cut_nodes, q_cut_subgraph = self.find_nearest_nodes_and_subgraph(
                node, q_cut_space, weighted=False, direction='both'
            )
            for n in q_cut_nodes:
                landmark_start_states.add(n[0])

        return landmark_states, landmark_start_states, targets

    def reset(
            self,
            seed=None,
            options=None,
            init_state_encode: int = None,
            init_strategy: str = "real_start_states",
            num_targets: int = 8,
            min_cut_max_flow_search_space: int = 128,
            q_cut_space: int = 4,
            weighted_search: bool = True,
            init_state_reward_prob_below_threshold: float = 0.2,
            quality_value_threshold: float = 1.0,
            re_init_landmarks: bool = False,
            return_actual_strategy: bool = False,
            take_done_states_as_targets: bool = False,
            do_print: bool = True,
    ):
        super().reset(seed=seed)
        self.step_count = 0

        if re_init_landmarks or self.landmark_states is None or self.landmark_start_states is None or self.targets is None:
            self.make_mdp_graph(use_encoded_states=True)
            self.landmark_states, self.landmark_start_states, self.targets = self.get_landmark_states(
                num_targets=num_targets,
                min_cut_max_flow_search_space=min_cut_max_flow_search_space,
                q_cut_space=q_cut_space,
                weighted_search=weighted_search,
                init_state_reward_prob_below_threshold=init_state_reward_prob_below_threshold,
                quality_value_threshold=quality_value_threshold,
                take_done_states_as_targets=take_done_states_as_targets,
            )
            if do_print:
                print(f"Initialized: {len(self.landmark_states)} landmark states; {len(self.landmark_start_states)} start states.")
                if take_done_states_as_targets:
                    print("Done states are also used as targets for landmark generation.")

        if init_state_encode is None or init_state_encode in self.done_set:
            if init_state_encode in self.done_set:
                print("Warning: Starting from a done state, reset to a random state.")
            if init_strategy == "random":
                init_state_encode = random.choice(tuple(self.forward_dict.keys()))
            elif init_strategy == "real_start_states":
                init_state_encode = random.choice(tuple(self.start_set))
            elif init_strategy == "landmarks":
                if len(self.landmark_states) == 0:
                    init_state_encode = random.choice(tuple(self.forward_dict.keys()))
                    init_strategy = "random"
                else:
                    init_state_encode = random.choice(tuple(self.landmark_start_states))
            else:
                raise ValueError(f"Init strategy not supported: {init_strategy}.")
            self.current_state = init_state_encode
            if return_actual_strategy:
                return self.current_state, {}, init_strategy
        return self.current_state, {}

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


class TabularDynaQAgent:
    def __init__(
            self,
            state_discretizer: Discretizer,
            action_discretizer: Discretizer,
            bonus_decay=0.9,
            max_steps: int = 500,
            rough_reward_resolution: int = -1,
    ):
        self.state_discretizer = state_discretizer
        self.action_discretizer = action_discretizer
        self.transition_table_env = TransitionalTableEnv(
            state_discretizer, action_discretizer, max_steps=max_steps, rough_reward_resolution=rough_reward_resolution,
        )
        self.q_table_agent = TabularQAgent(self.state_discretizer, self.action_discretizer)
        self.exploration_agent = TabularQAgent(self.state_discretizer, self.action_discretizer)
        self.exploration_agent.q_table = defaultdict(lambda: 1.0)
        self.bonus_states = defaultdict(lambda: 1.0)
        self.bonus_decay = bonus_decay

    def print_agent_info(self):
        self.q_table_agent.print_q_table_info()
        self.exploration_agent.print_q_table_info()
        self.transition_table_env.print_transition_table_info()

    def save_agent(self, file_path: str = None) -> tuple[DataFrame, DataFrame, DataFrame]:
        if file_path:
            q_table_file_path = file_path.split(".csv")[0] + "_q_table.csv"
            exploration_agent_q_table_file_path = file_path.split(".csv")[0] + "_exploration_agent_q_table.csv"
            transition_table_file_path = file_path.split(".csv")[0] + "_transition_table.csv"
        else:
            q_table_file_path = None
            exploration_agent_q_table_file_path = None
            transition_table_file_path = None
        q_table_df = self.q_table_agent.save_q_table(file_path=q_table_file_path)
        exploration_agent_q_table_df = self.exploration_agent.save_q_table(file_path=exploration_agent_q_table_file_path)
        transition_table_df = self.transition_table_env.save_transition_table(file_path=transition_table_file_path)
        return q_table_df, transition_table_df, exploration_agent_q_table_df

    def load_agent(self, file_path: str = None, dataframes: tuple[DataFrame, DataFrame, DataFrame] = (None, None, None)):
        if file_path:
            q_table_file_path = file_path.split(".csv")[0] + "_q_table.csv"
            exploration_agent_q_table_file_path = file_path.split(".csv")[0] + "_exploration_agent_q_table.csv"
            transition_table_file_path = file_path.split(".csv")[0] + "_transition_table.csv"
        else:
            q_table_file_path = None
            exploration_agent_q_table_file_path = None
            transition_table_file_path = None
        self.q_table_agent.load_q_table(file_path=q_table_file_path, df=dataframes[0])
        self.exploration_agent.load_q_table(file_path=exploration_agent_q_table_file_path, df=dataframes[0])
        self.transition_table_env.load_transition_table(
            file_path=transition_table_file_path, transition_table_df=dataframes[1]
        )

    def get_action_probabilities(self, state: np.ndarray, strategy: str = "greedy",
                                 temperature: float = 1.0) -> np.ndarray:
        if strategy == "softmax" or strategy == "greedy":
            return self.q_table_agent.get_action_probabilities(state, strategy=strategy, temperature=temperature)
        elif strategy == "explore_softmax":
            return self.exploration_agent.get_action_probabilities(state, strategy="softmax", temperature=temperature)
        elif strategy == "explore_greedy":
            return self.exploration_agent.get_action_probabilities(state, strategy="greedy", temperature=temperature)
        elif strategy == "weighted":
            encoded_state = self.state_discretizer.encode_indices([*self.state_discretizer.discretize(state)[1]])
            state_action_counts = self.transition_table_env.get_state_action_counts(encoded_state)
            sum_counts = sum(state_action_counts.values())
            if sum_counts == 0:
                return np.ones(len(self.q_table_agent.all_actions_encoded)) / len(self.q_table_agent.all_actions_encoded)
            return np.array([state_action_counts[a]/sum_counts for a in self.q_table_agent.all_actions_encoded])
        elif strategy == "random":
            return np.ones(len(self.q_table_agent.all_actions_encoded)) / len(self.q_table_agent.all_actions_encoded)
        else:
            raise ValueError(f"Select strategy not supported: {strategy}.")

    def choose_action(self, state: np.ndarray, strategy: str = "greedy", temperature: float = 1.0) -> np.ndarray:
        action_probabilities = self.get_action_probabilities(state, strategy=strategy, temperature=temperature)
        action = random.choices(self.q_table_agent.all_actions_encoded, weights=action_probabilities, k=1)[0]
        return np.array(self.action_discretizer.indices_to_midpoints(self.action_discretizer.decode_indices(action)))

    def choose_action_encoded(self, state: np.ndarray, strategy: str = "greedy", temperature: float = 1.0) -> np.ndarray:
        action_probabilities = self.get_action_probabilities(state, strategy=strategy, temperature=temperature)
        action = random.choices(self.q_table_agent.all_actions_encoded, weights=action_probabilities, k=1)[0]
        return action

    def update_from_env(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool,
               alpha: float = 0.1, gamma: float = 0.99, update_policy=True):
        if update_policy:
            self.q_table_agent.update(state, action, reward, next_state, done, alpha=alpha, gamma=gamma)
        self.transition_table_env.update(state, action, reward, next_state, done)
        encoded_next_state = self.state_discretizer.encode_indices(
            list(self.state_discretizer.discretize(next_state)[1]))
        reward_bonus = self.bonus_states[encoded_next_state]
        self.bonus_states[encoded_next_state] *= self.bonus_decay
        self.exploration_agent.update(state, action, reward_bonus, next_state, done, alpha=alpha, gamma=gamma)

    def update_from_transition_table(
            self,
            steps: int,
            epsilon: float,
            strategy: str = "greedy",
            alpha: float = 0.1,
            gamma: float = 0.99,
            transition_strategy: str = "weighted",
            init_strategy: str = "real_start_states",
            train_exploration_agent: bool = False,
            unknown_reward: float = None
    ):
        # Initialize variables
        num_episodes = 1
        num_truncated = 0
        num_terminated = 0
        sum_episode_rewards = 0

        old_truncate_steps = self.transition_table_env.max_steps
        if train_exploration_agent:
            self.transition_table_env.max_steps = np.inf
            self.exploration_agent.reset_q_table()

        agent = self.q_table_agent if not train_exploration_agent else self.exploration_agent
        unknown_reward = 0.0 if not train_exploration_agent else unknown_reward

        print(f"Starting for {steps} steps using transition table: ")
        if train_exploration_agent:
            print(f"Training unknown_reward agent with unknown_reward value: {unknown_reward}.")
        self.transition_table_env.print_transition_table_info()

        # Reset the environment and get the initial state
        state_encoded, info = self.transition_table_env.reset(init_strategy=init_strategy)

        # Initialize the progress bar
        progress_bar = tqdm.tqdm(total=steps, desc="Inner Training", unit="step")

        for step in range(steps):
            # Decode and compute the midpoint of the current state
            state = self.state_discretizer.indices_to_midpoints(self.state_discretizer.decode_indices(state_encoded))

            # Select action based on epsilon-greedy strategy
            if np.random.random() < epsilon:
                action_encoded = random.choice(agent.all_actions_encoded)
            else:
                action_encoded = self.choose_action_encoded(state, strategy=strategy, temperature=1.0)

            # Take a step in the environment
            next_state_encoded, reward, terminated, truncated, info = self.transition_table_env.step(
                action_encoded, transition_strategy, unknown_reward=unknown_reward,
            )
            if train_exploration_agent:
                if reward != unknown_reward:
                    reward = 0.0
                else:
                    terminated = True
            else:
                # if reward >= unknown_reward:
                #     reward += 1.0
                pass

            # Decode and compute the midpoint of the action and next state
            action = self.action_discretizer.indices_to_midpoints(
                self.action_discretizer.decode_indices(action_encoded))
            next_state = self.state_discretizer.indices_to_midpoints(
                self.state_discretizer.decode_indices(next_state_encoded))

            # Update Q-table using the chosen action
            agent.update(state, action, reward, next_state, terminated, alpha=alpha, gamma=gamma)

            # Update the current state
            state_encoded = next_state_encoded

            # Update counters and rewards
            if terminated:
                num_terminated += 1
            if truncated:
                num_truncated += 1
            sum_episode_rewards += reward

            # Reset the environment if an episode ends
            if terminated or truncated:
                num_episodes += 1
                state_encoded, info = self.transition_table_env.reset(init_strategy=init_strategy)

            # Update the progress bar
            progress_bar.set_postfix({
                "Episodes": num_episodes,
                "Terminated": num_terminated,
                "Truncated": num_truncated,
                "Reward (last)": reward,
                "Avg Episode Reward": sum_episode_rewards / num_episodes
            })
            progress_bar.update(1)

        # Close the progress bar
        progress_bar.close()

        print(f"Trained {num_episodes-1} episodes, including {num_truncated} truncated, {num_terminated} terminated.")
        print(f"Average episode reward: {sum_episode_rewards / num_episodes}.")
        self.transition_table_env.max_steps = old_truncate_steps


class QCutTabularDynaQAgent(TabularDynaQAgent):
    def __init__(
            self,
            state_discretizer: Discretizer,
            action_discretizer: Discretizer,
            bonus_decay=0.9,
            max_steps: int = 500,
            rough_reward_resolution: int = -1,):
        super().__init__(state_discretizer, action_discretizer, bonus_decay, max_steps, rough_reward_resolution)
        self.transition_table_env = QCutTransitionalTableEnv(state_discretizer, action_discretizer, max_steps, rough_reward_resolution)

    def update_from_transition_table(
            self,
            steps: int,
            epsilon: float,
            strategy: str = "greedy",
            alpha: float = 0.1,
            gamma: float = 0.99,
            transition_strategy: str = "weighted",
            init_strategy_distribution: Tuple[float] = (0.33, 0.33, 0.33),
            train_exploration_agent: bool = False,
            unknown_reward: float = None,
            num_targets: int = 8,
            min_cut_max_flow_search_space: int = 128,
            q_cut_space: int = 4,
            weighted_search: bool = True,
            init_state_reward_prob_below_threshold: float = 0.01,
            quality_value_threshold: float = 1.0,
            take_done_states_as_targets: bool = False,
            use_task_bar: bool = True,
            do_print: bool = True,
    ):
        # Initialize variables
        num_episodes = 1
        num_truncated = 0
        num_terminated = 0
        sum_episode_rewards = 0

        init_strategies = ["real_start_states", "random", "landmarks"]
        init_strategy = random.choices(init_strategies, weights=init_strategy_distribution, k=1)[0]
        strategy_counts = {s: 0 for s in init_strategies}
        strategy_step_counts = {s: 0 for s in init_strategies}

        old_truncate_steps = self.transition_table_env.max_steps
        if train_exploration_agent:
            self.transition_table_env.max_steps = np.inf
            self.exploration_agent.reset_q_table()

        agent = self.q_table_agent if not train_exploration_agent else self.exploration_agent
        unknown_reward = 0.0 if not train_exploration_agent else unknown_reward

        if do_print:
            print(f"Starting for {steps} steps using transition table: ")
            if train_exploration_agent:
                print(f"Training unknown_reward agent with unknown_reward value: {unknown_reward}.")
            self.transition_table_env.print_transition_table_info()

        # Reset the environment and get the initial state
        state_encoded, info, actual_strategy = self.transition_table_env.reset(
            init_strategy=init_strategy,
            num_targets=num_targets,
            min_cut_max_flow_search_space=min_cut_max_flow_search_space,
            q_cut_space=q_cut_space,
            weighted_search=weighted_search,
            init_state_reward_prob_below_threshold=init_state_reward_prob_below_threshold,
            quality_value_threshold=quality_value_threshold,
            re_init_landmarks=True,
            return_actual_strategy=True,
            take_done_states_as_targets=take_done_states_as_targets,
            do_print=do_print,
        )
        strategy_counts[actual_strategy] += 1

        # Initialize the progress bar
        progress_bar = tqdm.tqdm(total=steps, desc="Inner Training", unit="step") if use_task_bar else None
        episode_step_count = 0
        for step in range(steps):
            # Decode and compute the midpoint of the current state
            state = self.state_discretizer.indices_to_midpoints(self.state_discretizer.decode_indices(state_encoded))

            # Select action based on epsilon-greedy strategy
            if np.random.random() < epsilon:
                action_encoded = random.choice(agent.all_actions_encoded)
            else:
                action_encoded = self.choose_action_encoded(state, strategy=strategy, temperature=1.0)

            # Take a step in the environment
            next_state_encoded, reward, terminated, truncated, info = self.transition_table_env.step(
                action_encoded, transition_strategy, unknown_reward=unknown_reward,
            )
            if train_exploration_agent:
                if reward != unknown_reward:
                    reward = 0.0
                else:
                    terminated = True
            else:
                # if reward >= unknown_reward:
                #     reward += 1.0
                pass

            # Decode and compute the midpoint of the action and next state
            action = self.action_discretizer.indices_to_midpoints(
                self.action_discretizer.decode_indices(action_encoded))
            next_state = self.state_discretizer.indices_to_midpoints(
                self.state_discretizer.decode_indices(next_state_encoded))

            # Update Q-table using the chosen action
            agent.update(state, action, reward, next_state, terminated, alpha=alpha, gamma=gamma)

            # Update the current state
            state_encoded = next_state_encoded

            # Update counters and rewards
            if terminated:
                num_terminated += 1
            if truncated:
                num_truncated += 1
            sum_episode_rewards += reward

            # Reset the environment if an episode ends
            if terminated or truncated:
                num_episodes += 1
                strategy_step_counts[init_strategy] += episode_step_count
                strategy_selection_dict = {}
                for i, s in enumerate(init_strategies):
                    if init_strategy_distribution[i] != 0:
                        strategy_selection_dict[s] = strategy_step_counts[s] / init_strategy_distribution[i]
                    else:
                        strategy_selection_dict[s] = np.inf
                init_strategy = min(strategy_selection_dict, key=strategy_selection_dict.get)
                state_encoded, info, actual_strategy = self.transition_table_env.reset(
                    init_strategy=init_strategy,
                    num_targets=num_targets,
                    min_cut_max_flow_search_space=min_cut_max_flow_search_space,
                    q_cut_space=q_cut_space,
                    weighted_search=weighted_search,
                    init_state_reward_prob_below_threshold=init_state_reward_prob_below_threshold,
                    quality_value_threshold=quality_value_threshold,
                    re_init_landmarks=False,
                    return_actual_strategy=True,
                    do_print=do_print,
                )
                strategy_counts[actual_strategy] += 1
                episode_step_count = 0

            # Update the progress bar
            if use_task_bar:
                progress_bar.set_postfix({
                    "Episodes": num_episodes,
                    "Terminated": num_terminated,
                    "Truncated": num_truncated,
                    "Rwd (last)": reward,
                    "Avg Episode Rwd": sum_episode_rewards / num_episodes,
                    "Real Starts": strategy_counts["real_start_states"],
                    "Landmarks": strategy_counts["landmarks"],
                    "Random": strategy_counts["random"],
                })
                progress_bar.update(1)
            episode_step_count += 1

        # Close the progress bar
        if use_task_bar:
            progress_bar.close()

        if do_print:
            print(f"Trained {num_episodes-1} episodes, including {num_truncated} truncated, {num_terminated} terminated.")
            print(f"Real starts: {strategy_counts['real_start_states']}, Landmarks: {strategy_counts['landmarks']}, Random: {strategy_counts['random']}.")
            print(f"Average episode reward: {sum_episode_rewards / num_episodes}.")
        self.transition_table_env.max_steps = old_truncate_steps


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
