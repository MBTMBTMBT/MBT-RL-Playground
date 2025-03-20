import json
import os
import tempfile
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from itertools import product
from typing import List, Tuple, Optional, Union
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import scipy
from gymnasium import spaces
import gymnasium as gym
import tqdm
from gymnasium.core import Env
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv


def check_for_correct_spaces(env: Union[Env, "VecEnv"], observation_space: spaces.Space, action_space: spaces.Space) -> None:
    if observation_space != env.observation_space:
        raise ValueError(f"Observation spaces do not match: {observation_space} != {env.observation_space}")
    if action_space != env.action_space:
        raise ValueError(f"Action spaces do not match: {action_space} != {env.action_space}")


def generate_discretizer_params_from_space(
        space: spaces.Space,
        default_num_buckets_per_dim: int = 15
) -> Tuple[List[Tuple[float, float]], List[int]]:
    """
    Automatically generate `ranges` and `num_buckets` from a Gym space.

    :param space: Gymnasium space (Box, Discrete, MultiDiscrete)
    :param default_num_buckets_per_dim: Default number of buckets for Box spaces.
    :return: (ranges, num_buckets) lists for initializing the Discretizer.
    """
    ranges = []
    num_buckets = []

    # Handle Box space
    if isinstance(space, spaces.Box):
        low = space.low
        high = space.high

        # Sanity check on box shape
        assert low.shape == high.shape, "Box low and high must be the same shape."

        for i in range(low.size):
            l = low.flat[i]
            h = high.flat[i]

            # Check for infinite bounds
            if np.isinf(l) or np.isinf(h):
                raise ValueError(
                    f"Dimension {i} in Box space has infinite bounds: "
                    f"low={l}, high={h}. You must specify finite limits."
                )

            # Define the range
            ranges.append((float(l), float(h)))

            # Default number of buckets for continuous Box spaces
            num_buckets.append(default_num_buckets_per_dim)

    # Handle Discrete space
    elif isinstance(space, spaces.Discrete):
        # Range is [0, n-1]
        ranges.append((0, space.n - 1))

        # Use 0 to indicate integer discretization within range (as per your Discretizer)
        num_buckets.append(0)

    # Handle MultiDiscrete space
    elif isinstance(space, spaces.MultiDiscrete):
        for i, n in enumerate(space.nvec):
            ranges.append((0, n - 1))
            num_buckets.append(0)

    # Handle MultiBinary space
    elif isinstance(space, spaces.MultiBinary):
        for i in range(space.n):
            ranges.append((0, 1))
            num_buckets.append(0)

    else:
        raise NotImplementedError(f"Space type {type(space)} is not supported for discretization.")

    return ranges, num_buckets

def merge_params(
        user_values: Optional[List], auto_values: List
) -> List:
    """
    Merge user-specified parameters with auto-generated ones.
    If user provides a partial list, missing entries will be filled from auto_values.

    :param user_values: Partial or full user-provided list, or None.
    :param auto_values: Auto-generated list (default values).
    :return: Merged list with priority given to user_values.
    """
    if user_values is None:
        return auto_values

    merged = []
    for i in range(len(auto_values)):
        # Use user value if it exists, otherwise fallback to auto value
        if i < len(user_values) and user_values[i] is not None:
            merged.append(user_values[i])
        else:
            merged.append(auto_values[i])

    return merged


class Agent(ABC):
    """
    Abstract base Agent class for tabular methods, similar to Stable-Baselines3 structure.
    Handles both single Gym environments and VecEnvs.
    """

    def __init__(self, env: Union[Env, VecEnv, None]):
        """
        Initialize the Agent.
        :param env: A single environment or a vectorized environment.
        """
        # Automatically wrap a single environment into a DummyVecEnv
        if env and not isinstance(env, VecEnv):
            env = DummyVecEnv([lambda: env])

        self.env: Optional[VecEnv] = env
        self.observation_space: spaces.Space = self.env.observation_space
        self.action_space: spaces.Space = self.env.action_space

    def get_env(self) -> Optional[VecEnv]:
        """
        Return the current environment.
        :return: The current VecEnv environment.
        """
        return self.env

    def set_env(self, env: Union[Env, VecEnv]) -> None:
        """
        Set a new environment for the agent. Re-check spaces.
        :param env: A new environment (single or VecEnv).
        """
        # Auto-wrap if it's not a VecEnv
        if not isinstance(env, VecEnv):
            env = DummyVecEnv([lambda: env])

        # Validate that the new env spaces match the old ones
        check_for_correct_spaces(env, self.observation_space, self.action_space)

        # Set the new env
        self.env = env

    @abstractmethod
    def learn(
            self,
            total_timesteps: int,
            callback: Optional[callable] = None,
            reset_num_timesteps: bool = True,
            progress_bar: bool = False,
    ) -> "Agent":
        """
        Abstract learn method. Must be implemented by subclasses.
        :param total_timesteps: Number of timesteps for training.
        :param callback: Optional callback function during training.
        :param reset_num_timesteps: whether or not to reset the current timestep number (used in logging)
        :param progress_bar: Display a progress bar using tqdm and rich.
        :return: Returns self.
        """
        pass

    @abstractmethod
    def predict(self, observation, deterministic: bool = True):
        """
        Predict an action given an observation.
        :param observation: Observation from the environment.
        :param deterministic: Whether to return deterministic actions.
        :return: Action(s) to take.
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str, env: Optional[GymEnv] = None, print_system_info: bool = False,):
        """
        Abstract class method to load an agent from a file.

        :param path: Path to the saved agent.
        :param env: Optional environment to load the agent. If None, it will try to load the agent without it.
        :param print_system_info: Whether to print system info when loading.
        :return: An instance of the Agent.
        """
        pass

    @abstractmethod
    def save(self, path: str):
        """
        Save the model to the given path.
        Args:
            path: path to save the model to.
        Returns: None
        """
        pass


class Discretizer:
    def __init__(
        self,
        ranges: List[Tuple[float, float]],
        num_buckets: List[int],
        normal_params: List[Optional[Tuple[float, float]]] = None,
    ):
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
        assert len(ranges) == len(
            num_buckets
        ), "Ranges and num_buckets must have the same length."
        if normal_params:
            assert len(normal_params) == len(
                num_buckets
            ), "normal_params must match the length of num_buckets."

        self.ranges: List[Tuple[float, float]] = ranges
        self.input_num_buckets: List[int] = num_buckets
        self.num_buckets: List[int] = [
            int(np.floor(max_val) - np.ceil(min_val) + 1) if buckets == 0 else buckets
            for (min_val, max_val), buckets in zip(ranges, num_buckets)
        ]
        self.normal_params: List[Optional[Tuple[float, float]]] = (
            normal_params if normal_params else [None] * len(num_buckets)
        )
        self.bucket_midpoints: List[List[float]] = []

        for i, ((min_val, max_val), buckets, normal_param) in enumerate(
            zip(ranges, num_buckets, self.normal_params)
        ):
            if buckets == -1:
                self.bucket_midpoints.append([])
            elif buckets == 0:
                # Discretize into integers within range
                midpoints = list(
                    range(int(np.ceil(min_val)), int(np.floor(max_val)) + 1)
                )
                self.bucket_midpoints.append(midpoints)
            elif buckets == 1:
                midpoint = [(min_val + max_val) / 2]
                self.bucket_midpoints.append(midpoint)
            else:
                if normal_param:
                    mean, std = normal_param
                    # Restrict edges to a finite range if necessary
                    edges = [
                        scipy.stats.norm.ppf(
                            min(max((j / buckets), 1e-6), 1 - 1e-6), loc=mean, scale=std
                        )
                        for j in range(buckets + 1)
                    ]
                    midpoints = [
                        round((edges[j] + edges[j + 1]) / 2, 6) for j in range(buckets)
                    ]
                else:
                    step = (max_val - min_val) / buckets
                    midpoints = [
                        round(min_val + (i + 0.5) * step, 6) for i in range(buckets)
                    ]
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
        assert len(vector) == len(
            self.ranges
        ), "Input vector must have the same length as ranges."

        midpoints: List[float] = []
        bucket_indices: List[int] = []

        for i, (value, (min_val, max_val), buckets, normal_param) in enumerate(
            zip(vector, self.ranges, self.num_buckets, self.normal_params)
        ):
            if buckets == -1:
                # No discretization
                midpoints.append(value)
                bucket_indices.append(-1)
            elif buckets == 0:
                # Discretize into integers within range
                int_range = list(
                    range(int(np.ceil(min_val)), int(np.floor(max_val)) + 1)
                )
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
                    bucket_edges = [
                        scipy.stats.norm.ppf(
                            min(max((j / buckets), 1e-6), 1 - 1e-6), loc=mean, scale=std
                        )
                        for j in range(buckets + 1)
                    ]
                    for idx in range(buckets):
                        if bucket_edges[idx] <= value < bucket_edges[idx + 1]:
                            midpoints.append(
                                round(
                                    (bucket_edges[idx] + bucket_edges[idx + 1]) / 2, 6
                                )
                            )
                            bucket_indices.append(idx)
                            break
                    else:
                        midpoints.append(
                            round((bucket_edges[0] + bucket_edges[-1]) / 2, 6)
                        )  # Fallback to average if out of range
                        bucket_indices.append(-1)
                else:
                    step = (max_val - min_val) / buckets
                    bucket = int((value - min_val) / step)
                    bucket = min(
                        max(bucket, 0), buckets - 1
                    )  # Ensure bucket index is within bounds
                    midpoints.append(self.bucket_midpoints[i][bucket])
                    bucket_indices.append(bucket)

        return np.array(midpoints), np.array(bucket_indices)

    def encode_indices(self, indices: List[int]) -> int:
        """
        Encode bucket indices into a unique integer.

        :param indices: List of bucket indices.
        :return: Encoded integer.
        """
        assert len(indices) == len(
            self.num_buckets
        ), "Indices must match the number of dimensions."
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
                indices.append(
                    remaining_code % buckets
                )  # Extract the current dimension index
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

    def list_all_possible_combinations(
        self,
    ) -> Tuple[List[Tuple[float, ...]], List[Tuple[int, ...]]]:
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
        for i, ((min_val, max_val), buckets, normal_param) in enumerate(
            zip(self.ranges, self.num_buckets, self.normal_params)
        ):
            if buckets == -1:
                print(f"Dimension {i}: No discretization")
            elif buckets == 0:
                int_range = list(
                    range(int(np.ceil(min_val)), int(np.floor(max_val)) + 1)
                )
                print(f"Dimension {i}: Integer buckets {int_range}")
            elif buckets == 1:
                midpoint = round((min_val + max_val) / 2, 6)
                print(f"Dimension {i}: Single bucket at midpoint {midpoint}")
            else:
                if normal_param:
                    mean, std = normal_param
                    edges = [
                        scipy.stats.norm.ppf(
                            min(max((j / buckets), 1e-6), 1 - 1e-6), loc=mean, scale=std
                        )
                        for j in range(buckets + 1)
                    ]
                    for j in range(buckets):
                        bucket_min = round(edges[j], 6)
                        bucket_max = round(edges[j + 1], 6)
                        print(
                            f"Dimension {i}, Bucket {j}: Range [{bucket_min}, {bucket_max})"
                        )
                else:
                    step = (max_val - min_val) / buckets
                    for j in range(buckets):
                        bucket_min = round(min_val + j * step, 6)
                        bucket_max = round(bucket_min + step, 6)
                        print(
                            f"Dimension {i}, Bucket {j}: Range [{bucket_min}, {bucket_max})"
                        )

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
            return spaces.Box(
                low=np.array(low, dtype=np.float32),
                high=np.array(high, dtype=np.float32),
                dtype=np.float32,
            )

    def get_space_length(
        self,
    ):
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
        assert len(vector) == len(
            self.ranges
        ), "Input vector must have the same length as ranges."

        noisy_vector = np.copy(vector)

        for i, (value, (min_val, max_val), buckets) in enumerate(
            zip(vector, self.ranges, self.num_buckets)
        ):
            if buckets > 1:
                # Calculate bucket size
                bucket_size = (max_val - min_val) / buckets

                # Find the current bucket index
                bucket_index = int((value - min_val) / bucket_size)
                bucket_index = min(
                    max(bucket_index, 0), buckets - 1
                )  # Ensure index is within bounds

                # Calculate the current bucket's range
                bucket_start = min_val + bucket_index * bucket_size
                bucket_end = bucket_start + bucket_size

                # Add noise within the bucket's range
                noisy_vector[i] = np.random.uniform(bucket_start, bucket_end)

        return noisy_vector


@dataclass
class Transition:
    state: int         # Encoded state index
    action: int        # Encoded action index
    reward: float
    next_state: int    # Encoded next state index
    done: bool


class ReplayBuffer:
    """
    Replay Buffer with Prioritized Experience Replay (PER) support.
    Stores tabular state-action transitions, with priorities driven by TD error.
    """
    def __init__(self, capacity: int = 10000, alpha: float = 0.6):
        """
        :param capacity: Maximum number of transitions stored in the buffer.
        :param alpha: Degree of prioritization (0 = uniform, 1 = full prioritization).
        """
        self.capacity = capacity
        self.buffer: List[Transition] = []
        self.priorities: List[float] = []
        self.alpha = alpha
        self.pos = 0

    def add(self, transition: Transition, td_error: Optional[float] = None):
        """
        Add a transition with TD-error-based priority to the buffer.

        :param transition: The Transition to store.
        :param td_error: The TD error used to compute priority (optional).
        """
        if td_error is not None:
            priority = abs(td_error) + 1e-5  # Small epsilon to avoid zero priority
        else:
            priority = max(self.priorities, default=1.0)

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities.append(priority)
        else:
            self.buffer[self.pos] = transition
            self.priorities[self.pos] = priority
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int = 32, beta: float = 1.0) -> Tuple[List[Transition], List[int], np.ndarray]:
        """
        Sample a batch of transitions according to priorities.

        :param batch_size: Number of transitions to sample.
        :param beta: Importance-sampling weight adjustment.
        :return: Tuple (samples, indices, IS weights)
        """
        if len(self.buffer) == 0:
            return [], [], np.array([])

        priorities = np.array(self.priorities)
        scaled_priorities = priorities ** self.alpha
        probs = scaled_priorities / scaled_priorities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]

        # Importance-sampling weights to compensate bias
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()  # Normalize to 1

        return samples, indices, weights

    def update_priorities(self, indices: List[int], td_errors: List[float]):
        """
        Update the priorities of sampled transitions after TD-error recalculation.

        :param indices: List of indices for which priorities are updated.
        :param td_errors: New TD-errors corresponding to sampled transitions.
        """
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error) + 1e-5

    def clone(self):
        """
        Create a shallow copy of the replay buffer.
        """
        new_buffer = ReplayBuffer(capacity=self.capacity, alpha=self.alpha)
        new_buffer.buffer = self.buffer.copy()
        new_buffer.priorities = self.priorities.copy()
        new_buffer.pos = self.pos
        return new_buffer

    def save(self, directory: str):
        """
        Save ReplayBuffer content to the specified directory.

        :param directory: Directory path where files will be saved.
        """
        # --- Save transitions ---
        transition_data = []
        for t in self.buffer:
            transition_data.append({
                "state": t.state,
                "action": t.action,
                "reward": t.reward,
                "next_state": t.next_state,
                "done": t.done
            })

        df_transitions = pd.DataFrame(transition_data)
        df_transitions.to_csv(os.path.join(directory, "replay_buffer.csv"), index=False)

        # --- Save priorities ---
        df_priorities = pd.DataFrame({"priority": self.priorities})
        df_priorities.to_csv(os.path.join(directory, "replay_priorities.csv"), index=False)

        # --- Save buffer meta (pos, capacity, alpha) ---
        meta = {
            "pos": self.pos,
            "capacity": self.capacity,
            "alpha": self.alpha
        }
        with open(os.path.join(directory, "replay_buffer_meta.json"), "w") as f:
            json.dump(meta, f, indent=4)

        print(f"ReplayBuffer successfully saved to: {directory}")

    @classmethod
    def load(cls, directory: str) -> "ReplayBuffer":
        """
        Load ReplayBuffer from files in the specified directory.

        :param directory: Directory path where files are located.
        :return: Loaded ReplayBuffer instance.
        """
        # --- Load meta ---
        with open(os.path.join(directory, "replay_buffer_meta.json"), "r") as f:
            meta = json.load(f)

        buffer = cls(capacity=meta["capacity"], alpha=meta["alpha"])
        buffer.pos = meta["pos"]

        # --- Load transitions ---
        df_transitions = pd.read_csv(os.path.join(directory, "replay_buffer.csv"))
        buffer.buffer = []
        for _, row in df_transitions.iterrows():
            transition = Transition(
                state=int(row["state"]),
                action=int(row["action"]),
                reward=row["reward"],
                next_state=int(row["next_state"]),
                done=bool(row["done"])
            )
            buffer.buffer.append(transition)

        # --- Load priorities ---
        df_priorities = pd.read_csv(os.path.join(directory, "replay_priorities.csv"))
        buffer.priorities = df_priorities["priority"].tolist()

        print(f"ReplayBuffer successfully loaded from: {directory}")
        return buffer


class TabularQAgent(Agent):
    def __init__(
            self,
            env: gym.Env,
            *,
            state_discretizer: Union[Discretizer, None] = None,
            action_discretizer: Union[Discretizer, None] = None,
            state_ranges: Union[List[Tuple[float, float]], None] = None,
            num_state_buckets: Union[List[int], None] = None,
            state_normal_params: List[Optional[Tuple[float, float]]] = None,
            action_ranges: Union[List[Tuple[float, float]], None] = None,
            num_action_buckets: Union[List[int], None] = None,
            action_normal_params: List[Optional[Tuple[float, float]]] = None,
            learning_rate: float = 0.1,
            gamma: float = 0.99,
            buffer_size: int = 100_000,
            priority_exponent: float = 0.6,
            importance_sampling_correction = 1.0,
            max_temperature: float = 1.0,
            temperature_sensitivity = 0.1,
            batch_size = 32,
            batch_update_interval = 8,
            print_info: bool = True,
    ):
        super().__init__(env)

        # If discretizers are already provided, use them directly
        if state_discretizer is not None:
            self.state_discretizer = state_discretizer
        else:
            auto_state_ranges, auto_num_state_buckets = generate_discretizer_params_from_space(
                self.observation_space
            )

            final_state_ranges = [
                user if user is not None else auto
                for user, auto in zip(state_ranges or auto_state_ranges, auto_state_ranges)
            ]

            final_num_state_buckets = [
                user if user is not None else auto
                for user, auto in zip(num_state_buckets or auto_num_state_buckets, auto_num_state_buckets)
            ]

            final_state_normal_params = [
                param if param is not None else None
                for param in (state_normal_params or [None] * len(final_state_ranges))
            ]

            self.state_discretizer = Discretizer(
                ranges=final_state_ranges,
                num_buckets=final_num_state_buckets,
                normal_params=final_state_normal_params
            )

        if action_discretizer is not None:
            self.action_discretizer = action_discretizer
        else:
            auto_action_ranges, auto_num_action_buckets = generate_discretizer_params_from_space(
                self.action_space
            )

            final_action_ranges = [
                user if user is not None else auto
                for user, auto in zip(action_ranges or auto_action_ranges, auto_action_ranges)
            ]

            final_action_num_buckets = [
                user if user is not None else auto
                for user, auto in zip(num_action_buckets or auto_num_action_buckets, auto_num_action_buckets)
            ]

            final_action_normal_params = [
                param if param is not None else None
                for param in (action_normal_params or [None] * len(final_action_ranges))
            ]

            self.action_discretizer = Discretizer(
                ranges=final_action_ranges,
                num_buckets=final_action_num_buckets,
                normal_params=final_action_normal_params
            )

        self.q_table_1 = defaultdict(lambda: 0.0)
        self.q_table_2 = defaultdict(lambda: 0.0)
        self.visit_table = defaultdict(lambda: 0)  # Uses the same keys of the Q-Table to do visit count.

        # Q-Learning
        self.learning_rate = learning_rate
        self.gamma = gamma

        # Replay Buffer with PER
        self.replay_buffer = ReplayBuffer(
            capacity=buffer_size,
            alpha=priority_exponent,
        )
        self.importance_sampling_correction = importance_sampling_correction

        # VDBE temperature control
        self.T_max = max_temperature
        self.temperature_sensitivity = temperature_sensitivity
        self.state_temperature_table = defaultdict(lambda: self.T_max)

        # Batch update
        self.batch_size = batch_size
        self.batch_update_interval = batch_update_interval

        self.all_actions_encoded = sorted(
            [
                self.action_discretizer.encode_indices([*indices])
                for indices in self.action_discretizer.list_all_possible_combinations()[1]
            ]
        )

        self.print_info = print_info
        if print_info:
            self.print_q_table_info()

    @classmethod
    def load(cls, path: str, env: Optional[GymEnv] = None, print_system_info: bool = True) -> "TabularQAgent":
        """
        Load an agent from a zip file.

        :param path: Path to the saved zip file (including filename.zip)
        :param env: The environment instance (required)
        :param print_system_info: Whether to print Q-table info after loading
        :return: Loaded TabularQAgent instance
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # --- Unzip files ---
            with zipfile.ZipFile(path, "r") as zipf:
                zipf.extractall(temp_dir)

            # --- Load parameters ---
            with open(os.path.join(temp_dir, "parameters.json"), "r") as f:
                params = json.load(f)

            # --- Rebuild discretizers ---
            state_discretizer = Discretizer(
                ranges=params["state_discretizer"]["ranges"],
                num_buckets=params["state_discretizer"]["num_buckets"],
                normal_params=params["state_discretizer"]["normal_params"]
            )
            action_discretizer = Discretizer(
                ranges=params["action_discretizer"]["ranges"],
                num_buckets=params["action_discretizer"]["num_buckets"],
                normal_params=params["action_discretizer"]["normal_params"]
            )

            # --- Initialize agent ---
            agent = cls(
                env=env,
                state_discretizer=state_discretizer,
                action_discretizer=action_discretizer,
                learning_rate=params["learning_rate"],
                gamma=params["gamma"],
                buffer_size=params["buffer_size"],
                priority_exponent=params["priority_exponent"],
                importance_sampling_correction=params["importance_sampling_correction"],
                max_temperature=params["max_temperature"],
                temperature_sensitivity=params["temperature_sensitivity"],
                batch_size=params["batch_size"],
                batch_update_interval=params["batch_update_interval"],
                print_info=False
            )

            # --- Load Q-Table 1 ---
            df_q1 = pd.read_csv(os.path.join(temp_dir, "q_table_1.csv"))
            for _, row in df_q1.iterrows():
                agent.q_table_1[(int(row["state"]), int(row["action"]))] = row["q_value"]

            # --- Load Q-Table 2 ---
            df_q2 = pd.read_csv(os.path.join(temp_dir, "q_table_2.csv"))
            for _, row in df_q2.iterrows():
                agent.q_table_2[(int(row["state"]), int(row["action"]))] = row["q_value"]

            # --- Load visit table ---
            df_visit = pd.read_csv(os.path.join(temp_dir, "visit_table.csv"))
            for _, row in df_visit.iterrows():
                agent.visit_table[(int(row["state"]), int(row["action"]))] = int(row["visit_count"])

            # --- Load state temperature table ---
            df_temp = pd.read_csv(os.path.join(temp_dir, "state_temperature_table.csv"))
            for _, row in df_temp.iterrows():
                agent.state_temperature_table[int(row["state"])] = row["temperature"]

            # --- Load ReplayBuffer ---
            replay_dir = os.path.join(temp_dir, "replay_buffer")
            agent.replay_buffer = ReplayBuffer.load(replay_dir)

            print(f"Agent successfully loaded from: {path}")

            if print_system_info:
                agent.print_q_table_info()

            return agent

    def save(self, path: str):
        """
        Save the agent state to a zip file.

        :param path: Path to save the zip file (including filename.zip)
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # --- Save Q-Table 1 ---
            q1_data = [{"state": s, "action": a, "q_value": v}
                       for (s, a), v in self.q_table_1.items()]
            pd.DataFrame(q1_data).to_csv(os.path.join(temp_dir, "q_table_1.csv"), index=False)

            # --- Save Q-Table 2 ---
            q2_data = [{"state": s, "action": a, "q_value": v}
                       for (s, a), v in self.q_table_2.items()]
            pd.DataFrame(q2_data).to_csv(os.path.join(temp_dir, "q_table_2.csv"), index=False)

            # --- Save visit table ---
            visit_data = [{"state": s, "action": a, "visit_count": count}
                          for (s, a), count in self.visit_table.items()]
            pd.DataFrame(visit_data).to_csv(os.path.join(temp_dir, "visit_table.csv"), index=False)

            # --- Save state temperature table ---
            temp_data = [{"state": s, "temperature": t}
                         for s, t in self.state_temperature_table.items()]
            pd.DataFrame(temp_data).to_csv(os.path.join(temp_dir, "state_temperature_table.csv"), index=False)

            # --- Save hyperparameters & settings ---
            params = {
                "learning_rate": self.learning_rate,
                "gamma": self.gamma,
                "buffer_size": self.replay_buffer.capacity,
                "priority_exponent": self.replay_buffer.alpha,
                "importance_sampling_correction": self.importance_sampling_correction,
                "max_temperature": self.T_max,
                "temperature_sensitivity": self.temperature_sensitivity,
                "batch_size": self.batch_size,
                "batch_update_interval": self.batch_update_interval,
                # Discretizers
                "state_discretizer": {
                    "ranges": self.state_discretizer.ranges,
                    "num_buckets": self.state_discretizer.num_buckets,
                    "normal_params": self.state_discretizer.normal_params
                },
                "action_discretizer": {
                    "ranges": self.action_discretizer.ranges,
                    "num_buckets": self.action_discretizer.num_buckets,
                    "normal_params": self.action_discretizer.normal_params
                },
            }
            with open(os.path.join(temp_dir, "parameters.json"), "w") as f:
                json.dump(params, f, indent=4)

            # --- Save ReplayBuffer ---
            replay_dir = os.path.join(temp_dir, "replay_buffer")
            os.makedirs(replay_dir, exist_ok=True)
            self.replay_buffer.save(replay_dir)

            # --- Zip everything ---
            with zipfile.ZipFile(path, "w") as zipf:
                for root, _, files in os.walk(temp_dir):
                    for file in files:
                        full_path = os.path.join(root, file)
                        arcname = os.path.relpath(full_path, temp_dir)
                        zipf.write(full_path, arcname=arcname)

            print(f"Agent successfully saved to: {path}")

    def clone(self) -> "TabularQAgent":
        """
        Create a deep copy of the Tabular Q-Learning agent, including:
        - Q-tables
        - Visit table
        - State temperature table
        - Replay buffer (optional: shallow copy)

        :return: A new TabularQAgent instance with copied internal states.
        """
        # Create a new agent instance with the same hyperparameters
        new_agent = TabularQAgent(
            env=self.env,
            state_discretizer=self.state_discretizer,
            action_discretizer=self.action_discretizer,
            learning_rate=self.learning_rate,
            gamma=self.gamma,
            buffer_size=self.replay_buffer.capacity,
            priority_exponent=self.replay_buffer.alpha,
            importance_sampling_correction=self.importance_sampling_correction,
            max_temperature=self.T_max,
            temperature_sensitivity=self.temperature_sensitivity,
            batch_size=self.batch_size,
            batch_update_interval=self.batch_update_interval,
            print_info=False
        )

        # Deep copy all tables
        new_agent.q_table_1 = self.q_table_1.copy()
        new_agent.q_table_2 = self.q_table_2.copy()
        new_agent.visit_table = self.visit_table.copy()
        new_agent.state_temperature_table = self.state_temperature_table.copy()

        # Shallow copy of replay buffer (optional)
        new_agent.replay_buffer = self.replay_buffer.clone()

        if self.print_info:
            new_agent.print_q_table_info()

        return new_agent

    def print_q_table_info(self) -> None:
        """
        Print detailed information about the Q-tables, visit table, and state temperature table.
        """
        print("=" * 40)
        print("Tabular Q-Learning Agent Info")
        print("=" * 40)

        # --- State & Action discretizer info ---
        print("[State Discretizer]")
        self.state_discretizer.print_buckets()
        print("[Action Discretizer]")
        self.action_discretizer.print_buckets()

        total_state_combinations = self.state_discretizer.count_possible_combinations()
        total_action_combinations = self.action_discretizer.count_possible_combinations()
        total_state_action_combinations = total_state_combinations * total_action_combinations

        # --- Q-Tables ---
        q_table_1_size = len(self.q_table_1)
        q_table_2_size = len(self.q_table_2)
        print(f"[Q-Table 1] {q_table_1_size} state-action pairs.")
        print(f"[Q-Table 2] {q_table_2_size} state-action pairs.")
        print(f"Total State-Action Combinations: {total_state_action_combinations}")
        print(f"Q-Table 1 Coverage: {q_table_1_size / total_state_action_combinations * 100:.2f}%")
        print(f"Q-Table 2 Coverage: {q_table_2_size / total_state_action_combinations * 100:.2f}%")

        # --- Visit table ---
        visit_table_size = len(self.visit_table)
        print(f"[Visit Table] {visit_table_size} entries (state-action pairs tracked).")

        # --- State temperature table ---
        temp_values = list(self.state_temperature_table.values())
        if temp_values:
            print(f"[State Temperature Table]")
            print(f"  Number of states with temperature: {len(temp_values)}")
            print(f"  Temperature (min / mean / max): {min(temp_values):.4f} / "
                  f"{sum(temp_values) / len(temp_values):.4f} / {max(temp_values):.4f}")
        else:
            print(f"[State Temperature Table] No entries found.")

        # --- Replay Buffer info ---
        print(f"[Replay Buffer]")
        print(f"  Current size: {len(self.replay_buffer.buffer)} / {self.replay_buffer.capacity}")
        print(f"  Priority exponent (α): {self.replay_buffer.alpha}")
        print(f"  Importance sampling correction (β): {self.importance_sampling_correction}")

        print("=" * 40)

    def reset_q_table(self) -> None:
        self.q_table = defaultdict(
            lambda: 0.0
        )  # Flattened Q-Table with state-action tuple keys

    def reset_visit_table(self) -> None:
        self.visit_table = defaultdict(
            lambda: 0
        )  # Uses the same keys of the Q-Table to do visit count.

    def get_action_probabilities(self, state: np.ndarray) -> np.ndarray:
        """
        Calculate action probabilities for a given state using state-specific temperature.

        :param state: The current observation (raw state array).
        :return: An array of action probabilities.
        """
        # --- Encode the current state (discrete index) ---
        encoded_state = self.state_discretizer.encode_indices(
            [*self.state_discretizer.discretize(state)[1]]
        )

        # --- Retrieve Q-values for all actions ---
        q_values = np.array([
            (self.q_table_1[(encoded_state, a)] + self.q_table_2[(encoded_state, a)]) / 2.0
            for a in self.all_actions_encoded
        ])

        # --- Get temperature for this state ---
        temperature = self.state_temperature_table[encoded_state]

        # --- Handle all-zero Q-values ---
        if np.all(q_values == 0):
            probabilities = np.ones_like(q_values, dtype=float) / len(q_values)
            return probabilities

        # --- Stability control for temperature ---
        temperature = max(temperature, 1e-6)  # Prevent division by zero

        # --- Softmax computation ---
        q_values_stable = q_values - np.max(q_values)  # Numerical stability
        exp_values = np.exp(q_values_stable / temperature)
        probabilities = exp_values / (np.sum(exp_values) + 1e-10)

        return probabilities

    def choose_action(self, state: np.ndarray, greedy: bool = False) -> np.ndarray:
        """
        Select an action for a given state using state-specific temperature (softmax).
        If greedy is True, always select the highest probability action.

        :param state: The current observation (raw state array).
        :param greedy: Whether to use greedy action selection.
        :return: Action formatted for the environment (decoded if necessary).
        """
        # --- Get action probabilities using state-specific temperature ---
        action_probabilities = self.get_action_probabilities(state)

        # --- Select action ---
        if greedy:
            action_encoded = np.argmax(action_probabilities)
        else:
            action_encoded = np.random.choice(
                self.all_actions_encoded,
                p=action_probabilities
            )

        # --- Convert encoded action to actual action (for env.step) ---
        action = np.array(
            self.action_discretizer.indices_to_midpoints(
                self.action_discretizer.decode_indices(action_encoded)
            )
        )

        # --- Handle discrete action spaces ---
        if isinstance(self.env.action_space, spaces.Discrete):
            action = action.squeeze().item()

        return action

    def predict(self, state: np.ndarray, greedy: bool = False) -> np.ndarray:
        return self.choose_action(state, greedy)

    def update(
            self,
            state_encoded: int,
            action_encoded: int,
            reward: float,
            next_state_encoded: int,
            done: bool,
            is_weight: float = 1.0  # Importance-sampling weight, default no correction
    ) -> float:
        """
        Double Q-Learning update rule with IS weight.

        :param state_encoded: Encoded current state index.
        :param action_encoded: Encoded action index.
        :param reward: Immediate reward.
        :param next_state_encoded: Encoded next state index.
        :param done: Whether episode ended.
        :param is_weight: Importance-sampling weight (PER correction factor).
        :return: Absolute TD error (unweighted), for PER priority update.
        """
        # Randomly decide which table to update
        update_first = np.random.rand() < 0.5

        if update_first:
            q_update, q_target = self.q_table_1, self.q_table_2
        else:
            q_update, q_target = self.q_table_2, self.q_table_1

        # Compute TD target
        if done:
            td_target = reward
        else:
            best_next_action = max(
                self.all_actions_encoded,
                key=lambda a: q_update[(next_state_encoded, a)]
            )
            td_target = reward + self.gamma * q_target[(next_state_encoded, best_next_action)]

        # TD error
        td_error = td_target - q_update[(state_encoded, action_encoded)]

        # Apply importance-sampling weight on the TD error (scale update)
        q_update[(state_encoded, action_encoded)] += (
                self.learning_rate * is_weight * td_error
        )

        # Optional visit tracking
        self.visit_table[(state_encoded, action_encoded)] += 1

        # Return abs(TD error), not weighted → for priority updates
        return abs(td_error)

    def learn(
            self,
            total_timesteps: int,
            reset_num_timesteps: bool = True,
            progress_bar: bool = False
    ):
        """
        Double Q-Learning loop with Replay Buffer (PER), IS correction, and VDBE temperature.
        """

        state, info = self.env.reset()
        episode_step_count = 0
        num_episodes = 0
        num_truncated = 0
        num_terminated = 0
        sum_episode_rewards = 0

        pbar = tqdm.tqdm(total=total_timesteps, desc="Inner Training", unit="step") if progress_bar else None

        batch_size = self.batch_size
        batch_update_interval = self.batch_update_interval

        for timestep in range(total_timesteps):
            # --- Encode current state ---
            state_encoded = self.state_discretizer.encode_indices(
                [*self.state_discretizer.discretize(state)[1]]
            )

            # --- Select action with current_temperature (VDBE) ---
            action_probs = self.get_action_probabilities(state,)
            action_encoded = np.random.choice(self.all_actions_encoded, p=action_probs)

            action = np.array(
                self.action_discretizer.indices_to_midpoints(
                    self.action_discretizer.decode_indices(action_encoded)
                )
            )
            if isinstance(self.env.action_space, spaces.Discrete):
                action = action.squeeze().item()

            # --- Interact with environment ---
            next_state, reward, terminated, truncated, info = self.env.step(action)

            next_state_encoded = self.state_discretizer.encode_indices(
                [*self.state_discretizer.discretize(next_state)[1]]
            )

            # --- Online update (IS weight=1.0 for online update) ---
            td_error = self.update(
                state_encoded=state_encoded,
                action_encoded=action_encoded,
                reward=reward,
                next_state_encoded=next_state_encoded,
                done=terminated,
                is_weight=1.0  # Online step no correction
            )

            # --- Store in replay buffer with priority ---
            transition = Transition(
                state=state_encoded,
                action=action_encoded,
                reward=reward,
                next_state=next_state_encoded,
                done=terminated
            )
            self.replay_buffer.add(transition, td_error)

            # --- VDBE temperature update ---
            # VDBE-based temperature update for the current state
            self.state_temperature_table[state_encoded] = (
                    self.T_max * np.exp(-td_error / self.temperature_sensitivity)
            )

            # --- Periodic batch update from replay buffer ---
            if timestep % batch_update_interval == 0 and len(self.replay_buffer.buffer) >= batch_size:
                # Sample batch
                transitions, indices, weights = self.replay_buffer.sample(
                    batch_size=batch_size,
                    beta=self.importance_sampling_correction  # Fixed beta
                )

                batch_td_errors = []
                for trans, is_weight in zip(transitions, weights):
                    td_error_batch = self.update(
                        state_encoded=trans.state,
                        action_encoded=trans.action,
                        reward=trans.reward,
                        next_state_encoded=trans.next_state,
                        done=trans.done,
                        is_weight=is_weight
                    )
                    batch_td_errors.append(td_error_batch)

                # Update priorities after batch updates
                self.replay_buffer.update_priorities(indices, batch_td_errors)

            # --- Move to next state ---
            state = next_state

            # --- Episode control and statistics ---
            episode_step_count += 1
            sum_episode_rewards += reward
            if terminated or truncated:
                episode_step_count = 0
                num_episodes += 1
                num_terminated += 1 if terminated else 0
                num_truncated += 1 if truncated else 0
                state, info = self.env.reset()

            # --- Progress bar update ---
            if progress_bar:
                pbar.set_postfix({
                    "Episodes": num_episodes,
                    "Terminated": num_terminated,
                    "Truncated": num_truncated,
                    "Reward (last)": reward,
                    "Avg Episode Reward": (
                        sum_episode_rewards / num_episodes
                        if num_episodes > 0
                        else 0.0
                    ),
                })
                pbar.update(1)

        if progress_bar:
            pbar.close()


class _TabularQAgent(Agent):
    def __init__(
            self,
            env: gym.Env,
            *,
            state_discretizer: Union[Discretizer, None] = None,
            action_discretizer: Union[Discretizer, None] = None,
            state_ranges: Union[List[Tuple[float, float]], None] = None,
            num_state_buckets: Union[List[int], None] = None,
            state_normal_params: List[Optional[Tuple[float, float]]] = None,
            action_ranges: Union[List[Tuple[float, float]], None] = None,
            num_action_buckets: Union[List[int], None] = None,
            action_normal_params: List[Optional[Tuple[float, float]]] = None,
            learning_rate: float = 0.1,
            gamma: float = 0.99,
            print_info: bool = True,
    ):
        super().__init__(env)

        # If discretizers are already provided, use them directly
        if state_discretizer is not None:
            self.state_discretizer = state_discretizer
        else:
            auto_state_ranges, auto_num_state_buckets = generate_discretizer_params_from_space(
                self.observation_space
            )

            final_state_ranges = [
                user if user is not None else auto
                for user, auto in zip(state_ranges or auto_state_ranges, auto_state_ranges)
            ]

            final_num_state_buckets = [
                user if user is not None else auto
                for user, auto in zip(num_state_buckets or auto_num_state_buckets, auto_num_state_buckets)
            ]

            final_state_normal_params = [
                param if param is not None else None
                for param in (state_normal_params or [None] * len(final_state_ranges))
            ]

            self.state_discretizer = Discretizer(
                ranges=final_state_ranges,
                num_buckets=final_num_state_buckets,
                normal_params=final_state_normal_params
            )

        if action_discretizer is not None:
            self.action_discretizer = action_discretizer
        else:
            auto_action_ranges, auto_num_action_buckets = generate_discretizer_params_from_space(
                self.action_space
            )

            final_action_ranges = [
                user if user is not None else auto
                for user, auto in zip(action_ranges or auto_action_ranges, auto_action_ranges)
            ]

            final_action_num_buckets = [
                user if user is not None else auto
                for user, auto in zip(num_action_buckets or auto_num_action_buckets, auto_num_action_buckets)
            ]

            final_action_normal_params = [
                param if param is not None else None
                for param in (action_normal_params or [None] * len(final_action_ranges))
            ]

            self.action_discretizer = Discretizer(
                ranges=final_action_ranges,
                num_buckets=final_action_num_buckets,
                normal_params=final_action_normal_params
            )

        # Other hyperparameters
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.print_info = print_info
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.env = env
        self.q_table = defaultdict(
            lambda: 0.0
        )  # Flattened Q-Table with state-action tuple keys
        self.visit_table = defaultdict(
            lambda: 0
        )  # Uses the same keys of the Q-Table to do visit count.
        self.all_actions_encoded = sorted(
            [
                self.action_discretizer.encode_indices([*indices])
                for indices in self.action_discretizer.list_all_possible_combinations()[
                    1
                ]
            ]
        )
        if print_info:
            self.print_q_table_info()

    def reset_q_table(self) -> None:
        self.q_table = defaultdict(
            lambda: 0.0
        )  # Flattened Q-Table with state-action tuple keys

    def reset_visit_table(self) -> None:
        self.visit_table = defaultdict(
            lambda: 0
        )  # Uses the same keys of the Q-Table to do visit count.

    def clone(self) -> "TabularQAgent":
        """
        Create a deep copy of the Q-Table agent.

        :return: A new QTableAgent instance with the same Q-Table.
        """
        new_agent = TabularQAgent(
            self.env,
            state_discretizer=self.state_discretizer,
            action_discretizer=self.action_discretizer,
            learning_rate=self.learning_rate,
            gamma=self.gamma,
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
        total_combinations = (
            self.state_discretizer.count_possible_combinations()
            * self.action_discretizer.count_possible_combinations()
        )
        print(
            f"Q-Table Size: {len(self.q_table)} state-action pairs / total combinations: {total_combinations}."
        )
        print(
            f"State-Action usage: {len(self.q_table) / total_combinations * 100:.2f}%."
        )

    def save_q_table(self, file_path: str = None) -> pd.DataFrame:
        """
        Save the Q-Table to a CSV file and/or return as a DataFrame.
        IMPORTANT: This method is not a class method but an instance method, This is different from sb3!
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
            row.update(
                {
                    f"action_{a}_q_value": self.q_table[(encoded_state, a)]
                    for a in self.all_actions_encoded
                }
            )
            row.update(
                {
                    f"action_{a}_visit_count": self.visit_table[(encoded_state, a)]
                    for a in self.all_actions_encoded
                }
            )
            row.update(
                {
                    "total_visit_count": sum(
                        [
                            self.visit_table[(encoded_state, a)]
                            for a in self.all_actions_encoded
                        ]
                    )
                }
            )
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
            print(
                "Warning: Loading a Q-Table that already has data. Part of them might be overwritten."
            )

        for _, row in df.iterrows():
            encoded_state = int(row["state"])
            for a in self.all_actions_encoded:
                self.q_table[(encoded_state, a)] = row[f"action_{a}_q_value"]
                self.visit_table[(encoded_state, a)] = row[f"action_{a}_visit_count"]
        print(f"Q-Table loaded from {f'{file_path}' if file_path else 'DataFrame'}.")
        self.print_q_table_info()

    def get_action_probabilities(
        self, state: np.ndarray, temperature: float = None
    ) -> np.ndarray:
        """
        Calculate action probabilities based on the specified strategy.

        :param state: The current state.
        :param temperature: Temperature parameter for softmax. If None, uses ||Q||_2.
        :return: An array of action probabilities.
        """
        encoded_state = self.state_discretizer.encode_indices(
            [*self.state_discretizer.discretize(state)[1]]
        )

        # Retrieve Q-values for all actions
        q_values = np.array(
            [self.q_table[(encoded_state, a)] for a in self.all_actions_encoded]
        )
        visitation = sum(
            [self.visit_table[(encoded_state, a)] for a in self.all_actions_encoded]
        )

        if np.all(q_values == 0):  # Handle all-zero Q-values
            probabilities = np.ones_like(q_values, dtype=float) / len(q_values)
        else:
            # Compute ||Q||_2 if temperature is None
            temp = (
                np.linalg.norm(q_values, ord=2) if temperature is None else temperature
            )
            temp = max(temp, 1e-6)  # Avoid division by zero

            # Subtract the maximum value for numerical stability
            q_values_stable = q_values - np.max(q_values)
            exp_values = np.exp(q_values_stable / temp)
            probabilities = exp_values / (
                np.sum(exp_values) + 1e-10
            )  # Add small value to prevent division by zero

        if visitation == 0:
            probabilities = np.ones_like(q_values, dtype=float) / len(q_values)

        return probabilities

    def choose_action(
        self, state: np.ndarray, temperature: float = None, greedy: bool = False
    ) -> np.ndarray:
        action_probabilities = self.get_action_probabilities(state, temperature)
        if greedy:
            action_encoded = np.argmax(action_probabilities)
        else:
            action_encoded = np.random.choice(
                self.all_actions_encoded, p=action_probabilities
            )
        action = np.array(
            self.action_discretizer.indices_to_midpoints(
                self.action_discretizer.decode_indices(action_encoded)
            )
        )
        if isinstance(self.env.action_space, spaces.Discrete):
            action = action.squeeze().item()
        return action

    def update(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        alpha: float = 0.1,
        gamma: float = 0.99,
    ) -> None:
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
        state_encoded = self.state_discretizer.encode_indices(
            [*self.state_discretizer.discretize(state)[1]]
        )
        next_state_encoded = self.state_discretizer.encode_indices(
            [*self.state_discretizer.discretize(next_state)[1]]
        )
        action_encoded = self.action_discretizer.encode_indices(
            [*self.action_discretizer.discretize(action)[1]]
        )

        # reward /= 10.

        if done:
            td_target = reward  # No future reward if the episode is done
        else:
            # Compute the best next action's Q-value
            best_next_action_value = max(
                [
                    self.q_table.get((next_state_encoded, a), 0.0)
                    for a in self.all_actions_encoded
                ],
                default=0.0,
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
        temperature: float = None,
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
                pass
            action = self.choose_action(state, temperature=temperature)
            next_state, reward, terminated, truncated, info = self.env.step(action)
            self.update(
                state,
                action,
                reward,
                next_state,
                terminated,
                alpha=self.learning_rate,
                gamma=self.gamma,
            )
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
                pbar.set_postfix(
                    {
                        "Episodes": num_episodes,
                        "Terminated": num_terminated,
                        "Truncated": num_truncated,
                        "Reward (last)": reward,
                        "Avg Episode Reward": (
                            sum_episode_rewards / num_episodes
                            if num_episodes > 0
                            else 0.0
                        ),
                    }
                )
                pbar.update(1)
        if progress_bar:
            pbar.close()
