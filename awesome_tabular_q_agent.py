import json
import os
import tempfile
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from itertools import product
from typing import List, Tuple, Optional, Union, Callable, Any
from abc import ABC, abstractmethod

import cv2
import imageio
import numpy as np
import pandas as pd
import scipy
from gymnasium import spaces
import gymnasium as gym
import tqdm
from gymnasium.core import Env
from numpy import ndarray
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv


def check_for_correct_spaces(
    env: Union[Env, "VecEnv"],
    observation_space: spaces.Space,
    action_space: spaces.Space,
) -> None:
    if observation_space != env.observation_space:
        raise ValueError(
            f"Observation spaces do not match: {observation_space} != {env.observation_space}"
        )
    if action_space != env.action_space:
        raise ValueError(
            f"Action spaces do not match: {action_space} != {env.action_space}"
        )


def generate_discretizer_params_from_space(
    space: spaces.Space, default_num_buckets_per_dim: int = 15
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
        raise NotImplementedError(
            f"Space type {type(space)} is not supported for discretization."
        )

    return ranges, num_buckets


def merge_params(user_values: Optional[List], auto_values: List) -> List:
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


def safe_json(obj):
    """
    Recursively convert numpy types to native Python types
    """
    if isinstance(obj, dict):
        return {safe_json(k): safe_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [safe_json(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(safe_json(i) for i in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


class BaseCallback:
    """
    Base class for callbacks (compatible with SB3 style).
    """

    def __init__(self, verbose=0):
        self.verbose = verbose
        self.training_env = None
        self.locals = None
        self.globals = None
        self.num_timesteps = 0
        self.n_calls = 0
        self.n_episodes = 0
        self.model = None

    def init_callback(self, model):
        """
        Initialize callback before training starts.
        """
        self.model = model
        self._init_callback()

    def on_training_start(
        self, locals_: dict[str, Any], globals_: dict[str, Any]
    ) -> None:
        # Those are reference and will be updated automatically
        self.locals = locals_
        self.globals = globals_
        # Update num_timesteps in case training was done before
        self.num_timesteps = self.model.num_timesteps
        self._on_training_start()

    def on_step(self):
        """
        Called at each environment step.
        """
        self.n_calls += 1
        self.num_timesteps = self.model.num_timesteps

        return self._on_step()

    def on_rollout_end(self):
        """
        Called when an episode ends (rollout finished).
        """
        self.n_episodes += 1
        self._on_rollout_end()

    def on_training_end(self):
        """
        Called after the training loop.
        """
        self._on_training_end()

    def _init_callback(
        self,
    ):
        pass

    def _on_training_start(self):
        pass

    def _on_step(self):
        pass

    def _on_rollout_end(self):
        pass

    def _on_training_end(self):
        pass


class CallbackList(BaseCallback):
    """
    Combine multiple callbacks into one list callback.
    """

    def __init__(self, callbacks):
        super().__init__()
        self.callbacks = callbacks

    def _init_callback(self):
        for callback in self.callbacks:
            callback.init_callback(self.model)

    def _on_training_start(self) -> None:
        for callback in self.callbacks:
            callback.on_training_start(self.locals, self.globals)

    def _on_step(self) -> bool:
        continue_training = True
        for callback in self.callbacks:
            # Return False (stop training) if at least one callback returns False
            continue_training = callback.on_step() and continue_training
        return continue_training

    def _on_rollout_start(self) -> None:
        for callback in self.callbacks:
            callback.on_rollout_start()

    def _on_rollout_end(self) -> None:
        for callback in self.callbacks:
            callback.on_rollout_end()

    def _on_training_end(self) -> None:
        for callback in self.callbacks:
            callback.on_training_end()


class FunctionCallback(BaseCallback):
    """
    Wrap a callable into a BaseCallback.
    """

    def __init__(self, func):
        super().__init__()
        self.func = func

    def on_step(self):
        return self.func(self.model)


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
        self.num_timesteps = 0

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
        callback: Union[None, Callable, list["BaseCallback"], "BaseCallback"] = None,
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
    def predict(
            self,
            observation: Union[np.ndarray, dict[str, np.ndarray]],
            state: Optional[tuple[np.ndarray, ...]] = None,
            episode_start: Optional[np.ndarray] = None,
            deterministic: bool = False,
    ):
        """
        Predict an action given an observation.
        :param observation: Observation from the environment.
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether to return deterministic actions.
        :return: Action(s) to take.
        """
        pass

    @classmethod
    @abstractmethod
    def load(
        cls,
        path: str,
        env: Optional[GymEnv] = None,
        print_system_info: bool = False,
    ):
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
            or isinstance(vector, np.int32)
            or isinstance(vector, float)
            or isinstance(vector, np.float32)
            or isinstance(vector, np.float64)
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
    state: int  # Encoded state index
    action: int  # Encoded action index
    reward: float
    next_state: int  # Encoded next state index
    done: bool


class ReplayBuffer:
    """
    Simple Replay Buffer (no Prioritized Experience Replay).
    Stores tabular state-action transitions.
    """

    def __init__(
        self,
        capacity: int = 10000,
    ):
        """
        :param capacity: Maximum number of transitions stored in the buffer.
        """
        self.capacity = capacity
        self.buffer: List[Transition] = []
        self.pos = 0
        self.size = 0  # Track current number of valid transitions

    def add(self, transition: Transition, td_error: Optional[float] = None):
        """
        Add a transition to the buffer (no priority, no td_error needed).

        :param transition: The Transition to store.
        :param td_error: Ignored in uniform buffer (only for compatibility).
        """
        if self.size < self.capacity:
            self.buffer.append(transition)
            self.size += 1
        else:
            self.buffer[self.pos] = transition
            self.pos = (self.pos + 1) % self.capacity

    def sample(
        self,
        batch_size: int = 32,
    ) -> Tuple[List[Transition], List[int], np.ndarray]:
        """
        Uniformly sample a batch of transitions.

        :param batch_size: Number of transitions to sample.
        :return: Tuple (samples, indices, IS weights)
        """
        if self.size == 0:
            return [], [], np.array([])

        # Uniform random sampling
        indices = np.random.choice(self.size, batch_size, replace=False)
        samples = [self.buffer[i] for i in indices]

        # Importance sampling weights = 1.0 everywhere (no PER)
        weights = np.ones_like(indices, dtype=np.float32)

        return samples, indices.tolist(), weights

    def update_priorities(self, indices: List[int], td_errors: List[float]):
        """
        No-op for uniform buffer (kept for compatibility).
        """
        pass

    def clone(self):
        """
        Create a shallow copy of the replay buffer.
        """
        new_buffer = ReplayBuffer(
            capacity=self.capacity,
        )
        new_buffer.buffer = self.buffer.copy()
        new_buffer.size = self.size
        new_buffer.pos = self.pos
        return new_buffer

    def save(self, directory: str):
        """
        Save ReplayBuffer content to the specified directory.
        :param directory: Directory path where files will be saved.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)

        # --- Save transitions ---
        transition_data = []
        for t in self.buffer[: self.size]:
            transition_data.append(
                {
                    "state": t.state,
                    "action": t.action,
                    "reward": t.reward,
                    "next_state": t.next_state,
                    "done": t.done,
                }
            )

        df_transitions = pd.DataFrame(transition_data)
        df_transitions.to_csv(os.path.join(directory, "replay_buffer.csv"), index=False)

        # --- Save buffer meta ---
        meta = {"pos": self.pos, "capacity": self.capacity, "size": self.size}
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

        buffer = cls(
            capacity=meta["capacity"],
        )
        buffer.pos = meta["pos"]
        buffer.size = meta["size"]

        # --- Load transitions ---
        df_transitions = pd.read_csv(os.path.join(directory, "replay_buffer.csv"))
        buffer.buffer = []
        for _, row in df_transitions.iterrows():
            transition = Transition(
                state=int(row["state"]),
                action=int(row["action"]),
                reward=row["reward"],
                next_state=int(row["next_state"]),
                done=bool(row["done"]),
            )
            buffer.buffer.append(transition)

        print(f"ReplayBuffer successfully loaded from: {directory}")
        return buffer


class TabularQAgent(Agent):
    def __init__(
        self,
        env: Union[Env, VecEnv, None],
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
        learning_starts: int = 100_000,
        max_temperature: float = 1.0,
        min_temperature: float = 0.01,
        temperature_scale=1.0,
        update_steps=32,
        update_interval=8,
        print_info: bool = True,
    ):
        super().__init__(env)

        # If discretizers are already provided, use them directly
        if state_discretizer is not None:
            self.state_discretizer = state_discretizer
        else:
            auto_state_ranges, auto_num_state_buckets = (
                generate_discretizer_params_from_space(self.observation_space)
            )

            final_state_ranges = [
                user if user is not None else auto
                for user, auto in zip(
                    state_ranges or auto_state_ranges, auto_state_ranges
                )
            ]

            final_num_state_buckets = [
                user if user is not None else auto
                for user, auto in zip(
                    num_state_buckets or auto_num_state_buckets, auto_num_state_buckets
                )
            ]

            final_state_normal_params = [
                param if param is not None else None
                for param in (state_normal_params or [None] * len(final_state_ranges))
            ]

            self.state_discretizer = Discretizer(
                ranges=final_state_ranges,
                num_buckets=final_num_state_buckets,
                normal_params=final_state_normal_params,
            )

        if action_discretizer is not None:
            self.action_discretizer = action_discretizer
        else:
            auto_action_ranges, auto_num_action_buckets = (
                generate_discretizer_params_from_space(self.action_space)
            )

            final_action_ranges = [
                user if user is not None else auto
                for user, auto in zip(
                    action_ranges or auto_action_ranges, auto_action_ranges
                )
            ]

            final_action_num_buckets = [
                user if user is not None else auto
                for user, auto in zip(
                    num_action_buckets or auto_num_action_buckets,
                    auto_num_action_buckets,
                )
            ]

            final_action_normal_params = [
                param if param is not None else None
                for param in (action_normal_params or [None] * len(final_action_ranges))
            ]

            self.action_discretizer = Discretizer(
                ranges=final_action_ranges,
                num_buckets=final_action_num_buckets,
                normal_params=final_action_normal_params,
            )

        self.q_table_1 = defaultdict(lambda: 0.0)
        self.q_table_2 = defaultdict(lambda: 0.0)
        self.visit_table = defaultdict(
            lambda: 0
        )  # Uses the same keys of the Q-Table to do visit count.

        # Q-Learning
        self.learning_rate = learning_rate
        self.gamma = gamma

        # Replay Buffer with PER
        self.replay_buffer = ReplayBuffer(
            capacity=buffer_size,
        )
        self.learning_starts = learning_starts

        # temperature control
        self.T_max = max_temperature
        self.T_min = min_temperature
        self.temperature_scale = temperature_scale

        # Batch update
        self.update_steps = update_steps
        self.update_interval = update_interval

        self.all_actions_encoded = sorted(
            [
                self.action_discretizer.encode_indices([*indices])
                for indices in self.action_discretizer.list_all_possible_combinations()[
                    1
                ]
            ]
        )

        self.print_info = print_info
        if print_info:
            self.print_q_table_info()

    @classmethod
    def load(
        cls, path: str, env: Optional[GymEnv] = None, print_system_info: bool = True
    ) -> "TabularQAgent":
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
                normal_params=params["state_discretizer"]["normal_params"],
            )
            action_discretizer = Discretizer(
                ranges=params["action_discretizer"]["ranges"],
                num_buckets=params["action_discretizer"]["num_buckets"],
                normal_params=params["action_discretizer"]["normal_params"],
            )

            # --- Initialize agent ---
            agent = cls(
                env=env,
                state_discretizer=state_discretizer,
                action_discretizer=action_discretizer,
                learning_rate=params["learning_rate"],
                gamma=params["gamma"],
                buffer_size=params["buffer_size"],
                learning_starts=params["learning_starts"],
                max_temperature=params["max_temperature"],
                min_temperature=params["min_temperature"],
                temperature_scale=params["temperature_scale"],
                update_steps=params["update_steps"],
                update_interval=params["update_interval"],
                print_info=False,
            )

            # --- Load Q-Table 1 ---
            df_q1 = pd.read_csv(os.path.join(temp_dir, "q_table_1.csv"))
            for _, row in df_q1.iterrows():
                agent.q_table_1[(int(row["state"]), int(row["action"]))] = row[
                    "q_value"
                ]

            # --- Load Q-Table 2 ---
            df_q2 = pd.read_csv(os.path.join(temp_dir, "q_table_2.csv"))
            for _, row in df_q2.iterrows():
                agent.q_table_2[(int(row["state"]), int(row["action"]))] = row[
                    "q_value"
                ]

            # --- Load visit table ---
            df_visit = pd.read_csv(os.path.join(temp_dir, "visit_table.csv"))
            for _, row in df_visit.iterrows():
                agent.visit_table[(int(row["state"]), int(row["action"]))] = int(
                    row["visit_count"]
                )

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
            q1_data = [
                {"state": s, "action": a, "q_value": v}
                for (s, a), v in self.q_table_1.items()
            ]
            pd.DataFrame(q1_data).to_csv(
                os.path.join(temp_dir, "q_table_1.csv"), index=False
            )

            # --- Save Q-Table 2 ---
            q2_data = [
                {"state": s, "action": a, "q_value": v}
                for (s, a), v in self.q_table_2.items()
            ]
            pd.DataFrame(q2_data).to_csv(
                os.path.join(temp_dir, "q_table_2.csv"), index=False
            )

            # --- Save visit table ---
            visit_data = [
                {"state": s, "action": a, "visit_count": count}
                for (s, a), count in self.visit_table.items()
            ]
            pd.DataFrame(visit_data).to_csv(
                os.path.join(temp_dir, "visit_table.csv"), index=False
            )

            # --- Save hyperparameters & settings ---
            params = {
                "learning_rate": self.learning_rate,
                "gamma": self.gamma,
                "buffer_size": self.replay_buffer.capacity,
                "learning_starts": self.learning_starts,
                "max_temperature": self.T_max,
                "min_temperature": self.T_min,
                "temperature_scale": self.temperature_scale,
                "update_steps": self.update_steps,
                "update_interval": self.update_interval,
                # Discretizers
                "state_discretizer": {
                    "ranges": self.state_discretizer.ranges,
                    "num_buckets": self.state_discretizer.input_num_buckets,
                    "normal_params": self.state_discretizer.normal_params,
                },
                "action_discretizer": {
                    "ranges": self.action_discretizer.ranges,
                    "num_buckets": self.action_discretizer.input_num_buckets,
                    "normal_params": self.action_discretizer.normal_params,
                },
            }
            with open(os.path.join(temp_dir, "parameters.json"), "w") as f:
                json.dump(safe_json(params), f, indent=4)

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
            learning_starts=self.learning_starts,
            max_temperature=self.T_max,
            min_temperature=self.T_min,
            temperature_scale=self.temperature_scale,
            update_steps=self.update_steps,
            update_interval=self.update_interval,
            print_info=False,
        )

        # Deep copy all tables
        new_agent.q_table_1 = self.q_table_1.copy()
        new_agent.q_table_2 = self.q_table_2.copy()
        new_agent.visit_table = self.visit_table.copy()

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
        total_action_combinations = (
            self.action_discretizer.count_possible_combinations()
        )
        total_state_action_combinations = (
            total_state_combinations * total_action_combinations
        )

        # --- Q-Tables ---
        q_table_1_size = len(self.q_table_1)
        q_table_2_size = len(self.q_table_2)
        print(f"[Q-Table 1] {q_table_1_size} state-action pairs.")
        print(f"[Q-Table 2] {q_table_2_size} state-action pairs.")
        print(f"Total State-Action Combinations: {total_state_action_combinations}")
        print(
            f"Q-Table 1 Coverage: {q_table_1_size / total_state_action_combinations * 100:.2f}%"
        )
        print(
            f"Q-Table 2 Coverage: {q_table_2_size / total_state_action_combinations * 100:.2f}%"
        )

        # --- Visit table ---
        visit_table_size = len(self.visit_table)
        print(f"[Visit Table] {visit_table_size} entries (state-action pairs tracked).")

        # --- Replay Buffer info ---
        print(f"[Replay Buffer]")
        print(
            f"  Current size: {len(self.replay_buffer.buffer)} / {self.replay_buffer.capacity}"
        )

        print("=" * 40)

    def reset_q_table(self) -> None:
        self.q_table_1 = defaultdict(lambda: 0.0)
        self.q_table_2 = defaultdict(lambda: 0.0)

    def reset_visit_table(self) -> None:
        self.visit_table = defaultdict(
            lambda: 0
        )  # Uses the same keys of the Q-Table to do visit count.

    def get_action_probabilities(self, encoded_state: int) -> np.ndarray:
        """
        Calculate action probabilities for a given state using adaptive temperature
        based on the Q-value distribution.

        :param encoded_state: The current encoded state (int).
        :return: An array of action probabilities for each action (np.ndarray of shape [n_actions]).
        """
        # --- If the state is not well-learned, just take random action ---
        for a in self.all_actions_encoded:
            if self.visit_table[(encoded_state, a)] == 0:
                return np.array([1 / len(self.all_actions_encoded)] * len(self.all_actions_encoded))

        # --- Retrieve Q-values for all actions ---
        q_values = np.array([
            (self.q_table_1[(encoded_state, a)] + self.q_table_2[(encoded_state, a)]) / 2.0
            for a in self.all_actions_encoded
        ])

        # --- Handle all-zero Q-values (fully uninitialized state) ---
        if np.all(q_values == 0):
            # Uniform probabilities if no learning signal yet
            probabilities = np.ones_like(q_values, dtype=float) / len(q_values)
            return probabilities

        # --- Compute Q-value range (or other dispersion measure) ---
        q_max = np.max(q_values)
        q_min = np.min(q_values)
        delta_q = q_max - q_min

        # --- Compute adaptive temperature ---
        eps = 1e-8  # To avoid division by zero
        temperature = self.temperature_scale / (delta_q + eps)

        # --- Clip temperature to stay within [T_min, T_max] ---
        temperature = np.clip(temperature, self.T_min, self.T_max)

        # --- Softmax action selection ---
        q_values_stable = q_values - q_max  # for numerical stability
        exp_values = np.exp(q_values_stable / temperature)
        probabilities = exp_values / (np.sum(exp_values) + 1e-10)  # Avoid divide by zero

        # print(temperature, q_values, probabilities)

        return probabilities

    def choose_action(
        self, state: Union[np.ndarray, List[np.ndarray]], greedy: bool = False
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Select an action (or batch of actions) for the given state(s), using state-specific temperature (softmax).
        If greedy is True, always select the highest probability action.

        :param state: The current observation(s). Single state (np.ndarray) or batch of states (list or np.ndarray).
        :param greedy: Whether to use greedy action selection.
        :return: Action formatted for the environment (decoded if necessary).
                 Single action (np.ndarray) or batch of actions (list of np.ndarray)
        """

        # --- Detect batched observations directly ---
        is_discrete_obs = isinstance(self.observation_space, spaces.Discrete)
        if is_discrete_obs:
            # If Discrete observation_space, batch if ndim == 1 and len > 1
            is_batched = (
                isinstance(state, np.ndarray) and state.ndim == 1 and len(state) > 1
            )
        else:
            # Otherwise batch if ndim >= 2
            is_batched = isinstance(state, np.ndarray) and state.ndim >= 2

        # --- Handle batched observations ---
        if is_batched:
            actions = []
            for idx in range(len(state)):
                action = self.choose_action(state[idx], greedy=greedy)
                actions.append(action[0])
            return actions

        # --- Single observation case ---
        state_encoded = self.state_discretizer.encode_indices(
            [*self.state_discretizer.discretize(state)[1]]
        )
        action_probabilities = self.get_action_probabilities(state_encoded)

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

        if not isinstance(action, list):
            action = [action]

        return action

    def predict(
        self,
        observation: Union[np.ndarray, dict[str, np.ndarray]],
        state: Optional[tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> tuple[Union[ndarray, list[ndarray]], Optional[tuple[ndarray, ...]]]:
        """
        Alias for choose_action() to align with SB3 predict() interface.
        :param observation: the input observation.
        :param state: Not used.
        :param episode_start: Not used.
        :param deterministic: Whether or not to return deterministic actions.
        :return: Selected action(s).
        """
        return self.choose_action(observation, deterministic), state

    def update(
        self,
        state_encoded: int,
        action_encoded: int,
        reward: float,
        next_state_encoded: int,
        done: bool,
        is_weight: float = 1.0,  # Importance-sampling weight, default no correction
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
                key=lambda a: q_update[(next_state_encoded, a)],
            )
            td_target = (
                reward + self.gamma * q_target[(next_state_encoded, best_next_action)]
            )

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
        callback: Union[None, Callable, list, BaseCallback] = None,
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):
        """
        Double Q-Learning loop with Replay Buffer (PER), IS correction, and VDBE temperature.
        Fully Offline version (no online updates).
        """

        # --- Reset internal counter if requested ---
        if reset_num_timesteps:
            self.num_timesteps = 0

        # === Callback preparation ===
        callback = self._init_callback(callback)  # Wrap into CallbackList if needed
        locals_ = locals()
        globals_ = globals()

        # --- Callback training start ---
        callback.init_callback(self)
        callback.on_training_start(locals_=locals_, globals_=globals_)

        # === Environment reset ===
        state = self.env.reset()
        episode_step_count = 0
        num_episodes = 0
        num_terminated = 0
        sum_episode_rewards = 0

        # === Progress bar ===
        pbar = (
            tqdm.tqdm(
                total=total_timesteps, desc="Training", unit="step", dynamic_ncols=True
            )
            if progress_bar
            else None
        )

        update_steps = self.update_steps
        batch_update_interval = self.update_interval

        # === Main training loop ===
        for timestep in range(total_timesteps):

            # --- Detect batched observation directly ---
            is_discrete_obs = isinstance(self.observation_space, spaces.Discrete)
            if is_discrete_obs:
                is_batched = (
                    isinstance(state, np.ndarray) and state.ndim == 1 and len(state) > 1
                )
            else:
                is_batched = isinstance(state, np.ndarray) and state.ndim >= 2

            # --- Encode current state(s) ---
            if is_batched:
                state_encoded_batch = []
                for s in state:
                    if is_discrete_obs:
                        state_encoded = int(s)
                    else:
                        state_encoded = self.state_discretizer.encode_indices(
                            [*self.state_discretizer.discretize(s)[1]]
                        )
                    state_encoded_batch.append(state_encoded)
                state_encoded = state_encoded_batch
            else:
                if is_discrete_obs:
                    state_encoded = int(state)
                else:
                    state_encoded = self.state_discretizer.encode_indices(
                        [*self.state_discretizer.discretize(state)[1]]
                    )

            # --- Select action(s) ---
            actions = self.choose_action(state)

            # --- Step the environment ---
            next_state, reward, terminated, info = self.env.step(actions)

            # --- Encode next_state(s) ---
            if is_batched:
                next_state_encoded_batch = []
                for s in next_state:
                    if is_discrete_obs:
                        next_state_encoded = int(s)
                    else:
                        next_state_encoded = self.state_discretizer.encode_indices(
                            [*self.state_discretizer.discretize(s)[1]]
                        )
                    next_state_encoded_batch.append(next_state_encoded)
                next_state_encoded = next_state_encoded_batch
            else:
                if is_discrete_obs:
                    next_state_encoded = int(next_state)
                else:
                    next_state_encoded = self.state_discretizer.encode_indices(
                        [*self.state_discretizer.discretize(next_state)[1]]
                    )

            # --- Add transition(s) to replay buffer ---
            if is_batched:
                for idx in range(len(state)):
                    transition = Transition(
                        state=state_encoded[idx],
                        action=self.action_discretizer.encode_indices(
                            [*self.action_discretizer.discretize(actions[idx])[1]]
                        ),
                        reward=reward[idx],
                        next_state=next_state_encoded[idx],
                        done=terminated[idx],
                    )
                    self.replay_buffer.add(transition)
            else:
                transition = Transition(
                    state=state_encoded,
                    action=self.action_discretizer.encode_indices(
                        [*self.action_discretizer.discretize(actions)[1]]
                    ),
                    reward=reward,
                    next_state=next_state_encoded,
                    done=terminated,
                )
                self.replay_buffer.add(transition)

            # --- Periodic batch update ---
            if (
                self.num_timesteps % batch_update_interval == 0
                and len(self.replay_buffer.buffer) >= update_steps
                and self.replay_buffer.size >= self.learning_starts
            ):
                transitions, indices, weights = self.replay_buffer.sample(
                    batch_size=update_steps,
                )

                batch_td_errors = []
                for trans, is_weight in zip(transitions, weights):
                    td_error_batch = self.update(
                        state_encoded=trans.state,
                        action_encoded=trans.action,
                        reward=trans.reward,
                        next_state_encoded=trans.next_state,
                        done=trans.done,
                        is_weight=is_weight,
                    )
                    batch_td_errors.append(td_error_batch)

                # --- Update priorities ---
                self.replay_buffer.update_priorities(indices, batch_td_errors)

            # --- Move to next state ---
            state = next_state

            # --- Episode statistics ---
            episode_step_count += 1
            sum_episode_rewards += reward if not is_batched else np.mean(reward)

            # === Rollout/episode termination handling ===
            is_batched = isinstance(terminated, (list, np.ndarray))

            if is_batched:
                terminations_in_batch = np.sum(terminated)

                if terminations_in_batch > 0:
                    num_episodes += terminations_in_batch
                    num_terminated += terminations_in_batch

                    # callback on episode/rollout end
                    callback.n_episodes = num_episodes

                    callback.on_rollout_end()

            else:
                if terminated:
                    episode_step_count = 0
                    num_episodes += 1
                    num_terminated += int(terminated)

                    state = self.env.reset()

                    callback.n_episodes = num_episodes
                    if not callback.on_rollout_end():
                        print("Training aborted by callback (on_rollout_end).")
                        break

            # === Callback on every step ===
            callback.num_timesteps = self.num_timesteps
            if not callback.on_step():
                print("Training aborted by callback (on_step).")
                break

            # === Progress bar update ===
            if progress_bar:
                # pbar.set_postfix({
                #     "Episodes": num_episodes,
                #     "Terminated": num_terminated,
                #     "Avg Episode Reward": (
                #         sum_episode_rewards / num_episodes if num_episodes > 0 else 0.0
                #     ),
                # })
                pbar.update(1)

            # === Time step increment ===
            self.num_timesteps += 1

        # === End of training ===
        if progress_bar:
            pbar.close()

        callback.on_training_end()

        return self

    def _init_callback(self, callback):
        """
        Prepare the callback object.
        :param callback: None, BaseCallback, list, or callable.
        :return: CallbackList/BaseCallback
        """
        # --- If it's a list ---
        if isinstance(callback, list):
            callback_list = []
            for cb in callback:
                if isinstance(cb, BaseCallback):
                    callback_list.append(cb)
                elif callable(cb):
                    callback_list.append(FunctionCallback(cb))
                else:
                    raise ValueError("Unsupported callback type in list!")
            callback = CallbackList(callback_list)

        # --- Single callable ---
        elif callable(callback):
            callback = FunctionCallback(callback)

        # --- Single BaseCallback ---
        elif callback is None:
            callback = CallbackList([])

        # --- Init callback with agent reference ---
        callback.init_callback(self)
        return callback


class EvalCallback(BaseCallback):
    """
    General evaluation callback for tabular/actor-critic algorithms.
    Supports evaluation on VecEnv, best model saving, logging, plotting, and GIF recording.
    """

    def __init__(
        self,
        eval_env,
        eval_interval: int,
        eval_episodes: int = 10,
        save_dir: Optional[str] = "./eval_results",
        model_name: Optional[str] = "agent",
        verbose: int = 1,
        near_optimal_score: Optional[float] = None,
        gif_env: Union[Env, VecEnv, None] = None,
        gif_fps: int = 30,
    ):
        super().__init__(verbose)

        # === Auto-wrap single env into DummyVecEnv ===
        if hasattr(eval_env, "num_envs"):
            self.eval_env = eval_env
        else:
            self.eval_env = DummyVecEnv([lambda: eval_env])

        self.eval_interval = eval_interval
        self.eval_episodes = eval_episodes
        self.gif_fps = gif_fps
        self.record_gif = gif_env is not None
        if gif_env is not None:
            if hasattr(eval_env, "num_envs"):
                self.gif_env = gif_env
            else:
                self.gif_env = DummyVecEnv([lambda: gif_env])

        # === Directory and filenames ===
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.model_name = model_name
        self.best_model_save_path = os.path.join(self.save_dir, f"{self.model_name}.zip")
        self.log_path = os.path.join(self.save_dir, f"{self.model_name}_evaluation_determined_log.csv")
        self.log_path_ = os.path.join(self.save_dir, f"{self.model_name}_evaluation_non_determined_log.csv")
        self.gif_path = os.path.join(self.save_dir, f"{self.model_name}_eval.gif")

        self.near_optimal_score = near_optimal_score
        self.best_mean_reward = -np.inf
        self.step_reached_optimal = None
        self.records_determined = []
        self.records_non_determined = []

    def _on_training_start(self):
        self.records_determined = []
        self.records_non_determined = []
        self.step_reached_optimal = None
        if self.verbose > 0:
            print("[EvalCallback] Starting evaluation callback.")

    def _on_step(self):
        # --- Trigger evaluation ---
        if self.num_timesteps % self.eval_interval != 0:
            return True

        mean_reward, std_reward = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.eval_episodes,
                deterministic=True,
                render=False,
                warn=self.verbose > 0,
            )

        mean_reward_, std_reward_ = evaluate_policy(
            self.model,
            self.eval_env,
            n_eval_episodes=self.eval_episodes,
            deterministic=False,
            render=False,
            warn=self.verbose > 0,
        )

        # === Logging ===
        if self.verbose > 0:
            print(
                f"[EvalCallback] Step {self.num_timesteps} | "
                f"Mean reward (determined): {mean_reward:.2f} ± {std_reward:.2f}"
            )
            print(
                f"[EvalCallback] Step {self.num_timesteps} | "
                f"Mean reward (not determined): {mean_reward_:.2f} ± {std_reward_:.2f}"
            )

        self.records_determined.append((self.num_timesteps, mean_reward, std_reward))
        self.records_non_determined.append((self.num_timesteps, mean_reward_, std_reward_))

        # === Save best model ===
        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            if self.verbose > 0:
                print(f"[EvalCallback] New best mean reward: {mean_reward:.2f}. Saving model to {self.best_model_save_path}")
            self.model.save(self.best_model_save_path)
            self.record_eval_gif(
                save_path=self.gif_path,
                fps=self.gif_fps,
                deterministic=True,
                verbose=self.verbose,
            )

        # === Check optimal score ===
        if self.near_optimal_score is not None and mean_reward >= self.near_optimal_score:
            if self.step_reached_optimal is None:
                self.step_reached_optimal = self.num_timesteps

        return True

    def _on_training_end(self):
        if self.verbose > 0:
            print("[EvalCallback] Training finished. Saving logs.")
        df = pd.DataFrame(
            self.records_determined, columns=["Timesteps", "MeanReward", "StdReward"]
        )
        df_ = pd.DataFrame(
            self.records_non_determined, columns=["Timesteps", "MeanReward", "StdReward"]
        )
        if self.log_path is not None:
            df.to_csv(self.log_path, index=False)
            df_.to_csv(self.log_path_, index=False)
            if self.verbose > 0:
                print(f"[EvalCallback] Logs saved at: {self.log_path} and {self.log_path_}")

    def record_eval_gif(
            self,
            save_path: str,
            fps: int = 30,
            deterministic: bool = True,
            verbose: int = 1
    ):
        """
        Record a single episode as a GIF, from a DummyVecEnv (num_envs=1).

        Args:
            save_path (str): Path to save the GIF.
            fps (int): Frames per second for GIF playback.
            deterministic (bool): Whether to use deterministic actions.
            verbose (int): Verbosity level.
        """
        frames = []

        # === 1. Ensure gif_env is DummyVecEnv ===
        if not hasattr(self.gif_env, "num_envs") or not isinstance(self.gif_env, DummyVecEnv):
            raise ValueError("record_eval_gif requires a DummyVecEnv (num_envs=1) as gif_env.")

        if self.gif_env.num_envs != 1:
            raise ValueError(
                f"record_eval_gif only supports DummyVecEnv with 1 env, got num_envs={self.gif_env.num_envs}.")

        env = self.gif_env
        raw_env = env.envs[0]  # single raw env (for rendering)

        # === 2. Reset VecEnv ===
        obs = env.reset()

        # === 3. Initialize done flag ===
        done = np.array([False])  # VecEnv always returns batch results
        step_count = 0

        # === 4. Start recording episode ===
        while not done[0]:
            # --- Render raw env (avoid multi-frame issue) ---
            frame = raw_env.render()

            # --- Optional resize ---
            frame_resized = cv2.resize(
                frame,
                (frame.shape[1] // 2, frame.shape[0] // 2),
                interpolation=cv2.INTER_AREA
            )
            frames.append(frame_resized)

            # --- Select batched actions ---
            action = self.model.choose_action(obs, greedy=deterministic)

            # --- Step VecEnv ---
            obs, rewards, done, info = env.step(action)

            step_count += 1

        # === 5. Save GIF ===
        duration_per_frame = 1.0 / fps
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        imageio.mimsave(save_path, frames, duration=duration_per_frame, loop=0)

        if verbose > 0:
            print(f"[record_eval_gif] DummyVecEnv episode finished after {step_count} steps.")
            print(f"[record_eval_gif] GIF saved at: {save_path}")
