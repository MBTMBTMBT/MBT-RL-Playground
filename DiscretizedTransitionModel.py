from collections import defaultdict
from itertools import product
import numpy as np
import scipy.stats
from typing import List, Tuple, Optional, Dict

from gymnasium import spaces
from networkx.classes import DiGraph
import torch
from torch import nn
from torch.nn import functional as F


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
        Add noise to the input vector. The noise is uniformly sampled within the range
        of the corresponding bucket size for dimensions with buckets > 1.

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
                # Discretized dimension: Calculate bucket size and add noise
                bucket_size = (max_val - min_val) / buckets
                noise = np.random.uniform(-bucket_size / 2, bucket_size / 2)
                noisy_vector[i] = np.clip(value + noise, min_val, max_val)

        return noisy_vector


class TransitionTable:
    def __init__(
        self,
        state_discretizer: Discretizer,
        action_discretizer: Discretizer,
        reward_resolution: int = -1,
    ):
        self.state_discretizer = state_discretizer
        self.action_discretizer = action_discretizer
        self.transition_table: Dict[int, Dict[int, Dict[int, Dict[float, int]]]] = (
            defaultdict(
                lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
            )
        )  # {state: {action: {next_state: {reward: count}}}
        self.neighbour_dict = defaultdict(lambda: set())
        self.forward_dict = defaultdict(lambda: defaultdict(lambda: set()))
        self.inverse_dict = defaultdict(lambda: defaultdict(lambda: set()))

        # They will not be saved!
        self.state_count = defaultdict(lambda: 0)
        self.state_action_count = defaultdict(lambda: defaultdict(lambda: 0))
        self.transition_prob_table: Dict[int, Dict[int, Dict[int, float]]] = (
            defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
        )  # {state: {action: {next_state: rate}}
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
        for reward, reward_set in sorted(
            self.reward_set_dict.items(), key=lambda x: x[0]
        ):
            print(
                f"{reward}: {len(reward_set)} - {len(reward_set) / total_reward_count * 100:.2f}%"
            )

    def update(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        encoded_state = self.state_discretizer.encode_indices(
            list(self.state_discretizer.discretize(state)[1])
        )
        encoded_next_state = self.state_discretizer.encode_indices(
            list(self.state_discretizer.discretize(next_state)[1])
        )
        encoded_action = self.action_discretizer.encode_indices(
            list(self.action_discretizer.discretize(action)[1])
        )

        if done:
            self.done_set.add(encoded_next_state)

        self.transition_table[encoded_state][encoded_action][encoded_next_state][
            round(reward, 1)
        ] += 1
        self.neighbour_dict[encoded_state].add(encoded_action)
        self.neighbour_dict[encoded_next_state].add(encoded_action)
        self.forward_dict[encoded_state][encoded_next_state].add(encoded_action)
        self.inverse_dict[encoded_next_state][encoded_state].add(encoded_action)
        self.state_count[encoded_state] += 1
        self.state_action_count[encoded_state][encoded_action] += 1

        transition_state_avg_reward_and_prob = (
            self.get_transition_state_avg_reward_and_prob(encoded_state, encoded_action)
        )
        for encoded_next_state, (
            avg_reward,
            prob,
        ) in transition_state_avg_reward_and_prob.items():
            self.transition_prob_table[encoded_state][encoded_action][
                encoded_next_state
            ] = prob

        if self.reward_resolution > 0:
            self.reward_set_dict[
                round(reward / self.reward_resolution) * self.reward_resolution
            ].add(encoded_next_state)
        elif self.reward_resolution < 0:
            self.reward_set_dict[round(reward, abs(self.reward_resolution))].add(
                encoded_next_state
            )
        else:
            self.reward_set_dict[reward].add(encoded_next_state)

    def get_transition_state_avg_reward_and_prob(
        self, encoded_state: int, encoded_action: int
    ) -> Dict[int, Tuple[float, float]]:
        # Transition to state probs: from given state, with given action, probs of getting into next states
        # Avg Reward: from given state, with given action, ending up in certain state, the average reward it gets
        transition_state_reward_and_prob = {}
        _transition_state_reward_and_prob = {}
        encoded_next_states = []
        encoded_next_state_counts = []
        avg_rewards = []
        for encoded_next_state in self.transition_table[encoded_state][
            encoded_action
        ].keys():
            encoded_next_state_count = 0
            _transition_state_reward_and_prob[encoded_next_state] = {}
            rewards = []
            reward_counts = []
            for reward in self.transition_table[encoded_state][encoded_action][
                encoded_next_state
            ].keys():
                reward_count = self.transition_table[encoded_state][encoded_action][
                    encoded_next_state
                ][reward]
                _transition_state_reward_and_prob[encoded_next_state][
                    reward
                ] = reward_count
                encoded_next_state_count += reward_count
                rewards.append(reward)
                reward_counts.append(reward_count)
            avg_rewards.append(np.average(rewards, weights=reward_counts))
            encoded_next_states.append(encoded_next_state)
            encoded_next_state_counts.append(encoded_next_state_count)
        encoded_next_state_probs = np.array(encoded_next_state_counts) / np.sum(
            encoded_next_state_counts
        )
        for encoded_next_state, avg_reward, prob in zip(
            encoded_next_states, avg_rewards, encoded_next_state_probs
        ):
            transition_state_reward_and_prob[encoded_next_state] = (
                float(avg_reward),
                float(prob),
            )
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
        encoded_state = self.state_discretizer.encode_indices(
            list(self.state_discretizer.discretize(state)[1])
        )
        self.start_set.add(encoded_state)

    def get_start_set(self) -> set[int]:
        return self.start_set


class TransitionModel(nn.Module):
    def __init__(
        self,
        state_discretizer: Discretizer,
        action_discretizer: Discretizer,
        network_layers: list = None,
        dropout: float = 0.0,
        lr: float = 1e-4,
        device=torch.device("cpu"),
    ):
        super(TransitionModel, self).__init__()
        self.state_discretizer = state_discretizer
        self.action_discretizer = action_discretizer
        self.obs_space = state_discretizer.get_gym_space()
        self.action_space = action_discretizer.get_gym_space()
        self.obs_dim = state_discretizer.get_space_length()
        self.action_dim = action_discretizer.get_space_length()
        self.network_layers = network_layers if network_layers else [64, 64]
        self.dropout = dropout
        self.device = device

        layers = []
        input_dim = self.obs_dim + self.action_dim  # Concatenate observation and action

        # Add hidden layers
        for i, layer_size in enumerate(self.network_layers):
            layers.append(nn.Linear(input_dim, layer_size))
            layers.append(nn.LayerNorm(layer_size))  # LayerNorm for normalization
            if i < len(self.network_layers) - 1:
                layers.append(nn.LeakyReLU(negative_slope=0.01))
            else:
                layers.append(nn.ReLU())  # ReLU for the last hidden layer
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            input_dim = layer_size

        # Shared feature extractor
        self.feature_extractor = nn.Sequential(*layers)

        # Separate heads for predicting next state, reward, and terminal
        self.next_state_head = nn.Linear(
            self.network_layers[-1], self.obs_dim
        )  # Next state prediction
        self.reward_head = nn.Linear(self.network_layers[-1], 1)  # Reward prediction
        self.terminal_head = nn.Sequential(
            nn.Linear(self.network_layers[-1], 1), nn.Sigmoid()  # Terminal prediction
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.to(self.device)

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        """
        Forward pass of the transition model.

        Args:
            obs (torch.Tensor): Input observations, shape (batch_size, obs_dim).
            action (torch.Tensor): Input actions, shape (batch_size, action_dim).

        Returns:
            tuple: (next_state, reward, terminal)
                - next_state: Predicted next state, shape (batch_size, obs_dim).
                - reward: Predicted reward, shape (batch_size, 1).
                - terminal: Predicted terminal state probability, shape (batch_size, 1).
        """
        x = torch.cat([obs, action], dim=-1)  # Concatenate observations and actions
        features = self.feature_extractor(x)
        next_state = self.next_state_head(features)
        reward = self.reward_head(features)
        terminal = self.terminal_head(features)
        return next_state, reward, terminal

    def normalize_batch(
        self,
        batch,
    ):
        obs_space, action_space = self.obs_space, self.action_space
        # Normalize observations
        true_obs = torch.tensor(batch["state"], dtype=torch.float32, device=self.device)
        if isinstance(obs_space, spaces.Discrete):
            normalized_obs = torch.nn.functional.one_hot(
                true_obs.long(), num_classes=obs_space.n
            ).float()
        elif isinstance(obs_space, spaces.MultiDiscrete):
            split_obs = torch.split(true_obs.long(), 1, dim=-1)
            one_hot_obs_list = [
                torch.nn.functional.one_hot(dim.squeeze(-1), num_classes=n).float()
                for dim, n in zip(split_obs, obs_space.nvec)
            ]
            normalized_obs = torch.cat(one_hot_obs_list, dim=-1)
        elif isinstance(obs_space, spaces.Box):
            low = torch.tensor(obs_space.low, dtype=torch.float32, device=self.device)
            high = torch.tensor(obs_space.high, dtype=torch.float32, device=self.device)
            finite_mask = torch.isfinite(low) & torch.isfinite(high)
            normalized_obs = true_obs.clone()
            normalized_obs[..., finite_mask] = (
                2
                * (true_obs[..., finite_mask] - low[finite_mask])
                / (high[finite_mask] - low[finite_mask])
                - 1
            )
        else:
            raise NotImplementedError(
                f"Observation space {type(obs_space)} not supported."
            )

        next_obs = torch.tensor(
            batch["next_state"], dtype=torch.float32, device=self.device
        )
        if isinstance(obs_space, spaces.Discrete):
            normalized_next_obs = torch.nn.functional.one_hot(
                next_obs.long(), num_classes=obs_space.n
            ).float()
        elif isinstance(obs_space, spaces.MultiDiscrete):
            split_next_obs = torch.split(next_obs.long(), 1, dim=-1)
            one_hot_next_obs_list = [
                torch.nn.functional.one_hot(dim.squeeze(-1), num_classes=n).float()
                for dim, n in zip(split_next_obs, obs_space.nvec)
            ]
            normalized_next_obs = torch.cat(one_hot_next_obs_list, dim=-1)
        elif isinstance(obs_space, spaces.Box):
            low = torch.tensor(obs_space.low, dtype=torch.float32, device=self.device)
            high = torch.tensor(obs_space.high, dtype=torch.float32, device=self.device)
            finite_mask = torch.isfinite(low) & torch.isfinite(high)
            normalized_next_obs = next_obs.clone()
            normalized_next_obs[..., finite_mask] = (
                2
                * (next_obs[..., finite_mask] - low[finite_mask])
                / (high[finite_mask] - low[finite_mask])
                - 1
            )
        else:
            raise NotImplementedError(
                f"Observation space {type(obs_space)} not supported."
            )

        # Normalize actions
        true_actions = torch.tensor(
            batch["action"], dtype=torch.float32, device=self.device
        )
        if isinstance(action_space, spaces.Discrete):
            normalized_actions = torch.nn.functional.one_hot(
                true_actions.long(), num_classes=action_space.n
            ).float()
        elif isinstance(action_space, spaces.MultiDiscrete):
            split_actions = torch.split(true_actions.long(), 1, dim=-1)
            one_hot_action_list = [
                torch.nn.functional.one_hot(dim.squeeze(-1), num_classes=n).float()
                for dim, n in zip(split_actions, action_space.nvec)
            ]
            normalized_actions = torch.cat(one_hot_action_list, dim=-1)
        elif isinstance(action_space, spaces.Box):
            normalized_actions = true_actions
        else:
            raise NotImplementedError(
                f"Action space {type(action_space)} not supported."
            )

        # Process rewards and terminations
        true_rewards = torch.tensor(
            batch["reward"], dtype=torch.float32, device=self.device
        )
        true_terminations = torch.tensor(
            batch["terminal"], dtype=torch.float32, device=self.device
        )

        # Return the processed batch
        return {
            "state": normalized_obs,
            "next_state": normalized_next_obs,
            "action": normalized_actions,
            "reward": true_rewards,
            "terminal": true_terminations,
        }

    def train_batch(
        self,
        batch,
        recon_weight=1.0,
        reward_weight=1.0,
        termination_weight=1.0,
    ):
        self.train()

        # Convert batch data to tensors
        true_obs = torch.tensor(batch["state"], dtype=torch.float32, device=self.device)
        next_obs = torch.tensor(
            batch["next_state"], dtype=torch.float32, device=self.device
        )
        true_actions = torch.tensor(
            batch["action"], dtype=torch.float32, device=self.device
        )
        true_rewards = torch.tensor(
            batch["reward"], dtype=torch.float32, device=self.device
        )
        true_terminations = torch.tensor(
            batch["terminal"], dtype=torch.float32, device=self.device
        )
        masks = torch.tensor(batch["mask"], dtype=torch.float32, device=self.device)

        (
            batch_size,
            seq_len,
            _,
        ) = true_obs.size()
        time_weights = torch.linspace(1.0, 0.1, steps=seq_len, device=self.device)

        # Initialize loss accumulators as tensors
        recon_loss = torch.tensor(0.0, device=self.device)
        reward_loss = torch.tensor(0.0, device=self.device)
        termination_loss = torch.tensor(0.0, device=self.device)

        state = true_obs[:, 0]
        for t in range(seq_len):
            next_state, reward, terminal = self.forward(state, true_actions[:, t])

            # Reconstruction loss against next observation
            recon_loss += (
                F.mse_loss(
                    next_state * masks[:, t].unsqueeze(-1).unsqueeze(-1),
                    next_obs[:, t] * masks[:, t].unsqueeze(-1).unsqueeze(-1),
                )
                * time_weights[t]
            )
            reward_loss += (
                F.mse_loss(
                    reward.squeeze() * masks[:, t].unsqueeze(-1),
                    true_rewards[:, t] * masks[:, t].unsqueeze(-1),
                    reduction="mean",
                )
                * time_weights[t]
            )
            termination_loss += (
                F.binary_cross_entropy_with_logits(
                    terminal.squeeze() * masks[:, t].unsqueeze(-1),
                    true_terminations[:, t] * masks[:, t].unsqueeze(-1),
                    reduction="mean",
                )
                * time_weights[t]
            )
            state = next_state

        # Total loss
        total_loss = (
            recon_weight * recon_loss
            + reward_weight * reward_loss
            + termination_weight * termination_loss
        )

        # Backpropagation
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {
            "total_loss": total_loss.item(),
            "recon_loss": recon_loss.item(),
            "reward_loss": reward_loss.item(),
            "termination_loss": termination_loss.item(),
        }

    def predict(self, state: np.ndarray, action: np.ndarray or int):
        pass
