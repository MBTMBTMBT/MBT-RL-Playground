from collections import defaultdict
from itertools import product
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import scipy
from gymnasium import spaces
import gymnasium as gym


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

    def set_env(self, env: gym.Env):
        self.env = env

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
                if episode_step_count >= self.n_steps:
                    episode_step_count = 0
                    num_episodes += 1
                    state, info = self.env.reset()
            action = self.choose_action(state, temperature=temperature)
            next_state, reward, terminated, truncated, info = self.env.step(action)
            self.update(
                state,
                action,
                reward,
                next_state,
                terminated,
                alpha=self.lr,
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
