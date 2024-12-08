from collections import defaultdict
from itertools import product, combinations
from typing import List
import numpy as np
import pandas as pd


class Qtableagent:
    def __init__(self, action_space: List[int]):
        """
        Initialize the Q-Table agent.

        :param action_space: List of possible values for each action dimension.
        """
        self.q_table = defaultdict(lambda: 0.0)  # Flattened Q-Table with state-action tuple keys
        self.action_space = action_space
        self.print_q_table_info()

    def clone(self) -> 'Qtableagent':
        """
        Create a deep copy of the Q-Table agent.

        :return: A new QTableAgent instance with the same Q-Table.
        """
        new_agent = Qtableagent(self.action_space)
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
    def load_q_table(cls, file_path: str = None, df: pd.DataFrame = None) -> "Qtableagent":
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
        print(f"Q-Table loaded from {f'{file_path}' if file_path else 'DataFrame'}.")
        agent.print_q_table_info()
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

        # print(f"Updated Q-value for state {state_key}, action {action}: {self.q_table[(state_key, tuple(action))]}")

# Example Usage
if __name__ == "__main__":
    import random
    import gymnasium as gym
    from utils import merge_q_table_with_counts, compute_action_probabilities, compute_average_kl_divergence_between_dfs, \
    compute_mutual_information
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

    agent = Qtableagent(action_space=[len(action_ranges[i]) for i in range(len(action_ranges))])

    # Train the agent
    state, info = wrapped_env.reset()
    for episode in range(1000):
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

    # Train the agent
    state, info = wrapped_env.reset()
    for episode in range(1000):
        done = False
        while not done:
            action = random.choice(agent._generate_all_possible_actions())  # Choose a random action
            next_state, reward, done, truncated, _ = wrapped_env.step(
                action[0])  # Use the first action dimension for CartPole
            agent.update(state, action, reward, next_state)
            state = next_state if not done else wrapped_env.reset()[0]

    # Save Q-table and simulate wrapper counts
    q_table_df_ = agent.save_q_table("./q_table.csv")
    print("Saved Q-Table DataFrame:")
    print(q_table_df_.head())

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

    merged_df_ = merge_q_table_with_counts(q_table_df_, counts_df)
    merged_df_.to_csv("merged_test.csv", index=False)
    print("Merged Q-Table with Counts:")
    print(merged_df_.head())

    merged_df_ = compute_action_probabilities(merged_df_, "softmax")
    print("Action Probabilities:")
    print(merged_df_.head())

    # List of all state features
    state_features = ["state_dim_0", "state_dim_1", "state_dim_2", "state_dim_3"]

    # Iterate over all possible combinations of features
    for r in range(1, len(state_features) + 1):  # r is the number of features in each combination
        for feature_combination in combinations(state_features, r):
            # Compute mutual information for the current feature combination
            mi = compute_mutual_information(
                merged_df, list(feature_combination), "action_dim_0", use_visit_count=True
            )
            # Print the results with detailed explanation
            print(f"Mutual Information for Features {feature_combination} with action_dim_0: {mi}")

    # Compute mutual information for all features as values
    mi = compute_mutual_information(
        merged_df, state_features, "action_dim_0", use_visit_count=True
    )
    print(f"Mutual Information for All Features as Values with action_dim_0: {mi}")

    # Load agent from merged DataFrame
    loaded_agent_ = Qtableagent.load_q_table(df=q_table_df_)
    loaded_agent = Qtableagent.load_q_table(df=q_table_df)

    kl = compute_average_kl_divergence_between_dfs(merged_df_, merged_df, weighted_by_visitation=True)
    print(f"KL Divergence: {kl}")

    # Print loaded Q-table for verification
    loaded_agent.print_q_table_info()

    # Test action probabilities
    test_state = np.array([0.1, 0.0, -0.1, 0.0])
    print("Action probabilities (greedy):", loaded_agent.get_action_probabilities(test_state, strategy="greedy"))
    print("Action probabilities (softmax):", loaded_agent.get_action_probabilities(test_state, strategy="softmax", temperature=1.0))
