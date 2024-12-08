import numpy as np
import pandas as pd


def merge_q_table_with_counts(q_table_df: pd.DataFrame, counts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge the Q-Table DataFrame with the wrapper's state-action counts DataFrame.

    :param q_table_df: DataFrame containing the Q-Table (state, action, q_value).
    :param counts_df: DataFrame containing state-action counts (state, action, count).
    :return: A merged DataFrame containing states, actions, Q-values, and counts.
    """
    # Identify state and action columns
    state_columns = sorted([col for col in q_table_df.columns if col.startswith("state_dim_")])
    action_columns = sorted([col for col in q_table_df.columns if col.startswith("action_dim_")])

    # Merge on state and action columns
    merged_df = pd.merge(
        q_table_df,
        counts_df,
        on=state_columns + action_columns,
        how="inner",  # Use "inner" to keep only matching rows
        suffixes=('_q_table', '_counts')
    )

    # Keep only relevant columns and rename for clarity
    merged_df = merged_df[state_columns + action_columns + ["q_value", "count"]]

    return merged_df

def compute_action_probabilities(
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
    # state_index_columns = [col for col in df.columns if col.startswith("State_") and "Index" in col]
    state_value_columns = [col for col in df.columns if col.startswith("state_dim")]
    action_index_columns = [col for col in df.columns if col.startswith("action_dim")]

    if not action_index_columns:
        raise ValueError("No valid action index columns detected in the input DataFrame.")

    # Group by unique state and calculate probabilities for each action
    grouped = df.groupby(state_value_columns)  # df.groupby(state_index_columns + state_value_columns)

    # Dynamically generate all possible action combinations
    unique_action_combinations = np.array(
        np.meshgrid(*[df[col].unique() for col in action_index_columns])
    ).T.reshape(-1, len(action_index_columns))
    all_action_indices = [tuple(row) for row in unique_action_combinations]

    action_column_names = [
        f"{'_'.join(f'{col}_{value}' for col, value in zip(action_index_columns, action))}_Probability"
        for action in all_action_indices
    ]
    # new_columns = state_index_columns + state_value_columns + action_column_names + ["Visit_Count"]
    if "count" in df.columns:
        new_columns = state_value_columns + action_column_names + ["count"]
    else:
        new_columns = state_value_columns + action_column_names

    new_data = []

    for group_keys, group in grouped:
        # Dynamically unpack state indices and values
        # state_indices = group_keys[:len(state_index_columns)]
        state_values = group_keys[:]

        q_values = group["q_value"].values
        actions = group[action_index_columns].to_numpy()
        if "count" in group.columns:
            visit_count = group["count"].sum()  # Sum visit counts for this state

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
        # row = list(state_indices) + list(state_values) + list(probabilities) + [visit_count]
        if "count" in group.columns:
            row = list(state_values) + list(probabilities) + [visit_count]
        else:
            row = list(state_values) + list(probabilities)
        new_data.append(row)

    # Create and return the new DataFrame
    new_df = pd.DataFrame(new_data, columns=new_columns)
    return new_df
