from typing import Union, Optional, List

import numpy as np
import pandas as pd

from dqn_agent import DQNAgent


def merge_q_table_with_counts(
    q_table_df: pd.DataFrame, counts_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge the Q-Table DataFrame with the wrapper's state-action counts DataFrame.

    :param q_table_df: Sparse DataFrame containing the Q-Table (state, action, q_value).
    :param counts_df: Dense DataFrame containing state-action counts (state, action, count).
    :return: A merged DataFrame containing states, actions, Q-values, and counts.
    """
    # Identify state and action columns
    state_columns = sorted(
        [col for col in counts_df.columns if col.startswith("state_dim_")]
    )
    action_columns = sorted(
        [col for col in counts_df.columns if col.startswith("action_dim_")]
    )

    # Convert sparse Q-Table to dense for merging
    dense_q_table_df = q_table_df.copy()
    for col in state_columns + action_columns:
        dense_q_table_df[col] = dense_q_table_df[col].sparse.to_dense()

    # Merge on state and action columns
    merged_df = pd.merge(
        dense_q_table_df,
        counts_df,
        on=state_columns + action_columns,
        how="inner",  # Use "inner" to keep only matching rows
        suffixes=("_q_table", "_counts"),
    )

    # Keep only relevant columns and rename for clarity
    merged_df = merged_df[state_columns + action_columns + ["q_value", "count"]]

    return merged_df


def sample_q_table_with_counts(
    agent: DQNAgent, counts_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Sample Q-values from an agent using the states in counts_df and merge with the counts.

    :param agent: An agent with a `get_action_probabilities` method.
    :param counts_df: DataFrame containing state-action counts (state, action, count).
    :return: A merged DataFrame containing states, actions, Q-values, and counts.
    """
    # Identify state columns
    state_columns = sorted(
        [col for col in counts_df.columns if col.startswith("state_dim_")]
    )

    # Create a DataFrame to store sampled Q-values
    sampled_data = []

    for _, row in counts_df.iterrows():
        state = np.array([row[col] for col in state_columns])
        q_values = agent.get_action_probabilities(state, strategy="softmax")

        for action_idx, q_value in enumerate(q_values):
            sampled_row = {col: row[col] for col in state_columns}
            sampled_row[
                f"action_dim_0"
            ] = action_idx  # Assuming single-dimensional action
            sampled_row["q_value"] = q_value
            sampled_row["count"] = row["count"]
            sampled_data.append(sampled_row)

    sampled_df = pd.DataFrame(sampled_data)
    return sampled_df


def compute_action_probabilities(
    df: pd.DataFrame,
    strategy: str = "greedy",
    epsilon: float = 0.0,
    temperature: float = 1.0,
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
        raise ValueError(
            "No valid action index columns detected in the input DataFrame."
        )

    # Group by unique state and calculate probabilities for each action
    grouped = df.groupby(
        state_value_columns
    )  # df.groupby(state_index_columns + state_value_columns)

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
        # print(q_values)
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
            probabilities[
                all_action_indices.index(tuple(actions[best_action_idx]))
            ] += (1.0 - epsilon)
        elif strategy == "softmax":
            exp_q_values = np.exp(q_values / temperature)
            softmax_probs = exp_q_values / np.sum(exp_q_values)
            probabilities = np.zeros(len(all_action_indices), dtype=float)
            # print(q_values)
            # print(softmax_probs)
            for i, action in enumerate(actions):
                action_tuple = tuple(action)
                action_idx = all_action_indices.index(action_tuple)
                probabilities[action_idx] = softmax_probs[i]
            # print(probabilities)

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


def compute_mutual_information(
    df: pd.DataFrame,
    group1_columns: Union[str, List[str]],
    group2_columns: Optional[Union[str, List[str]]] = None,
    use_visit_count: bool = False,
) -> float:
    """
    Compute mutual information (MI) between two groups of features or between features and actions.

    :param df: A DataFrame containing the precomputed probability distributions and optionally visit counts.
    :param group1_columns: A single column or a list of columns representing the first group (e.g., features).
    :param group2_columns: A single column, a list of columns, or None. If None, computes self-MI for group1.
                           If the group represents actions, provide the base names, e.g., "Action_0_Index".
    :param use_visit_count: Whether to weight probabilities by visit counts. Default is False.
    :return: The mutual information (MI) value between the two groups.
    """
    # Make a copy of the DataFrame to avoid modifying the original
    df = df.copy()

    # Ensure group1_columns is a list
    if isinstance(group1_columns, str):
        group1_columns = [group1_columns]

    # Process the second group (group2_columns)
    if group2_columns is None:
        group2_columns = group1_columns  # Compute self-MI within group1
    elif isinstance(group2_columns, str):
        group2_columns = [group2_columns]

    # Check if the second group contains actions
    is_action_group = all(col.startswith("action_dim_") for col in group2_columns)

    if is_action_group:
        # If the group is actions, find corresponding probability columns
        full_action_columns = []
        for base_col in group2_columns:
            matching_columns = [
                col
                for col in df.columns
                if col.startswith(base_col) and "_Probability" in col
            ]
            if not matching_columns:
                raise ValueError(
                    f"Invalid or missing action columns for base: {base_col}"
                )
            full_action_columns.extend(matching_columns)
        group2_columns = full_action_columns

    # Check for visit counts if `use_visit_count` is enabled
    if use_visit_count and "count" not in df.columns:
        raise ValueError(
            "Visit count column ('count') is required when 'use_visit_count=True'."
        )

    # Normalize probabilities, optionally weighted by visit counts
    if use_visit_count:
        df["Weighted_Visit_Count"] = df["count"] / df["count"].sum()
        if is_action_group:
            for col in group2_columns:
                df[col] *= df["Weighted_Visit_Count"]

    # Normalize the probabilities to ensure valid distribution
    if is_action_group:
        total_prob = df[group2_columns].sum().sum()
        df[group2_columns] /= total_prob
    else:
        # For feature groups, normalize visit counts
        df["Visit_Prob"] = df["count"] / df["count"].sum()

    # Compute joint probability P(Group1, Group2)
    if is_action_group:
        joint_prob = df.groupby(group1_columns, as_index=False)[group2_columns].sum()
    else:
        # For feature groups, compute joint probabilities using Visit_Prob
        joint_prob = (
            df.groupby(group1_columns + group2_columns)["Visit_Prob"]
            .sum()
            .reset_index()
        )
        joint_prob.rename(columns={"Visit_Prob": "P(Group1,Group2)"}, inplace=True)

    # Compute marginal probabilities P(Group1) and P(Group2)
    if is_action_group:
        joint_prob["P(Group1)"] = joint_prob[group2_columns].sum(axis=1)
        marginal_group2_prob = df[group2_columns].sum()
    else:
        # For feature groups, compute marginals
        marginal_group1_prob = (
            joint_prob.groupby(group1_columns)["P(Group1,Group2)"].sum().reset_index()
        )
        marginal_group1_prob.rename(
            columns={"P(Group1,Group2)": "P(Group1)"}, inplace=True
        )

        marginal_group2_prob = (
            joint_prob.groupby(group2_columns)["P(Group1,Group2)"].sum().reset_index()
        )
        marginal_group2_prob.rename(
            columns={"P(Group1,Group2)": "P(Group2)"}, inplace=True
        )

    # Merge joint and marginal probabilities
    if not is_action_group:
        joint_prob = joint_prob.merge(marginal_group1_prob, on=group1_columns)
        joint_prob = joint_prob.merge(marginal_group2_prob, on=group2_columns)

    # Calculate mutual information
    mi = 0.0
    if is_action_group:
        for _, row in joint_prob.iterrows():
            px = row["P(Group1)"]  # P(Group1)
            for group2_col in group2_columns:
                pa = marginal_group2_prob[group2_col]  # P(Group2)
                p_group1_group2 = row[group2_col]  # P(Group1, Group2)
                if p_group1_group2 > 0:  # Avoid log(0)
                    mi += p_group1_group2 * np.log2(p_group1_group2 / (px * pa))
    else:
        for _, row in joint_prob.iterrows():
            p_group1_group2 = row["P(Group1,Group2)"]
            px = row["P(Group1)"]
            py = row["P(Group2)"]
            if p_group1_group2 > 0:  # Avoid log(0)
                mi += p_group1_group2 * np.log2(p_group1_group2 / (px * py))

    return mi


def compute_average_kl_divergence_between_dfs(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    visit_threshold: int = 0,
    weighted_by_visitation: bool = False,
) -> float:
    """
    Compute the average KL divergence between action distributions in two DataFrames for shared states.
    The filtering and optional weighting are based on the first DataFrame's visitation count.

    :param df1: The first DataFrame, generated using `compute_action_probabilities`.
    :param df2: The second DataFrame, generated using `compute_action_probabilities`.
    :param visit_threshold: Minimum visit count in df1 required for states to be considered. Default is 0.
    :param weighted_by_visitation: Whether to weight KL divergence by df1's visitation distribution. Default is False.
    :return: The average KL divergence for shared states based on action distributions.
    """
    # Identify the state columns
    state_columns = [col for col in df1.columns if col.startswith("state_dim")]

    # Identify action probability columns in both DataFrames
    action_prob_columns_df1 = [
        col for col in df1.columns if col.endswith("_Probability")
    ]
    action_prob_columns_df2 = [
        col for col in df2.columns if col.endswith("_Probability")
    ]

    # Ensure the action columns match between the two DataFrames
    if set(action_prob_columns_df1) != set(action_prob_columns_df2):
        raise ValueError(
            "Action probability columns do not match between the two DataFrames."
        )

    # Merge the two DataFrames on state columns
    merged_df = pd.merge(
        df1[state_columns + action_prob_columns_df1 + ["count"]],
        df2[state_columns + action_prob_columns_df2],
        on=state_columns,
        suffixes=("_df1", "_df2"),
    )

    # Apply visit threshold filter based on df1
    if visit_threshold > 0:
        merged_df = merged_df[merged_df["count"] > visit_threshold]

    # If no states meet the threshold, return 0 to avoid division by zero
    if merged_df.empty:
        return 0.0

    # Compute KL divergence for each shared state
    kl_divergences = []
    visitation_weights = []
    for _, row in merged_df.iterrows():
        # Extract the probabilities from both DataFrames
        p = row[[f"{col}_df1" for col in action_prob_columns_df1]].values
        q = row[[f"{col}_df2" for col in action_prob_columns_df2]].values

        # Avoid division by zero and log(0) by adding a small epsilon
        epsilon = 1e-20
        p = np.clip(p, epsilon, 1.0)
        q = np.clip(q, epsilon, 1.0)

        # Normalize the distributions to ensure they sum to 1
        p /= p.sum()
        q /= q.sum()

        # Compute the KL divergence for this state
        kl_divergence = np.sum(p * np.log(p / q))
        kl_divergences.append(kl_divergence)

        # Append the visit count from df1 if weighting is enabled
        if weighted_by_visitation:
            visitation_weights.append(row["count"])

    # Compute and return the average KL divergence
    if weighted_by_visitation:
        visitation_weights = np.array(visitation_weights) / np.sum(
            visitation_weights
        )  # Normalize weights
        average_kl_divergence = np.sum(np.array(kl_divergences) * visitation_weights)
    else:
        average_kl_divergence = np.mean(kl_divergences)

    return average_kl_divergence
