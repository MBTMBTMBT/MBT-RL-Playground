from typing import Union, Optional, List

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


def compute_mutual_information(
        df: pd.DataFrame,
        group1_columns: Union[str, List[str]],
        group2_columns: Optional[Union[str, List[str]]] = None,
        use_visit_count: bool = False
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
        is_action_group = all(col.startswith("Action_") for col in group2_columns)

        if is_action_group:
            # If the group is actions, find corresponding probability columns
            full_action_columns = []
            for base_col in group2_columns:
                matching_columns = [
                    col for col in df.columns if col.startswith(base_col) and "_Probability" in col
                ]
                if not matching_columns:
                    raise ValueError(f"Invalid or missing action columns for base: {base_col}")
                full_action_columns.extend(matching_columns)
            group2_columns = full_action_columns

        # Check for visit counts if `use_visit_count` is enabled
        if use_visit_count and "count" not in df.columns:
            raise ValueError("Visit count column ('count') is required when 'use_visit_count=True'.")

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
            df["Visit_Prob"] = df["Visit_Count"] / df["Visit_Count"].sum()

        # Compute joint probability P(Group1, Group2)
        if is_action_group:
            joint_prob = df.groupby(group1_columns, as_index=False)[group2_columns].sum()
        else:
            # For feature groups, compute joint probabilities using Visit_Prob
            joint_prob = df.groupby(group1_columns + group2_columns)["Visit_Prob"].sum().reset_index()
            joint_prob.rename(columns={"Visit_Prob": "P(Group1,Group2)"}, inplace=True)

        # Compute marginal probabilities P(Group1) and P(Group2)
        if is_action_group:
            joint_prob["P(Group1)"] = joint_prob[group2_columns].sum(axis=1)
            marginal_group2_prob = df[group2_columns].sum()
        else:
            # For feature groups, compute marginals
            marginal_group1_prob = joint_prob.groupby(group1_columns)["P(Group1,Group2)"].sum().reset_index()
            marginal_group1_prob.rename(columns={"P(Group1,Group2)": "P(Group1)"}, inplace=True)

            marginal_group2_prob = joint_prob.groupby(group2_columns)["P(Group1,Group2)"].sum().reset_index()
            marginal_group2_prob.rename(columns={"P(Group1,Group2)": "P(Group2)"}, inplace=True)

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
