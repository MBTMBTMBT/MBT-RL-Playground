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

