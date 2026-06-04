import pandas as pd


def leave_one_out_split(
    ratings: pd.DataFrame,
    user_col: str = "user_id",
    time_col: str = "timestamp",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split ratings so each user's most recent interaction becomes test data.

    Train: all earlier interactions
    Test: one latest interaction per user
    """
    if ratings.empty:
        raise ValueError("ratings dataframe is empty")

    required_columns = {user_col, time_col}
    missing_columns = required_columns - set(ratings.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    sorted_ratings = ratings.sort_values([user_col, time_col])

    test_indices = sorted_ratings.groupby(user_col).tail(1).index
    test = ratings.loc[test_indices].copy()
    train = ratings.drop(index=test_indices).copy()

    return train, test