from src.config.constants import TIME_SPLIT_RATIO


def time_aware_split(
    df,
    feature_cols,
    target_col="TotalKg",
    split_ratio=TIME_SPLIT_RATIO
):
    """
    Chronological train/test split using date quantile cutoff.
    """
    df = df.sort_values("Date").copy()

    split_point = df["Date"].quantile(split_ratio)

    train_df = df[df["Date"] <= split_point]
    test_df = df[df["Date"] > split_point]

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    return X_train, X_test, y_train, y_test, split_point
