from dataclasses import dataclass
import pandas as pd


@dataclass
class PreparedData:
    df: pd.DataFrame
    df_model: pd.DataFrame
    df_model_no_attempts: pd.DataFrame

    feature_cols: list

    X: pd.DataFrame
    y: pd.Series

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series

    Xb_train: pd.DataFrame
    Xb_test: pd.DataFrame
    yb_train: pd.Series
    yb_test: pd.Series

    X_train_scaled: pd.DataFrame
    X_test_scaled: pd.DataFrame
    Xb_train_scaled: pd.DataFrame
    Xb_test_scaled: pd.DataFrame

    split_idx: object