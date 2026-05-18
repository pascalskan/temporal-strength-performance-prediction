import numpy as np

from src.core.logging import get_logger
from src.features.temporal import feature_engineering
from src.utils.splitting import time_aware_split
from src.utils.scaling import scale_features
from src.pipelines.pipeline_objects import PreparedData

from src.config.features import (
    ATTEMPT_COLUMNS,
    ENGINEERED_FEATURES,
    BASELINE_FEATURES
)


logger = get_logger(__name__)


def prepare_modelling_data(df):
    """
    Prepare retrospective modelling dataset.

    Steps:
    - apply feature engineering
    - remove rows with insufficient historical data
    - remove attempt columns for ML modelling
    - create engineered and baseline feature matrices
    - perform time-aware train/test split
    - scale required feature sets
    """

    logger.info("🔹 Applying feature engineering...")
    df = feature_engineering(df)

    logger.info("🔹 Cleaning dataset for modelling (removing insufficient history rows)...")

    df_model = df.dropna(
        subset=[
            "Sex",
            "Age",
            "BodyweightKg",
            "TotalKg",
            "Prev_Total",
            "Rolling_Mean_3",
            "Rolling_Std_3",
            "Improvement",
            "Momentum",
            "Days_Since_Last",
            "PB",
        ]
    )

    df_model_no_attempts = df_model.drop(
        columns=ATTEMPT_COLUMNS,
        errors="ignore"
    )

    logger.info("✅ Remaining samples after cleaning: %s", len(df_model))

    X = df_model_no_attempts[ENGINEERED_FEATURES]
    y = df_model["TotalKg"]

    logger.info("🔹 Final feature cleaning (handling remaining NaNs)...")

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    logger.info("✅ Final feature cleaning complete")
    logger.info("✅ Features ready: %s features", len(ENGINEERED_FEATURES))

    logger.info("🔹 Using time-aware split (past → future)...")

    X_train, X_test, y_train, y_test, split_idx = time_aware_split(
        df_model_no_attempts,
        ENGINEERED_FEATURES
    )

    train_dates = df_model_no_attempts[
        df_model_no_attempts["Date"] <= split_idx
    ]["Date"]

    test_dates = df_model_no_attempts[
        df_model_no_attempts["Date"] > split_idx
    ]["Date"]

    assert train_dates.max() <= test_dates.min(), \
        "❌ Time leakage: train/test overlap"

    Xb_train, Xb_test, yb_train, yb_test, _ = time_aware_split(
        df_model_no_attempts,
        BASELINE_FEATURES
    )

    X_train_scaled, X_test_scaled, _ = scale_features(
        X_train,
        X_test
    )

    Xb_train_scaled, Xb_test_scaled, _ = scale_features(
        Xb_train,
        Xb_test
    )

    logger.info("Train size: %s", len(X_train))
    logger.info("Test size: %s", len(X_test))

    logger.info("📅 Date ranges:")

    train_df = df_model_no_attempts[
        df_model_no_attempts["Date"] <= split_idx
    ]

    test_df = df_model_no_attempts[
        df_model_no_attempts["Date"] > split_idx
    ]

    logger.info("Train: %s → %s", train_df["Date"].min(), train_df["Date"].max())
    logger.info("Test:  %s → %s", test_df["Date"].min(), test_df["Date"].max())

    return PreparedData(
        df=df,
        df_model=df_model,
        df_model_no_attempts=df_model_no_attempts,
        feature_cols=ENGINEERED_FEATURES,
        X=X,
        y=y,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        Xb_train=Xb_train,
        Xb_test=Xb_test,
        yb_train=yb_train,
        yb_test=yb_test,
        X_train_scaled=X_train_scaled,
        X_test_scaled=X_test_scaled,
        Xb_train_scaled=Xb_train_scaled,
        Xb_test_scaled=Xb_test_scaled,
        split_idx=split_idx
    )
