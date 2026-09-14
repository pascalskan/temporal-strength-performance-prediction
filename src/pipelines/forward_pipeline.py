import time

import pandas as pd
from src.data.identity import build_athlete_id
from src.features.temporal import feature_engineering
from src.utils.splitting import time_aware_split
from src.evaluation.forward_matched import save_matched_forward_metrics
from src.evaluation.forward_reporting import save_forward_metrics, save_forward_predictions, save_retrospective_metrics, save_traditional_predictions
from src.models.forward_training import train_forward_models, train_retrospective_reference_models
from src.models.forward_traditional import run_forward_traditional_predictions
from src.config.features import ENGINEERED_FEATURES, ATTEMPT_COLUMNS
from src.core.logging import get_logger
from src.io.paths import ProjectPaths
from src.reproducibility.environment import capture_experiment_metadata
from src.reproducibility.metadata import save_metadata

logger = get_logger(__name__)

def create_forward_target(df, athlete_col="Athlete_ID"):
    """
    Creates the forward prediction target by shifting TotalKg.

    Grouped by the constructed athlete identifier rather than by Name alone,
    so that every pipeline links an athlete's history the same way.
    """
    df = df.sort_values([athlete_col, "Date"]).copy()
    df["Target_Total"] = df.groupby(athlete_col)["TotalKg"].shift(-1)
    df = df.dropna(subset=["Target_Total"])
    return df

def run_forward_pipeline(df, group_name):
    """Runs the full forward-prediction pipeline."""
    start_time = time.time()
    logger.info("Starting forward prediction pipeline for group: %s", group_name)

    if df.empty:
        logger.warning("Dataset for group '%s' is empty. Skipping forward pipeline execution.", group_name)
        return

    base_dir = ProjectPaths.forward_results_dir(group_name)
    logger.info("Saving results to directory: %s", base_dir)

    metadata = capture_experiment_metadata(
        df=df,
        cohort=group_name,
        execution_mode="forward",
        random_seed=42  # Project standard baseline seed
    )
    save_metadata(metadata, base_dir / "run_metadata.json")

    df = df.copy()
    df["Athlete_ID"] = build_athlete_id(df)
    df = feature_engineering(df, athlete_col="Athlete_ID")
    assert df.groupby("Athlete_ID")["Date"].is_monotonic_increasing.all(), \
        "Data leakage risk: dates not sorted."

    df = create_forward_target(df)
    df["Target_Retro"] = df["TotalKg"]

    df_model = df.dropna(subset=ENGINEERED_FEATURES)
    logger.info("Samples after filtering for required features: %s", len(df_model))

    if len(df_model) < 2:
        logger.warning("Not enough data to perform train/test split for group '%s'. Skipping.", group_name)
        return

    df_model_no_attempts = df_model.drop(columns=ATTEMPT_COLUMNS, errors="ignore")

    X_train, X_test, y_train, y_test, _ = time_aware_split(df_model_no_attempts, ENGINEERED_FEATURES, target_col="Target_Total")
    X_train_r, X_test_r, y_train_r, y_test_r, _ = time_aware_split(df_model_no_attempts, ENGINEERED_FEATURES, target_col="Target_Retro")

    if X_train.empty or X_test.empty or X_train_r.empty or X_test_r.empty:
        logger.warning("Train or test split is empty for group '%s'. Skipping model training.", group_name)
        return

    logger.info("Training retrospective reference models...")
    retro_results = train_retrospective_reference_models(X_train_r, X_test_r, y_train_r, y_test_r)

    logger.info("Training forward prediction models...")
    forward_results = train_forward_models(X_train, X_test, y_train, y_test)

    logger.info("Running traditional methods for forward prediction...")
    (actual_next, epley_preds, brzycki_preds,
     traditional_metrics, traditional_index) = run_forward_traditional_predictions(
        df_model, y_test.index
    )

    logger.info("Saving all metrics and predictions...")
    save_forward_metrics(base_dir, forward_results, traditional_metrics)

    # Re-score every model over the observations the traditional equations
    # could address, so the comparison against them is made on shared data.
    save_matched_forward_metrics(
        model_predictions={
            "Linear Regression": forward_results.y_pred_lr,
            "Random Forest": forward_results.y_pred_rf,
            "Gradient Boosting": forward_results.y_pred_gb,
            "Epley": pd.Series(epley_preds, index=traditional_index).reindex(y_test.index),
            "Brzycki": pd.Series(brzycki_preds, index=traditional_index).reindex(y_test.index),
        },
        y_test=y_test,
        scored_index=traditional_index,
        base_dir=base_dir,
    )
    save_retrospective_metrics(
        base_dir,
        retro_results.linear_regression.metrics.mae, retro_results.linear_regression.metrics.rmse, retro_results.linear_regression.metrics.r2,
        retro_results.random_forest.metrics.mae, retro_results.random_forest.metrics.rmse, retro_results.random_forest.metrics.r2,
        retro_results.gradient_boosting.metrics.mae, retro_results.gradient_boosting.metrics.rmse, retro_results.gradient_boosting.metrics.r2
    )
    save_forward_predictions(base_dir, y_test, forward_results.y_pred_lr, forward_results.y_pred_rf, forward_results.y_pred_gb)
    save_traditional_predictions(base_dir, actual_next, epley_preds, brzycki_preds)

    logger.info("Forward pipeline completed in %.2f seconds", time.time() - start_time)
