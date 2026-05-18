import time
from src.features.temporal import feature_engineering
from src.utils.splitting import time_aware_split
from src.evaluation.forward_reporting import save_forward_metrics, save_forward_predictions, save_retrospective_metrics, save_traditional_predictions
from src.models.forward_training import train_forward_models, train_retrospective_reference_models
from src.models.forward_traditional import run_forward_traditional_predictions
from src.config.features import ENGINEERED_FEATURES, ATTEMPT_COLUMNS
from src.core.logging import get_logger
from src.io.paths import ProjectPaths
from src.reproducibility.environment import capture_experiment_metadata
from src.reproducibility.metadata import save_metadata

logger = get_logger(__name__)

def create_forward_target(df):
    """Creates the forward prediction target by shifting the TotalKg."""
    df = df.sort_values(["Name", "Date"]).copy()
    df["Target_Total"] = df.groupby("Name")["TotalKg"].shift(-1)
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
        dataset_name=group_name,
        execution_mode="forward",
        random_seed=42  # Project standard baseline seed
    )
    save_metadata(metadata, base_dir / "run_metadata.json")

    df = feature_engineering(df)
    assert df.groupby("Name")["Date"].is_monotonic_increasing.all(), "Data leakage risk: dates not sorted."

    df = create_forward_target(df)
    df["Target_Retro"] = df["TotalKg"]

    df_model = df.dropna(subset=ENGINEERED_FEATURES)
    logger.info("Samples after filtering for required features: %s", len(df_model))

    df_model_no_attempts = df_model.drop(columns=ATTEMPT_COLUMNS, errors="ignore")

    X_train, X_test, y_train, y_test, _ = time_aware_split(df_model_no_attempts, ENGINEERED_FEATURES, target_col="Target_Total")
    X_train_r, X_test_r, y_train_r, y_test_r, _ = time_aware_split(df_model_no_attempts, ENGINEERED_FEATURES, target_col="Target_Retro")

    logger.info("Training retrospective reference models...")
    retro_results = train_retrospective_reference_models(X_train_r, X_test_r, y_train_r, y_test_r)

    logger.info("Training forward prediction models...")
    forward_results = train_forward_models(X_train, X_test, y_train, y_test)

    logger.info("Running traditional methods for forward prediction...")
    actual_next, epley_preds, brzycki_preds, traditional_metrics = run_forward_traditional_predictions(df_model, y_test.index)

    logger.info("Saving all metrics and predictions...")
    save_forward_metrics(
        base_dir,
        forward_results.linear_regression.metrics.mae, forward_results.linear_regression.metrics.rmse, forward_results.linear_regression.metrics.r2,
        forward_results.random_forest.metrics.mae, forward_results.random_forest.metrics.rmse, forward_results.random_forest.metrics.r2,
        forward_results.gradient_boosting.metrics.mae, forward_results.gradient_boosting.metrics.rmse, forward_results.gradient_boosting.metrics.r2,
        traditional_metrics["Epley"]["MAE"], traditional_metrics["Epley"]["RMSE"], traditional_metrics["Epley"]["R2"],
        traditional_metrics["Brzycki"]["MAE"], traditional_metrics["Brzycki"]["RMSE"], traditional_metrics["Brzycki"]["R2"]
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
