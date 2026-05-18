from pathlib import Path

from src.analysis.correlation_analysis import run_importance_correlation_analysis
from src.analysis.importance_plots import (
    save_gradient_boosting_importance_plot,
    save_random_forest_importance_plot,
)
from src.core.logging import get_logger
from src.evaluation.subgroup_analysis import strength_level_analysis


logger = get_logger(__name__)


def run_feature_importance_analysis(
    df_model_no_attempts,
    rf_model,
    gb_model,
    X,
    y,
    feature_cols,
    evaluation_dir,
    importance_dir
):
    """
    Runs the full feature importance analysis pipeline.
    """
    logger.info("Starting feature importance analysis...")

    evaluation_dir = Path(evaluation_dir)
    importance_dir = Path(importance_dir)

    logger.info("Running strength-level subgroup analysis...")
    strength_level_analysis(
        df_model_no_attempts,
        rf_model,
        X,
        y,
        save_path=evaluation_dir / "strength_analysis_rf.csv"
    )
    strength_level_analysis(
        df_model_no_attempts,
        gb_model,
        X,
        y,
        save_path=evaluation_dir / "strength_analysis_gb.csv"
    )
    logger.info("Strength-level analysis complete.")

    logger.info("Calculating and plotting feature importances...")
    rf_df = save_random_forest_importance_plot(
        rf_model,
        feature_cols,
        importance_dir
    )
    gb_df = save_gradient_boosting_importance_plot(
        gb_model,
        feature_cols,
        importance_dir
    )
    logger.info("Feature importance plots saved.")

    logger.info("Running correlation analysis on feature importances...")
    run_importance_correlation_analysis(
        rf_df,
        gb_df,
        importance_dir
    )
    logger.info("Feature importance correlation analysis complete.")
    logger.info("Feature importance analysis finished.")
