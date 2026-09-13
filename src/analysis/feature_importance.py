from pathlib import Path

from src.analysis.correlation_analysis import (
    run_importance_correlation_analysis,
)

from src.analysis.importance_plots import (
    save_feature_importance_plot,
)

from src.core.logging import get_logger

from src.evaluation.subgroup_analysis import (
    strength_level_analysis,
)

from src.models.introspection import (
    supports_feature_importance,
    supports_coefficients,
)

logger = get_logger(__name__)


def run_feature_importance_analysis(
    df_model_no_attempts,
    models,
    X,
    y,
    feature_cols,
    evaluation_dir,
    importance_dir
):
    """
    Runs the full feature importance analysis pipeline.

    Supports:
    - tree-based feature importances
    - linear model coefficients
    - sklearn Pipelines
    """

    logger.info("Starting feature importance analysis...")

    evaluation_dir = Path(evaluation_dir)
    importance_dir = Path(importance_dir)

    importance_dfs = {}

    # --------------------------------------------------
    # Strength-level subgroup analysis
    # --------------------------------------------------

    for model_name, model in models.items():

        logger.info(
            f"Running strength-level subgroup analysis "
            f"for {model_name}..."
        )

        safe_name = (
            model_name.lower()
            .replace(" ", "_")
        )

        strength_level_analysis(
            df_model_no_attempts,
            model,
            X,
            y,
            save_path=(
                evaluation_dir /
                f"strength_analysis_{safe_name}.csv"
            )
        )

    logger.info("Strength-level analysis complete.")

    # --------------------------------------------------
    # Feature importance / coefficient analysis
    # --------------------------------------------------

    logger.info(
        "Calculating and plotting feature importances..."
    )

    for model_name, model in models.items():

        if (
            supports_feature_importance(model)
            or
            supports_coefficients(model)
        ):

            logger.info(
                f"Generating importance analysis "
                f"for {model_name}..."
            )

            df = save_feature_importance_plot(
                model,
                model_name,
                feature_cols,
                importance_dir
            )

            if df is not None:
                importance_dfs[model_name] = df

        else:

            logger.info(
                f"Skipping importance analysis for "
                f"{model_name} "
                f"(no supported importance interface)"
            )

    logger.info("Feature importance plots saved.")

    # --------------------------------------------------
    # Cross-model correlation analysis
    # --------------------------------------------------

    if len(importance_dfs) > 1:

        logger.info(
            "Running correlation analysis on "
            "feature importances..."
        )

        run_importance_correlation_analysis(
            list(importance_dfs.values()),
            list(importance_dfs.keys()),
            importance_dir
        )

        logger.info(
            "Feature importance correlation "
            "analysis complete."
        )

    logger.info(
        "Feature importance analysis finished."
    )