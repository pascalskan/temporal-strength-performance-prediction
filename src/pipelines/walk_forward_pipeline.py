import time

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config.features import FORECAST_FEATURES
from src.core.logging import get_logger
from src.evaluation.walk_forward import WalkForwardEvaluator
from src.features.temporal import feature_engineering
from src.io.paths import ProjectPaths
from src.models.baselines import (
    DriftBaseline,
    PersistenceBaseline,
    RollingMeanBaseline,
)
from src.reproducibility.environment import capture_experiment_metadata
from src.reproducibility.metadata import save_metadata

logger = get_logger(__name__)


def create_walk_forward_target(df, athlete_col="Athlete_ID"):
    """
    Creates next-event temporal forecasting targets.

    Each athlete's next competition total becomes the prediction target,
    enabling leakage-safe temporal forecasting evaluation.
    """
    df = df.sort_values([athlete_col, "Date"]).copy()

    df["Target_Total"] = (
        df.groupby(athlete_col)["TotalKg"]
        .shift(-1)
    )

    return df


def run_walk_forward_pipeline(df, group_name):
    """
    Runs the publication-grade expanding-window walk-forward forecasting pipeline.
    """

    start_time = time.time()

    logger.info(
        "Starting Walk-Forward pipeline for group: %s",
        group_name
    )

    if df.empty:
        logger.warning(
            "Dataset for group '%s' is empty. "
            "Skipping walk-forward pipeline.",
            group_name
        )
        return

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------

    base_dir = (
        ProjectPaths.forward_group_dir(group_name)
        / "walk_forward"
    )

    base_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Saving walk-forward results to directory: %s",
        base_dir
    )

    # ------------------------------------------------------------------
    # Reproducibility metadata
    # ------------------------------------------------------------------

    metadata = capture_experiment_metadata(
        df=df,
        dataset_name=group_name,
        execution_mode="walk_forward",
        random_seed=42,
    )

    save_metadata(
        metadata,
        base_dir / "run_metadata.json"
    )

    # ------------------------------------------------------------------
    # Machine Learning Models
    # ------------------------------------------------------------------

    models = {
        "Linear Regression": LinearRegression(),

        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(random_state=42)),
        ]),

        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200,
            random_state=42,
        ),
    }

    # ------------------------------------------------------------------
    # Forecasting Baselines
    # ------------------------------------------------------------------

    baselines = {
        "Persistence": PersistenceBaseline(),
        "Rolling Mean": RollingMeanBaseline(),
        "Drift": DriftBaseline(),
    }

    # ------------------------------------------------------------------
    # Walk-Forward Evaluator
    # ------------------------------------------------------------------

    # Scientific hardening:
    # Increase minimum training periods to reduce unstable early folds.
    evaluator = WalkForwardEvaluator(
        models=models,
        baselines=baselines,
        granularity="year",
        min_train_periods=5,
    )

    # ------------------------------------------------------------------
    # Execute Evaluation
    # ------------------------------------------------------------------

    evaluator.evaluate(
        df_raw=df,
        feature_engineering_fn=feature_engineering,
        target_fn=create_walk_forward_target,
        feature_cols=FORECAST_FEATURES,
    )

    # ------------------------------------------------------------------
    # Publication-Grade Statistical Comparisons
    # ------------------------------------------------------------------

    comparisons = [
        ("Rolling Mean", "Gradient Boosting"),
        ("Rolling Mean", "Ridge"),
        ("Rolling Mean", "Random Forest"),
        ("Rolling Mean", "Linear Regression"),
        ("Persistence", "Gradient Boosting"),
    ]

    # ------------------------------------------------------------------
    # Save All Outputs
    # ------------------------------------------------------------------

    evaluator.save_results(
        base_dir,
        comparisons=comparisons,
    )

    logger.info(
        "Walk-forward pipeline completed in %.2f seconds",
        time.time() - start_time,
    )