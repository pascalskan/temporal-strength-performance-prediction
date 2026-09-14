"""
Feature ablation stage.

A separate stage rather than part of the walk-forward pipeline because it costs
one full walk-forward run per feature set -- roughly four times the walk-forward
runtime on the default configuration. Bundling it would have made every routine
run several times slower for an analysis that is not needed on every run.
"""
import time

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.core.logging import get_logger
from src.evaluation.ablation import run_and_save_ablation
from src.evaluation.walk_forward import WalkForwardEvaluator
from src.features.temporal import feature_engineering
from src.io.paths import ProjectPaths
from src.models.baselines import (
    DriftBaseline,
    PersistenceBaseline,
    RollingMeanBaseline,
)
from src.pipelines.walk_forward_pipeline import create_walk_forward_target
from src.reproducibility.environment import capture_experiment_metadata
from src.reproducibility.metadata import save_metadata

logger = get_logger(__name__)


def build_evaluator() -> WalkForwardEvaluator:
    """
    A fresh evaluator with unfitted models and empty accumulators.

    Each feature set needs its own; reusing one would pool predictions from
    different configurations into a single result set.

    The temporal baselines are included and read their columns from the frame,
    so they are unaffected by ablation and act as fixed reference lines: the
    question is whether a model given more features beats simply carrying the
    last result forward.

    The traditional equations are excluded. They too are invariant to the
    feature set, but they score only the subpopulation with recorded attempts,
    which would make their row incomparable to the rest of the table.
    """
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(random_state=42)),
        ]),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, random_state=42
        ),
    }

    baselines = {
        "Persistence": PersistenceBaseline(),
        "Rolling Mean": RollingMeanBaseline(),
        "Drift": DriftBaseline(),
    }

    return WalkForwardEvaluator(
        models=models,
        baselines=baselines,
        granularity="year",
        min_train_periods=5,
    )


def run_ablation_pipeline(df, group_name):
    """Run the feature ablation for one cohort."""
    start_time = time.time()
    logger.info("Starting ablation pipeline for group: %s", group_name)

    if df.empty:
        logger.warning("Dataset for group '%s' is empty; skipping.", group_name)
        return

    base_dir = ProjectPaths.forward_group_dir(group_name) / "ablation"
    base_dir.mkdir(parents=True, exist_ok=True)

    save_metadata(
        capture_experiment_metadata(
            df=df,
            cohort=group_name,
            execution_mode="ablation",
            random_seed=42,
        ),
        base_dir / "run_metadata.json",
    )

    run_and_save_ablation(
        df=df,
        evaluator_factory=build_evaluator,
        feature_engineering_fn=feature_engineering,
        target_fn=create_walk_forward_target,
        output_dir=base_dir,
    )

    logger.info(
        "Ablation pipeline completed in %.2f seconds", time.time() - start_time
    )
