import time
from src.features.temporal import feature_engineering
from src.evaluation.walk_forward import WalkForwardEvaluator
from src.models.baselines import PersistenceBaseline, RollingMeanBaseline, DriftBaseline
from src.config.features import ENGINEERED_FEATURES
from src.core.logging import get_logger
from src.io.paths import ProjectPaths
from src.reproducibility.environment import capture_experiment_metadata
from src.reproducibility.metadata import save_metadata

# Currently using simple models for this framework (could pull from factories later if refactored)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

logger = get_logger(__name__)

def create_walk_forward_target(df, athlete_col="Athlete_ID"):
    """Creates the next-event temporal target for walk-forward validation."""
    df = df.sort_values([athlete_col, "Date"]).copy()
    df["Target_Total"] = df.groupby(athlete_col)["TotalKg"].shift(-1)
    # We do NOT dropna here because we need the most recent events to be the TEST targets
    # Dropping happens selectively inside the evaluator when constructing X and y pairs.
    return df

def run_walk_forward_pipeline(df, group_name):
    start_time = time.time()
    logger.info("Starting Walk-Forward pipeline for group: %s", group_name)

    if df.empty:
        logger.warning("Dataset for group '%s' is empty. Skipping walk-forward pipeline.", group_name)
        return

    # Use existing forward results dir or create a new dedicated one
    base_dir = ProjectPaths.forward_group_dir(group_name) / "walk_forward"
    base_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Saving walk-forward results to directory: %s", base_dir)

    metadata = capture_experiment_metadata(
        df=df,
        dataset_name=group_name,
        execution_mode="walk_forward",
        random_seed=42
    )
    save_metadata(metadata, base_dir / "run_metadata.json")

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=50, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=50, random_state=42)
    }

    baselines = {
        "Persistence": PersistenceBaseline(),
        "Rolling Mean": RollingMeanBaseline(),
        "Drift": DriftBaseline(),
        "Ridge": Ridge()
    }

    evaluator = WalkForwardEvaluator(models=models, baselines=baselines, granularity="year")

    # The evaluator dynamically handles features to prevent leakage
    evaluator.evaluate(
        df_raw=df,
        feature_engineering_fn=feature_engineering,
        target_fn=create_walk_forward_target,
        feature_cols=ENGINEERED_FEATURES
    )

    evaluator.save_results(base_dir)

    logger.info("Walk-forward pipeline completed in %.2f seconds", time.time() - start_time)
