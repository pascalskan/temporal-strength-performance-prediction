import time


from src.config.features import FORECAST_FEATURES
from src.config.experiment import (
    MIN_TRAIN_PERIODS,
    MODEL_COMPARISONS,
    WALK_FORWARD_GRANULARITY,
    build_all_baselines,
    build_models,
    build_tuned_models,
)
from src.core.logging import get_logger
from src.evaluation.walk_forward import WalkForwardEvaluator
from src.features.temporal import feature_engineering
from src.io.paths import ProjectPaths
from src.models.tuned import collect_selection_history
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


def run_walk_forward_pipeline(df, group_name, tuned: bool = False):
    """
    Runs the expanding-window walk-forward forecasting pipeline.

    Args:
        df: Cohort data, uncleaned and unengineered.
        group_name: Equipment cohort being evaluated.
        tuned: Select hyperparameters within each fold's training window rather
            than using fixed values. Writes to a separate directory so both
            variants coexist: the difference between them is what measures
            whether tuning changes the conclusion, which is not something to
            assert in either direction without evidence.
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
        / ("walk_forward_tuned" if tuned else "walk_forward")
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
        cohort=group_name,
        execution_mode="walk_forward_tuned" if tuned else "walk_forward",
        random_seed=42,
    )

    save_metadata(
        metadata,
        base_dir / "run_metadata.json"
    )

    # ------------------------------------------------------------------
    # Machine Learning Models
    # ------------------------------------------------------------------

    # Model set, protocol settings and comparison pairs all come from
    # src/config/experiment.py, so the ablation stage evaluates exactly the
    # same models as the headline run rather than a parallel definition that
    # can drift out of step.
    models = build_tuned_models() if tuned else build_models()
    baselines = build_all_baselines()

    if tuned:
        logger.info(
            "Hyperparameters will be selected within each fold's training "
            "window by forward-chaining search."
        )

    evaluator = WalkForwardEvaluator(
        models=models,
        baselines=baselines,
        granularity=WALK_FORWARD_GRANULARITY,
        min_train_periods=MIN_TRAIN_PERIODS,
        # Completed folds are persisted as they finish. A production run takes
        # tens of minutes to hours, and previously any interruption discarded
        # all of it.
        checkpoint_dir=base_dir,
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

    comparisons = MODEL_COMPARISONS

    # ------------------------------------------------------------------
    # Save All Outputs
    # ------------------------------------------------------------------

    evaluator.save_results(
        base_dir,
        comparisons=comparisons,
    )

    if tuned:
        # Whether the optimum moves as the training window grows is a result in
        # its own right: a drifting optimum indicates the data-generating
        # process is not stable, which every forecast here assumes.
        selection = collect_selection_history(models)
        if not selection.empty:
            selection_path = base_dir / "hyperparameter_selection.csv"
            selection.to_csv(selection_path, index=False)
            logger.info("Saved hyperparameter selection history to %s", selection_path)

    logger.info(
        "Walk-forward pipeline completed in %.2f seconds",
        time.time() - start_time,
    )