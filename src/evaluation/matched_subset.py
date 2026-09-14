"""
Like-for-like comparison on the subpopulation every model can score.

Traditional equations need recorded attempt loads, which are present for only
62.7% of raw records and 30.4% of equipped ones. They therefore predict on a
smaller, non-random subpopulation -- plausibly better-documented meets -- while
the machine learning models and temporal baselines are scored on everything.

Placing those numbers in one table, as the forward pipeline's metrics.csv
does, compares models across different evaluation sets. The difference is not
cosmetic: it is exactly the kind of subpopulation effect that can reverse a
ranking.

This module restricts every model to the observations where all models produced
a prediction, so a comparison against the traditional equations is made on
identical data.
"""
from pathlib import Path

import pandas as pd

from src.core.logging import get_logger
from src.evaluation.metrics import compute_global_forecast_metrics

logger = get_logger(__name__)


def find_common_observations(predictions_df: pd.DataFrame) -> pd.Index:
    """
    Observation ids for which every model produced a finite prediction.

    Returns an empty index if no observation is scored by all models.
    """
    scorable = predictions_df[predictions_df["y_pred"].notna()]

    counts = scorable.groupby("observation_id")["model"].nunique()
    n_models = predictions_df["model"].nunique()

    return counts[counts == n_models].index


def compute_matched_subset_metrics(predictions_df: pd.DataFrame):
    """
    Recompute pooled metrics over the observations common to every model.

    Returns:
        (metrics_df, coverage_df). metrics_df is empty when no observation is
        shared by all models.
    """
    common = find_common_observations(predictions_df)

    total_observations = predictions_df["observation_id"].nunique()
    coverage = (
        predictions_df.assign(scored=predictions_df["y_pred"].notna())
        .groupby("model")["scored"]
        .agg(["sum", "size"])
        .rename(columns={"sum": "scored_observations", "size": "eligible_observations"})
    )
    coverage["coverage_fraction"] = (
        coverage["scored_observations"] / coverage["eligible_observations"]
    )
    coverage["matched_subset_size"] = len(common)
    coverage["matched_fraction_of_all"] = (
        len(common) / total_observations if total_observations else 0.0
    )
    coverage = coverage.reset_index()

    if len(common) == 0:
        logger.warning(
            "No observation was scored by every model; matched-subset "
            "comparison is unavailable."
        )
        return pd.DataFrame(), coverage

    logger.info(
        "Matched subset: %d of %d observations scored by all %d models (%.1f%%).",
        len(common), total_observations, predictions_df["model"].nunique(),
        100 * len(common) / total_observations,
    )

    matched = predictions_df[predictions_df["observation_id"].isin(common)]
    return compute_global_forecast_metrics(matched), coverage


def run_and_save_matched_subset(predictions_df: pd.DataFrame, output_dir: Path):
    """Write the matched-subset metrics and the per-model coverage that explains them."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics, coverage = compute_matched_subset_metrics(predictions_df)

    coverage_path = output_dir / "model_coverage.csv"
    coverage.to_csv(coverage_path, index=False)
    logger.info("Saved model coverage to %s", coverage_path)

    if metrics.empty:
        return metrics, coverage

    metrics_path = output_dir / "matched_subset_summary.csv"
    metrics.to_csv(metrics_path, index=False)
    logger.info("Saved matched-subset metrics to %s", metrics_path)

    return metrics, coverage
