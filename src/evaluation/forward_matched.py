"""
Like-for-like comparison for the forward protocol.

The forward stage reported the traditional equations and the learned models in
one table, `metrics.csv`, with no indication that they were scored on different
populations. Epley and Brzycki require recorded attempts, present for only part
of the test set and not at random; that subpopulation is systematically easier.
On the Raw cohort the effect is large enough to invert how the table reads --
Brzycki appeared to beat gradient boosting by 11.7 kg where, on shared
observations, the difference is 2.1 kg.

The walk-forward protocol already produces a matched-subset comparison. This
gives the forward protocol the same treatment, so no results file in the
project presents an unequal comparison as an equal one.
"""
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from src.core.logging import get_logger
from src.evaluation.metrics import compute_basic_metrics

logger = get_logger(__name__)


def compute_matched_forward_metrics(
    model_predictions: Dict[str, pd.Series],
    y_test: pd.Series,
    scored_index: pd.Index,
) -> pd.DataFrame:
    """
    Re-score every model over the observations the traditional equations could
    address.

    Args:
        model_predictions: {model name: predictions indexed like y_test}.
        y_test: Forward targets for the full test set.
        scored_index: Observations the traditional equations scored.

    Returns:
        One row per model over the shared observations, empty if none overlap.
    """
    shared = y_test.index.intersection(scored_index)

    if len(shared) == 0:
        logger.warning(
            "No observations are shared between the traditional equations and "
            "the model test set; matched comparison unavailable."
        )
        return pd.DataFrame()

    records = []
    for name, predictions in model_predictions.items():
        aligned = pd.Series(predictions, index=y_test.index).loc[shared]
        truth = y_test.loc[shared]

        usable = aligned.notna() & truth.notna()
        if not usable.any():
            continue

        metrics = compute_basic_metrics(truth[usable].values, aligned[usable].values)
        records.append({
            "Model": name,
            "MAE": metrics.mae,
            "RMSE": metrics.rmse,
            "R2": metrics.r2,
            "n_scored": int(usable.sum()),
        })

    return pd.DataFrame(records)


def save_matched_forward_metrics(
    model_predictions: Dict[str, pd.Series],
    y_test: pd.Series,
    scored_index: pd.Index,
    base_dir: Path,
) -> pd.DataFrame:
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    matched = compute_matched_forward_metrics(model_predictions, y_test, scored_index)

    if matched.empty:
        return matched

    path = base_dir / "metrics_matched_subset.csv"
    matched.to_csv(path, index=False)

    logger.info(
        "Saved matched-subset forward metrics to %s (%d shared observations of %d).",
        path, int(matched["n_scored"].max()), len(y_test),
    )
    return matched
