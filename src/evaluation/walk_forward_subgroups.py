"""
Robustness of walk-forward forecasts across athlete subgroups.

Subgroup analysis existed only for the retrospective protocol, which the study
argues measures reconstruction rather than prediction. Claims about who the
models serve well therefore rested on the protocol whose validity the work
disputes. This runs the same question against the walk-forward predictions.

Every subgroup is reported with its sample size and athlete count, because a
difference in error between groups of very different size is often a statement
about sample size rather than about the groups.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.logging import get_logger
from src.evaluation.metrics import compute_global_forecast_metrics

logger = get_logger(__name__)

# Below this many predictions a subgroup's error estimate is too unstable to
# report as a finding. Rows are still emitted, flagged, so a reader sees the
# group exists and why it is not interpreted rather than finding it missing.
MIN_RELIABLE_PREDICTIONS = 100

SEX_LABELS = {1: "male", 0: "female", 1.0: "male", 0.0: "female"}


def _strength_bands(y_true: pd.Series, n_bands: int = 4) -> pd.Series:
    """
    Band observations by outcome magnitude.

    The original error analysis found a U-shaped error curve across strength
    levels, worst at the extremes. Banding by the realised total reproduces
    that view under walk-forward evaluation.
    """
    try:
        bands = pd.qcut(y_true, n_bands, labels=False, duplicates="drop")
    except (ValueError, IndexError):
        return pd.Series(np.nan, index=y_true.index)
    return bands


def build_subgroups(predictions_df: pd.DataFrame) -> dict:
    """
    Define the subgroup partitions available from the prediction record.

    Returns {dimension_name: Series of labels aligned to predictions_df}.
    Dimensions whose source column is absent are simply not offered.
    """
    subgroups = {}

    if "Sex" in predictions_df.columns:
        subgroups["sex"] = predictions_df["Sex"].map(SEX_LABELS).fillna("unknown")

    if "Equipment" in predictions_df.columns:
        subgroups["equipment"] = predictions_df["Equipment"].astype(str)

    if "BodyweightKg" in predictions_df.columns:
        weights = pd.to_numeric(predictions_df["BodyweightKg"], errors="coerce")
        subgroups["bodyweight_quartile"] = pd.qcut(
            weights, 4, labels=["q1_lightest", "q2", "q3", "q4_heaviest"],
            duplicates="drop",
        ).astype(str)

    if "Age" in predictions_df.columns:
        ages = pd.to_numeric(predictions_df["Age"], errors="coerce")
        subgroups["age_band"] = pd.cut(
            ages,
            bins=[0, 23, 34, 49, 200],
            labels=["under_24", "24_to_34", "35_to_49", "50_plus"],
        ).astype(str)

    bands = _strength_bands(predictions_df["y_true"])
    subgroups["strength_quartile"] = bands.map(
        {0: "q1_weakest", 1: "q2", 2: "q3", 3: "q4_strongest"}
    ).fillna("unknown")

    return subgroups


def compute_subgroup_metrics(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Pooled metrics per model per subgroup, with the sample sizes behind them."""
    records = []

    for dimension, labels in build_subgroups(predictions_df).items():
        frame = predictions_df.assign(_subgroup=labels)

        for label, group in frame.groupby("_subgroup", dropna=False):
            if str(label) in ("nan", "unknown"):
                continue

            metrics = compute_global_forecast_metrics(group)
            if metrics.empty:
                continue

            athletes_by_model = group.groupby("model")["athlete_id"].nunique()

            for _, row in metrics.iterrows():
                n_predictions = int(row["prediction_count"])
                records.append({
                    "dimension": dimension,
                    "subgroup": str(label),
                    "model": row["model"],
                    "prediction_count": n_predictions,
                    "athlete_count": int(athletes_by_model.get(row["model"], 0)),
                    "MAE": row["MAE"],
                    "RMSE": row["RMSE"],
                    "R2": row["R2"],
                    "mean_residual": row["mean_residual"],
                    "sMAPE": row["sMAPE"],
                    "sufficiently_powered": n_predictions >= MIN_RELIABLE_PREDICTIONS,
                })

    return pd.DataFrame(records)


# Schema for the disparity table. Declared once so that the "nothing to
# compare" paths return a frame with the same columns as a populated one --
# an empty frame built from an empty list has no columns at all, and sorting
# it raises KeyError mid-run.
DISPARITY_COLUMNS = [
    "dimension", "model",
    "best_subgroup", "best_MAE", "best_n",
    "worst_subgroup", "worst_MAE", "worst_n",
    "MAE_gap", "MAE_ratio", "subgroups_compared",
]


def _empty_disparities() -> pd.DataFrame:
    return pd.DataFrame(columns=DISPARITY_COLUMNS)


def summarise_disparities(subgroup_metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Spread in error between the best- and worst-served subgroup, per dimension.

    Reported only over adequately powered subgroups, and deliberately named a
    disparity rather than a bias: an error gap has many possible sources --
    representation, differing data-generating processes, feature fit, sampling
    noise -- and this measurement does not distinguish between them.
    """
    if subgroup_metrics.empty:
        return _empty_disparities()

    powered = subgroup_metrics[subgroup_metrics["sufficiently_powered"]]
    if powered.empty:
        return _empty_disparities()

    records = []
    for (dimension, model), group in powered.groupby(["dimension", "model"]):
        if len(group) < 2:
            continue

        best = group.loc[group["MAE"].idxmin()]
        worst = group.loc[group["MAE"].idxmax()]

        records.append({
            "dimension": dimension,
            "model": model,
            "best_subgroup": best["subgroup"],
            "best_MAE": best["MAE"],
            "best_n": int(best["prediction_count"]),
            "worst_subgroup": worst["subgroup"],
            "worst_MAE": worst["MAE"],
            "worst_n": int(worst["prediction_count"]),
            "MAE_gap": worst["MAE"] - best["MAE"],
            "MAE_ratio": worst["MAE"] / best["MAE"] if best["MAE"] else np.nan,
            "subgroups_compared": len(group),
        })

    if not records:
        # Every dimension had fewer than two adequately powered subgroups.
        return _empty_disparities()

    return pd.DataFrame(records).sort_values(
        ["dimension", "MAE_gap"], ascending=[True, False]
    )


def run_and_save_subgroup_analysis(predictions_df: pd.DataFrame, output_dir: Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = compute_subgroup_metrics(predictions_df)
    if metrics.empty:
        logger.warning("No subgroups could be formed from the prediction record.")
        return metrics, pd.DataFrame()

    metrics_path = output_dir / "subgroup_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    logger.info(
        "Saved subgroup metrics to %s (%d rows across %d dimensions)",
        metrics_path, len(metrics), metrics["dimension"].nunique(),
    )

    underpowered = (~metrics["sufficiently_powered"]).sum()
    if underpowered:
        logger.info(
            "%d subgroup rows fall below %d predictions and are flagged as "
            "insufficiently powered.",
            underpowered, MIN_RELIABLE_PREDICTIONS,
        )

    disparities = summarise_disparities(metrics)
    if not disparities.empty:
        disparities_path = output_dir / "subgroup_disparities.csv"
        disparities.to_csv(disparities_path, index=False)
        logger.info("Saved subgroup disparities to %s", disparities_path)

    return metrics, disparities
