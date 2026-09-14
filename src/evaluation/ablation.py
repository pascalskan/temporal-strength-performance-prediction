"""
Feature ablation under walk-forward evaluation.

The dissertation reports that engineered features lift R2 from roughly 0.34 to
0.60 and concludes that information availability matters more than algorithmic
complexity. That measurement was taken under retrospective evaluation, which
the same work argues measures reconstruction rather than prediction, so the
conclusion rests on the protocol whose validity it disputes.

This re-runs the comparison properly: the identical walk-forward protocol,
varying only which features the models receive. Because each configuration is a
full walk-forward run, cost is roughly linear in the number of feature sets --
budget accordingly before running against production data.

The feature sets are nested so that each step isolates one kind of information:

    demographics   who the athlete is, with no performance history at all
    last_result    demographics plus the single previous total
    recent_form    plus rolling mean and dispersion over recent competitions
    full           every forecasting-safe feature

Comparing 'last_result' against 'full' separates "knowing roughly where the
athlete is" from the entire engineered apparatus built on top of it.
"""
from pathlib import Path
from typing import Callable, Dict, List

import pandas as pd

from src.config.features import BASELINE_FEATURES, FORECAST_FEATURES
from src.core.logging import get_logger
from src.evaluation.metrics import compute_global_forecast_metrics

logger = get_logger(__name__)


def build_feature_sets() -> Dict[str, List[str]]:
    """
    Nested feature configurations, smallest first.

    Built by intersecting with FORECAST_FEATURES so a feature removed from the
    forecasting-safe set on leakage grounds cannot reappear here.
    """
    demographics = [f for f in BASELINE_FEATURES if f in FORECAST_FEATURES]

    last_result = demographics + [f for f in ("Prev_Total",) if f in FORECAST_FEATURES]

    recent_form = last_result + [
        f for f in ("Prev_Prev_Total", "Rolling_Mean_3", "Rolling_Std_3")
        if f in FORECAST_FEATURES
    ]

    return {
        "demographics": demographics,
        "last_result": last_result,
        "recent_form": recent_form,
        "full": list(FORECAST_FEATURES),
    }


def run_ablation(
    df: pd.DataFrame,
    evaluator_factory: Callable,
    feature_engineering_fn: Callable,
    target_fn: Callable,
    feature_sets: Dict[str, List[str]] = None,
) -> pd.DataFrame:
    """
    Run one full walk-forward evaluation per feature set.

    Args:
        df: Cohort data, uncleaned and unengineered.
        evaluator_factory: Zero-argument callable returning a fresh evaluator.
            A factory rather than an instance because each configuration needs
            unfitted models and empty result accumulators; reusing one
            evaluator would pool predictions across configurations.
        feature_engineering_fn: Fold-local feature construction.
        target_fn: Fold-local target construction.
        feature_sets: Overrides the default configurations.

    Returns:
        Pooled metrics for every model under every feature set.
    """
    feature_sets = feature_sets or build_feature_sets()
    records = []

    for set_name, feature_cols in feature_sets.items():
        if not feature_cols:
            logger.warning("Feature set '%s' is empty; skipping.", set_name)
            continue

        logger.info(
            "Ablation: evaluating feature set '%s' (%d features)",
            set_name, len(feature_cols),
        )

        evaluator = evaluator_factory()
        evaluator.evaluate(df, feature_engineering_fn, target_fn, feature_cols)

        if not evaluator.prediction_results:
            logger.warning("Feature set '%s' produced no predictions.", set_name)
            continue

        predictions = pd.DataFrame(evaluator.prediction_results)
        metrics = compute_global_forecast_metrics(predictions)
        metrics.insert(0, "feature_set", set_name)
        metrics.insert(1, "n_features", len(feature_cols))
        metrics["features"] = ", ".join(feature_cols)
        records.append(metrics)

    if not records:
        return pd.DataFrame()

    return pd.concat(records, ignore_index=True)


# Estimators that read their inputs by name from the frame and so predict
# identically under every feature set. They belong in the table as reference
# lines, but including them when computing "the best model for this feature
# set" would mask the effect being measured: an invariant baseline that happens
# to be strong would be reported as the winner everywhere, making every
# configuration look equally good and every information gain read as zero.
REFERENCE_MODELS = frozenset({"Persistence", "Rolling Mean", "Drift", "Epley", "Brzycki"})


def summarise_information_vs_algorithm(
    ablation_metrics: pd.DataFrame,
    reference_models: frozenset = REFERENCE_MODELS,
) -> pd.DataFrame:
    """
    Separate the gain from adding information from the gain from changing model.

    For each feature set: the spread in MAE across models (what choosing a
    different algorithm buys) against the improvement over the demographics-only
    configuration (what adding information buys). If the second dominates the
    first, model choice is the less important decision -- which is the
    dissertation's claim, here tested under a forecasting protocol.
    """
    if ablation_metrics.empty:
        return pd.DataFrame()

    learned = ablation_metrics[~ablation_metrics["model"].isin(reference_models)]
    if learned.empty:
        logger.warning("Ablation contains no learned models to summarise.")
        return pd.DataFrame()

    baseline_set = "demographics"
    baseline_best = None
    if baseline_set in set(learned["feature_set"]):
        baseline_best = learned[learned["feature_set"] == baseline_set]["MAE"].min()

    records = []
    for set_name, group in learned.groupby("feature_set", sort=False):
        best_mae = group["MAE"].min()
        worst_mae = group["MAE"].max()

        references = ablation_metrics[
            (ablation_metrics["feature_set"] == set_name)
            & (ablation_metrics["model"].isin(reference_models))
        ]
        best_reference = references["MAE"].min() if not references.empty else float("nan")

        records.append({
            "feature_set": set_name,
            "n_features": int(group["n_features"].iloc[0]),
            "best_model": group.loc[group["MAE"].idxmin(), "model"],
            "best_MAE": best_mae,
            "worst_MAE": worst_mae,
            # What switching algorithm is worth, holding information fixed.
            "spread_across_models": worst_mae - best_mae,
            # What adding this information is worth, best model at each step.
            "gain_over_demographics": (
                baseline_best - best_mae if baseline_best is not None else float("nan")
            ),
            # The invariant reference line, for scale.
            "best_reference_MAE": best_reference,
            # Positive means the learned models have overtaken simply carrying
            # recent performance forward; negative means they have not.
            "learned_advantage_over_reference": best_reference - best_mae,
        })

    return pd.DataFrame(records)


def run_and_save_ablation(
    df: pd.DataFrame,
    evaluator_factory: Callable,
    feature_engineering_fn: Callable,
    target_fn: Callable,
    output_dir: Path,
    feature_sets: Dict[str, List[str]] = None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = run_ablation(
        df, evaluator_factory, feature_engineering_fn, target_fn, feature_sets
    )

    if metrics.empty:
        logger.warning("Ablation produced no results.")
        return metrics, pd.DataFrame()

    metrics.to_csv(output_dir / "ablation_metrics.csv", index=False)

    summary = summarise_information_vs_algorithm(metrics)
    summary.to_csv(output_dir / "ablation_summary.csv", index=False)

    logger.info("Saved ablation results to %s", output_dir)
    for _, row in summary.iterrows():
        logger.info(
            "  %-14s (%2d feat) best learned MAE %7.2f via %-18s | "
            "model spread %6.2f | gain vs demographics %7.2f | "
            "vs best reference %+7.2f",
            row["feature_set"], row["n_features"], row["best_MAE"],
            row["best_model"], row["spread_across_models"],
            row["gain_over_demographics"],
            row["learned_advantage_over_reference"],
        )

    return metrics, summary
