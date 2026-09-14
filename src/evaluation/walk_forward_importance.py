"""
Feature attribution under walk-forward evaluation.

Feature importance existed only for the retrospective protocol. The
dissertation's headline observation -- that a rolling mean accounts for roughly
80-90% of total importance -- was therefore measured under the protocol the
study argues measures reconstruction. Whether the same concentration holds when
models are made to forecast is a different question, and this answers it.

Attribution is captured per fold while each model is still fitted on that
fold's training window, then aggregated. Fold-level capture also shows whether
a model's reliance on a feature is stable over time or drifts as the training
window grows.

Interpretation limits, which matter here more than usual:

- These are associations under a fitted model, not evidence that a feature
  drives performance. They cannot support a causal reading.
- Impurity-based importance inflates high-cardinality and correlated features.
  The engineered features are strongly correlated by construction -- rolling
  mean, previous total and personal best all encode recent performance -- so
  attribution is split among them somewhat arbitrarily, and a low score does
  not mean a feature carries no information.
- Tree importances and linear coefficients measure different things. They are
  normalised to shares so ranks can be compared, but their magnitudes are not
  equivalent.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.core.logging import get_logger

logger = get_logger(__name__)

TOP_N_FOR_PLOT = 12


def aggregate_importance(importance_df: pd.DataFrame) -> pd.DataFrame:
    """
    Mean attribution share per model per feature, across folds.

    Also reports the spread across folds: a feature with a high mean and a wide
    spread was decisive in some periods and ignored in others, which a single
    averaged number would conceal.
    """
    grouped = importance_df.groupby(["model", "feature"])["attribution_share"]

    aggregated = grouped.agg(
        mean_share="mean",
        std_share="std",
        min_share="min",
        max_share="max",
        folds="count",
    ).reset_index()

    aggregated["rank_within_model"] = (
        aggregated.groupby("model")["mean_share"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    return aggregated.sort_values(["model", "mean_share"], ascending=[True, False])


def summarise_concentration(aggregated: pd.DataFrame) -> pd.DataFrame:
    """
    How concentrated each model's attribution is.

    The dissertation's claim is one of concentration -- that a single feature
    dominates -- so the concentration itself is the quantity of interest, not
    any individual feature's score.
    """
    records = []

    for model, group in aggregated.groupby("model"):
        ordered = group.sort_values("mean_share", ascending=False)
        shares = ordered["mean_share"].to_numpy()

        records.append({
            "model": model,
            "top_feature": ordered.iloc[0]["feature"],
            "top_feature_share": shares[0],
            "top_3_share": shares[:3].sum(),
            "features_to_reach_80_percent": int(
                (shares.cumsum() < 0.80).sum() + 1
            ),
            "n_features": len(shares),
        })

    return pd.DataFrame(records).sort_values("top_feature_share", ascending=False)


def plot_importance(aggregated: pd.DataFrame, model: str, save_path: Path) -> None:
    group = (
        aggregated[aggregated["model"] == model]
        .sort_values("mean_share", ascending=False)
        .head(TOP_N_FOR_PLOT)
        .iloc[::-1]
    )
    if group.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(group["feature"], group["mean_share"], xerr=group["std_share"].fillna(0))
    ax.set_xlabel("Mean attribution share across folds (error bars: SD across folds)")
    ax.set_title(f"Walk-forward feature attribution - {model}")
    ax.grid(axis="x", alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_stability_over_time(importance_df: pd.DataFrame, model: str, save_path: Path) -> None:
    """Attribution share by forecast window, for the features that matter most."""
    subset = importance_df[importance_df["model"] == model]
    if subset.empty:
        return

    top_features = (
        subset.groupby("feature")["attribution_share"].mean()
        .sort_values(ascending=False).head(5).index
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    for feature in top_features:
        series = (
            subset[subset["feature"] == feature]
            .sort_values("forecast_window")
        )
        ax.plot(series["forecast_window"], series["attribution_share"],
                marker="o", markersize=3, label=feature)

    ax.set_xlabel("Forecast window")
    ax.set_ylabel("Attribution share")
    ax.set_title(f"Attribution stability across folds - {model}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def run_and_save_importance(importance_df: pd.DataFrame, output_dir: Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if importance_df.empty:
        logger.warning("No feature attributions were captured.")
        return pd.DataFrame(), pd.DataFrame()

    importance_df.to_csv(output_dir / "importance_by_fold.csv", index=False)

    aggregated = aggregate_importance(importance_df)
    aggregated.to_csv(output_dir / "importance_summary.csv", index=False)

    concentration = summarise_concentration(aggregated)
    concentration.to_csv(output_dir / "importance_concentration.csv", index=False)

    for model in aggregated["model"].unique():
        safe = model.lower().replace(" ", "_")
        plot_importance(aggregated, model, output_dir / f"importance_{safe}.png")
        plot_stability_over_time(
            importance_df, model, output_dir / f"importance_stability_{safe}.png"
        )

    logger.info(
        "Saved feature attribution for %d models to %s",
        aggregated["model"].nunique(), output_dir,
    )
    for _, row in concentration.iterrows():
        logger.info(
            "  %-20s top feature '%s' at %.1f%% share; %d features reach 80%%",
            row["model"], row["top_feature"],
            100 * row["top_feature_share"], row["features_to_reach_80_percent"],
        )

    return aggregated, concentration
