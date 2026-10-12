"""
Publication tables, assembled from stored results.

Built from the result CSVs rather than recomputed, so a table can never
disagree with the output it claims to summarise and restyling costs nothing.

Each table is emitted as CSV and as LaTeX. The LaTeX form is what a paper
actually needs, and hand-transcribing numbers into a manuscript is precisely
the manual step that puts a figure in print with no traceable provenance.
"""
from pathlib import Path
from typing import Optional

import pandas as pd

from src.core.logging import get_logger

logger = get_logger(__name__)

# Order models by family, then by role within it, so every table reads the same
# way: what simple continuation achieves, then the equations, then what was
# learned.
MODEL_ORDER = [
    "Persistence", "Rolling Mean", "Drift",
    "Epley", "Brzycki",
    "Linear Regression", "Ridge", "Random Forest", "Gradient Boosting",
]

MODEL_FAMILY = {
    "Persistence": "Temporal baseline",
    "Rolling Mean": "Temporal baseline",
    "Drift": "Temporal baseline",
    "Epley": "Traditional equation",
    "Brzycki": "Traditional equation",
    "Linear Regression": "Learned",
    "Ridge": "Learned",
    "Random Forest": "Learned",
    "Gradient Boosting": "Learned",
}


def order_models(frame: pd.DataFrame, column: str = "model") -> pd.DataFrame:
    """Sort by family order, leaving anything unrecognised at the end."""
    rank = {name: i for i, name in enumerate(MODEL_ORDER)}
    return (
        frame.assign(_rank=frame[column].map(lambda m: rank.get(m, len(rank))))
        .sort_values(["_rank", column])
        .drop(columns="_rank")
        .reset_index(drop=True)
    )


def to_latex(frame: pd.DataFrame, caption: str, label: str,
             float_format: str = "%.2f") -> str:
    """
    Render a LaTeX table.

    booktabs rules; no vertical lines. Column headers are escaped, since metric
    names contain characters LaTeX would otherwise interpret.
    """
    def escape(text: str) -> str:
        for char, replacement in (
            ("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
            ("$", r"\$"), ("#", r"\#"), ("_", r"\_"),
            ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"),
        ):
            text = text.replace(char, replacement)
        return text

    body = frame.copy()
    body.columns = [escape(str(c)) for c in body.columns]
    for column in body.select_dtypes(include="object").columns:
        body[column] = body[column].map(lambda v: escape(str(v)))

    rendered = body.to_latex(
        index=False,
        escape=False,
        float_format=lambda v: float_format % v,
        column_format="l" + "r" * (len(body.columns) - 1),
    )

    # to_latex emits tabular rules; swap in booktabs for publication.
    rendered = (
        rendered.replace(r"\toprule", r"\toprule")
        .replace(r"\midrule", r"\midrule")
        .replace(r"\bottomrule", r"\bottomrule")
    )

    return (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        f"\\caption{{{escape(caption)}}}\n"
        f"\\label{{tab:{label}}}\n"
        f"{rendered}"
        "\\end{table}\n"
    )


def write_table(frame: pd.DataFrame, output_dir: Path, name: str,
                caption: str, float_format: str = "%.2f") -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame.to_csv(output_dir / f"{name}.csv", index=False)
    (output_dir / f"{name}.tex").write_text(
        to_latex(frame, caption, name, float_format), encoding="utf-8"
    )
    logger.info("Table %-28s %d rows", name, len(frame))


# ---------------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------------

def dataset_characteristics(cohorts: dict, dataset_manifest: dict) -> pd.DataFrame:
    """
    What the study was run on.

    Reports the snapshot digest alongside the counts: OpenPowerlifting revises
    historical records, so a row count alone does not identify the data.
    """
    content = dataset_manifest.get("content", {})
    rows = []

    for cohort, predictions in cohorts.items():
        if predictions is None or predictions.empty:
            continue
        per_model = predictions.groupby("model").size()
        rows.append({
            "Cohort": cohort,
            "Forward predictions": int(per_model.max()),
            "Athletes": int(predictions["athlete_id"].nunique()),
            "Forecast windows": int(predictions["forecast_window"].nunique()),
            "First window": int(predictions["forecast_window"].min()),
            "Last window": int(predictions["forecast_window"].max()),
        })

    frame = pd.DataFrame(rows)
    frame.attrs["source_rows"] = content.get("rows")
    frame.attrs["sha256"] = dataset_manifest.get("sha256")
    return frame


def protocol_comparison(retro: pd.DataFrame, forward: pd.DataFrame,
                        walk_forward: pd.DataFrame) -> pd.DataFrame:
    """
    The study's central table: identical models, protocol the only difference.

    Restricted to models present in all three protocols. The retrospective and
    forward arms come from the forward pipeline, where both are built from the
    same factories with no tuning on either side -- so the difference between
    them is attributable to the protocol alone, which is not true of the
    standalone retrospective pipeline.
    """
    retro = retro.set_index("Model")
    forward = forward.set_index("Model")
    walk_forward = walk_forward.set_index("model")

    shared = [m for m in MODEL_ORDER
              if m in retro.index and m in forward.index and m in walk_forward.index]

    rows = []
    for model in shared:
        rows.append({
            "Model": model,
            "Retrospective R2": retro.loc[model, "R2"],
            "Forward R2": forward.loc[model, "R2"],
            "Walk-forward R2": walk_forward.loc[model, "R2"],
            "Retrospective MAE": retro.loc[model, "MAE"],
            "Forward MAE": forward.loc[model, "MAE"],
            "Walk-forward MAE": walk_forward.loc[model, "MAE"],
        })

    return pd.DataFrame(rows)


def primary_results(summary: pd.DataFrame) -> pd.DataFrame:
    """
    Pooled walk-forward performance.

    Prediction count travels with every row: the traditional equations score a
    subpopulation, and a table that omits n invites the reader to compare
    across different populations.
    """
    frame = order_models(summary)
    return pd.DataFrame({
        "Model": frame["model"],
        "Family": frame["model"].map(MODEL_FAMILY),
        "n": frame["prediction_count"],
        "MAE": frame["MAE"],
        "RMSE": frame["RMSE"],
        "R2": frame["R2"],
        "Median AE": frame["median_absolute_error"],
        "Mean residual": frame["mean_residual"],
    })


def matched_subset(matched: pd.DataFrame, coverage: pd.DataFrame) -> pd.DataFrame:
    """Every model re-scored over the observations all models could address."""
    frame = order_models(matched)
    coverage_by_model = coverage.set_index("model")["coverage_fraction"]

    return pd.DataFrame({
        "Model": frame["model"],
        "Family": frame["model"].map(MODEL_FAMILY),
        "Coverage": frame["model"].map(coverage_by_model),
        "n": frame["prediction_count"],
        "MAE": frame["MAE"],
        "RMSE": frame["RMSE"],
        "R2": frame["R2"],
    })


def ablation(summary: pd.DataFrame) -> pd.DataFrame:
    """Information gain against the gain from changing algorithm."""
    return pd.DataFrame({
        "Feature set": summary["feature_set"],
        "Features": summary["n_features"],
        "Best model": summary["best_model"],
        "Best MAE": summary["best_MAE"],
        "Gain vs demographics": summary["gain_over_demographics"],
        "Spread across models": summary["spread_across_models"],
        "vs best baseline": summary["learned_advantage_over_reference"],
    })


def pairwise_comparisons(comparisons: pd.DataFrame) -> pd.DataFrame:
    """
    Paired cluster bootstrap results with family-wise error control.

    Both the raw and adjusted p-values are shown. Which comparisons were run is
    part of what a reader needs in order to judge the adjustment, so hiding the
    unadjusted value would remove the evidence for the correction.
    """
    columns = {
        "model_a": "Model A", "model_b": "Model B", "metric": "Metric",
        "observed_difference": "Difference", "ci_lower": "CI lower",
        "ci_upper": "CI upper", "bootstrap_p": "p",
    }
    frame = comparisons.rename(columns=columns)
    ordered = [c for c in columns.values() if c in frame.columns]

    if "p_adjusted_holm" in comparisons.columns:
        frame["p (Holm)"] = comparisons["p_adjusted_holm"]
        ordered.append("p (Holm)")

    return frame[ordered]


def tuning_effect(untuned: pd.DataFrame, tuned: Optional[pd.DataFrame],
                  baselines_best: float) -> Optional[pd.DataFrame]:
    """
    Whether selecting hyperparameters changes the conclusion.

    The final column is the one that matters: not whether tuning helped, but
    whether it closed the gap to simple continuation.
    """
    if tuned is None or tuned.empty:
        return None

    untuned_mae = untuned.set_index("model")["MAE"]
    tuned_mae = tuned.set_index("model")["MAE"]

    learned = [m for m in MODEL_ORDER
               if MODEL_FAMILY.get(m) == "Learned" and m in tuned_mae.index]

    rows = []
    for model in learned:
        rows.append({
            "Model": model,
            "Fixed MAE": untuned_mae.get(model, float("nan")),
            "Tuned MAE": tuned_mae[model],
            "Gain from tuning": untuned_mae.get(model, float("nan")) - tuned_mae[model],
            "vs best baseline": baselines_best - tuned_mae[model],
        })

    return pd.DataFrame(rows)


def subgroup_robustness(disparities: pd.DataFrame) -> pd.DataFrame:
    """
    Spread between best- and worst-served subgroup, per dimension.

    Sample sizes are carried through: an error gap between groups of very
    different size is often a statement about sample size.
    """
    frame = disparities.copy()
    return pd.DataFrame({
        "Dimension": frame["dimension"],
        "Model": frame["model"],
        "Best subgroup": frame["best_subgroup"],
        "Best MAE": frame["best_MAE"],
        "n (best)": frame["best_n"],
        "Worst subgroup": frame["worst_subgroup"],
        "Worst MAE": frame["worst_MAE"],
        "n (worst)": frame["worst_n"],
        "Gap": frame["MAE_gap"],
    })
