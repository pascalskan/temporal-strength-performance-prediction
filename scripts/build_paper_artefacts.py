"""
Assemble publication figures and tables from stored results.

Reads the result CSVs and writes paper-ready artefacts to results/paper/.
Nothing is recomputed: a figure that re-derived its own numbers could disagree
with the output it claims to summarise, and restyling would then mean re-running
models. This is seconds of work and can be repeated freely.

    python scripts/build_paper_artefacts.py
    python scripts/build_paper_artefacts.py --cohort raw

Missing inputs are reported and skipped rather than fatal, so the script is
usable against a partial result tree -- which is the normal state while a long
suite is still running.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

from src.core.logging import get_logger
from src.data.provenance import MANIFEST_FILENAME, load_manifest
from src.io.paths import ProjectPaths
from src.reporting import figures, tables
from src.reporting.style import apply_style

logger = get_logger(__name__)

COHORTS = ("raw", "equipped")


def read(path: Path):
    """Load a result CSV, or None if this run did not produce it."""
    if not path.exists():
        logger.warning("Missing input: %s", path)
        return None
    frame = pd.read_csv(path)
    return frame if not frame.empty else None


def build_cohort(cohort: str, results_root: Path, output_root: Path,
                 dataset_manifest: dict) -> int:
    cohort_dir = results_root / cohort
    if not cohort_dir.exists():
        logger.warning("No results for cohort '%s'; skipping.", cohort)
        return 0

    figure_dir = output_root / "figures" / cohort
    table_dir = output_root / "tables" / cohort

    wf = cohort_dir / "walk_forward"
    summary = read(wf / "walk_forward_summary.csv")
    predictions_path = wf / "walk_forward_predictions.csv"
    comparisons = read(wf / "walk_forward_comparison.csv")
    matched = read(wf / "matched_subset" / "matched_subset_summary.csv")
    coverage = read(wf / "matched_subset" / "model_coverage.csv")
    temporal = read(wf / "temporal_analysis" / "temporal_metrics.csv")
    disparities = read(wf / "subgroups" / "subgroup_disparities.csv")

    retro = read(cohort_dir / "forward" / "metrics_retro.csv")
    forward = read(cohort_dir / "forward" / "metrics.csv")
    ablation_summary = read(cohort_dir / "ablation" / "ablation_summary.csv")
    tuned_summary = read(
        cohort_dir / "walk_forward_tuned" / "walk_forward_summary.csv"
    )

    if summary is None:
        logger.warning("Cohort '%s' has no walk-forward summary; skipping.", cohort)
        return 0

    produced = 0

    # ---- tables ---------------------------------------------------------
    if predictions_path.exists():
        # Only the columns the table needs: the prediction file runs to
        # hundreds of megabytes.
        predictions = pd.read_csv(
            predictions_path,
            usecols=["model", "athlete_id", "forecast_window"],
        )
        tables.write_table(
            tables.dataset_characteristics({cohort: predictions}, dataset_manifest),
            table_dir, "dataset_characteristics",
            f"Evaluation data for the {cohort} cohort.", "%.0f",
        )
        produced += 1

    tables.write_table(
        tables.primary_results(summary), table_dir, "primary_results",
        f"Walk-forward performance, {cohort} cohort, pooled over all predictions.",
    )
    produced += 1

    if retro is not None and forward is not None:
        protocol = tables.protocol_comparison(retro, forward, summary)
        tables.write_table(
            protocol, table_dir, "protocol_comparison",
            f"Identical models under three evaluation protocols, {cohort} cohort.",
            "%.3f",
        )
        figures.figure_protocol_collapse(protocol, figure_dir / "protocol_collapse")
        produced += 2

    if matched is not None and coverage is not None:
        tables.write_table(
            tables.matched_subset(matched, coverage), table_dir, "matched_subset",
            f"Every model restricted to shared observations, {cohort} cohort.",
        )
        figures.figure_coverage_effect(summary, matched, figure_dir / "coverage_effect")
        produced += 2

    if comparisons is not None:
        tables.write_table(
            tables.pairwise_comparisons(comparisons), table_dir, "pairwise_comparisons",
            f"Paired cluster bootstrap comparisons with Holm correction, "
            f"{cohort} cohort.", "%.4f",
        )
        produced += 1

    if disparities is not None:
        tables.write_table(
            tables.subgroup_robustness(disparities), table_dir, "subgroup_robustness",
            f"Subgroup disparity across adequately powered groups, {cohort} cohort.",
        )
        figures.figure_subgroup_disparity(
            disparities, "Gradient Boosting", figure_dir / "subgroup_disparity"
        )
        produced += 2

    baselines = summary[summary["model"].isin(["Persistence", "Rolling Mean"])]
    best_baseline = baselines["MAE"].min() if not baselines.empty else float("nan")

    if ablation_summary is not None:
        tables.write_table(
            tables.ablation(ablation_summary), table_dir, "ablation",
            f"Feature ablation under walk-forward evaluation, {cohort} cohort.",
        )
        figures.figure_ablation(
            ablation_summary, best_baseline, figure_dir / "ablation"
        )
        produced += 2

    tuning = tables.tuning_effect(summary, tuned_summary, best_baseline)
    if tuning is not None:
        tables.write_table(
            tuning, table_dir, "tuning_effect",
            f"Effect of in-fold hyperparameter selection, {cohort} cohort.",
        )
        produced += 1

    # ---- remaining figures ----------------------------------------------
    figures.figure_metric_disagreement(summary, figure_dir / "metric_disagreement")
    produced += 1

    if temporal is not None:
        figures.figure_temporal_stability(
            temporal,
            ["Rolling Mean", "Gradient Boosting", "Persistence"],
            figure_dir / "temporal_stability",
        )
        produced += 1

    logger.info("Cohort '%s': produced %d artefacts", cohort, produced)
    return produced


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=COHORTS,
                        help="Build for one cohort (default: all present).")
    parser.add_argument("--dataset", default="openpowerlifting",
                        help="Results subdirectory to read (default: openpowerlifting).")
    args = parser.parse_args()

    apply_style()

    results_root = ProjectPaths.project_root() / "results" / args.dataset
    if not results_root.exists():
        raise SystemExit(f"No results at {results_root}")

    output_root = results_root / "paper"
    dataset_manifest = load_manifest(ProjectPaths.data_dir() / MANIFEST_FILENAME) or {}

    cohorts = [args.cohort] if args.cohort else list(COHORTS)
    total = sum(
        build_cohort(cohort, results_root, output_root, dataset_manifest)
        for cohort in cohorts
    )

    if total == 0:
        print("No artefacts produced -- are there results to read?")
        return 1

    print(f"Wrote {total} artefacts to {output_root}")
    print("  figures: PNG for drafting, PDF for typesetting")
    print("  tables:  CSV for inspection, LaTeX for the manuscript")
    return 0


if __name__ == "__main__":
    sys.exit(main())
