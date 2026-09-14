"""
Assert that a fixture run actually produced the outputs each protocol promises.

`main.py --test` exiting zero is necessary but not sufficient: a stage can be
skipped, write an empty table, or silently lose a model and still exit cleanly.
This checks that every protocol wrote its artefacts, that they contain rows, and
that the walk-forward comparison covers the full model set -- the properties a
reader would assume when citing these outputs.

Intended for CI, but useful locally after any refactor:

    python scripts/verify_fixture_outputs.py
"""
import sys
from pathlib import Path

import pandas as pd

from src.io.paths import ProjectPaths

# Protocol -> files that must exist and be non-empty, relative to the cohort
# directory. Only the raw cohort is asserted: the fixture is small and the
# equipped cohort can legitimately be too sparse for some stages.
REQUIRED_OUTPUTS = {
    "retrospective": [
        "evaluation/model_comparison_full.csv",
        "evaluation/model_results.csv",
        "evaluation/feature_set_comparison.csv",
        "traditional/epley_results.csv",
        "traditional/brzycki_results.csv",
    ],
    "forward": [
        "forward/metrics.csv",
        "forward/metrics_retro.csv",
        "forward/ml_predictions.csv",
    ],
    "walk_forward": [
        "walk_forward/walk_forward_summary.csv",
        "walk_forward/walk_forward_predictions.csv",
        "walk_forward/walk_forward_comparison.csv",
        "walk_forward/walk_forward_confidence_intervals.csv",
        "walk_forward/diagnostics/diagnostic_summary.csv",
        "walk_forward/temporal_analysis/temporal_metrics.csv",
        "walk_forward/matched_subset/model_coverage.csv",
    ],
}

# Every estimator the walk-forward pipeline is configured to evaluate. A model
# silently dropping out of the comparison is exactly the kind of regression an
# exit code cannot reveal.
EXPECTED_WALK_FORWARD_MODELS = {
    "Persistence", "Rolling Mean", "Drift",
    "Epley", "Brzycki",
    "Linear Regression", "Ridge", "Random Forest", "Gradient Boosting",
}


def main() -> int:
    ProjectPaths.set_dataset_scope("test_fixture")
    cohort_dir = ProjectPaths.results_dir() / "raw"

    failures = []

    if not cohort_dir.exists():
        print(f"FAIL: no results for the raw cohort at {cohort_dir}")
        print("      Did `python main.py --test` run?")
        return 1

    for protocol, relative_paths in REQUIRED_OUTPUTS.items():
        for relative in relative_paths:
            path = cohort_dir / relative
            if not path.exists():
                failures.append(f"{protocol}: missing {relative}")
                continue
            try:
                frame = pd.read_csv(path)
            except Exception as error:  # noqa: BLE001 - report, do not mask
                failures.append(f"{protocol}: {relative} unreadable ({error})")
                continue
            if frame.empty:
                failures.append(f"{protocol}: {relative} has no rows")

    summary_path = cohort_dir / "walk_forward/walk_forward_summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        present = set(summary["model"])

        missing_models = EXPECTED_WALK_FORWARD_MODELS - present
        if missing_models:
            failures.append(
                f"walk_forward: models absent from the summary: {sorted(missing_models)}"
            )

        unscored = summary[summary["prediction_count"] == 0]["model"].tolist()
        if unscored:
            failures.append(f"walk_forward: models with zero predictions: {unscored}")

    if failures:
        print(f"FAIL: {len(failures)} problem(s) with the fixture run:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    checked = sum(len(paths) for paths in REQUIRED_OUTPUTS.values())
    print(
        f"OK: all {checked} expected outputs present and populated across "
        f"{len(REQUIRED_OUTPUTS)} protocols; walk-forward covers all "
        f"{len(EXPECTED_WALK_FORWARD_MODELS)} models."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
