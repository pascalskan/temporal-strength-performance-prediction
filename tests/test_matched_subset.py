"""Like-for-like comparison on the subpopulation every model can score."""
import numpy as np
import pandas as pd
import pytest

from src.evaluation.matched_subset import (
    compute_matched_subset_metrics,
    find_common_observations,
    run_and_save_matched_subset,
)


def build_predictions(rows):
    """rows: (observation_id, model, y_true, y_pred)"""
    return pd.DataFrame(rows, columns=["observation_id", "model", "y_true", "y_pred"])


def test_common_observations_exclude_those_a_model_could_not_score():
    df = build_predictions([
        ("o1", "ml", 400.0, 390.0),
        ("o1", "trad", 400.0, 405.0),
        ("o2", "ml", 500.0, 480.0),
        ("o2", "trad", 500.0, np.nan),   # no attempts recorded
    ])
    assert list(find_common_observations(df)) == ["o1"]


def test_metrics_are_recomputed_on_the_matched_subset_only():
    """
    The excluded observation is one the ML model predicts badly. If the subset
    were ignored, the ML model would look worse than it does on the data the
    traditional equation can actually address -- which is the comparison being
    made.
    """
    df = build_predictions([
        ("o1", "ml", 400.0, 400.0),
        ("o1", "trad", 400.0, 410.0),
        ("o2", "ml", 500.0, 300.0),     # large error, trad cannot score it
        ("o2", "trad", 500.0, np.nan),
    ])
    metrics, _ = compute_matched_subset_metrics(df)
    by_model = metrics.set_index("model")

    assert by_model.loc["ml", "prediction_count"] == 1
    assert by_model.loc["ml", "MAE"] == pytest.approx(0.0)
    assert by_model.loc["trad", "MAE"] == pytest.approx(10.0)


def test_coverage_reports_what_each_model_could_score():
    df = build_predictions([
        ("o1", "ml", 400.0, 390.0),
        ("o1", "trad", 400.0, 405.0),
        ("o2", "ml", 500.0, 480.0),
        ("o2", "trad", 500.0, np.nan),
    ])
    _, coverage = compute_matched_subset_metrics(df)
    by_model = coverage.set_index("model")

    assert by_model.loc["ml", "coverage_fraction"] == pytest.approx(1.0)
    assert by_model.loc["trad", "coverage_fraction"] == pytest.approx(0.5)
    assert by_model.loc["trad", "matched_subset_size"] == 1
    assert by_model.loc["ml", "matched_fraction_of_all"] == pytest.approx(0.5)


def test_no_shared_observations_returns_empty_without_raising():
    df = build_predictions([
        ("o1", "ml", 400.0, 390.0),
        ("o1", "trad", 400.0, np.nan),
        ("o2", "ml", 500.0, 480.0),
        ("o2", "trad", 500.0, np.nan),
    ])
    metrics, coverage = compute_matched_subset_metrics(df)

    assert metrics.empty
    assert not coverage.empty, "coverage must still explain why the subset is empty"


def test_all_models_scoring_everything_matches_the_full_population():
    df = build_predictions([
        ("o1", "a", 400.0, 390.0),
        ("o1", "b", 400.0, 410.0),
        ("o2", "a", 500.0, 480.0),
        ("o2", "b", 500.0, 520.0),
    ])
    metrics, coverage = compute_matched_subset_metrics(df)

    assert set(metrics["prediction_count"]) == {2}
    assert set(coverage["coverage_fraction"]) == {1.0}


def test_outputs_are_written(tmp_path):
    df = build_predictions([
        ("o1", "ml", 400.0, 390.0),
        ("o1", "trad", 400.0, 405.0),
        ("o2", "ml", 500.0, 480.0),
        ("o2", "trad", 500.0, np.nan),
    ])
    run_and_save_matched_subset(df, tmp_path)

    assert (tmp_path / "model_coverage.csv").exists()
    assert (tmp_path / "matched_subset_summary.csv").exists()


def test_coverage_written_even_when_no_matched_subset_exists(tmp_path):
    df = build_predictions([
        ("o1", "ml", 400.0, 390.0),
        ("o1", "trad", 400.0, np.nan),
    ])
    run_and_save_matched_subset(df, tmp_path)

    assert (tmp_path / "model_coverage.csv").exists()
    assert not (tmp_path / "matched_subset_summary.csv").exists()
