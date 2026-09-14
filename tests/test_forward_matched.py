"""
Like-for-like comparison in the forward protocol.

metrics.csv listed the traditional equations beside the learned models with no
indication they were scored on different populations. On the Raw cohort that
made Brzycki appear to beat gradient boosting by 11.7 kg where, on shared
observations, the gap is 2.1 kg.
"""
import numpy as np
import pandas as pd
import pytest

from src.evaluation.forward_matched import (
    compute_matched_forward_metrics,
    save_matched_forward_metrics,
)


@pytest.fixture
def scenario():
    """
    Four observations. The traditional equation scores only the two it has
    attempts for, and those two happen to be the ones the learned model
    predicts worst -- the direction that makes an unmatched table flattering
    to the traditional equation.
    """
    y_test = pd.Series([400.0, 500.0, 600.0, 700.0], index=[0, 1, 2, 3])

    learned = pd.Series([410.0, 510.0, 660.0, 760.0], index=[0, 1, 2, 3])
    traditional = pd.Series([np.nan, np.nan, 610.0, 710.0], index=[0, 1, 2, 3])

    return y_test, learned, traditional, pd.Index([2, 3])


def test_all_models_are_scored_on_the_same_observations(scenario):
    y_test, learned, traditional, scored_index = scenario

    matched = compute_matched_forward_metrics(
        {"Learned": learned, "Traditional": traditional}, y_test, scored_index
    ).set_index("Model")

    assert set(matched["n_scored"]) == {2}


def test_restriction_changes_the_apparent_comparison(scenario):
    """
    Across the full test set the learned model averages 20 kg of error; the
    traditional equation averages 10 kg on its own easier subset. Restricting
    to shared observations shows the learned model at 60 kg there -- the
    comparison the unmatched table concealed.
    """
    y_test, learned, traditional, scored_index = scenario

    full_population_learned_mae = (learned - y_test).abs().mean()
    assert full_population_learned_mae == pytest.approx(35.0)

    matched = compute_matched_forward_metrics(
        {"Learned": learned, "Traditional": traditional}, y_test, scored_index
    ).set_index("Model")

    assert matched.loc["Learned", "MAE"] == pytest.approx(60.0)
    assert matched.loc["Traditional", "MAE"] == pytest.approx(10.0)


def test_no_overlap_returns_empty_without_raising(scenario):
    y_test, learned, traditional, _ = scenario

    matched = compute_matched_forward_metrics(
        {"Learned": learned}, y_test, pd.Index([99])
    )
    assert matched.empty


def test_models_with_no_usable_predictions_are_omitted(scenario):
    y_test, learned, _, scored_index = scenario
    absent = pd.Series([np.nan] * 4, index=y_test.index)

    matched = compute_matched_forward_metrics(
        {"Learned": learned, "Absent": absent}, y_test, scored_index
    )
    assert set(matched["Model"]) == {"Learned"}


def test_output_is_written(scenario, tmp_path):
    y_test, learned, traditional, scored_index = scenario

    save_matched_forward_metrics(
        {"Learned": learned, "Traditional": traditional},
        y_test, scored_index, tmp_path,
    )
    assert (tmp_path / "metrics_matched_subset.csv").exists()


def test_nothing_written_when_there_is_no_overlap(scenario, tmp_path):
    y_test, learned, _, _ = scenario

    save_matched_forward_metrics({"Learned": learned}, y_test, pd.Index([99]), tmp_path)
    assert not (tmp_path / "metrics_matched_subset.csv").exists()
