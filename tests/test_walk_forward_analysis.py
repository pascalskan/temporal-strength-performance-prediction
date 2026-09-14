"""Subgroup robustness and feature attribution under walk-forward evaluation."""
import numpy as np
import pandas as pd
import pytest

from src.evaluation.walk_forward_importance import (
    aggregate_importance,
    run_and_save_importance,
    summarise_concentration,
)
from src.evaluation.walk_forward_subgroups import (
    MIN_RELIABLE_PREDICTIONS,
    build_subgroups,
    compute_subgroup_metrics,
    run_and_save_subgroup_analysis,
    summarise_disparities,
)


def make_predictions(n=400, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "observation_id": [f"o{i}" for i in range(n)],
        "athlete_id": [f"a{i % 50}" for i in range(n)],
        "forecast_window": rng.integers(2015, 2025, n),
        "model": ["Rolling Mean"] * (n // 2) + ["Gradient Boosting"] * (n - n // 2),
        "y_true": rng.normal(450, 90, n),
        "y_pred": rng.normal(450, 95, n),
        "Sex": rng.choice([0, 1], n),
        "Age": rng.integers(18, 60, n),
        "BodyweightKg": rng.normal(85, 18, n),
        "Equipment": rng.choice(["Raw"], n),
    })


# --------------------------------------------------------------- subgroups

def test_subgroup_dimensions_are_derived_from_the_prediction_record():
    subgroups = build_subgroups(make_predictions())
    assert {"sex", "bodyweight_quartile", "age_band", "strength_quartile"} <= set(subgroups)


def test_sex_is_decoded_to_labels_not_left_as_codes():
    """Sex is numerically encoded upstream; reporting '0' and '1' is unreadable."""
    subgroups = build_subgroups(make_predictions())
    assert set(subgroups["sex"].unique()) <= {"male", "female", "unknown"}


def test_missing_optional_columns_simply_omit_that_dimension():
    df = make_predictions().drop(columns=["Sex", "Age"])
    subgroups = build_subgroups(df)
    assert "sex" not in subgroups and "age_band" not in subgroups
    assert "strength_quartile" in subgroups, "always derivable from y_true"


def test_metrics_report_sample_size_alongside_error():
    """
    An error gap between groups of very different size is often a statement
    about sample size; reporting error without n invites misreading it.
    """
    metrics = compute_subgroup_metrics(make_predictions())
    assert {"prediction_count", "athlete_count"} <= set(metrics.columns)
    assert (metrics["prediction_count"] > 0).all()


def test_underpowered_subgroups_are_flagged_not_dropped():
    df = make_predictions(n=40)
    metrics = compute_subgroup_metrics(df)
    assert not metrics.empty
    assert not metrics["sufficiently_powered"].any()


def test_disparities_ignore_underpowered_subgroups():
    metrics = compute_subgroup_metrics(make_predictions(n=40))
    assert summarise_disparities(metrics).empty


def test_disparities_with_nothing_comparable_return_empty_frame_with_columns():
    """
    Regression: an empty records list built a column-less frame, and sorting it
    by 'dimension' raised KeyError mid-run.
    """
    metrics = pd.DataFrame([{
        "dimension": "sex", "subgroup": "male", "model": "RF",
        "prediction_count": 5, "athlete_count": 2, "MAE": 1.0, "RMSE": 1.0,
        "R2": 0.1, "mean_residual": 0.0, "sMAPE": 1.0,
        "sufficiently_powered": False,
    }])
    result = summarise_disparities(metrics)
    assert result.empty
    assert "dimension" in result.columns


def test_disparities_identify_the_extremes():
    metrics = pd.DataFrame([
        {"dimension": "sex", "subgroup": "male", "model": "RF", "prediction_count": 500,
         "athlete_count": 100, "MAE": 100.0, "RMSE": 1.0, "R2": 0.5,
         "mean_residual": 0.0, "sMAPE": 1.0, "sufficiently_powered": True},
        {"dimension": "sex", "subgroup": "female", "model": "RF", "prediction_count": 500,
         "athlete_count": 100, "MAE": 130.0, "RMSE": 1.0, "R2": 0.4,
         "mean_residual": 0.0, "sMAPE": 1.0, "sufficiently_powered": True},
    ])
    row = summarise_disparities(metrics).iloc[0]

    assert row["best_subgroup"] == "male"
    assert row["worst_subgroup"] == "female"
    assert row["MAE_gap"] == pytest.approx(30.0)
    assert row["MAE_ratio"] == pytest.approx(1.3)


def test_subgroup_outputs_are_written(tmp_path):
    run_and_save_subgroup_analysis(make_predictions(n=2000), tmp_path)
    assert (tmp_path / "subgroup_metrics.csv").exists()


# ------------------------------------------------------------- importance

def make_importance():
    rows = []
    for window in (2020, 2021, 2022):
        for feature, share in (("Rolling_Mean_3", 0.6), ("Prev_Total", 0.25), ("Age", 0.15)):
            rows.append({
                "forecast_window": window, "model": "Random Forest",
                "feature": feature, "attribution": share,
                "attribution_share": share, "attribution_kind": "impurity",
            })
    return pd.DataFrame(rows)


def test_aggregation_reports_spread_across_folds_not_just_the_mean():
    """A feature decisive in some periods and ignored in others must be visible."""
    aggregated = aggregate_importance(make_importance())
    assert {"mean_share", "std_share", "min_share", "max_share", "folds"} <= set(aggregated.columns)
    assert (aggregated["folds"] == 3).all()


def test_features_are_ranked_within_each_model():
    aggregated = aggregate_importance(make_importance())
    top = aggregated[aggregated["rank_within_model"] == 1].iloc[0]
    assert top["feature"] == "Rolling_Mean_3"


def test_concentration_quantifies_dominance():
    """
    The dissertation's claim is about concentration -- one feature dominating --
    so concentration is the quantity to measure, not any single score.
    """
    concentration = summarise_concentration(aggregate_importance(make_importance())).iloc[0]

    assert concentration["top_feature"] == "Rolling_Mean_3"
    assert concentration["top_feature_share"] == pytest.approx(0.6)
    assert concentration["features_to_reach_80_percent"] == 2


def test_importance_outputs_are_written(tmp_path):
    run_and_save_importance(make_importance(), tmp_path)
    for name in ("importance_by_fold.csv", "importance_summary.csv",
                 "importance_concentration.csv"):
        assert (tmp_path / name).exists()


def test_empty_importance_does_not_raise(tmp_path):
    aggregated, concentration = run_and_save_importance(pd.DataFrame(), tmp_path)
    assert aggregated.empty and concentration.empty
