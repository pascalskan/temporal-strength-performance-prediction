"""
Determinism: the same inputs must produce the same numbers.

Seeds are set throughout, and the dissertation claims reproducibility, but
nothing ever checked it. A fixed seed does not by itself guarantee stable
output -- unordered iteration, dictionary ordering, parallel reduction order
and unstable sorts can all introduce run-to-run variation while every seed
stays fixed. If results are not bit-stable, no published figure can be
verified, and a refactor that silently perturbs them cannot be detected.

These run the real pipeline components against the committed fixture rather
than synthetic stand-ins, because the property being asserted is about the
system as it actually executes.
"""
import numpy as np
import pandas as pd
import pytest

from src.config.features import FORECAST_FEATURES
from src.data.cleaning import clean_data, split_equipment_cohorts
from src.data.identity import build_athlete_id
from src.data.loader import load_data
from src.evaluation.statistics import (
    compute_model_confidence_intervals,
    paired_bootstrap_comparison,
)
from src.features.temporal import feature_engineering
from src.io.paths import ProjectPaths
from src.models.baselines import PersistenceBaseline, RollingMeanBaseline
from src.pipelines.walk_forward_pipeline import create_walk_forward_target


@pytest.fixture(scope="module")
def fixture_cohort():
    path = ProjectPaths.data_dir() / "test_fixture.csv"
    if not path.exists():
        pytest.skip("deterministic fixture not present in this checkout")

    df = clean_data(load_data(path))
    raw, _ = split_equipment_cohorts(df)
    return raw


def build_evaluator():
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression

    from src.evaluation.walk_forward import WalkForwardEvaluator

    return WalkForwardEvaluator(
        models={
            "Linear Regression": LinearRegression(),
            # The stochastic component: bootstrapped samples and random feature
            # subsets at every split. If anything is going to vary between runs,
            # it is this.
            "Random Forest": RandomForestRegressor(
                n_estimators=25, random_state=42, n_jobs=-1
            ),
        },
        baselines={
            "Persistence": PersistenceBaseline(),
            "Rolling Mean": RollingMeanBaseline(),
        },
        granularity="year",
        min_train_periods=5,
    )


def run_once(df):
    evaluator = build_evaluator()
    evaluator.evaluate(df, feature_engineering, create_walk_forward_target, FORECAST_FEATURES)
    return pd.DataFrame(evaluator.prediction_results)


def test_cleaning_is_deterministic(fixture_cohort):
    path = ProjectPaths.data_dir() / "test_fixture.csv"
    first = clean_data(load_data(path))
    second = clean_data(load_data(path))
    pd.testing.assert_frame_equal(first, second)


def test_athlete_identity_is_deterministic(fixture_cohort):
    first = build_athlete_id(fixture_cohort)
    second = build_athlete_id(fixture_cohort)
    pd.testing.assert_series_equal(first, second)


def test_feature_engineering_is_deterministic(fixture_cohort):
    df = fixture_cohort.copy()
    df["Athlete_ID"] = build_athlete_id(df)

    first = feature_engineering(df.copy(), athlete_col="Athlete_ID")
    second = feature_engineering(df.copy(), athlete_col="Athlete_ID")

    pd.testing.assert_frame_equal(
        first.sort_index(axis=1), second.sort_index(axis=1)
    )


def test_walk_forward_predictions_are_identical_across_runs(fixture_cohort):
    """
    The end-to-end property that matters: two runs of the full fold loop,
    including a stochastic ensemble, must agree exactly.
    """
    first = run_once(fixture_cohort)
    second = run_once(fixture_cohort)

    assert not first.empty, "fixture produced no predictions"

    key = ["model", "observation_id"]
    first = first.sort_values(key).reset_index(drop=True)
    second = second.sort_values(key).reset_index(drop=True)

    pd.testing.assert_frame_equal(first, second)


def test_bootstrap_is_reproducible_from_its_seed(fixture_cohort):
    predictions = run_once(fixture_cohort)
    if predictions.empty:
        pytest.skip("no predictions to compare")

    kwargs = dict(metric="mae", n_bootstraps=200, random_seed=7)
    first = paired_bootstrap_comparison(predictions, "Rolling Mean", "Random Forest", **kwargs)
    second = paired_bootstrap_comparison(predictions, "Rolling Mean", "Random Forest", **kwargs)

    assert first == second


def test_bootstrap_without_a_seed_is_allowed_to_vary(fixture_cohort):
    """
    Guards the opposite error: a seeded result that never varies regardless of
    seed would mean the resampling is not actually random, which would make
    every confidence interval meaningless.
    """
    predictions = run_once(fixture_cohort)
    if predictions.empty:
        pytest.skip("no predictions to compare")

    kwargs = dict(metric="mae", n_bootstraps=200)
    first = paired_bootstrap_comparison(predictions, "Rolling Mean", "Random Forest",
                                        random_seed=1, **kwargs)
    second = paired_bootstrap_comparison(predictions, "Rolling Mean", "Random Forest",
                                         random_seed=2, **kwargs)

    assert first["ci_lower"] != second["ci_lower"] or first["ci_upper"] != second["ci_upper"]


def test_confidence_intervals_are_reproducible_from_their_seed(fixture_cohort):
    predictions = run_once(fixture_cohort)
    if predictions.empty:
        pytest.skip("no predictions to compare")

    first = compute_model_confidence_intervals(predictions, n_bootstraps=100, random_seed=3)
    second = compute_model_confidence_intervals(predictions, n_bootstraps=100, random_seed=3)

    pd.testing.assert_frame_equal(first, second)


def test_numpy_default_rng_is_reproducible():
    """
    Sanity check on the assumption the bootstrap rests on. If this ever fails,
    every seeded result in the project is unreliable.
    """
    a = np.random.default_rng(42).normal(size=100)
    b = np.random.default_rng(42).normal(size=100)
    np.testing.assert_array_equal(a, b)
