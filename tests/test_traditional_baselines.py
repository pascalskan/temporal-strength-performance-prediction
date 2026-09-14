"""Traditional 1RM equations as walk-forward baselines."""
import numpy as np
import pandas as pd
import pytest

from src.models.traditional_baselines import (
    BrzyckiBaseline,
    EpleyBaseline,
    brzycki,
    epley,
    estimate_total,
    rpe_to_reps,
)


def _attempts(squat=(100, 110, 120), bench=(80, 85, 90), deadlift=(140, 150, 160), n=1):
    data = {}
    for lift, values in (("Squat", squat), ("Bench", bench), ("Deadlift", deadlift)):
        for i, v in enumerate(values, start=1):
            data[f"{lift}{i}Kg"] = [v] * n
    return pd.DataFrame(data)


def test_epley_matches_the_published_formula():
    assert epley(100, 5) == pytest.approx(100 * (1 + 5 / 30))


def test_brzycki_matches_the_published_formula():
    assert brzycki(100, 5) == pytest.approx(100 / (1.0278 - 0.0278 * 5))


def test_brzycki_caps_reps_to_stay_defined():
    """The denominator approaches zero as reps grow; beyond the cap it would flip sign."""
    assert brzycki(100, 40) == brzycki(100, 10)
    assert brzycki(100, 40) > 0


def test_rpe_maps_monotonically_to_reps():
    reps = [rpe_to_reps(r) for r in (10, 9.5, 9, 8.5, 8, 7, 6)]
    assert reps == sorted(reps), "higher RPE must imply fewer reps in reserve"


def test_estimates_a_total_from_complete_attempts():
    df = _attempts()
    for formula in ("epley", "brzycki"):
        total = estimate_total(df, formula)
        assert total.notna().all()
        assert total.iloc[0] > 0


@pytest.mark.parametrize("missing_lift", ["Squat", "Bench", "Deadlift"])
def test_missing_lift_yields_nan_not_a_partial_total(missing_lift):
    """
    A total cannot be assembled from two lifts. Summing with a missing lift
    treated as zero would produce a confidently wrong, badly low prediction
    rather than an honest absence.
    """
    df = _attempts()
    for i in (1, 2, 3):
        df[f"{missing_lift}{i}Kg"] = np.nan

    assert estimate_total(df, "epley").isna().all()


def test_partial_attempts_within_a_lift_still_estimate():
    """One recorded attempt per lift is enough; the mean skips the absent ones."""
    df = _attempts()
    df["Squat2Kg"] = np.nan
    df["Squat3Kg"] = np.nan

    assert estimate_total(df, "epley").notna().all()


def test_non_positive_loads_are_treated_as_absent():
    """Failed or unrecorded attempts appear as zero or negative loads."""
    df = _attempts(squat=(0, -100, 0))
    assert estimate_total(df, "epley").isna().all()


def test_baselines_declare_they_need_the_full_frame():
    """
    Attempt columns are deliberately excluded from the forecasting feature set,
    so the evaluator must hand these estimators the frame rather than X.
    """
    for baseline in (EpleyBaseline(), BrzyckiBaseline()):
        assert baseline.requires_frame is True


def test_baselines_do_not_learn_from_training_data():
    """Fixed equations: fit must not alter later predictions."""
    df = _attempts(n=3)
    baseline = EpleyBaseline()

    before = baseline.predict(df).tolist()
    baseline.fit(df, pd.Series([400.0, 410.0, 420.0]))
    after = baseline.predict(df).tolist()

    assert before == after


def test_baseline_rejects_a_frame_with_no_attempt_columns():
    baseline = BrzyckiBaseline()
    with pytest.raises(ValueError, match="requires attempt columns"):
        baseline.predict(pd.DataFrame({"Prev_Total": [400.0]}))


def test_unknown_formula_is_rejected():
    from src.models.traditional_baselines import TraditionalFormulaBaseline

    with pytest.raises(ValueError, match="Unknown formula"):
        TraditionalFormulaBaseline("lombardi")


def test_brzycki_and_epley_disagree_on_the_same_attempts():
    """Sanity check that the two formulas are genuinely distinct estimators."""
    df = _attempts()
    assert estimate_total(df, "epley").iloc[0] != pytest.approx(
        estimate_total(df, "brzycki").iloc[0]
    )
