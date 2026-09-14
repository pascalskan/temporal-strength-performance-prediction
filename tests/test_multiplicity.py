"""
Family-wise error control across the reported comparisons.

A default configuration reports fourteen tests -- seven model pairs on two
metrics each. At alpha = 0.05 with unadjusted p-values, the probability of at
least one spurious rejection across that family approaches one in two, so
"significant" in the comparison table would not mean what a reader takes it to
mean.
"""
import numpy as np
import pandas as pd
import pytest

from src.evaluation.statistics import add_multiplicity_correction, holm_bonferroni


def test_single_test_is_unchanged():
    assert holm_bonferroni([0.03]) == pytest.approx([0.03])


def test_smallest_p_is_scaled_by_the_family_size():
    """Holm's first step is the Bonferroni bound."""
    adjusted = holm_bonferroni([0.01, 0.04, 0.03])
    assert adjusted[0] == pytest.approx(0.03)


def test_adjusted_values_are_monotone_in_the_raw_values():
    """
    Step-down enforcement: a larger raw p-value can never receive a smaller
    adjusted one, which would make the ordering of evidence incoherent.
    """
    raw = np.array([0.001, 0.02, 0.03, 0.04, 0.2])
    adjusted = holm_bonferroni(raw)

    ordered = adjusted[np.argsort(raw)]
    assert np.all(np.diff(ordered) >= -1e-12)


def test_adjusted_values_never_shrink_below_raw():
    raw = np.array([0.01, 0.02, 0.5])
    assert np.all(holm_bonferroni(raw) >= raw - 1e-12)


def test_adjustment_is_capped_at_one():
    assert np.all(holm_bonferroni([0.4, 0.5, 0.9]) <= 1.0)


def test_holm_is_never_more_conservative_than_bonferroni():
    """Holm is uniformly more powerful; that is the reason for preferring it."""
    raw = np.array([0.001, 0.01, 0.02, 0.04])
    bonferroni = np.minimum(1.0, raw * len(raw))
    assert np.all(holm_bonferroni(raw) <= bonferroni + 1e-12)


def comparison_table(p_values):
    return pd.DataFrame({
        "model_a": ["A"] * len(p_values),
        "model_b": ["B"] * len(p_values),
        "metric": ["mae"] * len(p_values),
        "bootstrap_p": p_values,
    })


def test_correction_preserves_the_unadjusted_values():
    """
    Which comparisons were run is part of what a reader needs to judge the
    adjustment, so the raw p-values must remain visible.
    """
    annotated = add_multiplicity_correction(comparison_table([0.001, 0.02, 0.3]))

    assert list(annotated["bootstrap_p"]) == [0.001, 0.02, 0.3]
    assert "p_adjusted_holm" in annotated.columns


def test_borderline_significance_is_withdrawn_by_the_correction():
    """The case the correction exists for: a result that should not survive."""
    annotated = add_multiplicity_correction(comparison_table([0.001, 0.04, 0.045]))
    borderline = annotated.iloc[2]

    assert borderline["significant_unadjusted"]
    assert not borderline["significant_adjusted"]


def test_strong_results_survive_the_correction():
    annotated = add_multiplicity_correction(comparison_table([1e-6, 1e-5]))
    assert annotated["significant_adjusted"].all()


def test_family_size_is_recorded():
    """Without it the adjustment cannot be reproduced from the table alone."""
    annotated = add_multiplicity_correction(comparison_table([0.01] * 5))
    assert (annotated["n_tests_in_family"] == 5).all()


def test_alpha_is_configurable():
    table = comparison_table([0.02])
    assert add_multiplicity_correction(table, alpha=0.05)["significant_adjusted"].iloc[0]
    assert not add_multiplicity_correction(table, alpha=0.01)["significant_adjusted"].iloc[0]
