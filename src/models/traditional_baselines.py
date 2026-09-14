"""
Traditional 1RM estimation equations as walk-forward forecasting baselines.

These were available to the retrospective and forward protocols but not to
walk-forward, which is the project's primary methodology -- so the headline
evaluation could not compare against the very methods the study set out to
assess.

What these estimators do
------------------------
Each competition records up to three attempts per lift. An equation converts
an attempt's load into an estimated one-rep max, using an RPE-to-reps mapping
that assumes later attempts are taken closer to maximum. Per-lift estimates are
averaged and summed to a predicted total, which is used as the forecast for the
athlete's NEXT competition.

Why this is legitimate for forecasting
--------------------------------------
The attempts come from a competition that has already taken place at prediction
time, so no future information is used. This is the same construction the
forward pipeline uses, which keeps the two protocols comparable.

The comparability caveat that matters
-------------------------------------
An estimate requires at least one recorded attempt in each of the three lifts.
Attempt data is frequently absent: present for 62.7% of raw records and only
30.4% of equipped ones. These estimators therefore return NaN for the rest
rather than guessing, so their metrics are computed over a subpopulation that
is both smaller and non-random -- plausibly better-documented meets. Comparing
their pooled error against models scored on the full population is not
like-for-like. Use the matched-subset outputs for that comparison.
"""
import numpy as np
import pandas as pd

from src.config.features import ATTEMPT_COLUMNS

LIFTS = ("Squat", "Bench", "Deadlift")

# Attempts are assumed to be taken at increasing effort, so each maps to a
# different implied rep capacity. Mirrors the forward pipeline's assumption.
ATTEMPT_RPE = (7.0, 8.5, 9.5)

# Conservative adjustment applied to the assumed RPE before conversion.
RPE_ADJUSTMENT = 0.5
MIN_RPE = 6.0


def rpe_to_reps(rpe: float) -> float:
    """Implied reps-in-reserve capacity for an RPE."""
    if rpe >= 9.5:
        return 1.0
    if rpe >= 8.5:
        return 2.5
    if rpe >= 7.0:
        return 4.0
    return 5.0


def epley(weight: float, reps: float) -> float:
    return weight * (1 + reps / 30.0)


def brzycki(weight: float, reps: float) -> float:
    # The denominator approaches zero as reps grow; cap to stay well-defined.
    reps = min(reps, 10.0)
    return weight / (1.0278 - 0.0278 * reps)


FORMULAS = {"epley": epley, "brzycki": brzycki}


def estimate_one_rep_max(weight, rpe: float, formula: str) -> float:
    """Estimate a 1RM from a single attempt, or NaN if the attempt is unusable."""
    if pd.isna(weight) or weight <= 0:
        return np.nan
    adjusted = max(MIN_RPE, rpe - RPE_ADJUSTMENT)
    return FORMULAS[formula](weight, rpe_to_reps(adjusted))


def estimate_total(df: pd.DataFrame, formula: str) -> pd.Series:
    """
    Predict competition total for every row, vectorised over attempts.

    Returns NaN for any row lacking a usable attempt in one or more lifts,
    because a total cannot be assembled from a partial set of lifts.
    """
    lift_totals = []

    for lift in LIFTS:
        attempt_estimates = []

        for attempt_index, rpe in enumerate(ATTEMPT_RPE, start=1):
            column = f"{lift}{attempt_index}Kg"
            if column not in df.columns:
                continue

            loads = pd.to_numeric(df[column], errors="coerce")
            loads = loads.where(loads > 0)

            adjusted = max(MIN_RPE, rpe - RPE_ADJUSTMENT)
            reps = rpe_to_reps(adjusted)
            attempt_estimates.append(FORMULAS[formula](loads, reps))

        if not attempt_estimates:
            return pd.Series(np.nan, index=df.index)

        # Mean across whichever attempts were recorded; NaN if none were.
        stacked = pd.concat(attempt_estimates, axis=1)
        lift_totals.append(stacked.mean(axis=1, skipna=True))

    # Summing with skipna=False keeps the NaN when any lift is missing, rather
    # than silently treating an absent lift as zero.
    return pd.concat(lift_totals, axis=1).sum(axis=1, skipna=False)


class TraditionalFormulaBaseline:
    """
    Walk-forward baseline wrapping a traditional 1RM equation.

    Declares requires_frame so the evaluator hands it the full test frame:
    these estimators read attempt columns, which are deliberately excluded
    from the forecasting feature set given to the machine learning models.
    """

    requires_frame = True

    def __init__(self, formula: str):
        if formula not in FORMULAS:
            raise ValueError(
                f"Unknown formula {formula!r}; expected one of {sorted(FORMULAS)}"
            )
        self.formula = formula

    def fit(self, X, y=None):
        # Fixed equations; nothing is estimated from the training window.
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        missing = [c for c in ATTEMPT_COLUMNS if c not in df.columns]
        if len(missing) == len(ATTEMPT_COLUMNS):
            raise ValueError(
                f"{type(self).__name__} requires attempt columns; none present. "
                f"Expected some of: {ATTEMPT_COLUMNS}"
            )
        return estimate_total(df, self.formula)


class EpleyBaseline(TraditionalFormulaBaseline):
    def __init__(self):
        super().__init__("epley")


class BrzyckiBaseline(TraditionalFormulaBaseline):
    def __init__(self):
        super().__init__("brzycki")
