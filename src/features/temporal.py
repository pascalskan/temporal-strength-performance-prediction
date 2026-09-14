import numpy as np
from tqdm import tqdm

from src.config.features import ATTEMPT_COLUMNS
from src.core.logging import get_logger


logger = get_logger(__name__)


def feature_engineering(df, athlete_col):
    """
    Create temporal and engineered features, grouped per athlete.

    athlete_col is required rather than defaulted. It previously defaulted
    to "AthleteID", a column no part of this project produces, so callers
    that omitted it raised at runtime instead of grouping correctly -- which
    is how the retrospective and forward pipelines came to be broken while
    the walk-forward pipeline, the only caller passing it explicitly, kept
    working. Build the identifier with src.data.identity.build_athlete_id
    and pass its column name.
    """
    if athlete_col not in df.columns:
        raise ValueError(
            f"Missing required athlete identifier column: {athlete_col}"
        )

    logger.info(
        f"Starting feature engineering using athlete identifier: {athlete_col}..."
    )

    df = df.sort_values([athlete_col, "Date"]).copy()

    assert df.groupby(athlete_col)["Date"].is_monotonic_increasing.all(), (
        f"Data leakage risk: Dates are not strictly increasing per {athlete_col}"
    )

    logger.info("Creating historical features...")

    # Add Prev_Prev_Total for Drift Baseline
    df["Prev_Prev_Total"] = df.groupby(athlete_col)["TotalKg"].shift(2)
    df["Prev_Total"] = df.groupby(athlete_col)["TotalKg"].shift(1)

    # SCIENTIFIC FIX: Use group-safe transform and require full windows for rolling features
    df["Rolling_Mean_3"] = df.groupby(athlete_col)["TotalKg"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=3).mean()
    )
    df["Rolling_Std_3"] = df.groupby(athlete_col)["TotalKg"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=3).std()
    )

    df["Comp_Count"] = df.groupby(athlete_col).cumcount()
    df["Days_Since_Last"] = df.groupby(athlete_col)["Date"].diff().dt.days

    # SCIENTIFIC FIX: Ensure PB is calculated per athlete
    df["PB"] = df.groupby(athlete_col)["TotalKg"].transform(lambda x: x.shift(1).cummax())

    df["Peak_Distance"] = df["Prev_Total"] - df["PB"]

    logger.info("Creating progression features...")

    df["Improvement"] = (
        df.groupby(athlete_col)["TotalKg"].shift(1)
        - df.groupby(athlete_col)["TotalKg"].shift(2)
    )

    df["Momentum"] = (
        0.6 * df.groupby(athlete_col)["TotalKg"].shift(1)
        + 0.3 * df.groupby(athlete_col)["TotalKg"].shift(2)
        + 0.1 * df.groupby(athlete_col)["TotalKg"].shift(3)
    )

    logger.info("Creating consistency features...")

    df["CV"] = df["Rolling_Std_3"] / (df["Rolling_Mean_3"] + 1e-6)

    logger.info("Creating experience features...")

    first_date = df.groupby(athlete_col)["Date"].transform("min")

    df["Career_Length_Days"] = (df["Date"] - first_date).dt.days

    df["Experience_Density"] = (
        df["Comp_Count"] / (df["Career_Length_Days"] + 1)
    )

    logger.info("Creating age features...")

    df["Age_Squared"] = df["Age"] ** 2

    logger.info("Creating attempt behaviour features...")

    df["Success_Rate"] = df[ATTEMPT_COLUMNS].notna().sum(axis=1) / 9

    df["Aggression"] = (
        (df["Squat2Kg"] - df["Squat1Kg"])
        + (df["Squat3Kg"] - df["Squat2Kg"])
    ) / 2

    df["Clutch"] = (
        df[["Squat3Kg", "Bench3Kg", "Deadlift3Kg"]]
        .notna()
        .sum(axis=1) / 3
    )

    df["Risk"] = df["Aggression"] * (1 - df["Success_Rate"])

    logger.info("Feature engineering complete.")

    return df