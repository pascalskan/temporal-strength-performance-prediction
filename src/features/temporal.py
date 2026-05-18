import numpy as np
from tqdm import tqdm

from src.config.features import ATTEMPT_COLUMNS
from src.core.logging import get_logger


logger = get_logger(__name__)


def feature_engineering(df):
    """
    Create temporal and engineered features for athlete performance modelling.

    Assumes input dataframe contains:
    - Name
    - Date
    - TotalKg
    - Age
    - attempt columns
    """

    logger.info("🔹 Starting feature engineering...")

    df = df.sort_values(["Name", "Date"]).copy()

    assert df.groupby("Name")["Date"].is_monotonic_increasing.all(), (
        "❌ Data leakage risk: Dates are not strictly increasing per athlete"
    )

    logger.info("🔸 Creating historical features...")

    for step in tqdm(range(5), desc="Historical Features"):
        if step == 0:
            df["Prev_Total"] = df.groupby("Name")["TotalKg"].shift(1)

        elif step == 1:
            df["Rolling_Mean_3"] = (
                df.groupby("Name")["TotalKg"]
                .shift(1)
                .rolling(3)
                .mean()
            )

        elif step == 2:
            df["Rolling_Std_3"] = (
                df.groupby("Name")["TotalKg"]
                .shift(1)
                .rolling(3)
                .std()
            )

        elif step == 3:
            df["Comp_Count"] = df.groupby("Name").cumcount()

        elif step == 4:
            df["Days_Since_Last"] = (
                df.groupby("Name")["Date"]
                .diff()
                .dt.days
            )

    df["PB"] = df.groupby("Name")["TotalKg"].shift(1).cummax()

    df["Peak_Distance"] = df["Prev_Total"] - df["PB"]

    logger.info("🔸 Creating progression features...")

    df["Improvement"] = (
        df.groupby("Name")["TotalKg"].shift(1)
        - df.groupby("Name")["TotalKg"].shift(2)
    )

    df["Momentum"] = (
        0.6 * df.groupby("Name")["TotalKg"].shift(1)
        + 0.3 * df.groupby("Name")["TotalKg"].shift(2)
        + 0.1 * df.groupby("Name")["TotalKg"].shift(3)
    )

    logger.info("🔸 Creating consistency features...")

    df["CV"] = df["Rolling_Std_3"] / (df["Rolling_Mean_3"] + 1e-6)

    logger.info("🔸 Creating experience features...")

    first_date = df.groupby("Name")["Date"].transform("min")

    df["Career_Length_Days"] = (df["Date"] - first_date).dt.days

    df["Experience_Density"] = (
        df["Comp_Count"] / (df["Career_Length_Days"] + 1)
    )

    logger.info("🔸 Creating age features...")

    df["Age_Squared"] = df["Age"] ** 2

    logger.info("🔸 Creating attempt behaviour features...")

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

    logger.info("✅ Feature engineering complete.")

    return df
