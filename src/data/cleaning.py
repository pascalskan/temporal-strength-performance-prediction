import numpy as np

from src.config.features import ATTEMPT_COLUMNS


def clean_data(df):
    """
    Clean raw OpenPowerlifting dataset for modelling.

    Steps:
    - Remove rows missing core required values
    - Encode sex as numeric
    - Replace invalid attempt values (<= 0) with NaN
    """

    df = df.dropna(
        subset=[
            "TotalKg",
            "Age",
            "BodyweightKg",
            "Equipment"
        ]
    ).copy()

    # Encode Sex
    df["Sex"] = df["Sex"].map({
        "M": 1,
        "F": 0
    })

    # Clean attempt columns
    for col in ATTEMPT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: x if x > 0 else np.nan
            )

    return df