import numpy as np

from src.config.constants import (
    EQUIPPED_EQUIPMENT,
    RAW_EQUIPMENT,
    VALID_EQUIPMENT,
)
from src.config.features import ATTEMPT_COLUMNS


def clean_data(df):
    """
    Clean raw OpenPowerlifting dataset for modelling.

    Steps:
    - Remove rows missing core required values
    - Restrict dataset to supported equipment classes
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

    # Restrict to supported equipment categories
    df = df[df["Equipment"].isin(VALID_EQUIPMENT)].copy()

    # Encode sex
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


def split_equipment_cohorts(df):
    """
    Partition cleaned data into the raw and equipped evaluation cohorts.

    Both entrypoints previously hardcoded their own equipment lists, which is
    how they came to disagree with the cleaning filter. Routing every caller
    through this function keeps the split and the filter defined in one place.

    Args:
        df: Cleaned data, already restricted to VALID_EQUIPMENT.

    Returns:
        (df_raw, df_equipped) as independent copies.
    """
    df_raw = df[df["Equipment"].isin(RAW_EQUIPMENT)].copy()
    df_equipped = df[df["Equipment"].isin(EQUIPPED_EQUIPMENT)].copy()
    return df_raw, df_equipped
