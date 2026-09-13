import pandas as pd
from typing import List

def build_athlete_id(df: pd.DataFrame, stable_attributes: List[str] = None) -> pd.Series:
    """
    Builds a unique athlete identifier from a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        stable_attributes (List[str], optional): A list of columns to use for the ID. 
                                                 Defaults to ['Name', 'Sex', 'WeightClassKg'].

    Returns:
        pd.Series: A Series containing the unique athlete IDs.
    """
    if stable_attributes is None:
        stable_attributes = ['Name', 'Sex', 'WeightClassKg']

    # Ensure all required columns are present
    missing_cols = [col for col in stable_attributes if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns for athlete ID: {missing_cols}")

    # Create the composite ID
    return df[stable_attributes].astype(str).agg('_'.join, axis=1)
