import pandas as pd
from typing import List
from src.core.logging import get_logger

logger = get_logger(__name__)

def build_athlete_id(df: pd.DataFrame, stable_attributes: List[str] = None) -> pd.Series:
    """
    Builds a unique athlete identifier from a DataFrame.
    Enforces strict scientific fallback strategies to prevent cross-athlete contamination.
    """
    if stable_attributes is None:
        # Strict hierarchy of acceptable attributes
        acceptable_strategies = [
            ['AthleteID'],
            ['Name', 'Sex', 'Country'],
            ['Name', 'Sex']
        ]
        
        chosen_attrs = None
        for attrs in acceptable_strategies:
            if all(col in df.columns for col in attrs):
                chosen_attrs = attrs
                break
                
        if chosen_attrs is None:
            raise ValueError(
                "Cannot construct a scientifically safe Athlete ID. "
                "Dataset must contain either 'AthleteID' or ['Name', 'Sex', 'WeightClassKg']."
            )
            
        stable_attributes = chosen_attrs
        logger.info(f"Using {stable_attributes} for athlete identity construction.")

    else:
        missing_cols = [col for col in stable_attributes if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns for athlete ID: {missing_cols}")
        logger.info(f"Using provided {stable_attributes} for athlete identity construction.")

    # Create the composite ID, replacing NaNs with 'Unknown'
    return df[stable_attributes].fillna('Unknown').astype(str).agg('_'.join, axis=1)
