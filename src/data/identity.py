import pandas as pd
from typing import List
from src.core.logging import get_logger

logger = get_logger(__name__)

# Ordered identity strategies, strongest first.
#
# Scientific rationale for stopping at ['Name', 'Sex']:
#
# OpenPowerlifting curates `Name` as a person-level identifier and already
# disambiguates distinct athletes who share a name with a '#N' suffix
# ("Jason Nguyen #1", "Tony Nguyen #12"). Name collisions are therefore
# resolved upstream, and adding further attributes cannot improve separation
# of distinct people -- it can only split a single career into fragments.
#
# Measured on the cleaned Raw cohort (601,835 records):
#
#   Name+Sex                          199,607 identities   369,267 usable rows
#   Name+Sex+Country                  201,966 identities   (Country 35.5% missing;
#                                                           splits 1,676 real athletes)
#   Name+Sex+WeightClassKg            264,914 identities   275,032 usable rows
#   Name+Sex+WeightClassKg+AgeClass   305,723 identities   216,681 usable rows
#
# ("usable rows" = records belonging to an athlete with >=4 competitions, the
# minimum to form Rolling_Mean_3 plus a forecast target.)
#
# Weight class and age class are not stable athlete attributes: lifters move
# between weight classes and cross age-class boundaries as their careers
# progress. Including them discards 41% of usable records, and does so
# non-randomly -- it preferentially fragments the long, progressing careers
# that carry the temporal signal this study measures. Country is unstable and
# frequently missing. Both are rejected on those grounds.
IDENTITY_STRATEGIES = [
    ["AthleteID"],
    ["Name", "Sex"],
]


def build_athlete_id(df: pd.DataFrame, stable_attributes: List[str] = None) -> pd.Series:
    """
    Build a longitudinal athlete identifier.

    Uses a deterministic AthleteID where the dataset provides one, otherwise a
    composite of attributes that are stable across an athlete's career. See
    IDENTITY_STRATEGIES for why the composite deliberately excludes weight
    class, age class and country.

    Args:
        df: Records to identify.
        stable_attributes: Explicit override. Bypasses strategy selection;
            the caller is responsible for the choice.

    Returns:
        Series of string identifiers aligned to df.index.

    Raises:
        ValueError: If no strategy can be satisfied, or if an explicit
            override names columns the DataFrame does not contain.
    """
    if stable_attributes is None:
        chosen_attrs = next(
            (attrs for attrs in IDENTITY_STRATEGIES
             if all(col in df.columns for col in attrs)),
            None,
        )

        if chosen_attrs is None:
            raise ValueError(
                "Cannot construct a scientifically safe Athlete ID. Dataset must "
                "contain either 'AthleteID' or both 'Name' and 'Sex'. "
                f"Available columns: {sorted(df.columns)}"
            )

        stable_attributes = chosen_attrs
        logger.info("Using %s for athlete identity construction.", stable_attributes)

    else:
        missing_cols = [col for col in stable_attributes if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns for athlete ID: {missing_cols}")
        logger.info("Using provided %s for athlete identity construction.", stable_attributes)

    return df[stable_attributes].fillna("Unknown").astype(str).agg("_".join, axis=1)
