import pandas as pd
import pytest

from src.config.constants import EQUIPPED_EQUIPMENT, RAW_EQUIPMENT, VALID_EQUIPMENT
from src.data.cleaning import clean_data, split_equipment_cohorts


def _records(equipment_values):
    n = len(equipment_values)
    return pd.DataFrame({
        "Equipment": equipment_values,
        "TotalKg": [400.0] * n,
        "Age": [25.0] * n,
        "BodyweightKg": [80.0] * n,
        "Sex": ["M"] * n,
        "Name": [f"Athlete {i}" for i in range(n)],
    })


def test_valid_equipment_is_the_union_of_the_cohorts():
    """
    Structural invariant.

    The cleaning filter and the cohort definitions were previously independent
    literals in three files, and they disagreed: cleaning admitted {Raw, Wraps}
    while the entrypoints asked for {Wraps, Single-ply}. Single-ply was dropped
    before the cohort split ran, silently reducing 'equipped' to Wraps alone.
    Deriving the filter from the cohorts makes that divergence impossible, and
    this test pins the derivation.
    """
    assert VALID_EQUIPMENT == RAW_EQUIPMENT | EQUIPPED_EQUIPMENT
    assert not (RAW_EQUIPMENT & EQUIPPED_EQUIPMENT), "cohorts must be disjoint"


def test_single_ply_survives_cleaning():
    """The regression that made the equipped cohort ~11 forward predictions."""
    cleaned = clean_data(_records(["Single-ply"] * 3))
    assert len(cleaned) == 3


def test_cleaning_admits_every_cohort_member_and_nothing_else():
    df = _records(["Raw", "Single-ply", "Wraps", "Multi-ply", "Straps", "Unlimited"])
    cleaned = clean_data(df)

    assert set(cleaned["Equipment"]) == set(VALID_EQUIPMENT)
    assert "Multi-ply" not in set(cleaned["Equipment"])


def test_split_assigns_single_ply_to_the_equipped_cohort():
    df = _records(["Raw", "Raw", "Single-ply", "Single-ply", "Single-ply", "Wraps"])
    df_raw, df_equipped = split_equipment_cohorts(df)

    assert len(df_raw) == 2
    assert len(df_equipped) == 4
    assert set(df_equipped["Equipment"]) == {"Single-ply", "Wraps"}


def test_split_partitions_cleaned_data_without_loss_or_overlap():
    df = clean_data(_records(["Raw", "Single-ply", "Wraps", "Multi-ply"]))
    df_raw, df_equipped = split_equipment_cohorts(df)

    assert len(df_raw) + len(df_equipped) == len(df)
    assert set(df_raw.index).isdisjoint(df_equipped.index)


def test_split_returns_independent_copies():
    """Mutating a cohort must not write back into the cleaned frame."""
    df = _records(["Raw", "Single-ply"])
    df_raw, df_equipped = split_equipment_cohorts(df)

    df_raw.loc[df_raw.index[0], "TotalKg"] = 999.0
    assert df["TotalKg"].iloc[0] == 400.0


@pytest.mark.parametrize("equipment", sorted(EQUIPPED_EQUIPMENT))
def test_each_equipped_category_reaches_the_equipped_cohort(equipment):
    df = clean_data(_records([equipment] * 2))
    _, df_equipped = split_equipment_cohorts(df)
    assert len(df_equipped) == 2, f"{equipment} did not reach the equipped cohort"
