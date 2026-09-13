"""
Every pipeline must group athlete history by the shared constructed identity.

feature_engineering previously defaulted athlete_col to "AthleteID", a column
no part of this project creates. The walk-forward pipeline passed it
explicitly and worked; the retrospective and forward pipelines relied on the
default and raised at runtime. Because only walk-forward was being run, both
stayed broken through the entire scientific-evaluation refactor.
"""
import inspect

import pandas as pd
import pytest

from src.features.temporal import feature_engineering
from src.pipelines.forward_pipeline import create_forward_target


def test_feature_engineering_requires_an_explicit_athlete_column():
    """No default: a wrong grouping column must be impossible to reach silently."""
    signature = inspect.signature(feature_engineering)
    athlete_col = signature.parameters["athlete_col"]
    assert athlete_col.default is inspect.Parameter.empty, (
        "athlete_col must stay required; a default let two pipelines "
        "silently group by a non-existent column"
    )


def test_feature_engineering_rejects_a_missing_column():
    df = pd.DataFrame({
        "Name": ["A"], "Date": pd.to_datetime(["2020-01-01"]), "TotalKg": [400.0],
    })
    with pytest.raises(ValueError, match="Missing required athlete identifier column"):
        feature_engineering(df, athlete_col="Athlete_ID")


@pytest.mark.parametrize("module_name", [
    "src.pipelines.data_preparation",
    "src.pipelines.forward_pipeline",
    "src.pipelines.walk_forward_pipeline",
])
def test_every_pipeline_builds_the_shared_identity(module_name):
    """
    Each pipeline must construct the identity via build_athlete_id rather than
    inventing its own grouping key.
    """
    import importlib
    source = inspect.getsource(importlib.import_module(module_name))
    assert "build_athlete_id" in source or "Athlete_ID" in source, (
        f"{module_name} does not reference the shared athlete identity"
    )


def test_no_pipeline_groups_by_name_alone():
    """
    Name-only grouping bypasses the identity module, so it would not pick up
    the strategy hierarchy or its fallbacks.
    """
    import importlib
    for module_name in ("src.pipelines.data_preparation",
                        "src.pipelines.forward_pipeline",
                        "src.pipelines.walk_forward_pipeline"):
        source = inspect.getsource(importlib.import_module(module_name))
        assert 'groupby("Name")' not in source, f"{module_name} groups by Name alone"


def test_forward_target_groups_by_athlete_identity():
    """Two different athletes must not have their totals shifted into each other."""
    df = pd.DataFrame({
        "Athlete_ID": ["A_M", "A_M", "B_F"],
        "Date": pd.to_datetime(["2020-01-01", "2021-01-01", "2020-06-01"]),
        "TotalKg": [400.0, 420.0, 300.0],
    })
    out = create_forward_target(df)

    # B_F has no later competition, so it yields no target and is dropped.
    assert len(out) == 1
    assert out.iloc[0]["Athlete_ID"] == "A_M"
    assert out.iloc[0]["Target_Total"] == 420.0
