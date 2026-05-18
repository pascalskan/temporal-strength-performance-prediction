import pandas as pd
import pytest

from src.data.cleaning import clean_data


@pytest.fixture
def raw_data():
    data = {
        "Event": ["A", "B", "C", "D"],
        "Equipment": ["Raw", "Single-ply", "Wraps", "Multi-ply"],
        "TotalKg": [100, 200, None, 400],
        "Tested": ["Yes", "Yes", "No", "Yes"],
        "Age": [25, 30, 28, 35],
        "BodyweightKg": [80, 95, 90, 110],
        "Sex": ["M", "M", "F", "M"],
    }
    return pd.DataFrame(data)


def test_clean_data(raw_data):
    cleaned_df = clean_data(raw_data)

    assert "Multi-ply" not in cleaned_df["Equipment"].values
    assert "Single-ply" not in cleaned_df["Equipment"].values

    assert not cleaned_df["TotalKg"].isnull().any()

    assert "Raw" in cleaned_df["Equipment"].values

    assert len(cleaned_df) == 1

    expected_columns = [
        "Event",
        "Equipment",
        "TotalKg",
        "Tested",
        "Age",
        "BodyweightKg",
        "Sex",
    ]

    assert list(cleaned_df.columns) == expected_columns