import pandas as pd
import pytest

from src.utils.splitting import time_aware_split


@pytest.fixture
def synthetic_data():
    dates = pd.to_datetime(
        [
            "2020-01-01",
            "2020-02-01",
            "2020-03-01",
            "2020-04-01",
            "2020-05-01",
            "2020-06-01",
            "2020-07-01",
            "2020-08-01",
            "2020-09-01",
            "2020-10-01",
        ]
    )

    return pd.DataFrame(
        {
            "Date": dates,
            "Feature1": range(10),
            "Target": range(10, 20),
        }
    )


def test_time_aware_split_leakage(synthetic_data):
    X_train, X_test, y_train, y_test, split_point = time_aware_split(
        synthetic_data,
        feature_cols=["Feature1"],
        target_col="Target",
        split_ratio=0.7,
    )

    assert len(X_train) == 7
    assert len(X_test) == 3
    assert len(y_train) == 7
    assert len(y_test) == 3

    train_dates = synthetic_data[synthetic_data["Date"] <= split_point]["Date"]
    test_dates = synthetic_data[synthetic_data["Date"] > split_point]["Date"]

    assert train_dates.max() < test_dates.min()