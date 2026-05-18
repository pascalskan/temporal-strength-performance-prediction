import numpy as np
import pandas as pd
import pytest
from src.utils.metrics import compute_metrics, compute_basic_metrics
from src.contracts.metrics import EvaluationMetrics

@pytest.fixture
def sample_data():
    return pd.Series([1, 2, 3, 4, 5]), pd.Series([1.1, 2.2, 2.8, 4.3, 5.1])

def test_compute_metrics(sample_data):
    y_true, y_pred = sample_data
    metrics = compute_metrics(y_true, y_pred)
    assert isinstance(metrics, EvaluationMetrics)
    assert np.isclose(metrics.mae, 0.18)
    assert np.isclose(metrics.rmse, 0.204939)
    assert np.isclose(metrics.r2, 0.978)

def test_compute_basic_metrics(sample_data):
    y_true, y_pred = sample_data
    metrics = compute_basic_metrics(y_true, y_pred)
    assert isinstance(metrics, EvaluationMetrics)
    assert np.isclose(metrics.mae, 0.18)
    assert np.isclose(metrics.rmse, 0.204939)
    assert np.isclose(metrics.r2, 0.978)

def test_metrics_with_zeros():
    y_true = pd.Series([0, 0, 0])
    y_pred = pd.Series([0, 0, 0])
    metrics = compute_metrics(y_true, y_pred)
    assert metrics.mae == 0
    assert metrics.rmse == 0
    assert metrics.r2 == 1.0
