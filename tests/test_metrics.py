import numpy as np
import pandas as pd
import pytest

from src.contracts.metrics import EvaluationMetrics
from src.utils.metrics import compute_metrics
from src.evaluation.metrics import (
    compute_global_forecast_metrics,
    symmetric_mean_absolute_percentage_error,
    normalized_root_mean_squared_error
)


@pytest.fixture
def sample_data():
    return (
        pd.Series([1, 2, 3, 4, 5]),
        pd.Series([1.1, 2.2, 2.8, 4.3, 5.1]),
    )


def test_compute_metrics_returns_contract(sample_data):
    y_true, y_pred = sample_data

    metrics = compute_metrics(y_true, y_pred)

    assert isinstance(metrics, EvaluationMetrics)
    assert np.isclose(metrics.mae, 0.18)
    assert np.isclose(metrics.rmse, 0.19493588689617924)
    assert np.isclose(metrics.r2, 0.981)


def test_metrics_with_zero_error():
    y_true = pd.Series([0, 0, 0])
    y_pred = pd.Series([0, 0, 0])

    metrics = compute_metrics(y_true, y_pred)

    assert metrics.mae == 0
    assert metrics.rmse == 0
    assert metrics.r2 == 1.0


def test_compute_global_forecast_metrics():
    # Create a dummy predictions dataframe
    data = {
        "model": ["model_a", "model_a", "model_b", "model_b", "model_a"],
        "y_true": [10, 20, 30, 40, 50],
        "y_pred": [12, 22, 28, 42, 55],
    }
    predictions_df = pd.DataFrame(data)

    # Expected metrics for model_a
    y_true_a = np.array([10, 20, 50])
    y_pred_a = np.array([12, 22, 55])
    mae_a = np.mean(np.abs(y_true_a - y_pred_a))
    rmse_a = np.sqrt(np.mean((y_true_a - y_pred_a)**2))
    r2_a = 1 - (np.sum((y_true_a - y_pred_a)**2) / np.sum((y_true_a - np.mean(y_true_a))**2))
    median_ae_a = np.median(np.abs(y_true_a - y_pred_a))
    mean_res_a = np.mean(y_pred_a - y_true_a)

    # Expected metrics for model_b
    y_true_b = np.array([30, 40])
    y_pred_b = np.array([28, 42])
    mae_b = np.mean(np.abs(y_true_b - y_pred_b))
    rmse_b = np.sqrt(np.mean((y_true_b - y_pred_b)**2))
    r2_b = 1 - (np.sum((y_true_b - y_pred_b)**2) / np.sum((y_true_b - np.mean(y_true_b))**2))
    median_ae_b = np.median(np.abs(y_true_b - y_pred_b))
    mean_res_b = np.mean(y_pred_b - y_true_b)

    global_metrics = compute_global_forecast_metrics(predictions_df)

    assert len(global_metrics) == 2
    
    metrics_a = global_metrics[global_metrics["model"] == "model_a"].iloc[0]
    assert metrics_a["prediction_count"] == 3
    assert np.isclose(metrics_a["MAE"], mae_a)
    assert np.isclose(metrics_a["RMSE"], rmse_a)
    assert np.isclose(metrics_a["R2"], r2_a)
    assert np.isclose(metrics_a["median_absolute_error"], median_ae_a)
    assert np.isclose(metrics_a["mean_residual"], mean_res_a)
    assert 'sMAPE' in metrics_a
    assert 'NRMSE' in metrics_a

    metrics_b = global_metrics[global_metrics["model"] == "model_b"].iloc[0]
    assert metrics_b["prediction_count"] == 2
    assert np.isclose(metrics_b["MAE"], mae_b)
    assert np.isclose(metrics_b["RMSE"], rmse_b)
    assert np.isclose(metrics_b["R2"], r2_b)
    assert np.isclose(metrics_b["median_absolute_error"], median_ae_b)
    assert np.isclose(metrics_b["mean_residual"], mean_res_b)


def test_global_metrics_unequal_folds():
    # Fold 1 (small)
    fold1 = pd.DataFrame({
        "model": ["m1"] * 2,
        "y_true": [100, 100],
        "y_pred": [101, 101],  # MAE = 1
    })
    # Fold 2 (large)
    fold2 = pd.DataFrame({
        "model": ["m1"] * 10,
        "y_true": [100] * 10,
        "y_pred": [110] * 10,  # MAE = 10
    })
    
    predictions_df = pd.concat([fold1, fold2], ignore_index=True)
    
    # Invalid average of fold MAEs would be (1 + 10) / 2 = 5.5
    # Correct pooled MAE is ((2*1) + (10*10)) / 12 = 102 / 12 = 8.5
    
    global_metrics = compute_global_forecast_metrics(predictions_df)
    
    assert len(global_metrics) == 1
    assert np.isclose(global_metrics.iloc[0]["MAE"], 8.5)


def test_smape_calculation():
    y_true = np.array([100, 50, 0, 10])
    y_pred = np.array([110, 40, 0, 0])
    
    # obs 1: 200 * 10 / 210 = 9.523...
    # obs 2: 200 * 10 / 90 = 22.222...
    # obs 3: 0/0 -> should be excluded or handled
    # obs 4: 200 * 10 / 10 = 200
    
    # denominator mask: [True, True, False, True]
    # valid diffs: [10, 10, 10]
    # valid denoms: [210, 90, 10]
    # mean: (9.523809523809524 + 22.22222222222222 + 200) / 3 = 77.24867724867724
    
    expected_smape = 77.24867724867724
    actual_smape = symmetric_mean_absolute_percentage_error(y_true, y_pred)
    assert np.isclose(actual_smape, expected_smape)


def test_smape_zero_safe():
    y_true = np.array([0, 0, 0])
    y_pred = np.array([0, 0, 0])
    assert symmetric_mean_absolute_percentage_error(y_true, y_pred) == 0.0


def test_nrmse_calculation():
    y_true = np.array([10, -10, 20, -20])
    y_pred = np.array([12, -8, 15, -25])
    
    # errors: [2, 2, -5, -5]
    # sq errors: [4, 4, 25, 25]
    # mse: 58 / 4 = 14.5
    # rmse: sqrt(14.5) = 3.80788...
    # mean_abs_true: (10 + 10 + 20 + 20) / 4 = 15
    # nrmse: sqrt(14.5) / 15 = 0.2538...
    
    expected_nrmse = np.sqrt(14.5) / 15
    actual_nrmse = normalized_root_mean_squared_error(y_true, y_pred)
    assert np.isclose(actual_nrmse, expected_nrmse)


def test_nrmse_zero_safe():
    y_true = np.array([0, 0])
    y_pred = np.array([1, 1])
    assert np.isnan(normalized_root_mean_squared_error(y_true, y_pred))
