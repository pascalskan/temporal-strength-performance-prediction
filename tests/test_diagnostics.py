import pandas as pd
import numpy as np
import pytest
from pathlib import Path

from src.evaluation.diagnostics import (
    compute_diagnostic_metrics,
    compute_decile_performance,
    run_and_save_diagnostics
)

@pytest.fixture
def sample_predictions():
    data = {
        'model': ['model_a'] * 100,
        'y_true': np.linspace(0, 100, 100),
        'y_pred': np.linspace(0, 100, 100) + np.random.normal(0, 5, 100)
    }
    return pd.DataFrame(data)

def test_diagnostic_metrics_perfect_model():
    y_true = np.array([10, 20, 30])
    y_pred = np.array([10, 20, 30])
    metrics = compute_diagnostic_metrics(y_true, y_pred)
    
    assert np.isclose(metrics['mean_residual'], 0)
    assert np.isclose(metrics['mean_percentage_bias'], 0)
    assert np.isclose(metrics['calibration_slope'], 1.0)
    assert np.isclose(metrics['calibration_intercept'], 0)
    assert np.isclose(metrics['residual_std_dev'], 0)

def test_diagnostic_metrics_biased_model():
    y_true = np.array([10, 20, 30])
    y_pred = np.array([12, 22, 32]) # Systematic +2 bias
    metrics = compute_diagnostic_metrics(y_true, y_pred)
    
    assert np.isclose(metrics['mean_residual'], 2.0)
    assert np.isclose(metrics['calibration_slope'], 1.0)
    assert np.isclose(metrics['calibration_intercept'], 2.0)

def test_decile_performance(sample_predictions):
    deciles = compute_decile_performance(sample_predictions['y_true'].values, sample_predictions['y_pred'].values)
    assert isinstance(deciles, pd.DataFrame)
    assert 'decile' in deciles.columns
    assert 'mae' in deciles.columns
    assert len(deciles) <= 10

def test_run_and_save_diagnostics(sample_predictions, tmp_path):
    output_dir = Path(tmp_path)
    run_and_save_diagnostics(sample_predictions, output_dir)
    
    # Check for output files
    assert (output_dir / "diagnostic_summary.csv").exists()
    assert (output_dir / "decile_performance.csv").exists()
    assert (output_dir / "calibration_plot_model_a.png").exists()
    assert (output_dir / "residual_plot_model_a.png").exists()
    assert (output_dir / "bias_by_decile.png").exists()
    
    # Check content of summary
    summary_df = pd.read_csv(output_dir / "diagnostic_summary.csv")
    assert len(summary_df) == 1
    assert summary_df['model'].iloc[0] == 'model_a'
