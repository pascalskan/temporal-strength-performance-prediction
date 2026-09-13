import pandas as pd
import numpy as np
import pytest
from pathlib import Path

from src.evaluation.temporal_analysis import (
    compute_temporal_metrics,
    estimate_temporal_drift,
    run_and_save_temporal_analysis
)

@pytest.fixture
def synthetic_temporal_data():
    # 5 forecast windows (years)
    years = [2018, 2019, 2020, 2021, 2022]
    data = []
    
    for i, year in enumerate(years):
        # Stable model: constant MAE of ~2.0
        y_true = np.linspace(10, 50, 20)
        y_pred_stable = y_true + np.random.normal(0, 2, 20)
        
        # Degrading model: MAE increases by ~1.0 each year
        noise_level = 2 + (i * 1.0)
        y_pred_degrading = y_true + np.random.normal(0, noise_level, 20)
        
        for yt, yp_s, yp_d in zip(y_true, y_pred_stable, y_pred_degrading):
            data.append({
                'model': 'stable_model',
                'forecast_window': year,
                'y_true': yt,
                'y_pred': yp_s
            })
            data.append({
                'model': 'degrading_model',
                'forecast_window': year,
                'y_true': yt,
                'y_pred': yp_d
            })
            
    return pd.DataFrame(data)

def test_compute_temporal_metrics_schema(synthetic_temporal_data):
    metrics_df = compute_temporal_metrics(synthetic_temporal_data)
    
    assert isinstance(metrics_df, pd.DataFrame)
    expected_cols = ['model', 'forecast_window', 'prediction_count', 'MAE', 'RMSE', 'R2', 'sMAPE', 'NRMSE', 'mean_residual']
    assert all(col in metrics_df.columns for col in expected_cols)
    
    # 2 models * 5 windows = 10 rows
    assert len(metrics_df) == 10

def test_estimate_temporal_drift_logic(synthetic_temporal_data):
    # Set seed for reproducible synthetic generation in fixture
    np.random.seed(42)
    
    metrics_df = compute_temporal_metrics(synthetic_temporal_data)
    drift_df = estimate_temporal_drift(metrics_df)
    
    assert len(drift_df) == 2
    
    stable_drift = drift_df[drift_df['model'] == 'stable_model'].iloc[0]
    degrading_drift = drift_df[drift_df['model'] == 'degrading_model'].iloc[0]
    
    # Stable model should have near-zero trend slope
    assert abs(stable_drift['MAE_slope']) < 0.5
    
    # Degrading model should have positive MAE slope (error increasing)
    assert degrading_drift['MAE_slope'] > 0.5

def test_run_and_save_temporal_analysis(synthetic_temporal_data, tmp_path):
    output_dir = Path(tmp_path)
    run_and_save_temporal_analysis(synthetic_temporal_data, output_dir)
    
    # Check if files were created
    assert (output_dir / "temporal_metrics.csv").exists()
    assert (output_dir / "temporal_drift_summary.csv").exists()
    assert (output_dir / "temporal_mae.png").exists()
    assert (output_dir / "temporal_rmse.png").exists()
    assert (output_dir / "temporal_residual.png").exists()
