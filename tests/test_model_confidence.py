import pandas as pd
import numpy as np
import pytest

from src.evaluation.statistics import compute_model_confidence_intervals

@pytest.fixture
def sample_predictions():
    # Two models, two athletes, two predictions each
    data = {
        'model': ['model_a'] * 4 + ['model_b'] * 4,
        'athlete_id': [1, 1, 2, 2] * 2,
        'y_true': [10, 20, 30, 40] * 2,
        'y_pred': [11, 21, 31, 41, 12, 22, 32, 42]
    }
    return pd.DataFrame(data)

def test_ci_output_schema(sample_predictions):
    results = compute_model_confidence_intervals(sample_predictions, n_bootstraps=10, random_seed=42)
    
    assert isinstance(results, pd.DataFrame)
    expected_cols = ['model', 'metric', 'point_estimate', 'ci_lower', 'ci_upper', 'std_error']
    assert all(col in results.columns for col in expected_cols)
    
    # 5 metrics for 2 models
    assert len(results) == 10 

def test_ci_deterministic_reproducibility(sample_predictions):
    results1 = compute_model_confidence_intervals(sample_predictions, n_bootstraps=50, random_seed=42)
    results2 = compute_model_confidence_intervals(sample_predictions, n_bootstraps=50, random_seed=42)
    pd.testing.assert_frame_equal(results1, results2)

def test_point_estimate_within_ci(sample_predictions):
    results = compute_model_confidence_intervals(sample_predictions, n_bootstraps=100, random_seed=42)
    
    for _, row in results.iterrows():
        if pd.notna(row['ci_lower']) and pd.notna(row['ci_upper']):
            assert row['ci_lower'] <= row['point_estimate'] <= row['ci_upper']

def test_ci_synthetic_sanity_check():
    # A model with zero error should have zero width CI
    data = {
        'model': ['perfect_model'] * 10,
        'athlete_id': list(range(10)),
        'y_true': np.arange(10),
        'y_pred': np.arange(10)
    }
    df = pd.DataFrame(data)
    
    results = compute_model_confidence_intervals(df, n_bootstraps=100, random_seed=42)
    
    for _, row in results.iterrows():
        if row['metric'] in ['mae', 'rmse', 'median_absolute_error', 'mean_residual']:
            assert np.isclose(row['point_estimate'], 0)
            assert np.isclose(row['ci_lower'], 0)
            assert np.isclose(row['ci_upper'], 0)
            assert np.isclose(row['std_error'], 0)
        elif row['metric'] == 'r2':
            assert np.isclose(row['point_estimate'], 1.0)
            assert np.isclose(row['ci_lower'], 1.0)
            assert np.isclose(row['ci_upper'], 1.0)
            assert np.isclose(row['std_error'], 0)

def test_ci_handles_no_athlete_id(sample_predictions):
    # Test that it runs without athlete_id (standard bootstrap)
    df = sample_predictions.drop(columns=['athlete_id'])
    results = compute_model_confidence_intervals(df, n_bootstraps=10, random_seed=42)
    assert len(results) > 0
    assert 'model' in results.columns
