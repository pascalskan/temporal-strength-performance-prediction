import pandas as pd
import numpy as np
import pytest

from src.evaluation.statistics import paired_bootstrap_comparison, run_all_comparisons

@pytest.fixture
def sample_predictions():
    data = {
        'athlete_id': [1, 1, 1, 2, 2, 2] * 2,
        'date': ['2023-01-01'] * 12,
        'forecast_window': [1] * 12,
        'model': ['model_a'] * 6 + ['model_b'] * 6,
        'y_true': [10, 20, 30, 40, 50, 60] * 2,
        'y_pred': [11, 21, 31, 41, 51, 61, 12, 22, 32, 42, 52, 62]
    }
    return pd.DataFrame(data)

def test_paired_bootstrap_comparison(sample_predictions):
    result = paired_bootstrap_comparison(
        sample_predictions,
        'model_a',
        'model_b',
        metric='mae',
        random_seed=42
    )

    assert 'observed_difference' in result
    assert 'ci_lower' in result
    assert 'ci_upper' in result
    assert 'bootstrap_p' in result

    # With a deterministic seed, the results should be reproducible
    assert np.isclose(result['observed_difference'], -1.0)
    assert np.isclose(result['ci_lower'], -1.0)
    assert np.isclose(result['ci_upper'], -1.0)
    assert np.isclose(result['bootstrap_p'], 0.0)

def test_run_all_comparisons(sample_predictions):
    comparisons = [('model_a', 'model_b')]
    results = run_all_comparisons(
        sample_predictions,
        comparisons,
        metrics=['mae', 'rmse'],
        random_seed=42
    )

    assert len(results) == 2
    assert results['model_a'].iloc[0] == 'model_a'
    assert results['model_b'].iloc[0] == 'model_b'
    assert results['metric'].iloc[0] == 'mae'
    assert np.isclose(results['observed_difference'].iloc[0], -1.0)
