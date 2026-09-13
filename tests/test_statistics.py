import pandas as pd
import numpy as np
import pytest

from src.evaluation.statistics import paired_bootstrap_comparison, run_all_comparisons


@pytest.fixture
def sample_predictions():
    data = {
        'athlete_id': [1, 1, 1, 2, 2, 2] * 2,
        'date': pd.to_datetime(['2023-01-01', '2023-02-01', '2023-03-01'] * 4),
        'MeetID': ['meet_A', 'meet_B', 'meet_C'] * 4,
        'forecast_window': [1, 2, 3] * 4,
        'model': ['model_a'] * 6 + ['model_b'] * 6,
        'y_true': [10, 20, 30, 40, 50, 60] * 2,
        'y_pred': [11, 21, 31, 41, 51, 61, 12, 22, 32, 42, 52, 62]
    }
    df = pd.DataFrame(data)
    # Generate observation_id using the new stable logic
    df['observation_id'] = df.apply(
        lambda row: f"{row['athlete_id']}_{row['MeetID']}_{row['date'].strftime('%Y%m%d')}_{row.name}",
        axis=1
    )
    return df


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
    # For this synthetic data, model_b is always worse, so all bootstrap diffs are < 0.
    # This means mean(diffs >= 0) is 0, so p-value is 2 * 0 = 0.
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


def test_misordered_inputs(sample_predictions):
    shuffled_predictions = sample_predictions.sample(frac=1, random_state=42)
    result = paired_bootstrap_comparison(
        shuffled_predictions,
        'model_a',
        'model_b',
        metric='mae',
        random_seed=42
    )
    assert np.isclose(result['observed_difference'], -1.0)


def test_duplicate_predictions_error(sample_predictions):
    dup_data = {
        'athlete_id': [1],
        'date': [pd.to_datetime('2023-01-01')],
        'MeetID': ['meet_A'],
        'forecast_window': [1],
        'model': ['model_a'],
        'y_true': [10],
        'y_pred': [11]
    }
    dup_df = pd.DataFrame(dup_data)
    
    # Use 0 as original index to force duplicate
    def build_dup_id(row):
        return f"{row['athlete_id']}_{row['MeetID']}_{row['date'].strftime('%Y%m%d')}_0"
        
    dup_df['observation_id'] = dup_df.apply(build_dup_id, axis=1)
    
    bad_predictions = pd.concat([sample_predictions, dup_df], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate predictions found for model model_a"):
        paired_bootstrap_comparison(bad_predictions, 'model_a', 'model_b')


def test_mismatched_y_true_error(sample_predictions):
    # Find an observation_id for model_b to modify
    target_mask = (sample_predictions['model'] == 'model_b') & (sample_predictions['athlete_id'] == 1)
    target_obs_id = sample_predictions[target_mask].iloc[0]['observation_id']
    
    mask = (sample_predictions['model'] == 'model_b') & (sample_predictions['observation_id'] == target_obs_id)
    sample_predictions.loc[mask, 'y_true'] = 99

    with pytest.raises(ValueError, match="Aligned observations have mismatched true targets"):
        paired_bootstrap_comparison(sample_predictions, 'model_a', 'model_b')


def test_no_matching_predictions_error(sample_predictions):
    no_match_predictions = sample_predictions[sample_predictions['model'] == 'model_a']
    with pytest.raises(ValueError, match="No matching predictions found"):
        paired_bootstrap_comparison(no_match_predictions, 'model_a', 'model_b')


def test_cluster_bootstrap_multiplicity_regression():
    """
    This is a regression test to ensure that the bootstrap resampling
    correctly preserves the multiplicity of sampled clusters (athletes).
    It specifically guards against the misuse of `isin()`, which would
    incorrectly deduplicate the resampled data.
    """
    data = {
        'athlete_id': [1, 1, 2, 2, 1, 1, 2, 2],
        'date': pd.to_datetime(['2023-01-01', '2023-02-01'] * 4),
        'MeetID': ['meet_X', 'meet_Y'] * 4,
        'forecast_window': [1, 2] * 4,
        'model': ['model_a'] * 4 + ['model_b'] * 4,
        'y_true': [10, 20] * 4,
        'y_pred': [11, 21, 12, 22, 12, 22, 13, 23]
    }
    df = pd.DataFrame(data)
    df['observation_id'] = df.apply(
        lambda row: f"{row['athlete_id']}_{row['MeetID']}_{row['date'].strftime('%Y%m%d')}_{row.name}",
        axis=1
    )
    
    # Mock the rng.choice to return a specific sequence
    class MockRNG:
        def choice(self, a, size, replace):
            # Force athlete 1 to be sampled twice
            return np.array([1, 1])

    # We need to monkeypatch the np.random.default_rng to return our mock
    from unittest.mock import patch
    with patch('numpy.random.default_rng') as mock_rng_constructor:
        mock_rng_constructor.return_value = MockRNG()
        
        result = paired_bootstrap_comparison(df, 'model_a', 'model_b', metric='mae', random_seed=42)
        
        # The original df has 4 rows per model. If athlete 1 (2 rows per model) is sampled twice,
        # the resampled df should have 4 rows.
        # The MAE for athlete 1 is |10-11| + |20-21| / 2 = 1 for model a
        # and |10-12| + |20-22| / 2 = 2 for model b. Diff = -1.
        # Since we sample athlete 1 twice, the bootstrap diff should be -1.
        assert np.isclose(result['observed_difference'], -1.0)
        assert np.isclose(result['ci_lower'], -1.0)
        assert np.isclose(result['ci_upper'], -1.0)
