import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch

from src.evaluation.statistics import paired_bootstrap_comparison, run_all_comparisons


def build_predictions(observations, model_errors):
    """
    Assemble a prediction-level frame the way the walk-forward evaluator does.

    Crucially, observation_id is assigned ONCE per observation and reused across
    every model, because that is what makes the comparison paired. An earlier
    version of these fixtures derived observation_id from the row index after
    concatenating the models, which produced disjoint ids per model so nothing
    ever aligned.

    Args:
        observations: (athlete_id, meet_name, date, y_true) tuples.
        model_errors: {model_name: signed error added to y_true}.
    """
    base = pd.DataFrame(observations, columns=['athlete_id', 'MeetName', 'date', 'y_true'])
    base['date'] = pd.to_datetime(base['date'])
    base['observation_id'] = [
        f"{r.athlete_id}_{r.MeetName}_{r.date.strftime('%Y%m%d')}_{i}"
        for i, r in enumerate(base.itertuples())
    ]

    frames = []
    for model_name, error in model_errors.items():
        frame = base.copy()
        frame['model'] = model_name
        frame['y_pred'] = frame['y_true'] + error
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


OBSERVATIONS = [
    (1, 'meet_A', '2023-01-01', 10),
    (1, 'meet_B', '2023-02-01', 20),
    (1, 'meet_C', '2023-03-01', 30),
    (2, 'meet_A', '2023-01-01', 40),
    (2, 'meet_B', '2023-02-01', 50),
    (2, 'meet_C', '2023-03-01', 60),
]


@pytest.fixture
def sample_predictions():
    # model_a is uniformly 1kg out, model_b uniformly 2kg out,
    # so MAE(a) - MAE(b) = -1 exactly, with zero variance across athletes.
    return build_predictions(OBSERVATIONS, {'model_a': 1, 'model_b': 2})


def test_paired_bootstrap_comparison(sample_predictions):
    result = paired_bootstrap_comparison(
        sample_predictions, 'model_a', 'model_b', metric='mae', random_seed=42
    )

    assert set(result) == {'observed_difference', 'ci_lower', 'ci_upper', 'bootstrap_p'}
    assert np.isclose(result['observed_difference'], -1.0)
    # Every athlete has the same per-observation error, so every bootstrap
    # resample reproduces the same difference and the interval collapses.
    assert np.isclose(result['ci_lower'], -1.0)
    assert np.isclose(result['ci_upper'], -1.0)
    # All bootstrap differences are strictly negative, so the proportion at or
    # above zero is 0 and the two-sided p-value is 0.
    assert np.isclose(result['bootstrap_p'], 0.0)


def test_run_all_comparisons(sample_predictions):
    results = run_all_comparisons(
        sample_predictions, [('model_a', 'model_b')],
        metrics=['mae', 'rmse'], random_seed=42
    )

    assert len(results) == 2
    assert results['model_a'].iloc[0] == 'model_a'
    assert results['model_b'].iloc[0] == 'model_b'
    assert list(results['metric']) == ['mae', 'rmse']
    assert np.isclose(results['observed_difference'].iloc[0], -1.0)


def test_misordered_inputs(sample_predictions):
    """Row order must not affect the result; pairing is by observation_id."""
    shuffled = sample_predictions.sample(frac=1, random_state=42)
    result = paired_bootstrap_comparison(
        shuffled, 'model_a', 'model_b', metric='mae', random_seed=42
    )
    assert np.isclose(result['observed_difference'], -1.0)


def test_duplicate_predictions_error(sample_predictions):
    duplicated_row = sample_predictions[sample_predictions['model'] == 'model_a'].iloc[[0]]
    bad = pd.concat([sample_predictions, duplicated_row], ignore_index=True)

    with pytest.raises(ValueError, match="Duplicate predictions found for model model_a"):
        paired_bootstrap_comparison(bad, 'model_a', 'model_b')


def test_mismatched_y_true_error(sample_predictions):
    """The same observation must carry the same target under every model."""
    target = sample_predictions.loc[
        sample_predictions['model'] == 'model_b', 'observation_id'
    ].iloc[0]
    mask = (sample_predictions['model'] == 'model_b') & \
           (sample_predictions['observation_id'] == target)
    sample_predictions.loc[mask, 'y_true'] = 99

    with pytest.raises(ValueError, match="Aligned observations have mismatched true targets"):
        paired_bootstrap_comparison(sample_predictions, 'model_a', 'model_b')


def test_no_matching_predictions_error(sample_predictions):
    only_a = sample_predictions[sample_predictions['model'] == 'model_a']
    with pytest.raises(ValueError, match="No matching predictions found"):
        paired_bootstrap_comparison(only_a, 'model_a', 'model_b')


def test_unsupported_metric_error(sample_predictions):
    with pytest.raises(ValueError, match="Unsupported metric"):
        paired_bootstrap_comparison(sample_predictions, 'model_a', 'model_b', metric='r2')


def test_cluster_bootstrap_multiplicity_regression():
    """
    Resampling must preserve the multiplicity of sampled athletes.

    Guards against using isin() to gather resampled clusters, which would
    deduplicate an athlete drawn more than once and silently shrink the
    resample, understating the variance the cluster bootstrap exists to capture.
    """
    observations = [
        (1, 'meet_X', '2023-01-01', 10),
        (1, 'meet_Y', '2023-02-01', 20),
        (2, 'meet_X', '2023-01-01', 10),
        (2, 'meet_Y', '2023-02-01', 20),
    ]
    base = build_predictions(observations, {'model_a': 0, 'model_b': 0})
    # Athlete 1: model_a off by 1, model_b off by 2.
    # Athlete 2: model_a off by 2, model_b off by 3.
    errors = {('model_a', 1): 1, ('model_a', 2): 2, ('model_b', 1): 2, ('model_b', 2): 3}
    base['y_pred'] = base['y_true'] + [
        errors[(m, a)] for m, a in zip(base['model'], base['athlete_id'])
    ]

    class MockRNG:
        def choice(self, a, size, replace):
            # Force athlete 1 to be drawn twice and athlete 2 never.
            return np.array([1, 1])

    with patch('numpy.random.default_rng', return_value=MockRNG()):
        result = paired_bootstrap_comparison(
            base, 'model_a', 'model_b', metric='mae', random_seed=42
        )

    # Restricted to athlete 1: MAE(a)=1, MAE(b)=2, so the difference is -1.
    # Were multiplicity dropped, athlete 2 would leak in and shift this.
    assert np.isclose(result['observed_difference'], -1.0)
    assert np.isclose(result['ci_lower'], -1.0)
    assert np.isclose(result['ci_upper'], -1.0)
