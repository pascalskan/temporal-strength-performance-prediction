"""
Resumable walk-forward runs.

The evaluator held every prediction in memory and wrote only after the final
fold, so any interruption during a run of tens of minutes to hours discarded
all of it. These assert that completed folds survive an interruption and that a
resumed run does not corrupt the result by splicing together incompatible work.
"""
import json

import pandas as pd
import pytest

from src.evaluation.checkpointing import (
    CHECKPOINT_DIRNAME,
    WalkForwardCheckpoint,
    build_fingerprint,
)

FINGERPRINT = build_fingerprint(
    feature_cols=["Prev_Total", "Age"],
    model_names=["Ridge", "Persistence"],
    granularity="year",
    min_train_periods=5,
)


def predictions_for(window, n=3):
    return [
        {"observation_id": f"{window}_o{i}", "forecast_window": window,
         "model": "Ridge", "y_true": 400.0 + i, "y_pred": 405.0 + i}
        for i in range(n)
    ]


@pytest.fixture
def checkpoint(tmp_path):
    return WalkForwardCheckpoint(tmp_path, FINGERPRINT)


def test_nothing_completed_before_anything_is_written(checkpoint):
    assert checkpoint.completed_windows() == set()
    assert checkpoint.load_predictions() == []


def test_a_recorded_window_is_reported_complete(checkpoint):
    checkpoint.record_window(2015, predictions_for(2015))
    assert checkpoint.completed_windows() == {2015}


def test_predictions_survive_and_accumulate_across_windows(checkpoint):
    checkpoint.record_window(2015, predictions_for(2015))
    checkpoint.record_window(2016, predictions_for(2016))

    restored = checkpoint.load_predictions()
    assert len(restored) == 6
    assert {r["forecast_window"] for r in restored} == {2015, 2016}


def test_resuming_reads_back_what_a_previous_process_wrote(tmp_path):
    """The actual recovery path: a fresh object, as a restarted run would have."""
    first = WalkForwardCheckpoint(tmp_path, FINGERPRINT)
    first.record_window(2015, predictions_for(2015))

    resumed = WalkForwardCheckpoint(tmp_path, FINGERPRINT)
    assert resumed.completed_windows() == {2015}
    assert len(resumed.load_predictions()) == 3


def test_checkpoint_from_a_different_configuration_is_refused(tmp_path):
    """
    Resuming across a configuration change would combine folds computed under
    different experimental conditions into a single result set.
    """
    original = WalkForwardCheckpoint(tmp_path, FINGERPRINT)
    original.record_window(2015, predictions_for(2015))

    different = WalkForwardCheckpoint(
        tmp_path,
        build_fingerprint(
            feature_cols=["Prev_Total"],      # one fewer feature
            model_names=["Ridge", "Persistence"],
            granularity="year",
            min_train_periods=5,
        ),
    )

    assert different.completed_windows() == set()
    assert different.load_predictions() == []


def test_fingerprint_responds_to_every_configuration_axis():
    base = dict(feature_cols=["a"], model_names=["m"], granularity="year",
                min_train_periods=5)
    variations = [
        {**base, "feature_cols": ["a", "b"]},
        {**base, "model_names": ["m", "n"]},
        {**base, "granularity": "month"},
        {**base, "min_train_periods": 6},
    ]
    for variation in variations:
        assert build_fingerprint(**variation) != build_fingerprint(**base)


def test_fingerprint_ignores_ordering():
    """Model dict ordering is incidental and must not invalidate a checkpoint."""
    a = build_fingerprint(["x", "y"], ["m", "n"], "year", 5)
    b = build_fingerprint(["y", "x"], ["n", "m"], "year", 5)
    assert a == b


def test_clearing_removes_the_working_files(checkpoint, tmp_path):
    checkpoint.record_window(2015, predictions_for(2015))
    assert (tmp_path / CHECKPOINT_DIRNAME).exists()

    checkpoint.clear()

    assert not (tmp_path / CHECKPOINT_DIRNAME).exists()
    assert checkpoint.completed_windows() == set()


def test_clearing_an_absent_checkpoint_does_not_raise(checkpoint):
    checkpoint.clear()


def test_disabled_checkpoint_writes_nothing(tmp_path):
    disabled = WalkForwardCheckpoint(tmp_path, FINGERPRINT, enabled=False)
    disabled.record_window(2015, predictions_for(2015))

    assert not (tmp_path / CHECKPOINT_DIRNAME).exists()
    assert disabled.completed_windows() == set()


def test_empty_window_is_not_recorded_as_complete(checkpoint):
    """A fold that produced nothing must be retried, not skipped on resume."""
    checkpoint.record_window(2015, [])
    assert checkpoint.completed_windows() == set()


def test_state_file_records_the_fingerprint(checkpoint, tmp_path):
    checkpoint.record_window(2015, predictions_for(2015))
    state = json.loads(
        (tmp_path / "walk_forward_checkpoint_state.json").read_text(encoding="utf-8")
    )
    assert state["fingerprint"] == FINGERPRINT
    assert state["completed_windows"] == [2015]


def test_round_trip_preserves_dtypes_and_float_bits(checkpoint):
    """
    The property the format exists for. A CSV round-trip widened int32 to
    int64, stringified timestamps and lost the last bits of a float, making a
    resumed run disagree with an uninterrupted one by up to 1e-13.
    """
    original = pd.DataFrame({
        "observation_id": ["o1", "o2"],
        "forecast_window": [2015, 2015],
        "date": pd.to_datetime(["2015-03-01", "2015-09-01"]),
        # Values whose decimal expansions do not terminate, so any loss of the
        # trailing bits shows up immediately.
        "y_pred": [239.16666666666666, 1.0 / 3.0],
        "model": ["Ridge", "Ridge"],
    })
    checkpoint.record_window(2015, original.to_dict("records"))

    restored = pd.DataFrame(checkpoint.load_predictions())[original.columns]

    pd.testing.assert_frame_equal(original, restored)
    # Exact to the last bit, not merely close.
    assert (restored["y_pred"] == original["y_pred"]).all()
    assert restored["date"].dtype == original["date"].dtype
