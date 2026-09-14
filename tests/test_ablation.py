"""Feature ablation: separating information gain from algorithm choice."""
import pandas as pd
import pytest

from src.config.features import FORECAST_FEATURES
from src.evaluation.ablation import (
    REFERENCE_MODELS,
    build_feature_sets,
    run_ablation,
    summarise_information_vs_algorithm,
)


def test_feature_sets_are_nested_from_smallest_to_largest():
    """
    Each step must add information without removing any, or a change in error
    could not be attributed to the addition.
    """
    sets = build_feature_sets()
    order = ["demographics", "last_result", "recent_form", "full"]

    for smaller, larger in zip(order, order[1:]):
        assert set(sets[smaller]) <= set(sets[larger]), (
            f"{smaller} is not a subset of {larger}"
        )


def test_feature_sets_increase_in_size():
    sets = build_feature_sets()
    sizes = [len(sets[name]) for name in ("demographics", "last_result", "recent_form", "full")]
    assert sizes == sorted(sizes)
    assert len(set(sizes)) > 1, "ablation needs configurations of differing size"


def test_no_feature_set_reintroduces_an_excluded_feature():
    """
    Features dropped from FORECAST_FEATURES were dropped on leakage grounds.
    Ablation must not smuggle them back in.
    """
    for name, features in build_feature_sets().items():
        unexpected = set(features) - set(FORECAST_FEATURES)
        assert not unexpected, f"{name} contains non-forecasting features: {unexpected}"


def test_demographics_carries_no_performance_history():
    """The point of this configuration is to have no history at all."""
    demographics = build_feature_sets()["demographics"]
    for feature in demographics:
        assert not feature.startswith(("Prev_", "Rolling_", "Peak_")), feature
    assert "Momentum" not in demographics


def make_metrics():
    """Information helps a lot; algorithm choice helps a little."""
    rows = []
    for feature_set, n_features, learned_mae in (
        ("demographics", 3, 130.0),
        ("last_result", 4, 95.0),
        ("full", 17, 92.0),
    ):
        for offset, model in enumerate(("Random Forest", "Ridge")):
            rows.append({
                "feature_set": feature_set, "n_features": n_features,
                "model": model, "MAE": learned_mae + 5 * offset,
                "RMSE": 1.0, "R2": 0.5, "prediction_count": 1000,
            })
        # An invariant reference line, identical under every configuration.
        rows.append({
            "feature_set": feature_set, "n_features": n_features,
            "model": "Rolling Mean", "MAE": 100.0,
            "RMSE": 1.0, "R2": 0.5, "prediction_count": 1000,
        })
    return pd.DataFrame(rows)


def test_reference_models_are_excluded_from_best_model():
    """
    Regression: including invariant baselines made the same reference the
    winner under every configuration, so every information gain read as zero
    and the ablation measured nothing.
    """
    summary = summarise_information_vs_algorithm(make_metrics())

    assert "Rolling Mean" not in set(summary["best_model"])
    assert (summary["gain_over_demographics"] > 0).any()


def test_information_gain_is_measured_against_demographics():
    summary = summarise_information_vs_algorithm(make_metrics()).set_index("feature_set")

    assert summary.loc["demographics", "gain_over_demographics"] == pytest.approx(0.0)
    assert summary.loc["last_result", "gain_over_demographics"] == pytest.approx(35.0)
    assert summary.loc["full", "gain_over_demographics"] == pytest.approx(38.0)


def test_algorithm_spread_is_reported_separately_from_information_gain():
    """The comparison the study turns on: which decision matters more."""
    summary = summarise_information_vs_algorithm(make_metrics()).set_index("feature_set")

    assert summary.loc["last_result", "spread_across_models"] == pytest.approx(5.0)
    assert (
        summary.loc["last_result", "gain_over_demographics"]
        > summary.loc["last_result", "spread_across_models"]
    )


def test_reference_line_is_retained_for_scale():
    summary = summarise_information_vs_algorithm(make_metrics()).set_index("feature_set")

    assert summary.loc["demographics", "best_reference_MAE"] == pytest.approx(100.0)
    # Demographics-only is worse than simply carrying recent form forward.
    assert summary.loc["demographics", "learned_advantage_over_reference"] < 0
    # With the previous total, the learned models overtake it.
    assert summary.loc["last_result", "learned_advantage_over_reference"] > 0


def test_empty_metrics_summarise_without_raising():
    assert summarise_information_vs_algorithm(pd.DataFrame()).empty


def test_metrics_of_only_reference_models_summarise_without_raising():
    only_references = make_metrics()
    only_references = only_references[only_references["model"].isin(REFERENCE_MODELS)]
    assert summarise_information_vs_algorithm(only_references).empty


def test_each_feature_set_gets_a_fresh_evaluator():
    """
    Reusing one evaluator would pool predictions across configurations, so the
    factory must be called once per feature set.
    """
    created = []

    class FakeEvaluator:
        def __init__(self):
            self.prediction_results = []

        def evaluate(self, df, fe, tf, feature_cols):
            self.prediction_results = [
                {"observation_id": "o1", "model": "m", "y_true": 1.0, "y_pred": 1.0}
            ]

    def factory(checkpoint_dir=None):
        evaluator = FakeEvaluator()
        created.append((evaluator, checkpoint_dir))
        return evaluator

    feature_sets = {"a": ["Prev_Total"], "b": ["Prev_Total", "Age"]}
    run_ablation(pd.DataFrame({"x": [1]}), factory, lambda *a, **k: None,
                 lambda *a, **k: None, feature_sets)

    assert len(created) == 2
    assert created[0][0] is not created[1][0]


def test_each_feature_set_checkpoints_to_its_own_directory(tmp_path):
    """
    The checkpoint fingerprint covers the feature set, so a shared directory
    would make each configuration reject and delete the previous one's work,
    leaving only the last one resumable -- the opposite of what is wanted from
    a sequence of runs lasting hours.
    """
    seen = []

    class FakeEvaluator:
        def __init__(self):
            self.prediction_results = []

        def evaluate(self, df, fe, tf, feature_cols):
            self.prediction_results = [
                {"observation_id": "o1", "model": "m", "y_true": 1.0, "y_pred": 1.0}
            ]

    def factory(checkpoint_dir=None):
        seen.append(checkpoint_dir)
        return FakeEvaluator()

    feature_sets = {"a": ["Prev_Total"], "b": ["Prev_Total", "Age"]}
    run_ablation(pd.DataFrame({"x": [1]}), factory, lambda *a, **k: None,
                 lambda *a, **k: None, feature_sets, checkpoint_root=tmp_path)

    assert len(set(seen)) == 2, "feature sets must not share a checkpoint directory"
    assert {p.name for p in seen} == {"a", "b"}


def test_checkpointing_is_optional():
    """Absent a root, no checkpoint directory is imposed on the evaluator."""
    seen = []

    class FakeEvaluator:
        def __init__(self):
            self.prediction_results = []

        def evaluate(self, *args):
            self.prediction_results = [
                {"observation_id": "o1", "model": "m", "y_true": 1.0, "y_pred": 1.0}
            ]

    def factory(checkpoint_dir=None):
        seen.append(checkpoint_dir)
        return FakeEvaluator()

    run_ablation(pd.DataFrame({"x": [1]}), factory, lambda *a, **k: None,
                 lambda *a, **k: None, {"a": ["Prev_Total"]})

    assert seen == [None]
