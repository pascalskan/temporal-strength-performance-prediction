"""
The experimental configuration must have exactly one definition.

Model specifications previously existed separately in the walk-forward and
ablation pipelines. If those drift apart, the ablation measures different
models than the headline evaluation and the two result tables can no longer be
read together -- a correctness problem, not an untidiness one.
"""
import inspect

import pytest
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression

from src.config.experiment import (
    BOOTSTRAP_RESAMPLES,
    MIN_TRAIN_PERIODS,
    MODEL_COMPARISONS,
    WALK_FORWARD_GRANULARITY,
    build_all_baselines,
    build_models,
    build_temporal_baselines,
    build_traditional_baselines,
)


def test_models_are_freshly_constructed_each_call():
    """
    Evaluators fit models in place, so shared instances would carry one fold's
    fit into the next and silently leak training data across folds.
    """
    first, second = build_models(), build_models()
    assert first.keys() == second.keys()
    for name in first:
        assert first[name] is not second[name], f"{name} instance is shared"


def test_expected_model_families_are_present():
    models = build_models()
    assert isinstance(models["Linear Regression"], LinearRegression)
    assert isinstance(models["Random Forest"], RandomForestRegressor)
    assert isinstance(models["Gradient Boosting"], GradientBoostingRegressor)


def test_stochastic_models_are_seeded():
    """An unseeded ensemble would make every reported figure unreproducible."""
    models = build_models()
    assert models["Random Forest"].random_state is not None
    assert models["Gradient Boosting"].random_state is not None


def test_baseline_groups_compose_without_overlap():
    temporal = build_temporal_baselines()
    traditional = build_traditional_baselines()

    assert not set(temporal) & set(traditional)
    assert set(build_all_baselines()) == set(temporal) | set(traditional)


def test_traditional_baselines_read_the_frame():
    """They need attempt columns, which are excluded from the feature set."""
    for baseline in build_traditional_baselines().values():
        assert getattr(baseline, "requires_frame", False)


def test_comparisons_reference_only_defined_estimators():
    """A typo in a pair name would otherwise surface as a runtime failure."""
    available = set(build_models()) | set(build_all_baselines())
    for model_a, model_b in MODEL_COMPARISONS:
        assert model_a in available, f"unknown model in comparison: {model_a}"
        assert model_b in available, f"unknown model in comparison: {model_b}"


def test_comparisons_are_not_self_referential():
    for model_a, model_b in MODEL_COMPARISONS:
        assert model_a != model_b


def test_protocol_settings_are_sane():
    assert WALK_FORWARD_GRANULARITY in {"year", "month"}
    assert MIN_TRAIN_PERIODS >= 1
    assert BOOTSTRAP_RESAMPLES >= 1000, "too few resamples for stable 95% interval ends"


@pytest.mark.parametrize("module_name", [
    "src.pipelines.walk_forward_pipeline",
    "src.pipelines.ablation_pipeline",
])
def test_pipelines_do_not_define_their_own_models(module_name):
    """Regression: both pipelines previously constructed their own model sets."""
    import importlib

    source = inspect.getsource(importlib.import_module(module_name))
    for constructor in ("RandomForestRegressor(", "GradientBoostingRegressor(", "LinearRegression("):
        assert constructor not in source, (
            f"{module_name} constructs {constructor} directly; it should come "
            f"from src.config.experiment so both stages evaluate the same models"
        )
