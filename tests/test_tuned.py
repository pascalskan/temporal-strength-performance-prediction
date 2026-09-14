"""
Hyperparameter selection inside walk-forward folds.

Exists to close the objection that the learned models were never optimised,
which bears directly on the project's central negative result. The tuning must
itself be temporally safe, or it reintroduces the leakage the study measures.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.tree import DecisionTreeRegressor

from src.models.tuned import PeriodicallyTunedRegressor, collect_selection_history


def make_data(n=120, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = pd.Series(3 * X["a"] - 2 * X["b"] + rng.normal(scale=0.2, size=n))
    return X, y


def build(retune_every=5, **kwargs):
    return PeriodicallyTunedRegressor(
        estimator=Ridge(),
        param_grid={"alpha": [0.01, 1.0, 100.0]},
        retune_every=retune_every,
        name="Ridge",
        **kwargs,
    )


def test_selects_a_configuration_from_the_grid():
    X, y = make_data()
    model = build().fit(X, y)

    assert model.best_params_["alpha"] in (0.01, 1.0, 100.0)


def test_predicts_after_fitting():
    X, y = make_data()
    predictions = build().fit(X, y).predict(X)

    assert len(predictions) == len(X)
    assert np.isfinite(predictions).all()


def test_predicting_before_fitting_is_an_error():
    with pytest.raises(RuntimeError, match="has not been fitted"):
        build().predict(make_data()[0])


def test_search_uses_forward_chaining_not_k_fold(monkeypatch):
    """
    The property that makes this safe. K-fold would select hyperparameters
    using later records to predict earlier ones -- the leakage this project
    exists to measure, relocated into the tuning step.
    """
    seen = {}

    import src.models.tuned as tuned_module
    original = tuned_module.GridSearchCV

    def capture(estimator, param_grid, cv, **kwargs):
        seen["cv"] = cv
        return original(estimator, param_grid, cv=cv, **kwargs)

    monkeypatch.setattr(tuned_module, "GridSearchCV", capture)

    X, y = make_data()
    build().fit(X, y)

    assert isinstance(seen["cv"], TimeSeriesSplit)


def test_search_repeats_only_every_n_fits():
    """Cost control: a full search at every fold is several times the price."""
    X, y = make_data()
    model = build(retune_every=3)

    for _ in range(7):
        model.fit(X, y)

    # Fits 0, 3 and 6 search; the rest carry the last selection forward.
    assert len(model.selection_history) == 3
    assert [entry["fit_index"] for entry in model.selection_history] == [0, 3, 6]


def test_carried_forward_parameters_are_actually_applied():
    """A fit that skips the search must still use the selected configuration."""
    X, y = make_data()
    model = build(retune_every=100)

    model.fit(X, y)
    selected = model.best_params_["alpha"]

    model.fit(X, y)
    assert model.fitted_estimator_.get_params()["alpha"] == selected


def test_window_too_small_to_validate_falls_back_rather_than_failing():
    """
    Early walk-forward folds can hold a handful of rows. TimeSeriesSplit needs
    more samples than splits, so such a window cannot be searched -- it must
    still produce predictions rather than abort the fold.
    """
    X, y = make_data(n=2)
    model = build().fit(X, y)

    assert model.selection_history == []
    assert len(model.predict(X)) == 2


def test_attribution_is_readable_through_the_wrapper():
    """
    The evaluator captures feature importance off the fitted estimator; the
    wrapper must not hide it, or the tuned run would silently lose attribution.
    """
    X, y = make_data()
    model = PeriodicallyTunedRegressor(
        estimator=DecisionTreeRegressor(random_state=0),
        param_grid={"max_depth": [2, 4]},
        name="Tree",
    ).fit(X, y)

    assert len(model.feature_importances_) == X.shape[1]


def test_selection_history_records_what_was_chosen_and_when():
    X, y = make_data()
    model = build(retune_every=2)
    for _ in range(4):
        model.fit(X, y)

    history = collect_selection_history({"Ridge": model})

    assert len(history) == 2
    assert {"model", "fit_index", "n_train", "best_score"} <= set(history.columns)
    assert "param_alpha" in history.columns


def test_history_is_empty_for_untuned_models():
    """collect_selection_history is called over a mixed model set."""
    assert collect_selection_history({"Plain": Ridge()}).empty


def test_each_fit_starts_from_a_clean_estimator():
    """
    Folds must not inherit one another's fitted state; the evaluator refits the
    same instance on an expanding window at every fold.
    """
    X, y = make_data()
    model = build(retune_every=100)

    model.fit(X.iloc[:60], y.iloc[:60])
    first = model.fitted_estimator_

    model.fit(X, y)
    assert model.fitted_estimator_ is not first
