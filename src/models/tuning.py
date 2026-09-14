from typing import Tuple, Dict, Any
import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from src.config.constants import (
    GB_FULL_GRID,
    GB_SMALL_GRID,
    RF_FULL_GRID,
    RF_SMALL_GRID,
    SMALL_DATASET_THRESHOLD,
)
from src.models.factories import create_gradient_boosting, create_random_forest

# Cross-validation strategy for hyperparameter search.
#
# Forward-chaining rather than K-fold. The data reaching these tuners comes from
# a chronological split and remains in time order, so K-fold -- which holds out
# one contiguous block and trains on the rest, including blocks that come after
# it -- would select hyperparameters using later records to predict earlier
# ones. That is the leakage this project exists to measure, and hyperparameter
# selection is not exempt: parameters chosen with sight of the future are
# chosen partly for a task the model will never face at deployment.
#
# TimeSeriesSplit always trains on a prefix and validates on the segment that
# follows, matching how the model is ultimately evaluated.
CV_SPLITS = 3


def _cross_validator(n_samples: int) -> TimeSeriesSplit:
    """Forward-chaining splitter, sized so every fold has data to train on."""
    n_splits = min(CV_SPLITS, max(2, n_samples - 1))
    return TimeSeriesSplit(n_splits=n_splits)


def get_param_grids(dataset_size: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if dataset_size < SMALL_DATASET_THRESHOLD:
        return RF_SMALL_GRID, GB_SMALL_GRID
    return RF_FULL_GRID, GB_FULL_GRID


def tune_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[Any, Dict[str, Any]]:
    rf_param_grid, _ = get_param_grids(len(X_train))

    rf_grid = GridSearchCV(
        create_random_forest(),
        rf_param_grid,
        cv=_cross_validator(len(X_train)),
        scoring="r2",
        n_jobs=-1
    )

    rf_grid.fit(X_train, y_train)

    return rf_grid.best_estimator_, rf_grid.best_params_


def tune_gradient_boosting(X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[Any, Dict[str, Any]]:
    _, gb_param_grid = get_param_grids(len(X_train))

    gb_grid = GridSearchCV(
        create_gradient_boosting(),
        gb_param_grid,
        cv=_cross_validator(len(X_train)),
        scoring="r2",
        n_jobs=-1
    )

    gb_grid.fit(X_train, y_train)

    return gb_grid.best_estimator_, gb_grid.best_params_
