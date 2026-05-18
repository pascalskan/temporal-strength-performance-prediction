from typing import Tuple, Dict, Any
import pandas as pd
from sklearn.model_selection import GridSearchCV

from src.config.constants import (
    GB_FULL_GRID,
    GB_SMALL_GRID,
    RF_FULL_GRID,
    RF_SMALL_GRID,
    SMALL_DATASET_THRESHOLD,
)
from src.models.factories import create_gradient_boosting, create_random_forest


def get_param_grids(dataset_size: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if dataset_size < SMALL_DATASET_THRESHOLD:
        return RF_SMALL_GRID, GB_SMALL_GRID
    return RF_FULL_GRID, GB_FULL_GRID


def tune_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[Any, Dict[str, Any]]:
    rf_param_grid, _ = get_param_grids(len(X_train))

    rf_grid = GridSearchCV(
        create_random_forest(),
        rf_param_grid,
        cv=3,
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
        cv=3,
        scoring="r2",
        n_jobs=-1
    )

    gb_grid.fit(X_train, y_train)

    return gb_grid.best_estimator_, gb_grid.best_params_
