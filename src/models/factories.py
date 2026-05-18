from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator

from src.config.constants import GB_DEFAULT_PARAMS, RF_DEFAULT_PARAMS


def create_baseline_model() -> BaseEstimator:
    return LinearRegression()


def create_engineered_linear_model() -> BaseEstimator:
    return LinearRegression()


def create_random_forest() -> BaseEstimator:
    return RandomForestRegressor(**RF_DEFAULT_PARAMS)


def create_gradient_boosting() -> BaseEstimator:
    return GradientBoostingRegressor(**GB_DEFAULT_PARAMS)
