from typing import Any
import pandas as pd

from src.core.logging import get_logger
from src.models.factories import create_baseline_model, create_engineered_linear_model
from src.models.tuning import tune_gradient_boosting, tune_random_forest
from src.utils.metrics import compute_metrics
from src.contracts.training import TrainingResults, ModelPerformance

logger = get_logger(__name__)

def train_models(
    X_train: pd.DataFrame, 
    X_test: pd.DataFrame, 
    y_train: pd.Series, 
    y_test: pd.Series,
    X_train_scaled: pd.DataFrame, 
    X_test_scaled: pd.DataFrame,
    Xb_train_scaled: pd.DataFrame, 
    Xb_test_scaled: pd.DataFrame,
    yb_train: pd.Series, 
    yb_test: pd.Series
) -> TrainingResults:
    """Trains, tunes, and evaluates all machine learning models, returning a typed contract."""
    logger.info("Training baseline linear regression model...")
    baseline = create_baseline_model()
    baseline.fit(Xb_train_scaled, yb_train)
    y_pred_baseline = baseline.predict(Xb_test_scaled)
    baseline_metrics = compute_metrics(yb_test, y_pred_baseline)
    baseline_performance = ModelPerformance(metrics=baseline_metrics)
    logger.info("Baseline model training complete.")

    if len(X_train) < 100:
        logger.warning("Small dataset detected (< 100 samples), using simplified tuning.")

    logger.info("Tuning Random Forest model...")
    rf_model, rf_best_params = tune_random_forest(X_train, y_train)
    logger.info("Best Random Forest params: %s", rf_best_params)
    y_pred_rf = rf_model.predict(X_test)
    rf_metrics = compute_metrics(y_test, y_pred_rf)
    rf_performance = ModelPerformance(metrics=rf_metrics, cv_r2=rf_metrics.r2)

    logger.info("Tuning Gradient Boosting model...")
    gb_model, gb_best_params = tune_gradient_boosting(X_train, y_train)
    logger.info("Best Gradient Boosting params: %s", gb_best_params)
    y_pred_gb = gb_model.predict(X_test)
    gb_metrics = compute_metrics(y_test, y_pred_gb)
    gb_performance = ModelPerformance(metrics=gb_metrics, cv_r2=gb_metrics.r2)

    logger.info("Training engineered linear regression model...")
    lr_engineered = create_engineered_linear_model()
    lr_engineered.fit(X_train_scaled, y_train)
    y_pred_lr = lr_engineered.predict(X_test_scaled)
    lr_metrics = compute_metrics(y_test, y_pred_lr)
    lr_performance = ModelPerformance(metrics=lr_metrics, cv_r2=lr_metrics.r2)

    logger.info("All model training complete.")

    return TrainingResults(
        baseline=baseline_performance,
        linear_regression=lr_performance,
        random_forest=rf_performance,
        gradient_boosting=gb_performance,
        rf_model=rf_model,
        gb_model=gb_model
    )
