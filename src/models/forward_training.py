import pandas as pd

from src.core.logging import get_logger
from src.evaluation.metrics import compute_basic_metrics
from src.models.factories import create_engineered_linear_model, create_gradient_boosting, create_random_forest
from src.utils.scaling import scale_features
from src.contracts.training import ForwardTrainingResults, RetrospectiveReferenceResults, ModelPerformance

logger = get_logger(__name__)


def train_retrospective_reference_models(
    X_train_r: pd.DataFrame, 
    X_test_r: pd.DataFrame, 
    y_train_r: pd.Series, 
    y_test_r: pd.Series
) -> RetrospectiveReferenceResults:
    """Trains retrospective models to serve as a performance baseline."""
    logger.info("Training retrospective reference models...")
    X_train_r_scaled, X_test_r_scaled, _ = scale_features(X_train_r, X_test_r)

    lr_r = create_engineered_linear_model()
    lr_r.fit(X_train_r_scaled, y_train_r)
    y_pred_lr_r = lr_r.predict(X_test_r_scaled)
    lr_metrics = compute_basic_metrics(y_test_r, y_pred_lr_r)
    lr_perf = ModelPerformance(metrics=lr_metrics)
    logger.info("Retrospective Linear Regression: R2=%.3f", lr_metrics.r2)

    rf_r = create_random_forest()
    rf_r.fit(X_train_r, y_train_r)
    y_pred_rf_r = rf_r.predict(X_test_r)
    rf_metrics = compute_basic_metrics(y_test_r, y_pred_rf_r)
    rf_perf = ModelPerformance(metrics=rf_metrics)
    logger.info("Retrospective Random Forest: R2=%.3f", rf_metrics.r2)

    gb_r = create_gradient_boosting()
    gb_r.fit(X_train_r, y_train_r)
    y_pred_gb_r = gb_r.predict(X_test_r)
    gb_metrics = compute_basic_metrics(y_test_r, y_pred_gb_r)
    gb_perf = ModelPerformance(metrics=gb_metrics)
    logger.info("Retrospective Gradient Boosting: R2=%.3f", gb_metrics.r2)

    logger.info("Retrospective reference model training complete.")
    return RetrospectiveReferenceResults(
        linear_regression=lr_perf, 
        random_forest=rf_perf, 
        gradient_boosting=gb_perf
    )


def train_forward_models(
    X_train: pd.DataFrame, 
    X_test: pd.DataFrame, 
    y_train: pd.Series, 
    y_test: pd.Series
) -> ForwardTrainingResults:
    """Trains models for the forward prediction task."""
    logger.info("Training forward prediction models...")
    X_train_scaled, X_test_scaled, _ = scale_features(X_train, X_test)

    lr_model = create_engineered_linear_model()
    lr_model.fit(X_train_scaled, y_train)
    y_pred_lr = lr_model.predict(X_test_scaled)
    lr_metrics = compute_basic_metrics(y_test, y_pred_lr)
    lr_perf = ModelPerformance(metrics=lr_metrics)
    logger.info("Forward Linear Regression: R2=%.3f", lr_metrics.r2)

    rf_model = create_random_forest()
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    rf_metrics = compute_basic_metrics(y_test, y_pred_rf)
    rf_perf = ModelPerformance(metrics=rf_metrics)
    logger.info("Forward Random Forest: R2=%.3f", rf_metrics.r2)

    gb_model = create_gradient_boosting()
    gb_model.fit(X_train, y_train)
    y_pred_gb = gb_model.predict(X_test)
    gb_metrics = compute_basic_metrics(y_test, y_pred_gb)
    gb_perf = ModelPerformance(metrics=gb_metrics)
    logger.info("Forward Gradient Boosting: R2=%.3f", gb_metrics.r2)

    logger.info("Forward model training complete.")
    return ForwardTrainingResults(
        linear_regression=lr_perf,
        random_forest=rf_perf,
        gradient_boosting=gb_perf,
        y_pred_lr=y_pred_lr,
        y_pred_rf=y_pred_rf,
        y_pred_gb=y_pred_gb
    )
