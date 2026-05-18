import pytest
from src.contracts.metrics import EvaluationMetrics
from src.contracts.training import ModelPerformance, TrainingResults, ForwardTrainingResults

def test_evaluation_metrics():
    metrics = EvaluationMetrics(mae=1.0, rmse=2.0, r2=0.9)
    assert metrics.mae == 1.0
    assert metrics.rmse == 2.0
    assert metrics.r2 == 0.9

def test_evaluation_metrics_immutability():
    metrics = EvaluationMetrics(mae=1.0, rmse=2.0, r2=0.9)
    with pytest.raises(Exception):
        metrics.mae = 2.0

def test_model_performance():
    metrics = EvaluationMetrics(mae=1.0, rmse=2.0, r2=0.9)
    perf = ModelPerformance(metrics=metrics, cv_r2=0.85, cv_r2_ci=0.05)
    assert perf.metrics == metrics
    assert perf.cv_r2 == 0.85
    assert perf.cv_r2_ci == 0.05

def test_model_performance_immutability():
    metrics = EvaluationMetrics(mae=1.0, rmse=2.0, r2=0.9)
    perf = ModelPerformance(metrics=metrics)
    with pytest.raises(Exception):
        perf.cv_r2 = 0.85

def test_training_results():
    metrics = EvaluationMetrics(mae=1.0, rmse=2.0, r2=0.9)
    perf = ModelPerformance(metrics=metrics)
    
    results = TrainingResults(
        baseline=perf,
        linear_regression=perf,
        random_forest=perf,
        gradient_boosting=perf,
        rf_model="rf_mock",
        gb_model="gb_mock"
    )
    
    assert results.baseline == perf
    assert results.linear_regression == perf
    assert results.random_forest == perf
    assert results.gradient_boosting == perf
    assert results.rf_model == "rf_mock"
    assert results.gb_model == "gb_mock"

def test_forward_training_results():
    metrics = EvaluationMetrics(mae=1.0, rmse=2.0, r2=0.9)
    perf = ModelPerformance(metrics=metrics)
    
    results = ForwardTrainingResults(
        linear_regression=perf,
        random_forest=perf,
        gradient_boosting=perf,
        y_pred_lr=[1, 2, 3],
        y_pred_rf=[1, 2, 3],
        y_pred_gb=[1, 2, 3]
    )
    
    assert results.linear_regression == perf
    assert results.random_forest == perf
    assert results.gradient_boosting == perf
    assert results.y_pred_lr == [1, 2, 3]
    assert results.y_pred_rf == [1, 2, 3]
    assert results.y_pred_gb == [1, 2, 3]
