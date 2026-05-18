from dataclasses import dataclass
from typing import Any, Optional
from .metrics import EvaluationMetrics


@dataclass(frozen=True)
class ModelPerformance:
    """Holds the performance metrics for a single model."""
    metrics: EvaluationMetrics
    cv_r2: Optional[float] = None
    cv_r2_ci: Optional[float] = None


@dataclass
class TrainingResults:
    """Container for all results from a retrospective training run."""
    baseline: ModelPerformance
    linear_regression: ModelPerformance
    random_forest: ModelPerformance
    gradient_boosting: ModelPerformance
    rf_model: Any
    gb_model: Any


@dataclass(frozen=True)
class ForwardTrainingResults:
    """Container for results from a forward training run."""
    linear_regression: ModelPerformance
    random_forest: ModelPerformance
    gradient_boosting: ModelPerformance
    y_pred_lr: Any
    y_pred_rf: Any
    y_pred_gb: Any


@dataclass(frozen=True)
class RetrospectiveReferenceResults:
    """Container for results from retrospective reference models."""
    linear_regression: ModelPerformance
    random_forest: ModelPerformance
    gradient_boosting: ModelPerformance
