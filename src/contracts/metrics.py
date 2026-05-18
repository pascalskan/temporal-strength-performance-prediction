from dataclasses import dataclass

@dataclass(frozen=True)
class EvaluationMetrics:
    """A standard container for regression model evaluation metrics."""
    mae: float
    rmse: float
    r2: float
