import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.contracts.metrics import EvaluationMetrics

def compute_metrics(y_true, y_pred) -> EvaluationMetrics:
    """
    Compute standard regression metrics and return them in a typed contract.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return EvaluationMetrics(mae=mae, rmse=rmse, r2=r2)
