import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.contracts.metrics import EvaluationMetrics

def compute_basic_metrics(y_true, y_pred) -> EvaluationMetrics:
    """
    Compute standard regression metrics after handling NaNs and return a typed contract.
    """
    mask = ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return EvaluationMetrics(mae=float('nan'), rmse=float('nan'), r2=float('nan'))

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    return EvaluationMetrics(mae=mae, rmse=rmse, r2=r2)
