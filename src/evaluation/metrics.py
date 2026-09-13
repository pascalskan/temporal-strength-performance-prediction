import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.contracts.metrics import EvaluationMetrics

def compute_basic_metrics(y_true, y_pred) -> EvaluationMetrics:
    """
    Compute standard regression metrics after handling NaNs and return a typed contract.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)

    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return EvaluationMetrics(mae=float('nan'), rmse=float('nan'), r2=float('nan'))

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    return EvaluationMetrics(mae=mae, rmse=rmse, r2=r2)

def symmetric_mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the Symmetric Mean Absolute Percentage Error (sMAPE).
    The result is returned as a percentage (0-200).
    If the denominator is zero for a given observation, that observation is excluded.
    """
    denominator = np.abs(y_true) + np.abs(y_pred)
    # Avoid division by zero
    mask = denominator != 0
    if not np.any(mask):
        return 0.0  # Or np.nan, depending on desired behavior for all-zero inputs
    
    return np.mean(200 * np.abs(y_true[mask] - y_pred[mask]) / denominator[mask])

def normalized_root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the Normalized Root Mean Squared Error (NRMSE).
    The RMSE is normalized by the mean of the absolute true values.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mean_abs_true = np.mean(np.abs(y_true))
    if mean_abs_true == 0:
        return np.nan  # Undefined if the mean of the true values is zero
    return rmse / mean_abs_true

def compute_global_forecast_metrics(predictions_df: pd.DataFrame, model_col: str = "model", y_true_col: str = "y_true", y_pred_col: str = "y_pred") -> pd.DataFrame:
    """
    Compute headline performance metrics from pooled prediction-level outputs.
    This prevents invalid statistical weighting of walk-forward windows with unequal sample sizes.
    """
    records = []
    for model_name, group in predictions_df.groupby(model_col):
        mask = ~np.isnan(group[y_pred_col]) & ~np.isnan(group[y_true_col])
        valid_group = group[mask]
        
        y_true = valid_group[y_true_col].values
        y_pred = valid_group[y_pred_col].values
        
        n_samples = len(y_true)
        if n_samples == 0:
            records.append({
                "model": model_name,
                "prediction_count": 0,
                "MAE": float('nan'),
                "RMSE": float('nan'),
                "R2": float('nan'),
                "median_absolute_error": float('nan'),
                "mean_residual": float('nan'),
                "sMAPE": float('nan'),
                "NRMSE": float('nan')
            })
            continue

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        if np.var(y_true) > 0:
            r2 = r2_score(y_true, y_pred)
        else:
            r2 = float('nan')
            
        median_ae = np.median(np.abs(y_true - y_pred))
        mean_res = np.mean(y_pred - y_true)
        smape = symmetric_mean_absolute_percentage_error(y_true, y_pred)
        nrmse = normalized_root_mean_squared_error(y_true, y_pred)

        records.append({
            "model": model_name,
            "prediction_count": n_samples,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "median_absolute_error": median_ae,
            "mean_residual": mean_res,
            "sMAPE": smape,
            "NRMSE": nrmse
        })
        
    return pd.DataFrame(records)
