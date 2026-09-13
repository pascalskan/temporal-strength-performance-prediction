import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression
from src.evaluation.metrics import (
    symmetric_mean_absolute_percentage_error,
    normalized_root_mean_squared_error
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def compute_temporal_metrics(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Computes standard evaluation metrics per model, per forecast window."""
    results = []
    
    # Ensure forecast_window is treated appropriately (e.g., as numeric for sorting/plotting if possible)
    # But group by exactly as it is to respect the walk-forward structure.
    for (model, window), group in predictions_df.groupby(['model', 'forecast_window']):
        mask = ~np.isnan(group['y_pred']) & ~np.isnan(group['y_true'])
        valid_group = group[mask]
        
        y_true = valid_group['y_true'].values
        y_pred = valid_group['y_pred'].values
        
        n_samples = len(y_true)
        if n_samples == 0:
            continue
            
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        if np.var(y_true) > 0:
            r2 = r2_score(y_true, y_pred)
        else:
            r2 = float('nan')
            
        mean_res = np.mean(y_pred - y_true)
        smape = symmetric_mean_absolute_percentage_error(y_true, y_pred)
        nrmse = normalized_root_mean_squared_error(y_true, y_pred)
        
        results.append({
            'model': model,
            'forecast_window': window,
            'prediction_count': n_samples,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'sMAPE': smape,
            'NRMSE': nrmse,
            'mean_residual': mean_res
        })
        
    return pd.DataFrame(results)

def estimate_temporal_drift(temporal_metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Estimates the linear trend (drift) of performance metrics over time."""
    drift_results = []
    
    metrics_to_track = ['MAE', 'RMSE', 'mean_residual']
    
    for model, group in temporal_metrics_df.groupby('model'):
        # Ensure we sort by time to make sense of the trend
        # Note: assumes forecast_window is somewhat chronological/numeric
        # If it's a period object or string, we might need robust parsing, 
        # but standard walk-forward evaluates sequentially.
        sorted_group = group.sort_values('forecast_window')
        
        # We need a numeric time index for regression. 
        # Simple integer sequence representing evaluation steps:
        time_index = np.arange(len(sorted_group)).reshape(-1, 1)
        
        model_drift = {'model': model}
        
        for metric in metrics_to_track:
            y = sorted_group[metric].values
            
            # Handle possible NaNs (e.g. if a window failed)
            mask = ~np.isnan(y)
            if np.sum(mask) < 2:
                model_drift[f'{metric}_slope'] = float('nan')
                model_drift[f'{metric}_intercept'] = float('nan')
                continue
                
            lr = LinearRegression()
            lr.fit(time_index[mask], y[mask])
            
            model_drift[f'{metric}_slope'] = lr.coef_[0]
            model_drift[f'{metric}_intercept'] = lr.intercept_
            
        drift_results.append(model_drift)
        
    return pd.DataFrame(drift_results)


def plot_metric_over_time(temporal_metrics_df: pd.DataFrame, metric: str, ylabel: str, title: str, save_path: Path):
    """Plots a specific metric over the forecast windows for all models."""
    plt.figure(figsize=(10, 6))
    
    for model, group in temporal_metrics_df.groupby('model'):
        sorted_group = group.sort_values('forecast_window')
        # Convert forecast_window to string for categorical x-axis if it's not strictly numeric
        x_vals = sorted_group['forecast_window'].astype(str)
        plt.plot(x_vals, sorted_group[metric], marker='o', label=model)
        
    if metric == 'mean_residual':
        plt.axhline(0, color='black', linestyle='--', alpha=0.5)
        
    plt.xlabel('Forecast Window')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def run_and_save_temporal_analysis(predictions_df: pd.DataFrame, output_dir: Path):
    """Runs temporal stability analysis, generates plots, and saves outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Compute per-window metrics
    temporal_metrics_df = compute_temporal_metrics(predictions_df)
    metrics_path = output_dir / "temporal_metrics.csv"
    temporal_metrics_df.to_csv(metrics_path, index=False)
    
    # 2. Estimate Drift
    drift_df = estimate_temporal_drift(temporal_metrics_df)
    drift_path = output_dir / "temporal_drift_summary.csv"
    drift_df.to_csv(drift_path, index=False)
    
    # 3. Generate Plots
    plot_metric_over_time(
        temporal_metrics_df, 
        metric='MAE', 
        ylabel='Mean Absolute Error', 
        title='MAE by Forecast Window', 
        save_path=output_dir / "temporal_mae.png"
    )
    
    plot_metric_over_time(
        temporal_metrics_df, 
        metric='RMSE', 
        ylabel='Root Mean Squared Error', 
        title='RMSE by Forecast Window', 
        save_path=output_dir / "temporal_rmse.png"
    )
    
    plot_metric_over_time(
        temporal_metrics_df, 
        metric='mean_residual', 
        ylabel='Mean Residual (Bias)', 
        title='Mean Residual by Forecast Window', 
        save_path=output_dir / "temporal_residual.png"
    )
