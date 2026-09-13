import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from src.models.utils import get_model_attribute

def compute_diagnostic_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Computes scalar diagnostics for bias and calibration."""
    
    # Mean Residual
    mean_residual = np.mean(y_pred - y_true)
    
    # Mean Percentage Bias
    # Exclude cases where y_true is zero to avoid division by zero
    mask = y_true != 0
    mean_percentage_bias = 100 * np.mean((y_pred[mask] - y_true[mask]) / y_true[mask])
    
    # Calibration Regression
    lr = LinearRegression()
    lr.fit(y_pred.reshape(-1, 1), y_true)
    cal_slope = get_model_attribute(lr, "coef_")[0]
    cal_intercept = get_model_attribute(lr, "intercept_")
    
    # Residual Standard Deviation
    residuals = y_true - lr.predict(y_pred.reshape(-1, 1))
    residual_std = np.std(residuals)
    
    return {
        'mean_residual': mean_residual,
        'mean_percentage_bias': mean_percentage_bias,
        'calibration_slope': cal_slope,
        'calibration_intercept': cal_intercept,
        'residual_std_dev': residual_std
    }

def compute_decile_performance(y_true: np.ndarray, y_pred: np.ndarray, n_deciles: int = 10) -> pd.DataFrame:
    """Computes error metrics stratified by performance deciles."""
    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
    df['decile'] = pd.qcut(df['y_true'], n_deciles, labels=False, duplicates='drop')
    
    results = []
    for decile, group in df.groupby('decile'):
        mae = np.mean(np.abs(group['y_true'] - group['y_pred']))
        rmse = np.sqrt(np.mean((group['y_true'] - group['y_pred'])**2))
        mean_residual = np.mean(group['y_pred'] - group['y_true'])
        
        results.append({
            'decile': decile,
            'mae': mae,
            'rmse': rmse,
            'mean_residual': mean_residual,
            'n_samples': len(group)
        })
    return pd.DataFrame(results)

def run_and_save_diagnostics(predictions_df: pd.DataFrame, output_dir: Path):
    """Runs all diagnostic analyses for each model and saves the outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    scalar_results = []
    decile_results = []
    
    for model_name, group in predictions_df.groupby('model'):
        y_true = group['y_true'].values
        y_pred = group['y_pred'].values
        
        # Scalar diagnostics
        scalar_metrics = compute_diagnostic_metrics(y_true, y_pred)
        scalar_metrics['model'] = model_name
        scalar_results.append(scalar_metrics)
        
        # Decile performance
        decile_df = compute_decile_performance(y_true, y_pred)
        decile_df['model'] = model_name
        decile_results.append(decile_df)
        
        # Plots
        plot_calibration(y_true, y_pred, model_name, output_dir)
        plot_residuals(y_true, y_pred, model_name, output_dir)
    
    # Save scalar and decile results
    pd.DataFrame(scalar_results).to_csv(output_dir / "diagnostic_summary.csv", index=False)
    pd.concat(decile_results).to_csv(output_dir / "decile_performance.csv", index=False)
    
    # Plot combined decile performance
    plot_decile_performance(pd.concat(decile_results), output_dir)

def plot_calibration(y_true, y_pred, model_name, save_dir):
    plt.figure(figsize=(8, 8))
    plt.scatter(y_pred, y_true, alpha=0.3, label='Predictions')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', label='Ideal (y=x)')
    plt.xlabel("Predicted Values")
    plt.ylabel("Actual Values")
    plt.title(f"Calibration Plot - {model_name}")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir / f"calibration_plot_{model_name}.png")
    plt.close()

def plot_residuals(y_true, y_pred, model_name, save_dir):
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.3)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals (Actual - Predicted)")
    plt.title(f"Residual vs. Predicted Plot - {model_name}")
    plt.grid(True)
    plt.savefig(save_dir / f"residual_plot_{model_name}.png")
    plt.close()

def plot_decile_performance(decile_df, save_dir):
    plt.figure(figsize=(10, 6))
    for model_name, group in decile_df.groupby('model'):
        plt.plot(group['decile'], group['mean_residual'], marker='o', linestyle='--', label=f"{model_name} Bias")
    
    plt.axhline(0, color='k', linestyle='-')
    plt.xlabel("Performance Decile (by Actual Value)")
    plt.ylabel("Mean Residual (Bias)")
    plt.title("Bias by Performance Decile")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir / "bias_by_decile.png")
    plt.close()