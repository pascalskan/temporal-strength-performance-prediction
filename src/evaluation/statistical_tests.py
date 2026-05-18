from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from typing import Tuple, Optional

from src.core.logging import get_logger
from src.evaluation.metrics import compute_basic_metrics
from src.contracts.evaluation import StatisticalEvaluationResults
from src.contracts.training import TrainingResults

logger = get_logger(__name__)

def compute_ci(scores: np.ndarray) -> Tuple[float, float]:
    scores = scores[~np.isnan(scores)]
    if len(scores) == 0:
        return np.nan, np.nan
    mean = np.mean(scores)
    if len(scores) < 2:
        return mean, 0.0
    std = np.std(scores, ddof=1)
    return mean, 1.96 * std / np.sqrt(len(scores))

def run_statistical_evaluation(
    df: pd.DataFrame, 
    training_results: TrainingResults, 
    save_dir: Path, 
    traditional_results: Optional[pd.DataFrame] = None
) -> StatisticalEvaluationResults:
    save_dir.mkdir(parents=True, exist_ok=True)
    X, y = df[["Sex", "Age", "BodyweightKg"]], df["TotalKg"]
    baseline, rf, gb = LinearRegression(), RandomForestRegressor(random_state=42), GradientBoostingRegressor(random_state=42)

    n_samples = len(df)
    cv_folds = 5
    if n_samples < 100:
        logger.warning("Small dataset (%d samples), adjusting CV folds.", n_samples)
        cv_folds = max(2, n_samples // 20)
        if cv_folds < 3:
            logger.warning("Dataset too small for CV. Skipping.")
            baseline_scores, rf_scores, gb_scores = [np.array([np.nan])]*3
        else:
            logger.info("Using %d-fold CV for small dataset.", cv_folds)
            baseline_scores = cross_val_score(baseline, X, y, cv=cv_folds, scoring="r2")
            rf_scores = cross_val_score(rf, X, y, cv=cv_folds, scoring="r2")
            gb_scores = cross_val_score(gb, X, y, cv=cv_folds, scoring="r2")
    else:
        logger.info("Running %d-fold time-series CV.", cv_folds)
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        baseline_scores = cross_val_score(baseline, X, y, cv=tscv, scoring="r2")
        rf_scores = cross_val_score(rf, X, y, cv=cv_folds, scoring="r2")
        gb_scores = cross_val_score(gb, X, y, cv=cv_folds, scoring="r2")

    baseline_mean, baseline_ci = compute_ci(baseline_scores)
    rf_mean, rf_ci = compute_ci(rf_scores)
    gb_mean, gb_ci = compute_ci(gb_scores)
    logger.info("CV R² (95%% CI): Baseline=%.3f±%.3f, RF=%.3f±%.3f, GB=%.3f±%.3f", baseline_mean, baseline_ci, rf_mean, rf_ci, gb_mean, gb_ci)

    def safe_std(arr: np.ndarray) -> float:
        arr = arr[~np.isnan(arr)]
        return np.std(arr, ddof=1) if len(arr) > 1 else np.nan
    baseline_std, rf_std, gb_std = safe_std(baseline_scores), safe_std(rf_scores), safe_std(gb_scores)

    def compute_effect_size(a: np.ndarray, b: np.ndarray) -> float:
        diff = a - b
        return np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) != 0 else 0.0
    
    valid_mask_rf = ~np.isnan(rf_scores) & ~np.isnan(baseline_scores)
    ttest_rel(rf_scores[valid_mask_rf], baseline_scores[valid_mask_rf]) if sum(valid_mask_rf) > 1 else (np.nan, np.nan)
    rf_effect = compute_effect_size(rf_scores[valid_mask_rf], baseline_scores[valid_mask_rf]) if sum(valid_mask_rf) > 1 else np.nan

    valid_mask_gb = ~np.isnan(gb_scores) & ~np.isnan(baseline_scores)
    ttest_rel(gb_scores[valid_mask_gb], baseline_scores[valid_mask_gb]) if sum(valid_mask_gb) > 1 else (np.nan, np.nan)
    gb_effect = compute_effect_size(gb_scores[valid_mask_gb], baseline_scores[valid_mask_gb]) if sum(valid_mask_gb) > 1 else np.nan

    baseline_metrics = training_results.baseline.metrics
    rf_metrics = training_results.random_forest.metrics
    gb_metrics = training_results.gradient_boosting.metrics

    results_list = [
        {"Model": "Baseline", "MAE": baseline_metrics.mae, "RMSE": baseline_metrics.rmse, "R2": baseline_metrics.r2, "CV_R2": baseline_mean, "CV_R2_CI": baseline_ci, "Effect_Size": 0, "R2_STD": baseline_std},
        {"Model": "Random Forest", "MAE": rf_metrics.mae, "RMSE": rf_metrics.rmse, "R2": rf_metrics.r2, "CV_R2": rf_mean, "CV_R2_CI": rf_ci, "Effect_Size": rf_effect, "R2_STD": rf_std},
        {"Model": "Gradient Boosting", "MAE": gb_metrics.mae, "RMSE": gb_metrics.rmse, "R2": gb_metrics.r2, "CV_R2": gb_mean, "CV_R2_CI": gb_ci, "Effect_Size": gb_effect, "R2_STD": gb_std},
    ]

    if traditional_results is not None:
        y_true = traditional_results["Actual_Total"]
        epley_metrics = compute_basic_metrics(y_true, traditional_results["Epley_Prediction"])
        brzycki_metrics = compute_basic_metrics(y_true, traditional_results["Brzycki_Prediction"])
        results_list.extend([
            {"Model": "Epley", "MAE": epley_metrics.mae, "RMSE": epley_metrics.rmse, "R2": epley_metrics.r2, "CV_R2": np.nan, "CV_R2_CI": np.nan, "Effect_Size": np.nan, "R2_STD": np.nan},
            {"Model": "Brzycki", "MAE": brzycki_metrics.mae, "RMSE": brzycki_metrics.rmse, "R2": brzycki_metrics.r2, "CV_R2": np.nan, "CV_R2_CI": np.nan, "Effect_Size": np.nan, "R2_STD": np.nan},
        ])

    results_table = pd.DataFrame(results_list)
    results_table.to_csv(save_dir / "model_results.csv", index=False)
    logger.info("Saved final model results table to %s", save_dir / "model_results.csv")
    
    return StatisticalEvaluationResults(results_table=results_table)
