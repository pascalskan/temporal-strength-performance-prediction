from pathlib import Path
import pandas as pd
from typing import Dict
from src.core.logging import get_logger
from src.contracts.training import TrainingResults

logger = get_logger(__name__)

def generate_model_report(
    naive_results: Dict[str, float],
    training_results: TrainingResults,
    evaluation_dir: Path
) -> pd.DataFrame:
    """Generates a CSV report comparing the performance of all models."""
    logger.info("Generating model comparison report...")

    baseline_metrics = training_results.baseline.metrics

    results_data = [
        {"Model": "Naive (Prev_Total)", **naive_results},
        {"Model": "Baseline (Linear)", "MAE": baseline_metrics.mae, "RMSE": baseline_metrics.rmse, "R2": baseline_metrics.r2}
    ]
    
    ml_models = {
        "Random Forest": training_results.random_forest.metrics,
        "Gradient Boosting": training_results.gradient_boosting.metrics,
        "Linear Regression (Engineered)": training_results.linear_regression.metrics
    }

    for model_name, metrics in ml_models.items():
        results_data.append({
            "Model": model_name,
            "MAE": metrics.mae,
            "RMSE": metrics.rmse,
            "R2": metrics.r2
        })

    results_df = pd.DataFrame(results_data)
    results_path = evaluation_dir / "model_comparison_full.csv"
    results_df.to_csv(results_path, index=False)
    logger.info("Saved full model comparison report to %s", results_path)

    baseline_r2 = baseline_metrics.r2
    if pd.notna(baseline_r2):
        logger.info("Comparing ML models to baseline R2 score of %.3f...", baseline_r2)
        for model_name, metrics in ml_models.items():
            model_r2 = metrics.r2
            if pd.notna(model_r2) and abs(baseline_r2) > 1e-9:
                improvement = (model_r2 - baseline_r2) / abs(baseline_r2)
                logger.info(
                    "%-30s | R2: % .3f (Improvement vs. Baseline: % .2f%%)",
                    model_name, model_r2, improvement * 100
                )
            else:
                logger.info("%-30s | R2: % .3f", model_name, model_r2 if pd.notna(model_r2) else 0)

    return results_df
