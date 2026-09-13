import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional

def paired_bootstrap_comparison(
    predictions_df: pd.DataFrame,
    model_a_name: str,
    model_b_name: str,
    metric: str = 'mae',
    n_bootstraps: int = 5000,
    random_seed: Optional[int] = None
) -> Dict[str, float]:
    """
    Performs a paired bootstrap comparison between two models.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    model_a_preds = predictions_df[predictions_df['model'] == model_a_name]
    model_b_preds = predictions_df[predictions_df['model'] == model_b_name]

    # Align predictions
    merged_preds = pd.merge(
        model_a_preds,
        model_b_preds,
        on=['athlete_id', 'date', 'forecast_window', 'y_true'],
        suffixes=('_a', '_b')
    )

    if merged_preds.empty:
        return {
            'observed_difference': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'bootstrap_p': np.nan
        }

    y_true = merged_preds['y_true'].values
    y_pred_a = merged_preds['y_pred_a'].values
    y_pred_b = merged_preds['y_pred_b'].values

    if metric == 'mae':
        error_a = np.abs(y_true - y_pred_a)
        error_b = np.abs(y_true - y_pred_b)
    elif metric == 'rmse':
        error_a = (y_true - y_pred_a)**2
        error_b = (y_true - y_pred_b)**2
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    observed_diff = np.mean(error_a) - np.mean(error_b)
    if metric == 'rmse':
        observed_diff = np.sqrt(np.mean(error_a)) - np.sqrt(np.mean(error_b))

    # Bootstrap resampling
    n_obs = len(merged_preds)
    bootstrap_diffs = []
    for _ in range(n_bootstraps):
        indices = np.random.choice(n_obs, size=n_obs, replace=True)
        
        resampled_error_a = error_a[indices]
        resampled_error_b = error_b[indices]
        
        if metric == 'mae':
            diff = np.mean(resampled_error_a) - np.mean(resampled_error_b)
        elif metric == 'rmse':
            diff = np.sqrt(np.mean(resampled_error_a)) - np.sqrt(np.mean(resampled_error_b))
        
        bootstrap_diffs.append(diff)

    # Calculate confidence interval and p-value
    ci_lower = np.percentile(bootstrap_diffs, 2.5)
    ci_upper = np.percentile(bootstrap_diffs, 97.5)
    
    # Proportion of bootstrap differences on the opposite side of zero from the observed difference
    if observed_diff > 0:
        bootstrap_p = np.mean(np.array(bootstrap_diffs) <= 0)
    elif observed_diff < 0:
        bootstrap_p = np.mean(np.array(bootstrap_diffs) >= 0)
    else:
        bootstrap_p = 1.0

    return {
        'observed_difference': observed_diff,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'bootstrap_p': bootstrap_p
    }

def run_all_comparisons(predictions_df: pd.DataFrame, comparisons: List[Tuple[str, str]], metrics: List[str] = ['mae', 'rmse'], **kwargs) -> pd.DataFrame:
    """
    Runs paired bootstrap comparisons for a list of model pairs and metrics.
    """
    results = []
    for model_a, model_b in comparisons:
        for metric in metrics:
            result = paired_bootstrap_comparison(
                predictions_df,
                model_a,
                model_b,
                metric=metric,
                **kwargs
            )
            results.append({
                'model_a': model_a,
                'model_b': model_b,
                'metric': metric,
                **result
            })
    return pd.DataFrame(results)
