import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.evaluation.metrics import symmetric_mean_absolute_percentage_error, normalized_root_mean_squared_error

def paired_bootstrap_comparison(
    predictions_df: pd.DataFrame,
    model_a_name: str,
    model_b_name: str,
    metric: str = 'mae',
    n_bootstraps: int = 5000,
    random_seed: Optional[int] = None
) -> Dict[str, float]:
    """
    Performs a paired cluster bootstrap comparison at the athlete level.
    
    The bootstrap_p is a two-sided p-value testing the null hypothesis that the
    two models have equal performance on the chosen metric. It is calculated
    as twice the minimum proportion of bootstrap differences that fall strictly
    on either side of zero, capped at 1.0. This represents the probability
    that the true difference is zero or of the opposite sign.
    """
    rng = np.random.default_rng(random_seed)

    model_a_preds = predictions_df[predictions_df['model'] == model_a_name]
    model_b_preds = predictions_df[predictions_df['model'] == model_b_name]

    # Use the stable, unique observation ID for alignment
    merge_keys = ['observation_id']

    # Validate that the merge keys uniquely identify a prediction within each model
    if model_a_preds.duplicated(subset=merge_keys).any():
        raise ValueError(f"Duplicate predictions found for model {model_a_name} on keys {merge_keys}")
    if model_b_preds.duplicated(subset=merge_keys).any():
        raise ValueError(f"Duplicate predictions found for model {model_b_name} on keys {merge_keys}")

    # Align predictions
    merged_preds = pd.merge(
        model_a_preds,
        model_b_preds,
        on=merge_keys,
        suffixes=('_a', '_b'),
        how='inner'
    )

    if merged_preds.empty:
        raise ValueError(f"No matching predictions found between {model_a_name} and {model_b_name} to align.")

    # Confirm y_true matches after alignment
    if not np.allclose(merged_preds['y_true_a'], merged_preds['y_true_b'], equal_nan=True):
        raise ValueError(
            f"Aligned observations have mismatched true targets (y_true) for {model_a_name} and {model_b_name}."
        )

    # Calculate observed difference
    if metric == 'mae':
        error_a = np.abs(merged_preds['y_true_a'] - merged_preds['y_pred_a'])
        error_b = np.abs(merged_preds['y_true_b'] - merged_preds['y_pred_b'])
        observed_diff = np.mean(error_a) - np.mean(error_b)
    elif metric == 'rmse':
        error_a = (merged_preds['y_true_a'] - merged_preds['y_pred_a'])**2
        error_b = (merged_preds['y_true_b'] - merged_preds['y_pred_b'])**2
        observed_diff = np.sqrt(np.mean(error_a)) - np.sqrt(np.mean(error_b))
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    # Cluster bootstrap resampling
    unique_athletes = merged_preds['athlete_id_a'].unique() # Use athlete_id from one of the merged sides
    n_athletes = len(unique_athletes)
    bootstrap_diffs = []
    
    # Pre-group the data by athlete_id for faster lookups during the bootstrap loop
    grouped_preds = {athlete: df for athlete, df in merged_preds.groupby('athlete_id_a')}
    
    for _ in range(n_bootstraps):
        # Sample athlete IDs with replacement
        resampled_athlete_ids = rng.choice(unique_athletes, size=n_athletes, replace=True)
        
        # Collect all rows for the sampled athletes, preserving multiplicity
        bootstrap_parts = [grouped_preds[athlete] for athlete in resampled_athlete_ids]
        
        # It is guaranteed that bootstrap_parts is not empty because n_athletes >= 1
        resampled_df = pd.concat(bootstrap_parts, ignore_index=True)

        if metric == 'mae':
            resampled_error_a = np.abs(resampled_df['y_true_a'] - resampled_df['y_pred_a'])
            resampled_error_b = np.abs(resampled_df['y_true_b'] - resampled_df['y_pred_b'])
            diff = np.mean(resampled_error_a) - np.mean(resampled_error_b)
        elif metric == 'rmse':
            resampled_error_a = (resampled_df['y_true_a'] - resampled_df['y_pred_a'])**2
            resampled_error_b = (resampled_df['y_true_b'] - resampled_df['y_pred_b'])**2
            diff = np.sqrt(np.mean(resampled_error_a)) - np.sqrt(np.mean(resampled_error_b))
        
        bootstrap_diffs.append(diff)

    # Calculate confidence interval
    ci_lower = np.percentile(bootstrap_diffs, 2.5)
    ci_upper = np.percentile(bootstrap_diffs, 97.5)
    
    # Calculate two-sided bootstrap sign-based p-value
    bootstrap_diffs_arr = np.array(bootstrap_diffs)
    bootstrap_p = 2 * min(
        np.mean(bootstrap_diffs_arr <= 0),
        np.mean(bootstrap_diffs_arr >= 0)
    )
    bootstrap_p = min(1.0, bootstrap_p)

    return {
        'observed_difference': observed_diff,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'bootstrap_p': bootstrap_p
    }


def run_all_comparisons(
    predictions_df: pd.DataFrame,
    comparisons: List[Tuple[str, str]],
    metrics: List[str] = ['mae', 'rmse'],
    **kwargs
) -> pd.DataFrame:
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

def compute_model_confidence_intervals(
    predictions_df: pd.DataFrame,
    n_bootstraps: int = 5000,
    random_seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Computes bootstrap confidence intervals for individual model performance metrics
    using a cluster bootstrap (if athlete_id is present) or standard bootstrap.
    """
    rng = np.random.default_rng(random_seed)
    results = []
    
    # Check if athlete_id is available for cluster bootstrap
    has_athlete_id = 'athlete_id' in predictions_df.columns
    
    for model_name, group in predictions_df.groupby('model'):
        mask = ~np.isnan(group['y_pred']) & ~np.isnan(group['y_true'])
        valid_group = group[mask]
        
        if valid_group.empty:
            continue
            
        y_true_all = valid_group['y_true'].values
        y_pred_all = valid_group['y_pred'].values
        
        # Calculate point estimates
        point_estimates = {
            'mae': mean_absolute_error(y_true_all, y_pred_all),
            'rmse': np.sqrt(mean_squared_error(y_true_all, y_pred_all)),
            'r2': r2_score(y_true_all, y_pred_all) if np.var(y_true_all) > 0 else np.nan,
            'median_absolute_error': np.median(np.abs(y_true_all - y_pred_all)),
            'mean_residual': np.mean(y_pred_all - y_true_all),
            'smape': symmetric_mean_absolute_percentage_error(y_true_all, y_pred_all),
            'nrmse': normalized_root_mean_squared_error(y_true_all, y_pred_all)
        }
        
        bootstrap_metrics = {k: [] for k in point_estimates.keys()}
        
        if has_athlete_id:
            unique_athletes = valid_group['athlete_id'].unique()
            n_athletes = len(unique_athletes)
            grouped_preds = {athlete: df for athlete, df in valid_group.groupby('athlete_id')}
        else:
            n_obs = len(valid_group)
            
        for _ in range(n_bootstraps):
            if has_athlete_id:
                resampled_athlete_ids = rng.choice(unique_athletes, size=n_athletes, replace=True)
                bootstrap_parts = [grouped_preds[athlete] for athlete in resampled_athlete_ids]
                resampled_df = pd.concat(bootstrap_parts, ignore_index=True)
                y_true = resampled_df['y_true'].values
                y_pred = resampled_df['y_pred'].values
            else:
                indices = rng.choice(n_obs, size=n_obs, replace=True)
                y_true = y_true_all[indices]
                y_pred = y_pred_all[indices]
                
            if len(y_true) == 0:
                continue
                
            bootstrap_metrics['mae'].append(mean_absolute_error(y_true, y_pred))
            bootstrap_metrics['rmse'].append(np.sqrt(mean_squared_error(y_true, y_pred)))
            bootstrap_metrics['r2'].append(r2_score(y_true, y_pred) if np.var(y_true) > 0 else np.nan)
            bootstrap_metrics['median_absolute_error'].append(np.median(np.abs(y_true - y_pred)))
            bootstrap_metrics['mean_residual'].append(np.mean(y_pred - y_true))
            bootstrap_metrics['smape'].append(symmetric_mean_absolute_percentage_error(y_true, y_pred))
            bootstrap_metrics['nrmse'].append(normalized_root_mean_squared_error(y_true, y_pred))
            
        for metric_name, point_est in point_estimates.items():
            dist = np.array(bootstrap_metrics[metric_name])
            # Filter out NaNs (e.g. from R2 with no variance)
            dist = dist[~np.isnan(dist)]
            
            if len(dist) > 0:
                ci_lower = np.percentile(dist, 2.5)
                ci_upper = np.percentile(dist, 97.5)
                std_err = np.std(dist)
            else:
                ci_lower = np.nan
                ci_upper = np.nan
                std_err = np.nan
                
            results.append({
                'model': model_name,
                'metric': metric_name,
                'point_estimate': point_est,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'std_error': std_err
            })

    return pd.DataFrame(results)
