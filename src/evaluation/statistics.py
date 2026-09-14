import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.core.logging import get_logger
from src.evaluation.metrics import symmetric_mean_absolute_percentage_error, normalized_root_mean_squared_error

logger = get_logger(__name__)

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

    # Cluster bootstrap resampling.
    #
    # Both supported metrics are pooled functions of per-observation error
    # contributions, so a resample never needs to be materialised. For a drawn
    # multiset of athletes D:
    #
    #     MAE  = sum_{i in D} sum|err_i|  / sum_{i in D} n_i
    #     RMSE = sqrt( sum_{i in D} SSE_i / sum_{i in D} n_i )
    #
    # Pre-aggregating each athlete's error sum and row count therefore reduces
    # every iteration to two gathers and two sums over n_athletes, instead of
    # concatenating n_athletes DataFrames. This is an exact reformulation, not
    # an approximation. The RNG stream is preserved exactly, because
    # rng.choice(k, ...) draws the same underlying integers as
    # rng.choice(ids, ...) for ids of length k, so the same athletes are drawn
    # in the same order. Only floating-point summation order differs; agreement
    # with the previous row-materialising implementation was verified to 1e-12
    # on irregular cluster sizes for both metrics.
    #
    # The previous form rebuilt a full DataFrame on every iteration, which made
    # the production Raw cohort (~49,000 athletes) computationally infeasible.
    athlete_codes, unique_athletes = pd.factorize(merged_preds['athlete_id_a'])
    n_athletes = len(unique_athletes)

    # error_a / error_b already hold the per-row contribution for this metric:
    # absolute error for 'mae', squared error for 'rmse'.
    cluster_sum_a = np.bincount(athlete_codes, weights=np.asarray(error_a, dtype=float), minlength=n_athletes)
    cluster_sum_b = np.bincount(athlete_codes, weights=np.asarray(error_b, dtype=float), minlength=n_athletes)
    cluster_counts = np.bincount(athlete_codes, minlength=n_athletes).astype(float)

    bootstrap_diffs = []

    for _ in range(n_bootstraps):
        # Positions into unique_athletes, drawn with replacement. Multiplicity
        # is carried by the gather below, so a repeated athlete contributes
        # repeatedly -- the property the multiplicity regression test guards.
        draw = rng.choice(n_athletes, size=n_athletes, replace=True)

        total_rows = cluster_counts[draw].sum()
        pooled_a = cluster_sum_a[draw].sum() / total_rows
        pooled_b = cluster_sum_b[draw].sum() / total_rows

        if metric == 'mae':
            diff = pooled_a - pooled_b
        else:  # rmse
            diff = np.sqrt(pooled_a) - np.sqrt(pooled_b)

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

    comparisons_df = pd.DataFrame(results)

    if comparisons_df.empty:
        return comparisons_df

    # Correct for multiplicity. Every pair is tested on two metrics, so a
    # default configuration reports fourteen tests; at alpha = 0.05 the chance
    # of at least one spurious rejection across that family approaches one in
    # two. Raw p-values are retained alongside the adjusted ones so the
    # adjustment is visible rather than applied silently.
    comparisons_df = add_multiplicity_correction(comparisons_df)

    return comparisons_df


def holm_bonferroni(p_values: np.ndarray) -> np.ndarray:
    """
    Holm-Bonferroni step-down adjusted p-values.

    Controls the family-wise error rate without assuming the tests are
    independent, which matters here: the comparisons share a common reference
    model and are computed from the same predictions, so they are strongly
    dependent. Uniformly more powerful than Bonferroni and valid under
    arbitrary dependence, unlike Benjamini-Hochberg in its unmodified form.
    """
    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)

    order = np.argsort(p_values)
    adjusted = np.empty(n, dtype=float)

    running_max = 0.0
    for rank, index in enumerate(order):
        candidate = (n - rank) * p_values[index]
        # Step-down enforces monotonicity: an adjusted p-value may never fall
        # below one assigned to a smaller raw p-value.
        running_max = max(running_max, candidate)
        adjusted[index] = min(1.0, running_max)

    return adjusted


def add_multiplicity_correction(
    comparisons_df: pd.DataFrame,
    alpha: float = 0.05,
    p_column: str = "bootstrap_p",
) -> pd.DataFrame:
    """
    Annotate a comparison table with family-wise error control.

    The family is every test in the table. Adding columns rather than filtering
    keeps the unadjusted values available, since which comparisons were run is
    itself part of what a reader needs in order to judge the adjustment.
    """
    annotated = comparisons_df.copy()

    annotated["p_adjusted_holm"] = holm_bonferroni(annotated[p_column].to_numpy())
    annotated["significant_unadjusted"] = annotated[p_column] < alpha
    annotated["significant_adjusted"] = annotated["p_adjusted_holm"] < alpha
    annotated["n_tests_in_family"] = len(annotated)

    lost = int(
        (annotated["significant_unadjusted"] & ~annotated["significant_adjusted"]).sum()
    )
    if lost:
        logger.info(
            "%d of %d comparisons significant at alpha=%.2f unadjusted no longer "
            "reach significance after Holm correction.",
            lost, len(annotated), alpha,
        )

    return annotated

def _weighted_median(sorted_values: np.ndarray, weights: np.ndarray) -> float:
    """
    Median of a multiset given sorted values and integer multiplicities.

    Reproduces numpy.median semantics exactly, including averaging the two
    central order statistics when the total weight is even, so a cluster
    bootstrap can compute the median of a resample without materialising it.

    Args:
        sorted_values: Values in ascending order.
        weights: Multiplicity of each value, aligned to sorted_values.
            Zero-weight entries are never selected, because a zero weight
            leaves the cumulative total unchanged and the search returns the
            earlier index.
    """
    cumulative = np.cumsum(weights)
    total = cumulative[-1] if len(cumulative) else 0

    if total <= 0:
        return np.nan

    total = int(total)
    if total % 2 == 1:
        return float(sorted_values[np.searchsorted(cumulative, (total + 1) // 2, side="left")])

    lower = sorted_values[np.searchsorted(cumulative, total // 2, side="left")]
    upper = sorted_values[np.searchsorted(cumulative, total // 2 + 1, side="left")]
    return float((lower + upper) / 2.0)


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
            # Cluster bootstrap over athletes, computed from per-athlete
            # aggregates rather than by rebuilding the resampled frame on every
            # iteration. Six of the seven metrics are pooled functions of
            # per-observation quantities and so decompose exactly over clusters;
            # the median does not, and is obtained from a weighted median over
            # the pre-sorted absolute errors. See the note in
            # paired_bootstrap_comparison for why this matters at production
            # scale.
            codes, unique_athletes = pd.factorize(valid_group['athlete_id'])
            n_athletes = len(unique_athletes)

            residual = y_pred_all - y_true_all
            abs_error = np.abs(residual)

            smape_denominator = np.abs(y_true_all) + np.abs(y_pred_all)
            smape_valid = smape_denominator != 0
            smape_terms = np.zeros_like(smape_denominator, dtype=float)
            smape_terms[smape_valid] = (
                200 * abs_error[smape_valid] / smape_denominator[smape_valid]
            )

            def _by_cluster(values=None):
                return np.bincount(
                    codes,
                    weights=None if values is None else np.asarray(values, dtype=float),
                    minlength=n_athletes,
                ).astype(float)

            cluster_n = _by_cluster()
            cluster_abs_error = _by_cluster(abs_error)
            cluster_sq_error = _by_cluster(residual ** 2)
            cluster_residual = _by_cluster(residual)
            cluster_y = _by_cluster(y_true_all)
            cluster_y_squared = _by_cluster(y_true_all ** 2)
            cluster_abs_y = _by_cluster(np.abs(y_true_all))
            cluster_smape_sum = _by_cluster(smape_terms)
            cluster_smape_n = _by_cluster(smape_valid.astype(float))

            median_order = np.argsort(abs_error, kind="stable")
            sorted_abs_error = abs_error[median_order]
            sorted_codes = codes[median_order]
        else:
            n_obs = len(valid_group)

        for _ in range(n_bootstraps):
            if has_athlete_id:
                draw = rng.choice(n_athletes, size=n_athletes, replace=True)
                multiplicity = np.bincount(draw, minlength=n_athletes).astype(float)

                n_resampled = cluster_n @ multiplicity
                if n_resampled == 0:
                    continue

                sum_sq_error = cluster_sq_error @ multiplicity
                sum_y = cluster_y @ multiplicity
                # SST via the computational form: sum(y^2) - (sum y)^2 / n.
                total_sum_squares = (cluster_y_squared @ multiplicity) - (sum_y ** 2) / n_resampled

                rmse = np.sqrt(sum_sq_error / n_resampled)
                mean_abs_y = (cluster_abs_y @ multiplicity) / n_resampled
                smape_n = cluster_smape_n @ multiplicity

                bootstrap_metrics['mae'].append((cluster_abs_error @ multiplicity) / n_resampled)
                bootstrap_metrics['rmse'].append(rmse)
                bootstrap_metrics['r2'].append(
                    1 - sum_sq_error / total_sum_squares if total_sum_squares > 0 else np.nan
                )
                bootstrap_metrics['median_absolute_error'].append(
                    _weighted_median(sorted_abs_error, multiplicity[sorted_codes])
                )
                bootstrap_metrics['mean_residual'].append((cluster_residual @ multiplicity) / n_resampled)
                bootstrap_metrics['smape'].append(
                    (cluster_smape_sum @ multiplicity) / smape_n if smape_n > 0 else 0.0
                )
                bootstrap_metrics['nrmse'].append(rmse / mean_abs_y if mean_abs_y != 0 else np.nan)
                continue

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
