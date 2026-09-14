import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Callable, List, Tuple

from src.core.logging import get_logger
from src.utils.metrics import compute_metrics
from src.evaluation.metrics import compute_global_forecast_metrics
from src.evaluation.statistics import run_all_comparisons, compute_model_confidence_intervals
from src.evaluation.diagnostics import run_and_save_diagnostics
from src.evaluation.temporal_analysis import run_and_save_temporal_analysis
from src.evaluation.matched_subset import run_and_save_matched_subset
from src.evaluation.walk_forward_subgroups import run_and_save_subgroup_analysis
from src.evaluation.walk_forward_importance import run_and_save_importance
from src.models.baselines import PersistenceBaseline, RollingMeanBaseline, DriftBaseline
from src.data.identity import build_athlete_id

logger = get_logger(__name__)


class WalkForwardEvaluator:
    """
    Implements expanding-window walk-forward validation for temporal forecasting.
    Recomputes features iteratively to prevent temporal leakage.
    """

    # Columns the evaluator itself depends on, independent of feature_cols.
    REQUIRED_COLUMNS = ("Date", "MeetName")

    # Attributes carried onto each prediction so robustness analysis can run
    # from the prediction file alone. Joining back to the source afterwards
    # would have to reproduce the fold construction to align rows, which is
    # both fragile and an opportunity to reintroduce the leakage the fold
    # construction exists to prevent.
    SUBGROUP_ATTRIBUTES = ("Sex", "Age", "BodyweightKg", "Equipment")
    def __init__(
        self, 
        models: Dict[str, Any], 
        baselines: Dict[str, Any], 
        granularity: str = "year", 
        min_train_periods: int = 1
    ):
        self.models = models
        self.baselines = baselines
        self.granularity = granularity
        self.min_train_periods = min_train_periods
        self.metrics_results = []
        self.prediction_results = []
        # Per-fold feature attribution. Captured while each model is still
        # fitted on that fold's training window; recomputing later would
        # require refitting every fold.
        self.importance_results = []

    def _get_time_periods(self, df: pd.DataFrame) -> pd.Series:
        if self.granularity == "year":
            return df["Date"].dt.year
        elif self.granularity == "month":
            return df["Date"].dt.to_period("M")
        else:
            raise ValueError(f"Unsupported granularity: {self.granularity}")

    def evaluate(
        self,
        df_raw: pd.DataFrame,
        feature_engineering_fn: Callable,
        target_fn: Callable,
        feature_cols: List[str]
    ):
        """
        Runs the walk-forward evaluation.
        
        Args:
            df_raw: Raw, unsplit, un-engineered dataframe
            feature_engineering_fn: Function that takes a raw DF and returns an engineered DF
            target_fn: Function that takes an engineered DF and returns DF with target column
            feature_cols: List of features to be passed into models
        """
        df_raw = df_raw.copy()

        # Fail early and legibly. These are consumed deep inside the fold loop
        # (Date for period assignment, MeetName when composing observation_id),
        # where a missing column would otherwise surface as a bare KeyError
        # raised from inside a DataFrame.apply lambda.
        missing = [c for c in self.REQUIRED_COLUMNS if c not in df_raw.columns]
        if missing:
            raise ValueError(
                f"Walk-forward evaluation requires columns {missing}, which are "
                f"absent from the input. Available columns: {sorted(df_raw.columns)}"
            )

        # 1. Ensure correct athlete ID grouping to prevent collision
        df_raw["Athlete_ID"] = build_athlete_id(df_raw)
        
        time_periods = self._get_time_periods(df_raw)
        periods = sorted(time_periods.unique())
        
        if len(periods) < self.min_train_periods + 1:
            logger.warning(
                "Not enough time periods for walk-forward validation (requires %d, found %d).",
                self.min_train_periods + 1, len(periods)
            )
            return

        # Expanding window loop
        for i in range(self.min_train_periods, len(periods)):
            train_periods = periods[:i]
            test_period = periods[i]

            train_mask = time_periods.isin(train_periods)
            test_mask = time_periods == test_period

            # Subset RAW data
            df_train_raw = df_raw[train_mask].copy()
            df_test_context_raw = df_raw[train_mask | test_mask].copy()  # Need history to compute test features safely

            # Engineer features dynamically on subsets
            df_train_engineered = feature_engineering_fn(df_train_raw, athlete_col="Athlete_ID")
            df_test_context_engineered = feature_engineering_fn(df_test_context_raw, athlete_col="Athlete_ID")

            # Create target variable shift
            df_train_final = target_fn(df_train_engineered, athlete_col="Athlete_ID")
            df_test_context_final = target_fn(df_test_context_engineered, athlete_col="Athlete_ID")

            # Filter test context down to just the target prediction window
            test_engineered_mask = self._get_time_periods(df_test_context_final) == test_period
            df_test_final = df_test_context_final[test_engineered_mask].copy()

            self._run_iteration(df_train_final, df_test_final, feature_cols, test_period, periods[0], periods[i-1])

    def _run_iteration(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_cols: List[str],
        forecast_window: Any,
        train_start: Any,
        train_end: Any
    ):
        logger.info(f"Walk-Forward Window: Predict {forecast_window} (Train {train_start}-{train_end})")
        
        train_clean = train_df.dropna(subset=feature_cols + ["Target_Total"])
        test_clean = test_df.dropna(subset=feature_cols + ["Target_Total"]).copy()
        
        if train_clean.empty or test_clean.empty:
            logger.info(f"Empty train/test set after feature drop for window {forecast_window}. Skipping.")
            return

        # Create a stable, unique observation ID
        # Combine athlete_id, MeetID, Date, and the original index for robustness
        # The original index (row.name) is stable for a given row from the raw data.
        test_clean['observation_id'] = test_clean.apply(
            lambda row: f"{row['Athlete_ID']}_{row['MeetName']}_{row['Date'].strftime('%Y%m%d')}_{row.name}",
            axis=1
        )

        X_train, y_train = train_clean[feature_cols], train_clean["Target_Total"]
        X_test, y_test = test_clean[feature_cols], test_clean["Target_Total"]
        
        all_estimators = {**self.baselines, **self.models}
        
        for model_name, model in all_estimators.items():
            # Estimators declaring requires_frame read columns outside the
            # forecasting feature set: the traditional equations need raw
            # attempt loads, which are deliberately withheld from the machine
            # learning models. They receive the frame itself; every other
            # estimator sees only feature_cols, so what each model is given
            # stays explicit.
            needs_frame = getattr(model, "requires_frame", False)

            model.fit(train_clean if needs_frame else X_train, y_train)
            preds = pd.Series(
                model.predict(test_clean if needs_frame else X_test),
                index=y_test.index,
            )

            # A traditional equation yields NaN where a competition recorded no
            # usable attempts. Those rows are kept as NaN rather than imputed,
            # so the model is scored only on the subpopulation it can address.
            # Pooled metrics downstream already mask NaN per model.
            scored = preds.notna() & y_test.notna()

            if scored.any():
                metrics = compute_metrics(y_test[scored], preds[scored])

                self.metrics_results.append({
                    "forecast_window": forecast_window,
                    "model": model_name,
                    "MAE": metrics.mae,
                    "RMSE": metrics.rmse,
                    "R2": metrics.r2,
                    "n_samples": int(scored.sum()),
                    # Recorded separately so a model scored on a subset is
                    # visible in the per-window output rather than silently
                    # comparable.
                    "n_eligible": len(y_test),
                })
            else:
                logger.warning(
                    "Model '%s' produced no scorable predictions for window %s; "
                    "no window metrics recorded.",
                    model_name, forecast_window,
                )

            # Predictions are recorded regardless, including the NaN ones. A
            # model absent from a window entirely would understate its own
            # denominator, making coverage look better than it is.

            available_attributes = [
                c for c in self.SUBGROUP_ATTRIBUTES if c in test_clean.columns
            ]

            for idx, pred in zip(y_test.index, preds):
                record = {
                    "observation_id": test_clean.loc[idx, "observation_id"],
                    "athlete_id": test_clean.loc[idx, "Athlete_ID"],
                    "date": test_clean.loc[idx, "Date"],
                    "forecast_window": forecast_window,
                    "y_true": y_test.loc[idx],
                    "y_pred": pred,
                    "model": model_name,
                    "train_window_start": train_start,
                    "train_window_end": train_end,
                }
                for attribute in available_attributes:
                    record[attribute] = test_clean.loc[idx, attribute]

                self.prediction_results.append(record)

            self._record_feature_importance(
                model_name, model, feature_cols, forecast_window
            )
        
    def _record_feature_importance(self, model_name, model, feature_cols, forecast_window):
        """
        Capture what the fitted model attributed to each feature in this fold.

        Tree ensembles expose impurity-based importances; linear models expose
        coefficients, recorded as absolute magnitude so the two are comparable
        in rank. Baselines and the traditional equations have no learned
        parameters and are skipped.

        These are associations under the fitted model, not evidence that a
        feature drives performance. Impurity importance in particular inflates
        high-cardinality and highly correlated features, and the engineered
        features here are strongly correlated with one another by construction.
        """
        estimator = model
        # Unwrap a Pipeline so the scaler is not mistaken for the estimator.
        if hasattr(model, "named_steps"):
            estimator = list(model.named_steps.values())[-1]

        if hasattr(estimator, "feature_importances_"):
            values = np.asarray(estimator.feature_importances_, dtype=float)
            kind = "impurity"
        elif hasattr(estimator, "coef_"):
            values = np.abs(np.asarray(estimator.coef_, dtype=float).ravel())
            kind = "abs_coefficient"
        else:
            return

        if len(values) != len(feature_cols):
            logger.warning(
                "Model '%s' reported %d attributions for %d features; skipping.",
                model_name, len(values), len(feature_cols),
            )
            return

        total = values.sum()
        for feature, value in zip(feature_cols, values):
            self.importance_results.append({
                "forecast_window": forecast_window,
                "model": model_name,
                "feature": feature,
                "attribution": float(value),
                # Normalised so folds and model families are comparable;
                # coefficients and impurities are on unrelated scales.
                "attribution_share": float(value / total) if total > 0 else float("nan"),
                "attribution_kind": kind,
            })

    def save_results(self, output_dir: Path, comparisons: List[Tuple[str, str]]):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if self.prediction_results:
            df_predictions = pd.DataFrame(self.prediction_results)
            df_predictions.to_csv(output_dir / "walk_forward_predictions.csv", index=False)
            logger.info("Saved walk_forward_predictions.csv")
            
            summary = compute_global_forecast_metrics(df_predictions)
            summary.to_csv(output_dir / "walk_forward_summary.csv", index=False)
            logger.info("Saved walk_forward_summary.csv")

            comparison_results = run_all_comparisons(df_predictions, comparisons)
            comparison_results.to_csv(output_dir / "walk_forward_comparison.csv", index=False)
            logger.info("Saved walk_forward_comparison.csv")

            ci_results = compute_model_confidence_intervals(df_predictions)
            ci_results.to_csv(output_dir / "walk_forward_confidence_intervals.csv", index=False)
            logger.info("Saved walk_forward_confidence_intervals.csv")
            
            diagnostics_dir = output_dir / "diagnostics"
            run_and_save_diagnostics(df_predictions, diagnostics_dir)
            logger.info(f"Saved diagnostic plots and metrics to {diagnostics_dir}")
            
            temporal_dir = output_dir / "temporal_analysis"
            run_and_save_temporal_analysis(df_predictions, temporal_dir)
            logger.info(f"Saved temporal analysis plots and metrics to {temporal_dir}")

            # Like-for-like comparison restricted to observations every model
            # could score. Traditional equations cover only the subpopulation
            # with recorded attempts, so pooled metrics above are not directly
            # comparable across all models.
            run_and_save_matched_subset(df_predictions, output_dir / "matched_subset")

            run_and_save_subgroup_analysis(
                df_predictions, output_dir / "subgroups"
            )

        if self.importance_results:
            importance_dir = output_dir / "feature_importance"
            run_and_save_importance(
                pd.DataFrame(self.importance_results), importance_dir
            )
            
        if self.metrics_results:
            df_metrics = pd.DataFrame(self.metrics_results)
            df_metrics.to_csv(output_dir / "walk_forward_metrics.csv", index=False)
            logger.info("Saved walk_forward_metrics.csv (per-fold metrics)")

        self._validate_outputs(output_dir)

    def _validate_outputs(self, output_dir: Path):
        """Validates that all expected output files were successfully generated."""
        logger.info("Validating walk-forward outputs in %s...", output_dir)
        
        expected_files = {
            "walk_forward_predictions.csv": "Prediction level outputs",
            "walk_forward_summary.csv": "Global pooled metrics",
            "walk_forward_comparison.csv": "Bootstrap model comparisons",
            "walk_forward_confidence_intervals.csv": "Model confidence intervals",
            "diagnostics/diagnostic_summary.csv": "Scalar diagnostics",
            "diagnostics/decile_performance.csv": "Decile performance metrics",
            "temporal_analysis/temporal_metrics.csv": "Temporal metrics",
            "temporal_analysis/temporal_drift_summary.csv": "Temporal drift summary"
        }
        
        missing_files = []
        for file_path, description in expected_files.items():
            full_path = output_dir / file_path
            if full_path.exists():
                try:
                    df = pd.read_csv(full_path)
                    logger.info("Found %s (%s): %d rows", file_path, description, len(df))
                except Exception as e:
                    logger.error("Found %s but failed to read: %s", file_path, e)
                    missing_files.append(file_path)
            else:
                logger.error("Missing expected output: %s (%s)", file_path, description)
                missing_files.append(file_path)
                
        if missing_files:
            raise FileNotFoundError(
                f"Walk-forward evaluation failed to generate {len(missing_files)} expected outputs: {missing_files}"
            )
            
        logger.info("All expected walk-forward outputs successfully validated.")
