from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from src.core.logging import get_logger
from src.utils.splitting import time_aware_split
from src.utils.scaling import scale_features
from src.evaluation.metrics import compute_basic_metrics
from src.config.constants import (
    RF_DEFAULT_PARAMS,
    GB_DEFAULT_PARAMS,
    MIN_ANALYSIS_ROWS,
    SMALL_DATASET_THRESHOLD
)
from src.io.paths import ProjectPaths


logger = get_logger(__name__)


def run_lift_specific_models(df, feature_cols, base_dir):
    """
    Trains and evaluates models for each specific lift (Squat, Bench, Deadlift).
    """
    logger.info("Starting lift-specific model analysis...")

    lift_targets = {
        "Squat": "Best3SquatKg",
        "Bench": "Best3BenchKg",
        "Deadlift": "Best3DeadliftKg"
    }
    results = []

    for lift_name, target_col in lift_targets.items():
        logger.info("Processing lift-specific models for: %s", lift_name)
        df_lift = df.dropna(subset=feature_cols + [target_col])

        if len(df_lift) < MIN_ANALYSIS_ROWS:
            logger.warning("Skipping %s analysis: not enough data (found %d, required %d).", lift_name, len(df_lift), MIN_ANALYSIS_ROWS)
            continue

        if len(df_lift) < SMALL_DATASET_THRESHOLD:
            logger.warning("%s has a small dataset (%d samples), results may be unreliable.", lift_name, len(df_lift))

        X_train, X_test, y_train, y_test, _ = time_aware_split(
            df_lift, feature_cols, target_col=target_col
        )
        X_train_scaled, X_test_scaled, _ = scale_features(X_train, X_test)

        models = {
            "Linear": LinearRegression(),
            "Random Forest": RandomForestRegressor(**RF_DEFAULT_PARAMS),
            "Gradient Boosting": GradientBoostingRegressor(**GB_DEFAULT_PARAMS)
        }

        for model_name, model in models.items():
            if model_name == "Linear":
                model.fit(X_train_scaled, y_train)
                preds = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                preds = model.predict(X_test)

            metrics = compute_basic_metrics(y_test, preds)
            logger.info("%s | %s: R2=%.3f", lift_name, model_name, metrics.r2)
            results.append({
                "Lift": lift_name, "Model": model_name, "MAE": metrics.mae, "RMSE": metrics.rmse, "R2": metrics.r2
            })

    if not results:
        logger.warning("No lift-specific results were generated.")
        return pd.DataFrame()

    df_results = pd.DataFrame(results)
    group_name = Path(base_dir).name
    evaluation_dir = ProjectPaths.retrospective_evaluation_dir(group_name)
    save_path = evaluation_dir / "lift_specific_results.csv"
    df_results.to_csv(save_path, index=False)
    logger.info("Saved lift-specific results to %s", save_path)
    return df_results


def plot_lift_comparison(df, base_dir):
    """
    Generates and saves bar plots comparing model performance for each lift.
    """
    if df.empty or "Lift" not in df.columns:
        logger.warning("No lift-specific results to plot.")
        return

    group_name = Path(base_dir).name
    plot_dir = ProjectPaths.retrospective_evaluation_dir(group_name)

    for lift in df["Lift"].unique():
        subset = df[df["Lift"] == lift]
        plt.figure()
        plt.bar(subset["Model"], subset["R2"])
        plt.title(f"{lift} Prediction Performance (R²)")
        plt.ylabel("R² Score")

        for i, value in enumerate(subset["R2"]):
            plt.text(i, value, f"{value:.2f}", ha="center", va="bottom")

        save_path = plot_dir / f"{lift}_comparison.png"
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        logger.info("Saved %s comparison plot to %s", lift, save_path)
