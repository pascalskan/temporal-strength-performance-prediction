from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.core.logging import get_logger
from src.evaluation.metrics import compute_basic_metrics
from src.utils.splitting import time_aware_split


logger = get_logger(__name__)


# =========================
# FEATURE SET ANALYSIS
# =========================
def feature_set_comparison(df, feature_cols, save_dir):
    save_dir = Path(save_dir)

    logger.info("--- FEATURE SET COMPARISON ---")

    # -------------------------
    # BASIC FEATURES
    # -------------------------
    basic_features = ["Sex", "Age", "BodyweightKg"]

    X_train_b, X_test_b, y_train, y_test, _ = time_aware_split(
        df,
        basic_features
    )

    X_train_e, X_test_e, _, _, _ = time_aware_split(
        df,
        feature_cols
    )

    # -------------------------
    # SAFETY CHECK (ADD HERE)
    # -------------------------
    # -------------------------
    # SAFETY CHECK
    # -------------------------
    if len(X_train_b) < 50 or len(X_test_b) < 50:
        logger.warning("⚠️ Not enough data for feature comparison")

        results = pd.DataFrame([{
            "Feature_Set": "Skipped",
            "MAE": np.nan,
            "RMSE": np.nan,
            "R2": np.nan,
            "Reason": "Insufficient data"
        }])

        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / "feature_set_comparison.csv"
        results.to_csv(save_path, index=False)

        return results

    # -------------------------
    # TRAIN MODELS
    # -------------------------
    rf_basic = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
    rf_eng = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)

    rf_basic.fit(X_train_b, y_train)
    rf_eng.fit(X_train_e, y_train)

    # -------------------------
    # PREDICT
    # -------------------------
    y_pred_basic = rf_basic.predict(X_test_b)
    y_pred_eng = rf_eng.predict(X_test_e)

    metrics_b = compute_basic_metrics(y_test, y_pred_basic)
    metrics_e = compute_basic_metrics(y_test, y_pred_eng)

    logger.info("Basic Features → R2=%.3f", metrics_b.r2)
    logger.info("Engineered Features → R2=%.3f", metrics_e.r2)

    # -------------------------
    # SAVE RESULTS
    # -------------------------
    results = pd.DataFrame([
        {"Feature_Set": "Basic", "MAE": metrics_b.mae, "RMSE": metrics_b.rmse, "R2": metrics_b.r2},
        {"Feature_Set": "Engineered", "MAE": metrics_e.mae, "RMSE": metrics_e.rmse, "R2": metrics_e.r2}
    ])

    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "feature_set_comparison.csv"
    results.to_csv(save_path, index=False)

    logger.info("✅ Saved feature comparison to %s", save_path)

    plt.figure()
    bars = plt.bar(results["Feature_Set"], results["R2"])

    plt.title("Feature Set Comparison (R²)")
    plt.ylabel("R² Score")

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.2f}",
            ha='center',
            va='bottom'
        )

    plt.tight_layout()

    plot_path = save_dir / "feature_set_comparison.png"
    plt.savefig(plot_path)
    plt.close()

    logger.info("✅ Saved feature comparison plot to %s", plot_path)

    return results
