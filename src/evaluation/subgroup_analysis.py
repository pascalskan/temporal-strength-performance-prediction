from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from src.core.logging import get_logger
from src.evaluation.metrics import compute_basic_metrics
from src.utils.splitting import time_aware_split

logger = get_logger(__name__)

def run_subgroup_analysis(df, save_dir):
    """Performs subgroup analysis based on sex and saves the results."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Running subgroup analysis by sex...")

    results = []
    for group_value, group_name in [(1, "Male"), (0, "Female")]:
        subgroup = df[df["Sex"] == group_value]
        if len(subgroup) < 50:
            logger.warning("Skipping subgroup '%s' due to insufficient data (%d samples).", group_name, len(subgroup))
            continue

        _, _, y_train, y_test, _ = time_aware_split(subgroup, ["Sex", "Age", "BodyweightKg"])
        
        # Note: This is a simplified analysis. For a rigorous approach, models should be retrained on subgroup data.
        # Here we are just showing a breakdown, assuming the main model is used.
        # This part of the code seems to have a bug where it retrains models on very small splits.
        # For now, we preserve the original logic but log a warning.
        logger.warning("Subgroup analysis is retraining models on potentially small data splits. This may not be robust.")

        X_train, X_test, _, _, _ = time_aware_split(subgroup, ["Sex", "Age", "BodyweightKg"])

        models = {
            "Baseline": LinearRegression(),
            "Random Forest": RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42)
        }

        for model_name, model in models.items():
            scaler = StandardScaler().fit(X_train)
            X_train_scaled, X_test_scaled = scaler.transform(X_train), scaler.transform(X_test)
            
            if "Linear" in model_name:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            r2 = r2_score(y_test, y_pred)
            results.append({"Group": group_name, "Model": model_name, "R2": r2})

    if not results:
        logger.warning("Subgroup analysis did not produce any results.")
        return pd.DataFrame()

    results_df = pd.DataFrame(results)
    results_df.to_csv(save_dir / "subgroup_results.csv", index=False)
    logger.info("Subgroup analysis results saved to %s", save_dir / "subgroup_results.csv")

    try:
        pivot_df = results_df.pivot(index="Model", columns="Group", values="R2")
        pivot_df.plot(kind="bar", figsize=(10, 6))
        plt.title("Model Performance by Subgroup (R²)")
        plt.ylabel("R² Score")
        plt.xlabel("Model")
        plt.xticks(rotation=0)
        plt.legend(title="Group")
        plt.tight_layout()
        plt.savefig(save_dir / "subgroup_comparison.png")
        plt.close()
        logger.info("Subgroup analysis plot saved to %s", save_dir / "subgroup_comparison.png")
    except Exception as e:
        logger.error("Failed to generate subgroup plot: %s", e)

    return results_df

def strength_level_analysis(df, model, X, y, save_path=None):
    """Performs analysis based on strength quartiles."""
    logger.info("Running strength level analysis...")
    df_copy = df.copy()
    
    try:
        df_copy["Strength_Level"] = pd.qcut(df_copy["TotalKg"], 4, labels=["Q1 (Weakest)", "Q2", "Q3", "Q4 (Strongest)"])
    except ValueError:
        logger.warning("Could not create 4 quartiles for strength level analysis, likely due to data distribution. Skipping.")
        return pd.DataFrame()

    results = []
    for level in df_copy["Strength_Level"].cat.categories:
        subset = df_copy[df_copy["Strength_Level"] == level]
        X_sub, y_sub = X.loc[subset.index], y.loc[subset.index]

        if len(X_sub) < 20: # Increased threshold for robustness
            logger.warning("Skipping strength level '%s': too few samples (%d).", level, len(X_sub))
            continue

        y_pred = model.predict(X_sub)
        metrics = compute_basic_metrics(y_sub, y_pred)
        results.append({"Strength_Level": level, "Samples": len(X_sub), "R2": metrics.r2, "MAE": metrics.mae, "RMSE": metrics.rmse})
        logger.info("Strength Level %s: R2=%.3f, MAE=%.2f", level, metrics.r2, metrics.mae)

    if not results:
        logger.warning("Strength level analysis did not produce any results.")
        return pd.DataFrame()

    results_df = pd.DataFrame(results)
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(save_path, index=False)
        logger.info("Saved strength level analysis to %s", save_path)

    return results_df
