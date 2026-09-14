from pathlib import Path

import pandas as pd


def save_forward_metrics(base_dir, forward_results, traditional_metrics):
    """
    Write the forward-protocol metrics table.

    Every row carries the number of observations it was scored on. The
    traditional equations cover only competitions with recorded attempts, so
    without that column the table silently compares models measured on
    different populations -- see metrics_matched_subset.csv for the
    like-for-like comparison.
    """
    base_dir = Path(base_dir)

    n_learned = len(forward_results.y_pred_lr)

    rows = [
        {"Model": "Linear Regression",
         "MAE": forward_results.linear_regression.metrics.mae,
         "RMSE": forward_results.linear_regression.metrics.rmse,
         "R2": forward_results.linear_regression.metrics.r2,
         "n_scored": n_learned, "coverage": 1.0},
        {"Model": "Random Forest",
         "MAE": forward_results.random_forest.metrics.mae,
         "RMSE": forward_results.random_forest.metrics.rmse,
         "R2": forward_results.random_forest.metrics.r2,
         "n_scored": n_learned, "coverage": 1.0},
        {"Model": "Gradient Boosting",
         "MAE": forward_results.gradient_boosting.metrics.mae,
         "RMSE": forward_results.gradient_boosting.metrics.rmse,
         "R2": forward_results.gradient_boosting.metrics.r2,
         "n_scored": n_learned, "coverage": 1.0},
    ]

    for name in ("Epley", "Brzycki"):
        entry = traditional_metrics[name]
        rows.append({
            "Model": name,
            "MAE": entry["MAE"],
            "RMSE": entry["RMSE"],
            "R2": entry["R2"],
            "n_scored": entry.get("n_scored"),
            "coverage": entry.get("coverage"),
        })

    pd.DataFrame(rows).to_csv(base_dir / "metrics.csv", index=False)


def save_retrospective_metrics(
    base_dir,
    mae_lr_r,
    rmse_lr_r,
    r2_lr_r,
    mae_rf_r,
    rmse_rf_r,
    r2_rf_r,
    mae_gb_r,
    rmse_gb_r,
    r2_gb_r
):
    base_dir = Path(base_dir)
    retro_df = pd.DataFrame([
        {"Model": "Linear Regression", "MAE": mae_lr_r, "RMSE": rmse_lr_r, "R2": r2_lr_r},
        {"Model": "Random Forest", "MAE": mae_rf_r, "RMSE": rmse_rf_r, "R2": r2_rf_r},
        {"Model": "Gradient Boosting", "MAE": mae_gb_r, "RMSE": rmse_gb_r, "R2": r2_gb_r},
    ])

    retro_df.to_csv(
        base_dir / "metrics_retro.csv",
        index=False
    )


def save_forward_predictions(
    base_dir,
    y_test,
    y_pred_lr,
    y_pred_rf,
    y_pred_gb
):
    base_dir = Path(base_dir)
    predictions_df = pd.DataFrame({
        "Actual": y_test,
        "LR_Pred": y_pred_lr,
        "RF_Pred": y_pred_rf,
        "GB_Pred": y_pred_gb
    })

    predictions_df.to_csv(
        base_dir / "ml_predictions.csv",
        index=False
    )


def save_traditional_predictions(
    base_dir,
    actual_next,
    epley_preds,
    brzycki_preds
):
    base_dir = Path(base_dir)
    trad_save = pd.DataFrame({
        "Actual_Next": actual_next,
        "Epley": epley_preds,
        "Brzycki": brzycki_preds
    })

    trad_save.to_csv(
        base_dir / "traditional_predictions.csv",
        index=False
    )
