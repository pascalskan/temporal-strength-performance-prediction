from pathlib import Path

import pandas as pd


def save_forward_metrics(
    base_dir,
    mae_lr,
    rmse_lr,
    r2_lr,
    mae_rf,
    rmse_rf,
    r2_rf,
    mae_gb,
    rmse_gb,
    r2_gb,
    mae_e,
    rmse_e,
    r2_e,
    mae_b,
    rmse_b,
    r2_b
):
    base_dir = Path(base_dir)
    results_df = pd.DataFrame([
        {"Model": "Linear Regression", "MAE": mae_lr, "RMSE": rmse_lr, "R2": r2_lr},
        {"Model": "Random Forest", "MAE": mae_rf, "RMSE": rmse_rf, "R2": r2_rf},
        {"Model": "Gradient Boosting", "MAE": mae_gb, "RMSE": rmse_gb, "R2": r2_gb},
        {"Model": "Epley", "MAE": mae_e, "RMSE": rmse_e, "R2": r2_e},
        {"Model": "Brzycki", "MAE": mae_b, "RMSE": rmse_b, "R2": r2_b},
    ])

    results_df.to_csv(
        base_dir / "metrics.csv",
        index=False
    )


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
