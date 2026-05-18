from src.core.logging import get_logger
from src.utils.metrics import compute_metrics


logger = get_logger(__name__)


def run_naive_baseline(df_model_no_attempts, yb_test, split_idx):

    logger.info("🔹 Running Naive Baseline (Prev_Total)...")

    test_df = df_model_no_attempts[
        df_model_no_attempts["Date"] > split_idx
    ]

    prev_total_test = test_df["Prev_Total"]

    y_true = yb_test.loc[prev_total_test.index]
    y_pred = prev_total_test.copy()

    valid_idx = y_pred.notna()

    y_true = y_true.loc[valid_idx]
    y_pred = y_pred.loc[valid_idx]

    metrics = compute_metrics(y_true, y_pred)

    results = {
        "MAE": metrics.mae,
        "RMSE": metrics.rmse,
        "R2": metrics.r2
    }

    logger.info("Naive Baseline: R2=%.4f", metrics.r2)

    return results
