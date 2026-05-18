import numpy as np
import pandas as pd

from src.core.logging import get_logger
from src.evaluation.metrics import compute_basic_metrics


logger = get_logger(__name__)


def adjust_rpe(rpe):
    return max(6, rpe - 0.5)


def estimate_1rm_forward(weight, rpe, formula="epley"):
    if pd.isna(weight) or weight <= 0:
        return np.nan

    adj_rpe = adjust_rpe(rpe)

    if adj_rpe >= 9.5:
        reps = 1
    elif adj_rpe >= 8.5:
        reps = 2.5
    elif adj_rpe >= 7:
        reps = 4
    else:
        reps = 5

    if formula == "epley":
        return weight * (1 + reps / 30.0)

    reps = min(reps, 10)
    return weight / (1.0278 - 0.0278 * reps)


def run_forward_traditional_predictions(df_model, test_indices):
    epley_preds = []
    brzycki_preds = []
    actual_next = []

    for _, row in df_model.loc[test_indices].iterrows():

        squat_vals = [
            estimate_1rm_forward(row.get("Squat1Kg"), 7, "epley"),
            estimate_1rm_forward(row.get("Squat2Kg"), 8.5, "epley"),
            estimate_1rm_forward(row.get("Squat3Kg"), 9.5, "epley"),
        ]
        squat_vals = [v for v in squat_vals if not np.isnan(v)]
        squat_epley = np.mean(squat_vals) if squat_vals else np.nan

        squat_vals_b = [
            estimate_1rm_forward(row.get("Squat1Kg"), 7, "brzycki"),
            estimate_1rm_forward(row.get("Squat2Kg"), 8.5, "brzycki"),
            estimate_1rm_forward(row.get("Squat3Kg"), 9.5, "brzycki"),
        ]
        squat_vals_b = [v for v in squat_vals_b if not np.isnan(v)]
        squat_brzycki = np.mean(squat_vals_b) if squat_vals_b else np.nan

        bench_vals = [
            estimate_1rm_forward(row.get("Bench1Kg"), 7, "epley"),
            estimate_1rm_forward(row.get("Bench2Kg"), 8.5, "epley"),
            estimate_1rm_forward(row.get("Bench3Kg"), 9.5, "epley"),
        ]
        bench_vals = [v for v in bench_vals if not np.isnan(v)]
        bench_epley = np.mean(bench_vals) if bench_vals else np.nan

        bench_vals_b = [
            estimate_1rm_forward(row.get("Bench1Kg"), 7, "brzycki"),
            estimate_1rm_forward(row.get("Bench2Kg"), 8.5, "brzycki"),
            estimate_1rm_forward(row.get("Bench3Kg"), 9.5, "brzycki"),
        ]
        bench_vals_b = [v for v in bench_vals_b if not np.isnan(v)]
        bench_brzycki = np.mean(bench_vals_b) if bench_vals_b else np.nan

        dead_vals = [
            estimate_1rm_forward(row.get("Deadlift1Kg"), 7, "epley"),
            estimate_1rm_forward(row.get("Deadlift2Kg"), 8.5, "epley"),
            estimate_1rm_forward(row.get("Deadlift3Kg"), 9.5, "epley"),
        ]
        dead_vals = [v for v in dead_vals if not np.isnan(v)]
        dead_epley = np.mean(dead_vals) if dead_vals else np.nan

        dead_vals_b = [
            estimate_1rm_forward(row.get("Deadlift1Kg"), 7, "brzycki"),
            estimate_1rm_forward(row.get("Deadlift2Kg"), 8.5, "brzycki"),
            estimate_1rm_forward(row.get("Deadlift3Kg"), 9.5, "brzycki"),
        ]
        dead_vals_b = [v for v in dead_vals_b if not np.isnan(v)]
        dead_brzycki = np.mean(dead_vals_b) if dead_vals_b else np.nan

        total_epley = squat_epley + bench_epley + dead_epley
        total_brzycki = squat_brzycki + bench_brzycki + dead_brzycki

        if not np.isnan(total_epley) and not np.isnan(total_brzycki):
            epley_preds.append(total_epley)
            brzycki_preds.append(total_brzycki)
            actual_next.append(row["Target_Total"])

    actual_next = np.array(actual_next)
    epley_preds = np.array(epley_preds)
    brzycki_preds = np.array(brzycki_preds)

    metrics_e = compute_basic_metrics(actual_next, epley_preds)
    metrics_b = compute_basic_metrics(actual_next, brzycki_preds)

    logger.info("Epley (Forward Improved): MAE=%.2f, RMSE=%.2f, R2=%.3f", metrics_e.mae, metrics_e.rmse, metrics_e.r2)
    logger.info("Brzycki (Forward Improved): MAE=%.2f, RMSE=%.2f, R2=%.3f", metrics_b.mae, metrics_b.rmse, metrics_b.r2)

    metrics = {
        "Epley": {
            "MAE": metrics_e.mae,
            "RMSE": metrics_e.rmse,
            "R2": metrics_e.r2
        },
        "Brzycki": {
            "MAE": metrics_b.mae,
            "RMSE": metrics_b.rmse,
            "R2": metrics_b.r2
        }
    }

    return actual_next, epley_preds, brzycki_preds, metrics
