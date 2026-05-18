import pandas as pd
import numpy as np
from tqdm import tqdm

# -------------------------------
# RPE → Estimated Reps Mapping
# -------------------------------
def rpe_to_reps(rpe):
    """
    Convert RPE to estimated reps.
    """
    if rpe >= 9.5:
        return 1
    elif rpe >= 8.5:
        return 2.5
    elif rpe >= 7:
        return 4
    else:
        return 5  # fallback (rarely used)


# -------------------------------
# Epley Formula
# -------------------------------
def epley(weight, reps):
    return weight * (1 + reps / 30.0)


# -------------------------------
# Brzycki Formula
# -------------------------------
def brzycki(weight, reps):
    if reps >= 10:
        reps = 10  # avoid division issues
    return weight / (1.0278 - 0.0278 * reps)


# -------------------------------
# Estimate 1RM from attempts
# -------------------------------
def estimate_1rm_from_attempts(a1, a2, a3):

    attempts = [
        (a1, 7),
        (a2, 8.5),
        (a3, 9.5)
    ]

    epley_estimates = []
    brzycki_estimates = []

    for weight, rpe in attempts:
        if pd.isna(weight) or weight <= 0:
            continue

        reps = rpe_to_reps(rpe)

        epley_estimates.append(epley(weight, reps))
        brzycki_estimates.append(brzycki(weight, reps))

    # 🚨 FIX: fallback if no attempts available
    if len(epley_estimates) == 0:
        return np.nan, np.nan

    # 🚨 FIX: use max instead of mean when few values
    if len(epley_estimates) == 1:
        return epley_estimates[0], brzycki_estimates[0]

    return np.mean(epley_estimates), np.mean(brzycki_estimates)

# -------------------------------
# Main Prediction Function
# -------------------------------
def predict_traditional(df):
    """
    Predict total using traditional methods with attempts
    """

    results = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Traditional Predictions"):
        # -------------------------
        # Squat
        # -------------------------
        squat_epley, squat_brzycki = estimate_1rm_from_attempts(
            row.get("Squat1Kg", np.nan),
            row.get("Squat2Kg", np.nan),
            row.get("Squat3Kg", np.nan)
        )

        # -------------------------
        # Bench
        # -------------------------
        bench_epley, bench_brzycki = estimate_1rm_from_attempts(
            row.get("Bench1Kg", np.nan),
            row.get("Bench2Kg", np.nan),
            row.get("Bench3Kg", np.nan)
        )

        # -------------------------
        # Deadlift
        # -------------------------
        deadlift_epley, deadlift_brzycki = estimate_1rm_from_attempts(
            row.get("Deadlift1Kg", np.nan),
            row.get("Deadlift2Kg", np.nan),
            row.get("Deadlift3Kg", np.nan)
        )

        # Totals
        total_epley = squat_epley + bench_epley + deadlift_epley
        total_brzycki = squat_brzycki + bench_brzycki + deadlift_brzycki

        # 🚨 FIX: skip invalid rows
        if np.isnan(total_epley) or np.isnan(total_brzycki):
            continue

        results.append({
            "Actual_Total": row["TotalKg"],
            "Epley_Prediction": total_epley,
            "Brzycki_Prediction": total_brzycki
        })

    return pd.DataFrame(results)