from src.core.logging import get_logger
from src.utils.metrics import compute_metrics
import pandas as pd

logger = get_logger(__name__)


def run_naive_baseline(df_model_no_attempts, yb_test, split_idx):

    logger.info("Running Naive Baseline (Prev_Total)...")

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

# These baselines are defined by specific columns rather than by a feature
# matrix: Persistence *is* the previous total, Rolling Mean *is* the recent
# average. They therefore declare requires_frame and read those columns by name
# from the test frame, like the traditional equations do.
#
# Without this, feature ablation could not run: removing Prev_Total from the
# feature set to test what a model learns without it would also delete the
# Persistence baseline, when what is wanted is the opposite -- an unchanged
# reference line to measure the ablated models against.
class PersistenceBaseline:
    requires_frame = True

    def fit(self, X, y=None):
        return self
    def predict(self, X):
        if 'Prev_Total' not in X.columns:
            raise ValueError("PersistenceBaseline requires 'Prev_Total' feature.")
        return X['Prev_Total']

class RollingMeanBaseline:
    requires_frame = True

    def __init__(self, window=3):
        self.window = window

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        if 'Rolling_Mean_3' not in X.columns:
            raise ValueError("RollingMeanBaseline requires 'Rolling_Mean_3' feature.")
        return X['Rolling_Mean_3']

class DriftBaseline:
    requires_frame = True

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        if 'Prev_Total' not in X.columns or 'Prev_Prev_Total' not in X.columns:
            raise ValueError("DriftBaseline requires 'Prev_Total' and 'Prev_Prev_Total' features.")
        
        # Fill missing Prev_Prev_Total with Prev_Total to avoid NaN propagation
        prev_prev_total = X['Prev_Prev_Total'].fillna(X['Prev_Total'])
        
        return X['Prev_Total'] + (X['Prev_Total'] - prev_prev_total)