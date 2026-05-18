from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd


def walk_forward_validate(model, X, y, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)

    metrics = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        metrics.append({
            "fold": fold + 1,
            "mae": mean_absolute_error(y_test, preds),
            "rmse": np.sqrt(mean_squared_error(y_test, preds)),
            "r2": r2_score(y_test, preds)
        })

    return pd.DataFrame(metrics)