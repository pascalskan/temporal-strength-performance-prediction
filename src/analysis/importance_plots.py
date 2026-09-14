from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from src.core.logging import get_logger

from src.models.introspection import (
    get_model_attribute,
    supports_feature_importance,
    supports_coefficients,
)

logger = get_logger(__name__)


def save_feature_importance_plot(
    model,
    model_name,
    feature_cols,
    importance_dir
):
    """
    Generates and saves feature importance plots for:
    - tree-based models using feature_importances_
    - linear models using coef_
    """

    importance_dir = Path(importance_dir)

    importances = None

    if supports_feature_importance(model):
        importances = get_model_attribute(
            model,
            "feature_importances_"
        )

    elif supports_coefficients(model):
        importances = get_model_attribute(
            model,
            "coef_"
        )

    if importances is None:
        logger.warning(
            f"No importances available for {model_name}. Skipping."
        )
        return None

    # Linear coefficients may be signed
    importances = np.abs(importances)

    # Ensure 1D
    importances = np.ravel(importances)

    df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": importances
    }).sort_values(
        by="Importance",
        ascending=False
    )

    csv_name = (
        f"{model_name.lower().replace(' ', '_')}_importance.csv"
    )

    png_name = (
        f"{model_name.lower().replace(' ', '_')}_importance.png"
    )

    df.to_csv(
        importance_dir / csv_name,
        index=False
    )

    plt.figure(figsize=(10, 6))

    df.head(10).plot(
        kind="barh",
        x="Feature",
        y="Importance"
    )

    plt.gca().invert_yaxis()

    plt.title(f"{model_name} Feature Importance")

    plt.tight_layout()

    plt.savefig(
        importance_dir / png_name
    )

    plt.close()

    logger.info(
        f"{model_name} importance saved"
    )

    return df


def save_random_forest_importance_plot(
    rf_model,
    feature_cols,
    importance_dir
):
    return save_feature_importance_plot(
        rf_model,
        "Random Forest",
        feature_cols,
        importance_dir
    )


def save_gradient_boosting_importance_plot(
    gb_model,
    feature_cols,
    importance_dir
):
    return save_feature_importance_plot(
        gb_model,
        "Gradient Boosting",
        feature_cols,
        importance_dir
    )


def save_ridge_importance_plot(
    ridge_model,
    feature_cols,
    importance_dir
):
    return save_feature_importance_plot(
        ridge_model,
        "Ridge",
        feature_cols,
        importance_dir
    )