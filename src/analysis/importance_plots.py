from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.core.logging import get_logger


logger = get_logger(__name__)


def save_random_forest_importance_plot(
    rf_model,
    feature_cols,
    importance_dir
):
    importance_dir = Path(importance_dir)
    rf_importance = rf_model.feature_importances_

    rf_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": rf_importance
    }).sort_values(by="Importance", ascending=False)

    rf_df.to_csv(
        importance_dir / "rf_importance.csv",
        index=False
    )

    plt.figure()
    rf_df.head(10).plot(kind="barh", x="Feature", y="Importance")
    plt.gca().invert_yaxis()
    plt.title("Random Forest Feature Importance")
    plt.tight_layout()
    plt.savefig(importance_dir / "rf_importance.png")
    plt.close()

    logger.info("   ✅ Random Forest importance saved")

    return rf_df


def save_gradient_boosting_importance_plot(
    gb_model,
    feature_cols,
    importance_dir
):
    importance_dir = Path(importance_dir)
    gb_importance = gb_model.feature_importances_

    gb_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": gb_importance
    }).sort_values(by="Importance", ascending=False)

    gb_df.to_csv(
        importance_dir / "gb_importance.csv",
        index=False
    )

    plt.figure()
    gb_df.head(10).plot(kind="barh", x="Feature", y="Importance")
    plt.gca().invert_yaxis()
    plt.title("Gradient Boosting Feature Importance")
    plt.tight_layout()
    plt.savefig(importance_dir / "gb_importance.png")
    plt.close()

    logger.info("   ✅ Gradient Boosting importance saved")

    return gb_df
