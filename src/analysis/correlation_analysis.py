from pathlib import Path
import numpy as np
from src.core.logging import get_logger

logger = get_logger(__name__)

def run_importance_correlation_analysis(rf_df, gb_df, importance_dir):
    """
    Calculates and saves the correlation between Random Forest and Gradient Boosting feature importances.
    """
    importance_dir = Path(importance_dir)
    rf_vals = rf_df.set_index("Feature")["Importance"]
    gb_vals = gb_df.set_index("Feature")["Importance"]

    common_features = rf_vals.index.intersection(gb_vals.index)
    rf_vals = rf_vals.loc[common_features]
    gb_vals = gb_vals.loc[common_features]

    if len(common_features) < 2:
        logger.warning("Cannot calculate feature importance correlation with less than 2 common features.")
        return None

    corr = np.corrcoef(rf_vals, gb_vals)[0, 1]
    logger.info("Feature importance correlation (RF vs GB): %.3f", corr)

    consistency_path = importance_dir / "importance_correlation.txt"
    try:
        with open(consistency_path, "w", encoding="utf-8") as f:
            f.write(f"Correlation between RF and GB feature importance: {corr:.4f}\n")
        logger.info("Saved importance correlation to %s", consistency_path)
    except IOError as e:
        logger.error("Failed to save importance correlation file: %s", e)

    return corr
