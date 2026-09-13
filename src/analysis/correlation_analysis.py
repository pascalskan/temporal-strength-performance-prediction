from pathlib import Path
import numpy as np
import pandas as pd
from src.core.logging import get_logger
from itertools import combinations

logger = get_logger(__name__)

def run_importance_correlation_analysis(importance_dfs, model_names, importance_dir):
    """
    Calculates and saves the correlation between all pairs of feature importances.
    """
    importance_dir = Path(importance_dir)
    
    if len(importance_dfs) < 2:
        logger.warning("Cannot calculate feature importance correlation with less than 2 models.")
        return

    # Create a unified dataframe of all importances
    all_importances = pd.DataFrame()
    for df, name in zip(importance_dfs, model_names):
        if all_importances.empty:
            all_importances = df.set_index("Feature")[["Importance"]].rename(columns={"Importance": name})
        else:
            all_importances = all_importances.join(
                df.set_index("Feature")[["Importance"]].rename(columns={"Importance": name}),
                how="outer"
            )
    
    all_importances = all_importances.fillna(0) # Assume 0 importance if a feature is not in a model
    
    # Calculate pairwise correlations
    correlation_matrix = all_importances.corr()
    
    logger.info("Feature importance correlation matrix:\n%s", correlation_matrix)

    consistency_path = importance_dir / "importance_correlation.txt"
    try:
        with open(consistency_path, "w", encoding="utf-8") as f:
            f.write("Feature Importance Correlation Matrix:\n")
            f.write(correlation_matrix.to_string())
        logger.info("Saved importance correlation to %s", consistency_path)
    except IOError as e:
        logger.error("Failed to save importance correlation file: %s", e)

    return correlation_matrix
