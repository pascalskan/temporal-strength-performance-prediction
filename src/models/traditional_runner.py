from pathlib import Path
import pandas as pd

from src.core.logging import get_logger
from src.models.traditional_models import predict_traditional


logger = get_logger(__name__)


def run_traditional_models(df_model, trad_dir):
    """
    Runs traditional strength prediction models (Epley, Brzycki) and saves the results.

    Args:
        df_model (pd.DataFrame): The dataframe containing the necessary columns for prediction.
        trad_dir (str or Path): The directory to save the prediction results.
    """
    trad_dir = Path(trad_dir)
    trad_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Preparing data for traditional models (aligned with ML dataset)...")
    logger.info("Traditional model dataset size: %s", len(df_model))

    traditional_results = predict_traditional(df_model)

    logger.info("Structuring and saving traditional method results...")

    actual = pd.Series(traditional_results["Actual_Total"], name="Actual")
    epley = pd.Series(traditional_results["Epley_Prediction"], name="Predicted")
    brzycki = pd.Series(traditional_results["Brzycki_Prediction"], name="Predicted")

    epley_df = pd.concat([actual, epley], axis=1)
    brzycki_df = pd.concat([actual, brzycki], axis=1)

    epley_path = trad_dir / "epley_results.csv"
    brzycki_path = trad_dir / "brzycki_results.csv"

    epley_df.to_csv(epley_path, index=False)
    brzycki_df.to_csv(brzycki_path, index=False)

    logger.info("Saved Epley results to %s", epley_path)
    logger.info("Saved Brzycki results to %s", brzycki_path)

    # Log a brief summary of the outputs
    for name, df_check in {"Epley": epley_df, "Brzycki": brzycki_df}.items():
        if not df_check.empty:
            logger.info("%s predictions generated for %d athletes.", name, len(df_check))
        else:
            logger.warning("%s prediction output was empty.", name)

    logger.info("Traditional model processing complete.")

    return traditional_results
