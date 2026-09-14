from pathlib import Path
import pandas as pd

from src.core.logging import get_logger
from src.evaluation.metrics import compute_basic_metrics
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

    # Emit metrics, not only raw prediction pairs. Storing Actual/Predicted
    # alone meant every reported traditional figure -- including the Brzycki
    # R2 of 0.99 that anchors the reconstruction argument -- had to be derived
    # by hand afterwards, leaving the project's most striking number without
    # traceable provenance.
    metrics_rows = []
    for name, frame in (("Epley", epley_df), ("Brzycki", brzycki_df)):
        usable = frame["Actual"].notna() & frame["Predicted"].notna()
        n_scored = int(usable.sum())

        if n_scored == 0:
            logger.warning("%s produced no scorable predictions.", name)
            continue

        metrics = compute_basic_metrics(
            frame.loc[usable, "Actual"].values,
            frame.loc[usable, "Predicted"].values,
        )
        metrics_rows.append({
            "Model": name,
            "MAE": metrics.mae,
            "RMSE": metrics.rmse,
            "R2": metrics.r2,
            "n_scored": n_scored,
            "n_eligible": len(frame),
            # These equations need recorded attempts, so they describe a
            # subpopulation. Reported here so the restriction travels with the
            # numbers rather than being rediscovered later.
            "coverage": n_scored / len(frame) if len(frame) else 0.0,
        })
        logger.info(
            "%s (retrospective): MAE=%.2f RMSE=%.2f R2=%.4f on %d of %d observations",
            name, metrics.mae, metrics.rmse, metrics.r2, n_scored, len(frame),
        )

    if metrics_rows:
        metrics_path = trad_dir / "traditional_metrics.csv"
        pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)
        logger.info("Saved traditional metrics to %s", metrics_path)

    logger.info("Traditional model processing complete.")

    return traditional_results
