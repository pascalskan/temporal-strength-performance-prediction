import argparse
import pandas as pd
from src.data.loader import load_data
from src.data.cleaning import clean_data
from src.core.logging import get_logger
from src.io.paths import ProjectPaths
from src.pipelines.retrospective_pipeline import run_pipeline
from src.pipelines.forward_pipeline import run_forward_pipeline
from src.pipelines.walk_forward_pipeline import run_walk_forward_pipeline


logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run the strength performance prediction pipeline.")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode using the test fixture dataset.",
    )
    args = parser.parse_args()

    if args.test:
        logger.info("TEST MODE")
        dataset_filename = "test_fixture.csv"
    else:
        logger.info("PRODUCTION MODE")
        dataset_filename = "openpowerlifting.csv"

    dataset_path = ProjectPaths.dataset_path(dataset_filename)
    logger.info("Loading dataset from: %s", dataset_path)
    
    df = load_data(dataset_path)

    logger.info("Cleaning data...")
    df = clean_data(df)

    logger.info("Splitting dataset into 'raw' and 'equipped' cohorts...")

    df_raw = df[df["Equipment"] == "Raw"]
    
    # SCIENTIFIC FIX: Merge 'Wraps' and 'Single-ply' into a single 'equipped' cohort.
    # The wraps-only subgroup lacked sufficient longitudinal observations for valid walk-forward forecasting.
    # To preserve temporal validity while maintaining equipment-specific analysis, wraps observations 
    # were merged with single-ply into a unified equipped cohort.
    df_equipped = df[df["Equipment"].isin(["Wraps", "Single-ply"])].copy()
    
    logger.info("Raw samples: %d", len(df_raw))
    logger.info("Equipped (Single-ply + Wraps) samples: %d", len(df_equipped))

    logger.info("Running Retrospective Pipelines...")
    if not df_raw.empty:
        run_pipeline(df_raw, "raw")
    if not df_equipped.empty:
        run_pipeline(df_equipped, "equipped")
    
    logger.info("Running Forward Pipelines...")
    if not df_raw.empty:
        run_forward_pipeline(df_raw, "raw")
    if not df_equipped.empty:
        run_forward_pipeline(df_equipped, "equipped")

    logger.info("Running Walk-Forward Pipelines...")
    if not df_raw.empty:
        run_walk_forward_pipeline(df_raw, "raw")
    if not df_equipped.empty:
        run_walk_forward_pipeline(df_equipped, "equipped")

    logger.info("All pipelines complete.")


if __name__ == "__main__":
    main()
