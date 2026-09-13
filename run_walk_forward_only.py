"""
Minimal execution entrypoint for running ONLY the walk-forward evaluation pipeline.
"""
import argparse
from src.data.loader import load_data
from src.data.cleaning import clean_data, split_equipment_cohorts
from src.core.logging import get_logger
from src.io.paths import ProjectPaths
from src.pipelines.walk_forward_pipeline import run_walk_forward_pipeline

logger = get_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run the walk-forward evaluation pipeline.")
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

    # Scopes results to this dataset, so a --test run cannot land in
    # (or overwrite) the production results tree.
    dataset_path = ProjectPaths.use_dataset(dataset_filename)
    logger.info("Loading dataset from: %s", dataset_path)
    
    df = load_data(dataset_path)

    logger.info("Cleaning data...")
    df = clean_data(df)

    logger.info("Splitting dataset into 'raw' and 'equipped' cohorts...")

    df_raw, df_equipped = split_equipment_cohorts(df)

    # The equipped cohort is Single-ply in all but name (Wraps contributes
    # ~0.12% of records). See src/config/constants.py for composition.
    logger.info("Raw samples: %d", len(df_raw))
    logger.info("Equipped samples: %d", len(df_equipped))

    logger.info("Starting walk-forward evaluation...")
    if not df_raw.empty:
        run_walk_forward_pipeline(df_raw, "raw")
    if not df_equipped.empty:
        run_walk_forward_pipeline(df_equipped, "equipped")

    logger.info("Walk-forward evaluation complete.")

if __name__ == "__main__":
    main()
