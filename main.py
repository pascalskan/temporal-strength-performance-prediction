import argparse
from src.data.loader import load_data
from src.data.cleaning import clean_data
from src.core.logging import get_logger
from src.io.paths import ProjectPaths
from src.pipelines.retrospective_pipeline import run_pipeline
from src.pipelines.forward_pipeline import run_forward_pipeline


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

    logger.info("Splitting dataset...")

    df_raw = df[df["Equipment"] == "Raw"]
    df_non_raw = df[df["Equipment"] != "Raw"]

    logger.info("Raw samples: %s", len(df_raw))
    logger.info("Non-Raw (wraps) samples: %s", len(df_non_raw))

    logger.info("Running Retrospective Pipelines...")
    if not df_raw.empty:
        run_pipeline(df_raw, "raw")
    if not df_non_raw.empty:
        run_pipeline(df_non_raw, "wraps")
    
    logger.info("Running Forward Pipelines...")
    if not df_raw.empty:
        run_forward_pipeline(df_raw, "raw")
    if not df_non_raw.empty:
        run_forward_pipeline(df_non_raw, "wraps")

    logger.info("All pipelines complete.")


if __name__ == "__main__":
    main()
