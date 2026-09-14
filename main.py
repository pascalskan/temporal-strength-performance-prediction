import argparse

from src.core.logging import get_logger
from src.data.cleaning import clean_data, split_equipment_cohorts
from src.data.loader import load_data
from src.data.provenance import (
    MANIFEST_FILENAME,
    build_manifest,
    compare_to_manifest,
    load_manifest,
    report_comparison,
)
from src.io.paths import ProjectPaths
from src.pipelines.ablation_pipeline import run_ablation_pipeline
from src.pipelines.forward_pipeline import run_forward_pipeline
from src.pipelines.retrospective_pipeline import run_pipeline
from src.pipelines.walk_forward_pipeline import run_walk_forward_pipeline

logger = get_logger(__name__)

# Evaluation protocols, in the order they are reported. Each maps to the
# function that runs it for a single cohort.
STAGES = {
    "retrospective": run_pipeline,
    "forward": run_forward_pipeline,
    "walk_forward": run_walk_forward_pipeline,
    # Costs one full walk-forward run per feature set, so it is excluded from
    # DEFAULT_STAGES and must be requested explicitly.
    "ablation": run_ablation_pipeline,
}

# Ablation is omitted: at roughly four times the walk-forward runtime it should
# be a deliberate choice, not something a routine run pays for.
DEFAULT_STAGES = ["retrospective", "forward", "walk_forward"]

COHORTS = ("raw", "equipped")


def main():
    parser = argparse.ArgumentParser(
        description="Run the strength performance prediction pipeline.",
        epilog=(
            "Stages and cohorts are selectable because a full production run takes\n"
            "hours and its parts are independent. Walk-forward alone took ~50 min for\n"
            "raw and ~2h for equipped, so regenerating one protocol should not require\n"
            "recomputing the others. Examples:\n"
            "  python main.py --stages retrospective forward\n"
            "  python main.py --stages walk_forward --cohorts equipped"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode using the deterministic test fixture dataset.",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=list(STAGES),
        default=list(DEFAULT_STAGES),
        metavar="STAGE",
        help=(
            f"Stages to run: {', '.join(STAGES)} "
            f"(default: {', '.join(DEFAULT_STAGES)}; ablation is costly and opt-in)."
        ),
    )
    parser.add_argument(
        "--cohorts",
        nargs="+",
        choices=COHORTS,
        default=list(COHORTS),
        metavar="COHORT",
        help=f"Equipment cohorts to run: {', '.join(COHORTS)} (default: all).",
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

    # Fingerprint the input before touching it. OpenPowerlifting is a living
    # database, so a filename does not identify a snapshot; results are only
    # reproducible against the data that produced them.
    manifest = build_manifest(dataset_path, df)
    ProjectPaths.set_dataset_digest(manifest["sha256"])
    manifest_path = ProjectPaths.data_dir() / MANIFEST_FILENAME
    report_comparison(
        compare_to_manifest(manifest, load_manifest(manifest_path)),
        manifest_path,
    )

    logger.info("Cleaning data...")
    df = clean_data(df)

    logger.info("Splitting dataset into 'raw' and 'equipped' cohorts...")
    df_raw, df_equipped = split_equipment_cohorts(df)

    # The equipped cohort is Single-ply in all but name (Wraps contributes
    # ~0.12% of records). See src/config/constants.py for composition.
    logger.info("Raw samples: %d", len(df_raw))
    logger.info("Equipped samples: %d", len(df_equipped))

    cohort_frames = {"raw": df_raw, "equipped": df_equipped}

    logger.info(
        "Running stages [%s] for cohorts [%s]",
        ", ".join(args.stages),
        ", ".join(args.cohorts),
    )

    for stage_name in args.stages:
        run_stage = STAGES[stage_name]
        logger.info("=== Stage: %s ===", stage_name)

        for cohort_name in args.cohorts:
            cohort_df = cohort_frames[cohort_name]

            if cohort_df.empty:
                logger.warning(
                    "Cohort '%s' is empty; skipping %s.", cohort_name, stage_name
                )
                continue

            logger.info("Running %s for cohort '%s'...", stage_name, cohort_name)
            run_stage(cohort_df, cohort_name)

    logger.info("All requested pipelines complete.")


if __name__ == "__main__":
    main()
