"""
Record the fingerprint of the dataset currently on disk.

Run this once after downloading OpenPowerlifting, and again whenever you
deliberately move to a newer snapshot. The manifest it writes is committed, so
anyone reading the results can tell which snapshot produced them and whether
their own download matches.

    python scripts/record_dataset_manifest.py
    python scripts/record_dataset_manifest.py --dataset test_fixture.csv
"""
import argparse

from src.core.logging import get_logger
from src.data.loader import load_data
from src.data.provenance import (
    MANIFEST_FILENAME,
    build_manifest,
    compare_to_manifest,
    load_manifest,
    write_manifest,
)
from src.io.paths import ProjectPaths

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="openpowerlifting.csv",
        help="Dataset filename inside data/ (default: openpowerlifting.csv).",
    )
    parser.add_argument(
        "--source-url",
        default="https://openpowerlifting.gitlab.io/opl-csv/bulk-csv.html",
        help="Where this snapshot was obtained, recorded in the manifest.",
    )
    args = parser.parse_args()

    dataset_path = ProjectPaths.use_dataset(args.dataset)
    if not dataset_path.exists():
        raise SystemExit(f"Dataset not found: {dataset_path}")

    logger.info("Fingerprinting %s (this reads the whole file)...", dataset_path)
    df = load_data(dataset_path)
    manifest = build_manifest(dataset_path, df)
    manifest["source_url"] = args.source_url

    manifest_path = ProjectPaths.data_dir() / MANIFEST_FILENAME
    previous = load_manifest(manifest_path)

    if previous is not None:
        comparison = compare_to_manifest(manifest, previous)
        if comparison["status"] == "match":
            logger.info("Manifest already matches this dataset; nothing to change.")
            return
        logger.warning("Replacing the existing manifest. Differences:")
        for difference in comparison["differences"]:
            logger.warning(
                "  %s: recorded=%s current=%s",
                difference["field"], difference["recorded"], difference["current"],
            )

    write_manifest(manifest, manifest_path)
    content = manifest.get("content", {})
    logger.info(
        "Recorded snapshot: %s rows, %s columns, %s to %s, sha256 %s",
        f"{content.get('rows', 0):,}",
        content.get("columns"),
        content.get("earliest_record"),
        content.get("latest_record"),
        manifest["sha256"][:16] + "...",
    )


if __name__ == "__main__":
    main()
