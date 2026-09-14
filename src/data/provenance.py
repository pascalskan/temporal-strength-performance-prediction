"""
Dataset provenance: identify exactly which snapshot produced a set of results.

OpenPowerlifting is a living database. Meets are added continuously and
historical records are retroactively corrected, so the file behind
`data/openpowerlifting.csv` is not a fixed artefact -- two downloads weeks
apart are different datasets carrying the same name. Without a recorded
fingerprint, published numbers cannot be tied to the data that produced them,
and a reader who re-downloads cannot tell whether a discrepancy is their
mistake or a different snapshot.

This module fingerprints the input and records the fingerprint alongside every
run. The manifest committed at data/dataset_manifest.json states which snapshot
the reported results came from; a mismatch is surfaced loudly but does not stop
the run, because working against a newer snapshot is legitimate as long as it
is disclosed.
"""
import hashlib
import json
from datetime import datetime, UTC
from pathlib import Path
from typing import Optional

import pandas as pd

from src.core.logging import get_logger

logger = get_logger(__name__)

MANIFEST_FILENAME = "dataset_manifest.json"

# 1 MiB chunks: large enough to keep hashing throughput-bound, small enough
# that a 300MB+ dataset never needs to be resident in memory.
_CHUNK_BYTES = 1024 * 1024


def compute_file_digest(path: Path, algorithm: str = "sha256") -> str:
    """Streaming content digest of a file, so dataset size does not bound memory."""
    digest = hashlib.new(algorithm)

    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_BYTES), b""):
            digest.update(chunk)

    return digest.hexdigest()


def describe_dataframe(df: pd.DataFrame, date_column: str = "Date") -> dict:
    """
    Shape and coverage summary.

    Recorded alongside the digest because a digest alone says only "different",
    never "different how". These fields make a snapshot change legible: more
    rows means meets were added, a later end date means the window moved, a
    changed row count at an identical end date means records were revised.
    """
    description = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "column_names": sorted(map(str, df.columns)),
    }

    if date_column in df.columns:
        dates = pd.to_datetime(df[date_column], errors="coerce")
        valid = dates.dropna()
        if not valid.empty:
            description["earliest_record"] = str(valid.min().date())
            description["latest_record"] = str(valid.max().date())
            description["records_with_unparseable_date"] = int(dates.isna().sum())

    if "Equipment" in df.columns:
        counts = df["Equipment"].value_counts()
        description["equipment_counts"] = {str(k): int(v) for k, v in counts.items()}

    return description


def build_manifest(dataset_path: Path, df: Optional[pd.DataFrame] = None) -> dict:
    """Fingerprint a dataset file, optionally including a summary of its contents."""
    dataset_path = Path(dataset_path)

    manifest = {
        "filename": dataset_path.name,
        "size_bytes": dataset_path.stat().st_size,
        "sha256": compute_file_digest(dataset_path),
        "fingerprinted_at": datetime.now(UTC).isoformat(),
    }

    if df is not None:
        manifest["content"] = describe_dataframe(df)

    return manifest


def write_manifest(manifest: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=4, sort_keys=True), encoding="utf-8")
    logger.info("Wrote dataset manifest to %s", path)


def load_manifest(path: Path) -> Optional[dict]:
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def compare_to_manifest(current: dict, recorded: Optional[dict]) -> dict:
    """
    Compare a freshly computed fingerprint against the committed one.

    Returns a result dict with a 'status' of:
        no_manifest -- nothing to compare against
        match       -- same snapshot as the published results
        mismatch    -- a different snapshot; results will not reproduce exactly
    """
    if recorded is None:
        return {"status": "no_manifest", "differences": []}

    if current.get("sha256") == recorded.get("sha256"):
        return {"status": "match", "differences": []}

    differences = []
    for field in ("size_bytes", "sha256"):
        if current.get(field) != recorded.get(field):
            differences.append(
                {"field": field, "recorded": recorded.get(field), "current": current.get(field)}
            )

    current_content = current.get("content", {})
    recorded_content = recorded.get("content", {})
    for field in ("rows", "columns", "earliest_record", "latest_record"):
        if field in current_content or field in recorded_content:
            if current_content.get(field) != recorded_content.get(field):
                differences.append(
                    {
                        "field": f"content.{field}",
                        "recorded": recorded_content.get(field),
                        "current": current_content.get(field),
                    }
                )

    return {"status": "mismatch", "differences": differences}


def report_comparison(comparison: dict, manifest_path: Path) -> None:
    """Log the outcome at a severity matching its consequences for reproducibility."""
    status = comparison["status"]

    if status == "match":
        logger.info(
            "Dataset matches the recorded manifest; results are comparable to "
            "those published from this snapshot."
        )
        return

    if status == "no_manifest":
        logger.warning(
            "No dataset manifest at %s. Results from this run cannot be tied to a "
            "known snapshot. Write one with scripts/record_dataset_manifest.py.",
            manifest_path,
        )
        return

    logger.warning(
        "Dataset DIFFERS from the recorded manifest. OpenPowerlifting adds meets "
        "and revises historical records continuously, so results from this run "
        "will not exactly reproduce those published from the recorded snapshot. "
        "This is expected when working against a newer download; report the "
        "snapshot actually used."
    )
    for difference in comparison["differences"]:
        logger.warning(
            "  %s: recorded=%s current=%s",
            difference["field"], difference["recorded"], difference["current"],
        )
