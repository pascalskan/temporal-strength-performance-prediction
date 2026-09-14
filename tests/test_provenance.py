"""
Dataset provenance.

OpenPowerlifting adds meets and revises historical records continuously, so
the filename does not identify a snapshot. Without a fingerprint recorded with
each run, published numbers cannot be tied to the data that produced them.
"""
import json

import pandas as pd
import pytest

from src.data.provenance import (
    build_manifest,
    compare_to_manifest,
    compute_file_digest,
    describe_dataframe,
    load_manifest,
    write_manifest,
)


@pytest.fixture
def dataset(tmp_path):
    path = tmp_path / "snapshot.csv"
    df = pd.DataFrame({
        "Name": ["A", "B"],
        "Equipment": ["Raw", "Single-ply"],
        "Date": ["2020-01-01", "2021-06-01"],
        "TotalKg": [400.0, 500.0],
    })
    df.to_csv(path, index=False)
    return path, df


def test_digest_is_stable_for_identical_content(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    a.write_text("x,y\n1,2\n", encoding="utf-8")
    b.write_text("x,y\n1,2\n", encoding="utf-8")
    assert compute_file_digest(a) == compute_file_digest(b)


def test_digest_changes_when_a_single_record_is_revised(tmp_path):
    """
    The case this exists for: a retroactive correction to one historical result
    leaves row and column counts identical but is a different dataset.
    """
    before, after = tmp_path / "before.csv", tmp_path / "after.csv"
    before.write_text("Name,TotalKg\nA,400.0\nB,500.0\n", encoding="utf-8")
    after.write_text("Name,TotalKg\nA,402.5\nB,500.0\n", encoding="utf-8")

    assert compute_file_digest(before) != compute_file_digest(after)


def test_description_captures_shape_and_coverage(dataset):
    _, df = dataset
    description = describe_dataframe(df)

    assert description["rows"] == 2
    assert description["columns"] == 4
    assert description["earliest_record"] == "2020-01-01"
    assert description["latest_record"] == "2021-06-01"
    assert description["equipment_counts"] == {"Raw": 1, "Single-ply": 1}


def test_description_tolerates_missing_date_column():
    description = describe_dataframe(pd.DataFrame({"TotalKg": [400.0]}))
    assert description["rows"] == 1
    assert "earliest_record" not in description


def test_unparseable_dates_are_counted_not_dropped_silently():
    df = pd.DataFrame({"Date": ["2020-01-01", "not-a-date"], "TotalKg": [1.0, 2.0]})
    assert describe_dataframe(df)["records_with_unparseable_date"] == 1


def test_manifest_round_trips(dataset, tmp_path):
    path, df = dataset
    manifest = build_manifest(path, df)

    manifest_path = tmp_path / "dataset_manifest.json"
    write_manifest(manifest, manifest_path)

    assert load_manifest(manifest_path) == manifest
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["sha256"] == manifest["sha256"]


def test_missing_manifest_loads_as_none(tmp_path):
    assert load_manifest(tmp_path / "absent.json") is None


def test_identical_snapshot_reports_match(dataset):
    path, df = dataset
    manifest = build_manifest(path, df)
    assert compare_to_manifest(manifest, manifest)["status"] == "match"


def test_absent_manifest_reports_no_manifest(dataset):
    path, df = dataset
    assert compare_to_manifest(build_manifest(path, df), None)["status"] == "no_manifest"


def test_changed_snapshot_reports_mismatch_and_names_the_differences(dataset, tmp_path):
    path, df = dataset
    recorded = build_manifest(path, df)

    # A newer download: one additional meet.
    grown = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    newer_path = tmp_path / "newer.csv"
    grown.to_csv(newer_path, index=False)
    current = build_manifest(newer_path, grown)

    comparison = compare_to_manifest(current, recorded)
    assert comparison["status"] == "mismatch"

    fields = {d["field"] for d in comparison["differences"]}
    assert "sha256" in fields
    assert "content.rows" in fields

    rows = next(d for d in comparison["differences"] if d["field"] == "content.rows")
    assert rows["recorded"] == 2 and rows["current"] == 3


def test_manifest_records_size_and_filename(dataset):
    path, df = dataset
    manifest = build_manifest(path, df)

    assert manifest["filename"] == "snapshot.csv"
    assert manifest["size_bytes"] == path.stat().st_size
    assert len(manifest["sha256"]) == 64


def test_committed_manifest_describes_the_production_dataset():
    """
    The manifest shipped with the repository must identify the snapshot the
    published results came from, so a reader can verify their own download.
    """
    from src.io.paths import ProjectPaths
    from src.data.provenance import MANIFEST_FILENAME

    manifest = load_manifest(ProjectPaths.data_dir() / MANIFEST_FILENAME)
    if manifest is None:
        pytest.skip("no dataset manifest recorded in this checkout")

    assert len(manifest["sha256"]) == 64
    assert manifest["content"]["rows"] > 0
    assert "source_url" in manifest, "manifest must say where the snapshot came from"
