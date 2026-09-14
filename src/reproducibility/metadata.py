from dataclasses import dataclass, asdict
from pathlib import Path
import json
from typing import Dict, Optional

@dataclass(frozen=True)
class EnvironmentMetadata:
    python_version: str
    platform: str
    package_versions: Dict[str, str]

@dataclass(frozen=True)
class ExperimentMetadata:
    timestamp: str
    git_commit: Optional[str]
    random_seed: Optional[int]
    # The dataset file the run consumed (e.g. "openpowerlifting",
    # "test_fixture") and the equipment cohort within it ("raw", "equipped").
    # These are independent: the cohort alone does not identify the source, and
    # recording only the cohort is what previously allowed fixture output to be
    # stamped as production output.
    dataset_source: str
    # Content digest of the input file. OpenPowerlifting revises historical
    # records, so the filename alone does not identify which data produced a
    # result; the digest does.
    dataset_sha256: Optional[str]
    cohort: str
    dataset_rows: int
    dataset_columns: int
    execution_mode: str
    environment: EnvironmentMetadata

def save_metadata(metadata: ExperimentMetadata, path: Path):
    """Saves experiment metadata to a JSON file."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(asdict(metadata), f, ensure_ascii=False, indent=4)
