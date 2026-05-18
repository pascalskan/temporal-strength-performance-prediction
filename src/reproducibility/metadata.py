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
    dataset_name: str
    dataset_rows: int
    dataset_columns: int
    execution_mode: str
    environment: EnvironmentMetadata

def save_metadata(metadata: ExperimentMetadata, path: Path):
    """Saves experiment metadata to a JSON file."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(asdict(metadata), f, ensure_ascii=False, indent=4)
