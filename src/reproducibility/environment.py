import platform
import sys
import subprocess
from datetime import datetime
import importlib.metadata
import pandas as pd
from typing import Optional, Dict
from src.reproducibility.metadata import EnvironmentMetadata, ExperimentMetadata

def get_git_commit() -> Optional[str]:
    """Gracefully attempts to get the current git commit hash."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], 
            stderr=subprocess.DEVNULL, 
            text=True
        ).strip()
        return commit
    except Exception:
        return None

def get_package_versions() -> Dict[str, str]:
    """Captures key package versions."""
    packages = ["pandas", "numpy", "scikit-learn", "scipy", "matplotlib", "pytest"]
    versions = {}
    for pkg in packages:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = "unknown"
    return versions

def capture_environment() -> EnvironmentMetadata:
    """Captures current Python and OS environment."""
    return EnvironmentMetadata(
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        package_versions=get_package_versions()
    )

def capture_experiment_metadata(
    df: pd.DataFrame, 
    dataset_name: str, 
    execution_mode: str, 
    random_seed: Optional[int] = None
) -> ExperimentMetadata:
    """Generates a full metadata object for the current experiment run."""
    return ExperimentMetadata(
        timestamp=datetime.utcnow().isoformat() + "Z",
        git_commit=get_git_commit(),
        random_seed=random_seed,
        dataset_name=dataset_name,
        dataset_rows=len(df),
        dataset_columns=len(df.columns),
        execution_mode=execution_mode,
        environment=capture_environment()
    )
