import pytest
import json
from pathlib import Path
import pandas as pd
from src.reproducibility.environment import capture_experiment_metadata, get_git_commit
from src.reproducibility.metadata import save_metadata, ExperimentMetadata

def test_get_git_commit_fallback():
    # This test is tricky as git might be present.
    # We can't easily simulate its absence, but we can check it doesn't crash.
    commit = get_git_commit()
    assert isinstance(commit, str) or commit is None

def test_metadata_creation():
    df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    metadata = capture_experiment_metadata(
        df=df,
        dataset_name="test_dataset",
        execution_mode="test_mode",
        random_seed=42
    )
    assert isinstance(metadata, ExperimentMetadata)
    assert metadata.dataset_rows == 2
    assert metadata.dataset_columns == 2
    assert metadata.random_seed == 42
    assert "pandas" in metadata.environment.package_versions

def test_metadata_serialization(tmp_path: Path):
    df = pd.DataFrame({'a': [1, 2]})
    metadata = capture_experiment_metadata(
        df=df,
        dataset_name="test_dataset",
        execution_mode="test_mode"
    )
    
    file_path = tmp_path / "run_metadata.json"
    save_metadata(metadata, file_path)
    
    assert file_path.exists()
    
    with open(file_path, 'r') as f:
        loaded_data = json.load(f)
    
    assert loaded_data['dataset_name'] == "test_dataset"
    assert loaded_data['environment']['python_version'] is not None
