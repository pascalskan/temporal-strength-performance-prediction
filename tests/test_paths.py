import pytest

from src.io.paths import ProjectPaths


@pytest.fixture(autouse=True)
def isolated_project_root(tmp_path, monkeypatch):
    """
    Scope is class-level state; leave it as found. Also redirect project_root,
    because the directory helpers mkdir as a side effect and tests must not
    create directories in the real results tree.
    """
    original = ProjectPaths.dataset_scope()
    monkeypatch.setattr(ProjectPaths, "project_root", staticmethod(lambda: tmp_path))
    yield
    ProjectPaths._dataset_scope = original


def test_results_dir_requires_a_declared_dataset():
    """
    Results must not be writable without provenance.

    Previously results_dir() resolved to results/<cohort>, and the cohort
    ("raw", "equipped") is chosen independently of the dataset file. A --test
    run therefore wrote fixture output into the production path and stamped it
    as the raw cohort.
    """
    ProjectPaths.clear_dataset_scope()
    with pytest.raises(RuntimeError, match="No dataset scope declared"):
        ProjectPaths.results_dir()


def test_use_dataset_sets_scope_and_resolves_input():
    path = ProjectPaths.use_dataset("test_fixture.csv")
    assert path.name == "test_fixture.csv"
    assert path.parent == ProjectPaths.data_dir()
    assert ProjectPaths.dataset_scope() == "test_fixture"


def test_fixture_and_production_results_cannot_collide():
    ProjectPaths.use_dataset("test_fixture.csv")
    fixture_raw = ProjectPaths.retrospective_group_dir("raw")

    ProjectPaths.use_dataset("openpowerlifting.csv")
    production_raw = ProjectPaths.retrospective_group_dir("raw")

    assert fixture_raw != production_raw
    assert fixture_raw.name == production_raw.name == "raw"
    assert fixture_raw.parent.name == "test_fixture"
    assert production_raw.parent.name == "openpowerlifting"


def test_every_results_path_sits_under_the_dataset_scope():
    ProjectPaths.use_dataset("test_fixture.csv")
    scope_root = ProjectPaths.results_dir()

    paths = [
        ProjectPaths.retrospective_group_dir("raw"),
        ProjectPaths.retrospective_evaluation_dir("raw"),
        ProjectPaths.retrospective_traditional_dir("raw"),
        ProjectPaths.retrospective_importance_dir("raw"),
        ProjectPaths.forward_group_dir("equipped"),
        ProjectPaths.forward_results_dir("equipped"),
        ProjectPaths.comparison_plots_dir("equipped"),
    ]
    for path in paths:
        assert scope_root in path.parents or path == scope_root, path


def test_set_dataset_scope_rejects_empty_label():
    for bad in ("", None):
        with pytest.raises(ValueError):
            ProjectPaths.set_dataset_scope(bad)
