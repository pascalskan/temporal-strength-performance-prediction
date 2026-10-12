"""
CLI behaviour for stage and cohort selection.

A full production run takes hours and its stages are independent, so selecting
a subset must be reliable: a wrong stage name should fail loudly rather than
silently running everything, and an unselected stage must not execute.
"""
import pandas as pd
import pytest

import main as main_module


@pytest.fixture
def stub_pipeline(monkeypatch):
    """Replace data loading and every stage with recorders."""
    calls = []

    frame = pd.DataFrame({
        "Equipment": ["Raw", "Raw", "Single-ply"],
        "TotalKg": [400.0, 410.0, 500.0],
        "Age": [25.0, 26.0, 30.0],
        "BodyweightKg": [80.0, 81.0, 100.0],
        "Sex": ["M", "M", "F"],
        "Name": ["A", "A", "B"],
    })

    monkeypatch.setattr(main_module, "load_data", lambda path: frame)
    monkeypatch.setattr(main_module, "clean_data", lambda df: df)

    stages = {}
    for name in ("retrospective", "forward", "walk_forward"):
        def recorder(df, cohort, _name=name):
            calls.append((_name, cohort, len(df)))
        stages[name] = recorder
    monkeypatch.setattr(main_module, "STAGES", stages)

    return calls


def run_cli(monkeypatch, argv):
    monkeypatch.setattr("sys.argv", ["main.py", "--test", *argv])
    main_module.main()


def test_defaults_run_every_stage_and_cohort(stub_pipeline, monkeypatch):
    run_cli(monkeypatch, [])
    assert {c[0] for c in stub_pipeline} == {"retrospective", "forward", "walk_forward"}
    assert {c[1] for c in stub_pipeline} == {"raw", "equipped"}
    assert len(stub_pipeline) == 6


def test_selecting_stages_skips_the_others(stub_pipeline, monkeypatch):
    run_cli(monkeypatch, ["--stages", "retrospective", "forward"])
    assert {c[0] for c in stub_pipeline} == {"retrospective", "forward"}
    assert "walk_forward" not in {c[0] for c in stub_pipeline}


def test_selecting_a_single_cohort_skips_the_other(stub_pipeline, monkeypatch):
    run_cli(monkeypatch, ["--stages", "walk_forward", "--cohorts", "equipped"])
    assert stub_pipeline == [("walk_forward", "equipped", 1)]


def test_stages_run_in_declared_order(stub_pipeline, monkeypatch):
    run_cli(monkeypatch, ["--stages", "walk_forward", "retrospective", "--cohorts", "raw"])
    assert [c[0] for c in stub_pipeline] == ["walk_forward", "retrospective"]


def test_unknown_stage_is_rejected(stub_pipeline, monkeypatch):
    """Must fail loudly; silently falling back to all stages would cost hours."""
    with pytest.raises(SystemExit):
        run_cli(monkeypatch, ["--stages", "retrospectiv"])
    assert stub_pipeline == []


def test_unknown_cohort_is_rejected(stub_pipeline, monkeypatch):
    with pytest.raises(SystemExit):
        run_cli(monkeypatch, ["--cohorts", "wraps"])
    assert stub_pipeline == []


def test_empty_cohort_is_skipped_not_fatal(monkeypatch):
    calls = []
    frame = pd.DataFrame({
        "Equipment": ["Raw"],
        "TotalKg": [400.0],
        "Age": [25.0],
        "BodyweightKg": [80.0],
        "Sex": ["M"],
        "Name": ["A"],
    })
    monkeypatch.setattr(main_module, "load_data", lambda path: frame)
    monkeypatch.setattr(main_module, "clean_data", lambda df: df)
    monkeypatch.setattr(main_module, "STAGES", {
        "retrospective": lambda df, cohort: calls.append((cohort, len(df))),
    })

    run_cli(monkeypatch, ["--stages", "retrospective"])

    # Only raw has records; equipped is skipped without raising.
    assert calls == [("raw", 1)]


def test_one_failing_unit_does_not_abort_the_rest(monkeypatch, capsys):
    """
    A full suite runs unattended for hours over independent units. A failure in
    one must not discard the others' work -- that would be the difference
    between losing twenty minutes and losing a night.
    """
    calls = []

    frame = pd.DataFrame({
        "Equipment": ["Raw", "Single-ply"],
        "TotalKg": [400.0, 500.0],
        "Age": [25.0, 30.0],
        "BodyweightKg": [80.0, 100.0],
        "Sex": ["M", "F"],
        "Name": ["A", "B"],
    })
    monkeypatch.setattr(main_module, "load_data", lambda path: frame)
    monkeypatch.setattr(main_module, "clean_data", lambda df: df)

    def failing(df, cohort):
        calls.append(("failing", cohort))
        raise RuntimeError("simulated stage failure")

    def working(df, cohort):
        calls.append(("working", cohort))

    monkeypatch.setattr(main_module, "STAGES",
                        {"failing": failing, "working": working})

    monkeypatch.setattr("sys.argv",
                        ["main.py", "--test", "--stages", "failing", "working"])
    exit_code = main_module.main()

    # Both cohorts of both stages were attempted despite the first failing.
    assert ("working", "raw") in calls
    assert ("working", "equipped") in calls
    assert len([c for c in calls if c[0] == "failing"]) == 2

    # And the run reports failure rather than looking clean.
    assert exit_code == 1


def test_a_clean_run_exits_zero(monkeypatch):
    frame = pd.DataFrame({
        "Equipment": ["Raw"], "TotalKg": [400.0], "Age": [25.0],
        "BodyweightKg": [80.0], "Sex": ["M"], "Name": ["A"],
    })
    monkeypatch.setattr(main_module, "load_data", lambda path: frame)
    monkeypatch.setattr(main_module, "clean_data", lambda df: df)
    monkeypatch.setattr(main_module, "STAGES", {"ok": lambda df, cohort: None})
    monkeypatch.setattr("sys.argv", ["main.py", "--test", "--stages", "ok"])

    assert main_module.main() == 0
