import pandas as pd
import pytest
from pathlib import Path
from src.analysis.identity_forensic_audit import run_identity_audit


def _athlete(name, sex, country, dates, ages, bodyweights, totals):
    return pd.DataFrame({
        'Name': name,
        'Sex': sex,
        'Country': country,
        'Date': pd.to_datetime(dates),
        'Age': ages,
        'BodyweightKg': bodyweights,
        'TotalKg': totals,
    })


@pytest.fixture
def sample_cleaned_data():
    """
    Synthetic cleaned dataset with predictable identity-collision risks.

    Built by concatenating per-athlete frames. An earlier version of this
    fixture used a single dict literal with one key per athlete, which silently
    collapsed to the last athlete only, because duplicate keys in a dict literal
    overwrite rather than append.
    """
    return pd.concat([
        # Normal two-competition athlete.
        _athlete('A', 'M', 'USA', ['2010-01-01', '2011-01-01'], [20, 21], [80, 81], [500, 510]),
        # Three competitions, still unremarkable.
        _athlete('B', 'F', 'CAN', ['2010-01-01', '2011-01-01', '2012-01-01'],
                 [25, 26, 27], [60, 61, 62], [300, 310, 320]),
        # 25-year career span.
        _athlete('C', 'M', 'USA', ['2000-01-01', '2025-01-01'], [20, 45], [90, 95], [600, 650]),
        # Implausible bodyweight swing (70 -> 120kg).
        _athlete('D', 'F', 'USA', ['2015-01-01', '2016-01-01'], [30, 31], [70, 120], [400, 450]),
        # Age reversal (40 -> 39) as time advances.
        _athlete('E', 'M', 'USA', ['2018-01-01', '2019-01-01'], [40, 39], [100, 101], [700, 710]),
        # Same person recorded under two countries.
        _athlete('F', 'M', ['USA', 'MEX'], ['2020-01-01', '2021-01-01'], [28, 29], [85, 86], [550, 560]),
    ], ignore_index=True)


def test_run_identity_audit_calculations(sample_cleaned_data, tmp_path: Path):
    run_identity_audit(sample_cleaned_data, tmp_path)

    pop_summary = pd.read_csv(tmp_path / "identity_population_summary.csv")
    assert pop_summary['total_cleaned_records'].iloc[0] == 13
    # Identity key is Name+Sex, so F's country change does not split them.
    assert pop_summary['total_constructed_identities'].iloc[0] == 6

    # The disambiguation comparison is a sensitivity analysis over alternative
    # keys, independent of the key actually used above.
    disambiguation = pd.read_csv(tmp_path / "identity_disambiguation_comparison.csv")
    by_strategy = disambiguation.set_index('strategy')['unique_identities']
    assert by_strategy['Name_only'] == 6
    assert by_strategy['Name_Sex'] == 6
    # Adding Country fragments F into two identities -- the behaviour that
    # disqualified it from the production key.
    assert by_strategy['Name_Sex_Country'] == 7

    high_history = pd.read_csv(tmp_path / "suspicious_high_history_identities.csv")
    assert len(high_history) == 0

    long_span = pd.read_csv(tmp_path / "suspicious_temporal_span_identities.csv")
    assert len(long_span) == 1
    assert long_span['athlete_id'].iloc[0] == 'C_M'
    assert long_span['career_span_years'].iloc[0] == pytest.approx(25.0, abs=0.01)

    phys_anomaly = pd.read_csv(tmp_path / "suspicious_physiological_inconsistencies.csv")
    assert set(phys_anomaly['athlete_id']) == {'D_F', 'E_M'}

    risk_summary = pd.read_csv(tmp_path / "identity_collision_risk_summary.csv")
    reversal_row = risk_summary[risk_summary['risk_heuristic'].str.startswith('Age Reversal')]
    assert reversal_row['suspicious_identity_count'].iloc[0] == 1

    for plot in ("identity_competition_count_distribution.png",
                 "identity_career_span_distribution.png",
                 "identity_disambiguation_comparison.png"):
        assert (tmp_path / plot).exists()


def test_age_reversal_within_tolerance_is_not_flagged(tmp_path: Path):
    """Half-year age approximations must not be reported as collisions."""
    df = _athlete('G', 'M', 'USA', ['2020-01-01', '2021-01-01'],
                  [30.5, 30.0], [90, 91], [500, 510])
    run_identity_audit(df, tmp_path)

    phys_anomaly = pd.read_csv(tmp_path / "suspicious_physiological_inconsistencies.csv")
    assert 'G_M' not in set(phys_anomaly['athlete_id'])
