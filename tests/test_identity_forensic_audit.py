import pandas as pd
import pytest
from pathlib import Path
from src.analysis.identity_forensic_audit import run_identity_audit

@pytest.fixture
def sample_cleaned_data():
    """Creates a synthetic cleaned dataset with predictable identity collision risks."""
    data = {
        # Identity 1: Normal
        'Name': ['A', 'A'], 'Sex': ['M', 'M'], 'Country': ['USA', 'USA'],
        'Date': pd.to_datetime(['2010-01-01', '2011-01-01']),
        'Age': [20, 21], 'BodyweightKg': [80, 81], 'TotalKg': [500, 510],

        # Identity 2: High History (merged)
        'Name': ['B', 'B', 'B'], 'Sex': ['F', 'F', 'F'], 'Country': ['CAN', 'CAN', 'CAN'],
        'Date': pd.to_datetime(['2010-01-01', '2011-01-01', '2012-01-01']),
        'Age': [25, 26, 27], 'BodyweightKg': [60, 61, 62], 'TotalKg': [300, 310, 320],

        # Identity 3: Long Span
        'Name': ['C', 'C'], 'Sex': ['M', 'M'], 'Country': ['USA', 'USA'],
        'Date': pd.to_datetime(['2000-01-01', '2025-01-01']),
        'Age': [20, 45], 'BodyweightKg': [90, 95], 'TotalKg': [600, 650],

        # Identity 4: Physiological Anomaly (BW swing)
        'Name': ['D', 'D'], 'Sex': ['F', 'F'], 'Country': ['USA', 'USA'],
        'Date': pd.to_datetime(['2015-01-01', '2016-01-01']),
        'Age': [30, 31], 'BodyweightKg': [70, 120], 'TotalKg': [400, 450],
        
        # Identity 5: Physiological Anomaly (Age reversal)
        'Name': ['E', 'E'], 'Sex': ['M', 'M'], 'Country': ['USA', 'USA'],
        'Date': pd.to_datetime(['2018-01-01', '2019-01-01']),
        'Age': [40, 39], 'BodyweightKg': [100, 101], 'TotalKg': [700, 710],
        
        # Identity 6: Cross-country ambiguity
        'Name': ['F', 'F'], 'Sex': ['M', 'M'], 'Country': ['USA', 'MEX'],
        'Date': pd.to_datetime(['2020-01-01', '2021-01-01']),
        'Age': [28, 29], 'BodyweightKg': [85, 86], 'TotalKg': [550, 560],
    }
    return pd.DataFrame(data)

def test_run_identity_audit_calculations(sample_cleaned_data, tmp_path: Path):
    """Validates the identity audit's calculations and risk detection."""
    
    run_identity_audit(sample_cleaned_data, tmp_path)
    
    # Verify Population Summary
    pop_summary = pd.read_csv(tmp_path / "identity_population_summary.csv")
    assert pop_summary['total_cleaned_records'].iloc[0] == 13
    # A_M_USA, B_F_CAN, C_M_USA, D_F_USA, E_M_USA, F_M_USA, F_M_MEX -> 7 identities
    assert pop_summary['total_constructed_identities'].iloc[0] == 7
    
    # Verify Disambiguation Comparison
    disambiguation = pd.read_csv(tmp_path / "identity_disambiguation_comparison.csv")
    assert disambiguation[disambiguation['strategy'] == 'Name_only']['unique_identities'].iloc[0] == 6 # F is one name
    assert disambiguation[disambiguation['strategy'] == 'Name_Sex']['unique_identities'].iloc[0] == 6 # F_M is one
    assert disambiguation[disambiguation['strategy'] == 'Name_Sex_Country']['unique_identities'].iloc[0] == 7 # F_M_USA, F_M_MEX
    
    # Verify High History
    high_history = pd.read_csv(tmp_path / "suspicious_high_history_identities.csv")
    assert len(high_history) == 0 # No one has >= 20 comps
    
    # Verify Temporal Span
    long_span = pd.read_csv(tmp_path / "suspicious_temporal_span_identities.csv")
    assert len(long_span) == 1
    assert long_span['athlete_id'].iloc[0] == 'C_M_USA'
    assert long_span['career_span_years'].iloc[0] == pytest.approx(25.0, abs=0.01)
    
    # Verify Physiological Inconsistencies
    phys_anomaly = pd.read_csv(tmp_path / "suspicious_physiological_inconsistencies.csv")
    assert len(phys_anomaly) == 2
    assert 'D_F_USA' in phys_anomaly['athlete_id'].values
    assert 'E_M_USA' in phys_anomaly['athlete_id'].values
    
    # Verify plots were generated
    assert (tmp_path / "identity_competition_count_distribution.png").exists()
    assert (tmp_path / "identity_career_span_distribution.png").exists()
    assert (tmp_path / "identity_disambiguation_comparison.png").exists()
