import pandas as pd
import pytest
from pathlib import Path
from src.analysis.dataset_forensic_audit import run_audit

def test_run_audit_calculations(tmp_path: Path):
    """
    Validates that the forensic audit correctly calculates dataset populations, 
    athlete history distributions, and attrition without double-counting.
    """
    # Create synthetic dataset representing athletes with different history depths
    # Athlete A: 1 comp
    # Athlete B: 2 comps
    # Athlete C: 3 comps
    # Athlete D: 4 comps (Note: 1 raw, 3 equipped -> will still be 4 total before we do equipment splitting, 
    # but cleaning drops NaNs in core fields. Let's make sure none are dropped for testing the grouping logic.)
    
    data = {
        'Name': ['A', 'B', 'B', 'C', 'C', 'C', 'D', 'D', 'D', 'D', 'E_DROP'],
        'Sex': ['M', 'F', 'F', 'M', 'M', 'M', 'F', 'F', 'F', 'F', 'M'],
        'Equipment': ['Raw', 'Raw', 'Raw', 'Raw', 'Raw', 'Raw', 'Wraps', 'Wraps', 'Wraps', 'Wraps', 'Raw'],
        'Age': [20, 25, 26, 30, 31, 32, 22, 23, 24, 25, None], # E_DROP will be dropped due to missing Age
        'BodyweightKg': [80, 60, 61, 90, 91, 92, 70, 71, 72, 73, 85],
        'TotalKg': [500, 300, 310, 600, 610, 620, 400, 410, 420, 430, 400]
    }
    
    raw_df = pd.DataFrame(data)
    
    # Run audit, outputs go to tmp_path
    run_audit(raw_df, tmp_path)
    
    # Verify Population Summary
    pop_summary = pd.read_csv(tmp_path / "dataset_population_summary.csv")
    assert len(pop_summary) == 1
    assert pop_summary.iloc[0]['total_raw_records'] == 11
    assert pop_summary.iloc[0]['total_cleaned_records'] == 10
    assert pop_summary.iloc[0]['records_removed'] == 1
    assert pop_summary.iloc[0]['removal_percentage'] == pytest.approx((1 / 11) * 100)
    
    # Verify Athlete History Distribution
    # Unique athletes in cleaned data: A, B, C, D (4 athletes)
    # A has 1, B has 2, C has 3, D has 4
    history_dist = pd.read_csv(tmp_path / "athlete_history_distribution.csv")
    
    # threshold >= 1: A, B, C, D (4)
    assert history_dist[history_dist['threshold'] == '>=1']['athlete_count'].iloc[0] == 4
    # threshold >= 2: B, C, D (3)
    assert history_dist[history_dist['threshold'] == '>=2']['athlete_count'].iloc[0] == 3
    # threshold >= 3: C, D (2)
    assert history_dist[history_dist['threshold'] == '>=3']['athlete_count'].iloc[0] == 2
    # threshold >= 5: None (0)
    assert history_dist[history_dist['threshold'] == '>=5']['athlete_count'].iloc[0] == 0
    
    # Verify Attrition Analysis
    attrition = pd.read_csv(tmp_path / "feature_eligibility_attrition.csv")
    
    # Baseline (all unique)
    baseline = attrition[attrition['feature_requirement'].str.contains('Baseline')]
    assert baseline['eligible_athletes'].iloc[0] == 4
    
    # Lag 1 (>=2 comps)
    lag1 = attrition[attrition['feature_requirement'].str.contains('Lag 1)', regex=False)]
    assert lag1['eligible_athletes'].iloc[0] == 3
    assert lag1['attrition_from_baseline_percentage'].iloc[0] == pytest.approx(25.0) # 1 - (3/4) = 25% attrition

    # Check that plots were generated
    assert (tmp_path / "competition_counts_histogram.png").exists()
    assert (tmp_path / "cumulative_eligibility_curve.png").exists()