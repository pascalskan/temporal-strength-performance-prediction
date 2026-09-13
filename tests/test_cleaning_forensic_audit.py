import pandas as pd
import pytest
from pathlib import Path
from src.analysis.cleaning_forensic_audit import run_cleaning_audit

def test_run_cleaning_audit_calculations(tmp_path: Path):
    """
    Validates that the cleaning audit correctly calculates raw missingness, 
    sequential attrition, and independent overlap.
    """
    data = {
        'Name': ['A', 'B', 'C', 'D', 'E'],
        'Sex': ['M', 'F', None, 'M', 'F'], # 1 missing
        'Age': [20, None, 25, 30, 22], # 1 missing
        'BodyweightKg': [80, 60, None, 90, 70], # 1 missing
        'Equipment': ['Raw', 'Raw', 'Raw', 'Multi-ply', None], # 1 invalid, 1 missing
        'TotalKg': [500, 300, 400, 600, None] # 1 missing
    }
    
    # Missing core columns (Age, BodyweightKg, Equipment, TotalKg):
    # A: None missing. Equipment=Raw (Valid). -> KEEP
    # B: Missing Age. Equipment=Raw (Valid). -> DROP in core
    # C: Missing BodyweightKg. Equipment=Raw (Valid). -> DROP in core
    # D: None missing. Equipment=Multi-ply (Invalid). -> DROP in equip
    # E: Missing Equipment, missing TotalKg. Equipment=None. -> DROP in core
    
    # Sequential attrition expected:
    # Initial: 5
    # Drop Core: B, C, E are dropped. (3 dropped). Remaining: 2 (A, D)
    # Filter Equip: D is dropped. (1 dropped). Remaining: 1 (A)
    # Final: 1
    
    raw_df = pd.DataFrame(data)
    
    run_cleaning_audit(raw_df, tmp_path)
    
    # Verify Raw Missingness
    missingness = pd.read_csv(tmp_path / "raw_missingness_audit.csv")
    assert missingness[missingness['column'] == 'Age']['missing_count'].iloc[0] == 1
    assert missingness[missingness['column'] == 'Sex']['missing_count'].iloc[0] == 1
    assert missingness[missingness['column'] == 'Equipment']['missing_count'].iloc[0] == 1
    assert missingness[missingness['column'] == 'TotalKg']['missing_count'].iloc[0] == 1
    assert missingness[missingness['column'] == 'BodyweightKg']['missing_count'].iloc[0] == 1
    
    # Verify Sequential Attrition
    attrition = pd.read_csv(tmp_path / "cleaning_attrition_decomposition.csv")
    
    assert attrition[attrition['step'] == 'Initial']['records_after'].iloc[0] == 5
    
    core_drop = attrition[attrition['step'] == 'Drop Core NaNs']
    assert core_drop['records_removed'].iloc[0] == 3
    assert core_drop['records_after'].iloc[0] == 2
    
    equip_drop = attrition[attrition['step'] == 'Filter Equipment']
    assert equip_drop['records_removed'].iloc[0] == 1
    assert equip_drop['records_after'].iloc[0] == 1
    
    assert attrition[attrition['step'] == 'Final']['records_after'].iloc[0] == 1
    
    # Verify Overlap
    overlap = pd.read_csv(tmp_path / "cleaning_rule_overlap_summary.csv")
    
    indep_core = overlap[overlap['rule'] == 'Drop Core NaNs']['independent_removals'].iloc[0]
    assert indep_core == 3 # B, C, E
    
    indep_equip = overlap[overlap['rule'] == 'Filter Equipment']['independent_removals'].iloc[0]
    assert indep_equip == 2 # D (Multi-ply), E (None)
    
    total_removed = overlap[overlap['rule'] == 'Overlap (records failing both rules)']['independent_removals'].iloc[0]
    assert total_removed == 1 # E fails both (missing Equipment, so it's not in VALID_EQUIPMENT, and missing TotalKg/Equipment so it's dropped in core)
    
    assert (tmp_path / "cleaning_attrition_by_rule.png").exists()
    assert (tmp_path / "raw_missingness_by_column.png").exists()
