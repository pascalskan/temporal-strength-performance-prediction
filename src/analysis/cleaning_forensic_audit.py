import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.data.loader import load_data
from src.data.cleaning import VALID_EQUIPMENT
from src.core.logging import get_logger
from src.io.paths import ProjectPaths

logger = get_logger(__name__)

KEY_COLUMNS = [
    'Name', 'Sex', 'Country', 'Date', 'Age', 'BodyweightKg', 
    'Equipment', 'TotalKg', 'Federation', 'Event', 'Division', 
    'WeightClassKg', 'Best3SquatKg', 'Best3BenchKg', 'Best3DeadliftKg'
]


def run_cleaning_audit(raw_df: pd.DataFrame, output_dir: Path):
    """
    Performs a forensic audit of the cleaning pipeline to quantify missingness and attrition.
    """
    logger.info("Starting cleaning forensic audit.")
    output_dir.mkdir(parents=True, exist_ok=True)

    total_raw_records = len(raw_df)

    # 1. Raw Missingness Audit
    logger.info("Running raw missingness audit...")
    present_cols = [col for col in KEY_COLUMNS if col in raw_df.columns]
    missingness_records = []
    for col in present_cols:
        missing_count = raw_df[col].isnull().sum()
        missing_percentage = (missing_count / total_raw_records) * 100 if total_raw_records > 0 else 0
        missingness_records.append({
            "column": col,
            "missing_count": missing_count,
            "missing_percentage": missing_percentage
        })
    
    missingness_df = pd.DataFrame(missingness_records)
    missingness_df.to_csv(output_dir / "raw_missingness_audit.csv", index=False)
    logger.info("Raw missingness audit saved.")

    # 2. Cleaning Pipeline Attrition Decomposition (Sequential)
    logger.info("Running cleaning pipeline attrition decomposition...")
    
    attrition_steps = []
    df_step = raw_df.copy()
    
    # Initial state
    attrition_steps.append({
        "step": "Initial",
        "rule_description": "Raw dataset loaded",
        "records_before": total_raw_records,
        "records_after": total_raw_records,
        "records_removed": 0,
        "percentage_of_raw": 0.0,
        "percentage_of_step": 0.0
    })

    # Step 1: Dropna on core columns
    core_na_cols = ["TotalKg", "Age", "BodyweightKg", "Equipment"]
    records_before_dropna = len(df_step)
    df_after_dropna = df_step.dropna(subset=core_na_cols).copy()
    records_after_dropna = len(df_after_dropna)
    removed_dropna = records_before_dropna - records_after_dropna
    
    attrition_steps.append({
        "step": "Drop Core NaNs",
        "rule_description": f"dropna(subset={core_na_cols})",
        "records_before": records_before_dropna,
        "records_after": records_after_dropna,
        "records_removed": removed_dropna,
        "percentage_of_raw": (removed_dropna / total_raw_records) * 100,
        "percentage_of_step": (removed_dropna / records_before_dropna) * 100
    })
    df_step = df_after_dropna

    # Step 2: Filter by Equipment
    records_before_equip = len(df_step)
    df_after_equip = df_step[df_step["Equipment"].isin(VALID_EQUIPMENT)].copy()
    records_after_equip = len(df_after_equip)
    removed_equip = records_before_equip - records_after_equip

    attrition_steps.append({
        "step": "Filter Equipment",
        "rule_description": f"Equipment.isin({VALID_EQUIPMENT})",
        "records_before": records_before_equip,
        "records_after": records_after_equip,
        "records_removed": removed_equip,
        "percentage_of_raw": (removed_equip / total_raw_records) * 100,
        "percentage_of_step": (removed_equip / records_before_equip) * 100
    })
    df_step = df_after_equip

    # Final state
    attrition_steps.append({
        "step": "Final",
        "rule_description": "Final cleaned dataset",
        "records_before": records_after_equip,
        "records_after": records_after_equip,
        "records_removed": 0,
        "percentage_of_raw": 0.0,
        "percentage_of_step": 0.0
    })

    attrition_df = pd.DataFrame(attrition_steps)
    attrition_df.to_csv(output_dir / "cleaning_attrition_decomposition.csv", index=False)
    logger.info("Cleaning attrition decomposition saved.")

    # 3. Cleaning Rule Overlap Summary
    logger.info("Running cleaning rule overlap summary...")
    
    # Independent (non-sequential) removal counts
    na_removals_independent = raw_df[core_na_cols].isnull().any(axis=1).sum()
    equip_removals_independent = (~raw_df["Equipment"].isin(VALID_EQUIPMENT)).sum()
    
    # Overlap calculation
    # A and B = A + B - (A or B)
    # A or B = total removed
    total_removed_by_rules = (
        raw_df[core_na_cols].isnull().any(axis=1) | 
        (~raw_df["Equipment"].isin(VALID_EQUIPMENT))
    ).sum()
    overlap = (na_removals_independent + equip_removals_independent) - total_removed_by_rules

    overlap_summary = pd.DataFrame([{
        "rule": "Drop Core NaNs",
        "independent_removals": na_removals_independent,
        "sequential_removals": removed_dropna
    }, {
        "rule": "Filter Equipment",
        "independent_removals": equip_removals_independent,
        "sequential_removals": removed_equip
    }, {
        "rule": "Overlap (records failing both rules)",
        "independent_removals": overlap,
        "sequential_removals": np.nan  # Not applicable in sequential view
    }])
    overlap_summary.to_csv(output_dir / "cleaning_rule_overlap_summary.csv", index=False)
    logger.info("Cleaning rule overlap summary saved.")

    # 4. Visuals
    logger.info("Generating plots...")
    
    # Bar chart for sequential removals
    plot_df = attrition_df[attrition_df['records_removed'] > 0].set_index('step')
    plt.figure(figsize=(10, 6))
    plot_df['records_removed'].plot(kind='barh')
    plt.title('Rows Removed by Each Cleaning Step (Sequential)')
    plt.xlabel('Number of Rows Removed')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_dir / "cleaning_attrition_by_rule.png")
    plt.close()

    # Bar chart for raw missingness
    missing_plot_df = missingness_df.sort_values('missing_percentage', ascending=True)
    plt.figure(figsize=(10, 8))
    plt.barh(missing_plot_df['column'], missing_plot_df['missing_percentage'])
    plt.title('Raw Data Missingness Percentage by Column')
    plt.xlabel('Percentage of Rows Missing (%)')
    plt.tight_layout()
    plt.savefig(output_dir / "raw_missingness_by_column.png")
    plt.close()
    
    logger.info("Cleaning forensic audit complete.")


if __name__ == '__main__':
    dataset_path = ProjectPaths.dataset_path("openpowerlifting.csv")
    if not dataset_path.exists():
        logger.error(f"Dataset not found at {dataset_path}")
    else:
        logger.info(f"Loading data from {dataset_path}")
        raw_df = load_data(dataset_path)
        
        audit_dir = ProjectPaths.results_dir() / "forensic_audit"
        run_cleaning_audit(raw_df, audit_dir)
