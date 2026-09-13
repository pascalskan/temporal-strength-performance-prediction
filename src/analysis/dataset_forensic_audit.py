import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.data.loader import load_data
from src.data.cleaning import clean_data
from src.data.identity import build_athlete_id
from src.core.logging import get_logger
from src.io.paths import ProjectPaths

logger = get_logger(__name__)


def run_audit(raw_df: pd.DataFrame, output_dir: Path):
    """
    Performs a forensic audit of the dataset to quantify population and longitudinal history.
    """
    logger.info("Starting dataset forensic audit.")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Dataset Level Summaries
    total_raw_records = len(raw_df)
    logger.info(f"Total raw records: {total_raw_records}")

    cleaned_df = clean_data(raw_df.copy())
    total_cleaned_records = len(cleaned_df)
    
    records_removed = total_raw_records - total_cleaned_records
    removal_percentage = (records_removed / total_raw_records) * 100 if total_raw_records > 0 else 0

    population_summary = pd.DataFrame([{
        "total_raw_records": total_raw_records,
        "total_cleaned_records": total_cleaned_records,
        "records_removed": records_removed,
        "removal_percentage": removal_percentage,
    }])
    population_summary.to_csv(output_dir / "dataset_population_summary.csv", index=False)
    logger.info(f"Dataset population summary saved. Removed {records_removed} records ({removal_percentage:.2f}%).")

    # 2. Athlete Identity and Longitudinal History Distribution
    # Use existing athlete identity construction
    cleaned_df['athlete_id'] = build_athlete_id(cleaned_df)
    unique_athletes = cleaned_df['athlete_id'].nunique()
    logger.info(f"Found {unique_athletes} unique athletes after cleaning.")

    # Calculate competition counts per athlete
    competition_counts = cleaned_df.groupby('athlete_id').size().reset_index(name='competition_count')
    
    thresholds = [1, 2, 3, 5, 10, 15, 20]
    history_records = []
    
    for threshold in thresholds:
        count = (competition_counts['competition_count'] >= threshold).sum()
        percentage = (count / unique_athletes) * 100 if unique_athletes > 0 else 0
        history_records.append({
            "threshold": f">={threshold}",
            "athlete_count": count,
            "percentage_of_population": percentage
        })

    history_df = pd.DataFrame(history_records)
    history_df.to_csv(output_dir / "athlete_history_distribution.csv", index=False)
    logger.info("Athlete history distribution saved.")

    # 3. Attrition Analysis for Feature Eligibility
    # To forecast, we require lag features. 
    # Previous competition only -> requires at least 2 comps (1 for lag, 1 to predict)
    # 2 prior comps -> requires at least 3 comps
    # 3 prior comps -> requires at least 4 comps
    # Rolling-window features (often e.g., 3-5 comps) -> assuming typical 3 prior comps
    
    attrition_records = [
        {
            "feature_requirement": "Baseline (No lags, all unique athletes)",
            "required_competitions": 1,
            "eligible_athletes": unique_athletes,
        },
        {
            "feature_requirement": "Previous competition only (Lag 1)",
            "required_competitions": 2,
            "eligible_athletes": (competition_counts['competition_count'] >= 2).sum(),
        },
        {
            "feature_requirement": "2 prior competitions (Lag 1 & 2)",
            "required_competitions": 3,
            "eligible_athletes": (competition_counts['competition_count'] >= 3).sum(),
        },
        {
            "feature_requirement": "3 prior competitions (Lag 1, 2, & 3) / Rolling window",
            "required_competitions": 4,
            "eligible_athletes": (competition_counts['competition_count'] >= 4).sum(),
        }
    ]
    
    attrition_df = pd.DataFrame(attrition_records)
    attrition_df["attrition_from_baseline_percentage"] = 100 * (1 - attrition_df["eligible_athletes"] / unique_athletes)
    attrition_df.to_csv(output_dir / "feature_eligibility_attrition.csv", index=False)
    logger.info("Feature eligibility attrition saved.")

    # 4. Distribution Outputs (Plots)
    
    # Histogram of competition counts
    plt.figure(figsize=(10, 6))
    
    max_count = int(competition_counts['competition_count'].max()) if len(competition_counts) > 0 else 30
    bins = range(1, min(max_count + 2, 31))
    
    plt.hist(competition_counts['competition_count'], bins=bins, edgecolor='black', alpha=0.7, align='left')
    plt.title('Distribution of Competition Depth per Athlete')
    plt.xlabel('Number of Competitions')
    plt.ylabel('Number of Athletes (Log Scale)')
    plt.yscale('log')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "competition_counts_histogram.png")
    plt.close()

    # Cumulative eligibility curve (Reverse ECDF)
    # Shows % of population eligible as required history depth increases
    max_comps = min(max_count, 30)
    x_vals = np.arange(1, max_comps + 1)
    y_vals = [(competition_counts['competition_count'] >= x).sum() / unique_athletes * 100 for x in x_vals]
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals, marker='o', linestyle='-')
    plt.title('Cumulative Eligibility Curve (Attrition by History Depth)')
    plt.xlabel('Minimum Required Competitions')
    plt.ylabel('Percentage of Population Eligible (%)')
    plt.grid(True, alpha=0.5)
    
    if max_comps > 1:
        plt.xticks(np.arange(1, max_comps + 1, step=2))

    plt.tight_layout()
    plt.savefig(output_dir / "cumulative_eligibility_curve.png")
    plt.close()

    logger.info(f"Audit completed successfully. Outputs stored in {output_dir}")


if __name__ == '__main__':
    # Script execution capability for independent running
    dataset_path = ProjectPaths.dataset_path("openpowerlifting.csv")
    if not dataset_path.exists():
        logger.error(f"Dataset not found at {dataset_path}")
    else:
        logger.info(f"Loading data from {dataset_path}")
        raw_df_instance = load_data(dataset_path)
        
        audit_dir = ProjectPaths.results_dir() / "forensic_audit"
        run_audit(raw_df_instance, audit_dir)
