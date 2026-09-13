import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.loader import load_data
from src.data.cleaning import clean_data
from src.data.identity import build_athlete_id
from src.core.logging import get_logger

logger = get_logger(__name__)

def run_audit(raw_data_path: Path, output_dir: Path):
    """
    Performs a forensic audit of the dataset to quantify population and history.
    """
    logger.info("Starting dataset forensic audit.")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Data Loading and Cleaning
    raw_df = load_data(raw_data_path)
    total_raw_records = len(raw_df)
    logger.info(f"Loaded {total_raw_records} raw records.")

    cleaned_df = clean_data(raw_df.copy())
    total_cleaned_records = len(cleaned_df)
    records_removed = total_raw_records - total_cleaned_records
    removal_percentage = (records_removed / total_raw_records) * 100 if total_raw_records > 0 else 0

    population_summary = {
        "total_raw_records": total_raw_records,
        "total_cleaned_records": total_cleaned_records,
        "records_removed": records_removed,
        "removal_percentage": removal_percentage,
    }
    pd.DataFrame([population_summary]).to_csv(output_dir / "dataset_population_summary.csv", index=False)
    logger.info(f"Removed {records_removed} records ({removal_percentage:.2f}%) during cleaning.")

    # 2. Athlete Identity and History
    cleaned_df['athlete_id'] = build_athlete_id(cleaned_df)
    unique_athletes = cleaned_df['athlete_id'].nunique()
    logger.info(f"Found {unique_athletes} unique athletes.")

    competition_counts = cleaned_df.groupby('athlete_id').size().reset_index(name='competition_count')
    
    history_thresholds = [1, 2, 3, 5, 10, 15, 20]
    history_distribution = {"threshold": [], "athlete_count": [], "percentage": []}

    for threshold in history_thresholds:
        count = competition_counts[competition_counts['competition_count'] >= threshold].shape[0]
        percentage = (count / unique_athletes) * 100 if unique_athletes > 0 else 0
        history_distribution["threshold"].append(f">={threshold}")
        history_distribution["athlete_count"].append(count)
        history_distribution["percentage"].append(percentage)

    history_df = pd.DataFrame(history_distribution)
    history_df.to_csv(output_dir / "athlete_history_distribution.csv", index=False)
    logger.info("Athlete history distribution calculated.")

    # 3. Attrition Analysis
    # This is a simplified estimation. A full implementation would need to lag features.
    attrition_summary = {
        "threshold": ["all_unique_athletes", ">=2_competitions", ">=3_competitions", ">=5_competitions"],
        "eligible_athletes": [
            unique_athletes,
            history_distribution["athlete_count"][1],
            history_distribution["athlete_count"][2],
            history_distribution["athlete_count"][3],
        ]
    }
    attrition_df = pd.DataFrame(attrition_summary)
    attrition_df["attrition_percentage"] = 100 * (1 - attrition_df["eligible_athletes"] / unique_athletes)
    attrition_df.to_csv(output_dir / "feature_eligibility_attrition.csv", index=False)
    logger.info("Feature eligibility attrition estimated.")

    # 4. Plots
    plt.figure(figsize=(12, 6))
    sns.histplot(competition_counts['competition_count'], bins=30, kde=False)
    plt.title('Distribution of Competition Counts per Athlete')
    plt.xlabel('Number of Competitions')
    plt.ylabel('Number of Athletes')
    plt.yscale('log')
    plt.savefig(output_dir / "competition_counts_histogram.png")
    logger.info("Competition count histogram saved.")

    logger.info("Dataset forensic audit complete.")

if __name__ == '__main__':
    # This allows the script to be run directly, pointing to the data and output folders.
    # In a real pipeline, this would be orchestrated by a higher-level script.
    from src.config.core import PROCESSED_DATA_DIR, RESULTS_DIR
    
    run_audit(
        raw_data_path=PROCESSED_DATA_DIR / "openpowerlifting-latest.csv",
        output_dir=RESULTS_DIR / "forensic_audit"
    )
