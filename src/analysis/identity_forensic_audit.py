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

# Tolerated backwards movement in recorded age between consecutive competitions,
# absorbing OpenPowerlifting's half-year age approximations. Anything beyond
# this is treated as an identity-collision signal.
AGE_REVERSAL_TOLERANCE_YEARS = 0.5


def run_identity_audit(cleaned_df: pd.DataFrame, output_dir: Path):
    """
    Performs a forensic audit of the athlete identity construction to quantify collision risk.
    """
    logger.info("Starting athlete identity forensic audit.")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use the project's default identity builder
    cleaned_df['athlete_id'] = build_athlete_id(cleaned_df)
    
    # 1. Identity Population Summary
    logger.info("Running identity population summary...")
    total_records = len(cleaned_df)
    total_identities = cleaned_df['athlete_id'].nunique()
    competition_counts = cleaned_df.groupby('athlete_id').size()

    identity_summary = pd.DataFrame([{
        "total_cleaned_records": total_records,
        "total_constructed_identities": total_identities,
        "average_competitions_per_identity": competition_counts.mean(),
        "median_competitions_per_identity": competition_counts.median(),
        "max_competitions_per_identity": competition_counts.max(),
    }])
    identity_summary.to_csv(output_dir / "identity_population_summary.csv", index=False)
    logger.info("Identity population summary saved.")

    # 2. Collision Risk Heuristics
    logger.info("Running collision risk heuristics...")
    
    # A. High-History Suspicious Identities
    history_thresholds = [20, 30, 40, 50]
    high_history_mask = competition_counts >= min(history_thresholds)
    high_history_identities = competition_counts[high_history_mask].reset_index(name='competition_count')
    high_history_identities.to_csv(output_dir / "suspicious_high_history_identities.csv", index=False)

    # B. Temporal Span Audit
    date_stats = cleaned_df.groupby('athlete_id')['Date'].agg(['min', 'max'])
    date_stats['career_span_years'] = (date_stats['max'] - date_stats['min']).dt.days / 365.25
    span_thresholds = [15, 20, 25, 30]
    suspicious_spans = date_stats[date_stats['career_span_years'] >= min(span_thresholds)].reset_index()
    suspicious_spans.to_csv(output_dir / "suspicious_temporal_span_identities.csv", index=False)

    # C. Physiological Inconsistency Heuristics
    phys_stats = cleaned_df.groupby('athlete_id').agg(
        age_min=('Age', 'min'),
        age_max=('Age', 'max'),
        bw_min=('BodyweightKg', 'min'),
        bw_max=('BodyweightKg', 'max'),
        date_min=('Date', 'min'),
        date_max=('Date', 'max')
    )
    phys_stats['age_range'] = phys_stats['age_max'] - phys_stats['age_min']
    phys_stats['bw_range'] = phys_stats['bw_max'] - phys_stats['bw_min']
    phys_stats['span_years'] = (phys_stats['date_max'] - phys_stats['date_min']).dt.days / 365.25
    
    phys_stats['age_progression_ok'] = phys_stats.apply(
        lambda row: (row['age_max'] - row['age_min']) >= (row['span_years'] - 1) if row['span_years'] > 1 else True, 
        axis=1
    )

    # Age reversal: an athlete's recorded age must not fall as time advances.
    # The age_range heuristic above cannot detect this, because max - min is
    # always non-negative -- a 40 -> 39 sequence is invisible to it. A reversal
    # is a direct signal that two people have been merged into one identity.
    # OpenPowerlifting records some ages as approximations (e.g. 24.5 meaning
    # "24 or 25"), so a tolerance absorbs that rounding rather than flagging it.
    phys_stats['max_age_reversal_years'] = (
        cleaned_df
        .sort_values('Date')
        .groupby('athlete_id')['Age']
        .apply(lambda ages: max(0.0, float(-ages.diff().min()))
               if len(ages) > 1 and pd.notna(ages.diff().min()) else 0.0)
    )

    suspicious_phys = phys_stats[
        (phys_stats['bw_range'] > 40) | 
        (phys_stats['age_range'] > phys_stats['span_years'] + 1) | 
        (phys_stats['max_age_reversal_years'] > AGE_REVERSAL_TOLERANCE_YEARS) |
        (~phys_stats['age_progression_ok'])
    ].reset_index()
    suspicious_phys.to_csv(output_dir / "suspicious_physiological_inconsistencies.csv", index=False)

    # Summary of collision risks
    risk_summary = pd.DataFrame([{
        "risk_heuristic": "High Competition History (>=20)",
        "suspicious_identity_count": len(high_history_identities[high_history_identities['competition_count'] >= 20]),
    }, {
        "risk_heuristic": "Long Career Span (>=15 years)",
        "suspicious_identity_count": len(suspicious_spans[suspicious_spans['career_span_years'] >= 15]),
    }, {
        "risk_heuristic": "Physiological Inconsistency",
        "suspicious_identity_count": len(suspicious_phys),
    }, {
        "risk_heuristic": f"Age Reversal (>{AGE_REVERSAL_TOLERANCE_YEARS}y)",
        "suspicious_identity_count": int(
            (phys_stats['max_age_reversal_years'] > AGE_REVERSAL_TOLERANCE_YEARS).sum()
        ),
    }])
    risk_summary.to_csv(output_dir / "identity_collision_risk_summary.csv", index=False)
    logger.info("Collision risk summaries saved.")

    # D & E. Disambiguation Comparison
    logger.info("Running identity disambiguation comparison...")
    strategies = {
        "Name_only": ['Name'],
        "Name_Sex": ['Name', 'Sex'],
        "Name_Sex_Country": ['Name', 'Sex', 'Country']
    }
    disambiguation_results = []
    for name, attrs in strategies.items():
        if all(col in cleaned_df.columns for col in attrs):
            count = cleaned_df.fillna('Unknown').astype(str).agg('_'.join, axis=1).nunique()
            disambiguation_results.append({
                "strategy": name,
                "unique_identities": cleaned_df[attrs].fillna('Unknown').astype(str).agg('_'.join, axis=1).nunique()
            })
    
    disambiguation_df = pd.DataFrame(disambiguation_results)
    disambiguation_df.to_csv(output_dir / "identity_disambiguation_comparison.csv", index=False)
    logger.info("Identity disambiguation comparison saved.")

    # 5. Visualizations
    logger.info("Generating plots...")
    
    # Competition count distribution
    plt.figure(figsize=(10, 6))
    plt.hist(competition_counts, bins=range(1, 51), log=True)
    plt.title('Distribution of Competitions per Athlete Identity')
    plt.xlabel('Number of Competitions')
    plt.ylabel('Number of Identities (Log Scale)')
    plt.grid(axis='y', alpha=0.5)
    plt.savefig(output_dir / "identity_competition_count_distribution.png")
    plt.close()

    # Career span distribution
    plt.figure(figsize=(10, 6))
    plt.hist(date_stats['career_span_years'].dropna(), bins=30, log=True)
    plt.title('Distribution of Career Spans per Athlete Identity')
    plt.xlabel('Career Span (Years)')
    plt.ylabel('Number of Identities (Log Scale)')
    plt.grid(axis='y', alpha=0.5)
    plt.savefig(output_dir / "identity_career_span_distribution.png")
    plt.close()

    # Disambiguation comparison plot
    if not disambiguation_df.empty:
        plt.figure(figsize=(8, 5))
        plt.bar(disambiguation_df['strategy'], disambiguation_df['unique_identities'])
        plt.title('Identity Disambiguation Comparison')
        plt.ylabel('Number of Unique Identities')
        plt.tight_layout()
        plt.savefig(output_dir / "identity_disambiguation_comparison.png")
        plt.close()

    logger.info("Identity forensic audit complete.")


if __name__ == '__main__':
    dataset_path = ProjectPaths.use_dataset("openpowerlifting.csv")
    if not dataset_path.exists():
        logger.error(f"Dataset not found at {dataset_path}")
    else:
        logger.info(f"Loading data from {dataset_path}")
        raw_df = load_data(dataset_path)
        cleaned_df = clean_data(raw_df)
        
        audit_dir = ProjectPaths.results_dir() / "forensic_audit"
        run_identity_audit(cleaned_df, audit_dir)
