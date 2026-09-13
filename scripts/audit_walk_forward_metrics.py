import pandas as pd
from pathlib import Path
import sys

# Ensure src is in the python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.evaluation.metrics import compute_global_forecast_metrics

def main():
    pred_path = project_root / "results" / "raw" / "walk_forward" / "walk_forward_predictions.csv"
    old_summary_path = project_root / "results" / "raw" / "walk_forward" / "walk_forward_summary_raw.csv"

    if not pred_path.exists():
        print(f"Error: Predictions file not found at {pred_path}")
        return

    # Load predictions
    print(f"Loading predictions from: {pred_path}")
    df = pd.read_csv(pred_path)
    
    # Check if we need to map column names
    y_true_col = "y_true"
    y_pred_col = "y_pred"
    
    # Map common column names if the defaults aren't found
    if "actual" in df.columns:
        y_true_col = "actual"
    elif "Actual" in df.columns:
        y_true_col = "Actual"
        
    if "predicted" in df.columns:
        y_pred_col = "predicted"
    elif "Predicted" in df.columns:
        y_pred_col = "Predicted"

    # Compute corrected metrics
    corrected = compute_global_forecast_metrics(
        df, 
        y_true_col=y_true_col, 
        y_pred_col=y_pred_col
    )
    
    corrected_sorted = corrected.sort_values("MAE")

    print("\n" + "="*50)
    print("CORRECTED METRICS (Pooled Global):")
    print("="*50)
    print(corrected_sorted.to_string(index=False))

    # Attempt to load old summary
    print("\n" + "="*50)
    print("OLD SUMMARY (Potentially Averaged):")
    print("="*50)
    try:
        old = pd.read_csv(old_summary_path)
        print(old.sort_values("MAE").to_string(index=False))
    except Exception as e:
        print(f"Could not load old summary: {e}")

    # Print MAE ranking explicitly
    print("\n" + "="*50)
    print("MAE RANKING (CORRECTED):")
    print("="*50)
    print(corrected_sorted[["model", "prediction_count", "MAE", "RMSE", "R2"]].to_string(index=False))

if __name__ == "__main__":
    main()
