from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class StatisticalEvaluationResults:
    """Container for statistical evaluation results."""
    results_table: Any  # pandas DataFrame
