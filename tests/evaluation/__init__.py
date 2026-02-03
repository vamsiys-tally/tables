"""
Evaluation Framework for Bank Statement Extraction.

This module provides tools for:
- Comparing extracted transactions against ground truth CSVs
- Calculating accuracy metrics (transaction-level, file-level)
- Generating error reports for debugging
- Tracking run history for regression detection

Usage:
    python -m tests.evaluation.runner

    # Run on specific bank
    python -m tests.evaluation.runner --bank hdfc

    # Compare runs
    python -m tests.evaluation.runner --compare run_001 run_002

See docs/test_suite_prd.md for full specification.
"""

# CSV Loading
from tests.evaluation.csv_loader import (
    load_ground_truth,
    discover_ground_truth_files,
    load_all_ground_truth,
    GroundTruthTransaction,
    GroundTruthFile,
)

# Comparison
from tests.evaluation.comparator import (
    TransactionComparator,
    ComparisonResult,
    TransactionMatch,
    MismatchType,
)

# Metrics
from tests.evaluation.metrics import (
    AccuracyMetrics,
    FileMetrics,
    calculate_accuracy,
    compare_runs,
)

# Run Tracking
from tests.evaluation.run_tracker import (
    RunTracker,
    RunSummary,
    load_run_history,
)

# Runner
from tests.evaluation.runner import EvaluationRunner

__all__ = [
    # CSV Loading
    "load_ground_truth",
    "discover_ground_truth_files",
    "load_all_ground_truth",
    "GroundTruthTransaction",
    "GroundTruthFile",
    # Comparison
    "TransactionComparator",
    "ComparisonResult",
    "TransactionMatch",
    "MismatchType",
    # Metrics
    "AccuracyMetrics",
    "FileMetrics",
    "calculate_accuracy",
    "compare_runs",
    # Run Tracking
    "RunTracker",
    "RunSummary",
    "load_run_history",
    # Runner
    "EvaluationRunner",
]
