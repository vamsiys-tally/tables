"""
Evaluation Framework for Bank Statement Extraction.

This module provides tools for:
- Comparing extracted transactions against ground truth CSVs
- Calculating accuracy metrics (transaction-level, file-level)
- Generating error reports for debugging
- Tracking run history for regression detection

Usage:
    python -m tests.evaluation.runner

See docs/test_suite_prd.md for full specification.
"""

from tests.evaluation.comparator import TransactionComparator, ComparisonResult
from tests.evaluation.metrics import AccuracyMetrics, calculate_accuracy
from tests.evaluation.csv_loader import load_ground_truth, GroundTruthTransaction

__all__ = [
    "TransactionComparator",
    "ComparisonResult",
    "AccuracyMetrics",
    "calculate_accuracy",
    "load_ground_truth",
    "GroundTruthTransaction",
]
