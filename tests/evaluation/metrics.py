"""
Accuracy Metrics Module.

Calculates accuracy metrics at file and aggregate levels.

Metrics:
    - File accuracy: matched_transactions / ground_truth_transactions
    - Overall accuracy: total_matched / total_ground_truth
    - Error breakdown by type
    - Bank/category level rollups

Example:
    >>> metrics = AccuracyMetrics()
    >>> metrics.add_result(comparison_result)
    >>> print(f"Overall: {metrics.overall_accuracy:.2%}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Any
import logging

from tests.evaluation.comparator import ComparisonResult, MismatchType

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class FileMetrics:
    """
    Metrics for a single file.

    Attributes:
        file_name: Name of the file
        gt_count: Number of ground truth transactions
        extracted_count: Number of extracted transactions
        matched_count: Number of matched transactions
        missed_count: GT rows not extracted
        extra_count: Extracted rows not in GT
        accuracy: matched / gt_count
        error_breakdown: Count by error type
    """

    file_name: str
    gt_count: int = 0
    extracted_count: int = 0
    matched_count: int = 0
    missed_count: int = 0
    extra_count: int = 0
    error_breakdown: dict[str, int] = field(default_factory=dict)

    @property
    def accuracy(self) -> float:
        """Calculate accuracy as matched / ground truth count."""
        if self.gt_count == 0:
            return 1.0 if self.extracted_count == 0 else 0.0
        return self.matched_count / self.gt_count

    @property
    def is_perfect(self) -> bool:
        """Check if file has 100% accuracy with no extra rows."""
        return self.matched_count == self.gt_count and self.extra_count == 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "file_name": self.file_name,
            "gt_count": self.gt_count,
            "extracted_count": self.extracted_count,
            "matched_count": self.matched_count,
            "missed_count": self.missed_count,
            "extra_count": self.extra_count,
            "accuracy": self.accuracy,
            "is_perfect": self.is_perfect,
            "error_breakdown": self.error_breakdown,
        }


@dataclass
class CategoryMetrics:
    """
    Metrics for a category (e.g., bank, account type).

    Attributes:
        category_name: Name of the category
        file_count: Number of files in category
        perfect_count: Files with 100% accuracy
        total_gt: Total ground truth transactions
        total_matched: Total matched transactions
        accuracy: total_matched / total_gt
    """

    category_name: str
    file_count: int = 0
    perfect_count: int = 0
    total_gt: int = 0
    total_matched: int = 0
    total_missed: int = 0
    total_extra: int = 0
    error_breakdown: dict[str, int] = field(default_factory=dict)

    @property
    def accuracy(self) -> float:
        """Calculate accuracy."""
        if self.total_gt == 0:
            return 1.0 if self.total_matched == 0 else 0.0
        return self.total_matched / self.total_gt

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "category_name": self.category_name,
            "file_count": self.file_count,
            "perfect_count": self.perfect_count,
            "total_gt": self.total_gt,
            "total_matched": self.total_matched,
            "total_missed": self.total_missed,
            "total_extra": self.total_extra,
            "accuracy": self.accuracy,
            "error_breakdown": self.error_breakdown,
        }


@dataclass
class AccuracyMetrics:
    """
    Aggregate accuracy metrics across multiple files.

    Tracks overall accuracy, per-file metrics, and category breakdowns.

    Example:
        >>> metrics = AccuracyMetrics()
        >>> metrics.add_result(comparison_result)
        >>> metrics.add_result(comparison_result2)
        >>> print(f"Overall: {metrics.overall_accuracy:.2%}")
    """

    # File-level metrics
    file_metrics: list[FileMetrics] = field(default_factory=list)

    # Category breakdowns
    by_bank: dict[str, CategoryMetrics] = field(default_factory=dict)
    by_classification: dict[str, CategoryMetrics] = field(default_factory=dict)

    # Aggregate counts
    total_files: int = 0
    perfect_files: int = 0
    total_gt_transactions: int = 0
    total_extracted_transactions: int = 0
    total_matched_transactions: int = 0
    total_missed_transactions: int = 0
    total_extra_transactions: int = 0

    # Error breakdown
    error_breakdown: dict[str, int] = field(default_factory=dict)

    # Timing
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    @property
    def overall_accuracy(self) -> float:
        """Calculate overall accuracy across all files."""
        if self.total_gt_transactions == 0:
            return 1.0 if self.total_extracted_transactions == 0 else 0.0
        return self.total_matched_transactions / self.total_gt_transactions

    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate duration if timing is available."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    def add_result(
        self,
        result: ComparisonResult,
        bank: Optional[str] = None,
        classification: Optional[str] = None,
    ) -> None:
        """
        Add a comparison result to the metrics.

        Args:
            result: ComparisonResult from comparator
            bank: Optional bank name for categorization
            classification: Optional classification tag (e.g., "STANDARD")
        """
        # Create file metrics
        file_metric = FileMetrics(
            file_name=result.file_name,
            gt_count=result.gt_count,
            extracted_count=result.extracted_count,
            matched_count=result.matched_count,
            missed_count=result.missed_count,
            extra_count=result.extra_count,
            error_breakdown=result.get_error_breakdown(),
        )
        self.file_metrics.append(file_metric)

        # Update aggregate counts
        self.total_files += 1
        if file_metric.is_perfect:
            self.perfect_files += 1

        self.total_gt_transactions += result.gt_count
        self.total_extracted_transactions += result.extracted_count
        self.total_matched_transactions += result.matched_count
        self.total_missed_transactions += result.missed_count
        self.total_extra_transactions += result.extra_count

        # Update error breakdown
        for error_type, count in result.get_error_breakdown().items():
            self.error_breakdown[error_type] = (
                self.error_breakdown.get(error_type, 0) + count
            )

        # Update bank metrics
        if bank:
            self._update_category(self.by_bank, bank, result)

        # Update classification metrics
        if classification:
            self._update_category(self.by_classification, classification, result)

    def _update_category(
        self,
        category_dict: dict[str, CategoryMetrics],
        category_name: str,
        result: ComparisonResult,
    ) -> None:
        """Update category metrics."""
        if category_name not in category_dict:
            category_dict[category_name] = CategoryMetrics(category_name=category_name)

        cat = category_dict[category_name]
        cat.file_count += 1
        if result.is_perfect:
            cat.perfect_count += 1
        cat.total_gt += result.gt_count
        cat.total_matched += result.matched_count
        cat.total_missed += result.missed_count
        cat.total_extra += result.extra_count

        # Update error breakdown
        for error_type, count in result.get_error_breakdown().items():
            cat.error_breakdown[error_type] = (
                cat.error_breakdown.get(error_type, 0) + count
            )

    def get_files_with_errors(self) -> list[FileMetrics]:
        """Get all files that have errors."""
        return [f for f in self.file_metrics if not f.is_perfect]

    def get_files_by_accuracy(self, ascending: bool = True) -> list[FileMetrics]:
        """Get files sorted by accuracy."""
        return sorted(
            self.file_metrics,
            key=lambda f: f.accuracy,
            reverse=not ascending,
        )

    def get_top_error_types(self, limit: int = 5) -> list[tuple[str, int]]:
        """Get the most common error types."""
        sorted_errors = sorted(
            self.error_breakdown.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return sorted_errors[:limit]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "total_files": self.total_files,
            "perfect_files": self.perfect_files,
            "total_gt_transactions": self.total_gt_transactions,
            "total_extracted_transactions": self.total_extracted_transactions,
            "total_matched_transactions": self.total_matched_transactions,
            "total_missed_transactions": self.total_missed_transactions,
            "total_extra_transactions": self.total_extra_transactions,
            "overall_accuracy": self.overall_accuracy,
            "error_breakdown": self.error_breakdown,
            "duration_seconds": self.duration_seconds,
            "by_bank": {k: v.to_dict() for k, v in self.by_bank.items()},
            "by_classification": {
                k: v.to_dict() for k, v in self.by_classification.items()
            },
        }

    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            "=" * 60,
            "EVALUATION SUMMARY",
            "=" * 60,
            f"Files Tested:     {self.total_files}",
            f"Perfect Files:    {self.perfect_files} ({self.perfect_files/max(self.total_files, 1):.1%})",
            "",
            f"Transactions:",
            f"  Ground Truth:   {self.total_gt_transactions}",
            f"  Extracted:      {self.total_extracted_transactions}",
            f"  Matched:        {self.total_matched_transactions}",
            f"  Missed:         {self.total_missed_transactions}",
            f"  Extra:          {self.total_extra_transactions}",
            "",
            f"Overall Accuracy: {self.overall_accuracy:.2%}",
        ]

        if self.error_breakdown:
            lines.append("")
            lines.append("Error Breakdown:")
            for error_type, count in sorted(
                self.error_breakdown.items(), key=lambda x: -x[1]
            ):
                lines.append(f"  {error_type}: {count}")

        if self.duration_seconds:
            lines.append("")
            lines.append(f"Duration: {self.duration_seconds:.1f}s")

        lines.append("=" * 60)
        return "\n".join(lines)


# =============================================================================
# Convenience Functions
# =============================================================================

def calculate_accuracy(
    results: list[ComparisonResult],
    bank_extractor: Optional[callable] = None,
    classification_extractor: Optional[callable] = None,
) -> AccuracyMetrics:
    """
    Calculate accuracy metrics from a list of comparison results.

    Args:
        results: List of ComparisonResult objects
        bank_extractor: Optional function to extract bank from file name
        classification_extractor: Optional function to extract classification

    Returns:
        AccuracyMetrics with aggregated results

    Example:
        >>> results = [comparator.compare(gt1, ext1), comparator.compare(gt2, ext2)]
        >>> metrics = calculate_accuracy(results)
        >>> print(metrics.summary())
    """
    metrics = AccuracyMetrics()
    metrics.start_time = datetime.now()

    for result in results:
        bank = None
        classification = None

        if bank_extractor:
            try:
                bank = bank_extractor(result.file_name)
            except Exception:
                pass

        if classification_extractor:
            try:
                classification = classification_extractor(result.file_name)
            except Exception:
                pass

        metrics.add_result(result, bank=bank, classification=classification)

    metrics.end_time = datetime.now()
    return metrics


def extract_bank_from_path(file_path: str) -> Optional[str]:
    """
    Extract bank name from file path.

    Assumes structure: .../ground_truth/{bank}/{file}.csv

    Args:
        file_path: Path to the file

    Returns:
        Bank name or None
    """
    from pathlib import Path

    path = Path(file_path)

    # Look for ground_truth parent
    parts = path.parts
    for i, part in enumerate(parts):
        if part == "ground_truth" and i + 1 < len(parts):
            return parts[i + 1]

    # Fallback: try to extract from filename prefix
    name = path.stem
    if "_" in name:
        return name.split("_")[0]

    return None


def compare_runs(
    current: AccuracyMetrics,
    previous: AccuracyMetrics,
) -> dict[str, Any]:
    """
    Compare two runs to detect improvements and regressions.

    Args:
        current: Current run metrics
        previous: Previous run metrics

    Returns:
        Dictionary with comparison results
    """
    # Build file accuracy lookup for previous run
    prev_files = {f.file_name: f for f in previous.file_metrics}
    curr_files = {f.file_name: f for f in current.file_metrics}

    # Find improvements and regressions
    improved = []
    regressed = []
    new_files = []

    for file_name, curr_metric in curr_files.items():
        if file_name in prev_files:
            prev_metric = prev_files[file_name]
            if curr_metric.accuracy > prev_metric.accuracy:
                improved.append({
                    "file": file_name,
                    "previous_accuracy": prev_metric.accuracy,
                    "current_accuracy": curr_metric.accuracy,
                    "delta": curr_metric.accuracy - prev_metric.accuracy,
                })
            elif curr_metric.accuracy < prev_metric.accuracy:
                regressed.append({
                    "file": file_name,
                    "previous_accuracy": prev_metric.accuracy,
                    "current_accuracy": curr_metric.accuracy,
                    "delta": curr_metric.accuracy - prev_metric.accuracy,
                })
        else:
            new_files.append(file_name)

    # Removed files
    removed_files = [f for f in prev_files if f not in curr_files]

    return {
        "accuracy_delta": current.overall_accuracy - previous.overall_accuracy,
        "files_improved": improved,
        "files_regressed": regressed,
        "new_files": new_files,
        "removed_files": removed_files,
        "perfect_files_delta": current.perfect_files - previous.perfect_files,
    }
