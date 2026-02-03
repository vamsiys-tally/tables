"""
Run Tracker Module.

Manages evaluation run history and enables cross-run comparison.

Storage:
    - run_history.csv: Summary of all runs
    - run_{id}/summary.json: Detailed run results
    - run_{id}/errors/: Per-file error reports

Example:
    >>> tracker = RunTracker("tests/results")
    >>> run_id = tracker.start_run(notes="Initial baseline")
    >>> tracker.add_result(comparison_result)
    >>> tracker.finish_run()
"""

from __future__ import annotations

import csv
import json
import subprocess
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Any
import logging

from tests.evaluation.comparator import ComparisonResult
from tests.evaluation.metrics import AccuracyMetrics, compare_runs

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class RunSummary:
    """
    Summary of a single evaluation run.

    Stored in run_history.csv for quick lookup.
    """

    run_id: str
    timestamp: str
    git_commit: Optional[str] = None
    total_files: int = 0
    files_100pct: int = 0
    total_txns_gt: int = 0
    total_txns_matched: int = 0
    overall_accuracy: float = 0.0
    notes: str = ""

    def to_csv_row(self) -> list[str]:
        """Convert to CSV row."""
        return [
            self.run_id,
            self.timestamp,
            self.git_commit or "",
            str(self.total_files),
            str(self.files_100pct),
            str(self.total_txns_gt),
            str(self.total_txns_matched),
            f"{self.overall_accuracy:.4f}",
            self.notes,
        ]

    @classmethod
    def from_csv_row(cls, row: list[str]) -> "RunSummary":
        """Create from CSV row."""
        return cls(
            run_id=row[0],
            timestamp=row[1],
            git_commit=row[2] or None,
            total_files=int(row[3]),
            files_100pct=int(row[4]),
            total_txns_gt=int(row[5]),
            total_txns_matched=int(row[6]),
            overall_accuracy=float(row[7]),
            notes=row[8] if len(row) > 8 else "",
        )

    @classmethod
    def csv_headers(cls) -> list[str]:
        """Get CSV headers."""
        return [
            "run_id",
            "timestamp",
            "git_commit",
            "total_files",
            "files_100pct",
            "total_txns_gt",
            "total_txns_matched",
            "overall_accuracy",
            "notes",
        ]


@dataclass
class DetailedRunReport:
    """
    Detailed report for a single run.

    Stored as JSON in run_{id}/summary.json.
    """

    run_id: str
    timestamp: str
    git_commit: Optional[str] = None
    duration_seconds: Optional[float] = None
    notes: str = ""

    # Aggregate metrics
    metrics: dict[str, Any] = field(default_factory=dict)

    # Files with errors
    files_with_errors: list[dict[str, Any]] = field(default_factory=list)

    # Comparison to previous run
    comparison_to_previous: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "git_commit": self.git_commit,
            "duration_seconds": self.duration_seconds,
            "notes": self.notes,
            "metrics": self.metrics,
            "files_with_errors": self.files_with_errors,
            "comparison_to_previous": self.comparison_to_previous,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DetailedRunReport":
        """Create from dictionary."""
        return cls(
            run_id=data["run_id"],
            timestamp=data["timestamp"],
            git_commit=data.get("git_commit"),
            duration_seconds=data.get("duration_seconds"),
            notes=data.get("notes", ""),
            metrics=data.get("metrics", {}),
            files_with_errors=data.get("files_with_errors", []),
            comparison_to_previous=data.get("comparison_to_previous"),
        )


# =============================================================================
# Run Tracker
# =============================================================================

class RunTracker:
    """
    Tracks evaluation runs and manages run history.

    Provides:
        - Run ID generation
        - Run history storage (CSV)
        - Detailed run reports (JSON)
        - Cross-run comparison

    Example:
        >>> tracker = RunTracker("tests/results")
        >>> run_id = tracker.start_run(notes="Testing new parser")
        >>> # ... run evaluation ...
        >>> tracker.finish_run(metrics)
    """

    HISTORY_FILE = "run_history.csv"

    def __init__(self, results_dir: str | Path):
        """
        Initialize the run tracker.

        Args:
            results_dir: Directory for storing run results
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.history_path = self.results_dir / self.HISTORY_FILE

        # Current run state
        self.current_run_id: Optional[str] = None
        self.current_run_start: Optional[datetime] = None
        self.current_metrics: Optional[AccuracyMetrics] = None
        self.current_notes: str = ""

    def start_run(self, notes: str = "") -> str:
        """
        Start a new evaluation run.

        Args:
            notes: Optional notes about this run

        Returns:
            Generated run ID
        """
        self.current_run_id = self._generate_run_id()
        self.current_run_start = datetime.now()
        self.current_notes = notes
        self.current_metrics = AccuracyMetrics()
        self.current_metrics.start_time = self.current_run_start

        # Create run directory
        run_dir = self._get_run_dir(self.current_run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "errors").mkdir(exist_ok=True)

        logger.info(f"Started run: {self.current_run_id}")
        return self.current_run_id

    def finish_run(self, metrics: Optional[AccuracyMetrics] = None) -> RunSummary:
        """
        Finish the current run and save results.

        Args:
            metrics: Final AccuracyMetrics (if not using add_result)

        Returns:
            RunSummary for this run
        """
        if self.current_run_id is None:
            raise RuntimeError("No run in progress. Call start_run() first.")

        if metrics:
            self.current_metrics = metrics

        self.current_metrics.end_time = datetime.now()

        # Get git commit
        git_commit = self._get_git_commit()

        # Create summary
        summary = RunSummary(
            run_id=self.current_run_id,
            timestamp=self.current_run_start.isoformat(),
            git_commit=git_commit,
            total_files=self.current_metrics.total_files,
            files_100pct=self.current_metrics.perfect_files,
            total_txns_gt=self.current_metrics.total_gt_transactions,
            total_txns_matched=self.current_metrics.total_matched_transactions,
            overall_accuracy=self.current_metrics.overall_accuracy,
            notes=self.current_notes,
        )

        # Save to history
        self._append_to_history(summary)

        # Create detailed report
        detailed = self._create_detailed_report()

        # Save detailed report
        run_dir = self._get_run_dir(self.current_run_id)
        with open(run_dir / "summary.json", "w") as f:
            json.dump(detailed.to_dict(), f, indent=2)

        logger.info(
            f"Finished run {self.current_run_id}: "
            f"{summary.overall_accuracy:.2%} accuracy"
        )

        # Reset state
        run_id = self.current_run_id
        self.current_run_id = None
        self.current_run_start = None
        self.current_metrics = None
        self.current_notes = ""

        return summary

    def add_result(
        self,
        result: ComparisonResult,
        bank: Optional[str] = None,
        classification: Optional[str] = None,
    ) -> None:
        """
        Add a comparison result to the current run.

        Args:
            result: ComparisonResult from comparator
            bank: Optional bank name
            classification: Optional file classification
        """
        if self.current_metrics is None:
            raise RuntimeError("No run in progress. Call start_run() first.")

        self.current_metrics.add_result(result, bank=bank, classification=classification)

    def get_run_history(self) -> list[RunSummary]:
        """
        Get all historical runs.

        Returns:
            List of RunSummary objects, most recent first
        """
        if not self.history_path.exists():
            return []

        runs = []
        with open(self.history_path, "r", newline="") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                if row:  # Skip empty rows
                    runs.append(RunSummary.from_csv_row(row))

        return list(reversed(runs))

    def get_run_details(self, run_id: str) -> Optional[DetailedRunReport]:
        """
        Get detailed report for a specific run.

        Args:
            run_id: The run ID to retrieve

        Returns:
            DetailedRunReport or None if not found
        """
        run_dir = self._get_run_dir(run_id)
        summary_path = run_dir / "summary.json"

        if not summary_path.exists():
            return None

        with open(summary_path, "r") as f:
            data = json.load(f)

        return DetailedRunReport.from_dict(data)

    def get_previous_run(self) -> Optional[RunSummary]:
        """Get the most recent completed run."""
        history = self.get_run_history()
        if history:
            return history[0]
        return None

    def compare_with_previous(
        self,
        current_metrics: AccuracyMetrics,
    ) -> Optional[dict[str, Any]]:
        """
        Compare current metrics with the previous run.

        Args:
            current_metrics: Current run metrics

        Returns:
            Comparison dictionary or None if no previous run
        """
        previous = self.get_previous_run()
        if previous is None:
            return None

        previous_details = self.get_run_details(previous.run_id)
        if previous_details is None:
            return None

        # Load previous metrics from detailed report
        # Note: This is a simplified comparison using summary data
        return {
            "previous_run": previous.run_id,
            "accuracy_delta": current_metrics.overall_accuracy - previous.overall_accuracy,
            "perfect_files_delta": current_metrics.perfect_files - previous.files_100pct,
            "matched_delta": (
                current_metrics.total_matched_transactions - previous.total_txns_matched
            ),
        }

    def _generate_run_id(self) -> str:
        """Generate a unique run ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Get next sequence number
        existing = self.get_run_history()
        seq = len(existing) + 1
        return f"run_{seq:03d}_{timestamp}"

    def _get_run_dir(self, run_id: str) -> Path:
        """Get directory for a specific run."""
        return self.results_dir / run_id

    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None

    def _append_to_history(self, summary: RunSummary) -> None:
        """Append a run summary to the history file."""
        write_header = not self.history_path.exists()

        with open(self.history_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(RunSummary.csv_headers())
            writer.writerow(summary.to_csv_row())

    def _create_detailed_report(self) -> DetailedRunReport:
        """Create detailed report for the current run."""
        metrics = self.current_metrics

        # Get files with errors
        files_with_errors = []
        for file_metric in metrics.get_files_with_errors():
            # Determine issue type from error breakdown
            issue = "UNKNOWN"
            if file_metric.error_breakdown:
                issue = max(
                    file_metric.error_breakdown.items(),
                    key=lambda x: x[1],
                )[0]

            files_with_errors.append({
                "file": file_metric.file_name,
                "accuracy": file_metric.accuracy,
                "matched": file_metric.matched_count,
                "gt": file_metric.gt_count,
                "issue": issue,
            })

        # Compare with previous run
        comparison = self.compare_with_previous(metrics)

        return DetailedRunReport(
            run_id=self.current_run_id,
            timestamp=self.current_run_start.isoformat(),
            git_commit=self._get_git_commit(),
            duration_seconds=metrics.duration_seconds,
            notes=self.current_notes,
            metrics={
                "files": {
                    "total": metrics.total_files,
                    "perfect": metrics.perfect_files,
                    "with_errors": metrics.total_files - metrics.perfect_files,
                },
                "transactions": {
                    "ground_truth": metrics.total_gt_transactions,
                    "extracted": metrics.total_extracted_transactions,
                    "matched": metrics.total_matched_transactions,
                    "missed": metrics.total_missed_transactions,
                    "extra": metrics.total_extra_transactions,
                    "accuracy": metrics.overall_accuracy,
                },
                "error_breakdown": metrics.error_breakdown,
                "by_bank": {k: v.to_dict() for k, v in metrics.by_bank.items()},
            },
            files_with_errors=files_with_errors,
            comparison_to_previous=comparison,
        )


# =============================================================================
# Convenience Functions
# =============================================================================

def load_run_history(results_dir: str | Path) -> list[RunSummary]:
    """
    Load run history from results directory.

    Args:
        results_dir: Path to results directory

    Returns:
        List of RunSummary objects
    """
    tracker = RunTracker(results_dir)
    return tracker.get_run_history()


def compare_two_runs(
    results_dir: str | Path,
    run_id_1: str,
    run_id_2: str,
) -> dict[str, Any]:
    """
    Compare two specific runs.

    Args:
        results_dir: Path to results directory
        run_id_1: First run ID
        run_id_2: Second run ID

    Returns:
        Comparison dictionary
    """
    tracker = RunTracker(results_dir)

    details_1 = tracker.get_run_details(run_id_1)
    details_2 = tracker.get_run_details(run_id_2)

    if details_1 is None or details_2 is None:
        raise ValueError("One or both runs not found")

    m1 = details_1.metrics
    m2 = details_2.metrics

    return {
        "run_1": run_id_1,
        "run_2": run_id_2,
        "accuracy_1": m1["transactions"]["accuracy"],
        "accuracy_2": m2["transactions"]["accuracy"],
        "accuracy_delta": m2["transactions"]["accuracy"] - m1["transactions"]["accuracy"],
        "files_perfect_1": m1["files"]["perfect"],
        "files_perfect_2": m2["files"]["perfect"],
        "files_perfect_delta": m2["files"]["perfect"] - m1["files"]["perfect"],
    }
