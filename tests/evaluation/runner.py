"""
Evaluation Runner Module.

Main CLI for running extraction evaluation against ground truth.

Commands:
    - Run evaluation on all ground truth files
    - Run on specific bank or file
    - Compare runs
    - Generate coverage report

Example:
    $ python -m tests.evaluation.runner
    $ python -m tests.evaluation.runner --bank hdfc
    $ python -m tests.evaluation.runner --file hdfc_001.pdf
    $ python -m tests.evaluation.runner --compare run_001 run_002
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Any
import logging

from tests.evaluation.csv_loader import (
    load_ground_truth,
    discover_ground_truth_files,
    GroundTruthFile,
)
from tests.evaluation.comparator import TransactionComparator, ComparisonResult
from tests.evaluation.metrics import (
    AccuracyMetrics,
    calculate_accuracy,
    extract_bank_from_path,
)
from tests.evaluation.report_generator import ReportGenerator
from tests.evaluation.run_tracker import RunTracker, compare_two_runs

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_GROUND_TRUTH_DIR = "tests/data/ground_truth"
DEFAULT_RESULTS_DIR = "tests/results"


# =============================================================================
# Evaluation Runner
# =============================================================================

class EvaluationRunner:
    """
    Main evaluation runner.

    Orchestrates:
        1. Ground truth discovery
        2. PDF extraction
        3. Comparison
        4. Metrics calculation
        5. Report generation
        6. Run tracking

    Example:
        >>> runner = EvaluationRunner()
        >>> results = runner.run()
        >>> print(f"Accuracy: {results.overall_accuracy:.2%}")
    """

    def __init__(
        self,
        ground_truth_dir: str | Path = DEFAULT_GROUND_TRUTH_DIR,
        results_dir: str | Path = DEFAULT_RESULTS_DIR,
        generate_reports: bool = True,
    ):
        """
        Initialize the evaluation runner.

        Args:
            ground_truth_dir: Directory containing ground truth CSVs
            results_dir: Directory for storing results
            generate_reports: Whether to generate Excel error reports
        """
        self.ground_truth_dir = Path(ground_truth_dir)
        self.results_dir = Path(results_dir)
        self.generate_reports = generate_reports

        # Initialize components
        self.comparator = TransactionComparator()
        self.tracker = RunTracker(results_dir)
        self.report_generator = None
        if generate_reports:
            try:
                self.report_generator = ReportGenerator(
                    output_dir=self.results_dir / "errors"
                )
            except ImportError:
                logger.warning(
                    "openpyxl not installed. Excel reports will be skipped."
                )

    def run(
        self,
        bank: Optional[str] = None,
        file_name: Optional[str] = None,
        notes: str = "",
    ) -> AccuracyMetrics:
        """
        Run evaluation.

        Args:
            bank: Filter by bank name (e.g., "hdfc")
            file_name: Run on specific file only
            notes: Notes for this run

        Returns:
            AccuracyMetrics with results
        """
        # Start tracking
        run_id = self.tracker.start_run(notes=notes)
        logger.info(f"Starting evaluation run: {run_id}")

        # Discover ground truth files
        if file_name:
            gt_files = self._find_file(file_name)
        else:
            gt_files = discover_ground_truth_files(self.ground_truth_dir, bank=bank)

        if not gt_files:
            logger.warning("No ground truth files found")
            return self.tracker.current_metrics

        logger.info(f"Found {len(gt_files)} ground truth files")

        # Process each file
        for gt_file in gt_files:
            try:
                result = self._evaluate_file(gt_file)
                if result:
                    # Add to tracker
                    file_bank = extract_bank_from_path(str(gt_file.file_path))
                    self.tracker.add_result(result, bank=file_bank)

                    # Generate error report if needed
                    if not result.is_perfect and self.report_generator:
                        self._generate_error_report(result, run_id)

            except Exception as e:
                logger.error(f"Error processing {gt_file.file_path}: {e}")

        # Finish run
        summary = self.tracker.finish_run()

        # Print summary
        print("\n" + self.tracker.current_metrics.summary() if hasattr(self.tracker, 'current_metrics') and self.tracker.current_metrics else "")

        return self.tracker.get_run_details(summary.run_id).metrics if self.tracker.get_run_details(summary.run_id) else {}

    def _find_file(self, file_name: str) -> list[GroundTruthFile]:
        """Find a specific ground truth file."""
        # Search in ground truth directory
        pattern = f"**/{file_name}"
        if not file_name.endswith(".csv"):
            pattern = f"**/{file_name}.csv"

        matches = list(self.ground_truth_dir.glob(pattern))

        if not matches:
            # Try with PDF extension
            pattern = f"**/{Path(file_name).stem}.csv"
            matches = list(self.ground_truth_dir.glob(pattern))

        return [
            GroundTruthFile(file_path=m)
            for m in matches
        ]

    def _evaluate_file(self, gt_file: GroundTruthFile) -> Optional[ComparisonResult]:
        """
        Evaluate a single file.

        Args:
            gt_file: Ground truth file to evaluate

        Returns:
            ComparisonResult or None if extraction fails
        """
        logger.info(f"Evaluating: {gt_file.file_path.name}")

        # Load ground truth
        gt_loaded = load_ground_truth(gt_file.file_path)
        if gt_loaded.has_errors:
            logger.warning(
                f"Errors loading ground truth: {gt_loaded.parse_errors}"
            )

        if not gt_loaded.transactions:
            logger.warning(f"No transactions in ground truth: {gt_file.file_path}")
            return None

        # Find corresponding PDF
        pdf_path = gt_file.file_path.with_suffix(".pdf")
        if not pdf_path.exists():
            # Try in same directory
            pdf_path = gt_file.file_path.parent / f"{gt_file.file_path.stem}.pdf"

        if not pdf_path.exists():
            logger.warning(f"PDF not found for: {gt_file.file_path}")
            # Return comparison with empty extraction (all missed)
            return self.comparator.compare(
                gt_loaded.transactions,
                [],
                file_name=gt_file.file_path.name,
            )

        # Extract transactions from PDF
        extracted = self._extract_from_pdf(pdf_path)

        # Compare
        result = self.comparator.compare(
            gt_loaded.transactions,
            extracted,
            file_name=gt_file.file_path.name,
        )

        logger.info(
            f"  Result: {result.matched_count}/{result.gt_count} "
            f"({result.accuracy:.1%})"
        )

        return result

    def _extract_from_pdf(self, pdf_path: Path) -> list[dict[str, Any]]:
        """
        Extract transactions from a PDF file.

        This method should be implemented to call the actual extraction pipeline.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of extracted transaction dictionaries
        """
        # TODO: Integrate with actual extraction pipeline
        # For now, return empty list - extraction not yet implemented
        #
        # Example integration:
        # from tables.reader.file_classifier import classify_file
        # from tables.detector.table_detector import TableDetector
        # from tables.recognizer.cell_extractor import CellExtractor
        #
        # classification = classify_file(pdf_path)
        # if classification.status != "success":
        #     return []
        #
        # detector = TableDetector()
        # tables = detector.detect(classification.pdf_document)
        #
        # extractor = CellExtractor()
        # transactions = []
        # for table in tables:
        #     rows = extractor.extract(table)
        #     for row in rows:
        #         transactions.append(row.to_dict())
        #
        # return transactions

        logger.debug(f"Extraction not yet integrated for: {pdf_path}")
        return []

    def _generate_error_report(
        self,
        result: ComparisonResult,
        run_id: str,
    ) -> None:
        """Generate error report for a file with errors."""
        if self.report_generator is None:
            return

        run_errors_dir = self.results_dir / run_id / "errors"
        run_errors_dir.mkdir(parents=True, exist_ok=True)

        output_path = run_errors_dir / f"{Path(result.file_name).stem}_errors.xlsx"

        try:
            self.report_generator.generate_file_report(result, output_path)
        except Exception as e:
            logger.error(f"Failed to generate error report: {e}")


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run extraction evaluation against ground truth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run evaluation on all ground truth files
  python -m tests.evaluation.runner

  # Run on specific bank
  python -m tests.evaluation.runner --bank hdfc

  # Run on specific file
  python -m tests.evaluation.runner --file hdfc_001.pdf

  # Compare two runs
  python -m tests.evaluation.runner --compare run_001 run_002

  # Show run history
  python -m tests.evaluation.runner --history
        """,
    )

    parser.add_argument(
        "--bank",
        help="Filter by bank name (e.g., hdfc, icici, sbi)",
    )
    parser.add_argument(
        "--file",
        help="Run on specific file only",
    )
    parser.add_argument(
        "--notes",
        default="",
        help="Notes for this run",
    )
    parser.add_argument(
        "--ground-truth-dir",
        default=DEFAULT_GROUND_TRUTH_DIR,
        help=f"Ground truth directory (default: {DEFAULT_GROUND_TRUTH_DIR})",
    )
    parser.add_argument(
        "--results-dir",
        default=DEFAULT_RESULTS_DIR,
        help=f"Results directory (default: {DEFAULT_RESULTS_DIR})",
    )
    parser.add_argument(
        "--no-reports",
        action="store_true",
        help="Skip generating Excel error reports",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("RUN1", "RUN2"),
        help="Compare two runs",
    )
    parser.add_argument(
        "--history",
        action="store_true",
        help="Show run history",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Handle compare command
    if args.compare:
        run_compare(args.results_dir, args.compare[0], args.compare[1])
        return

    # Handle history command
    if args.history:
        run_history(args.results_dir)
        return

    # Run evaluation
    runner = EvaluationRunner(
        ground_truth_dir=args.ground_truth_dir,
        results_dir=args.results_dir,
        generate_reports=not args.no_reports,
    )

    metrics = runner.run(
        bank=args.bank,
        file_name=args.file,
        notes=args.notes,
    )

    # Exit with error code if accuracy is below threshold
    if isinstance(metrics, dict):
        accuracy = metrics.get("transactions", {}).get("accuracy", 0)
    else:
        accuracy = metrics.overall_accuracy if hasattr(metrics, 'overall_accuracy') else 0

    if accuracy < 1.0:
        sys.exit(1)


def run_compare(results_dir: str, run_id_1: str, run_id_2: str) -> None:
    """Compare two runs and print results."""
    try:
        comparison = compare_two_runs(results_dir, run_id_1, run_id_2)

        print("\n" + "=" * 60)
        print("RUN COMPARISON")
        print("=" * 60)
        print(f"\nRun 1: {comparison['run_1']}")
        print(f"  Accuracy: {comparison['accuracy_1']:.2%}")
        print(f"  Perfect Files: {comparison['files_perfect_1']}")

        print(f"\nRun 2: {comparison['run_2']}")
        print(f"  Accuracy: {comparison['accuracy_2']:.2%}")
        print(f"  Perfect Files: {comparison['files_perfect_2']}")

        print("\nDelta:")
        delta = comparison['accuracy_delta']
        delta_str = f"+{delta:.2%}" if delta > 0 else f"{delta:.2%}"
        print(f"  Accuracy: {delta_str}")

        files_delta = comparison['files_perfect_delta']
        files_delta_str = f"+{files_delta}" if files_delta > 0 else str(files_delta)
        print(f"  Perfect Files: {files_delta_str}")

        print("=" * 60)

    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)


def run_history(results_dir: str) -> None:
    """Show run history."""
    tracker = RunTracker(results_dir)
    history = tracker.get_run_history()

    if not history:
        print("No runs found.")
        return

    print("\n" + "=" * 80)
    print("RUN HISTORY")
    print("=" * 80)
    print(
        f"{'Run ID':<25} {'Timestamp':<20} {'Files':<8} "
        f"{'Perfect':<8} {'Accuracy':<10} {'Notes'}"
    )
    print("-" * 80)

    for run in history:
        print(
            f"{run.run_id:<25} "
            f"{run.timestamp[:19]:<20} "
            f"{run.total_files:<8} "
            f"{run.files_100pct:<8} "
            f"{run.overall_accuracy:.2%}     "
            f"{run.notes[:20]}"
        )

    print("=" * 80)


if __name__ == "__main__":
    main()
