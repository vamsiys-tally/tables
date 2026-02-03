"""
Unit tests for unbordered table column detection.

Tests the whitespace gap detection algorithm for column boundary
detection in tables without visible lines.
"""

import pytest
from unittest.mock import Mock, MagicMock

from tables.detector.table_detector import (
    TableDetector,
    TableRegion,
    TextBlock,
)
from tables.detector.header_detector import HeaderMatch, HeaderRow
from tables.detector.structure_classifier import StructureAnalysis
from tables.models.table import StructureType, ColumnDefinition
from tables.utils.geometry import BoundingBox


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def detector():
    """Create a table detector instance."""
    return TableDetector(use_slm=False)


@pytest.fixture
def sample_header_row():
    """Create a sample header row with 4 columns."""
    matches = [
        HeaderMatch(
            text="Date",
            semantic_type="date",
            confidence=1.0,
            match_method="keyword",
            column_index=0,
        ),
        HeaderMatch(
            text="Description",
            semantic_type="description",
            confidence=1.0,
            match_method="keyword",
            column_index=1,
        ),
        HeaderMatch(
            text="Debit",
            semantic_type="debit",
            confidence=1.0,
            match_method="keyword",
            column_index=2,
        ),
        HeaderMatch(
            text="Credit",
            semantic_type="credit",
            confidence=1.0,
            match_method="keyword",
            column_index=3,
        ),
    ]
    header_row = HeaderRow(row_index=0, matches=matches)
    header_row._calculate_metrics()
    header_row.is_header = True
    return header_row


def create_text_block(text: str, x0: float, x1: float, y0: float, y1: float) -> TextBlock:
    """Helper to create a text block."""
    return TextBlock(
        text=text,
        bbox=BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1),
        page=0,
    )


# =============================================================================
# Test Gap Detection Algorithm
# =============================================================================

class TestFindColumnBoundariesFromGaps:
    """Tests for whitespace gap detection."""

    def test_clear_gaps_between_columns(self, detector):
        """Test detection of clear gaps between columns."""
        # 4 columns with clear gaps
        x_positions = [
            (50, 100),    # Column 1
            (50, 100),
            (150, 300),   # Column 2 (wider)
            (150, 280),
            (350, 400),   # Column 3
            (360, 410),
            (450, 500),   # Column 4
            (455, 505),
        ]
        region_bounds = BoundingBox(x0=40, y0=0, x1=520, y1=100)

        boundaries = detector._find_column_boundaries_from_gaps(
            x_positions, region_bounds
        )

        # Should detect 4 columns
        assert len(boundaries) == 4

        # Each column should have reasonable boundaries
        # Column 1: should cover 50-100 range
        assert boundaries[0][0] < 55
        assert boundaries[0][1] > 95

        # Column 4: should cover 450-505 range
        assert boundaries[3][0] < 455
        assert boundaries[3][1] > 500

    def test_overlapping_positions(self, detector):
        """Test with overlapping text positions (same column)."""
        # Multiple texts in same column region
        x_positions = [
            (50, 100),
            (55, 110),   # Overlaps with first
            (52, 105),   # Also overlaps
            (200, 300),  # Different column
            (205, 290),
        ]
        region_bounds = BoundingBox(x0=40, y0=0, x1=320, y1=100)

        boundaries = detector._find_column_boundaries_from_gaps(
            x_positions, region_bounds
        )

        # Should detect 2 columns
        assert len(boundaries) == 2

    def test_single_column(self, detector):
        """Test with single column (no gaps)."""
        x_positions = [
            (100, 200),
            (100, 200),
            (100, 200),
        ]
        region_bounds = BoundingBox(x0=50, y0=0, x1=250, y1=100)

        boundaries = detector._find_column_boundaries_from_gaps(
            x_positions, region_bounds
        )

        # Should detect 1 column
        assert len(boundaries) == 1
        assert boundaries[0][0] < 105
        assert boundaries[0][1] > 195

    def test_empty_positions(self, detector):
        """Test with empty positions list."""
        boundaries = detector._find_column_boundaries_from_gaps(
            [], BoundingBox(x0=0, y0=0, x1=100, y1=100)
        )
        assert boundaries == []


# =============================================================================
# Test Unbordered Column Detection
# =============================================================================

class TestUnborderedColumnDetection:
    """Tests for unbordered table column detection."""

    def test_detect_columns_by_gaps(self, detector, sample_header_row):
        """Test column detection using gap analysis."""
        # Create header blocks
        header_blocks = [
            create_text_block("Date", 50, 100, 10, 20),
            create_text_block("Description", 150, 280, 10, 20),
            create_text_block("Debit", 350, 400, 10, 20),
            create_text_block("Credit", 450, 510, 10, 20),
        ]

        # Create data rows with clear column separation
        data_row1 = [
            create_text_block("01/01/24", 52, 98, 30, 40),
            create_text_block("ATM Withdrawal from XYZ", 155, 295, 30, 40),
            create_text_block("5000.00", 355, 398, 30, 40),
            create_text_block("", 450, 450, 30, 40),  # Empty
        ]
        data_row2 = [
            create_text_block("02/01/24", 50, 95, 50, 60),
            create_text_block("Salary Credit", 152, 250, 50, 60),
            create_text_block("", 350, 350, 50, 60),  # Empty
            create_text_block("50000.00", 455, 520, 50, 60),
        ]

        # Create unbordered region
        structure = StructureAnalysis()
        structure.structure_type = StructureType.UNBORDERED
        structure.confidence = 0.8

        region = TableRegion(
            bounds=BoundingBox(x0=40, y0=0, x1=540, y1=100),
            page=0,
            text_blocks=header_blocks + data_row1 + data_row2,
            rows=[header_blocks, data_row1, data_row2],
            structure=structure,
        )

        columns = detector._detect_columns_by_gaps(
            region, [sample_header_row], header_blocks, sample_header_row
        )

        assert len(columns) == 4
        assert columns[0].semantic_type == "date"
        assert columns[1].semantic_type == "description"
        assert columns[2].semantic_type == "debit"
        assert columns[3].semantic_type == "credit"

    def test_fallback_to_alignment_when_no_data_rows(
        self, detector, sample_header_row
    ):
        """Test fallback when no data rows available."""
        header_blocks = [
            create_text_block("Date", 50, 100, 10, 20),
            create_text_block("Description", 150, 280, 10, 20),
        ]

        structure = StructureAnalysis()
        structure.structure_type = StructureType.UNBORDERED

        region = TableRegion(
            bounds=BoundingBox(x0=40, y0=0, x1=300, y1=100),
            page=0,
            rows=[header_blocks],  # Only header row
            structure=structure,
        )

        # Should fall back to alignment-based detection
        columns = detector._detect_columns_by_gaps(
            region,
            [sample_header_row],
            header_blocks,
            sample_header_row,
        )

        # Should still produce columns (via fallback)
        assert len(columns) >= 2


# =============================================================================
# Test Integration with _detect_columns
# =============================================================================

class TestDetectColumnsIntegration:
    """Integration tests for _detect_columns method."""

    def test_uses_gap_detection_for_unbordered(self, detector, sample_header_row):
        """Test that unbordered tables use gap detection."""
        header_blocks = [
            create_text_block("Date", 50, 100, 10, 20),
            create_text_block("Description", 200, 350, 10, 20),
        ]
        data_row = [
            create_text_block("01/01/24", 55, 95, 30, 40),
            create_text_block("Test transaction", 205, 340, 30, 40),
        ]

        structure = StructureAnalysis()
        structure.structure_type = StructureType.UNBORDERED

        region = TableRegion(
            bounds=BoundingBox(x0=40, y0=0, x1=400, y1=100),
            page=0,
            rows=[header_blocks, data_row],
            structure=structure,
        )

        # Adjust header row to match
        header_row = HeaderRow(
            row_index=0,
            matches=[
                HeaderMatch(text="Date", semantic_type="date", confidence=1.0,
                           match_method="keyword", column_index=0),
                HeaderMatch(text="Description", semantic_type="description",
                           confidence=1.0, match_method="keyword", column_index=1),
            ],
        )
        header_row._calculate_metrics()
        header_row.is_header = True

        columns = detector._detect_columns(region, [header_row])

        assert len(columns) == 2
        # Gap detection should produce cleaner boundaries
        assert columns[0].x0 < 60
        assert columns[1].x0 > 100  # Clear separation

    def test_uses_alignment_for_bordered(self, detector):
        """Test that bordered tables use alignment detection."""
        header_blocks = [
            create_text_block("Date", 50, 100, 10, 20),
            create_text_block("Amount", 150, 200, 10, 20),
        ]

        structure = StructureAnalysis()
        structure.structure_type = StructureType.BORDERED

        region = TableRegion(
            bounds=BoundingBox(x0=40, y0=0, x1=250, y1=100),
            page=0,
            rows=[header_blocks],
            structure=structure,
        )

        header_row = HeaderRow(
            row_index=0,
            matches=[
                HeaderMatch(text="Date", semantic_type="date", confidence=1.0,
                           match_method="keyword", column_index=0),
                HeaderMatch(text="Amount", semantic_type="amount", confidence=1.0,
                           match_method="keyword", column_index=1),
            ],
        )
        header_row._calculate_metrics()
        header_row.is_header = True

        columns = detector._detect_columns(region, [header_row])

        assert len(columns) == 2
        # Alignment detection should use header positions
        assert columns[0].header_text == "Date"
        assert columns[1].header_text == "Amount"


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Edge case tests for column detection."""

    def test_variable_width_data(self, detector, sample_header_row):
        """Test with highly variable width data in columns."""
        header_blocks = [
            create_text_block("Date", 50, 100, 10, 20),
            create_text_block("Description", 150, 280, 10, 20),
        ]

        # Description column has very variable width
        data_rows = [
            [
                create_text_block("01/01/24", 52, 98, 30, 40),
                create_text_block("A", 160, 170, 30, 40),  # Short
            ],
            [
                create_text_block("02/01/24", 50, 95, 50, 60),
                create_text_block("Very long description text here", 155, 350, 50, 60),
            ],
        ]

        structure = StructureAnalysis()
        structure.structure_type = StructureType.UNBORDERED

        region = TableRegion(
            bounds=BoundingBox(x0=40, y0=0, x1=400, y1=100),
            page=0,
            rows=[header_blocks] + data_rows,
            structure=structure,
        )

        header_row = HeaderRow(
            row_index=0,
            matches=[
                HeaderMatch(text="Date", semantic_type="date", confidence=1.0,
                           match_method="keyword", column_index=0),
                HeaderMatch(text="Description", semantic_type="description",
                           confidence=1.0, match_method="keyword", column_index=1),
            ],
        )
        header_row._calculate_metrics()
        header_row.is_header = True

        columns = detector._detect_columns(region, [header_row])

        assert len(columns) == 2
        # Description column should accommodate the widest entry
        assert columns[1].x1 > 340

    def test_no_structure_defaults_to_alignment(self, detector):
        """Test that missing structure uses alignment detection."""
        header_blocks = [
            create_text_block("Date", 50, 100, 10, 20),
        ]

        region = TableRegion(
            bounds=BoundingBox(x0=40, y0=0, x1=150, y1=100),
            page=0,
            rows=[header_blocks],
            structure=None,  # No structure analysis
        )

        header_row = HeaderRow(
            row_index=0,
            matches=[
                HeaderMatch(text="Date", semantic_type="date", confidence=1.0,
                           match_method="keyword", column_index=0),
            ],
        )
        header_row._calculate_metrics()
        header_row.is_header = True

        columns = detector._detect_columns(region, [header_row])

        # Should still work (falls back to alignment)
        assert len(columns) == 1
