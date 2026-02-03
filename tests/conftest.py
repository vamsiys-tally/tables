"""
Pytest configuration and shared fixtures for the tables test suite.

This module provides:
- Common test fixtures (sample PDFs, mock data)
- Test markers configuration
- Helper functions for test data generation

Note on Ground Truth:
    Tests can use two approaches for establishing ground truth:
    1. Golden files: Pre-computed expected outputs stored as JSON
    2. LLM-assisted: For complex cases, LLMs can validate extraction accuracy

    The choice depends on test complexity and maintenance burden.
    Simple unit tests use hardcoded expectations; complex integration
    tests may use LLM validation for flexibility.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Generator

import pytest


# Test data directory
TEST_DATA_DIR = Path(__file__).parent / "data"
VALID_DATA_DIR = TEST_DATA_DIR / "valid"
INVALID_DATA_DIR = TEST_DATA_DIR / "invalid"
EDGE_CASES_DIR = TEST_DATA_DIR / "edge_cases"


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """
    Create a temporary directory for test files.

    Yields:
        Path to temporary directory (cleaned up after test)
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_pdf_content() -> bytes:
    """
    Minimal valid PDF content for testing.

    Returns:
        Bytes of a minimal valid PDF
    """
    # Minimal PDF structure (empty page)
    return b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>
endobj
xref
0 4
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
trailer
<< /Size 4 /Root 1 0 R >>
startxref
196
%%EOF"""


@pytest.fixture
def valid_pdf_path(temp_dir: Path, sample_pdf_content: bytes) -> Path:
    """
    Create a valid PDF file for testing.

    Args:
        temp_dir: Temporary directory fixture
        sample_pdf_content: PDF content fixture

    Returns:
        Path to the created PDF file
    """
    pdf_path = temp_dir / "test.pdf"
    pdf_path.write_bytes(sample_pdf_content)
    return pdf_path


@pytest.fixture
def empty_file_path(temp_dir: Path) -> Path:
    """
    Create an empty file for testing.

    Args:
        temp_dir: Temporary directory fixture

    Returns:
        Path to the empty file
    """
    empty_path = temp_dir / "empty.pdf"
    empty_path.touch()
    return empty_path


@pytest.fixture
def non_pdf_file_path(temp_dir: Path) -> Path:
    """
    Create a non-PDF file with wrong extension for testing.

    Args:
        temp_dir: Temporary directory fixture

    Returns:
        Path to the non-PDF file
    """
    non_pdf_path = temp_dir / "fake.pdf"
    non_pdf_path.write_text("This is not a PDF file")
    return non_pdf_path


@pytest.fixture
def png_file_path(temp_dir: Path) -> Path:
    """
    Create a PNG file for testing file type detection.

    Args:
        temp_dir: Temporary directory fixture

    Returns:
        Path to the PNG file
    """
    png_path = temp_dir / "image.png"
    # Minimal PNG header
    png_content = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
    png_path.write_bytes(png_content)
    return png_path


@pytest.fixture
def large_file_path(temp_dir: Path) -> Path:
    """
    Create a large file (>100MB) for testing size limits.

    Note: This creates a sparse file to avoid disk usage.

    Args:
        temp_dir: Temporary directory fixture

    Returns:
        Path to the large file
    """
    large_path = temp_dir / "large.pdf"
    # Create a 101MB file
    with open(large_path, "wb") as f:
        f.write(b"%PDF")  # Valid PDF header
        f.seek(101 * 1024 * 1024)
        f.write(b"\x00")
    return large_path


def load_golden_file(name: str) -> dict[str, Any]:
    """
    Load a golden file (expected output) for comparison.

    Args:
        name: Name of the golden file (without extension)

    Returns:
        Dictionary with expected values

    Raises:
        FileNotFoundError: If golden file doesn't exist
    """
    golden_path = TEST_DATA_DIR / "golden" / f"{name}.json"
    if not golden_path.exists():
        raise FileNotFoundError(f"Golden file not found: {golden_path}")

    with open(golden_path) as f:
        return json.load(f)


def save_golden_file(name: str, data: dict[str, Any]) -> None:
    """
    Save a golden file for future comparisons.

    Useful for generating expected outputs during test development.

    Args:
        name: Name of the golden file (without extension)
        data: Dictionary to save
    """
    golden_dir = TEST_DATA_DIR / "golden"
    golden_dir.mkdir(exist_ok=True)

    golden_path = golden_dir / f"{name}.json"
    with open(golden_path, "w") as f:
        json.dump(data, f, indent=2, default=str)


# Pytest markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "regression: Bank-specific regression tests")
    config.addinivalue_line("markers", "slow: Tests that take > 1s")
    config.addinivalue_line("markers", "ml: Tests requiring ML models")
    config.addinivalue_line("markers", "requires_pdf: Tests requiring real PDF files")
