"""
Unit tests for file validators.

Tests cover:
- File existence validation
- Empty file detection
- File size limits
- PDF magic byte verification
- File type detection
"""

import pytest
from pathlib import Path

from tables.reader.validators import FileValidator, validate_file
from tables.models.errors import ProcessingError, ErrorCode


class TestFileValidator:
    """Tests for the FileValidator class."""

    def test_validate_file_exists_success(self, valid_pdf_path: Path):
        """Test that validation passes for existing files."""
        validator = FileValidator()
        # Should not raise
        validator.validate_file_exists(str(valid_pdf_path))

    def test_validate_file_exists_not_found(self, temp_dir: Path):
        """Test that validation fails for non-existent files."""
        validator = FileValidator()
        non_existent = temp_dir / "does_not_exist.pdf"

        with pytest.raises(ProcessingError) as exc_info:
            validator.validate_file_exists(str(non_existent))

        assert exc_info.value.code == ErrorCode.FILE_NOT_FOUND
        assert str(non_existent) in exc_info.value.message

    def test_validate_file_not_empty_success(self, valid_pdf_path: Path):
        """Test that validation passes for non-empty files."""
        validator = FileValidator()
        file_size = validator.validate_file_not_empty(str(valid_pdf_path))
        assert file_size > 0

    def test_validate_file_not_empty_fails_for_empty(self, empty_file_path: Path):
        """Test that validation fails for empty files."""
        validator = FileValidator()

        with pytest.raises(ProcessingError) as exc_info:
            validator.validate_file_not_empty(str(empty_file_path))

        assert exc_info.value.code == ErrorCode.FILE_EMPTY

    def test_validate_file_size_within_limit(self, valid_pdf_path: Path):
        """Test that validation passes for files within size limit."""
        validator = FileValidator(max_size_bytes=10 * 1024 * 1024)  # 10MB
        file_size = validator.validate_file_size(str(valid_pdf_path))
        assert file_size > 0

    def test_validate_file_size_exceeds_limit(self, large_file_path: Path):
        """Test that validation fails for files exceeding size limit."""
        validator = FileValidator(max_size_bytes=100 * 1024 * 1024)  # 100MB

        with pytest.raises(ProcessingError) as exc_info:
            validator.validate_file_size(str(large_file_path))

        assert exc_info.value.code == ErrorCode.FILE_TOO_LARGE
        assert "100" in exc_info.value.message  # Max size in message

    def test_validate_file_type_valid_pdf(self, valid_pdf_path: Path):
        """Test that validation passes for valid PDF files."""
        validator = FileValidator()
        file_type = validator.validate_file_type(str(valid_pdf_path))
        assert file_type == "pdf"

    def test_validate_file_type_invalid_not_pdf(self, non_pdf_file_path: Path):
        """Test that validation fails for non-PDF files."""
        validator = FileValidator()

        with pytest.raises(ProcessingError) as exc_info:
            validator.validate_file_type(str(non_pdf_file_path))

        assert exc_info.value.code == ErrorCode.FILE_INVALID_TYPE

    def test_validate_file_type_detects_png(self, png_file_path: Path):
        """Test that PNG files are correctly identified in error message."""
        validator = FileValidator()

        with pytest.raises(ProcessingError) as exc_info:
            validator.validate_file_type(str(png_file_path))

        assert exc_info.value.code == ErrorCode.FILE_INVALID_TYPE
        assert "PNG" in exc_info.value.message

    def test_validate_all_success(self, valid_pdf_path: Path):
        """Test that validate_all passes for valid PDF files."""
        validator = FileValidator()
        file_size, file_type = validator.validate_all(str(valid_pdf_path))

        assert file_size > 0
        assert file_type == "pdf"

    def test_validate_all_fails_early_on_not_found(self, temp_dir: Path):
        """Test that validate_all fails fast on file not found."""
        validator = FileValidator()
        non_existent = temp_dir / "does_not_exist.pdf"

        with pytest.raises(ProcessingError) as exc_info:
            validator.validate_all(str(non_existent))

        assert exc_info.value.code == ErrorCode.FILE_NOT_FOUND


class TestValidateFileFunction:
    """Tests for the validate_file convenience function."""

    def test_validate_file_success(self, valid_pdf_path: Path):
        """Test convenience function with valid PDF."""
        file_size, file_type = validate_file(str(valid_pdf_path))

        assert file_size > 0
        assert file_type == "pdf"

    def test_validate_file_with_custom_max_size(self, valid_pdf_path: Path):
        """Test convenience function with custom max size."""
        # Set very small max size
        with pytest.raises(ProcessingError) as exc_info:
            validate_file(str(valid_pdf_path), max_size_bytes=10)

        assert exc_info.value.code == ErrorCode.FILE_TOO_LARGE

    def test_validate_file_not_found(self, temp_dir: Path):
        """Test convenience function with non-existent file."""
        non_existent = temp_dir / "missing.pdf"

        with pytest.raises(ProcessingError) as exc_info:
            validate_file(str(non_existent))

        assert exc_info.value.code == ErrorCode.FILE_NOT_FOUND


class TestFileTypeDetection:
    """Tests for file type detection from magic bytes."""

    def test_detect_pdf(self, valid_pdf_path: Path):
        """Test PDF detection from magic bytes."""
        validator = FileValidator()
        file_type = validator.validate_file_type(str(valid_pdf_path))
        assert file_type == "pdf"

    def test_detect_png_in_error(self, png_file_path: Path):
        """Test PNG detection in error message."""
        validator = FileValidator()

        with pytest.raises(ProcessingError) as exc_info:
            validator.validate_file_type(str(png_file_path))

        error_msg = exc_info.value.message.lower()
        assert "png" in error_msg

    def test_detect_unknown_type(self, temp_dir: Path):
        """Test handling of unknown file types."""
        unknown_path = temp_dir / "unknown.pdf"
        unknown_path.write_bytes(b"\x00\x01\x02\x03")  # Unknown magic bytes

        validator = FileValidator()

        with pytest.raises(ProcessingError) as exc_info:
            validator.validate_file_type(str(unknown_path))

        assert exc_info.value.code == ErrorCode.FILE_INVALID_TYPE
