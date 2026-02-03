"""
Unit tests for error handling models.

Tests cover:
- ProcessingError creation and serialization
- Error code enumeration
- Message templating
- ProcessingWarning creation
"""

import pytest

from tables.models.errors import (
    ErrorCode,
    ProcessingError,
    ProcessingWarning,
    WarningCode,
    ERROR_MESSAGES,
    WARNING_MESSAGES,
)


class TestErrorCode:
    """Tests for the ErrorCode enumeration."""

    def test_error_codes_are_strings(self):
        """Test that error codes are string-based."""
        assert ErrorCode.FILE_NOT_FOUND.value == "FILE_NOT_FOUND"
        assert ErrorCode.FILE_CORRUPTED.value == "FILE_CORRUPTED"

    def test_all_error_codes_have_messages(self):
        """Test that all error codes have message templates."""
        for code in ErrorCode:
            assert code in ERROR_MESSAGES, f"Missing message for {code}"

    def test_error_messages_are_templates(self):
        """Test that error messages can be formatted."""
        # FILE_NOT_FOUND should have {path} placeholder
        msg = ERROR_MESSAGES[ErrorCode.FILE_NOT_FOUND]
        assert "{path}" in msg

        # FILE_TOO_LARGE should have {max_size} and {actual_size}
        msg = ERROR_MESSAGES[ErrorCode.FILE_TOO_LARGE]
        assert "{max_size}" in msg
        assert "{actual_size}" in msg


class TestProcessingError:
    """Tests for the ProcessingError exception class."""

    def test_create_error_with_details(self):
        """Test creating error with details for templating."""
        error = ProcessingError(
            ErrorCode.FILE_NOT_FOUND,
            details={"path": "/some/path.pdf"},
        )

        assert error.code == ErrorCode.FILE_NOT_FOUND
        assert "/some/path.pdf" in error.message
        assert error.details["path"] == "/some/path.pdf"

    def test_create_error_with_custom_message(self):
        """Test creating error with custom message."""
        error = ProcessingError(
            ErrorCode.FILE_CORRUPTED,
            message="Custom error message",
        )

        assert error.code == ErrorCode.FILE_CORRUPTED
        assert error.message == "Custom error message"

    def test_error_is_exception(self):
        """Test that ProcessingError is an Exception."""
        error = ProcessingError(ErrorCode.FILE_NOT_FOUND)
        assert isinstance(error, Exception)

    def test_error_can_be_raised(self):
        """Test that ProcessingError can be raised and caught."""
        with pytest.raises(ProcessingError) as exc_info:
            raise ProcessingError(
                ErrorCode.FILE_NOT_FOUND,
                details={"path": "/test.pdf"},
            )

        assert exc_info.value.code == ErrorCode.FILE_NOT_FOUND

    def test_error_to_dict(self):
        """Test serialization to dictionary."""
        error = ProcessingError(
            ErrorCode.FILE_TOO_LARGE,
            details={"max_size": "100", "actual_size": "150"},
        )

        result = error.to_dict()

        assert result["code"] == "FILE_TOO_LARGE"
        assert "message" in result
        assert result["details"]["max_size"] == "100"

    def test_error_repr(self):
        """Test string representation of error."""
        error = ProcessingError(ErrorCode.FILE_EMPTY)
        repr_str = repr(error)

        assert "ProcessingError" in repr_str
        assert "FILE_EMPTY" in repr_str

    def test_error_missing_template_params(self):
        """Test error handles missing template parameters gracefully."""
        # Creating error without required path param
        error = ProcessingError(ErrorCode.FILE_NOT_FOUND)

        # Should not raise, message should be the template
        assert "{path}" in error.message or "file" in error.message.lower()


class TestWarningCode:
    """Tests for the WarningCode enumeration."""

    def test_warning_codes_are_strings(self):
        """Test that warning codes are string-based."""
        assert WarningCode.LOW_CONFIDENCE.value == "LOW_CONFIDENCE"

    def test_all_warning_codes_have_messages(self):
        """Test that all warning codes have message templates."""
        for code in WarningCode:
            assert code in WARNING_MESSAGES, f"Missing message for {code}"


class TestProcessingWarning:
    """Tests for the ProcessingWarning dataclass."""

    def test_create_warning_with_factory(self):
        """Test creating warning with factory method."""
        warning = ProcessingWarning.create(
            WarningCode.LOW_CONFIDENCE,
            details={"confidence": 0.65},
        )

        assert warning.code == WarningCode.LOW_CONFIDENCE
        assert "0.65" in warning.message

    def test_create_warning_with_page(self):
        """Test creating warning with page number."""
        warning = ProcessingWarning.create(
            WarningCode.PAGE_SKIPPED,
            details={"page": 3, "reason": "scanned"},
            page=3,
        )

        assert warning.page == 3
        assert "3" in warning.message

    def test_warning_to_dict(self):
        """Test serialization to dictionary."""
        warning = ProcessingWarning.create(
            WarningCode.ROW_MERGED,
            details={"rows": "5-7"},
            row=5,
        )

        result = warning.to_dict()

        assert result["code"] == "ROW_MERGED"
        assert "message" in result
        assert result["row"] == 5

    def test_warning_to_dict_excludes_none(self):
        """Test that None values are excluded from dict."""
        warning = ProcessingWarning.create(
            WarningCode.LARGE_FILE,
            details={"page_count": 200},
        )

        result = warning.to_dict()

        # page and row should not be in dict if None
        assert "page" not in result
        assert "row" not in result

    def test_warning_direct_creation(self):
        """Test creating warning directly (not via factory)."""
        warning = ProcessingWarning(
            code=WarningCode.MIXED_CONTENT,
            message="Custom warning message",
            details={"info": "test"},
        )

        assert warning.code == WarningCode.MIXED_CONTENT
        assert warning.message == "Custom warning message"
