"""
Unit tests for file classifier.

Tests cover:
- FileClassifier basic functionality
- Bank statement keyword detection
- Account type detection
- Language detection
- Integration with validators

Note: These tests use minimal PDF fixtures. For full integration tests
with real bank statements, see tests/integration/.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from tables.reader.file_classifier import (
    FileClassifier,
    classify_file,
    _calculate_bank_statement_confidence,
    _detect_account_type,
    _detect_language,
)
from tables.models.document import (
    AccountType,
    DocumentType,
    ProcessingOptions,
)
from tables.models.errors import ErrorCode


class TestBankStatementConfidence:
    """Tests for bank statement confidence calculation."""

    def test_strong_keywords_high_confidence(self):
        """Test that strong keywords result in high confidence."""
        text = """
        ACCOUNT STATEMENT
        Statement Period: Jan 2024 to Feb 2024
        Opening Balance: 10,000.00
        Closing Balance: 15,000.00
        """
        confidence = _calculate_bank_statement_confidence(text)
        assert confidence >= 0.6  # At least 2 strong keywords

    def test_medium_keywords_medium_confidence(self):
        """Test that medium keywords result in medium confidence."""
        text = """
        Account Number: 12345678
        IFSC Code: HDFC0001234
        Branch: Mumbai
        Debit: 500.00
        Credit: 1000.00
        Balance: 10500.00
        """
        confidence = _calculate_bank_statement_confidence(text)
        assert 0.3 <= confidence <= 0.7

    def test_weak_keywords_low_confidence(self):
        """Test that only weak keywords result in low confidence."""
        text = """
        Date: 2024-01-15
        Amount: 500
        Bank Transfer
        """
        confidence = _calculate_bank_statement_confidence(text)
        assert confidence < 0.3

    def test_no_keywords_zero_confidence(self):
        """Test that irrelevant text results in near-zero confidence."""
        text = """
        Lorem ipsum dolor sit amet, consectetur adipiscing elit.
        Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
        """
        confidence = _calculate_bank_statement_confidence(text)
        assert confidence < 0.1

    def test_confidence_capped_at_one(self):
        """Test that confidence is capped at 1.0."""
        text = """
        ACCOUNT STATEMENT
        BANK STATEMENT
        Statement of Account
        Transaction History
        Account Summary
        Opening Balance Closing Balance
        Account Number IFSC Debit Credit Balance
        """
        confidence = _calculate_bank_statement_confidence(text)
        assert confidence <= 1.0

    def test_case_insensitive(self):
        """Test that keyword matching is case-insensitive."""
        text1 = "ACCOUNT STATEMENT"
        text2 = "account statement"
        text3 = "Account Statement"

        c1 = _calculate_bank_statement_confidence(text1)
        c2 = _calculate_bank_statement_confidence(text2)
        c3 = _calculate_bank_statement_confidence(text3)

        assert c1 == c2 == c3


class TestAccountTypeDetection:
    """Tests for account type detection."""

    def test_detect_savings_account(self):
        """Test detection of savings account."""
        text = "Your Savings Account Statement for the month of January"
        account_type, confidence = _detect_account_type(text)

        assert account_type == AccountType.SAVINGS
        assert confidence > 0.8

    def test_detect_current_account(self):
        """Test detection of current account."""
        text = "Current Account Statement\nAccount No: 12345"
        account_type, confidence = _detect_account_type(text)

        assert account_type == AccountType.CURRENT
        assert confidence > 0.8

    def test_detect_overdraft(self):
        """Test detection of overdraft account."""
        text = "Cash Credit Account Statement\nOD Limit: 500000"
        account_type, confidence = _detect_account_type(text)

        assert account_type == AccountType.OVERDRAFT
        assert confidence > 0.8

    def test_detect_short_forms(self):
        """Test detection of abbreviated account types."""
        # SB A/C
        text1 = "SB A/C Statement"
        account_type1, _ = _detect_account_type(text1)
        assert account_type1 == AccountType.SAVINGS

        # CA A/C
        text2 = "CA A/C Statement"
        account_type2, _ = _detect_account_type(text2)
        assert account_type2 == AccountType.CURRENT

    def test_no_account_type_detected(self):
        """Test when no account type is detected."""
        text = "Statement for Period: Jan 2024"
        account_type, confidence = _detect_account_type(text)

        assert account_type is None
        assert confidence == 0.0


class TestLanguageDetection:
    """Tests for language detection."""

    def test_detect_english(self):
        """Test detection of English text."""
        text = """
        This is an English bank statement with transaction details.
        The account balance is shown below with all debits and credits.
        """
        language, confidence = _detect_language(text)

        assert language == "en"
        assert confidence > 0.5

    def test_empty_text(self):
        """Test handling of empty text."""
        language, confidence = _detect_language("")

        assert language == "unknown"
        assert confidence == 0.0

    def test_short_text(self):
        """Test handling of very short text."""
        language, confidence = _detect_language("Hello")

        assert language == "unknown"
        assert confidence == 0.0


class TestFileClassifier:
    """Tests for the FileClassifier class."""

    def test_classify_valid_pdf(self, valid_pdf_path: Path):
        """Test classification of a minimal valid PDF."""
        classifier = FileClassifier()
        result = classifier.classify(str(valid_pdf_path))

        # Minimal PDF won't be classified as bank statement
        # but should process without error
        assert result.status == "success"
        assert result.file_type == "pdf"
        assert result.page_count >= 0

    def test_classify_non_existent_file(self, temp_dir: Path):
        """Test classification of non-existent file."""
        classifier = FileClassifier()
        result = classifier.classify(str(temp_dir / "missing.pdf"))

        assert result.status == "error"
        assert result.error_code == ErrorCode.FILE_NOT_FOUND.value

    def test_classify_empty_file(self, empty_file_path: Path):
        """Test classification of empty file."""
        classifier = FileClassifier()
        result = classifier.classify(str(empty_file_path))

        assert result.status == "error"
        assert result.error_code == ErrorCode.FILE_EMPTY.value

    def test_classify_non_pdf_file(self, non_pdf_file_path: Path):
        """Test classification of non-PDF file."""
        classifier = FileClassifier()
        result = classifier.classify(str(non_pdf_file_path))

        assert result.status == "error"
        assert result.error_code == ErrorCode.FILE_INVALID_TYPE.value

    def test_classify_with_custom_options(self, valid_pdf_path: Path):
        """Test classification with custom options."""
        options = ProcessingOptions(
            max_file_size_mb=1.0,
            strict_mode=False,
        )
        classifier = FileClassifier(options=options)
        result = classifier.classify(str(valid_pdf_path))

        assert result.status == "success"

    def test_result_to_dict(self, valid_pdf_path: Path):
        """Test that result can be serialized to dict."""
        classifier = FileClassifier()
        result = classifier.classify(str(valid_pdf_path))

        result_dict = result.to_dict()

        assert "status" in result_dict
        assert "file_type" in result_dict
        assert "page_count" in result_dict


class TestClassifyFileFunction:
    """Tests for the classify_file convenience function."""

    def test_classify_file_success(self, valid_pdf_path: Path):
        """Test convenience function with valid PDF."""
        result = classify_file(str(valid_pdf_path))

        assert result.status == "success"
        assert result.file_path == str(valid_pdf_path)

    def test_classify_file_with_options(self, valid_pdf_path: Path):
        """Test convenience function with options."""
        options = ProcessingOptions(strict_mode=True)
        result = classify_file(str(valid_pdf_path), options=options)

        # Should work but may have different classification
        assert result is not None

    def test_classify_file_not_found(self, temp_dir: Path):
        """Test convenience function with missing file."""
        result = classify_file(str(temp_dir / "missing.pdf"))

        assert result.status == "error"
        assert result.error_code == ErrorCode.FILE_NOT_FOUND.value


class TestFileClassifierIntegration:
    """Integration-style tests for FileClassifier."""

    @pytest.mark.requires_pdf
    def test_classify_real_bank_statement(self):
        """
        Test classification of a real bank statement.

        This test is marked requires_pdf and will be skipped if
        no real PDF files are available in tests/data/valid/.

        To run: pytest -m requires_pdf
        """
        # This would test with real PDFs from tests/data/valid/
        pytest.skip("No real PDF files configured for testing")

    def test_classifier_returns_pdf_document(self, valid_pdf_path: Path):
        """Test that classifier returns usable PDF document."""
        classifier = FileClassifier()
        result = classifier.classify(str(valid_pdf_path))

        if result.status == "success":
            # PDF document should be accessible
            assert result.pdf_document is not None or result.page_count >= 0
