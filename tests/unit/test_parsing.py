"""
Unit tests for parsing utilities.

Tests date parsing, amount parsing (including Indian formats),
and text cleaning functions.
"""

import pytest
from datetime import date
from decimal import Decimal

from tables.utils.parsing import (
    parse_amount,
    parse_amount_detailed,
    parse_date,
    parse_date_detailed,
    clean_text,
    clean_description,
    is_empty_cell,
    extract_reference,
    normalize_amount,
    AmountParseResult,
    DateParseResult,
)


# =============================================================================
# Amount Parsing Tests
# =============================================================================

class TestParseAmount:
    """Tests for parse_amount function."""

    @pytest.mark.parametrize("input_str,expected", [
        # Standard formats
        ("1000", Decimal("1000")),
        ("1000.00", Decimal("1000.00")),
        ("1000.50", Decimal("1000.50")),
        ("1,000.00", Decimal("1000.00")),
        ("10,000.00", Decimal("10000.00")),
        ("100,000.00", Decimal("100000.00")),
        ("1,000,000.00", Decimal("1000000.00")),
    ])
    def test_standard_formats(self, input_str, expected):
        """Test standard number formats."""
        result = parse_amount(input_str)
        assert result == expected

    @pytest.mark.parametrize("input_str,expected", [
        # Indian formats (lakhs)
        ("1,00,000", Decimal("100000")),
        ("1,00,000.00", Decimal("100000.00")),
        ("10,00,000", Decimal("1000000")),
        ("1,00,00,000", Decimal("10000000")),
        ("50,000", Decimal("50000")),
        ("5,00,000.50", Decimal("500000.50")),
    ])
    def test_indian_lakh_format(self, input_str, expected):
        """Test Indian lakh number formats."""
        result = parse_amount(input_str)
        assert result == expected

    @pytest.mark.parametrize("input_str,expected", [
        # Currency symbols
        ("₹ 1000", Decimal("1000")),
        ("₹1000", Decimal("1000")),
        ("Rs 1000", Decimal("1000")),
        ("Rs. 1000", Decimal("1000")),
        ("INR 1000", Decimal("1000")),
        ("$ 100.00", Decimal("100.00")),
        ("₹ 1,00,000.00", Decimal("100000.00")),
    ])
    def test_currency_symbols(self, input_str, expected):
        """Test amounts with currency symbols."""
        result = parse_amount(input_str)
        assert result == expected

    @pytest.mark.parametrize("input_str,expected", [
        # Negative formats
        ("-1000", Decimal("-1000")),
        ("−1000", Decimal("-1000")),  # Unicode minus
        ("(1000)", Decimal("-1000")),
        ("(1,000.00)", Decimal("-1000.00")),
        ("1000-", Decimal("-1000")),
        ("1000 Dr", Decimal("-1000")),
        ("1000Dr", Decimal("-1000")),
    ])
    def test_negative_formats(self, input_str, expected):
        """Test negative amount formats."""
        result = parse_amount(input_str)
        assert result == expected

    @pytest.mark.parametrize("input_str,expected", [
        # Credit indicators (positive)
        ("1000 Cr", Decimal("1000")),
        ("1000Cr", Decimal("1000")),
    ])
    def test_credit_indicators(self, input_str, expected):
        """Test credit indicator formats."""
        result = parse_amount(input_str)
        assert result == expected

    @pytest.mark.parametrize("input_str", [
        "",
        "   ",
        "abc",
        "not a number",
        "12.34.56",
    ])
    def test_invalid_amounts(self, input_str):
        """Test that invalid inputs return None."""
        result = parse_amount(input_str)
        assert result is None

    def test_detailed_result_success(self):
        """Test detailed result on successful parse."""
        result = parse_amount_detailed("₹ 1,00,000.50")
        assert result.success is True
        assert result.value == Decimal("100000.50")
        assert result.is_negative is False
        assert result.original == "₹ 1,00,000.50"

    def test_detailed_result_negative(self):
        """Test detailed result for negative amount."""
        result = parse_amount_detailed("(5,000.00)")
        assert result.success is True
        assert result.is_negative is True
        assert result.value == Decimal("-5000.00")

    def test_detailed_result_failure(self):
        """Test detailed result on parse failure."""
        result = parse_amount_detailed("invalid")
        assert result.success is False
        assert result.value is None
        assert result.error is not None


class TestNormalizeAmount:
    """Tests for normalize_amount function."""

    def test_normalize_to_two_decimal_places(self):
        """Test normalizing to 2 decimal places."""
        result = normalize_amount(Decimal("100.5"), 2)
        assert result == Decimal("100.50")

    def test_normalize_whole_number(self):
        """Test normalizing whole number."""
        result = normalize_amount(Decimal("1000"), 2)
        assert result == Decimal("1000.00")


# =============================================================================
# Date Parsing Tests
# =============================================================================

class TestParseDate:
    """Tests for parse_date function."""

    @pytest.mark.parametrize("input_str,expected", [
        # DD/MM/YYYY formats
        ("15/01/2024", date(2024, 1, 15)),
        ("01/12/2024", date(2024, 12, 1)),
        ("1/1/2024", date(2024, 1, 1)),
        ("15-01-2024", date(2024, 1, 15)),
    ])
    def test_ddmmyyyy_formats(self, input_str, expected):
        """Test DD/MM/YYYY date formats."""
        result = parse_date(input_str)
        assert result == expected

    @pytest.mark.parametrize("input_str,expected", [
        # DD/MM/YY formats
        ("15/01/24", date(2024, 1, 15)),
        ("15-01-24", date(2024, 1, 15)),
        ("01/12/23", date(2023, 12, 1)),
    ])
    def test_ddmmyy_formats(self, input_str, expected):
        """Test DD/MM/YY date formats."""
        result = parse_date(input_str)
        assert result == expected

    @pytest.mark.parametrize("input_str,expected", [
        # DD-MMM-YYYY formats
        ("15-Jan-2024", date(2024, 1, 15)),
        ("15 Jan 2024", date(2024, 1, 15)),
        ("1-January-2024", date(2024, 1, 1)),
        ("15-DEC-2024", date(2024, 12, 15)),
        ("15/Feb/2024", date(2024, 2, 15)),
    ])
    def test_month_name_formats(self, input_str, expected):
        """Test date formats with month names."""
        result = parse_date(input_str)
        assert result == expected

    @pytest.mark.parametrize("input_str,expected", [
        # ISO formats
        ("2024-01-15", date(2024, 1, 15)),
        ("2024/01/15", date(2024, 1, 15)),
    ])
    def test_iso_formats(self, input_str, expected):
        """Test ISO date formats."""
        result = parse_date(input_str)
        assert result == expected

    @pytest.mark.parametrize("input_str", [
        "",
        "   ",
        "not a date",
        "32/01/2024",  # Invalid day
        "15/13/2024",  # Invalid month
    ])
    def test_invalid_dates(self, input_str):
        """Test that invalid inputs return None."""
        result = parse_date(input_str)
        assert result is None

    def test_detailed_result_success(self):
        """Test detailed result on successful parse."""
        result = parse_date_detailed("15/01/2024")
        assert result.success is True
        assert result.value == date(2024, 1, 15)
        assert result.format_used is not None
        assert result.original == "15/01/2024"

    def test_detailed_result_failure(self):
        """Test detailed result on parse failure."""
        result = parse_date_detailed("invalid")
        assert result.success is False
        assert result.value is None
        assert result.error is not None


# =============================================================================
# Text Cleaning Tests
# =============================================================================

class TestCleanText:
    """Tests for clean_text function."""

    def test_strips_whitespace(self):
        """Test that leading/trailing whitespace is stripped."""
        assert clean_text("  hello  ") == "hello"

    def test_normalizes_internal_whitespace(self):
        """Test that multiple spaces become single space."""
        assert clean_text("hello   world") == "hello world"

    def test_handles_tabs_and_newlines(self):
        """Test handling of tabs and newlines."""
        assert clean_text("hello\tworld\ntest") == "hello world test"

    def test_empty_string(self):
        """Test empty string handling."""
        assert clean_text("") == ""
        assert clean_text("   ") == ""

    def test_none_input(self):
        """Test None input handling."""
        assert clean_text(None) == ""


class TestCleanDescription:
    """Tests for clean_description function."""

    def test_normalizes_dashes(self):
        """Test that different dash characters are normalized."""
        assert clean_description("ATM – Withdrawal") == "ATM - Withdrawal"
        assert clean_description("ATM — Withdrawal") == "ATM - Withdrawal"

    def test_removes_multiple_dashes(self):
        """Test removal of multiple consecutive dashes."""
        assert clean_description("ATM -- Withdrawal") == "ATM - Withdrawal"

    def test_removes_trailing_punctuation(self):
        """Test removal of trailing punctuation."""
        assert clean_description("Withdrawal.") == "Withdrawal"
        assert clean_description("Withdrawal,") == "Withdrawal"


class TestIsEmptyCell:
    """Tests for is_empty_cell function."""

    @pytest.mark.parametrize("input_str", [
        "",
        "   ",
        "-",
        "--",
        "---",
        "nil",
        "NIL",
        "n/a",
        "N/A",
        "NA",
        "none",
        "None",
        ".",
    ])
    def test_empty_values(self, input_str):
        """Test that empty/placeholder values are detected."""
        assert is_empty_cell(input_str) is True

    @pytest.mark.parametrize("input_str", [
        "hello",
        "1000",
        "15/01/2024",
        "ATM Withdrawal",
    ])
    def test_non_empty_values(self, input_str):
        """Test that actual values are not detected as empty."""
        assert is_empty_cell(input_str) is False

    def test_none_input(self):
        """Test None input."""
        assert is_empty_cell(None) is True


class TestExtractReference:
    """Tests for extract_reference function."""

    def test_extract_utr(self):
        """Test UTR number extraction."""
        result = extract_reference("UTR: HDFC24012345678901234")
        assert result == "HDFC24012345678901234"

    def test_extract_cheque_number(self):
        """Test cheque number extraction."""
        result = extract_reference("CHQ NO: 123456")
        assert result == "CHQ123456"
        result = extract_reference("Cheque 789012")
        assert result == "CHQ789012"

    def test_alphanumeric_reference(self):
        """Test plain alphanumeric reference."""
        result = extract_reference("REF123456")
        assert result == "REF123456"

    def test_empty_input(self):
        """Test empty/nil inputs."""
        assert extract_reference("") is None
        assert extract_reference("-") is None
        assert extract_reference("nil") is None


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_amount_with_spaces(self):
        """Test amount with spaces in number."""
        # Some formats use spaces: "1 000 000"
        result = parse_amount("1 000 000")
        assert result == Decimal("1000000")

    def test_date_swapped_format(self):
        """Test that ambiguous dates default to DD/MM."""
        # 01/02/2024 should be interpreted as Feb 1, not Jan 2
        result = parse_date("01/02/2024")
        assert result == date(2024, 2, 1)

    def test_two_digit_year_boundary(self):
        """Test two-digit year handling at boundary."""
        # Years < 50 should be 20xx
        result = parse_date("15/01/49")
        assert result == date(2049, 1, 15)
        # Years >= 50 should be 19xx
        result = parse_date("15/01/99")
        assert result == date(1999, 1, 15)
