"""
File classification module for bank statement PDF processing.

This module provides the main entry point for classifying PDF files as bank
statements and extracting relevant metadata. It performs a series of validation
and classification steps:

1. File Validation: Check file exists, is valid PDF, not too large
2. Password Handling: Decrypt if password provided, prompt if needed
3. Content Analysis: Detect scanned vs generated, extract text
4. Language Detection: Verify document is in English
5. Document Classification: Determine if it's a bank statement
6. Account Type Detection: Classify as Savings/Current/OD

Example Usage:
    ```python
    from tables.reader.file_classifier import FileClassifier

    classifier = FileClassifier()

    # Classify a file
    result = classifier.classify("statement.pdf")

    if result.status == "success":
        print(f"Document type: {result.document_type}")
        print(f"Account type: {result.account_type}")
        print(f"Pages with tables: {result.pages_with_tables}")
        # Access PDF document for further processing
        pdf_doc = result.pdf_document
    else:
        print(f"Error: {result.error_code} - {result.error_message}")
    ```

For password-protected files:
    ```python
    result = classifier.classify("protected.pdf", password="secret")
    ```

Design Philosophy:
    - Fail fast on blocking errors (file not found, corrupted, etc.)
    - Warn and proceed for non-blocking issues (low confidence, etc.)
    - Return structured results with confidence scores
    - All classifications include confidence for transparency
"""

import os
from typing import Optional

from tables.models.document import (
    AccountType,
    DocumentType,
    FileClassification,
    ProcessingOptions,
)
from tables.models.errors import (
    ErrorCode,
    ProcessingError,
    ProcessingWarning,
    WarningCode,
)
from tables.reader.pdf_document import PDFDocument
from tables.reader.validators import FileValidator


# Keywords indicating bank statement content
BANK_STATEMENT_KEYWORDS = {
    # Strong indicators (high weight)
    "strong": [
        "account statement",
        "bank statement",
        "statement of account",
        "transaction history",
        "account summary",
        "opening balance",
        "closing balance",
        "statement period",
    ],
    # Medium indicators (medium weight)
    "medium": [
        "account number",
        "account no",
        "ifsc code",
        "branch",
        "customer id",
        "debit",
        "credit",
        "withdrawal",
        "deposit",
        "balance",
        "transaction",
        "particulars",
        "narration",
    ],
    # Weak indicators (low weight)
    "weak": [
        "bank",
        "date",
        "amount",
        "description",
        "reference",
    ],
}

# Keywords indicating account type
ACCOUNT_TYPE_KEYWORDS = {
    AccountType.SAVINGS: [
        "savings account",
        "savings a/c",
        "sb account",
        "sb a/c",
        "saving account",
    ],
    AccountType.CURRENT: [
        "current account",
        "current a/c",
        "ca account",
        "ca a/c",
    ],
    AccountType.OVERDRAFT: [
        "overdraft",
        "od account",
        "od a/c",
        "cash credit",
        "cc account",
        "cc a/c",
    ],
}


def _detect_language(text: str) -> tuple[str, float]:
    """
    Detect the language of the given text.

    Args:
        text: Text to analyze

    Returns:
        Tuple of (language_code, confidence)

    Note:
        Requires langdetect library. Falls back to "unknown" if detection fails.
    """
    if not text or len(text.strip()) < 20:
        return "unknown", 0.0

    try:
        from langdetect import detect_langs
        from langdetect.lang_detect_exception import LangDetectException

        results = detect_langs(text[:5000])  # Use first 5000 chars for efficiency
        if results:
            top_result = results[0]
            return top_result.lang, top_result.prob
        return "unknown", 0.0

    except LangDetectException:
        return "unknown", 0.0
    except ImportError:
        # langdetect not installed - assume English with low confidence
        return "en", 0.5


def _calculate_bank_statement_confidence(text: str) -> float:
    """
    Calculate confidence that the document is a bank statement.

    Uses keyword matching with weighted scoring:
    - Strong keywords: 0.3 each (max 0.6)
    - Medium keywords: 0.1 each (max 0.3)
    - Weak keywords: 0.02 each (max 0.1)

    Args:
        text: Document text to analyze

    Returns:
        Confidence score between 0.0 and 1.0
    """
    text_lower = text.lower()
    confidence = 0.0

    # Check strong indicators
    strong_matches = sum(1 for kw in BANK_STATEMENT_KEYWORDS["strong"] if kw in text_lower)
    confidence += min(strong_matches * 0.3, 0.6)

    # Check medium indicators
    medium_matches = sum(1 for kw in BANK_STATEMENT_KEYWORDS["medium"] if kw in text_lower)
    confidence += min(medium_matches * 0.1, 0.3)

    # Check weak indicators
    weak_matches = sum(1 for kw in BANK_STATEMENT_KEYWORDS["weak"] if kw in text_lower)
    confidence += min(weak_matches * 0.02, 0.1)

    return min(confidence, 1.0)


def _detect_account_type(text: str) -> tuple[Optional[AccountType], float]:
    """
    Detect the account type from document text.

    Args:
        text: Document text to analyze

    Returns:
        Tuple of (AccountType, confidence)
        Returns (None, 0.0) if no account type detected
    """
    text_lower = text.lower()

    for account_type, keywords in ACCOUNT_TYPE_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text_lower:
                # Found a match - return with high confidence
                return account_type, 0.9

    # No explicit match - try to infer from context
    # Default to SAVINGS as most common type if it looks like a bank statement
    return None, 0.0


def _detect_pages_with_tables(pdf_doc: PDFDocument) -> list[int]:
    """
    Detect which pages likely contain tables.

    Uses heuristics based on line density:
    - Pages with many horizontal and vertical lines likely have tables
    - Minimum 4 horizontal and 4 vertical lines required

    Args:
        pdf_doc: PDFDocument instance

    Returns:
        List of zero-indexed page numbers with tables
    """
    pages_with_tables = []

    for page_num in range(pdf_doc.page_count):
        lines = pdf_doc.get_page_lines(page_num)

        # Count horizontal and vertical lines
        h_lines = sum(1 for line in lines if line.orientation == "horizontal" and line.length > 50)
        v_lines = sum(1 for line in lines if line.orientation == "vertical" and line.length > 20)

        # Heuristic: need at least 4 horizontal and 4 vertical lines for a table
        if h_lines >= 4 and v_lines >= 4:
            pages_with_tables.append(page_num)
        elif h_lines >= 2:
            # Semi-bordered table (just horizontal lines)
            # Check if there's structured text (column alignment)
            text = pdf_doc.get_page_text(page_num)
            if text and len(text) > 100:
                pages_with_tables.append(page_num)

    return pages_with_tables


class FileClassifier:
    """
    Classifies PDF files as bank statements and extracts metadata.

    This is the main entry point for file classification. It performs:
    1. File validation (exists, valid PDF, size limits)
    2. PDF parsing and password handling
    3. Content analysis (scanned detection, language)
    4. Bank statement classification
    5. Account type detection
    6. Table presence detection

    Attributes:
        options: ProcessingOptions for customizing behavior

    Example:
        ```python
        classifier = FileClassifier()
        result = classifier.classify("statement.pdf")

        if result.status == "success":
            print(f"Bank statement: {result.document_type}")
            print(f"Account: {result.account_type}")
            print(f"Confidence: {result.classification_confidence}")
        ```
    """

    def __init__(self, options: Optional[ProcessingOptions] = None):
        """
        Initialize the classifier.

        Args:
            options: Optional ProcessingOptions to customize behavior
        """
        self.options = options or ProcessingOptions()
        self._validator = FileValidator(
            max_size_bytes=int(self.options.max_file_size_mb * 1024 * 1024)
        )

    def classify(
        self,
        file_path: str,
        password: Optional[str] = None,
    ) -> FileClassification:
        """
        Classify a PDF file and extract metadata.

        This is the main entry point for classification. It performs all
        validation and classification steps, returning a structured result.

        Args:
            file_path: Path to the PDF file
            password: Optional password for encrypted PDFs

        Returns:
            FileClassification with classification results or error info

        Example:
            ```python
            result = classifier.classify("statement.pdf")

            if result.status == "success":
                # Access classification results
                print(result.document_type)
                print(result.account_type)

                # Use PDF document for further processing
                pdf_doc = result.pdf_document
                text = pdf_doc.get_page_text(0)
            else:
                # Handle error
                print(f"Error: {result.error_code}")
            ```
        """
        warnings: list[ProcessingWarning] = []

        # Step 1: Validate file
        try:
            file_size, file_type = self._validator.validate_all(file_path)
        except ProcessingError as e:
            return FileClassification.from_error(e, file_path)

        # Step 2: Open PDF (with password if provided)
        try:
            pdf_doc = PDFDocument.open(file_path, password=password)
        except ProcessingError as e:
            return FileClassification.from_error(e, file_path)

        # Step 3: Check for large files
        if pdf_doc.page_count > 100:
            warnings.append(
                ProcessingWarning.create(
                    WarningCode.LARGE_FILE,
                    details={"page_count": pdf_doc.page_count},
                )
            )

        # Step 4: Check if scanned
        if pdf_doc.is_scanned:
            pdf_doc.close()
            return FileClassification.from_error(
                ProcessingError(
                    ErrorCode.FILE_SCANNED_NOT_SUPPORTED,
                    details={"path": file_path},
                ),
                file_path,
            )

        # Step 5: Extract text for analysis
        # Use first few pages for efficiency
        sample_pages = min(3, pdf_doc.page_count)
        sample_text = "\n".join(
            pdf_doc.get_page_text(i) for i in range(sample_pages)
        )

        # Step 6: Detect language
        language, lang_confidence = _detect_language(sample_text)

        if language != "en" and lang_confidence > self.options.language_confidence_threshold:
            pdf_doc.close()
            return FileClassification.from_error(
                ProcessingError(
                    ErrorCode.LANGUAGE_NOT_SUPPORTED,
                    details={"detected_language": language},
                ),
                file_path,
            )

        # Step 7: Classify as bank statement
        bank_confidence = _calculate_bank_statement_confidence(sample_text)

        if bank_confidence < self.options.bank_statement_confidence_threshold:
            if self.options.strict_mode:
                pdf_doc.close()
                return FileClassification.from_error(
                    ProcessingError(ErrorCode.NOT_BANK_STATEMENT),
                    file_path,
                )
            else:
                warnings.append(
                    ProcessingWarning.create(
                        WarningCode.LOW_CONFIDENCE,
                        details={"confidence": bank_confidence},
                    )
                )

        document_type = (
            DocumentType.BANK_STATEMENT
            if bank_confidence >= self.options.bank_statement_confidence_threshold
            else DocumentType.UNKNOWN
        )

        # Step 8: Detect account type
        account_type, account_confidence = _detect_account_type(sample_text)

        # Validate account type is supported
        if account_type and account_type not in AccountType.supported_types():
            pdf_doc.close()
            return FileClassification.from_error(
                ProcessingError(
                    ErrorCode.ACCOUNT_TYPE_NOT_SUPPORTED,
                    details={"account_type": account_type.value},
                ),
                file_path,
            )

        # Step 9: Detect pages with tables
        pages_with_tables = _detect_pages_with_tables(pdf_doc)

        # Build result
        result = FileClassification(
            status="success",
            file_path=file_path,
            file_type=file_type,
            file_size_bytes=file_size,
            is_password_protected=pdf_doc.is_encrypted,
            is_scanned=False,
            page_count=pdf_doc.page_count,
            pages_with_tables=pages_with_tables,
            language=language,
            language_confidence=lang_confidence,
            document_type=document_type,
            account_type=account_type,
            classification_confidence=bank_confidence,
            warnings=warnings,
            pdf_document=pdf_doc._pdf,  # Pass underlying pdfplumber object
        )

        return result


def classify_file(
    file_path: str,
    password: Optional[str] = None,
    options: Optional[ProcessingOptions] = None,
) -> FileClassification:
    """
    Convenience function to classify a file.

    Args:
        file_path: Path to the PDF file
        password: Optional password for encrypted PDFs
        options: Optional processing options

    Returns:
        FileClassification with results

    Example:
        ```python
        from tables.reader.file_classifier import classify_file

        result = classify_file("statement.pdf")
        print(result.to_dict())
        ```
    """
    classifier = FileClassifier(options=options)
    return classifier.classify(file_path, password=password)
