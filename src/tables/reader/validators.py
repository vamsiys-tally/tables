"""File validation logic for PDF processing."""

import os
from pathlib import Path
from typing import Optional

from tables.models.errors import ErrorCode, ProcessingError


class FileValidator:
    """Validates files before PDF processing."""

    # PDF magic bytes (first 4 bytes of a PDF file)
    PDF_MAGIC_BYTES = b"%PDF"

    # Maximum file size in bytes (default 100MB)
    DEFAULT_MAX_SIZE_BYTES = 100 * 1024 * 1024

    def __init__(self, max_size_bytes: Optional[int] = None):
        """
        Initialize the validator.

        Args:
            max_size_bytes: Maximum allowed file size in bytes
        """
        self.max_size_bytes = max_size_bytes or self.DEFAULT_MAX_SIZE_BYTES

    def validate_file_exists(self, file_path: str) -> None:
        """
        Validate that the file exists.

        Args:
            file_path: Path to the file

        Raises:
            ProcessingError: If file does not exist
        """
        if not os.path.exists(file_path):
            raise ProcessingError(
                ErrorCode.FILE_NOT_FOUND,
                details={"path": file_path},
            )

    def validate_file_not_empty(self, file_path: str) -> int:
        """
        Validate that the file is not empty.

        Args:
            file_path: Path to the file

        Returns:
            File size in bytes

        Raises:
            ProcessingError: If file is empty
        """
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            raise ProcessingError(
                ErrorCode.FILE_EMPTY,
                details={"path": file_path},
            )
        return file_size

    def validate_file_size(self, file_path: str, file_size: Optional[int] = None) -> int:
        """
        Validate that the file is not too large.

        Args:
            file_path: Path to the file
            file_size: Optional pre-computed file size

        Returns:
            File size in bytes

        Raises:
            ProcessingError: If file exceeds maximum size
        """
        if file_size is None:
            file_size = os.path.getsize(file_path)

        if file_size > self.max_size_bytes:
            max_size_mb = self.max_size_bytes / (1024 * 1024)
            actual_size_mb = file_size / (1024 * 1024)
            raise ProcessingError(
                ErrorCode.FILE_TOO_LARGE,
                details={
                    "path": file_path,
                    "max_size": f"{max_size_mb:.1f}",
                    "actual_size": f"{actual_size_mb:.1f}",
                },
            )
        return file_size

    def validate_file_type(self, file_path: str) -> str:
        """
        Validate that the file is a PDF by checking magic bytes and extension.

        Args:
            file_path: Path to the file

        Returns:
            Detected file type ("pdf")

        Raises:
            ProcessingError: If file is not a valid PDF
        """
        # Check extension
        path = Path(file_path)
        extension = path.suffix.lower()

        # Read magic bytes
        try:
            with open(file_path, "rb") as f:
                magic_bytes = f.read(4)
        except IOError as e:
            raise ProcessingError(
                ErrorCode.FILE_CORRUPTED,
                details={"path": file_path, "error": str(e)},
            )

        # Validate magic bytes
        if magic_bytes != self.PDF_MAGIC_BYTES:
            # Determine actual type for error message
            actual_type = self._detect_file_type(magic_bytes, extension)
            raise ProcessingError(
                ErrorCode.FILE_INVALID_TYPE,
                details={"path": file_path, "actual_type": actual_type},
            )

        return "pdf"

    def _detect_file_type(self, magic_bytes: bytes, extension: str) -> str:
        """
        Attempt to detect the actual file type for error reporting.

        Args:
            magic_bytes: First 4 bytes of the file
            extension: File extension

        Returns:
            Detected file type string
        """
        # Common file signatures
        signatures = {
            b"\x89PNG": "PNG image",
            b"\xff\xd8\xff": "JPEG image",
            b"GIF8": "GIF image",
            b"RIFF": "RIFF (possibly WebP)",
            b"PK\x03\x04": "ZIP archive (possibly DOCX/XLSX)",
            b"\x00\x00\x00": "Unknown binary",
        }

        for sig, file_type in signatures.items():
            if magic_bytes.startswith(sig):
                return file_type

        # Fall back to extension
        if extension:
            return f"file with extension '{extension}'"

        return "unknown file type"

    def validate_all(self, file_path: str) -> tuple[int, str]:
        """
        Run all file validations.

        Args:
            file_path: Path to the file

        Returns:
            Tuple of (file_size_bytes, file_type)

        Raises:
            ProcessingError: If any validation fails
        """
        self.validate_file_exists(file_path)
        file_size = self.validate_file_not_empty(file_path)
        self.validate_file_size(file_path, file_size)
        file_type = self.validate_file_type(file_path)

        return file_size, file_type


def validate_file(
    file_path: str, max_size_bytes: Optional[int] = None
) -> tuple[int, str]:
    """
    Convenience function to validate a file.

    Args:
        file_path: Path to the file
        max_size_bytes: Optional maximum file size

    Returns:
        Tuple of (file_size_bytes, file_type)

    Raises:
        ProcessingError: If validation fails
    """
    validator = FileValidator(max_size_bytes=max_size_bytes)
    return validator.validate_all(file_path)
