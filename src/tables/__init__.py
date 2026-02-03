"""
Bank Statement PDF Processing System

A high-accuracy, low-latency system to extract structured transaction data
from bank statement PDFs.
"""

__version__ = "0.1.0"

from tables.reader.file_classifier import FileClassifier
from tables.models.document import FileClassification, ProcessingResult
from tables.models.errors import ProcessingError

__all__ = [
    "FileClassifier",
    "FileClassification",
    "ProcessingResult",
    "ProcessingError",
]
