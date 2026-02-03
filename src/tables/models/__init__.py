"""
Data Models and Type Definitions.

This package provides all data models used throughout the tables module:

Document Models:
    - FileClassification: Result of file validation and classification
    - ProcessingResult: Container for processing results
    - ProcessingOptions: Configuration options for processing

Table Models:
    - TableDefinition: Complete table structure definition
    - ColumnDefinition: Individual column definition
    - TableDetectionResult: Container for detection results

Error Models:
    - ProcessingError: Exception class for processing errors
    - ProcessingWarning: Warning information

Enumerations:
    - StructureType: Visual table structure types
    - ContentType: Semantic table content types
    - DataType: Column data types
"""

from tables.models.errors import ProcessingError, ProcessingWarning
from tables.models.document import (
    FileClassification,
    ProcessingResult,
    ProcessingOptions,
)
from tables.models.table import (
    TableDefinition,
    ColumnDefinition,
    TableDetectionResult,
    StructureType,
    ContentType,
    DataType,
)

__all__ = [
    # Error models
    "ProcessingError",
    "ProcessingWarning",
    # Document models
    "FileClassification",
    "ProcessingResult",
    "ProcessingOptions",
    # Table models
    "TableDefinition",
    "ColumnDefinition",
    "TableDetectionResult",
    # Enumerations
    "StructureType",
    "ContentType",
    "DataType",
]
