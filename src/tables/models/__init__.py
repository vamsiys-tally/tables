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

Transaction Models:
    - TransactionTable: Extracted transaction table with parsed data
    - TransactionRow: Single parsed transaction row
    - ColumnSchema: Column type and parsing schema
    - ExtractionResult: Complete extraction result container

Error Models:
    - ProcessingError: Exception class for processing errors
    - ProcessingWarning: Warning information
    - ParseWarning: Warning about parsing issues

Enumerations:
    - StructureType: Visual table structure types
    - ContentType: Semantic table content types
    - DataType: Column data types
    - SemanticType: Semantic column types for transactions
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
from tables.models.transaction import (
    TransactionTable,
    TransactionRow,
    ColumnSchema,
    ExtractionResult,
    ParseWarning,
    SemanticType,
    ParseStatus,
)

__all__ = [
    # Error models
    "ProcessingError",
    "ProcessingWarning",
    "ParseWarning",
    # Document models
    "FileClassification",
    "ProcessingResult",
    "ProcessingOptions",
    # Table models
    "TableDefinition",
    "ColumnDefinition",
    "TableDetectionResult",
    # Transaction models
    "TransactionTable",
    "TransactionRow",
    "ColumnSchema",
    "ExtractionResult",
    # Enumerations
    "StructureType",
    "ContentType",
    "DataType",
    "SemanticType",
    "ParseStatus",
]
