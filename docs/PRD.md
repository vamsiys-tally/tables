# Bank Statement PDF Processing System - Detailed PRD

## Executive Summary

A high-accuracy, low-latency system to extract structured transaction data from bank statement PDFs. The system must handle diverse Indian bank formats with near-100% accuracy using primarily deterministic methods, with targeted ML augmentation where it adds clear value.

---

## Module 1: File Reader & Classifier

### 1.1 Operations

#### A. File Acquisition
- Fetch file from local path or URL
- Validate file existence and accessibility

#### B. File Parameter Classification

| Parameter | Detection Method | Notes |
|-----------|-----------------|-------|
| File type | Extension + magic bytes | PDF, images (JPG, PNG, TIFF) |
| Password protection | pdfplumber/PyPDF2 exception handling | Detect encrypted PDFs |
| Corruption status | PDF parser validation | Malformed PDF detection |
| Scanned vs Generated | Text layer presence + char density analysis | Generated PDFs have extractable text; scanned have images |

> **Decision:** Edit detection is skipped - technically unreliable and not worth the complexity.

#### C. File Content Classification

| Parameter | Detection Method | Confidence |
|-----------|-----------------|------------|
| Page count | PDF metadata | Deterministic |
| Pages with tables | Line/rect detection + text clustering | Deterministic |
| Language | Character encoding + langdetect on extracted text | High |
| Document type | Keyword matching + layout heuristics | **SLM opportunity** |
| Account type | Header/keyword extraction | **SLM opportunity** |

**Document Type Classification:**
- Bank statement vs Non-bank statement
- If bank statement (Supported in v1):
  - Savings account ✓
  - Current account ✓
  - OD (Overdraft) account ✓
- Future consideration:
  - Credit card statement (different structure - deferred)
  - Loan statement
  - Fixed Deposit statement
  - Others/Unknown

### 1.2 Edge Cases

| Edge Case | Detection | Handling |
|-----------|-----------|----------|
| Zero-byte file | File size check | Error: `FILE_EMPTY` |
| Corrupted PDF | Parser exception | Error: `FILE_CORRUPTED` |
| Password-protected (no password) | Encryption flag | Error: `FILE_PASSWORD_REQUIRED` - prompt for password |
| Password-protected (wrong password) | Decryption failure | Error: `FILE_PASSWORD_INCORRECT` |
| Image-only PDF (scanned) | No text layer + high image content | Error: `FILE_SCANNED_NOT_SUPPORTED` |
| Mixed scanned/generated | Per-page text density analysis | Process generated pages, flag scanned |
| Non-English content | Language detection | Error: `LANGUAGE_NOT_SUPPORTED` |
| Multi-language document | Per-page/section language check | Process English sections |
| Very large files (>100 pages) | Page count check | Warning, process with pagination |
| Unusual page sizes | Page dimension extraction | Normalize coordinates |
| Rotated pages | Rotation metadata | Apply rotation correction |
| PDF/A, PDF/X variants | PDF version check | Handle appropriately |

### 1.3 Flow

```
Input: file_path, password (optional)
    │
    ├─► File exists? ──No──► Error: FILE_NOT_FOUND
    │
    ├─► Valid PDF? ──No──► Error: FILE_CORRUPTED or FILE_INVALID_TYPE
    │
    ├─► Password protected?
    │   ├─ Yes + password provided ──► Attempt unlock
    │   │   └─ Unlock failed? ──► Error: FILE_PASSWORD_INCORRECT
    │   └─ Yes + no password ──► Error: FILE_PASSWORD_REQUIRED (prompt user)
    │
    ├─► Has text layer? ──No──► Error: FILE_SCANNED_NOT_SUPPORTED
    │
    ├─► Language = English? ──No──► Error: LANGUAGE_NOT_SUPPORTED
    │
    ├─► Is bank statement? ──No──► Error: NOT_BANK_STATEMENT
    │
    ├─► Account type supported? ──No──► Error: ACCOUNT_TYPE_NOT_SUPPORTED
    │   (Savings/Current/OD)
    │
    └─► Success: Return PDFDocument object with metadata
```

> **Note:** In production, the password will be provided along with the file upload. The API accepts an optional `password` parameter.

### 1.4 I/O Specification

**Input:**
```python
@dataclass
class FileInput:
    file_path: str  # Local path or URL
    password: Optional[str] = None  # PDF password if protected
    options: Optional[ProcessingOptions] = None  # Override defaults
```

**Output (JSON/Dict Structure):**
```python
@dataclass
class FileClassification:
    status: Literal["success", "error"]
    error_code: Optional[str] = None
    error_message: Optional[str] = None

    # File metadata
    file_type: str
    file_size_bytes: int
    is_password_protected: bool
    is_scanned: bool

    # Content metadata
    page_count: int
    pages_with_tables: List[int]
    language: str
    document_type: str
    account_type: Optional[str]

    # Confidence & warnings
    classification_confidence: float  # 0.0 - 1.0
    warnings: List[str] = field(default_factory=list)

    # For downstream processing (not serialized)
    pdf_document: Optional[PDFDocument] = None

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict (excludes pdf_document)"""
        ...
```

> **Decision:** Primary output is JSON/Dict structure for serialization and downstream API use.

---

## Module 2: Table Detection

### 2.1 Operations

#### A. Table Presence Detection
- Identify pages containing tabular data
- Classify table structure type (bordered, semi-bordered, unbordered)

#### B. Table Type Classification

| Table Type | Characteristics | Detection Method |
|------------|-----------------|------------------|
| **Bordered** | Full grid lines | Line intersection detection |
| **Semi-bordered** | Partial lines (headers, rows) | Horizontal line + column alignment |
| **Unbordered** | No lines, whitespace-separated | Text clustering + column alignment |
| **Shaded** | Background color differentiation | Rectangle fill color analysis |

#### C. Table Content Classification

| Content Type | Indicators | Detection |
|--------------|------------|-----------|
| **Transaction table** | Date, Amount, Balance columns | Header keyword matching |
| **Summary table** | Totals, period summary | Keyword + position (often top/bottom) |
| **Account info table** | Account number, name, branch | Keyword + typically page 1 |
| **Interest/charges table** | Interest, charges, fees | Keyword matching |

#### D. Table Structure Analysis

| Property | Description |
|----------|-------------|
| Table bounds | (x0, y0, x1, y1) coordinates |
| Column definitions | x-coordinates of column separators |
| Row definitions | y-coordinates of row separators |
| Header row(s) | Identified header row indices |
| Multi-page span | Whether table continues across pages |
| Header repetition | Whether headers repeat on each page |

### 2.2 Header Detection Strategy

**Primary Method (Deterministic):**
```python
HEADER_KEYWORDS = {
    "date": ["date", "txn date", "transaction date", "value date", "posting date"],
    "description": ["particulars", "description", "narration", "remarks", "transaction details"],
    "reference": ["ref", "reference", "chq no", "cheque no", "utr", "transaction id"],
    "debit": ["debit", "withdrawal", "dr", "withdrawals", "debit amount"],
    "credit": ["credit", "deposit", "cr", "deposits", "credit amount"],
    "balance": ["balance", "closing balance", "running balance", "available balance"],
    # ... extensive keyword list
}
```

**Secondary Method (SLM-assisted) - For ambiguous cases:**
- Use lightweight embedding model for semantic similarity
- Only invoked when deterministic matching confidence < threshold
- Model: `all-MiniLM-L6-v2` or similar lightweight model

### 2.3 Edge Cases

| Edge Case | Detection | Handling |
|-----------|-----------|----------|
| No tables on page | Low line density + no text clusters | Skip page, log info |
| Multiple tables per page | Spatial separation analysis | Identify and process each separately |
| Nested tables | Hierarchy detection via containment | Process outer table, flag nested |
| Tables split mid-row across pages | Row continuity analysis | Mark for cross-page merging |
| Rotated tables | Text angle analysis | Apply rotation before processing |
| Tables with merged cells | Cell span detection | Track colspan/rowspan |
| Very wide tables (>page width) | Column overflow detection | Error or multi-column handling |
| Empty rows within table | Row content analysis | Preserve structure, mark empty |
| Header spans multiple rows | Multi-row header detection | Combine header rows |
| No clear headers | Header detection failure | Use column position heuristics |
| Watermarks/logos in table area | Image vs text differentiation | Filter non-text elements |
| Footer rows (totals) | Position + keyword detection | Mark as footer, not transaction |

### 2.4 Flow

```
Input: PDFDocument from Module 1
    │
    For each page:
    │
    ├─► Extract lines (horizontal, vertical)
    │
    ├─► Extract rectangles (bordered regions)
    │
    ├─► Extract text with coordinates
    │
    ├─► Detect table structure type:
    │   ├─ Bordered: Line intersection grid
    │   ├─ Semi-bordered: Partial lines + text alignment
    │   └─ Unbordered: Text clustering + whitespace analysis
    │
    ├─► Identify table boundaries
    │
    ├─► Detect headers (keyword matching → SLM fallback)
    │
    ├─► Classify table content (transaction vs other)
    │
    └─► Analyze cross-page continuity
    │
Output: List[TableDefinition] with metadata
```

### 2.5 I/O Specification

**Output:**
```python
class TableDefinition:
    table_id: str
    page_numbers: List[int]  # Pages this table spans

    # Structure
    structure_type: Literal["bordered", "semi_bordered", "unbordered", "shaded"]
    bounds_per_page: Dict[int, BoundingBox]  # Page -> bounds

    # Columns
    columns: List[ColumnDefinition]  # x0, x1, header_text, data_type

    # Headers
    header_rows: List[int]  # Row indices that are headers
    header_repeats: bool  # True if headers repeat on each page

    # Classification
    content_type: Literal["transaction", "summary", "account_info", "other"]

    # Cross-page
    is_multi_page: bool
    continuation_pages: List[int]

class ColumnDefinition:
    column_id: int
    x0: float
    x1: float
    header_text: str
    semantic_type: Optional[str]  # "date", "amount", "description", etc.
    data_type: Literal["date", "numeric", "text", "mixed"]
```

---

## Module 3: Table Structure Recognition (TSR)

### 3.1 Operations

#### A. Cell Extraction
- Define cell boundaries from column/row intersections
- Extract text content per cell
- Handle multi-line cell content

#### B. Row Construction
- Group cells into logical rows
- Handle wrapped text (single transaction across multiple visual rows)
- Detect row continuity breaks

#### C. Cross-Page Table Merging
- Identify continuation tables
- Merge tables maintaining row integrity
- Handle mid-row page breaks

#### D. Data Structuring
- Map cells to semantic columns
- Validate data types per column
- Construct final table output

### 3.2 Row Merging Logic

**Challenge:** A single transaction may span multiple visual lines due to text wrapping.

**Detection Signals:**
1. Date column empty in subsequent row
2. Balance column empty (only populated for complete transactions)
3. Continuation keywords ("...", carried forward)
4. Indentation pattern in description

**Merging Rules:**
```python
def should_merge_with_previous(current_row, previous_row):
    # Rule 1: No date in current row
    if is_empty(current_row.date_cell):
        return True

    # Rule 2: No balance in current row (for statements with running balance)
    if has_balance_column and is_empty(current_row.balance_cell):
        return True

    # Rule 3: Indentation suggests continuation
    if current_row.description_indent > threshold:
        return True

    return False
```

### 3.3 Edge Cases

| Edge Case | Detection | Handling |
|-----------|-----------|----------|
| Text wrapping within cell | Multi-line content in single cell bounds | Preserve with newlines |
| Transaction spans 3+ visual rows | Extended merge detection | Recursive merge check |
| Page break mid-transaction | Incomplete row at page end | Merge with next page start |
| Misaligned columns across pages | Column position drift | Re-detect columns per page, map semantically |
| Missing cells in row | Expected column count mismatch | Insert null placeholder |
| Extra columns in some rows | Column count variation | Handle as merged cell or error |
| Numeric parsing issues | Non-standard formats (lakhs, parentheses for negative) | Robust number parser |
| Date format variations | Multiple date formats in same doc | Multi-format date parser |
| Currency symbols mixed in amounts | Symbol detection | Strip and normalize |
| Thousands separators (Indian: 1,00,000) | Indian number format | Locale-aware parsing |

### 3.4 Flow

```
Input: List[TableDefinition] from Module 2
    │
    For each transaction table:
    │
    ├─► Extract cells with text content
    │   ├─ Use column definitions for x-boundaries
    │   └─ Use row separators or text clustering for y-boundaries
    │
    ├─► Construct initial rows
    │
    ├─► Apply row merging logic
    │   └─ Combine wrapped transaction rows
    │
    ├─► Cross-page merging
    │   ├─ Detect continuation (no header on next page OR repeated header)
    │   └─ Merge maintaining row indices
    │
    ├─► Semantic column mapping
    │   └─ Map detected columns to standard schema
    │
    ├─► Data type validation & parsing
    │   ├─ Parse dates
    │   ├─ Parse amounts (handle Indian format)
    │   └─ Clean description text
    │
    └─► Construct output DataFrame/DTO
    │
Output: List[TransactionTable]
```

### 3.5 I/O Specification

**Output (JSON/Dict Structure):**
```python
@dataclass
class TransactionTable:
    table_id: str
    source_pages: List[int]

    # Schema
    columns: List[ColumnSchema]

    # Data
    rows: List[TransactionRow]

    # Metadata
    row_count: int
    date_range: Optional[Tuple[date, date]]
    total_debit: Optional[Decimal]
    total_credit: Optional[Decimal]

    # Quality metrics - CRITICAL for low-confidence handling
    confidence_score: float  # 0.0 - 1.0
    parse_warnings: List[ParseWarning] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict"""
        ...

@dataclass
class TransactionRow:
    row_id: int
    source_page: int
    source_row_indices: List[int]  # Original visual row indices (for merged rows)

    # Standard fields (nullable)
    transaction_date: Optional[date]
    value_date: Optional[date]
    description: str
    reference: Optional[str]
    debit_amount: Optional[Decimal]
    credit_amount: Optional[Decimal]
    balance: Optional[Decimal]

    # Per-row confidence and warnings
    confidence: float  # 0.0 - 1.0
    warnings: List[str] = field(default_factory=list)

    # Raw values for debugging/audit
    raw_cells: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict"""
        ...

class ColumnSchema:
    name: str
    semantic_type: str  # "date", "description", "debit", "credit", "balance", "reference"
    data_type: str  # "date", "decimal", "string"
    nullable: bool
    source_column_index: int
```

---

## Tech Stack Recommendation

### Core Libraries

| Component | Library | Rationale |
|-----------|---------|-----------|
| PDF Parsing | `pdfplumber` | Best balance of accuracy and speed for text/line extraction |
| PDF Fallback | `PyMuPDF (fitz)` | Faster for large files, good fallback |
| Data Manipulation | `pandas` | Industry standard, efficient |
| Numeric Precision | `decimal.Decimal` | Financial data requires precision |
| Date Parsing | `dateutil.parser` | Handles diverse formats |

### ML Components (Targeted Use)

| Component | Model | Use Case | Latency Impact |
|-----------|-------|----------|----------------|
| Header detection fallback | `all-MiniLM-L6-v2` | When keyword matching fails | ~50ms per row |
| Document classification | Custom classifier or `DistilBERT` | Bank statement vs other | ~100ms per doc |
| Account type detection | Rule-based + keyword | Savings/Current/CC/Loan | Minimal |

**ML Usage Philosophy:**
- ML is a **fallback**, not primary
- Pre-compute embeddings for known headers (one-time cost)
- Lazy-load models only when needed
- Consider ONNX optimization for production

### Coordinate Processing

| Component | Library | Rationale |
|-----------|---------|-----------|
| Clustering | `scikit-learn` DBSCAN | Coordinate normalization |
| Geometry | `shapely` | Robust rectangle/intersection ops |

### Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Single page processing | < 500ms | Without ML |
| Single page with ML fallback | < 1s | With header detection ML |
| 10-page document | < 5s | End-to-end |
| Memory per page | < 50MB | Peak usage |

---

## Testing Framework

### Test Categories

#### 1. Unit Tests
```
tests/
├── unit/
│   ├── test_file_classifier.py
│   ├── test_table_detector.py
│   ├── test_structure_recognizer.py
│   ├── test_row_merger.py
│   ├── test_date_parser.py
│   └── test_amount_parser.py
```

#### 2. Integration Tests
```
tests/
├── integration/
│   ├── test_end_to_end.py
│   ├── test_multi_page.py
│   └── test_error_handling.py
```

#### 3. Bank-Specific Regression Tests
```
tests/
├── regression/
│   ├── test_hdfc_formats.py
│   ├── test_icici_formats.py
│   ├── test_sbi_formats.py
│   └── ...
```

#### 4. Golden File Tests
- Store expected outputs for each test PDF
- Compare extracted data against golden files
- Flag any regression

### Test Data Organization

```
tests/
├── data/
│   ├── valid/
│   │   ├── hdfc/
│   │   │   ├── savings_format1.pdf
│   │   │   ├── savings_format1.expected.json  # Golden file
│   │   │   └── ...
│   │   ├── icici/
│   │   └── ...
│   ├── invalid/
│   │   ├── corrupted.pdf
│   │   ├── password_protected.pdf
│   │   ├── scanned.pdf
│   │   └── non_english.pdf
│   └── edge_cases/
│       ├── multi_table_page.pdf
│       ├── wrapped_transactions.pdf
│       ├── mid_row_page_break.pdf
│       └── ...
```

### Test Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Row extraction accuracy | 99.5% | Correct rows / Total rows |
| Cell extraction accuracy | 99% | Correct cells / Total cells |
| Amount parsing accuracy | 99.9% | Financial data critical |
| Date parsing accuracy | 99.5% | Date format handling |
| Header detection accuracy | 98% | Correct header identification |
| Error message clarity | 100% | All errors have actionable messages |

### Continuous Testing

```python
# pytest configuration
# pytest.ini or pyproject.toml

[tool.pytest.ini_options]
markers = [
    "unit: Unit tests",
    "integration: Integration tests",
    "regression: Bank-specific regression tests",
    "slow: Tests that take > 1s",
    "ml: Tests requiring ML models",
]
```

---

## Error Handling & Messages

### Error Code Structure

```python
class ProcessingError(Exception):
    def __init__(self, code: str, message: str, details: dict = None):
        self.code = code
        self.message = message
        self.details = details or {}

# Error codes by module
FILE_ERRORS = {
    "FILE_NOT_FOUND": "The specified file does not exist: {path}",
    "FILE_EMPTY": "The file is empty (0 bytes): {path}",
    "FILE_CORRUPTED": "The PDF file is corrupted and cannot be read: {path}",
    "FILE_PASSWORD_REQUIRED": "The PDF is password protected. Please provide the password.",
    "FILE_PASSWORD_INCORRECT": "The provided password is incorrect for this PDF.",
    "FILE_INVALID_TYPE": "Expected PDF file, got: {actual_type}",
    "FILE_TOO_LARGE": "File exceeds maximum size of {max_size}MB: {actual_size}MB",
}

CONTENT_ERRORS = {
    "FILE_SCANNED_NOT_SUPPORTED": "This appears to be a scanned document. Only digitally generated PDFs are supported.",
    "LANGUAGE_NOT_SUPPORTED": "Document language '{detected_language}' is not supported. Only English documents are supported.",
    "NOT_BANK_STATEMENT": "This document does not appear to be a bank statement.",
    "ACCOUNT_TYPE_NOT_SUPPORTED": "Account type '{account_type}' is not currently supported. Supported types: Savings, Current, OD.",
}

TABLE_ERRORS = {
    "NO_TABLES_FOUND": "No transaction tables found in the document.",
    "TABLE_STRUCTURE_UNCLEAR": "Could not determine table structure on page {page}. The table format may not be supported.",
    "HEADER_NOT_DETECTED": "Could not identify table headers on page {page}.",
}

PARSING_ERRORS = {
    "DATE_PARSE_FAILED": "Could not parse date '{value}' in row {row}, column '{column}'.",
    "AMOUNT_PARSE_FAILED": "Could not parse amount '{value}' in row {row}, column '{column}'.",
    "ROW_STRUCTURE_INVALID": "Row {row} on page {page} has unexpected structure.",
}
```

### Error Response Format

```python
class ProcessingResult:
    success: bool
    data: Optional[List[TransactionTable]]
    errors: List[ProcessingError]
    warnings: List[ProcessingWarning]

    # Processing metadata
    processing_time_ms: int
    pages_processed: int
    pages_skipped: List[Tuple[int, str]]  # (page_num, reason)
```

---

## Key Decisions Made

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Edit Detection | **Skip entirely** | Technically unreliable, not worth complexity |
| Output Format | **JSON/Dict structure** | Serializable for APIs, includes full metadata |
| Low Confidence Handling | **Warn and proceed** | Return data with confidence flags, let consumer decide |
| Credit Card Statements | **Unsupported in v1** | Focus on Savings/Current/OD first, fundamentally different structure |

### Confidence Handling Strategy

```python
# Processing always continues, but results include confidence metadata
class ProcessingResult:
    success: bool  # True even for low-confidence extractions
    data: List[TransactionTable]

    # Confidence levels with semantic meaning
    overall_confidence: float  # 0.0 - 1.0

    # Warnings for consumer to evaluate
    warnings: List[ProcessingWarning]

    # Per-row confidence available in TransactionRow.confidence
```

**Confidence Thresholds (informational, not blocking):**
- `>= 0.9`: High confidence - likely accurate
- `0.7 - 0.9`: Medium confidence - review recommended
- `< 0.7`: Low confidence - manual verification needed

---

## Implementation Phases

### Phase 1: Core Infrastructure
- File reader with classification
- Basic PDF parsing
- Error handling framework
- Test infrastructure

### Phase 2: Table Detection
- Bordered table detection
- Header detection (keyword-based)
- Single-page processing

### Phase 3: Structure Recognition
- Cell extraction
- Row construction
- Basic data parsing

### Phase 4: Advanced Features
- Multi-page table handling
- Row merging logic
- Unbordered table support
- ML fallback integration

### Phase 5: Hardening
- Bank-specific regression tests
- Performance optimization
- Edge case handling
- Production readiness

---

## Files to Create

```
src/
├── tables/
│   ├── __init__.py
│   ├── reader/
│   │   ├── __init__.py
│   │   ├── file_classifier.py
│   │   ├── pdf_document.py
│   │   └── validators.py
│   ├── detector/
│   │   ├── __init__.py
│   │   ├── table_detector.py
│   │   ├── header_detector.py
│   │   ├── structure_classifier.py
│   │   └── keywords.py
│   ├── recognizer/
│   │   ├── __init__.py
│   │   ├── cell_extractor.py
│   │   ├── row_builder.py
│   │   ├── table_merger.py
│   │   └── data_parser.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── document.py
│   │   ├── table.py
│   │   ├── transaction.py
│   │   └── errors.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── geometry.py
│   │   ├── clustering.py
│   │   └── parsing.py
│   └── config.py
├── tests/
│   └── [as described above]
├── requirements.txt
└── setup.py
```
