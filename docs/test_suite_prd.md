# Test Suite PRD - Bank Statement Extraction Evaluation Framework

## Overview

A comprehensive evaluation framework to measure extraction accuracy at multiple granularity levels, track performance across runs, and generate detailed error reports for debugging.

---

## 1. Ground Truth Format

### 1.1 CSV Structure

Ground truth files are CSV extracts from PDFs with headers matching `TransactionRow` fields:

```csv
date,value_date,description,reference,debit,credit,balance
15/01/2024,,ATM WITHDRAWAL XYZ BRANCH,CHQ123456,5000.00,,45000.00
16/01/2024,16/01/2024,SALARY CREDIT ACME CORP,NEFT12345,,50000.00,95000.00
```

| Column | Required | Notes |
|--------|----------|-------|
| date | Yes | Transaction date (DD/MM/YYYY or other formats) |
| value_date | No | Value date if different from transaction date |
| description | Yes | Full transaction description |
| reference | No | Cheque number, UTR, NEFT ref, etc. |
| debit | No* | Debit amount (positive number) |
| credit | No* | Credit amount (positive number) |
| balance | No | Running balance after transaction |

*At least one of debit/credit should be present per row.

### 1.2 File Organization

```
tests/
├── data/
│   ├── ground_truth/
│   │   ├── hdfc/
│   │   │   ├── hdfc_savings_001.pdf
│   │   │   ├── hdfc_savings_001.csv      # Ground truth
│   │   │   └── hdfc_savings_001.meta.json # File metadata
│   │   ├── icici/
│   │   ├── sbi/
│   │   └── ...
│   ├── edge_cases/
│   │   ├── scanned/
│   │   ├── password_protected/
│   │   ├── multi_page/
│   │   └── unbordered_tables/
```

### 1.3 Metadata File (Optional)

```json
{
  "bank": "HDFC",
  "account_type": "savings",
  "page_count": 3,
  "transaction_count": 45,
  "date_range": ["2024-01-01", "2024-01-31"],
  "known_issues": ["Page 2 has watermark"],
  "classification": "standard"
}
```

---

## 2. Evaluation Levels

### 2.1 Transaction Level (Primary Metric)

**Definition:** A single transaction row. Match is binary: 1 (correct) or 0 (incorrect).

**A transaction matches if ALL fields match:**

| Field | Comparison Method |
|-------|-------------------|
| date | Parse both to date, compare values |
| value_date | Parse both to date, compare values (if present) |
| description | Normalized text comparison (lowercase, collapse whitespace, trim) |
| reference | Normalized text comparison (if present) |
| debit | Parse to Decimal, compare values (5000 = 5,000.00 = ₹5000) |
| credit | Parse to Decimal, compare values |
| balance | Parse to Decimal, compare values (if present) |

**If ANY field doesn't match → Transaction Score = 0**
**If ALL fields match → Transaction Score = 1**

### 2.2 Transaction Matching Strategy

1. **Primary key:** Date + normalized description (fuzzy match)
2. **Tiebreaker:** Amount (debit or credit)
3. **Unmatched extracted rows:** Extra transactions (penalize)
4. **Unmatched ground truth rows:** Missed transactions (penalize)

### 2.3 File Level (Rollup)

**Definition:** Accuracy across all transactions in a file.

```
File Accuracy = Matched Transactions / Total Ground Truth Transactions

Where:
- Matched = Extracted row exists AND all fields match
- Missed = Ground truth row not found in extracted
- Extra = Extracted row not found in ground truth (also penalized)
```

**Example:**
```
Ground Truth: 10 transactions
Extracted: 12 transactions
  - 8 exact matches
  - 2 GT rows not extracted (missed)
  - 4 extracted rows not in GT (extra)

File Accuracy = 8 / 10 = 80%

Note: Extra rows don't increase denominator but are tracked separately.
```

### 2.4 Overall Accuracy (Across All Files)

```
Overall Accuracy = Total Matched Transactions / Total Ground Truth Transactions

Example:
  File 1: 8/10 matched
  File 2: 45/45 matched
  File 3: 18/20 matched

  Overall = (8 + 45 + 18) / (10 + 45 + 20) = 71/75 = 94.7%
```

---

## 3. Error Output Files

### 3.1 Per-File Error Report

For each file with errors, generate: `{filename}_errors.xlsx`

**Sheet 1: Summary**
| Metric | Value |
|--------|-------|
| File | hdfc_savings_001.pdf |
| Ground Truth Transactions | 45 |
| Extracted Transactions | 47 |
| Matched Transactions | 43 |
| Missed (in GT, not extracted) | 2 |
| Extra (extracted, not in GT) | 4 |
| **Accuracy** | **95.6%** (43/45) |
| Error Breakdown | DATE_MISMATCH: 1, AMOUNT_MISMATCH: 1, MISSING: 2 |

**Sheet 2: Ground Truth - Unmatched Rows**
| GT_Row# | Date | Description | Debit | Credit | Balance | Issue |
|---------|------|-------------|-------|--------|---------|-------|
| 12 | 15/01/2024 | ATM WITHDRAWAL | 5000 | | 45000 | NOT_EXTRACTED |
| 23 | 20/01/2024 | NEFT TRANSFER | | 10000 | 55000 | AMOUNT_MISMATCH |

**Sheet 3: Extracted - Unmatched/Mismatched Rows**
| Ext_Row# | Date | Description | Debit | Credit | Balance | Issue | Matched_GT_Row |
|----------|------|-------------|-------|--------|---------|-------|----------------|
| 14 | 15/01/2024 | ATM WITHDRAWAL XYZ | 5000 | | 45000 | DESC_MISMATCH | 12 |
| 25 | 20/01/2024 | NEFT TRANSFER | | 10500 | 55000 | AMOUNT_MISMATCH | 23 |
| 48 | | OPENING BALANCE | | | 40000 | EXTRA_ROW | - |

**Sheet 4: Side-by-Side Comparison (Mismatched Transactions)**
| GT_Row | Field | Ground Truth | Extracted | Match |
|--------|-------|--------------|-----------|-------|
| 23 | date | 20/01/2024 | 20/01/2024 | ✓ |
| 23 | description | NEFT TRANSFER TO ABC | NEFT TRANSFER TO ABC | ✓ |
| 23 | credit | 10000.00 | 10500.00 | ✗ |
| 23 | balance | 55000.00 | 55000.00 | ✓ |

---

## 4. Run Summary Tracking

### 4.1 Summary CSV Format

File: `tests/results/run_history.csv`

```csv
run_id,timestamp,git_commit,total_files,files_100pct,total_txns_gt,total_txns_matched,overall_accuracy,notes
run_001,2024-01-15T10:30:00,abc123f,50,45,1250,1190,0.952,Initial baseline
run_002,2024-01-16T14:20:00,def456a,50,47,1250,1215,0.972,Fixed date parsing
run_003,2024-01-17T09:15:00,ghi789b,55,52,1380,1350,0.978,Added 5 new test files
```

| Column | Description |
|--------|-------------|
| run_id | Unique identifier for the run |
| timestamp | When the run was executed |
| git_commit | Git commit hash (for reproducibility) |
| total_files | Number of files tested |
| files_100pct | Files with 100% transaction accuracy |
| total_txns_gt | Total transactions in ground truth |
| total_txns_matched | Transactions correctly extracted |
| overall_accuracy | total_txns_matched / total_txns_gt |
| notes | Optional description of changes |

### 4.2 Detailed Run Report

File: `tests/results/run_{run_id}/summary.json`

```json
{
  "run_id": "run_003",
  "timestamp": "2024-01-17T09:15:00",
  "git_commit": "ghi789b",
  "duration_seconds": 125,
  "metrics": {
    "files": {
      "total": 55,
      "perfect": 52,
      "with_errors": 3
    },
    "transactions": {
      "ground_truth": 1380,
      "extracted": 1375,
      "matched": 1350,
      "missed": 30,
      "extra": 25,
      "accuracy": 0.978
    }
  },
  "files_with_errors": [
    {"file": "axis_current_003.pdf", "accuracy": 0.85, "matched": 17, "gt": 20, "issue": "UNBORDERED_TABLE"},
    {"file": "sbi_savings_007.pdf", "accuracy": 0.92, "matched": 23, "gt": 25, "issue": "DATE_FORMAT"},
    {"file": "bob_od_002.pdf", "accuracy": 0.88, "matched": 22, "gt": 25, "issue": "MULTI_PAGE_MERGE"}
  ],
  "comparison_to_previous": {
    "previous_run": "run_002",
    "accuracy_delta": "+0.006",
    "files_improved": ["icici_savings_002.pdf"],
    "files_regressed": []
  }
}
```

---

## 5. Edge Case Classification

### 5.1 File Classification Tags

| Tag | Description | Priority |
|-----|-------------|----------|
| `STANDARD` | Clean, bordered table, single page | P0 (test first) |
| `MULTI_PAGE` | Table spans multiple pages | P0 |
| `UNBORDERED` | No table lines, whitespace-separated | P1 |
| `SEMI_BORDERED` | Partial lines (horizontal only) | P1 |
| `WRAPPED_TEXT` | Transactions span multiple visual rows | P1 |
| `WATERMARK` | Has watermark overlapping table | P2 |
| `SCANNED` | Scanned document (no text layer) | P3 (unsupported v1) |
| `PASSWORD` | Password protected | P3 |
| `ROTATED` | Rotated pages | P2 |
| `MULTI_TABLE` | Multiple tables per page | P2 |
| `NON_ENGLISH` | Non-English content | P3 (unsupported v1) |

### 5.2 Testing Priority

**Phase 1 (Core):** `STANDARD`, `MULTI_PAGE`
**Phase 2 (Common Edge Cases):** `UNBORDERED`, `SEMI_BORDERED`, `WRAPPED_TEXT`
**Phase 3 (Rare Edge Cases):** `WATERMARK`, `ROTATED`, `MULTI_TABLE`
**Phase 4 (Unsupported):** `SCANNED`, `PASSWORD`, `NON_ENGLISH` (verify error handling)

---

## 6. Additional Considerations

### 6.1 Performance Benchmarking

Track extraction speed alongside accuracy:

```csv
file,pages,transactions,extraction_time_ms,time_per_page_ms
hdfc_001.pdf,3,45,1250,417
icici_002.pdf,5,120,2800,560
```

**Targets:**
- < 500ms per page (without ML)
- < 1s per page (with SLM fallback)

### 6.2 Regression Detection

Automatically flag when:
- Any previously passing file starts failing (100% → <100%)
- Overall accuracy drops by > 0.5% from previous run
- New error types appear

### 6.3 Bank Coverage Matrix

Track which banks/formats are tested:

| Bank | Savings | Current | OD | Files | Last Updated |
|------|---------|---------|-----|-------|--------------|
| HDFC | ✓ (3) | ✓ (2) | ✗ | 5 | 2024-01-15 |
| ICICI | ✓ (2) | ✓ (1) | ✓ (1) | 4 | 2024-01-14 |
| SBI | ✓ (4) | ✓ (2) | ✗ | 6 | 2024-01-16 |

### 6.4 Confidence Calibration (Future)

Compare system confidence scores with actual accuracy:

```
If confidence = 0.95, actual accuracy should be ~95%
```

Track calibration curve to identify over/under-confident extractions.

### 6.5 Error Categorization

Standardized error codes for analysis:

| Code | Description |
|------|-------------|
| `DATE_PARSE_ERROR` | Date not parsed or parsed incorrectly |
| `AMOUNT_MISMATCH` | Amount differs from ground truth |
| `DESC_PARTIAL` | Description partially extracted |
| `DESC_WRONG` | Wrong description entirely |
| `ROW_MISSING` | Ground truth row not extracted |
| `ROW_EXTRA` | Extracted row not in ground truth |
| `ROW_DUPLICATE` | Same transaction extracted twice |
| `COLUMN_SHIFT` | Values shifted to wrong columns |
| `MERGE_ERROR` | Wrapped rows merged incorrectly |
| `PAGE_BREAK_ERROR` | Cross-page merge failed |

---

## 7. Implementation Files

```
tests/
├── evaluation/
│   ├── __init__.py
│   ├── comparator.py        # Transaction comparison logic
│   ├── metrics.py           # Accuracy calculations
│   ├── csv_loader.py        # Ground truth CSV loading
│   ├── report_generator.py  # Excel error reports
│   ├── run_tracker.py       # Run history management
│   └── runner.py            # Main evaluation runner
├── results/
│   ├── run_history.csv
│   └── run_{id}/
│       ├── summary.json
│       └── errors/
│           └── {file}_errors.xlsx
├── data/
│   └── ground_truth/
│       └── {bank}/
│           ├── {file}.pdf
│           └── {file}.csv
└── conftest.py              # Fixtures for evaluation tests
```

---

## 8. Test Types & Ground Truth Scope

### 8.1 What Your Ground Truth Validates

Your CSV ground truth validates the **end-to-end pipeline**:

```
PDF → [Module 1] → [Module 2] → [Module 3] → Transactions CSV
                                                    ↑
                                            Ground Truth Here
```

This means you can only measure **final output accuracy**, not intermediate module accuracy.

### 8.2 Test Type Comparison

| Test Type | Ground Truth Needed | What It Tests | Your Coverage |
|-----------|---------------------|---------------|---------------|
| **Unit Tests** (existing) | Mock data (synthetic) | Code logic, edge cases | ✅ 430 tests |
| **Integration Tests** | File-level CSV | End-to-end accuracy | ✅ Your CSVs |
| **Module-specific Tests** | Intermediate outputs | Individual module accuracy | ❌ Not available |

### 8.3 What You CAN Test with File-Level Ground Truth

1. **End-to-End Accuracy:** Does the system produce correct transactions?
2. **Regression Detection:** Did a change break previously working files?
3. **Bank Coverage:** Which banks/formats work correctly?
4. **Error Patterns:** What types of errors are most common?

### 8.4 What You CANNOT Test Without Module-Specific Ground Truth

| Module | What You'd Need | Why It's Hard |
|--------|-----------------|---------------|
| Table Detection | Annotated bounding boxes | Manual labeling tedious |
| Column Detection | Column x-coordinates | Varies per file |
| Header Detection | Semantic type labels | Could derive from CSV headers |
| Cell Extraction | Word → cell assignments | Very granular |

### 8.5 Practical Approach: Derive Confidence from End-to-End

Even without module-specific ground truth, you can infer module issues from error patterns:

| Error Pattern | Likely Module Issue |
|---------------|---------------------|
| All rows shifted one column | Column detection |
| Header row appears in data | Header detection |
| Transactions split incorrectly | Row merging |
| Multi-page files fail | Table merger |
| Amounts wrong, text correct | Amount parsing |
| Dates wrong | Date parsing |

### 8.6 Recommendation

**Phase 1:** Use file-level ground truth for integration tests
- This is what matters for production accuracy
- Catches real-world issues

**Phase 2 (Optional):** Create module-specific golden files for critical cases
- Run system on a few files
- Manually verify intermediate outputs
- Save as golden files for regression testing

**Keep existing unit tests:** They validate code logic even though they don't validate real extraction accuracy.

---

## 9. CLI Interface

```bash
# Run evaluation on all ground truth files
python -m tests.evaluation.runner

# Run on specific bank
python -m tests.evaluation.runner --bank hdfc

# Run on specific file
python -m tests.evaluation.runner --file tests/data/ground_truth/hdfc/hdfc_001.pdf

# Compare two runs
python -m tests.evaluation.runner --compare run_001 run_002

# Generate coverage report
python -m tests.evaluation.runner --coverage
```
