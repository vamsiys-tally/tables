# CLAUDE.md - AI Assistant Guide for Tables Project

## Project Overview

**Tables** is a PDF table detection and extraction system designed to identify, extract, and analyze tabular data from bank statement PDFs. The project focuses on processing Indian bank statement formats with various table layouts (bordered and unbordered).

**Key Capabilities:**
- PDF table structure detection using geometric analysis
- Semantic header identification using sentence transformers
- Cell, row, and column extraction with coordinate alignment
- Multi-page document processing

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .

# Run header detection demo
python src/header_detector/detector.py

# Explore table detection interactively
jupyter notebook src/tables/Interactive\ -\ inference.py.ipynb
```

## Repository Structure

```
/home/user/tables/
├── src/
│   ├── tables/                   # Core table detection module
│   │   ├── classes.py            # Data structures (Cell, Row, Column, Table, Page, Document)
│   │   ├── inference.py          # Advanced PDF inference with rectangle detection
│   │   ├── headers.py            # Header keywords and sentence transformer embeddings
│   │   ├── config.yaml           # Detection algorithm parameters
│   │   └── workspace.py          # PDF analysis utilities
│   └── header_detector/
│       └── detector.py           # Semantic header detection algorithm
├── tests/
│   ├── data/                     # 146 PDF test samples from various Indian banks
│   ├── Priority Banks/           # Specialized test data (unbordered tables)
│   └── results/                  # Test output results
├── main.py                       # Entry point (currently empty)
├── requirements.txt              # Python dependencies
└── setup.py                      # Package configuration
```

## Key Files

| File | Purpose |
|------|---------|
| `src/tables/classes.py` | Core data structures: `Cell`, `Row`, `Column`, `Table`, `Page`, `Document` |
| `src/tables/inference.py` | Advanced PDF processing with DBSCAN clustering and rectangle detection |
| `src/tables/headers.py` | Pre-computed header embeddings (150+ banking keywords) |
| `src/tables/config.yaml` | Configurable detection parameters |
| `src/header_detector/detector.py` | Semantic header block detection |

## Technology Stack

- **Python 3.11+**
- **PDF Processing:** pdfplumber, pdfminer.six, pypdfium2
- **ML/NLP:** sentence-transformers (all-MiniLM-L6-v2), torch, scikit-learn
- **Data Processing:** pandas, numpy, scipy
- **Visualization:** matplotlib, Jupyter notebooks

## Development Workflows

### Table Detection Pipeline

```
Load PDF → Extract page elements → Clean coordinates →
Detect rectangles → Initialize cells → Group into rows/columns →
Identify headers → Detect table structure
```

### Header Detection Pipeline

```
Parse PDF → Extract horizontal edges → Extract characters →
Group chars to words → Convert words to phrases →
Compute semantic similarity → Rank header blocks → Export results
```

## Code Conventions

### Naming
- **Classes:** PascalCase (`Cell`, `Page`, `Document`)
- **Methods/Functions:** snake_case (`get_rows`, `extract_horizontal_edges`)
- **Constants:** UPPERCASE (`HEADER_KEYWORDS`, `HEADER_EMBEDDINGS`)
- **Private methods:** Prefix with underscore (`_merge_lines`)

### Coordinate System
- **PDF Origin:** Bottom-left (0, 0)
- **Bounding Box Format:** `(x0, y0, x1, y1)` where (x0, y0) is bottom-left
- **Tolerance:** Default 3.0 points for fuzzy coordinate matching
- **Grid Resolution:** Snapping to 1 point resolution by default

### Type Hints
Modern Python 3.9+ type hints are used throughout:
```python
def get_rows(self) -> list[Row]:
def get_bounding_box(self) -> tuple[float, float, float, float]:
```

### Algorithm Patterns
- **DBSCAN Clustering:** For grouping nearby coordinates
- **Graph Traversal:** Depth-first search for connected cells
- **Intersection Detection:** Finding rectangles from line intersections
- **Semantic Similarity:** Sentence transformers for header matching

## Configuration

Key parameters in `src/tables/config.yaml`:

```yaml
cell_detection:
  tol: 3.0              # Coordinate tolerance
  min_height: 10        # Minimum cell height
  min_width: 10         # Minimum cell width
  db_scan_metric: euclidean
  db_scan_algorithm: kd_tree

header_identification:
  sentence_transformer: all-MiniLM-L6-v2
  header_similarity_threshold: 0.8

table_detection:
  rows_after_non_table_row: 10
```

## Testing

### Test Data
- **146 PDF files** from various Indian banks in `tests/data/`
- **Priority Banks** subset with 54+ unbordered table PDFs

### Running Tests
The project uses script-based testing embedded in main modules:

```python
# In classes.py (lines 1042-1114) - batch processing 36 sample PDFs
# In inference.py (lines 696-752) - multi-page processing with header detection
# In detector.py (lines 299-330) - header detection batch processing
```

Test results are written to `tests/results/` with timestamped filenames.

## Important Notes for AI Assistants

### When Modifying Code

1. **Preserve coordinate conventions:** Always use (x0, y0, x1, y1) format with bottom-left origin
2. **Respect tolerance values:** Use configurable tolerances from config.yaml, not hardcoded values
3. **Maintain type hints:** Add proper type annotations to new methods
4. **Follow class structure:** New data structures should follow the Cell/Row/Column/Table/Page/Document hierarchy

### When Adding Features

1. **Use sentence transformers** for any text similarity comparisons
2. **Apply DBSCAN clustering** for coordinate-based grouping
3. **Add test PDFs** to `tests/data/` for new bank formats
4. **Update config.yaml** for new configurable parameters

### Common Pitfalls

1. **PDF coordinates vs image coordinates:** PDF origin is bottom-left, not top-left
2. **Floating-point precision:** Always use tolerance-based comparisons for coordinates
3. **Memory management:** Call `gc.collect()` after processing large PDFs
4. **Duplicate cells:** Use `drop_duplicates()` on DataFrames to avoid duplicate entries

### File Paths
Test scripts contain hardcoded paths. When modifying, update paths relative to project root:
```python
# Use relative paths from project root
tests_dir = "tests/data/"
results_dir = "tests/results/"
```

## Git Workflow

**Branch naming:** `{author}/{feature-name}`

**Recent development focus:**
- Header detection implementation (PRs #9, #10)
- Rows and columns detection (PR #7)
- Character snapping improvements
- Table detection core (PRs #5, #6)

## Architecture Diagram

```
                    ┌──────────────────────────────────────┐
                    │              Document                │
                    │  (multi-page PDF management)         │
                    └──────────────┬───────────────────────┘
                                   │
                    ┌──────────────▼───────────────────────┐
                    │                Page                  │
                    │  - extract lines/edges/chars         │
                    │  - detect rectangles                 │
                    │  - initialize cells                  │
                    └──────────────┬───────────────────────┘
                                   │
          ┌────────────────────────┼────────────────────────┐
          │                        │                        │
   ┌──────▼──────┐          ┌──────▼──────┐          ┌──────▼──────┐
   │    Cell     │          │    Row      │          │   Column    │
   │ (x0,y0,x1,y1)│◄─────────│ (cell list) │──────────►│ (cell list) │
   │   + text    │          └─────────────┘          └─────────────┘
   └─────────────┘
          │
          ▼
   ┌─────────────┐
   │   Table     │
   │ rows/columns│
   │   headers   │
   └─────────────┘
```

## Dependencies Installation

Full dependency list with version requirements:

```bash
# Core dependencies
pip install pdfplumber==0.11.7
pip install pdfminer.six==20250506
pip install pypdfium2==4.30.0
pip install sentence-transformers
pip install pandas==2.3.2
pip install numpy==2.3.2
pip install matplotlib==3.10.5
pip install Pillow==11.3.0
pip install scikit-learn

# Development dependencies
pip install ipykernel ipython jupyter_client jupyter_core
```

## Future Development Areas

Areas identified for improvement:
1. Add formal unit tests with pytest
2. Implement CI/CD pipeline
3. Create API documentation
4. Add support for more international bank formats
5. Improve error recovery in batch processing
6. Remove duplicate `classes copy.py` file
