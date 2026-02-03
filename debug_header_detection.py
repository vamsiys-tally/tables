#!/usr/bin/env python3
"""Debug script to investigate header detection issues."""

import sys
sys.path.insert(0, 'src')

from tables.reader import PDFDocument
from tables.detector.table_detector import TableDetector, TextBlock
from tables.utils.geometry import BoundingBox, cluster_by_y_coordinate

def debug_pdf(pdf_path: str):
    """Debug header detection for a specific PDF."""
    print(f"\n{'='*60}")
    print(f"Debugging: {pdf_path}")
    print(f"{'='*60}")

    # Open PDF
    pdf = PDFDocument.open(pdf_path)
    print(f"Pages: {pdf.page_count}")

    # Create detector
    detector = TableDetector(use_slm=False)  # Disable SLM for faster debugging

    # Process page 0
    page_num = 0
    print(f"\n--- Page {page_num} ---")

    # Extract text blocks
    text_blocks = detector._extract_text_blocks(pdf, page_num)
    print(f"Total text blocks: {len(text_blocks)}")

    # Group by y-coordinate
    boxes = [block.bbox for block in text_blocks]
    row_clusters = cluster_by_y_coordinate(boxes, tolerance=5.0)
    print(f"Row clusters: {len(row_clusters)}")

    # Show first 15 rows (sorted by y)
    print("\n--- First 15 row clusters (sorted by y) ---")
    sorted_clusters = sorted(row_clusters, key=lambda c: min(b.y0 for b in c))

    for i, cluster in enumerate(sorted_clusters[:15]):
        y_val = min(b.y0 for b in cluster)
        # Get text for this cluster
        row_texts = []
        for block in text_blocks:
            if any(block.bbox.is_horizontally_aligned(box, tolerance=5.0) for box in cluster):
                row_texts.append((block.bbox.x0, block.text))
        row_texts.sort(key=lambda x: x[0])
        text_str = " | ".join([t[1] for t in row_texts])
        print(f"Row {i} (y={y_val:.1f}, cols={len(cluster)}): {text_str[:100]}")

    # Now detect table regions
    print("\n--- Table Regions ---")
    regions = detector._detect_table_regions(text_blocks, page_num)
    print(f"Number of regions: {len(regions)}")

    for r_idx, region in enumerate(regions):
        print(f"\nRegion {r_idx}:")
        print(f"  Bounds: ({region.bounds.x0:.1f}, {region.bounds.y0:.1f}) - ({region.bounds.x1:.1f}, {region.bounds.y1:.1f})")
        print(f"  Text blocks in region: {len(region.text_blocks)}")

        # Group into rows
        region.rows = detector._group_into_rows(region.text_blocks)
        print(f"  Rows in region: {len(region.rows)}")

        # Show first 5 rows
        print(f"  First 5 rows:")
        for row_idx, row in enumerate(region.rows[:5]):
            y_val = min(b.bbox.y0 for b in row) if row else 0
            texts = [b.text for b in row]
            print(f"    Row {row_idx} (y={y_val:.1f}): {texts}")

        # Check if header row texts match header keywords
        print(f"\n  Checking for headers in first 3 rows:")
        for row_idx, row in enumerate(region.rows[:3]):
            texts = [b.text for b in row]
            from tables.detector.keywords import is_likely_header_row, HEADER_KEYWORDS
            is_header = is_likely_header_row(texts)
            print(f"    Row {row_idx}: is_header={is_header}, texts={texts}")

    # Now run full detection
    print("\n--- Full Detection Result ---")
    detector2 = TableDetector(use_slm=True)
    result = detector2.detect(pdf)

    print(f"Tables found: {len(result.tables)}")
    for table in result.tables:
        print(f"\nTable: {table.table_id}")
        print(f"  Content type: {table.content_type.value if table.content_type else None}")
        print(f"  Header rows: {table.header_row_indices}")
        print(f"  Columns ({len(table.columns)}):")
        for col in table.columns:
            print(f"    - {col.semantic_type}: '{col.header_text}'")


if __name__ == "__main__":
    import glob

    # Find first PDF in ground truth
    pdf_files = glob.glob("tests/data/ground_truth/**/*.pdf", recursive=True)

    if pdf_files:
        debug_pdf(pdf_files[0])
    else:
        print("No PDF files found in tests/data/ground_truth/")
