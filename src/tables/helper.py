import pdfplumber as plumber
import pandas as pd
import numpy as np
from collections import defaultdict
import os
import gc

class Word:
    def __init__(self, text, x0, y0, x1, y1, page_number, page_height):
        self.text = text
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.page_number = page_number
        self.width = x1 - x0
        self.height = y1 - y0


        # Flipped coordinates (top-left origin)
        self.flipped_y0 = page_height - y1
        self.flipped_y1 = page_height - y0


    def __repr__(self):
        return f"Word('{self.text}', Pg:{self.page_number}, ({self.x0:.1f}, {self.y0:.1f}))"

class Cell:
    def __init__(self, x0, y0, x1, y1, page_number: int, page_height: float):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.page_number = page_number
        self.height = abs(self.y1 - self.y0)
        self.width = abs(self.x1 - self.x0)
        self.text = ""
        self.words = []  # Store Word objects within this cell

        # Flipped coordinates
        self.flipped_y0 = page_height - self.y1
        self.flipped_y1 = page_height - self.y0

    def __repr__(self):
        return f"Cell(Pg:{self.page_number}, ({self.x0}, {self.y0})-({self.x1}, {self.y1}))"

class Row:
    def __init__(self, cells: list[Cell], page_number: int, page_height: float) -> None:
        self.cells = cells
        self.length = len(cells)
        self.page_number = page_number
        self.x0 = min(cell.x0 for cell in cells)
        self.y0 = min(cell.y0 for cell in cells)
        self.x1 = max(cell.x1 for cell in cells)
        self.y1 = max(cell.y1 for cell in cells)
        self.text = [cell.text for cell in cells]

        # Flipped coordinates
        self.flipped_y0 = page_height - self.y1
        self.flipped_y1 = page_height - self.y0

    def __repr__(self):
        return f"Row(Pg:{self.page_number}, ({self.x0}, {self.y0})-({self.x1}, {self.y1}))"

class ComprehensivePDFProcessor:
    def __init__(self, path: str, password: str = None):
        self.path = path
        self.password = password
        
        try:
            self.pdf = plumber.open(path, password=password)
        except Exception as e:
            raise RuntimeError(f"Failed to open PDF {path}. Error: {e}")
        
        # Word building results
        self.words = []
        self.text_lines = []
        
        # Geometric structure results
        self.cells = []
        self.rows = []
        
        # Raw extraction results
        self.all_chars = []
        self.all_lines = []
        self.all_edges = []
        self.all_rectangles = []
    
    def _cluster_characters_to_words(self, chars_df, page_height, x_tolerance=0, y_tolerance=0):
        """Fast word building by clustering adjacent characters"""
        if chars_df.empty:
            return []
        
        # Sort characters by position (left to right, top to bottom)
        chars_sorted = chars_df.sort_values(['y0', 'x0']).reset_index(drop=True)
        
        words = []
        current_word_chars = []
        
        for idx, char in chars_sorted.iterrows():
            if not current_word_chars:
                current_word_chars.append(char)
                continue
            
            last_char = current_word_chars[-1]
            
            # Check if character belongs to current word
            same_line = abs(char['y0'] - last_char['y0']) <= y_tolerance
            close_horizontally = (char['x0'] - last_char['x1']) <= x_tolerance
            
            if same_line and close_horizontally:
                current_word_chars.append(char)
            else:
                # Finalize current word
                if current_word_chars:
                    word = self._create_word_from_chars(current_word_chars, page_height)
                    if word.text.strip():
                        words.append(word)
                
                current_word_chars = [char]
        
        # Don't forget the last word
        if current_word_chars:
            word = self._create_word_from_chars(current_word_chars, page_height)
            if word.text.strip():
                words.append(word)
        
        return words

    def _create_word_from_chars(self, chars, page_height):
        """Create a Word object from a list of character dictionaries"""
        text = "".join([c['text'] for c in chars])
        x0 = min(c['x0'] for c in chars)
        y0 = min(c['y0'] for c in chars)
        x1 = max(c['x1'] for c in chars)
        y1 = max(c['y1'] for c in chars)
        page_number = chars[0].get('page_number', 0)
        
        return Word(text, x0, y0, x1, y1, page_number, page_height)
    
    @staticmethod
    def _detect_orientation(x0, y0, x1, y1, tol=0.0):
        """Detect if a line is horizontal, vertical, or other"""
        if np.isclose(x0, x1, atol=tol):
            return "vertical"
        elif np.isclose(y0, y1, atol=tol):
            return "horizontal"
        else:
            return "other"

    @staticmethod
    def _is_intersection(hor_line: dict, vert_line: dict) -> tuple:
        """Check if horizontal and vertical lines intersect"""
        h_y = hor_line["y0"]
        h_x0, h_x1 = hor_line["x0"], hor_line["x1"]
        v_x = vert_line["x0"]
        v_y0, v_y1 = vert_line["y0"], vert_line["y1"]

        if (h_x0 <= v_x <= h_x1) and (v_y0 <= h_y <= v_y1):
            return (v_x, h_y)
        else:
            return (None, None)

    @staticmethod
    def _find_intersections(hor_lines: pd.DataFrame, vert_lines: pd.DataFrame) -> list:
        """Find all intersections between horizontal and vertical lines (optimized)."""
        if hor_lines.empty or vert_lines.empty:
            return []

        intersections = []
        for col in ["x0", "x1", "y0"]:
            if col in hor_lines.columns:
                hor_lines[col] = hor_lines[col].astype(float)
        for col in ["x0", "y0", "y1"]:
            if col in vert_lines.columns:
                vert_lines[col] = vert_lines[col].astype(float)

        # Work with numpy arrays for speed
        hor = hor_lines[["x0", "x1", "y0"]].to_numpy()
        vert = vert_lines[["x0", "y0", "y1"]].to_numpy()

        for x0_h, x1_h, y_h in hor:
            # Filter verticals that can intersect this horizontal line
            mask = (vert[:, 0] >= x0_h) & (vert[:, 0] <= x1_h) & \
                (vert[:, 1] <= y_h) & (vert[:, 2] >= y_h)
            candidates = vert[mask]
            if candidates.size > 0:
                for v in candidates:
                    intersections.append((v[0], y_h))

        return intersections

    @staticmethod
    def _detect_minimal_rectangles(intersections: list, x_coords: list, y_coords: list) -> list:
        """Detect minimal rectangles from intersection points"""
        if len(x_coords) < 2 or len(y_coords) < 2:
            return []

        inter_set = set(intersections)
        rectangles = []

        for j in range(len(y_coords) - 1):
            y1, y2 = y_coords[j], y_coords[j + 1]
            
            # Find x coordinates that form valid boundaries for this row
            row_x_set = set()
            for x, y in intersections:
                if y == y1 and (x, y2) in inter_set:
                    row_x_set.add(x)
                elif y == y2 and (x, y1) in inter_set:
                    row_x_set.add(x)

            if len(row_x_set) < 2:
                continue

            row_x = sorted(row_x_set)
            
            # Create rectangles from consecutive x coordinates
            for i in range(len(row_x) - 1):
                x1, x2 = row_x[i], row_x[i + 1]
                rect = {
                    "top_left": (x1, y1),
                    "top_right": (x2, y1),
                    "bottom_left": (x1, y2),
                    "bottom_right": (x2, y2),
                }
                rectangles.append(rect)

        return rectangles

    @staticmethod
    def _is_adjacent(cell1: Cell, cell2: Cell, type="row", tol=1.0):
        """Check if two cells are adjacent"""
        if type == "row":
            return np.isclose(cell1.x1, cell2.x0, atol=tol)
        elif type == "column":
            return np.isclose(cell1.y1, cell2.y0, atol=tol)

    def _get_page_data(self, page_num):
        """Extract characters, lines, and edges from a page (but do NOT store chars globally)"""
        page = self.pdf.pages[page_num]

        # Get characters (only used for building words on this page)
        chars = pd.DataFrame(page.chars)
        if not chars.empty:
            chars = chars[['text', 'x0', 'y0', 'x1', 'y1']].drop_duplicates().reset_index(drop=True)
            chars['page_number'] = page_num

        # Get lines
        lines = pd.DataFrame(page.lines) if page.lines else pd.DataFrame()
        if not lines.empty:
            lines = lines[['x0', 'y0', 'x1', 'y1']].drop_duplicates()
            lines['orientation'] = lines.apply(
                lambda row: self._detect_orientation(row['x0'], row['y0'], row['x1'], row['y1']), axis=1
            )
            lines['page_number'] = page_num

        # Get edges
        edges = pd.DataFrame(page.edges) if page.edges else pd.DataFrame()
        if not edges.empty:
            edges = edges[['x0', 'y0', 'x1', 'y1']].drop_duplicates()
            edges['orientation'] = edges.apply(
                lambda row: self._detect_orientation(row['x0'], row['y0'], row['x1'], row['y1']), axis=1
            )
            edges['page_number'] = page_num

        # Store raw data (NO characters anymore)
        if not lines.empty:
            self.all_lines.extend(lines.to_dict('records'))
        if not edges.empty:
            self.all_edges.extend(edges.to_dict('records'))

        return chars, lines, edges, page.height, page.width

    def _clean_coordinates(self, chars, lines, edges, height, width, tol=3.0):
        """Clean and snap coordinates to grid using simple rounding (no clustering)."""
        # Skip chars as they are unused for geometry
        for df, df_name in [(lines, 'lines'), (edges, 'edges')]:
            if df.empty:
                continue

            # Ensure all coordinate columns are float
            for col in ["x0", "y0", "x1", "y1"]:
                if col in df.columns:
                    df[col] = df[col].astype(float)

            # Simple snapping: round to nearest integer for alignment
            for col in ["x0", "x1", "y0", "y1"]:
                df[col] = np.round(df[col].values).astype(float)

            # Fix line orientations
            if df_name in ['lines', 'edges']:
                for idx, row in df.iterrows():
                    if row["orientation"] == "horizontal":
                        df.at[idx, "y1"] = float(row["y0"])
                    elif row["orientation"] == "vertical":
                        df.at[idx, "x1"] = float(row["x0"])

        return chars, lines, edges

    # def _detect_rectangles(self, lines, edges):
    #     """Detect rectangles from lines and edges"""
    #     hor_lines = pd.concat([
    #         lines[lines["orientation"] == "horizontal"] if not lines.empty else pd.DataFrame(),
    #         edges[edges["orientation"] == "horizontal"] if not edges.empty else pd.DataFrame()
    #     ])
        
    #     vert_lines = pd.concat([
    #         lines[lines["orientation"] == "vertical"] if not lines.empty else pd.DataFrame(),
    #         edges[edges["orientation"] == "vertical"] if not edges.empty else pd.DataFrame()
    #     ])
        
    #     if hor_lines.empty or vert_lines.empty:
    #         return []
        
    #     intersections = self._find_intersections(hor_lines, vert_lines)
        
    #     if not intersections:
    #         return []
        
    #     x_coords = sorted(set(x for x, y in intersections))
    #     y_coords = sorted(set(y for x, y in intersections))
        
    #     rectangles = self._detect_minimal_rectangles(intersections, x_coords, y_coords)
        
    #     # Store detected rectangles
    #     for rect in rectangles:
    #         rect_copy = rect.copy()
    #         rect_copy['page_number'] = lines['page_number'].iloc[0] if not lines.empty else edges['page_number'].iloc[0]
    #         self.all_rectangles.append(rect_copy)
        
    #     return rectangles

    def _create_cells_from_rectangles(self, rectangles, words, page_num, page_height, tol=3.0):
        """Create cells from detected rectangles and fill with words (not just characters)"""
        cells = []
        
        for rect in rectangles:
            x0, y0 = rect["top_left"]
            x1, y1 = rect["bottom_right"]
            
            cell = Cell(x0, y0, x1, y1, page_num, page_height)
            
            # Fill cell with words that fall within bounds
            for word in words:
                if (word.x0 >= x0 - tol and word.x1 <= x1 + tol and
                    word.y0 >= y0 - tol and word.y1 <= y1 + tol):
                    cell.words.append(word)
                    cell.text += word.text + " "
            
            cell.text = cell.text.strip()
            cells.append(cell)
        
        return sorted(cells, key=lambda x: (x.x0, x.flipped_y0))

    def _build_rows(self, cells, page_height, tol=1.0):
        """Group cells into rows"""
        cells_sorted = sorted(cells, key=lambda c: (c.flipped_y0, c.x0))
        
        rows = []
        current_group = []
        
        for cell in cells_sorted:
            if not current_group:
                current_group.append(cell)
                continue
            
            prev_cell = current_group[-1]
            same_level = abs(cell.flipped_y0 - prev_cell.flipped_y0) < tol
            adjacent = same_level and self._is_adjacent(prev_cell, cell, type="row", tol=tol)
            
            if same_level and adjacent:
                current_group.append(cell)
            else:
                if current_group:
                    rows.append(Row(current_group, cell.page_number, page_height))
                current_group = [cell]
        
        if current_group:
            rows.append(Row(current_group, current_group[0].page_number, page_height))
        
        return sorted(rows, key=lambda x: x.flipped_y0)

    def process_pages(self, page_numbers=None, 
                  word_x_tolerance=2.0, word_y_tolerance=1.0):
        """
        Extract only words, lines, and edges from the PDF pages.
        Skips rectangle, cell, and row processing.
        """
        if page_numbers is None:
            page_numbers = list(range(len(self.pdf.pages)))

        all_words = []

        for page_num in page_numbers:
            print(f"Processing page {page_num + 1}...")

            # Get page data
            chars, lines, edges, height, width = self._get_page_data(page_num)

            # === FAST WORD BUILDING ===
            page_words = self._cluster_characters_to_words(
                chars, height, word_x_tolerance, word_y_tolerance
            )
            all_words.extend(page_words)

            # === CLEAN LINES AND EDGES ===
            _, lines_clean, edges_clean = self._clean_coordinates(
                chars.copy(), lines.copy(), edges.copy(), height, width
            )

        # Store results
        self.words = all_words

        return self.words, self.all_lines, self.all_edges

    def save_comprehensive_results(self, output_path):
        """Save extracted words, lines, and edges (skip cells/rows/rectangles)"""
        # Sort words by flipped coordinates (top-to-bottom, left-to-right)
        sorted_words = sorted(self.words, key=lambda w: (w.page_number, w.flipped_y0, w.x0))

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"PDF Word, Line, and Edge Extraction Results\n")
            f.write(f"Source: {self.path}\n")
            f.write(f"Total Words: {len(sorted_words)}\n")
            f.write(f"Total Lines: {len(self.all_lines)}\n")
            f.write(f"Total Edges: {len(self.all_edges)}\n")
            f.write("="*80 + "\n\n")

            # === WORDS ===
            f.write("DETECTED WORDS:\n")
            f.write("-"*60 + "\n")
            for i, word in enumerate(sorted_words):
                f.write(f"Word {i+1}: '{word.text}' [BBox: ({word.x0:.1f}, {word.y0:.1f})-({word.x1:.1f}, {word.y1:.1f})] "
                        f"[Flipped: ({word.x0:.1f}, {word.flipped_y0:.1f})-({word.x1:.1f}, {word.flipped_y1:.1f})] "
                        f"Page {word.page_number} [Size: {word.width:.1f}x{word.height:.1f}]\n")

            f.write("\n" + "="*80 + "\n")

            # === LINES ===
            f.write("DETECTED LINES:\n")
            f.write("-"*60 + "\n")
            for i, line in enumerate(self.all_lines):
                orient = line.get('orientation', 'unknown')
                f.write(f"Line {i+1}: [BBox: ({line['x0']:.1f}, {line['y0']:.1f})-({line['x1']:.1f}, {line['y1']:.1f})] "
                        f"Type: {orient} Page {line.get('page_number', 0)} "
                        f"[Length: {abs(line['x1']-line['x0']):.1f}x{abs(line['y1']-line['y0']):.1f}]\n")

            f.write("\n" + "="*80 + "\n")

            # === EDGES ===
            f.write("DETECTED EDGES:\n")
            f.write("-"*60 + "\n")
            for i, edge in enumerate(self.all_edges):
                orient = edge.get('orientation', 'unknown')
                f.write(f"Edge {i+1}: [BBox: ({edge['x0']:.1f}, {edge['y0']:.1f})-({edge['x1']:.1f}, {edge['y1']:.1f})] "
                        f"Type: {orient} Page {edge.get('page_number', 0)} "
                        f"[Length: {abs(edge['x1']-edge['x0']):.1f}x{abs(edge['y1']-edge['y0']):.1f}]\n")

    def close(self):
        """Close the PDF file"""
        if hasattr(self, 'pdf'):
            self.pdf.close()

def main():
    """Main function to extract all elements with their bounding boxes from all PDFs in a folder"""
    # Configuration
    folder_path = "tests/Priority Banks"   # Change this to your folder path
    pages_to_process = None                # None = process ALL pages
    password = None                        # Set password if PDF is protected


    # Tolerance settings
    word_x_tolerance = 2.0      # Character spacing within words
    word_y_tolerance = 1.0      # Vertical alignment tolerance for words


    try:
        # Loop through all PDF files in folder
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(".pdf"):
                pdf_path = os.path.join(folder_path, filename)
                output_file = os.path.splitext(filename)[0] + "_analysis.txt"


                print(f"\n🔍 Processing: {pdf_path}")


                # Initialize processor
                processor = ComprehensivePDFProcessor(pdf_path, password=password)


                # Process ALL pages
                words, cells, rows = processor.process_pages(
                    page_numbers=pages_to_process,
                    word_x_tolerance=word_x_tolerance,
                    word_y_tolerance=word_y_tolerance,
                )


                print(f" Extraction completed for {filename}")
                print(f"   Found {len(words)} words, {len(cells)} cells, {len(rows)} rows")
                # print(f"   Found {len(processor.all_chars)} characters, {len(processor.all_lines)} lines, {len(processor.all_edges)} edges")
                print(f"   Found {len(processor.all_lines)} lines, {len(processor.all_edges)} edges")
                print(f"   Found {len(processor.all_rectangles)} detected rectangles")


                # Save comprehensive results with all elements
                processor.save_comprehensive_results(output_file)
                print(f"   Results saved to: {output_file}")


                # Cleanup
                processor.close()
                gc.collect()


    except Exception as e:
        print(f" Error processing PDFs: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()