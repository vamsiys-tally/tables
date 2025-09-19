# %%
import pdfplumber as plumber
import pandas as pd
import numpy as np
import torch
import os
import gc
from sklearn.cluster import DBSCAN
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util
from tables.headers import HEADER_EMBEDDINGS,header_words
import json
import math
# %%


class Cell:
    def __init__(self, x0, y0, x1, y1, page_number: int):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.page_number = page_number

        self.height = abs(self.y1 - self.y0)
        self.width = abs(self.x1 - self.x0)

        self.top = None
        self.bottom = None
        self.left = None
        self.right = None

        self.text = ""

    def __repr__(self):
        return f"Cell(Pg:{self.page_number}, ({self.x0}, {self.y0})-({self.x1}, {self.y1}))"


# %%
class Row:
    active_rows = []

    def __init__(self, cells: list[Cell], page_number: int) -> None:
        self.cells = cells
        self.length = len(cells)
        self.page_number = page_number

        Row.active_rows.append(self)

        self.x0 = min(cell.x0 for cell in cells)
        self.y0 = min(cell.y0 for cell in cells)
        self.x1 = max(cell.x1 for cell in cells)
        self.y1 = max(cell.y1 for cell in cells)

        self.text = [cell.text for cell in cells]
        self.header_row = False

    def __repr__(self):
        return (
            f"Row(Pg:{self.page_number}, ({self.x0}, {self.y0})-({self.x1}, {self.y1}))"
        )

    @classmethod
    def get_rows(cls, page_number: int = None):
        if page_number is None:
            return cls.active_rows
        else:
            return [row for row in cls.active_rows if row.page_number == page_number]


class Column:
    active_columns = []

    def __init__(self, cells: list[Cell], page_number: int) -> None:
        self.cells = cells
        self.length = len(cells)
        self.page_number = page_number

        Column.active_columns.append(self)

        self.x0 = min(cell.x0 for cell in cells)
        self.y0 = min(cell.y0 for cell in cells)
        self.x1 = max(cell.x1 for cell in cells)
        self.y1 = max(cell.y1 for cell in cells)

        self.text = [cell.text for cell in cells]

    def __repr__(self):
        return f"Column(Pg:{self.page_number}, ({self.x0}, {self.y0})-({self.x1}, {self.y1}))"

    @classmethod
    def get_columns(cls, page_number: int = None):
        if page_number is None:
            return cls.active_columns
        else:
            return [
                column
                for column in cls.active_columns
                if column.page_number == page_number
            ]


# %%
class PageRegion:
    pass


# %%
class Table:
    def __init__(self) -> None:
        self.rows = []
        self.columns = []

        self.header_row = None

    def __repr__(self):
        return f"Table(({len(self.rows)}, {len(self.columns)}))"

    def add_row(self, row: Row):
        self.rows.append(row)

    def add_column(self, column: Column):
        self.columns.append(column)

    def set_header_row(self, header_row: Row):
        self.header_row = header_row

    def get_header_row(self):
        return self.header_row


# %%
class Page:
    def __init__(self, page) -> None:
        self.page = page

        self.page_number = self.page.page_number
        self.height = self.page.height
        self.width = self.page.width

        self.row_grid = Page._generate_grid(self.height)
        self.column_grid = Page._generate_grid(self.width)

        self.chars = pd.DataFrame()
        self.lines = pd.DataFrame()
        self.edges = pd.DataFrame()
        self.rects = pd.DataFrame()

        self.cells = []
        self.rows = []
        self.columns = []

        self.model = None
        self.tables = []

    def __repr__(self) -> str:
        return f"Page({self.page_number}, {self.height}x{self.width})"

    def get_page_characters(self):
        if self.chars.empty:
            chars = self.page.chars

            char_df = Page._attribute_df(
                chars,
                mandatory=["text", "x0", "y0", "x1", "y1"],
                selection=["size", "width", "height"],
            )

            self.chars = char_df.drop_duplicates().reset_index(drop=True)

        return self.chars

    def get_page_lines(self):
        if self.lines.empty:
            lines = self.page.lines

            lines_df = Page._attribute_df(
                lines,
                mandatory=["x0", "y0", "x1", "y1"],
                selection=["orientation"],
            )

            lines_df["orientation"] = lines_df.apply(
                lambda row: Page._detect_orientation(
                    row["x0"], row["y0"], row["x1"], row["y1"]
                ),
                axis=1,
            )

            self.lines = lines_df.drop_duplicates().reset_index(drop=True)

        return self.lines

    def get_page_edges(self, tol: float = 2.0):
        if self.edges.empty:
            edges = self.page.edges

            edges_df = Page._attribute_df(
                edges,
                mandatory=["x0", "y0", "x1", "y1"],
                selection=["orientation"],
            )

            edges_df["orientation"] = edges_df.apply(
                lambda row: Page._detect_orientation(
                    row["x0"], row["y0"], row["x1"], row["y1"]
                ),
                axis=1,
            )

            self.edges = edges_df.drop_duplicates().reset_index(drop=True)

        return self.edges

    def get_page_rects(self):
        if self.rects.empty:
            rects = self.page.rects

            rects_df = Page._attribute_df(
                rects,
                mandatory=["x0", "y0", "x1", "y1"],
                selection=[],
            )

            self.rects = rects_df.drop_duplicates().reset_index(drop=True)

        return self.rects

    def clean_all_coords(
        self,
        tol: float = 3.0,
    ):
        groups = [["x0", "y0"], ["x1", "y1"]]

        for name in ["chars", "lines", "edges", "rects"]:
            c = getattr(self, name)
            if c.empty:
                continue

            for group in groups:
                points = np.vstack((c[group].values))

                db = DBSCAN(
                    eps=tol, min_samples=1, metric="euclidean", algorithm="kd_tree"
                ).fit(points)
                labels = db.labels_

                point_df = pd.DataFrame(points, columns=["x", "y"])
                point_df["cluster"] = labels
                centroids = point_df.groupby("cluster")[["x", "y"]].mean()

                mapped_points = np.array(
                    [centroids.loc[label].values for label in labels]
                )
                # new_coords = mapped_points.reshape(2, -1, 2)

                for i, col in enumerate(group):
                    c.loc[:, col] = mapped_points[:, i].astype(float)

            for col in ["x0", "y0", "x1", "y1"]:
                c[col] = c[col].apply(
                    lambda x: Page._snap_to_grid(
                        x, self.column_grid if col in ["x0", "x1"] else self.row_grid
                    )
                )

            if name in ["lines", "edges"]:
                for _, row in c.iterrows():
                    if row["orientation"] == "horizontal":
                        row["y1"] = row["y0"]
                    elif row["orientation"] == "vertical":
                        row["x1"] = row["x0"]
                    else:
                        row["x0"] = row["x0"]
                        row["x1"] = row["x1"]
                        row["y0"] = row["y0"]
                        row["y1"] = row["y1"]

            c = c.drop_duplicates().reset_index(drop=True)
            setattr(self, name, c)

        return None

    @staticmethod
    def _assign_rectangle_hierarchy(rectangles: list[dict]) -> list[dict]:
        rects = []
        for i, rect in enumerate(rectangles):
            bbox = Page._convert_rectangle_format(rect)
            bbox["id"] = i
            rects.append(bbox)

        # Initialize containment map: rect_id -> list of contained rect_ids
        contains_map = {r["id"]: [] for r in rects}

        # Populate contains_map
        for r_outer in rects:
            for r_inner in rects:
                if r_outer["id"] != r_inner["id"]:
                    if Page._rectangle_contains(r_outer, r_inner):
                        contains_map[r_outer["id"]].append(r_inner["id"])

        # Initialize all levels as None
        levels = {r["id"]: None for r in rects}

        # Rectangles that contain no other rectangles are level 0
        for r in rects:
            if not contains_map[r["id"]]:
                levels[r["id"]] = 0

        # Iteratively assign levels
        changed = True
        while changed:
            changed = False
            for r in rects:
                if levels[r["id"]] is None:
                    child_levels = [levels[cid] for cid in contains_map[r["id"]]]
                    # Only assign level if all child levels are assigned
                    if None not in child_levels:
                        levels[r["id"]] = max(child_levels) + 1 if child_levels else 0
                        changed = True

        # Collect results
        results = [(rectangles[r["id"]], levels[r["id"]]) for r in rects]
        return results

    @staticmethod
    def _detect_rectangles_from_intersections(
        intersections: list[tuple[float, float]],
    ) -> list[dict]:
        points_by_x = defaultdict(list)
        points_by_y = defaultdict(list)

        for point in intersections:
            x, y = point
            points_by_x[x].append(point)
            points_by_y[y].append(point)

        x_coords = sorted(points_by_x.keys())
        y_coords = sorted(points_by_y.keys())

        rectangles = []
        rectangle_count = 0

        for i in range(len(x_coords)):
            for j in range(i + 1, len(x_coords)):
                x1, x2 = x_coords[i], x_coords[j]

                for k in range(len(y_coords)):
                    for l in range(k + 1, len(y_coords)):
                        y1, y2 = y_coords[k], y_coords[l]

                        corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]

                        if all(corner in intersections for corner in corners):
                            rectangle = {
                                "top_left": (x1, y1),
                                "top_right": (x2, y1),
                                "bottom_left": (x1, y2),
                                "bottom_right": (x2, y2),
                            }
                            rectangles.append(rectangle)
                            rectangle_count += 1

        return rectangles

    @staticmethod
    def _detect_minimal_rectangles(
        intersections: list[tuple[float, float]],
        x_coords: list[float],
        y_coords: list[float],
    ) -> list[dict]:
        """
        Detect minimal rectangles row-wise, handling per-row merges.

        Args:
            intersections: List of (x, y) points.
            x_coords: Sorted unique x-coordinates (global, for reference).
            y_coords: Sorted unique y-coordinates.

        Returns:
            List of rectangle dicts (level 0 per row).
        """
        if len(x_coords) < 2 or len(y_coords) < 2:
            return []

        inter_set = set(intersections)  # For fast lookup
        rectangles = []

        for j in range(len(y_coords) - 1):
            y1, y2 = y_coords[j], y_coords[j + 1]

            # Collect x where BOTH (x, y1) and (x, y2) exist (boundaries for this row)
            row_x_set = set()
            for x, y in intersections:
                if y == y1 and (x, y2) in inter_set:
                    row_x_set.add(x)
                elif y == y2 and (x, y1) in inter_set:
                    row_x_set.add(x)

            if len(row_x_set) < 2:
                continue  # No valid row boundaries

            row_x = sorted(row_x_set)

            # Now, create rects from consecutive row_x (max consecutive spans where boundaries exist)
            for i in range(len(row_x) - 1):
                x1, x2 = row_x[i], row_x[i + 1]
                # By construction, all 4 corners exist
                rect = {
                    "top_left": (x1, y1),
                    "top_right": (x2, y1),
                    "bottom_left": (x1, y2),
                    "bottom_right": (x2, y2),
                }
                rectangles.append(rect)

        return rectangles

    def _detect_rectangles(self) -> list[dict]:
        hor_lines = pd.concat(
            [
                self.lines[self.lines["orientation"] == "horizontal"],
                self.edges[self.edges["orientation"] == "horizontal"],
            ]
        )
        vert_lines = pd.concat(
            [
                self.lines[self.lines["orientation"] == "vertical"],
                self.edges[self.edges["orientation"] == "vertical"],
            ]
        )

        intersections = Page._find_intersections(hor_lines, vert_lines)

        # rectangles = Page._detect_rectangles_from_intersections(intersections)

        # hierarchy = Page._assign_rectangle_hierarchy(rectangles)

        # l0_rectangles = [rect for rect, level in hierarchy if level == 0]

        # return l0_rectangles

        if not intersections:
            return []
        x_coords = sorted(set(x for x,y in intersections))
        y_coords = sorted(set(y for x,y in intersections))
        return Page._detect_minimal_rectangles(intersections, x_coords, y_coords)
    
    def initialise_cells(self, tol: float = 3.0):
        # Check if the rectangles directly need to be used here.
        line_rectangles = self._detect_rectangles()

        for _, rect in enumerate(line_rectangles):
            x0, y0 = rect["top_left"]
            x1, y1 = rect["bottom_right"]

            self.cells.append(Cell(x0, y0, x1, y1, self.page_number))

            for _, char in self.chars.iterrows():
                if (
                    char["x0"] >= x0 - tol
                    and char["x1"] <= x1 + tol
                    and char["y0"] >= y0 - tol
                    and char["y1"] <= y1 + tol
                ):
                    self.cells[-1].text += char["text"]

        self.cells = sorted(self.cells, key=lambda x: (x.x0, -x.y0))

        return self.cells

    def get_cell_groups(self, tol=1.0):
        rows = self._build_cell_groups(self.cells, type="row", tol=tol)
        self.rows = sorted(rows, key=lambda x: -x.y0)

        columns = self._build_cell_groups(self.cells, type="column", tol=tol)
        self.columns = sorted(columns, key=lambda x: x.x0)

        return self.rows, self.columns

    @staticmethod
    def _generate_grid(dimension, resolution=1):
        return [g for g in range(0, int(dimension), resolution)]

    @staticmethod
    def _snap_to_grid(val, grid):
        return max([g for g in grid if g <= val], default=val)

    @staticmethod
    def _detect_orientation(x0, y0, x1, y1, tol=0.0):
        if np.isclose(x0, x1, atol=tol):
            return "vertical"
        elif np.isclose(y0, y1, atol=tol):
            return "horizontal"
        else:
            return "other"

    @staticmethod
    def _attribute_df(
        list_dict: list[dict], mandatory: list[str], selection: list[str]
    ) -> pd.DataFrame:
        """
        Create a DataFrame from a list of dictionaries, ensuring mandatory columns are present.
        """

        if not list_dict:
            # Return empty DataFrame with all mandatory + selection columns as placeholders
            columns = mandatory + selection
            df = pd.DataFrame(columns=columns)
            # Fill selection columns with NaN explicitly (optional here as empty)
            for col in selection:
                df[col] = np.nan
            return df

        df = pd.DataFrame(list_dict)

        for col in mandatory:
            if col not in df.columns:
                raise ValueError(f"Mandatory column '{col}' missing in data.")

        for col in selection:
            if col not in df.columns:
                df[col] = np.nan

        final = mandatory + selection

        return df[final]

    @staticmethod
    def _is_intersection(
        hor_line: dict, vert_line: dict
    ) -> tuple[float, float] | tuple[None, None]:
        h_y = hor_line["y0"]
        h_x0, h_x1 = hor_line["x0"], hor_line["x1"]

        v_x = vert_line["x0"]
        v_y0, v_y1 = vert_line["y0"], vert_line["y1"]

        if (h_x0 <= v_x <= h_x1) and (v_y0 <= h_y <= v_y1):
            return (v_x, h_y)
        else:
            return (None, None)

    @staticmethod
    def _find_intersections(
        hor_lines: pd.DataFrame, vert_lines: pd.DataFrame
    ) -> list[dict]:
        intersections = set()

        for _, hor_line in hor_lines.iterrows():
            for _, vert_line in vert_lines.iterrows():
                intersection = Page._is_intersection(hor_line, vert_line)
                if intersection[0] is not None and intersection[1] is not None:
                    intersections.add(intersection)

        return list(intersections)

    @staticmethod
    def _convert_rectangle_format(rectangle: dict) -> dict:
        min_x, min_y = rectangle["top_left"]
        max_x, max_y = rectangle["bottom_right"]
        return {
            "min_x": min_x,
            "min_y": min_y,
            "max_x": max_x,
            "max_y": max_y,
        }

    @staticmethod
    def _rectangle_contains(r_outer, r_inner) -> bool:
        """Return True if r_outer fully contains r_inner.

        Parameters:
            r_outer: dict with keys 'min_x', 'min_y', 'max_x', 'max_y'
            r_inner: dict with keys 'min_x', 'min_y', 'max_x', 'max_y'

        Returns:
            True if r_outer fully contains r_inner, False otherwise
        """
        return (
            r_outer["min_x"] <= r_inner["min_x"]
            and r_outer["min_y"] <= r_inner["min_y"]
            and r_outer["max_x"] >= r_inner["max_x"]
            and r_outer["max_y"] >= r_inner["max_y"]
        )

    @staticmethod
    def _is_adjacent(cell1: Cell, cell2: Cell, type="row", tol=1.0):
        if type not in ["row", "column"]:
            raise ValueError("Type must be either 'row' or 'column'")

        if type == "row":
            return np.isclose(cell1.x1, cell2.x0, atol=tol)
        elif type == "column":
            return np.isclose(cell1.y1, cell2.y0, atol=tol)
        else:
            raise ValueError("Type must be either 'row' or 'column'")

    @staticmethod
    def _build_cell_groups(cells: list[Cell], type="row", tol=1.0):
        if type not in ["row", "column"]:
            raise ValueError("Type must be either 'row' or 'column'")

        if type == "row":
            coord_col = "y0"
            sort_col = "x0"
        elif type == "column":
            coord_col = "x0"
            sort_col = "y0"
        else:
            raise ValueError("Type must be either 'row' or 'column'")

        cells_sorted = sorted(
            cells, key=lambda c: (getattr(c, coord_col), getattr(c, sort_col))
        )

        groups = []
        current_group = []

        for i, cell in enumerate(cells_sorted):
            if not current_group:
                current_group.append(cell)
                continue

            prev_cell = current_group[-1]

            same_level = (
                abs(getattr(cell, coord_col) - getattr(prev_cell, coord_col)) < tol
            )
            adjacent = same_level and Page._is_adjacent(prev_cell, cell, type=type)

            if same_level and adjacent:
                current_group.append(cell)
            else:
                groups.append(
                    Row(current_group, cell.page_number)
                    if type == "row"
                    else Column(current_group, cell.page_number)
                )
                current_group = [cell]

        if current_group:
            groups.append(
                Row(current_group, cell.page_number)
                if type == "row"
                else Column(current_group, cell.page_number)
            )

        return groups


# %%


class Document:
    def __init__(self, path: str, password: str | None = None) -> None:
        self.path = path
        self.password = password

        try:
            self.pdf = plumber.open(path, password=password)
        except Exception as e:
            raise RuntimeError(
                f"Failed to open PDF {path}. "
                f"{'Password may be incorrect or missing.' if password else ''} "
                f"Error: {e}"
            )

        self.pages = []
        self.cells = []
        self.rows = []
        self.columns = []
        self.tables = []

        self.model = None
        self.header_row = None
        self.table_structure = {}

        self._initialise_pages()

    def _initialise_pages(self):
        for page in self.pdf.pages:
            self.pages.append(Page(page))

    def process_pages(self, pages: list[int] = None):
        doc_cells = []
        doc_rows = []
        doc_columns = []

        if pages is None:
            pages = self.pages
        else:
            pages = [self.pages[i] for i in pages]

        for page in pages:
            _ = page.get_page_characters()
            _ = page.get_page_lines()
            _ = page.get_page_edges()
            _ = page.get_page_rects()

            _ = page.clean_all_coords()
            _ = page.initialise_cells()

            rows, columns = page.get_cell_groups()
            doc_cells.extend(page.cells)
            doc_rows.extend(rows)
            doc_columns.extend(columns)

        self.cells = sorted(doc_cells, key=lambda x: (x.page_number, -x.y0, x.x0))
        self.rows = sorted(doc_rows, key=lambda x: (x.page_number, -x.y0))
        self.columns = sorted(doc_columns, key=lambda x: (x.page_number, x.x0))

        return self.rows, self.columns

    def identify_header_rows(self):
        if self.model is None:
            self.model = SentenceTransformer("all-MiniLM-L6-v2")

        for row in self.rows:
            if self._is_header_row(row, self.model):
                self.header_row = row
                break

        return self.header_row

    def detect_table_structure(self):
        if self.header_row is None:
            self.identify_header_rows()
        print(f"Header Row: {self.header_row}")
        self.table_structure = Document._detect_table_structure(self.header_row)

        return self.table_structure

    def get_tables(self):
        pass

    @staticmethod
    def _detect_table_structure(row: Row):
        num_cells = len(row.cells)
        column_x = [cell.x0 for cell in row.cells]
        column_width = [cell.width for cell in row.cells]

        return {
            "num_cells": num_cells,
            "column_x": column_x,
            "column_width": column_width,
        }

    @staticmethod
    def _is_header_row(row: Row, model=None):
        if model is None:
            model = SentenceTransformer("all-MiniLM-L6-v2")

        row_text = [text.lower() for text in row.text]
        row_embeddings = model.encode(row_text, convert_to_tensor=True)
        similarities = util.pytorch_cos_sim(row_embeddings, HEADER_EMBEDDINGS)
        max_similarity = similarities.max()

        return max_similarity > 0.8

    
    def map_table_to_json(self):
        """
        Single function that converts table data to JSON format.
        Dynamically finds headers and maps data using x-coordinates.
        """
        if self.header_row is None:
            self.identify_header_rows()
        
        if self.header_row is None:
            return []  # Return empty list if no header found

        # Create header mapping with coordinates
        header_columns = []
        for cell in self.header_row.cells:
            text = cell.text.lower().strip()
            text_emb = self.model.encode([text], convert_to_tensor=True)
            scores = util.pytorch_cos_sim(text_emb, HEADER_EMBEDDINGS)[0]
            best_idx = int(torch.argmax(scores))
            best_match = header_words[best_idx]
            
            header_columns.append({
                "name": best_match.upper().replace(".", "").replace("/", "/"),
                "x0": cell.x0,
                "x1": cell.x1
            })
        
        header_columns.sort(key=lambda x: x['x0'])
        
        # Process data rows
        json_data = []
        row_counter = 1
        
        for row in self.rows:
            if row == self.header_row or all(cell.text.strip() == "" for cell in row.cells):
                continue
            
            row_data = {"#": str(row_counter)}
            
            for header_col in header_columns:
                column_name = header_col["name"]
                matching_cell = None
                best_match_score = float('inf')
                
                # Find best matching cell based on x-coordinate
                for data_cell in row.cells:
                    x_distance = abs(data_cell.x0 - header_col["x0"])
                    if x_distance < best_match_score:
                        best_match_score = x_distance
                        matching_cell = data_cell
                
                # Add cell data
                if matching_cell and best_match_score <= 20:  # 20 pixel tolerance
                    cell_text = matching_cell.text.strip()
                    row_data[column_name] = "" if cell_text in ["-", "", "–", "—"] else cell_text
                else:
                    row_data[column_name] = ""
            
            # Only add rows with actual data
            if any(value != "" for key, value in row_data.items() if key != "#"):
                json_data.append(row_data)
                row_counter += 1
        
        return json_data


# %%
if __name__ == "__main__":
    # from skimage.draw import line as skline
    # import matplotlib.pyplot as plt

    # path = "./tests/data/UBI Format 2.pdf"
    path = "./tests/Priority Banks"
    bordered_list = [
        "Axis bank format 2.pdf",
        "Punjab national bank.pdf",
        "SBi format 2.pdf",
        "ICICI Bank format 4.pdf"
    ]

    for file in bordered_list:
        try:
            print(f"Processing {file}...")
            doc = Document(os.path.join(path, file), password="")

            pages = None 
            doc.process_pages(pages=pages)

            doc.identify_header_rows()
            doc.detect_table_structure()
            json_mapping = doc.map_table_to_json()

            stub = file.split(".pdf")[0]

            with open(
        f"./tests/results/doc/{stub}_output_20250911_doc_run1.txt", "w"
            ) as f:
                for i, cell in enumerate(doc.cells):
                    f.write(
                        f"Pg: {cell.page_number}, Cell {i}: {cell}, Text: '{cell.text}'\n"
                    )

            # 🔹 also dump the header mapping JSON here
            json_mapping = doc.map_table_to_json()
            with open(
                f"./tests/results/doc/{stub}_header_mapping.json", "w"
            ) as jf:
                json.dump(json_mapping, jf, indent=4)

            del doc
            gc.collect()
        except Exception as e:
            print(f"Error processing {file}: {e}")
            stub = file.split(".pdf")[0]

            with open(
                f"./tests/results/doc/{stub}_output_20250911_doc_run1.txt", "w"
            ) as f:
                f.write(f"Error processing {file}: {e}\n")

            del doc
            gc.collect()
            continue