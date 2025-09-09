# %%

import pdfplumber as plumber
import pandas as pd
import numpy as np

from sklearn.cluster import DBSCAN
from collections import defaultdict

# %%


# class Rectangle:
#     def __init__(self, top_left, top_right, bottom_left, bottom_right):
#         self.top_left = top_left
#         self.top_right = top_right
#         self.bottom_left = bottom_left
#         self.bottom_right = bottom_right

#         self.x0, self.y0 = top_left
#         self.x1, self.y1 = bottom_right


# %%


class Cell:
    def __init__(self, x0, y0, x1, y1):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1

        self.height = abs(self.y1 - self.y0)
        self.width = abs(self.x1 - self.x0)

        self.top = None
        self.bottom = None
        self.left = None
        self.right = None

        self.text = ""

    def __repr__(self):
        return f"Cell(({self.x0}, {self.y0})-({self.x1}, {self.y1}))"


# %%
class Row:
    def __init__(self, cells: list[Cell]) -> None:
        self.cells = cells
        self.length = len(cells)

        self.x0 = min(cell.x0 for cell in cells)
        self.y0 = min(cell.y0 for cell in cells)
        self.x1 = max(cell.x1 for cell in cells)
        self.y1 = max(cell.y1 for cell in cells)

        self.text = [cell.text for cell in cells]

    def __repr__(self):
        return f"Row(({self.x0}, {self.y0})-({self.x1}, {self.y1}))"


class Column:
    def __init__(self, cells: list[Cell]) -> None:
        self.cells = cells
        self.length = len(cells)

        self.x0 = min(cell.x0 for cell in cells)
        self.y0 = min(cell.y0 for cell in cells)
        self.x1 = max(cell.x1 for cell in cells)
        self.y1 = max(cell.y1 for cell in cells)

        self.text = [cell.text for cell in cells]

    def __repr__(self):
        return f"Column(({self.x0}, {self.y0})-({self.x1}, {self.y1}))"


# %%
class Table:
    def __init__(self, rows: list[Row], columns: list[Column]) -> None:
        self.rows = rows
        self.columns = columns

    def __repr__(self):
        return f"Table(({len(self.rows)}, {len(self.columns)}))"


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

            # for col in ["x0", "y0", "x1", "y1"]:
            #     char_df[col] = char_df[col].apply(
            #         lambda x: Page._snap_to_grid(
            #             x, self.column_grid if col in ["x0", "x1"] else self.row_grid
            #         )
            #     )

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

            # for col in ["x0", "y0", "x1", "y1"]:
            #     lines_df[col] = lines_df[col].apply(
            #         lambda x: Page._snap_to_grid(
            #             x, self.column_grid if col in ["x0", "x1"] else self.row_grid
            #         )
            #     )

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

            # coordinate_groups = [["x0", "y0"], ["x1", "y1"]]

            # for orientation in ["horizontal", "vertical"]:
            #     subset = edges_df[edges_df["orientation"] == orientation].copy()
            #     indices = list(subset.index)

            #     if not subset.empty:
            #         for group in coordinate_groups:
            #             points = np.vstack((subset[group].values))

            #             clustered = Page._cluster_points(
            #                 points,
            #                 self.column_grid,
            #                 self.row_grid,
            #                 tol=tol,
            #                 original_indices=indices,
            #             )

            #             for col in clustered.columns:
            #                 subset_col = group[0] if col == "x" else group[1]
            #                 subset.loc[indices, subset_col] = clustered[col].values

            #     edges_df.loc[indices, group] = subset[group].values

            # for col in ["x0", "y0", "x1", "y1"]:
            #     edges_df[col] = edges_df[col].apply(
            #         lambda x: Page._snap_to_grid(
            #             x, self.column_grid if col in ["x0", "x1"] else self.row_grid
            #         )
            #     )

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

            # for col in ["x0", "y0", "x1", "y1"]:
            #     rects_df[col] = rects_df[col].apply(
            #         lambda x: Page._snap_to_grid(
            #             x, self.column_grid if col in ["x0", "x1"] else self.row_grid
            #         )
            #     )

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
        rectangles = Page._detect_rectangles_from_intersections(intersections)

        hierarchy = Page._assign_rectangle_hierarchy(rectangles)

        l0_rectangles = [rect for rect, level in hierarchy if level == 0]

        return l0_rectangles

    def _initialise_cells(self, tol: float = 3.0):
        # for _, rect in self.rects.iterrows():
        #     self.cells.append(Cell(rect["x0"], rect["y0"], rect["x1"], rect["y1"]))
        #     for _, char in self.chars.iterrows():
        #         if (
        #             char["x0"] >= rect["x0"] - tol
        #             and char["x1"] <= rect["x1"] + tol
        #             and char["y0"] >= rect["y0"] - tol
        #             and char["y1"] <= rect["y1"] + tol
        #         ):
        #             self.cells[-1].text += char["text"]

        line_rectangles = self._detect_rectangles()

        for _, rect in enumerate(line_rectangles):
            x0, y0 = rect["top_left"]
            x1, y1 = rect["bottom_right"]

            self.cells.append(Cell(x0, y0, x1, y1))

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
        self.rows = self._build_cell_groups(self.cells, type="row", tol=tol)
        self.rows = sorted(self.rows, key=lambda x: -x.y0)

        self.columns = self._build_cell_groups(self.cells, type="column", tol=tol)
        self.columns = sorted(self.columns, key=lambda x: x.x0)

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
                    Row(current_group) if type == "row" else Column(current_group)
                )
                current_group = [cell]

        if current_group:
            groups.append(
                Row(current_group) if type == "row" else Column(current_group)
            )

        return groups


# %%


class Document:
    def __init__(self, path: str) -> None:
        self.path = path
        self.pdf = plumber.open(
            path
        )  # TODO: Change to an encapsulated function that handles PWD protection

        self.pages = []
        self._initialise_pages()

    def _initialise_pages(self):
        for page in self.pdf.pages:
            self.pages.append(Page(page))


# %%
if __name__ == "__main__":
    # from skimage.draw import line as skline
    # import matplotlib.pyplot as plt

    import os
    import gc

    # path = "./tests/data/UBI Format 2.pdf"
    path = "./tests/data/"
    bordered_list = [
        "AKOLA JANATA COMMERCIAL COOPERATIVE BANK Statement_For_193775_012103301000485.pdf",
        "Al Ilmna.pdf",
        "Axis bank format 2.pdf",
        "Axis bank format 3.pdf",
        "Axis bank format 4.pdf",
        "Axis bank format 5.pdf",
        "Axis bank format 6.pdf",
        "AXIS Bank.pdf",
        "Bhuj Commercial Co-operative Bank Ltd.PDF",
        "Canara Bank format 4.pdf",
        "Canara bank Format 6.pdf",
        "Central Bank of India.pdf",
        "CITIZENCREDIT Co-operative Bank Ltd.pdf",
        "DBS.pdf",
        "DEOGIRI NAGARI SAHAKARI BANK LTD.pdf",
        # "FEDERAL BANK.pdf",
        "ICICI Bank format 4.pdf",
        "ICICI Bank format 5.pdf",
        "ICICI Bank.pdf",
        "IDBI bank format 2.pdf",
        "idbi bank format 3.pdf",
        "IDFC format 2.pdf",
        "IDFC_Ashish.pdf",
        "Kalpana Awade Bank.pdf",
        "Maharasta gramin bank.pdf",
        "Omprakash Deora Peoples Co-Operative Bank Ltd..pdf",
        "Punjab national bank.pdf",
        "Punjab Nationl Bank format 2.pdf",
        "Sawji Bank.pdf",
        "SBi format 2.pdf",
        "SBI format 3.pdf",
        "SBI format 4.pdf",
        "SBI Format 5.pdf",
        "SBI format 6.pdf",
        "SBI format 7.pdf",
        "SBI format 8.pdf",
        "SBI format 9.pdf",
        "UBI Format 2.pdf",
    ]

    for file in bordered_list:
        try:
            print(f"Processing {file}...")
            doc = Document(os.path.join(path, file))
            page = doc.pages[1]

            _ = page.get_page_characters()
            _ = page.get_page_lines()
            _ = page.get_page_edges()
            _ = page.get_page_rects()

            page.clean_all_coords()
            cells = page._initialise_cells()
            print(f"Cells initialized for {file}: {len(cells)}")

            stub = file.split(".pdf")[0]

            with open(
                f"./tests/results/page 2/{stub}_output_20250908_page2_run1.txt", "w"
            ) as f:
                for i, cell in enumerate(cells):
                    f.write(f"Cell {i}: {cell}, Text: '{cell.text}'\n")

            del doc
            gc.collect()
        except Exception as e:
            print(f"Error processing {file}: {e}")
            stub = file.split(".pdf")[0]

            with open(f"./tests/results/{stub}_output_20250908_run1.txt", "w") as f:
                f.write(f"Error processing {file}: {e}\n")

            del doc
            gc.collect()
            continue

    # def pdf_to_array_coords(x, y, height):
    #     # y-axis flip because PDF origin (0,0) is bottom-left
    #     row = height - int(round(y))
    #     col = int(round(x))
    #     return row, col

    # height = int(page.height)
    # width = int(page.width)

    # # Create a blank canvas
    # canvas = np.zeros((height, width), dtype=np.uint8)

    # for cell in cells:
    #     x0, y0 = cell.x0, cell.y0
    #     x1, y1 = cell.x1, cell.y1

    #     # Convert PDF coords to array indices (row, col)
    #     r0, c0 = pdf_to_array_coords(x0, y0, height)
    #     r1, c1 = pdf_to_array_coords(x1, y1, height)

    #     # Ensure ordered indices
    #     r_min, r_max = min(r0, r1), max(r0, r1)
    #     c_min, c_max = min(c0, c1), max(c0, c1)

    #     r_min_clamped = np.clip(r_min, 0, height - 1)
    #     r_max_clamped = np.clip(r_max, 0, height - 1)
    #     c_min_clamped = np.clip(c_min, 0, width - 1)
    #     c_max_clamped = np.clip(c_max, 0, width - 1)

    #     # Draw top edge
    #     rr, cc = skline(r_min_clamped, c_min_clamped, r_min_clamped, c_max_clamped)
    #     canvas[rr, cc] = 1

    #     # Draw bottom edge
    #     rr, cc = skline(r_max_clamped, c_min_clamped, r_max_clamped, c_max_clamped)
    #     canvas[rr, cc] = 1

    #     # Draw left edge
    #     rr, cc = skline(r_min_clamped, c_min_clamped, r_max_clamped, c_min_clamped)
    #     canvas[rr, cc] = 1

    #     # Draw right edge
    #     rr, cc = skline(r_min_clamped, c_max_clamped, r_max_clamped, c_max_clamped)
    #     canvas[rr, cc] = 1

    # plt.imshow(canvas, cmap="gray")
    # plt.show()
