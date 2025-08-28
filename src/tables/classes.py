"""
Table classes for processing and analyzing tabular data.
"""

import pdfplumber as plumber
import pandas as pd
import numpy as np

from collections import defaultdict

# %%


class Char:
    def __init__(self, char, x0, y0, x1, y1, height, width):
        self.char = char

        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1

        self.height = height
        self.width = width

        # Neighbor links
        self.left = None
        self.right = None
        self.top = None
        self.bottom = None


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

    def __repr__(self):
        return f"Row(({self.x0}, {self.y0})-({self.x1}, {self.y1}))"


# %%


class Column:
    def __init__(self, cells: list[Cell]) -> None:
        self.cells = cells
        self.length = len(cells)

        self.x0 = min(cell.x0 for cell in cells)
        self.y0 = min(cell.y0 for cell in cells)
        self.x1 = max(cell.x1 for cell in cells)
        self.y1 = max(cell.y1 for cell in cells)

    def __repr__(self):
        return f"Column(({self.x0}, {self.y0})-({self.x1}, {self.y1}))"


# %%


class Table:
    def __init__(self, cells: list[Cell]) -> None:
        self.cells = cells
        self.length = len(cells)

        self.x0 = min(cell.x0 for cell in cells)
        self.y0 = min(cell.y0 for cell in cells)
        self.x1 = max(cell.x1 for cell in cells)
        self.y1 = max(cell.y1 for cell in cells)

    def __repr__(self):
        return f"Table(({self.x0}, {self.y0})-({self.x1}, {self.y1}))"


# %%


class PageVector:
    def __init__(self, page_width, page_height, resolution=1):
        self.page_width = page_width
        self.page_height = page_height
        self.resolution = resolution  # grid cell size in points, default 1

        self.grid = self._init_grid()

    def __repr__(self):
        return f"PageVector(({self.page_width}, {self.page_height})-{self.resolution})"

    def _init_grid(self):
        width = int(np.ceil(self.page_width / self.resolution))
        height = int(np.ceil(self.page_height / self.resolution))
        grid = np.empty((height, width), dtype=object)
        grid[:] = None

        return grid


# %%


class Page:
    def __init__(self, pdf_page: plumber.page.Page) -> None:  # pyright: ignore[reportAttributeAccessIssue]
        self.page = pdf_page

        self.page_number = self.page.page_number
        self.height = self.page.height
        self.width = self.page.width

        self.page_vector = PageVector(self.width, self.height, resolution=1)

        self.chars = self.page.chars
        self.lines = self.page.lines
        self.rects = self.page.rects

        self.char_df = Page._attribute_df(
            self.chars,
            mandatory=["text", "x0", "y0", "x1", "y1"],
            selection=["size", "width", "height"],
        )

        self.line_df = Page._attribute_df(
            self.lines,
            mandatory=["x0", "y0", "x1", "y1"],
            selection=["linewidth", "orientation"],
        )

        if not self.line_df.empty:
            y0 = pd.to_numeric(self.line_df["y0"], errors="coerce")
            y1 = pd.to_numeric(self.line_df["y1"], errors="coerce")
            self.line_df["orientation"] = np.where(
                np.isclose(y0, y1), "horizontal", "vertical"
            )
        else:
            self.line_df["orientation"] = pd.Series(dtype="object")

        self.rect_df = Page._attribute_df(
            self.rects,
            mandatory=["x0", "y0", "x1", "y1"],
            selection=["linewidth"],
        )

        self.all_lines_df = pd.DataFrame()

        self.cells = []

        self.rows = []
        self.columns = []
        self.tables = []

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

    def _rect_to_lines(self) -> pd.DataFrame:
        """
        Convert rectangles into the component lines.
        """
        lines = []

        for _, rect in self.rect_df.iterrows():
            x0, y0, x1, y1, linewidth = (
                rect["x0"],
                rect["y0"],
                rect["x1"],
                rect["y1"],
                rect["linewidth"],
            )

            # Top Edge
            lines.append(
                {
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y0,
                    "linewidth": linewidth,
                }
            )

            # Right Edge
            lines.append(
                {
                    "x0": x1,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                    "linewidth": linewidth,
                }
            )

            # Bottom Edge
            lines.append(
                {
                    "x0": x0,
                    "y0": y1,
                    "x1": x1,
                    "y1": y1,
                    "linewidth": linewidth,
                }
            )

            # Left Edge
            lines.append(
                {
                    "x0": x0,
                    "y0": y0,
                    "x1": x0,
                    "y1": y1,
                    "linewidth": linewidth,
                }
            )

        return pd.DataFrame(lines)

    @staticmethod
    def _merge_intervals(intervals, threshold: float = 2.0) -> list:
        """Merge overlapping or nearly overlapping intervals."""
        if not intervals:
            return []
        # Sort intervals by start
        intervals = sorted(intervals, key=lambda x: x[0])
        merged = [intervals[0]]
        for current in intervals[1:]:
            prev = merged[-1]
            # If overlapping or within threshold, merge
            if current[0] <= prev[1] + threshold:
                merged[-1] = (prev[0], max(prev[1], current[1]))
            else:
                merged.append(current)
        return merged

    @staticmethod
    def _cluster_horizontal_lines(
        df_lines: pd.DataFrame, threshold: float = 3.0
    ) -> pd.DataFrame:
        """
        Cluster horizontal lines that are close to each other into groups.
        """
        lines = df_lines[np.isclose(df_lines["y0"], df_lines["y1"])].copy()

        clustered = []

        grouping = ["y0", "linewidth"]

        for group_key, group_df in lines.groupby(grouping):
            # If the group has more than one line, we consider it a cluster
            y0 = group_key[0]
            lw = group_key[1]

            intervals = [(x0, x1) for x0, x1 in zip(group_df["x0"], group_df["x1"])]

            merged_intervals = Page._merge_intervals(intervals, threshold=threshold)

            for start, end in merged_intervals:
                clustered.append(
                    {
                        "x0": start,
                        "y0": y0,
                        "x1": end,
                        "y1": y0,
                        "linewidth": lw,
                    }
                )

        return pd.DataFrame(clustered)

    @staticmethod
    def _cluster_vertical_lines(
        df_lines: pd.DataFrame, threshold: float = 3.0
    ) -> pd.DataFrame:
        """
        Cluster vertical lines that are close to each other into groups.
        """
        lines = df_lines[np.isclose(df_lines["x0"], df_lines["x1"])].copy()

        clustered = []

        grouping = ["x0", "linewidth"]

        for group_key, group_df in lines.groupby(grouping):
            # If the group has more than one line, we consider it a cluster
            x0 = group_key[0]
            lw = group_key[1]

            intervals = [(y0, y1) for y0, y1 in zip(group_df["y0"], group_df["y1"])]

            merged_intervals = Page._merge_intervals(intervals, threshold=threshold)

            for start, end in merged_intervals:
                clustered.append(
                    {
                        "x0": x0,
                        "y0": start,
                        "x1": x0,
                        "y1": end,
                        "linewidth": lw,
                    }
                )

        return pd.DataFrame(clustered)

    @staticmethod
    def _squash_lines(
        array: np.ndarray, threshold: float
    ) -> tuple[list[np.float64], dict]:
        # array: sorted list of y (or x); returns new canonical values with mapping

        positions = sorted(array)
        squashed = []
        mapping = {}
        group = [positions[0]]

        for pos in positions[1:]:
            if abs(pos - group[-1]) <= threshold:
                group.append(pos)
            else:
                target = np.mean(group)
                for g in group:
                    mapping[g] = target
                squashed.append(target)
                group = [pos]

        # last group
        target = np.mean(group)
        for g in group:
            mapping[g] = target
        squashed.append(target)

        return squashed, mapping

    def get_all_lines(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get all lines from the detected rectangles and lines.
        """
        if self.rect_df.empty:
            # No rectangles → no lines from rects
            rect_lines = pd.DataFrame(columns=["x0", "y0", "x1", "y1", "linewidth"])
        else:
            rect_lines = self._rect_to_lines()

        if rect_lines.empty:
            # Return empty hor_lines and ver_lines with expected columns
            hor_lines = pd.DataFrame(columns=["x0", "y0", "x1", "y1", "linewidth"])
            ver_lines = pd.DataFrame(columns=["x0", "y0", "x1", "y1", "linewidth"])
            return hor_lines, ver_lines

        hor_lines = self._cluster_horizontal_lines(rect_lines)
        ver_lines = self._cluster_vertical_lines(rect_lines)

        return hor_lines, ver_lines

    def get_clustered_lines(self, threshold: float = 3.0) -> pd.DataFrame:
        hor_lines, ver_lines = self.get_all_lines()

        if not hor_lines.empty:
            ys = hor_lines.y0.unique()
            _, y_mapping = Page._squash_lines(ys, threshold=threshold)
            hor_lines["y0"] = hor_lines["y0"].map(y_mapping)
            hor_lines["y1"] = hor_lines["y1"].map(y_mapping)

            clustered_horizontal = Page._cluster_horizontal_lines(
                hor_lines, threshold=threshold
            )
            clustered_horizontal["orientation"] = "horizontal"
        else:
            clustered_horizontal = pd.DataFrame(
                columns=["x0", "y0", "x1", "y1", "linewidth", "orientation"]
            )

        if not ver_lines.empty:
            xs = ver_lines.x0.unique()
            _, x_mapping = Page._squash_lines(xs, threshold=threshold)
            ver_lines["x0"] = ver_lines["x0"].map(x_mapping)
            ver_lines["x1"] = ver_lines["x1"].map(x_mapping)

            clustered_vertical = Page._cluster_vertical_lines(
                ver_lines, threshold=threshold
            )
            clustered_vertical["orientation"] = "vertical"
        else:
            clustered_vertical = pd.DataFrame(
                columns=["x0", "y0", "x1", "y1", "linewidth", "orientation"]
            )

        self.all_lines_df = (
            pd.concat([self.line_df, clustered_horizontal, clustered_vertical])
            .drop_duplicates()
            .reset_index(drop=True)
        )

        return self.all_lines_df

    def _hash_lines_as_spans(self):
        # For horizontals: (y, x_start, x_end)
        horiz_lines = [
            (row.y0, min(row.x0, row.x1), max(row.x0, row.x1))
            for _, row in self.all_lines_df[
                self.all_lines_df.orientation == "horizontal"
            ].iterrows()
        ]
        # For verticals: (x, y_start, y_end)
        vert_lines = [
            (row.x0, min(row.y0, row.y1), max(row.y0, row.y1))
            for _, row in self.all_lines_df[
                self.all_lines_df.orientation == "vertical"
            ].iterrows()
        ]

        return horiz_lines, vert_lines

    @staticmethod
    def _assign_cell_text(char_df: pd.DataFrame, cell: Cell, tol: float = 2.0) -> str:
        chars_in_cell = char_df[
            (char_df["x0"] >= cell.x0 - tol)
            & (char_df["x1"] <= cell.x1 + tol)
            & (char_df["y0"] >= cell.y0 - tol)
            & (char_df["y1"] <= cell.y1 + tol)
        ].copy()

        chars_in_cell.sort_values(
            by=["y0", "x0"], ascending=[False, True], inplace=True
        )  # Sort top-to-bottom, left-to-right

        text = "".join(chars_in_cell["text"].tolist()).strip()
        cell.text = text

        return text

    def initialise_cells(self) -> list[Cell]:
        horiz_lines, vert_lines = self._hash_lines_as_spans()

        x_vals = sorted(set(self.all_lines_df["x0"]).union(self.all_lines_df["x1"]))
        y_vals = sorted(set(self.all_lines_df["y0"]).union(self.all_lines_df["y1"]))

        cells = []
        for i in range(len(x_vals) - 1):
            for j in range(len(y_vals) - 1):
                x0, x1 = x_vals[i], x_vals[i + 1]
                y0, y1 = y_vals[j], y_vals[j + 1]

                # For top: horizontal line at y0 that covers [x0, x1]
                top = any(
                    y == y0 and x_start <= x0 and x_end >= x1
                    for y, x_start, x_end in horiz_lines
                )
                # For bottom: horizontal at y1
                bottom = any(
                    y == y1 and x_start <= x0 and x_end >= x1
                    for y, x_start, x_end in horiz_lines
                )
                # For left: vertical at x0 that covers [y0, y1]
                left = any(
                    x == x0 and y_start <= y0 and y_end >= y1
                    for x, y_start, y_end in vert_lines
                )
                # For right: vertical at x1
                right = any(
                    x == x1 and y_start <= y0 and y_end >= y1
                    for x, y_start, y_end in vert_lines
                )

                if top and bottom and left and right:
                    cell = Cell(x0, y0, x1, y1)
                    Page._assign_cell_text(self.char_df, cell)
                    cells.append(cell)

        self.cells = cells

        return cells

    @staticmethod
    def bin_edge(val: float, tol: float = 1.0) -> float:
        return round(val / tol) * tol

    @staticmethod
    def overlap_1d(a0, a1, b0, b1, tol):
        # Check if 1D intervals [a0,a1] and [b0,b1] overlap or nearly overlap
        return not (a1 + tol < b0 or b1 + tol < a0)

    @staticmethod
    def strict_linking_1d(a0, a1, b0, b1, tol):
        return abs(a0 - b0) < tol and abs(a1 - b1) < tol

    def group_cells(
        self, tol: float = 2.0
    ) -> tuple[defaultdict, defaultdict, defaultdict, defaultdict]:
        """
        Group cells based on their edges.
        """
        left_groups = defaultdict(list)
        right_groups = defaultdict(list)
        top_groups = defaultdict(list)
        bottom_groups = defaultdict(list)

        for cell in self.cells:
            left_groups[Page.bin_edge(cell.x0, tol)].append(cell)
            right_groups[Page.bin_edge(cell.x1, tol)].append(cell)
            top_groups[Page.bin_edge(cell.y0, tol)].append(cell)
            bottom_groups[Page.bin_edge(cell.y1, tol)].append(cell)

        return left_groups, right_groups, top_groups, bottom_groups

    def link_cells(self, tol: float = 2.0) -> None:
        left_groups, right_groups, top_groups, bottom_groups = self.group_cells()

        for cell in self.cells:
            x0_bin = Page.bin_edge(cell.x0, tol)
            x1_bin = Page.bin_edge(cell.x1, tol)
            y0_bin = Page.bin_edge(cell.y0, tol)
            y1_bin = Page.bin_edge(cell.y1, tol)

            for left_neighbor in right_groups[x0_bin]:
                if Page.strict_linking_1d(
                    cell.y0, cell.y1, left_neighbor.y0, left_neighbor.y1, tol
                ):
                    cell.left = left_neighbor
                    left_neighbor.right = cell

            for right_neighbor in left_groups[x1_bin]:
                if Page.strict_linking_1d(
                    cell.y0, cell.y1, right_neighbor.y0, right_neighbor.y1, tol
                ):
                    cell.right = right_neighbor
                    right_neighbor.left = cell

            for top_neighbor in bottom_groups[y0_bin]:
                if Page.strict_linking_1d(
                    cell.x0, cell.x1, top_neighbor.x0, top_neighbor.x1, tol
                ):
                    cell.top = top_neighbor
                    top_neighbor.bottom = cell

            for bottom_neighbor in top_groups[y1_bin]:
                if Page.strict_linking_1d(
                    cell.x0, cell.x1, bottom_neighbor.x0, bottom_neighbor.x1, tol
                ):
                    cell.bottom = bottom_neighbor
                    bottom_neighbor.top = cell

        return None

    @staticmethod
    def depth_first_search(start_cell, visited, direction="row"):
        directions = {
            "row": ["left", "right"],
            "column": ["top", "bottom"],
            "table": ["left", "right", "top", "bottom"],
        }

        stack = [start_cell]
        connected_cells = []

        while stack:
            cell = stack.pop()
            if cell not in visited:
                visited.add(cell)
                connected_cells.append(cell)

                for dir in directions[direction]:
                    neighbor = getattr(cell, dir)
                    if neighbor is not None and neighbor not in visited:
                        stack.append(neighbor)

        return connected_cells

    def get_rows(self):
        visited = set()
        rows = []
        for cell in self.cells:
            if cell not in visited:
                row = Page.depth_first_search(cell, visited, direction="row")
                rows.append(row)

        if len(rows) > 0:
            rows.sort(key=lambda r: min(c.x0 for c in r))

            self.rows = [Row(cells) for cells in rows]

        return rows

    def get_columns(self):
        visited = set()
        columns = []
        for cell in self.cells:
            if cell not in visited:
                column = Page.depth_first_search(cell, visited, direction="column")
                columns.append(column)

        if len(columns) > 0:
            columns.sort(key=lambda c: min(c.y0 for c in c))

            self.columns = [Column(cells) for cells in columns]

        return columns

    def get_tables(self):
        visited = set()
        tables = []

        for cell in self.cells:
            if cell not in visited:
                cluster = Page.depth_first_search(cell, visited, direction="table")

                if len(cluster) > 1:  # ignore stray cells
                    tables.append(cluster)

        if len(tables) > 0:
            # Create Table objects similarly to Row and Column
            self.tables = [Table(cells) for cells in tables]

        return tables


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
    # path = "./tests/data/UBI Format 2.pdf"
    path = "../../tests/data/ICICI Bank.pdf"
    doc = Document(path)
    page_0 = doc.pages[0]

    hor_lines, ver_lines = page_0.get_all_lines()
    lines = page_0.get_clustered_lines()
    cells = page_0.initialise_cells()
    print(cells)

    page_0.link_cells()
    rows = page_0.get_rows()
    columns = page_0.get_columns()
    tables = page_0.get_tables()
