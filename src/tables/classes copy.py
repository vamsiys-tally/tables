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

        self.text = [text for cell in cells if (text := cell.text)]

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

        self.text = [text for cell in cells if (text := cell.text)]

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


# class PageVector:
#     def __init__(self, page_width, page_height, resolution=1):
#         self.page_width = page_width
#         self.page_height = page_height
#         self.resolution = resolution  # grid cell size in points, default 1

#         self.grid = self._init_grid()

#     def __repr__(self):
#         return f"PageVector(({self.page_width}, {self.page_height})-{self.resolution})"

#     def _init_grid(self):
#         width = int(np.ceil(self.page_width / self.resolution))
#         height = int(np.ceil(self.page_height / self.resolution))
#         grid = np.empty((height, width), dtype=object)
#         grid[:] = None

#         return grid


# %%


class Page:
    def __init__(self, pdf_page: plumber.page.Page) -> None:  # pyright: ignore[reportAttributeAccessIssue]
        self.page = pdf_page

        self.page_number = self.page.page_number
        self.height = self.page.height
        self.width = self.page.width

        # self.page_vector = PageVector(self.width, self.height, resolution=1)

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
            selection=["orientation"],
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
            selection=[],
        )

        self.grid = (Page._generate_grid(self.width), Page._generate_grid(self.height))
        self.all_lines_df = pd.DataFrame(
            columns=["x0", "y0", "x1", "y1", "orientation"]
        )

        self.cells = []

        self.rows = []
        self.columns = []
        self.tables = []

    # Static Methods

    @staticmethod
    def _generate_grid(dimension, resolution=1):
        return [g for g in range(0, int(dimension), resolution)]

    @staticmethod
    def _snap_to_grid(val, grid):
        return max([g for g in grid if g <= val], default=val)

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
    def _merge_intervals(intervals, tol: float = 3.0) -> list:
        """Merge overlapping or nearly overlapping intervals."""
        if not intervals:
            return []
        # Sort intervals by start
        intervals = sorted(intervals, key=lambda x: x[0])
        merged = [intervals[0]]
        for current in intervals[1:]:
            prev = merged[-1]
            # If overlapping or within threshold, merge
            if current[0] <= prev[1] + tol:
                merged[-1] = (prev[0], max(prev[1], current[1]))
            else:
                merged.append(current)
        return merged

    @staticmethod
    def _merge_horizontal_lines(
        df_lines: pd.DataFrame, tol: float = 3.0
    ) -> pd.DataFrame:
        """
        Merge horizontal lines that are close to each other into groups.
        """
        lines = df_lines[np.isclose(df_lines["y0"], df_lines["y1"])].copy()
        clustered = []
        grouping = ["y0"]
        for group_key, group_df in lines.groupby(grouping):
            # If the group has more than one line, we consider it a cluster
            y0 = group_key[0]
            intervals = [(x0, x1) for x0, x1 in zip(group_df["x0"], group_df["x1"])]
            merged_intervals = Page._merge_intervals(intervals, tol=tol)
            for start, end in merged_intervals:
                clustered.append(
                    {
                        "x0": start,
                        "y0": y0,
                        "x1": end,
                        "y1": y0,
                    }
                )
        return pd.DataFrame(clustered)

    @staticmethod
    def _merge_vertical_lines(df_lines: pd.DataFrame, tol: float = 3.0) -> pd.DataFrame:
        """
        Merge vertical lines that are close to each other into groups.
        """
        lines = df_lines[np.isclose(df_lines["x0"], df_lines["x1"])].copy()
        clustered = []
        grouping = ["x0"]
        for group_key, group_df in lines.groupby(grouping):
            # If the group has more than one line, we consider it a cluster
            x0 = group_key[0]
            intervals = [(y0, y1) for y0, y1 in zip(group_df["y0"], group_df["y1"])]
            merged_intervals = Page._merge_intervals(intervals, tol=tol)
            for start, end in merged_intervals:
                clustered.append(
                    {
                        "x0": x0,
                        "y0": start,
                        "x1": x0,
                        "y1": end,
                    }
                )
        return pd.DataFrame(clustered)

    @staticmethod
    def _cluster_horizontal_lines(
        df_lines: pd.DataFrame, tol: float = 3.0, max_span: float = 6.0
    ) -> pd.DataFrame:
        """
        Cluster horizontal lines that are close to each other into groups.
        """
        lines = df_lines[np.isclose(df_lines["y0"], df_lines["y1"])].copy()

        clustered = []
        unique_ys = lines["y0"].unique()
        groups = []
        current_group = [unique_ys[0]]

        for y in sorted(unique_ys[1:]):
            if abs(y - current_group[-1]) <= tol:
                current_group.append(y)
            else:
                groups.append(current_group)
                current_group = [y]

        groups.append(current_group)

        for group_ys in groups:
            group_df = lines[lines["y0"].isin(group_ys)].copy()

            group_y_min, group_y_max = min(group_ys), max(group_ys)
            if group_y_max - group_y_min > max_span:
                pass

            intervals = [(x0, x1) for x0, x1 in zip(group_df["x0"], group_df["x1"])]
            merged_intervals = Page._merge_intervals(intervals, tol=tol)
            y_mean = np.mean(group_ys)
            for start, end in merged_intervals:
                clustered.append({"x0": start, "y0": y_mean, "x1": end, "y1": y_mean})

        return pd.DataFrame(clustered)

    @staticmethod
    def _cluster_vertical_lines(
        df_lines: pd.DataFrame, tol: float = 3.0, max_span: float = 6.0
    ) -> pd.DataFrame:
        """
        Cluster vertical lines that are close to each other into groups.
        """
        lines = df_lines[np.isclose(df_lines["x0"], df_lines["x1"])].copy()

        clustered = []
        unique_xs = lines["x0"].unique()
        groups = []
        current_group = [unique_xs[0]]

        for x in sorted(unique_xs[1:]):
            if abs(x - current_group[-1]) <= tol:
                current_group.append(x)
            else:
                groups.append(current_group)
                current_group = [x]

        groups.append(current_group)

        for group_xs in groups:
            group_df = lines[lines["x0"].isin(group_xs)].copy()

            group_x_min, group_x_max = min(group_xs), max(group_xs)
            if group_x_max - group_x_min > max_span:
                pass

            intervals = [(y0, y1) for y0, y1 in zip(group_df["y0"], group_df["y1"])]
            merged_intervals = Page._merge_intervals(intervals, tol=tol)
            x_mean = np.mean(group_xs)
            for start, end in merged_intervals:
                clustered.append({"x0": x_mean, "y0": start, "x1": x_mean, "y1": end})

        return pd.DataFrame(clustered)

    @staticmethod
    def _squash_lines(
        array: np.ndarray, tol: float = 3.0
    ) -> tuple[list[np.float64], dict]:
        # array: sorted list of y (or x); returns new canonical values with mapping

        positions = sorted(array)
        squashed = []
        mapping = {}
        group = [positions[0]]

        for pos in positions[1:]:
            if abs(pos - group[-1]) <= tol:
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

    @staticmethod
    def _extend_lines(
        coords: tuple[float, float, float, float], extension: float = 2.0
    ) -> tuple[float, float, float, float]:
        if coords:
            x0, y0, x1, y1 = coords

            if np.isclose(y0, y1):
                return (x0 - extension, y0, x1 + extension, y1)
            elif np.isclose(x0, x1):
                return (x0, y0 - extension, x1, y1 + extension)

        return (0.0, 0.0, 0.0, 0.0)

    # TODO 1. Snap chars to grid
    @staticmethod
    def _assign_cell_text(
        char_df: pd.DataFrame, cell: Cell, tol: float = 3.0
    ) -> tuple[str, list[str]]:
        chars_in_cell = char_df[
            (char_df["x0"] >= cell.x0 - tol)
            & (char_df["x1"] <= cell.x1 + tol)
            & (char_df["y0"] >= cell.y0 - tol)
            & (char_df["y1"] <= cell.y1 + tol)
        ].copy()

        chars_in_cell.sort_values(
            by=["y0", "x0"], ascending=[False, True], inplace=True
        )  # Sort top-to-bottom, left-to-right

        char_list = chars_in_cell["text"].tolist()
        text = "".join(char_list).strip()
        cell.text = text

        return text, char_list

    @staticmethod
    def bin_edge(val: float, tol: float = 3.0) -> float:
        return round(val / tol) * tol

    @staticmethod
    def overlap_1d(a0, a1, b0, b1, tol):
        # Check if 1D intervals [a0,a1] and [b0,b1] overlap or nearly overlap
        return not (a1 + tol < b0 or b1 + tol < a0)

    @staticmethod
    def strict_linking_1d(a0, a1, b0, b1, tol):
        return abs(a0 - b0) < tol and abs(a1 - b1) < tol

    @staticmethod
    def _shared_boundary_length(cell1: Cell, cell2: Cell) -> float:
        if abs(cell1.x1 - cell2.x0) < 1e-6:  # cell1 right adjacent to cell2 left
            overlap = max(0, min(cell1.y1, cell2.y1) - max(cell1.y0, cell2.y0))
            return overlap

        if abs(cell1.x0 - cell2.x1) < 1e-6:  # cell1 left adjacent to cell2 right
            overlap = max(0, min(cell1.y1, cell2.y1) - max(cell1.y0, cell2.y0))
            return overlap
        # if they share a horizontal boundary (top-bottom)

        if abs(cell1.y1 - cell2.y0) < 1e-6:  # cell1 bottom adjacent to cell2 top
            overlap = max(0, min(cell1.x1, cell2.x1) - max(cell1.x0, cell2.x0))
            return overlap

        if abs(cell1.y0 - cell2.y1) < 1e-6:  # cell1 top adjacent to cell2 bottom
            overlap = max(0, min(cell1.x1, cell2.x1) - max(cell1.x0, cell2.x0))
            return overlap

        return 0.0

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

    # Object Methods

    def _rect_to_lines(self) -> pd.DataFrame:
        """
        Convert rectangles into the component lines.
        """
        lines = []

        for _, rect in self.rect_df.iterrows():
            x0, y0, x1, y1 = (
                rect["x0"],
                rect["y0"],
                rect["x1"],
                rect["y1"],
            )

            # Top Edge
            lines.append(
                {
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y0,
                }
            )

            # Right Edge
            lines.append(
                {
                    "x0": x1,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                }
            )

            # Bottom Edge
            lines.append(
                {
                    "x0": x0,
                    "y0": y1,
                    "x1": x1,
                    "y1": y1,
                }
            )

            # Left Edge
            lines.append(
                {
                    "x0": x0,
                    "y0": y0,
                    "x1": x0,
                    "y1": y1,
                }
            )

        return pd.DataFrame(lines)

    def get_all_lines(self) -> pd.DataFrame:
        """
        Find all lines in the page.
        """
        rect_lines = self._rect_to_lines()

        all_lines_df = pd.concat(
            [self.all_lines_df, self.line_df, rect_lines], ignore_index=True
        )

        all_lines_df["orientation"] = all_lines_df.apply(
            lambda row: "horizontal" if np.isclose(row.y0, row.y1) else "vertical",
            axis=1,
        )

        self.all_lines_df = all_lines_df.drop_duplicates().reset_index(drop=True)

        return self.all_lines_df

    def clean_lines(self, tol: float = 3.0, min_length: float = 5.0) -> pd.DataFrame:
        """
        Clean the lines DataFrame by applying various operations.
        """

        # Snap all lines to the grid
        for col in ["x0", "y0", "x1", "y1"]:
            self.all_lines_df[col] = self.all_lines_df[col].apply(
                lambda x: Page._snap_to_grid(
                    x, self.grid[0] if "x" in col else self.grid[1]
                )
            )

        # Merge overlapping lines
        hor_lines = Page._merge_horizontal_lines(
            self.all_lines_df[self.all_lines_df.orientation == "horizontal"], tol=tol
        )
        ver_lines = Page._merge_vertical_lines(
            self.all_lines_df[self.all_lines_df.orientation == "vertical"], tol=tol
        )

        # Squash close lines
        if not hor_lines.empty:
            ys = hor_lines.y0.unique()
            _, y_mapping = Page._squash_lines(ys, tol=tol)
            hor_lines["y0"] = hor_lines["y0"].map(y_mapping)
            hor_lines["y1"] = hor_lines["y1"].map(y_mapping)

        if not ver_lines.empty:
            xs = ver_lines.x0.unique()
            _, x_mapping = Page._squash_lines(xs, tol=tol)
            ver_lines["x0"] = ver_lines["x0"].map(x_mapping)
            ver_lines["x1"] = ver_lines["x1"].map(x_mapping)

        # Remove short lines
        hor_lines = hor_lines[abs(hor_lines["x1"] - hor_lines["x0"]) > min_length]
        ver_lines = ver_lines[abs(ver_lines["y1"] - ver_lines["y0"]) > min_length]

        all_lines_df = pd.concat(
            [self.all_lines_df, hor_lines, ver_lines], ignore_index=True
        )

        all_lines_df["orientation"] = all_lines_df.apply(
            lambda row: "horizontal" if np.isclose(row.y0, row.y1) else "vertical",
            axis=1,
        )

        self.all_lines_df = all_lines_df.drop_duplicates().reset_index(drop=True)

        return self.all_lines_df

    def finalise_lines(
        self, tol: float = 3.0, max_span: float = 6.0, extension: float = 2.0
    ) -> pd.DataFrame:
        hor_lines = self.all_lines_df[self.all_lines_df.orientation == "horizontal"]
        ver_lines = self.all_lines_df[self.all_lines_df.orientation == "vertical"]

        # Cluster lines
        hor_lines = Page._cluster_horizontal_lines(
            hor_lines,
            tol=tol,
            max_span=max_span,
        )

        ver_lines = Page._cluster_vertical_lines(
            ver_lines,
            tol=tol,
            max_span=max_span,
        )

        # Extend lines

        all_lines_df = pd.concat([hor_lines, ver_lines], ignore_index=True)

        all_lines_df[["x0", "y0", "x1", "y1"]] = all_lines_df.apply(
            lambda row: pd.Series(
                Page._extend_lines(
                    (row.x0, row.y0, row.x1, row.y1), extension=extension
                )
            ),
            axis=1,
        )

        all_lines_df["orientation"] = all_lines_df.apply(
            lambda row: "horizontal" if np.isclose(row.y0, row.y1) else "vertical",
            axis=1,
        )

        self.all_lines_df = all_lines_df.drop_duplicates().reset_index(drop=True)

        return self.all_lines_df

    # # Operations to perform
    # # 1. Snap to grid
    # # 2. Merge overlapping lines
    # # 3. Cluster lines
    # # 4. Squash lines
    # # 5. Remove short lines
    # # 6. Remove duplicates
    # # 7. Extend lines
    # # 8. Finalize lines

    # # for col in ["x0", "y0", "x1", "y1"]:
    # # self.rect_df[col] = self.rect_df[col].apply(
    # #     lambda x: Page._snap_to_grid(
    # #         x, self.grid_x if "x" in col else self.grid_y
    # #     )
    # # )

    # def get_clustered_lines(
    #     self, tol: float = 3.0, extension: float = 2.0
    # ) -> pd.DataFrame:
    #     hor_lines, ver_lines = self.get_all_lines()

    #     if not hor_lines.empty:
    #         ys = hor_lines.y0.unique()
    #         _, y_mapping = Page._squash_lines(ys, tol=tol)
    #         hor_lines["y0"] = hor_lines["y0"].map(y_mapping)
    #         hor_lines["y1"] = hor_lines["y1"].map(y_mapping)

    #         clustered_horizontal = Page._cluster_horizontal_lines(hor_lines, tol=tol)
    #         clustered_horizontal["orientation"] = "horizontal"
    #     else:
    #         clustered_horizontal = pd.DataFrame(
    #             columns=["x0", "y0", "x1", "y1", "orientation"]
    #         )

    #     if not ver_lines.empty:
    #         xs = ver_lines.x0.unique()
    #         _, x_mapping = Page._squash_lines(xs, tol=tol)
    #         ver_lines["x0"] = ver_lines["x0"].map(x_mapping)
    #         ver_lines["x1"] = ver_lines["x1"].map(x_mapping)

    #         clustered_vertical = Page._cluster_vertical_lines(ver_lines, tol=tol)
    #         clustered_vertical["orientation"] = "vertical"
    #     else:
    #         clustered_vertical = pd.DataFrame(
    #             columns=["x0", "y0", "x1", "y1", "orientation"]
    #         )

    #     all_lines_df = (
    #         pd.concat([self.line_df, clustered_horizontal, clustered_vertical])
    #         .drop_duplicates()
    #         .reset_index(drop=True)
    #     )

    #     all_lines_df[["x0", "y0", "x1", "y1"]] = all_lines_df.apply(
    #         lambda row: pd.Series(
    #             Page._extend_lines(
    #                 (row.x0, row.y0, row.x1, row.y1), extension=extension
    #             )
    #         ),
    #         axis=1,
    #     )

    #     self.all_lines_df = all_lines_df

    #     return self.all_lines_df

    @staticmethod
    def _approx_span_cover(
        span_start: float,
        span_end: float,
        target_start: float,
        target_end: float,
        tol: float = 3.0,
        min_overlap_ratio: float = 0.7,
    ) -> bool:
        overlap = max(
            0, min(span_end + tol, target_end) - max(span_start - tol, target_start)
        )
        required = (target_end - target_start) * min_overlap_ratio

        return overlap >= required

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

    def initialise_cells(
        self, tol: float = 3.0, min_overlap_ratio: float = 0.7
    ) -> list[Cell]:
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
                    np.isclose(y, y0, atol=tol)
                    and Page._approx_span_cover(
                        x_start,
                        x_end,
                        x0,
                        x1,
                        tol=tol,
                        min_overlap_ratio=min_overlap_ratio,
                    )
                    for y, x_start, x_end in horiz_lines
                )
                # For bottom: horizontal at y1
                bottom = any(
                    np.isclose(y, y1, atol=tol)
                    and Page._approx_span_cover(
                        x_start,
                        x_end,
                        x0,
                        x1,
                        tol=tol,
                        min_overlap_ratio=min_overlap_ratio,
                    )
                    for y, x_start, x_end in horiz_lines
                )
                # For left: vertical at x0 that covers [y0, y1]
                left = any(
                    np.isclose(x, x0, atol=tol)
                    and Page._approx_span_cover(
                        y_start,
                        y_end,
                        y0,
                        y1,
                        tol=tol,
                        min_overlap_ratio=min_overlap_ratio,
                    )
                    for x, y_start, y_end in vert_lines
                )
                # For right: vertical at x1
                right = any(
                    np.isclose(x, x1, atol=tol)
                    and Page._approx_span_cover(
                        y_start,
                        y_end,
                        y0,
                        y1,
                        tol=tol,
                        min_overlap_ratio=min_overlap_ratio,
                    )
                    for x, y_start, y_end in vert_lines
                )

                if top and bottom and left and right:
                    cell = Cell(x0, y0, x1, y1)
                    Page._assign_cell_text(self.char_df, cell)
                    cells.append(cell)

        self.cells = cells

        return cells

    def group_cells(
        self, tol: float = 3.0
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

    def link_cells(self, tol: float = 3.0) -> None:
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

        merged = self.merge_small_cells(tol=tol)

        if merged:
            self.link_cells(tol=tol)

        return None

    def merge_small_cells(self, tol: float = 3.0):
        small_cells = [
            cell for cell in self.cells if (cell.width < tol or cell.height < tol)
        ]
        merged_any = False

        for small_cell in small_cells:
            neighbors = [
                n
                for n in [
                    small_cell.left,
                    small_cell.right,
                    small_cell.top,
                    small_cell.bottom,
                ]
                if n is not None
            ]

            if not neighbors:
                continue

            max_overlap = 0
            best_neighbor = None
            for neighbor in neighbors:
                overlap = Page._shared_boundary_length(small_cell, neighbor)
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_neighbor = neighbor

            if best_neighbor is None:
                continue

            best_neighbor.x0 = min(best_neighbor.x0, small_cell.x0)
            best_neighbor.y0 = min(best_neighbor.y0, small_cell.y0)
            best_neighbor.x1 = max(best_neighbor.x1, small_cell.x1)
            best_neighbor.y1 = max(best_neighbor.y1, small_cell.y1)
            best_neighbor.width = abs(best_neighbor.x1 - best_neighbor.x0)
            best_neighbor.height = abs(best_neighbor.y1 - best_neighbor.y0)

            if best_neighbor.text and small_cell.text:
                best_neighbor.text += " " + small_cell.text
            elif small_cell.text:
                best_neighbor.text = small_cell.text

            self.cells.remove(small_cell)

            for neighbor in [
                small_cell.left,
                small_cell.right,
                small_cell.top,
                small_cell.bottom,
            ]:
                if neighbor is not None:
                    # Unlink small cell from neighbors
                    if neighbor.left == small_cell:
                        neighbor.left = None
                    if neighbor.right == small_cell:
                        neighbor.right = None
                    if neighbor.top == small_cell:
                        neighbor.top = None
                    if neighbor.bottom == small_cell:
                        neighbor.bottom = None

            merged_any = True

        return merged_any

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
    import os
    import gc

    # path = "./tests/data/UBI Format 2.pdf"
    path = "./tests/data/"
    bordered_list = [
        "Canara Bank format 4.pdf",
    ]

    for file in bordered_list:
        print(f"Processing {file}...")
        doc = Document(os.path.join(path, file))
        page_0 = doc.pages[0]

        lines = page_0.get_all_lines()
        clean_lines = page_0.clean_lines()
        final_lines = page_0.finalise_lines()
        cells = page_0.initialise_cells()
        print(f"Cells initialized for {file}: {len(cells)}")

        page_0.link_cells()
        rows = page_0.get_rows()
        columns = page_0.get_columns()
        tables = page_0.get_tables()
        print(f"Tables detected for {file}: {len(tables)}")

        stub = file.split(".pdf")[0]

        with open(f"./tests/results/{stub}_output_20250829_run2.txt", "w") as f:
            for i, cell in enumerate(page_0.cells):
                f.write(f"Cell {i}: {cell}, Text: '{cell.text}'\n")

        del doc
        gc.collect()

# %%


# from skimage.draw import line as skline


# # Function to convert PDF coordinate (x, y) to NumPy array indices (row, col)
# def pdf_to_array_coords(x, y, height):
#     # y-axis flip because PDF origin (0,0) is bottom-left
#     row = height - int(round(y))
#     col = int(round(x))
#     return row, col


# # %%
# height = int(page_0.height)
# width = int(page_0.width)

# # %%
# import matplotlib.pyplot as plt

# # Create a blank canvas
# canvas = np.zeros((height, width), dtype=np.uint8)

# # %%
# # Draw lines on the canvas
# for cell in page_0.cells:
#     # Assuming cell is a patch or has attributes x0, y0, x1, y1 bounding the table box
#     x0, y0 = cell.x0, cell.y0
#     x1, y1 = cell.x1, cell.y1

#     # Convert PDF coords to array indices (row, col)
#     r0, c0 = pdf_to_array_coords(x0, y0, height)
#     r1, c1 = pdf_to_array_coords(x1, y1, height)

#     # Ensure row/col indices are ordered properly (r0 < r1, c0 < c1)
#     r_min, r_max = min(r0, r1), max(r0, r1)
#     c_min, c_max = min(c0, c1), max(c0, c1)

#     # Clip indices within array bounds
#     r_min = np.clip(r_min, 0, height - 1)
#     r_max = np.clip(r_max, 0, height - 1)
#     c_min = np.clip(c_min, 0, width - 1)
#     c_max = np.clip(c_max, 0, width - 1)

#     # Fill the rectangular region in the canvas
#     canvas[r_min : r_max + 1, c_min : c_max + 1] = 1

# # %%


# from skimage.draw import line as skline
# import matplotlib.pyplot as plt

# # Create a blank canvas
# canvas = np.zeros((height, width), dtype=np.uint8)

# # %%
# # Draw lines on the canvas
# for _, line_obj in page_0.all_lines_df.iterrows():
#     x0, y0 = line_obj["x0"], line_obj["y0"]
#     x1, y1 = line_obj["x1"], line_obj["y1"]
#     r0, c0 = pdf_to_array_coords(x0, y0, height)
#     r1, c1 = pdf_to_array_coords(x1, y1, height)

#     # Draw line pixels on the array using skimage's line function
#     rr, cc = skline(r0, c0, r1, c1)
#     # Ensure indices are inside array bounds
#     rr = np.clip(rr, 0, height - 1)
#     cc = np.clip(cc, 0, width - 1)

#     # Mark line pixels as 1
#     canvas[rr, cc] = 1
# # %%
