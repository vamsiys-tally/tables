# %%
import os

import pandas as pd
import numpy as np
import pdfplumber as plumber

from tables.classes import Document, Cell
from collections import namedtuple
# %%

path = "../../tests/data/"
file = "DEOGIRI NAGARI SAHAKARI BANK LTD.pdf"

doc = Document(os.path.join(path, file))
page = doc.pages[0]
# %%

lines = page.page.lines
edges = page.page.edges
rects = page.page.rects

chars = page.page.chars
# %%

Line = namedtuple("Line", ["x0", "y0", "x1", "y1"])
Edge = namedtuple("Edge", ["x0", "y0", "x1", "y1"])
Rect = namedtuple("Rect", ["x0", "y0", "x1", "y1"])
Char = namedtuple("Char", ["x0", "y0", "x1", "y1", "text"])

line_list = []
edge_list = []
rect_list = []
char_list = []

for line in lines:
    new_line = Line(line["x0"], line["y0"], line["x1"], line["y1"])
    if new_line not in set(line_list):
        line_list.append(new_line)

for edge in edges:
    new_edge = Edge(edge["x0"], edge["y0"], edge["x1"], edge["y1"])
    if new_edge not in set(edge_list):
        edge_list.append(new_edge)

for rect in rects:
    new_rect = Rect(rect["x0"], rect["y0"], rect["x1"], rect["y1"])
    if new_rect not in set(rect_list):
        rect_list.append(new_rect)

for char in chars:
    new_char = Char(char["x0"], char["y0"], char["x1"], char["y1"], char["text"])
    if new_char not in set(char_list):
        char_list.append(new_char)

# %%line_list = []
line_list = []
edge_list = []
rect_list = []
char_list = []

for line in lines:
    new_line = {"x0": line["x0"], "y0": line["y0"], "x1": line["x1"], "y1": line["y1"]}
    if new_line not in line_list:
        line_list.append(new_line)

for edge in edges:
    new_edge = {"x0": edge["x0"], "y0": edge["y0"], "x1": edge["x1"], "y1": edge["y1"]}
    if new_edge not in edge_list:
        edge_list.append(new_edge)

for rect in rects:
    new_rect = {"x0": rect["x0"], "y0": rect["y0"], "x1": rect["x1"], "y1": rect["y1"]}
    if new_rect not in rect_list:
        rect_list.append(new_rect)

for char in chars:
    new_char = {
        "x0": char["x0"],
        "y0": char["y0"],
        "x1": char["x1"],
        "y1": char["y1"],
        "text": char["text"],
    }
    if new_char not in char_list:
        char_list.append(new_char)


# %%
line_df = pd.DataFrame(line_list)
edge_df = pd.DataFrame(edge_list)
rect_df = pd.DataFrame(rect_list)
char_df = pd.DataFrame(char_list)

# %%

cells = []
tol = 3.0
for rect in rect_list:
    cells.append(Cell(rect.x0, rect.y0, rect.x1, rect.y1))
    for char in char_list:
        if (
            char.x0 >= rect.x0 - tol
            and char.x1 <= rect.x1 + tol
            and char.y0 >= rect.y0 - tol
            and char.y1 <= rect.y1 + tol
        ):
            cells[-1].text += char.text

# %%
for cell in cells:
    print(cell.text)

# %%
vert_edge = []
hor_edge = []
other_edge = []

for edge in edge_list:
    if edge.x0 == edge.x1:
        vert_edge.append(edge)
    elif edge.y0 == edge.y1:
        hor_edge.append(edge)
    else:
        other_edge.append(edge)

# %%

vert_edge = sorted(vert_edge, key=lambda e: (e.x0, e.x1))
hor_edge = sorted(hor_edge, key=lambda e: (e.y0, e.y1))


# %%
for ve in vert_edge:
    x0 = ve.x0
    y0 = ve.y1
    x1 = ve.x1
    y1 = ve.y1

    cell_x0 = None
    cell_y0 = None
    cell_x1 = None
    cell_y1 = None

    for he in hor_edge:
        if he.x0 == x0 and he.y0 == y0:
            cell_x0 = he.x0
            cell_y0 = he.y0

        he_x1 = he.x1
        he_y1 = he.y1

        for ve in vert_edge:
            if he_x1 == ve.x0 and (he_y1 == ve.y0 or he_y1 == ve.y1):
                cell_x1 = he.x1
                cell_y1 = he.y1


# %%
cells = []

for i, v_edge in enumerate(vert_edge):
    next_v_edge = vert_edge[i + 1] if i + 1 < len(vert_edge) else None
    current_cell = None

    for j, h_edge in enumerate(hor_edge):
        next_h_edge = hor_edge[j + 1] if j + 1 < len(hor_edge) else None

        if next_h_edge and next_v_edge:
            if v_edge.x0 == h_edge.x0 and v_edge.y0 == h_edge.y0:
                if next_h_edge.x1 == next_v_edge.x0 and (
                    next_h_edge.y1 == next_v_edge.y0 or next_h_edge.y1 == next_v_edge.y1
                ):
                    cell = Cell(v_edge.x0, v_edge.y0, next_v_edge.x1, next_h_edge.y1)
                    cells.append(cell)

# %%

Intersection = namedtuple("Intersection", ["x", "y"])

intersections = []
seen = set()

for v_edge in vert_edge:
    x = v_edge.x0
    y_start = v_edge.y0
    y_end = v_edge.y1

    for h_edge in hor_edge:
        if (h_edge.x0 <= x <= h_edge.x1) and (y_start <= h_edge.y0 <= y_end):
            if (x, h_edge.y0) not in seen:
                seen.add((x, h_edge.y0))
                intersections.append(Intersection(x, h_edge.y0))

# %%
Rectangle = namedtuple(
    "Rectangle", ["top_left", "top_right", "bottom_left", "bottom_right"]
)

rectangles = []

unique_x = sorted(set(line.x0 for line in vert_edge))
unique_y = sorted(set(line.y0 for line in hor_edge))

for i in range(len(unique_x) - 1):
    for j in range(i + 1, len(unique_x)):
        x1, x2 = unique_x[i], unique_x[j]

        for k in range(len(unique_y) - 1):
            for l in range(k + 1, len(unique_y)):
                y1, y2 = unique_y[k], unique_y[l]

                # Check if all 4 corners exist in intersections
                if (
                    (x1, y1) in intersections
                    and (x2, y1) in intersections
                    and (x1, y2) in intersections
                    and (x2, y2) in intersections
                ):
                    rect = Rectangle(
                        top_left=(x1, y1),
                        top_right=(x2, y1),
                        bottom_left=(x1, y2),
                        bottom_right=(x2, y2),
                    )
                    rectangles.append(rect)

# %%
from skimage.draw import line as skline
import numpy as np


def pdf_to_array_coords(x, y, height):
    # y-axis flip because PDF origin (0,0) is bottom-left
    row = height - int(round(y))
    col = int(round(x))
    return row, col


# %%
height = int(page.height)
width = int(page.width)

# %%
import matplotlib.pyplot as plt

# Create a blank canvas
canvas = np.zeros((height, width), dtype=np.uint8)

for rectangle in rectangles:
    x0, y0 = rectangle.top_left[0], rectangle.top_left[1]
    x1, y1 = rectangle.bottom_right[0], rectangle.bottom_right[1]

    # Convert PDF coords to array indices (row, col)
    r0, c0 = pdf_to_array_coords(x0, y0, height)
    r1, c1 = pdf_to_array_coords(x1, y1, height)

    # Ensure ordered indices
    r_min, r_max = min(r0, r1), max(r0, r1)
    c_min, c_max = min(c0, c1), max(c0, c1)

    r_min_clamped = np.clip(r_min, 0, height - 1)
    r_max_clamped = np.clip(r_max, 0, height - 1)
    c_min_clamped = np.clip(c_min, 0, width - 1)
    c_max_clamped = np.clip(c_max, 0, width - 1)

    # Draw top edge
    rr, cc = skline(r_min_clamped, c_min_clamped, r_min_clamped, c_max_clamped)
    canvas[rr, cc] = 1

    # Draw bottom edge
    rr, cc = skline(r_max_clamped, c_min_clamped, r_max_clamped, c_max_clamped)
    canvas[rr, cc] = 1

    # Draw left edge
    rr, cc = skline(r_min_clamped, c_min_clamped, r_max_clamped, c_min_clamped)
    canvas[rr, cc] = 1

    # Draw right edge
    rr, cc = skline(r_min_clamped, c_max_clamped, r_max_clamped, c_max_clamped)
    canvas[rr, cc] = 1

# %%


for rect in rectangles:
    cell = Cell(
        rect.top_left[0],
        rect.top_left[1],
        rect.bottom_right[0],
        rect.bottom_right[1],
    )

    for char in char_list:
        if (
            char.x0 >= cell.x0 - tol
            and char.x1 <= cell.x1 + tol
            and char.y0 >= cell.y0 - tol
            and char.y1 <= cell.y1 + tol
        ):
            cell.text += char.text

    cells.append(cell)


# %%
rectangles = []


def is_cell_rectangle(x1, x2, y1, y2, intersections):
    for x in unique_x:
        for y in unique_y:
            if x1 < x < x2 and y1 < y < y2 and (x, y) in intersections:
                return False
    return True


for i in range(len(unique_x) - 1):
    for j in range(i + 1, len(unique_x)):
        x1, x2 = unique_x[i], unique_x[j]
        for k in range(len(unique_y) - 1):
            for l in range(k + 1, len(unique_y)):
                y1, y2 = unique_y[k], unique_y[l]
                if (
                    (x1, y1) in intersections
                    and (x2, y1) in intersections
                    and (x1, y2) in intersections
                    and (x2, y2) in intersections
                ):
                    if is_cell_rectangle(x1, x2, y1, y2, intersections):
                        rect = Rectangle(
                            top_left=(x1, y1),
                            top_right=(x2, y1),
                            bottom_left=(x1, y2),
                            bottom_right=(x2, y2),
                        )
                        rectangles.append(rect)

# %%
