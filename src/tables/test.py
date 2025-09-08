# %%
from tables.classes import Char, Cell, Page, Document

import pdfplumber as plumber
import numpy as np
import matplotlib.pyplot as plt

from skimage.draw import line as skline

# %%
path = "../../tests/data/DEOGIRI NAGARI SAHAKARI BANK LTD.pdf"

doc = Document(path)
page = doc.pages[0]
# %%
height = int(page.height)
width = int(page.width)


# %%
# Function to convert PDF coordinate (x, y) to NumPy array indices (row, col)
def pdf_to_array_coords(x, y, height):
    # y-axis flip because PDF origin (0,0) is bottom-left
    row = height - int(round(y))
    col = int(round(x))
    return row, col


# %%

rect_df = page.rect_df.copy()

cells = []
for _, cell in rect_df.iterrows():
    x0, y0, x1, y1 = cell
    cells.append(Cell(x0, y0, x1, y1))

for cell in cells:
    Page._assign_cell_text(page.char_df, cell, tol=3.0)

# %%

for cell in cells:
    print(cell.text)
# %%
