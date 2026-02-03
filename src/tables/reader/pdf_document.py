"""
PDF Document wrapper for bank statement processing.

This module provides a high-level wrapper around pdfplumber for PDF operations,
with specific functionality for bank statement processing including:
- Password-protected PDF handling
- Text layer detection (scanned vs generated)
- Page-level text and line extraction
- Coordinate normalization

Example Usage:
    ```python
    from tables.reader.pdf_document import PDFDocument

    # Open a PDF (with optional password)
    doc = PDFDocument.open("statement.pdf", password="secret")

    # Check if scanned
    if doc.is_scanned:
        print("Scanned document - not supported")

    # Extract text from first page
    text = doc.get_page_text(0)

    # Get lines for table detection
    lines = doc.get_page_lines(0)

    # Always close when done
    doc.close()

    # Or use context manager
    with PDFDocument.open("statement.pdf") as doc:
        for page_num in range(doc.page_count):
            text = doc.get_page_text(page_num)
    ```

Design Notes:
    - Uses pdfplumber as primary PDF library for text/line extraction
    - Lazy loading of pages to minimize memory usage
    - Caches extracted data per page to avoid redundant processing
    - Thread-safe for read operations (but not for concurrent page processing)
"""

from dataclasses import dataclass, field
from typing import Any, Optional

import pdfplumber
from pdfplumber.page import Page as PdfplumberPage

from tables.models.document import BoundingBox
from tables.models.errors import ErrorCode, ProcessingError


@dataclass
class PageInfo:
    """
    Information about a single PDF page.

    Attributes:
        page_number: Zero-indexed page number
        width: Page width in points
        height: Page height in points
        rotation: Page rotation in degrees (0, 90, 180, 270)
        has_text: Whether the page has extractable text
        text_density: Characters per square point (indicator of scanned vs generated)
    """

    page_number: int
    width: float
    height: float
    rotation: int = 0
    has_text: bool = False
    text_density: float = 0.0

    @property
    def is_likely_scanned(self) -> bool:
        """
        Determine if page is likely scanned based on text density.

        Scanned pages typically have very low text density (<0.001 chars/pt²)
        because OCR isn't applied by default.

        Returns:
            True if page appears to be scanned
        """
        # Threshold: 0.001 characters per square point
        # A typical A4 page is ~595 x 842 points = ~501,000 sq points
        # A page with 500 chars would have density 0.001
        return self.text_density < 0.001

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "page_number": self.page_number,
            "width": self.width,
            "height": self.height,
            "rotation": self.rotation,
            "has_text": self.has_text,
            "text_density": self.text_density,
            "is_likely_scanned": self.is_likely_scanned,
        }


@dataclass
class Line:
    """
    Represents a line segment extracted from a PDF page.

    Lines are used for table detection - horizontal and vertical lines
    typically indicate table borders.

    Attributes:
        x0: Starting x coordinate
        y0: Starting y coordinate
        x1: Ending x coordinate
        y1: Ending y coordinate
        width: Line stroke width
        orientation: 'horizontal', 'vertical', or 'diagonal'
    """

    x0: float
    y0: float
    x1: float
    y1: float
    width: float = 1.0
    orientation: str = "unknown"

    def __post_init__(self):
        """Determine line orientation after initialization."""
        if self.orientation == "unknown":
            dx = abs(self.x1 - self.x0)
            dy = abs(self.y1 - self.y0)
            if dy < 2 and dx > 2:  # Tolerance of 2 points
                self.orientation = "horizontal"
            elif dx < 2 and dy > 2:
                self.orientation = "vertical"
            else:
                self.orientation = "diagonal"

    @property
    def length(self) -> float:
        """Calculate line length."""
        return ((self.x1 - self.x0) ** 2 + (self.y1 - self.y0) ** 2) ** 0.5

    @property
    def bounding_box(self) -> BoundingBox:
        """Get bounding box of the line."""
        return BoundingBox(
            x0=min(self.x0, self.x1),
            y0=min(self.y0, self.y1),
            x1=max(self.x0, self.x1),
            y1=max(self.y0, self.y1),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "x0": self.x0,
            "y0": self.y0,
            "x1": self.x1,
            "y1": self.y1,
            "width": self.width,
            "orientation": self.orientation,
            "length": self.length,
        }


@dataclass
class Character:
    """
    Represents a single character extracted from a PDF page.

    Attributes:
        char: The character string
        x0, y0, x1, y1: Bounding box coordinates
        font_name: Name of the font
        font_size: Size of the font in points
    """

    char: str
    x0: float
    y0: float
    x1: float
    y1: float
    font_name: str = ""
    font_size: float = 0.0

    @property
    def bounding_box(self) -> BoundingBox:
        """Get bounding box of the character."""
        return BoundingBox(x0=self.x0, y0=self.y0, x1=self.x1, y1=self.y1)


@dataclass
class PDFDocument:
    """
    High-level wrapper for PDF document operations.

    This class provides a convenient interface for PDF operations needed
    for bank statement processing, including:
    - Opening password-protected PDFs
    - Detecting scanned vs generated content
    - Extracting text, lines, and characters per page
    - Page information and metadata

    Attributes:
        file_path: Path to the PDF file
        page_count: Number of pages in the document
        is_encrypted: Whether the PDF is password-protected
        is_scanned: Whether the document appears to be scanned
        metadata: PDF metadata dictionary

    Note:
        Always close the document when done, or use as context manager:
        ```python
        with PDFDocument.open("file.pdf") as doc:
            # work with doc
        ```
    """

    file_path: str
    _pdf: Optional[pdfplumber.PDF] = field(default=None, repr=False)
    _page_cache: dict[int, PdfplumberPage] = field(default_factory=dict, repr=False)
    _page_info_cache: dict[int, PageInfo] = field(default_factory=dict, repr=False)

    @classmethod
    def open(
        cls, file_path: str, password: Optional[str] = None
    ) -> "PDFDocument":
        """
        Open a PDF file for processing.

        Args:
            file_path: Path to the PDF file
            password: Optional password for encrypted PDFs

        Returns:
            PDFDocument instance

        Raises:
            ProcessingError: If file cannot be opened or password is incorrect

        Example:
            ```python
            doc = PDFDocument.open("statement.pdf")
            # or with password
            doc = PDFDocument.open("statement.pdf", password="secret")
            ```
        """
        try:
            pdf = pdfplumber.open(file_path, password=password)
        except Exception as e:
            error_str = str(e).lower()

            # Check for password-related errors
            if "password" in error_str or "encrypted" in error_str:
                if password is None:
                    raise ProcessingError(
                        ErrorCode.FILE_PASSWORD_REQUIRED,
                        details={"path": file_path},
                    )
                else:
                    raise ProcessingError(
                        ErrorCode.FILE_PASSWORD_INCORRECT,
                        details={"path": file_path},
                    )

            # Generic corruption/parse error
            raise ProcessingError(
                ErrorCode.FILE_CORRUPTED,
                details={"path": file_path, "error": str(e)},
            )

        doc = cls(file_path=file_path, _pdf=pdf)
        return doc

    def __enter__(self) -> "PDFDocument":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - ensures PDF is closed."""
        self.close()

    def close(self) -> None:
        """
        Close the PDF document and release resources.

        Should be called when done processing, or use context manager.
        """
        if self._pdf is not None:
            self._pdf.close()
            self._pdf = None
        self._page_cache.clear()
        self._page_info_cache.clear()

    @property
    def page_count(self) -> int:
        """Get the number of pages in the document."""
        if self._pdf is None:
            return 0
        return len(self._pdf.pages)

    @property
    def is_encrypted(self) -> bool:
        """Check if the PDF was encrypted (password-protected)."""
        if self._pdf is None:
            return False
        # pdfplumber doesn't expose is_encrypted directly
        # Check if a password was used to open the document
        return self._pdf.password is not None

    @property
    def metadata(self) -> dict[str, Any]:
        """Get PDF metadata dictionary."""
        if self._pdf is None:
            return {}
        return self._pdf.metadata or {}

    @property
    def is_scanned(self) -> bool:
        """
        Determine if the document appears to be scanned.

        A document is considered scanned if more than 50% of its pages
        have very low text density, indicating they are image-based
        without OCR.

        Returns:
            True if document appears to be scanned
        """
        if self.page_count == 0:
            return False

        scanned_count = 0
        for page_num in range(self.page_count):
            page_info = self.get_page_info(page_num)
            if page_info.is_likely_scanned:
                scanned_count += 1

        return scanned_count > (self.page_count / 2)

    def _get_page(self, page_number: int) -> PdfplumberPage:
        """
        Get a pdfplumber page object, with caching.

        Args:
            page_number: Zero-indexed page number

        Returns:
            pdfplumber Page object
        """
        if self._pdf is None:
            raise ValueError("PDF document is closed")

        if page_number not in self._page_cache:
            if page_number < 0 or page_number >= self.page_count:
                raise ValueError(f"Page {page_number} out of range (0-{self.page_count - 1})")
            self._page_cache[page_number] = self._pdf.pages[page_number]

        return self._page_cache[page_number]

    def get_page_info(self, page_number: int) -> PageInfo:
        """
        Get information about a specific page.

        Args:
            page_number: Zero-indexed page number

        Returns:
            PageInfo object with page dimensions and text info
        """
        if page_number in self._page_info_cache:
            return self._page_info_cache[page_number]

        page = self._get_page(page_number)

        # Extract text to calculate density
        text = page.extract_text() or ""
        char_count = len(text.replace(" ", "").replace("\n", ""))

        # Calculate text density (chars per square point)
        area = page.width * page.height
        text_density = char_count / area if area > 0 else 0

        page_info = PageInfo(
            page_number=page_number,
            width=page.width,
            height=page.height,
            rotation=page.rotation or 0,
            has_text=char_count > 0,
            text_density=text_density,
        )

        self._page_info_cache[page_number] = page_info
        return page_info

    def get_page_text(self, page_number: int) -> str:
        """
        Extract all text from a page.

        Args:
            page_number: Zero-indexed page number

        Returns:
            Extracted text string (may be empty for scanned pages)
        """
        page = self._get_page(page_number)
        return page.extract_text() or ""

    def get_page_characters(self, page_number: int) -> list[Character]:
        """
        Extract all characters with positions from a page.

        This is useful for precise text positioning needed for
        table cell text assignment.

        Args:
            page_number: Zero-indexed page number

        Returns:
            List of Character objects with positions
        """
        page = self._get_page(page_number)
        chars = page.chars or []

        return [
            Character(
                char=c.get("text", ""),
                x0=c.get("x0", 0),
                y0=c.get("y0", 0),
                x1=c.get("x1", 0),
                y1=c.get("y1", 0),
                font_name=c.get("fontname", ""),
                font_size=c.get("size", 0),
            )
            for c in chars
            if c.get("text", "").strip()  # Skip whitespace-only chars
        ]

    def get_page_lines(self, page_number: int) -> list[Line]:
        """
        Extract all line segments from a page.

        Lines include both explicit line objects and edges of rectangles.
        These are essential for detecting table borders.

        Args:
            page_number: Zero-indexed page number

        Returns:
            List of Line objects
        """
        page = self._get_page(page_number)
        lines: list[Line] = []

        # Get explicit lines
        for line in page.lines or []:
            lines.append(
                Line(
                    x0=line.get("x0", 0),
                    y0=line.get("y0", 0),
                    x1=line.get("x1", 0),
                    y1=line.get("y1", 0),
                    width=line.get("linewidth", 1),
                )
            )

        # Get edges (from rectangles)
        for edge in page.edges or []:
            lines.append(
                Line(
                    x0=edge.get("x0", 0),
                    y0=edge.get("y0", 0),
                    x1=edge.get("x1", 0),
                    y1=edge.get("y1", 0),
                    width=edge.get("linewidth", 1),
                )
            )

        return lines

    def get_page_rectangles(self, page_number: int) -> list[BoundingBox]:
        """
        Extract all rectangles from a page.

        Rectangles may represent table cells, shaded regions, or other
        structural elements.

        Args:
            page_number: Zero-indexed page number

        Returns:
            List of BoundingBox objects
        """
        page = self._get_page(page_number)
        rects: list[BoundingBox] = []

        for rect in page.rects or []:
            rects.append(
                BoundingBox(
                    x0=rect.get("x0", 0),
                    y0=rect.get("y0", 0),
                    x1=rect.get("x1", 0),
                    y1=rect.get("y1", 0),
                )
            )

        return rects

    def get_page_words(self, page_number: int) -> list[dict[str, Any]]:
        """
        Extract all words with positions from a page.

        Returns raw pdfplumber word dictionaries with bounding boxes.
        Useful for text clustering in table detection.

        Args:
            page_number: Zero-indexed page number

        Returns:
            List of word dictionaries with keys:
            - text: The word text
            - x0, top, x1, bottom: Bounding box coordinates
            - fontname, size: Font information

        Example:
            >>> words = doc.get_page_words(0)
            >>> for word in words:
            ...     print(f"{word['text']} at ({word['x0']}, {word['top']})")
        """
        page = self._get_page(page_number)
        return page.extract_words() or []

    def get_page_rects(self, page_number: int) -> list[dict[str, Any]]:
        """
        Extract raw rectangle dictionaries from a page.

        Returns pdfplumber rectangle dictionaries for use in
        table structure classification.

        Args:
            page_number: Zero-indexed page number

        Returns:
            List of rectangle dictionaries with coordinate keys
        """
        page = self._get_page(page_number)
        return page.rects or []

    def get_page_lines_raw(self, page_number: int) -> list[dict[str, Any]]:
        """
        Extract raw line dictionaries from a page.

        Returns pdfplumber line and edge dictionaries for use in
        table structure classification.

        Args:
            page_number: Zero-indexed page number

        Returns:
            List of line dictionaries with coordinate keys
        """
        page = self._get_page(page_number)
        lines = list(page.lines or [])
        edges = list(page.edges or [])
        return lines + edges

    def get_all_text(self) -> str:
        """
        Extract text from all pages concatenated.

        Returns:
            All text from the document
        """
        texts = []
        for page_num in range(self.page_count):
            texts.append(self.get_page_text(page_num))
        return "\n\n".join(texts)

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize document metadata to dictionary.

        Note: Does not include page content, only metadata.
        """
        return {
            "file_path": self.file_path,
            "page_count": self.page_count,
            "is_encrypted": self.is_encrypted,
            "is_scanned": self.is_scanned,
            "metadata": self.metadata,
        }
