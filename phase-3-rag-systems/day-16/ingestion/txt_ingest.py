"""
txt_ingest.py

Plain text ingestion module for Day 16 (Document Ingestion & Chunking Strategies).

Strategy:
    This is deliberately the simplest ingestion module, because .txt files
    carry NO structural metadata at all:
      - No pages (like DOCX) -> page_number is always None.
      - No styles/tags to mark headings (unlike DOCX's "Heading 1" style,
        or PDF's font-size signal) -> heading detection here is a weak
        heuristic at best, not a real signal. We apply one anyway (see
        _looks_like_heading) so section_heading isn't always None, but
        this is the module where you should expect the LOWEST heading
        detection accuracy of the three — worth calling out explicitly
        in day16_report.md as a trade-off of using plain text as a source.

    Paragraph splitting is done on blank lines (one or more consecutive
    blank lines = paragraph boundary), which is the closest plain text
    gets to a structural signal.

Output shape:
    extract_txt(path) -> list[ExtractedElement]
    One element per paragraph (or detected heading), in file order.
"""

import sys
from pathlib import Path
from typing import Optional

# See pdf_ingest.py for why this is needed — makes metadata_schema
# importable regardless of the working directory this script is run from.
sys.path.append(str(Path(__file__).resolve().parent.parent))

from metadata_schema import ExtractedElement, ElementType, elements_to_json


def _looks_like_heading(line: str, next_line_is_blank: bool) -> bool:
    """
    Weak heuristic only — plain text has no real structural markers.
    A line is treated as a possible heading if it's short, doesn't end in
    typical sentence punctuation, and is followed by a blank line (i.e. it
    stands alone rather than flowing into a paragraph).

    This WILL produce false positives/negatives on real-world files. It's
    included so section_heading isn't uniformly None, but treat its output
    with more skepticism than pdf_ingest.py's font-size heuristic or
    docx_ingest.py's explicit style-name check — this is genuinely the
    weakest of the three signals, and that gap is itself worth reporting.
    """
    stripped = line.strip()
    if not stripped:
        return False
    if len(stripped) > 80:
        return False
    if stripped.endswith((".", ",", ";", ":")):
        return False
    return next_line_is_blank


def extract_txt(path: str) -> list[ExtractedElement]:
    """
    Extracts a plain text file into an ordered list of ExtractedElement.
    Splits on blank lines to approximate paragraph boundaries, and applies
    a weak heading heuristic (see _looks_like_heading).
    """
    source_filename = Path(path).name

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    elements: list[ExtractedElement] = []
    element_index = 0
    current_heading: Optional[str] = None
    paragraph_buffer: list[str] = []

    def flush_paragraph():
        nonlocal element_index
        text = " ".join(paragraph_buffer).strip()
        if text:
            elements.append(ExtractedElement(
                text=text,
                source_filename=source_filename,
                page_number=None,  # plain text has no page concept at all
                section_heading=current_heading,
                element_type=ElementType.PARAGRAPH,
                element_index=element_index,
            ))
            element_index += 1
        paragraph_buffer.clear()

    for i, raw_line in enumerate(lines):
        line = raw_line.rstrip("\n")
        is_blank = not line.strip()
        next_is_blank = (i + 1 >= len(lines)) or (not lines[i + 1].strip())

        if is_blank:
            flush_paragraph()
            continue

        # Only treat a line as a heading candidate if it's alone in its
        # paragraph (buffer empty so far AND next line is blank) — a short
        # non-punctuated line in the MIDDLE of a paragraph isn't a heading.
        if not paragraph_buffer and _looks_like_heading(line, next_is_blank):
            current_heading = line.strip()
            elements.append(ExtractedElement(
                text=current_heading,
                source_filename=source_filename,
                page_number=None,
                section_heading=current_heading,
                element_type=ElementType.HEADING,
                element_index=element_index,
            ))
            element_index += 1
        else:
            paragraph_buffer.append(line.strip())

    flush_paragraph()  # catch any trailing paragraph with no final blank line

    return elements


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    RAW_PATH = sys.argv[1] if len(sys.argv) > 1 else "data/raw/sample.txt"
    OUT_PATH = "data/extracted/txt_text.json"

    elements = extract_txt(RAW_PATH)
    print(f"[txt_ingest] Extracted {len(elements)} elements from {RAW_PATH}")

    elements_to_json(elements, OUT_PATH)
    print(f"[txt_ingest] Wrote {OUT_PATH}")