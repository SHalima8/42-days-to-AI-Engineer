"""
pdf_ingest.py

PDF ingestion using both pdfplumber and pypdf, each doing what it's actually
good at:

  - pdfplumber: structured extraction. Uses font-size to guess headings
    (native PDFs don't tag headings explicitly — size/boldness relative to
    body text is the only real signal available), and extracts tables.
    This is the function that feeds the rest of the pipeline.

  - pypdf: fast, naive baseline extraction — no layout awareness, no
    heading detection. Kept specifically so you can compare its output
    against pdfplumber's for the Day 16 trade-offs write-up.
"""

import sys
import os
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metadata_schema import ExtractedElement, ElementType
from table_registry import (
    detect_caption, generate_table_id, summarize_table_with_llm, register_table,
)

import pdfplumber
from pypdf import PdfReader


# ---------- pypdf: fast, naive baseline ----------

def extract_with_pypdf(pdf_path: str) -> list[ExtractedElement]:
    """
    Simple per-page extraction. No heading detection, no table structure —
    pypdf has no font-size or layout info to work with, so every page just
    becomes one PARAGRAPH element. This is the "naive" baseline to compare
    pdfplumber against.

    Edge cases handled:
      - encrypted/password-protected PDFs raise a clear error instead of
        failing deep inside pypdf with a confusing traceback
      - a page that fails to parse is skipped with a warning, not fatal
      - a page with no extractable text (e.g. a scanned page mixed into an
        otherwise normal PDF) is skipped — that's what ocr_ingest.py is for
    """
    filename = os.path.basename(pdf_path)
    reader = PdfReader(pdf_path)

    if reader.is_encrypted:
        raise ValueError(
            f"{filename} is password-protected. Decrypt it first — "
            f"pypdf can only read it with reader.decrypt('password')."
        )

    elements = []
    for page_num, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as e:
            print(f"  [warn] pypdf failed on page {page_num} of {filename}: {e} — skipping page")
            continue

        if not text.strip():
            print(f"  [warn] page {page_num} of {filename} has no extractable text "
                  f"(likely a scanned page) — skipping, use ocr_ingest.py for this page")
            continue

        elements.append(ExtractedElement(
            text=text.strip(),
            source_filename=filename,
            page_number=page_num,
            section_heading=None,   # pypdf has no way to know this
            element_type=ElementType.PARAGRAPH,
            element_index=len(elements),
        ))
    return elements


# ---------- pdfplumber: structured extraction ----------

def _guess_body_font_size(page) -> float:
    """
    Body text is whatever font size appears most often on the page.
    Headings are then detected as text meaningfully larger than this.
    """
    words = page.extract_words(extra_attrs=["size"])
    if not words:
        return 10.0  # fallback default
    sizes = [round(w["size"]) for w in words]
    return Counter(sizes).most_common(1)[0][0]


def _group_words_into_lines(words):
    """Group words with extra_attrs into lines based on vertical position."""
    lines = {}
    for w in words:
        top = round(w["top"])  # round to merge words on the same visual line
        lines.setdefault(top, []).append(w)
    # sort lines top-to-bottom, words within a line left-to-right
    sorted_tops = sorted(lines.keys())
    return [sorted(lines[t], key=lambda w: w["x0"]) for t in sorted_tops]


def extract_with_pdfplumber(pdf_path: str, heading_size_ratio: float = 1.15) -> list[ExtractedElement]:
    """
    Structured extraction: detects headings via font size, extracts tables
    separately, and tags every paragraph with the nearest heading above it.

    Table regions are excluded from paragraph/heading text BEFORE building
    lines — otherwise extract_words() grabs table cell text too, and it gets
    silently duplicated inside the surrounding paragraph.
    """
    filename = os.path.basename(pdf_path)
    elements = []
    current_heading = None
    table_index = 0  # increments across the WHOLE document, not per page, so table_ids stay unique

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            try:
                # find table regions FIRST so we can exclude their words from
                # the paragraph/heading pass below
                found_tables = page.find_tables()
                table_bboxes = [t.bbox for t in found_tables]  # (x0, top, x1, bottom)

                def word_in_table(w):
                    for (x0, top, x1, bottom) in table_bboxes:
                        if w["x0"] >= x0 - 1 and w["x1"] <= x1 + 1 and w["top"] >= top - 1 and w["bottom"] <= bottom + 1:
                            return True
                    return False

                body_size = _guess_body_font_size(page)
                heading_threshold = body_size * heading_size_ratio

                all_words = page.extract_words(extra_attrs=["size"])
                words = [w for w in all_words if not word_in_table(w)]
            except Exception as e:
                print(f"  [warn] pdfplumber failed reading layout on page {page_num} of {filename}: {e} — skipping page")
                continue

            if not all_words:
                print(f"  [warn] page {page_num} of {filename} has no extractable text "
                      f"(likely a scanned page) — skipping, use ocr_ingest.py for this page")
                continue

            lines = _group_words_into_lines(words)

            paragraph_buffer = []
            last_paragraph_text = [None]  # mutable holder so flush_paragraph can update it

            def flush_paragraph():
                if paragraph_buffer:
                    text = " ".join(paragraph_buffer).strip()
                    if text:
                        elements.append(ExtractedElement(
                            text=text,
                            source_filename=filename,
                            page_number=page_num,
                            section_heading=current_heading,
                            element_type=ElementType.PARAGRAPH,
                            element_index=len(elements),
                        ))
                        last_paragraph_text[0] = text  # remember for caption detection below
                    paragraph_buffer.clear()

            for line in lines:
                line_text = " ".join(w["text"] for w in line)
                avg_size = sum(w["size"] for w in line) / len(line)

                if avg_size >= heading_threshold and len(line_text) < 120:
                    # looks like a heading: flush whatever paragraph we were
                    # building, record the heading, and start tracking under it
                    flush_paragraph()
                    current_heading = line_text.strip()
                    elements.append(ExtractedElement(
                        text=current_heading,
                        source_filename=filename,
                        page_number=page_num,
                        section_heading=current_heading,
                        element_type=ElementType.HEADING,
                        element_index=len(elements),
                    ))
                else:
                    paragraph_buffer.append(line_text)

            flush_paragraph()

            # Tables: instead of inlining full content into the main element
            # stream, route each one through the shared table registry.
            # The chunker will only ever see a short pointer element here —
            # the real content (LLM summary + raw table) lives in
            # tables_lookup.json / tables_summary.md, kept separate from
            # everything that gets chunked.
            for table_obj in found_tables:
                table = table_obj.extract()
                if not table:
                    continue
                raw_markdown = "\n".join(
                    " | ".join(cell if cell else "" for cell in row)
                    for row in table
                )
                if not raw_markdown.strip():
                    continue

                table_id = generate_table_id(filename, table_index)
                table_index += 1

                # captions conventionally sit in the paragraph right before
                # the table (e.g. "Table 2.13: ..."), so that's what we check
                caption = detect_caption(last_paragraph_text[0])

                summary = summarize_table_with_llm(raw_markdown)

                register_table(
                    table_id=table_id,
                    source_filename=filename,
                    page_number=page_num,
                    section_heading=current_heading,
                    caption=caption,
                    raw_markdown=raw_markdown,
                    summary=summary,
                )

                placeholder = (
                    f"[{caption or 'Table'} — see table registry, id: {table_id}]"
                )
                elements.append(ExtractedElement(
                    text=placeholder,
                    source_filename=filename,
                    page_number=page_num,
                    section_heading=current_heading,
                    element_type=ElementType.TABLE,
                    element_index=len(elements),
                    table_id=table_id,
                ))

    return elements


# ---------- main entry point used by the rest of the pipeline ----------

def ingest_pdf(pdf_path: str) -> list[ExtractedElement]:
    """The canonical PDF ingestion function the rest of the pipeline calls."""
    return extract_with_pdfplumber(pdf_path)


if __name__ == "__main__":
    import sys as _sys
    from metadata_schema import elements_to_json
    from table_registry import reset_registry, TABLES_LOOKUP_PATH, TABLES_SUMMARY_PATH

    path = _sys.argv[1] if len(_sys.argv) > 1 else "data/raw/sample.pdf"

    # clears tables_lookup.json / tables_summary.md so re-running this
    # script doesn't duplicate table entries. If you're ingesting multiple
    # documents in one pipeline run, call reset_registry() ONCE before the
    # first document, not before each one.

    print(f"Extracting with pypdf (baseline)...")
    pypdf_elements = extract_with_pypdf(path)
    print(f"  -> {len(pypdf_elements)} elements (page-level, no headings/tables)")

    print(f"Extracting with pdfplumber (structured)...")
    plumber_elements = extract_with_pdfplumber(path)
    headings = [e for e in plumber_elements if e.element_type.value == "heading"]
    tables = [e for e in plumber_elements if e.element_type.value == "table"]
    paragraphs = [e for e in plumber_elements if e.element_type.value == "paragraph"]
    print(f"  -> {len(plumber_elements)} elements total")
    print(f"     {len(headings)} headings detected")
    print(f"     {len(paragraphs)} paragraphs")
    print(f"     {len(tables)} table pointers (full content routed to registry)")

    elements_to_json(plumber_elements, "data/extracted/pdf_text.json")
    print("Saved structured output to data/extracted/pdf_text.json")
    if tables:
        print(f"Table registry: {TABLES_LOOKUP_PATH}, {TABLES_SUMMARY_PATH}")