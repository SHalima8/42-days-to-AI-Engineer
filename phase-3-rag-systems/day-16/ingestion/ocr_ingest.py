"""
ocr_ingest.py (v3 — docling-based, tables routed through table_registry)

OCR ingestion module for Day 16 (Document Ingestion & Chunking Strategies).

WHY THIS VERSION REPLACES THE PLAIN pytesseract APPROACH:
    The original version of this module (pytesseract + pdf2image) treated
    each scanned page as one flat blob of OCR text. That's fine for prose,
    but it's actively BAD for tables: Tesseract has no concept of rows/
    columns, so it reads cell text in raw left-to-right, top-to-bottom
    pixel order. On a real multi-column table, this interleaves cells from
    different columns and produces garbled nonsense — e.g. a 4-column
    error-rate table came out as fragments like "Dialectal pronunciation
    variance\\nast\\nspeech / phoneme dropping\\napes topes", which is
    unusable for retrieval or for a human reader.

    This version uses docling instead, which runs actual OCR (same
    underlying idea as pytesseract) but ALSO runs a trained table-
    structure-recognition model that detects table boundaries and
    reconstructs true row/column structure — even on scanned/image-only
    pages, where there's no native text layer to hint at columns.

WHY THIS VERSION (v3) CHANGES AGAIN FROM v2:
    v2 inlined each table's LLM summary (or markdown fallback) directly
    into the main element stream, same mistake pdf_ingest.py originally
    made — a table still ended up as one big blob of text sitting right
    next to paragraphs, which a chunker could still split or absorb into
    surrounding prose.

    v3 routes every detected table through table_registry.py instead —
    the SAME shared module pdf_ingest.py and docx_ingest.py now use. This
    means:
      - one summarization implementation shared across all three ingestion
        modules, instead of a duplicate copy living in this file
      - the table's real content (summary + raw markdown) lives in
        tables_summary.md / tables_lookup.json, not in ocr_text.json
      - the element left behind here is just a short pointer, so a
        chunker running on ocr_text.json never sees raw table content

Requires:
    pip install docling
    (Optional, for LLM table summarization) pip install google-genai
    (Optional) set GEMINI_API_KEY environment variable

Note: docling downloads its layout + table-structure models from Hugging
Face on first use (a few hundred MB, cached afterward). Requires normal
internet access the first time you run this.

Usage:
    From the project root:
        python ingestion/ocr_ingest.py data/raw/scanned_document.pdf
    Writes to:
        data/extracted/ocr_text.json
"""

import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc.document import TableItem, TextItem

from metadata_schema import ExtractedElement, ElementType, elements_to_json
from table_registry import (
    detect_caption, generate_table_id, summarize_table_with_llm, register_table,
)


def _build_converter() -> DocumentConverter:
    """
    Configures docling to run OCR (for the scanned/image page text) AND
    table-structure recognition (so tables come back as real rows/columns,
    not garbled OCR text).
    """
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )


def extract_scanned_pdf(path: str) -> list[ExtractedElement]:
    """
    Extracts a scanned/image-only PDF via docling (OCR + table structure
    recognition) into an ordered list of ExtractedElement.

    Text content -> ElementType.OCR_TEXT (OCR text has no reliable heading/
    paragraph structure, so we don't pretend to detect it — same reasoning
    as before).
    Table content -> routed through table_registry.py, same as
    pdf_ingest.py and docx_ingest.py. The element left behind here is a
    short pointer (table_id + placeholder), never the full table.
    """
    source_filename = Path(path).name
    converter = _build_converter()

    result = converter.convert(path)
    doc = result.document

    elements: list[ExtractedElement] = []
    element_index = 0
    table_index = 0
    last_text = None  # most recent OCR_TEXT seen, used for caption detection

    for item, _level in doc.iterate_items():
        page_no = item.prov[0].page_no if item.prov else None

        if isinstance(item, TableItem):
            raw_markdown = item.export_to_markdown(doc=doc)
            if not raw_markdown.strip():
                continue

            table_id = generate_table_id(source_filename, table_index)
            table_index += 1

            # captions conventionally sit in the text immediately before a
            # table (e.g. "Table 2.13: ..."); on a scanned page this is the
            # most recent OCR_TEXT item, which is imperfect (OCR errors can
            # corrupt the caption text itself) but the best signal available
            caption = detect_caption(last_text)
            summary = summarize_table_with_llm(raw_markdown)

            register_table(
                table_id=table_id,
                source_filename=source_filename,
                page_number=page_no,
                section_heading=None,  # OCR has no heading context to attach
                caption=caption,
                raw_markdown=raw_markdown,
                summary=summary,
            )

            placeholder = f"[{caption or 'Table'} — see table registry, id: {table_id}]"
            elements.append(ExtractedElement(
                text=placeholder,
                source_filename=source_filename,
                page_number=page_no,
                section_heading=None,
                element_type=ElementType.TABLE,
                element_index=element_index,
                table_id=table_id,
            ))
            element_index += 1

        elif isinstance(item, TextItem):
            text = item.text.strip()
            if not text:
                continue
            elements.append(ExtractedElement(
                text=text,
                source_filename=source_filename,
                page_number=page_no,
                section_heading=None,
                element_type=ElementType.OCR_TEXT,
                element_index=element_index,
            ))
            element_index += 1
            last_text = text  # remember for the next table's caption detection

    return elements


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from table_registry import TABLES_LOOKUP_PATH, TABLES_SUMMARY_PATH

    # NOTE: does NOT call reset_registry() here — same reasoning as
    # docx_ingest.py. Reset once at the start of a full multi-document
    # pipeline run, not inside each individual ingestion script.

    RAW_PATH = sys.argv[1] if len(sys.argv) > 1 else "data/raw/scanned_sample.pdf"
    OUT_PATH = "data/extracted/ocr_text.json"

    elements = extract_scanned_pdf(RAW_PATH)
    ocr_text_count = sum(1 for e in elements if e.element_type.value == "ocr_text")
    tables = [e for e in elements if e.element_type.value == "table"]

    print(f"[ocr_ingest] Extracted {len(elements)} element(s) from {RAW_PATH}")
    print(f"  {ocr_text_count} OCR text elements")
    print(f"  {len(tables)} table pointers (full content routed to registry)")

    Path("data/extracted").mkdir(parents=True, exist_ok=True)
    elements_to_json(elements, OUT_PATH)
    print(f"[ocr_ingest] Wrote {OUT_PATH}")
    if tables:
        print(f"Table registry: {TABLES_LOOKUP_PATH}, {TABLES_SUMMARY_PATH}")