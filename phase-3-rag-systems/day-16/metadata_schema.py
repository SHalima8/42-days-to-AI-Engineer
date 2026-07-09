"""
metadata_schema.py

Shared data structures used by every ingestion module (pdf_ingest, docx_ingest,
txt_ingest, ocr_ingest) and every chunker (fixed_size, token, recursive,
semantic, hierarchical).

Defining this once, up front, is what makes "no chunk should lose its
metadata" actually checkable later — every module speaks the same shape.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional
import json


class ElementType(str, Enum):
    """What kind of content a piece of extracted text represents."""
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    LIST_ITEM = "list_item"
    OCR_TEXT = "ocr_text"  # OCR output, since we can't always tell heading vs paragraph from OCR alone


@dataclass
class ExtractedElement:
    """
    One unit of content produced by an INGESTION module.
    A document becomes a list[ExtractedElement], not one big string.
    """
    text: str
    source_filename: str
    page_number: Optional[int]          # None for DOCX/TXT, which don't have "pages" the same way
    section_heading: Optional[str]      # nearest heading this element falls under, if any
    element_type: ElementType
    element_index: int                  # position of this element within the document, in order
    table_id: Optional[str] = None      # set ONLY for TABLE elements: the full table content lives
                                         # in table_registry's tables_summary.md, not here — this
                                         # element just points to it, so chunkers never see raw table text

    def to_dict(self) -> dict:
        d = asdict(self)
        d["element_type"] = self.element_type.value
        return d


@dataclass
class ChunkMetadata:
    """
    Metadata attached to one CHUNK, produced by a chunking module.
    A chunk can span multiple pages/elements, so page_numbers and
    section_headings are lists, not single values.
    """
    source_filename: str
    chunk_index: int                    # position of this chunk within the full chunk sequence
    chunking_strategy: str               # "fixed_size" | "token" | "recursive" | "semantic" | "hierarchical"
    page_numbers: list                  # e.g. [3] or [3, 4] if the chunk spans a page break; [] if not applicable
    section_headings: list              # every distinct heading the chunk's source elements fell under
    char_count: int
    token_count: Optional[int] = None   # filled in by token-aware chunkers; left None otherwise


@dataclass
class Chunk:
    """A chunk of text ready for embedding, paired with its metadata."""
    text: str
    metadata: ChunkMetadata

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "metadata": asdict(self.metadata),
        }


def elements_to_json(elements: list[ExtractedElement], path: str) -> None:
    """Dump ingestion output to disk for inspection before chunking."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump([e.to_dict() for e in elements], f, ensure_ascii=False, indent=2)


def chunks_to_json(chunks: list[Chunk], path: str) -> None:
    """Dump chunking output to disk for inspection/comparison."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump([c.to_dict() for c in chunks], f, ensure_ascii=False, indent=2)


def verify_metadata_integrity(chunks: list[Chunk]) -> list[str]:
    """
    Checks every chunk has a non-empty source_filename and valid chunk_index.
    Returns a list of problem descriptions — empty list means everything passed.
    This is the function your Day 16 "verify no chunk loses its source
    reference" requirement maps directly onto.
    """
    problems = []
    for i, c in enumerate(chunks):
        m = c.metadata
        if not m.source_filename:
            problems.append(f"Chunk {i}: missing source_filename")
        if m.chunk_index is None:
            problems.append(f"Chunk {i}: missing chunk_index")
        if not c.text or not c.text.strip():
            problems.append(f"Chunk {i}: empty text")
        if m.char_count != len(c.text):
            problems.append(f"Chunk {i}: char_count ({m.char_count}) doesn't match actual text length ({len(c.text)})")
    return problems