"""
fixed_size_chunker.py

Fixed-size chunking strategy for Day 16 (Document Ingestion & Chunking Strategies).

Strategy:
    The simplest possible chunker: concatenate all of a document's extracted
    elements into one continuous string, then slice it into fixed-length
    character windows with a configurable overlap between consecutive
    chunks. No awareness of sentences, words, or structure — a chunk
    boundary can (and will) land in the middle of a word.

    Because a fixed-size window can span MULTIPLE ExtractedElements (e.g.
    the tail of one paragraph + the start of the next, or even across a
    page break), this module tracks, for every chunk, the FULL SET of
    page_numbers and section_headings touched by that window — not just
    a single value — matching ChunkMetadata's list-typed fields exactly
    for this reason.

Usage:
    From the project root:
        python chunking/fixed_size_chunker.py data/extracted/pdf_text.json
    Writes to:
        outputs/fixed_size_<source>.json  (or wherever you point OUT_PATH)
"""

import sys
import json
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from metadata_schema import (
    ExtractedElement, ElementType, Chunk, ChunkMetadata,
    chunks_to_json, verify_metadata_integrity,
)


def _load_elements(path: str) -> list[ExtractedElement]:
    """Reloads ExtractedElement objects from an ingestion module's JSON output."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    elements = []
    for item in raw:
        item = dict(item)
        item["element_type"] = ElementType(item["element_type"])
        elements.append(ExtractedElement(**item))
    return elements


def chunk_fixed_size(
    elements: list[ExtractedElement],
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[Chunk]:
    """
    Splits the full text of `elements` into fixed-size character windows.

    chunk_size: target number of characters per chunk.
    overlap: number of characters repeated between consecutive chunks, so
             a sentence cut off at the end of one chunk still appears
             (with its preceding context) at the start of the next.

    Returns a list[Chunk], each carrying every page_number and
    section_heading its character window overlaps.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size, or chunking never advances")
    if not elements:
        return []

    source_filename = elements[0].source_filename

    # Concatenate all element text into one string, tracking each element's
    # (start_offset, end_offset) in that concatenated string so we can later
    # map any character window back to the elements it overlaps.
    full_text_parts = []
    offsets = []  # list of (start, end, page_number, section_heading)
    cursor = 0
    separator = "\n\n"

    for el in elements:
        start = cursor
        full_text_parts.append(el.text)
        cursor += len(el.text)
        end = cursor
        offsets.append((start, end, el.page_number, el.section_heading))

        full_text_parts.append(separator)
        cursor += len(separator)

    full_text = "".join(full_text_parts)

    chunks: list[Chunk] = []
    chunk_index = 0
    start = 0
    step = chunk_size - overlap

    while start < len(full_text):
        end = min(start + chunk_size, len(full_text))
        chunk_text = full_text[start:end]

        if chunk_text.strip():
            # Find every element whose offset range overlaps [start, end)
            page_numbers = []
            section_headings = []
            for el_start, el_end, page_number, section_heading in offsets:
                if el_start < end and el_end > start:  # ranges overlap
                    if page_number is not None and page_number not in page_numbers:
                        page_numbers.append(page_number)
                    if section_heading is not None and section_heading not in section_headings:
                        section_headings.append(section_heading)

            metadata = ChunkMetadata(
                source_filename=source_filename,
                chunk_index=chunk_index,
                chunking_strategy="fixed_size",
                page_numbers=sorted(page_numbers) if page_numbers else [],
                section_headings=section_headings,
                char_count=len(chunk_text),
            )
            chunks.append(Chunk(text=chunk_text, metadata=metadata))
            chunk_index += 1

        if end == len(full_text):
            break
        start += step

    return chunks


if __name__ == "__main__":
    IN_PATH = sys.argv[1] if len(sys.argv) > 1 else "data/extracted/pdf_text.json"
    stem = Path(IN_PATH).stem  # e.g. "pdf_text"
    OUT_PATH = f"outputs/fixed_size_{stem}.json"

    elements = _load_elements(IN_PATH)
    chunks = chunk_fixed_size(elements, chunk_size=500, overlap=50)

    problems = verify_metadata_integrity(chunks)
    if problems:
        print(f"[fixed_size_chunker] METADATA PROBLEMS FOUND:")
        for p in problems:
            print(f"  - {p}")
    else:
        print(f"[fixed_size_chunker] Metadata integrity check passed ({len(chunks)} chunks)")

    Path("outputs").mkdir(exist_ok=True)
    chunks_to_json(chunks, OUT_PATH)
    print(f"[fixed_size_chunker] Wrote {OUT_PATH}")