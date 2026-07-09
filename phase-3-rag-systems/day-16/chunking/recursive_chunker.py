"""
recursive_chunker.py

Recursive chunking strategy for Day 16 (Document Ingestion & Chunking Strategies).

Strategy:
    Uses LangChain's RecursiveCharacterTextSplitter, which tries a priority
    list of separators — by default ["\\n\\n", "\\n", " ", ""] — attempting
    to split on the MOST semantically meaningful boundary first (paragraph
    breaks), and only falling back to cruder ones (single newlines, then
    spaces, then raw characters) if a piece is still too big after trying
    the gentler separator.

    Net effect vs. fixed_size_chunker.py / token_chunker.py: chunk
    boundaries land on paragraph/sentence/word edges far more often,
    instead of slicing blindly through the middle of a word. The
    trade-off: chunk sizes become UNEVEN — the splitter stops as soon as a
    piece is under the size limit, so some chunks are noticeably smaller
    than the target while others sit right at the limit.

    Metadata mapping challenge:
        LangChain's splitter operates on plain text and returns a list of
        chunk strings — it does NOT tell you which character offsets each
        chunk came from. To recover page_numbers/section_headings per
        chunk (same requirement as every other chunker here), we locate
        each returned chunk's start offset in the original concatenated
        text via str.find(), searching forward from the previous chunk's
        start position (chunks are emitted in order, and even with overlap
        a later chunk's start position is never earlier than the previous
        chunk's start).

Usage:
    From the project root:
        python chunking/recursive_chunker.py data/extracted/pdf_text.json
    Writes to:
        outputs/recursive_<source>.json
"""

import sys
import json
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from langchain_text_splitters import RecursiveCharacterTextSplitter

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


def chunk_recursive(
    elements: list[ExtractedElement],
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[Chunk]:
    """
    Splits the full text of `elements` using LangChain's
    RecursiveCharacterTextSplitter (paragraph -> line -> word -> char
    fallback), then maps each resulting chunk back to the page_numbers and
    section_headings it overlaps.
    """
    if not elements:
        return []

    source_filename = elements[0].source_filename
    separator = "\n\n"

    # Same offset-tracking approach as fixed_size_chunker.py: build one
    # concatenated string, and remember each element's (start, end) range
    # within it so any later character window can be mapped back to the
    # elements (and their page/heading metadata) it overlaps.
    full_text_parts = []
    offsets = []  # list of (start, end, page_number, section_heading)
    cursor = 0

    for el in elements:
        start = cursor
        full_text_parts.append(el.text)
        cursor += len(el.text)
        end = cursor
        offsets.append((start, end, el.page_number, el.section_heading))

        full_text_parts.append(separator)
        cursor += len(separator)

    full_text = "".join(full_text_parts)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    raw_chunks = splitter.split_text(full_text)

    chunks: list[Chunk] = []
    search_from = 0

    for chunk_index, chunk_text in enumerate(raw_chunks):
        if not chunk_text.strip():
            continue

        # Locate this chunk's position in the original text. Search
        # forward from search_from (not from the previous chunk's END,
        # since overlap means the next chunk can start BEFORE the
        # previous one ends).
        found_at = full_text.find(chunk_text, search_from)
        if found_at == -1:
            # Fallback: the splitter can occasionally strip/alter
            # whitespace slightly, so retry from the very start if the
            # forward-only search fails.
            found_at = full_text.find(chunk_text)
        if found_at == -1:
            # Truly can't locate it (rare) — still emit the chunk, just
            # without positional metadata, rather than dropping the text.
            chunk_start, chunk_end = None, None
        else:
            chunk_start = found_at
            chunk_end = found_at + len(chunk_text)
            search_from = chunk_start + 1  # allow overlap on the next search

        page_numbers = []
        section_headings = []
        if chunk_start is not None:
            for el_start, el_end, page_number, section_heading in offsets:
                if el_start < chunk_end and el_end > chunk_start:
                    if page_number is not None and page_number not in page_numbers:
                        page_numbers.append(page_number)
                    if section_heading is not None and section_heading not in section_headings:
                        section_headings.append(section_heading)

        metadata = ChunkMetadata(
            source_filename=source_filename,
            chunk_index=chunk_index,
            chunking_strategy="recursive",
            page_numbers=sorted(page_numbers) if page_numbers else [],
            section_headings=section_headings,
            char_count=len(chunk_text),
        )
        chunks.append(Chunk(text=chunk_text, metadata=metadata))

    return chunks


if __name__ == "__main__":
    IN_PATH = sys.argv[1] if len(sys.argv) > 1 else "data/extracted/pdf_text.json"
    stem = Path(IN_PATH).stem
    OUT_PATH = f"outputs/recursive_{stem}.json"

    elements = _load_elements(IN_PATH)
    chunks = chunk_recursive(elements, chunk_size=500, overlap=50)

    problems = verify_metadata_integrity(chunks)
    if problems:
        print(f"[recursive_chunker] METADATA PROBLEMS FOUND:")
        for p in problems:
            print(f"  - {p}")
    else:
        print(f"[recursive_chunker] Metadata integrity check passed ({len(chunks)} chunks)")

    Path("outputs").mkdir(exist_ok=True)
    chunks_to_json(chunks, OUT_PATH)
    print(f"[recursive_chunker] Wrote {OUT_PATH}")