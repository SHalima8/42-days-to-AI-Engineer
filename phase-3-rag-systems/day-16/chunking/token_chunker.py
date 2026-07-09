"""
token_chunker.py

Token-based chunking strategy for Day 16 (Document Ingestion & Chunking Strategies).

Strategy:
    Nearly identical mechanically to fixed_size_chunker.py — same "slice a
    concatenated string into fixed windows with overlap" approach — but
    the unit of measurement is TOKENS, not characters, using tiktoken.

    Why this matters: embedding models and LLMs have context limits
    measured in tokens, not characters. 500 characters can be anywhere
    from ~80 tokens (long words, little punctuation) to ~150+ tokens
    (heavy punctuation, short words) depending on the text — so a
    character-based limit gives inconsistent guarantees about whether a
    chunk actually fits a model's input window. A token-based limit gives
    an exact guarantee.

    Implementation approach:
      1. Concatenate all element text (same offset-tracking trick as
         fixed_size_chunker.py, but tracking TOKEN offsets instead of
         character offsets).
      2. Encode the full text once into a list of token ids.
      3. Slide a fixed-size token window across that list, with overlap.
      4. Decode each token window back into text for the chunk's `text`
         field, and fill in `token_count` (which fixed_size_chunker.py
         deliberately leaves as None, since it never counts tokens).

Usage:
    From the project root:
        python chunking/token_chunker.py data/extracted/pdf_text.json
    Writes to:
        outputs/token_<source>.json
"""

import sys
import json
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import tiktoken

from metadata_schema import (
    ExtractedElement, ElementType, Chunk, ChunkMetadata,
    chunks_to_json, verify_metadata_integrity,
)

# cl100k_base is the encoding used by GPT-3.5/GPT-4-era models and is a
# reasonable, widely-used default for "how many tokens will this cost/fit"
# estimates even when you're not targeting an OpenAI model specifically.
ENCODING_NAME = "cl100k_base"


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


def chunk_by_tokens(
    elements: list[ExtractedElement],
    chunk_size: int = 200,
    overlap: int = 20,
    encoding_name: str = ENCODING_NAME,
) -> list[Chunk]:
    """
    Splits the full text of `elements` into fixed-size TOKEN windows.

    chunk_size: target number of tokens per chunk.
    overlap: number of tokens repeated between consecutive chunks.

    Returns a list[Chunk], each carrying every page_number and
    section_heading its token window overlaps, plus an accurate
    token_count (unlike fixed_size_chunker.py, which leaves this None).
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size, or chunking never advances")
    if not elements:
        return []

    source_filename = elements[0].source_filename
    encoding = tiktoken.get_encoding(encoding_name)

    # Same offset-tracking idea as fixed_size_chunker.py, but in TOKEN
    # space: encode each element's text separately (plus a separator) so
    # we know exactly which token range belongs to which element.
    all_tokens: list[int] = []
    offsets = []  # list of (token_start, token_end, page_number, section_heading)
    separator_tokens = encoding.encode("\n\n")

    for el in elements:
        el_tokens = encoding.encode(el.text)
        start = len(all_tokens)
        all_tokens.extend(el_tokens)
        end = len(all_tokens)
        offsets.append((start, end, el.page_number, el.section_heading))
        all_tokens.extend(separator_tokens)

    chunks: list[Chunk] = []
    chunk_index = 0
    start = 0
    step = chunk_size - overlap

    while start < len(all_tokens):
        end = min(start + chunk_size, len(all_tokens))
        window_tokens = all_tokens[start:end]
        chunk_text = encoding.decode(window_tokens).strip()

        if chunk_text:
            page_numbers = []
            section_headings = []
            for tok_start, tok_end, page_number, section_heading in offsets:
                if tok_start < end and tok_end > start:  # ranges overlap
                    if page_number is not None and page_number not in page_numbers:
                        page_numbers.append(page_number)
                    if section_heading is not None and section_heading not in section_headings:
                        section_headings.append(section_heading)

            metadata = ChunkMetadata(
                source_filename=source_filename,
                chunk_index=chunk_index,
                chunking_strategy="token",
                page_numbers=sorted(page_numbers) if page_numbers else [],
                section_headings=section_headings,
                char_count=len(chunk_text),
                token_count=len(window_tokens),
            )
            chunks.append(Chunk(text=chunk_text, metadata=metadata))
            chunk_index += 1

        if end == len(all_tokens):
            break
        start += step

    return chunks


if __name__ == "__main__":
    IN_PATH = sys.argv[1] if len(sys.argv) > 1 else "data/extracted/pdf_text.json"
    stem = Path(IN_PATH).stem
    OUT_PATH = f"outputs/token_{stem}.json"

    elements = _load_elements(IN_PATH)
    chunks = chunk_by_tokens(elements, chunk_size=200, overlap=20)

    problems = verify_metadata_integrity(chunks)
    if problems:
        print(f"[token_chunker] METADATA PROBLEMS FOUND:")
        for p in problems:
            print(f"  - {p}")
    else:
        print(f"[token_chunker] Metadata integrity check passed ({len(chunks)} chunks)")

    Path("outputs").mkdir(exist_ok=True)
    chunks_to_json(chunks, OUT_PATH)
    print(f"[token_chunker] Wrote {OUT_PATH}")