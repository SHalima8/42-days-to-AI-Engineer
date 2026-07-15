"""
Day 19 — Chunk Index Finder (one-off helper, not part of the pipeline)
Backfills expected_chunk_index in test_questions.py. Give it a keyword or
short phrase from the expected answer, and it prints every chunk that
contains it — copy the chunk_index of the right one into test_questions.py.

Usage:
    python find_chunk.py "twelve"
    python find_chunk.py "eighteen months"
"""

import sys
from bm25_retrieval import CHUNKS_PATH, load_chunks


def find_chunks(keyword: str) -> None:
    chunks = load_chunks(CHUNKS_PATH)
    keyword_lower = keyword.lower()
    matches = [c for c in chunks if keyword_lower in c["text"].lower()]

    if not matches:
        print(f"No chunks contain '{keyword}' — try a shorter/simpler phrase.")
        return

    for c in matches:
        meta = c["metadata"]
        print(f"\n{meta['source_filename']} #{meta['chunk_index']}")
        print(f"  {c['text'][:200].strip()}...")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python find_chunk.py "keyword or phrase"')
        sys.exit(1)
    find_chunks(" ".join(sys.argv[1:]))