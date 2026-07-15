"""
Day 19 — BM25 Retrieval Pipeline
Loads chunks from Day 16's recursive chunking output and builds a BM25
sparse retriever. Designed to run side-by-side with the Day 18 vector
retriever so results can be compared on the same 20 test questions.

Install:
    pip install rank-bm25
"""

import json
import re
import string
from pathlib import Path
from rank_bm25 import BM25Okapi

CHUNKS_PATH = "../day-16/outputs/recursive_docx_text.json"


def load_chunks(path: str) -> list[dict]:
    """Load chunk records produced by the Day 16 recursive chunker."""
    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return chunks


def tokenize(text: str) -> list[str]:
    """
    Lowercase + strip punctuation + whitespace split.
    Stopwords are intentionally kept — BM25's IDF term already
    down-weights common words, so removing them adds complexity
    without a clear precision gain (same call made in SmartDocAI).
    """
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()


class BM25Retriever:
    """Thin wrapper so this has the same call shape as a vector retriever
    (query in, ranked chunks + scores out) — makes RRF fusion later easier."""

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.tokenized_corpus = [tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        tokenized_query = tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # pair each chunk with its score, sort descending
        scored = list(zip(self.chunks, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        results = []
        for rank, (chunk, score) in enumerate(scored[:top_k], start=1):
            results.append({
                "rank": rank,
                "score": float(score),
                "text": chunk["text"],
                "source_filename": chunk["metadata"]["source_filename"],
                "chunk_index": chunk["metadata"]["chunk_index"],
                "section_headings": chunk["metadata"]["section_headings"],
            })
        return results


def preview_results(query: str, results: list[dict]) -> None:
    print(f"\nQuery: {query}")
    print("-" * 60)
    for r in results:
        print(f"[{r['rank']}] score={r['score']:.3f} "
              f"({r['source_filename']} #{r['chunk_index']})")
        print(f"    {r['text'][:150].strip()}...")


if __name__ == "__main__":
    chunks = load_chunks(CHUNKS_PATH)
    print(f"Loaded {len(chunks)} chunks from {CHUNKS_PATH}")

    retriever = BM25Retriever(chunks)

    # Sanity-check query — replace with your real 20 test questions
    # once you point me to where they live
    test_query = "Which subteam would need to be involved if a change to the shared front end caused a repeat of the Q2 latency incident"
    results = retriever.retrieve(test_query, top_k=5)
    preview_results(test_query, results)