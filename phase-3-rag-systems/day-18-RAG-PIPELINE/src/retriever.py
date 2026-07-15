"""
retriever.py
Embeds the query with your Day 17 embedder (minilm), queries the existing
ChromaDB collection, and — critically — resolves any TABLE-type chunk's
placeholder text into real content via table_lookup.py before returning.

Without that resolution step, a retrieved table chunk would just hand the
LLM a useless string like "[Table 4.1 — see table registry, id: ...]".
"""

import sys
import os
import re
import chromadb

# Day 17's embedder_factory.py lives in its own folder, not this project —
# add it to sys.path rather than duplicating the file (same pattern your
# pdf_ingest.py already uses for metadata_schema/table_registry).
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "day-17", "embeddings"))
from embedder_factory import get_embedder  # noqa: E402

from src import config
from src.table_lookup import get_table_block, get_table_summary_only

# Matches a leaked table placeholder wherever it appears, e.g.:
# "[Table 4.2 — see table registry, id: multilingual_speech_pipeline_table_0001]"
# Captures the table_id so it can be resolved even when it's stuck mid-paragraph
# inside a chunk that was never tagged element_type="table" (Case 2 bug — the
# Day 16 chunker sometimes merges a table pointer into surrounding text
# instead of keeping it isolated).
_PLACEHOLDER_PATTERN = re.compile(r"\[[^\[\]]*?id:\s*([\w\-]+)\]")


def _resolve_leaked_placeholders(text: str) -> str:
    """
    Defensive patch, not a root-cause fix: scans ANY chunk's text (not just
    ones already tagged element_type="table") for a stray placeholder and
    swaps it for the real summary. Root cause is in Day 16 ingestion — this
    just stops junk placeholder text from ever reaching the LLM regardless
    of which chunk it hides in.
    """
    def _replace(match):
        table_id = match.group(1)
        return get_table_summary_only(table_id)

    return _PLACEHOLDER_PATTERN.sub(_replace, text)

_embedder = get_embedder(config.EMBEDDING_MODEL_NAME)
_chroma_client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
_collection = _chroma_client.get_collection(config.COLLECTION_NAME)


def retrieve(query: str, top_k: int = None) -> list[dict]:
    """
    Returns ranked chunks: {"text", "source", "section_heading", "distance",
    "confidence", "element_type"}.

    If a retrieved chunk is a table pointer, "text" is replaced with the
    resolved table content (caption + summary + raw markdown) pulled from
    tables_summary.md — everything downstream (prompt_builder) just sees
    real content, never the placeholder.
    """
    top_k = top_k or config.TOP_K
    query_vector = _embedder.embed_query(query)

    results = _collection.query(
        query_embeddings=[query_vector.tolist()],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for text, meta, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        element_type = meta.get("element_type", "paragraph")

        if element_type == "table" and meta.get("table_id"):
            text = get_table_block(meta["table_id"])
        else:
            # Defensive check: even a chunk tagged "paragraph" might have a
            # table placeholder leaked into it (Case 2). Resolve it inline
            # if found, otherwise this is a no-op.
            text = _resolve_leaked_placeholders(text)

        chunks.append({
            "text": text,
            "source": meta.get("source_filename", meta.get("source", "unknown")),
            "section_heading": meta.get("section_heading"),
            "element_type": element_type,
            "distance": distance,
            "confidence": round(max(0.0, 1 - (distance / 2)), 3),
        })

    return chunks


if __name__ == "__main__":
    test_query = "What are the error types for Pashto?"
    for i, chunk in enumerate(retrieve(test_query), 1):
        print(f"\n[{i}] type={chunk['element_type']} confidence={chunk['confidence']} source={chunk['source']}")
        print(chunk["text"][:200], "...")