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
import chromadb

# Day 17's embedder_factory.py lives in its own folder, not this project —
# add it to sys.path rather than duplicating the file (same pattern your
# pdf_ingest.py already uses for metadata_schema/table_registry).
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "day-17", "embeddings"))
from embedder_factory import get_embedder  # noqa: E402

from src import config
from src.table_lookup import get_table_block

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
    test_query = "What GPU utilization percentage did the Sindhi model achieve last quarter?"
    for i, chunk in enumerate(retrieve(test_query), 1):
        print(f"\n[{i}] type={chunk['element_type']} confidence={chunk['confidence']} source={chunk['source']}")
        print(chunk["text"][:200], "...")