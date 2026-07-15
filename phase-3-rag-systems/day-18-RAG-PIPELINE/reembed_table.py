"""
reembed_tables.py
ONE-TIME fix script for the Case 1 failure (table retrieval miss).

Problem: table chunks in chunks_minilm were originally embedded from their
placeholder string ("[Table 4.1 -- see table registry, id: ...]"), which is
semantically thin and loses the retrieval race against prose that mentions
the same facts in words. The table's REAL content (the summary + values)
was never what got embedded.

Fix: read tables_lookup.json for the list of tables, pull each one's real
summary text from tables_summary.md (via table_lookup.py), embed THAT text
instead, and upsert it into the collection -- replacing any old
placeholder-based chunk for the same table_id so we don't end up with two
competing versions of the same table.

Run this once after ingestion changes, or any time tables_summary.md is
regenerated. Safe to re-run: it deletes-then-adds per table_id, so it
won't duplicate entries on repeated runs.
"""

import sys
import os
import json

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "day-17", "embeddings"))
from embedder_factory import get_embedder  # noqa: E402

import chromadb
from src import config
from src.table_lookup import get_table_summary_only


def load_table_metadata() -> list[dict]:
    with open(config.TABLES_LOOKUP_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    embedder = get_embedder(config.EMBEDDING_MODEL_NAME)
    client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
    collection = client.get_or_create_collection(config.COLLECTION_NAME)

    tables = load_table_metadata()
    print(f"Found {len(tables)} tables in {config.TABLES_LOOKUP_PATH}")

    for t in tables:
        table_id = t["table_id"]

        # Pull the REAL summary text (not the placeholder) to embed.
        # Prepending caption + section gives the embedding extra semantic
        # anchoring (e.g. "Table 4.1" + "Model Evaluation") on top of the
        # summary content itself.
        summary_text = get_table_summary_only(table_id)
        embedding_text = (
            f"{t.get('caption', '')} ({t.get('section_heading', '')}): {summary_text}"
        )

        vector = embedder.embed_docs([embedding_text])[0]

        # Remove any existing chunk(s) for this table_id first (e.g. the
        # original placeholder-embedded entry from ingestion) so we don't
        # end up with two competing versions of the same table.
        existing = collection.get(where={"table_id": table_id})
        if existing["ids"]:
            collection.delete(ids=existing["ids"])
            print(f"  Removed {len(existing['ids'])} old chunk(s) for {table_id}")

        collection.upsert(
            ids=[f"reembedded_{table_id}"],
            embeddings=[vector.tolist()],
            documents=[summary_text],
            metadatas=[{
                "element_type": "table",
                "table_id": table_id,
                "source_filename": t.get("source_filename", "unknown"),
                "section_heading": t.get("section_heading"),
                "caption": t.get("caption"),
            }],
        )
        print(f"  Re-embedded {table_id} using real summary text")

    print(f"\nDone. {len(tables)} table(s) re-embedded in '{config.COLLECTION_NAME}'.")


if __name__ == "__main__":
    main()