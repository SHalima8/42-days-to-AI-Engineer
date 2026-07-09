# day-17/ingest_chroma.py

import json
from embeddings.embedder_factory import get_embedder
from vector_stores.chroma_store import ChromaStore

CHUNKS_PATH = "../day-16/outputs/recursive_docx_text.json"
MODELS_TO_RUN = ["minilm", "mpnet", "bge"]


def load_chunks(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def sanitize_metadata(raw_meta):
    """
    ChromaDB only accepts str/int/float/bool as metadata values —
    no lists, no None. Convert anything else into a safe equivalent.
    """
    clean = {}
    for key, value in raw_meta.items():
        if value is None:
            clean[key] = ""  # empty string instead of None
        elif isinstance(value, list):
            clean[key] = ", ".join(str(v) for v in value)  # list -> comma-joined string
        else:
            clean[key] = value  # already a safe type (str, int, float, bool)
    return clean


def ingest_for_model(model_name, chunks):
    print(f"\n--- Ingesting into ChromaDB using model: {model_name} ---")
    embedder = get_embedder(model_name)
    store = ChromaStore(model_name=model_name)

    texts = [c["text"] for c in chunks]
    ids = [f"chunk_{c['metadata']['chunk_index']}" for c in chunks]
    metadatas = [sanitize_metadata(c["metadata"]) for c in chunks]

    embeddings = embedder.embed_docs(texts)
    store.add_chunks(ids, embeddings, texts, metadatas)

    print(f"Done. Collection now has {store.count()} chunks.")


if __name__ == "__main__":
    chunks = load_chunks(CHUNKS_PATH)
    print(f"Loaded {len(chunks)} chunks from {CHUNKS_PATH}")

    for model_name in MODELS_TO_RUN:
        ingest_for_model(model_name, chunks)