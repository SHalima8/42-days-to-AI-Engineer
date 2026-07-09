# day-17/ingest_faiss.py

import json
from embeddings.embedder_factory import get_embedder
from vector_stores.faiss_store import FaissStore

CHUNKS_PATH = "../day-16/outputs/recursive_docx_text.json"
MODELS_TO_RUN = ["minilm", "mpnet", "bge"]


def load_chunks(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def sanitize_metadata(raw_meta):
    """Same sanitization as Chroma — keeps metadata consistent across both stores,
    even though FAISS itself doesn't enforce type restrictions like Chroma does."""
    clean = {}
    for key, value in raw_meta.items():
        if value is None:
            clean[key] = ""
        elif isinstance(value, list):
            clean[key] = ", ".join(str(v) for v in value)
        else:
            clean[key] = value
    return clean


def ingest_for_model(model_name, chunks):
    print(f"\n--- Ingesting into FAISS using model: {model_name} ---")
    embedder = get_embedder(model_name)
    store = FaissStore(model_name=model_name)

    texts = [c["text"] for c in chunks]
    ids = [f"chunk_{c['metadata']['chunk_index']}" for c in chunks]
    metadatas = [sanitize_metadata(c["metadata"]) for c in chunks]

    embeddings = embedder.embed_docs(texts)
    store.add_chunks(ids, embeddings, texts, metadatas)
    store.save()  # persist to disk — this step was missing/unclear in our earlier dummy test

    print(f"Done. Index now has {store.count()} chunks.")


if __name__ == "__main__":
    chunks = load_chunks(CHUNKS_PATH)
    print(f"Loaded {len(chunks)} chunks from {CHUNKS_PATH}")

    for model_name in MODELS_TO_RUN:
        ingest_for_model(model_name, chunks)