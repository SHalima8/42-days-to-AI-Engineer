# day-17/collection_utils_demo.py

from embeddings.embedder_factory import get_embedder
from vector_stores.chroma_store import ChromaStore
from vector_stores.faiss_store import FaissStore
from vector_stores.store_utils import (
    chroma_list_collections, chroma_delete_chunks, chroma_add_incremental,
    faiss_list_chunk_ids, faiss_delete_chunks, faiss_add_incremental
)

MODEL = "minilm"  # demo utilities against one model — behavior is identical across all 3

print("=== ChromaDB utilities demo ===\n")

chroma_store = ChromaStore(model_name=MODEL)
print("1. list_collections():", chroma_list_collections(chroma_store))
print("2. count():", chroma_store.count())

# --- incremental add ---
embedder = get_embedder(MODEL)
new_text = "This is a demo chunk added incrementally for utility testing."
new_id = "chunk_demo_999"
new_vec = embedder.embed_docs([new_text])
chroma_add_incremental(chroma_store, [new_id], new_vec, [new_text], [{"source": "demo"}])
print("3. after incremental add, count():", chroma_store.count())

# --- delete ---
chroma_delete_chunks(chroma_store, [new_id])
print("4. after delete, count():", chroma_store.count(), "\n")


print("=== FAISS utilities demo ===\n")

faiss_store = FaissStore(model_name=MODEL)
print("1. list_chunk_ids() [first 5]:", faiss_list_chunk_ids(faiss_store)[:5])
print("2. count():", faiss_store.count())

# --- incremental add ---
faiss_add_incremental(faiss_store, [new_id], new_vec, [new_text], [{"source": "demo"}])
print("3. after incremental add, count():", faiss_store.count())

# --- delete ---
faiss_delete_chunks(faiss_store, [new_id])
print("4. after delete, count():", faiss_store.count())
faiss_store.save()  # persist the state back to disk after cleanup