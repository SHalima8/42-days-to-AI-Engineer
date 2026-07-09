# scratch_test_utils.py — run this fresh, from inside day-17/

from embeddings.embedder_factory import get_embedder
from vector_stores.chroma_store import ChromaStore
from vector_stores.faiss_store import FaissStore
from vector_stores.store_utils import chroma_list_collections, faiss_list_chunk_ids

embedder = get_embedder("minilm")

# --- Create/reuse the Chroma store and check it ---
chroma_store = ChromaStore(model_name="minilm")
print("Chroma collections:", chroma_list_collections(chroma_store))

# --- Create/reuse the FAISS store and check it ---
faiss_store = FaissStore(model_name="minilm")
print("FAISS chunk ids:", faiss_list_chunk_ids(faiss_store))