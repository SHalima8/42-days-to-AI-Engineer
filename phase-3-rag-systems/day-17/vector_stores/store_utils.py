# day-17/vector_stores/store_utils.py

# ---------- ChromaDB utilities ----------

def chroma_list_collections(chroma_store):
    """List all collection names that exist in this ChromaDB client."""
    return [c.name for c in chroma_store.client.list_collections()]

def chroma_delete_chunks(chroma_store, chunk_ids):
    """Delete specific chunks by their IDs from a ChromaDB collection."""
    chroma_store.collection.delete(ids=chunk_ids)

def chroma_add_incremental(chroma_store, new_ids, new_embeddings, new_texts, new_metadatas):
    """Add new chunks to an existing ChromaDB collection without touching existing ones."""
    chroma_store.add_chunks(new_ids, new_embeddings, new_texts, new_metadatas)
    # Note: ChromaDB's .add() naturally supports incremental addition —
    # calling it again with new IDs just appends, it doesn't wipe existing data.


# ---------- FAISS utilities ----------

def faiss_list_chunk_ids(faiss_store):
    """List all chunk IDs currently stored in this FAISS index."""
    return faiss_store.chunk_ids

def faiss_delete_chunks(faiss_store, chunk_ids_to_delete):
    """
    FAISS's IndexFlatIP has no native 'delete by id' — the only reliable way
    is to rebuild the index from the chunks we want to KEEP.
    This is a real limitation of FAISS vs ChromaDB, worth noting in findings.md.
    """
    import faiss
    import numpy as np

    keep_positions = [
        i for i, cid in enumerate(faiss_store.chunk_ids)
        if cid not in chunk_ids_to_delete
    ]

    if not keep_positions:
        # nothing left — reset to empty
        faiss_store.index = None
        faiss_store.chunk_ids = []
        faiss_store.texts = []
        faiss_store.metadatas = []
        return

    # Reconstruct vectors for the ones we're keeping
    dim = faiss_store.index.d
    kept_vectors = np.array([
        faiss_store.index.reconstruct(i) for i in keep_positions
    ]).astype("float32")

    new_index = faiss.IndexFlatIP(dim)
    new_index.add(kept_vectors)

    faiss_store.index = new_index
    faiss_store.chunk_ids = [faiss_store.chunk_ids[i] for i in keep_positions]
    faiss_store.texts = [faiss_store.texts[i] for i in keep_positions]
    faiss_store.metadatas = [faiss_store.metadatas[i] for i in keep_positions]

def faiss_add_incremental(faiss_store, new_ids, new_embeddings, new_texts, new_metadatas):
    """Add new chunks to an existing FAISS index without touching existing ones."""
    faiss_store.add_chunks(new_ids, new_embeddings, new_texts, new_metadatas)
    # FAISS's .add() also naturally appends — same incremental behavior as ChromaDB.