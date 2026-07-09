# day-17/vector_stores/chroma_store.py

import chromadb

class ChromaStore:
    def __init__(self, model_name, persist_path="./outputs/chroma_db"):
        # One separate collection per embedding model — vectors from
        # different models are NOT comparable and must never mix.
        self.model_name = model_name
        self.client = chromadb.PersistentClient(path=persist_path)
        self.collection = self.client.get_or_create_collection(
            name=f"chunks_{model_name}",
            metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunk_ids, embeddings, texts, metadatas):
        self.collection.add(
            ids=chunk_ids,
            embeddings=[e.tolist() if hasattr(e, "tolist") else e for e in embeddings],
            documents=texts,
            metadatas=metadatas
        )

    def query(self, query_embedding, top_k=3):
        result = self.collection.query(
            query_embeddings=[query_embedding.tolist() if hasattr(query_embedding, "tolist") else query_embedding],
            n_results=top_k
        )
        # Returns list of (chunk_id, document_text, distance) tuples
        ids = result["ids"][0]
        docs = result["documents"][0]
        distances = result["distances"][0]
        return list(zip(ids, docs, distances))

    def count(self):
        return self.collection.count()