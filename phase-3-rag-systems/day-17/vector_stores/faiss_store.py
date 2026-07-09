# day-17/vector_stores/faiss_store.py

import faiss
import numpy as np
import pickle
import os

class FaissStore:
    def __init__(self, model_name, persist_dir="./outputs/faiss_index"):
        self.model_name = model_name
        self.persist_dir = persist_dir
        self.index_path = os.path.join(persist_dir, f"{model_name}.index")
        self.meta_path = os.path.join(persist_dir, f"{model_name}_meta.pkl")

        self.index = None
        self.chunk_ids = []     # position i -> chunk_id
        self.texts = []         # position i -> chunk text
        self.metadatas = []     # position i -> metadata dict

        os.makedirs(persist_dir, exist_ok=True)

        # If a saved index already exists, load it instead of starting empty
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self._load()

    def add_chunks(self, chunk_ids, embeddings, texts, metadatas):
        embeddings = np.array(embeddings).astype("float32")
        faiss.normalize_L2(embeddings)  # required for cosine-equivalent similarity

        if self.index is None:
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)  # IP = inner product (dot product)

        self.index.add(embeddings)
        self.chunk_ids.extend(chunk_ids)
        self.texts.extend(texts)
        self.metadatas.extend(metadatas)

    def query(self, query_embedding, top_k=3):
        query_embedding = np.array(query_embedding).astype("float32").reshape(1, -1)
        faiss.normalize_L2(query_embedding)

        scores, positions = self.index.search(query_embedding, top_k)

        results = []
        for score, pos in zip(scores[0], positions[0]):
            if pos == -1:  # FAISS returns -1 if fewer than top_k items exist
                continue
            results.append((self.chunk_ids[pos], self.texts[pos], float(score)))
        return results

    def count(self):
        return self.index.ntotal if self.index else 0

    def save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump({
                "chunk_ids": self.chunk_ids,
                "texts": self.texts,
                "metadatas": self.metadatas
            }, f)

    def _load(self):
        self.index = faiss.read_index(self.index_path)
        with open(self.meta_path, "rb") as f:
            data = pickle.load(f)
            self.chunk_ids = data["chunk_ids"]
            self.texts = data["texts"]
            self.metadatas = data["metadatas"]