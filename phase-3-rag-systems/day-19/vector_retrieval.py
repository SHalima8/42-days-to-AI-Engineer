"""
Day 19 — Vector Retrieval Pipeline
Wraps a persistent ChromaDB collection (MiniLM embeddings) with the same
call shape as BM25Retriever: .retrieve(query, top_k) -> list[dict].
If the collection doesn't exist yet, builds it from the same Day 16
chunks JSON used by bm25_retrieval.py.

Install:
    pip install chromadb sentence-transformers
"""

import json
import chromadb
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = "../day-16/outputs/recursive_docx_text.json"
PERSIST_DIR = "../day-17/outputs/chroma_db"          # adjust if Day 18 already has one
COLLECTION_NAME = "chunks_minilm"  # adjust to match Day 18 if it exists
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def load_chunks(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class VectorRetriever:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.client = chromadb.PersistentClient(path=PERSIST_DIR)

        existing = [c.name for c in self.client.list_collections()]
        if COLLECTION_NAME in existing:
            self.collection = self.client.get_collection(COLLECTION_NAME)
            print(f"Loaded existing collection '{COLLECTION_NAME}' "
                  f"({self.collection.count()} chunks)")
        else:
            self.collection = self._build_collection()

    def _build_collection(self):
        collection = self.client.create_collection(COLLECTION_NAME)

        texts = [c["text"] for c in self.chunks]
        ids = [f"chunk_{c['metadata']['source_filename']}_{c['metadata']['chunk_index']}"
               for c in self.chunks]
        metadatas = [
            {
                "source_filename": c["metadata"]["source_filename"],
                "chunk_index": c["metadata"]["chunk_index"],
                "section_headings": " | ".join(c["metadata"]["section_headings"]),
            }
            for c in self.chunks
        ]

        print(f"Embedding {len(texts)} chunks with {EMBEDDING_MODEL}...")
        embeddings = self.model.encode(texts, show_progress_bar=True).tolist()

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        print(f"Built and persisted collection '{COLLECTION_NAME}'")
        return collection

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        query_embedding = self.model.encode([query]).tolist()

        raw = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
        )

        results = []
        for rank, (doc, meta, dist) in enumerate(
            zip(raw["documents"][0], raw["metadatas"][0], raw["distances"][0]),
            start=1,
        ):
            results.append({
                "rank": rank,
                # Chroma returns distance (lower = closer); convert to a
                # similarity-style score so it reads consistently with BM25
                "score": 1 - dist,
                "text": doc,
                "source_filename": meta["source_filename"],
                "chunk_index": meta["chunk_index"],
                "section_headings": meta["section_headings"].split(" | ") if meta["section_headings"] else [],
            })
        return results


def preview_results(query: str, results: list[dict]) -> None:
    print(f"\nQuery: {query}")
    print("-" * 60)
    for r in results:
        print(f"[{r['rank']}] score={r['score']:.3f} "
              f"({r['source_filename']} #{r['chunk_index']})")
        print(f"    {r['text'][:150].strip()}...")


if __name__ == "__main__":
    chunks = load_chunks(CHUNKS_PATH)
    print(f"Loaded {len(chunks)} chunks from {CHUNKS_PATH}")

    retriever = VectorRetriever(chunks)

    # Same sanity-check query as bm25_retrieval.py — run both and eyeball the difference
    test_query = "Which subteam would need to be involved if a change to the shared front end caused a repeat of the Q2 latency incident?"
    results = retriever.retrieve(test_query, top_k=5)
    preview_results(test_query, results)