"""
Day 19 — Cross-Encoder Re-ranking
Takes the RRF-fused top-20 (BM25 + vector) and re-scores each chunk
against the query with a cross-encoder, which reads query and chunk
together instead of comparing separate embeddings. Slower per-pair than
BM25/vector, which is exactly why it only runs on a shortlist of 20
instead of the whole corpus.

Install:
    pip install sentence-transformers
    (same package vector_retrieval.py already needs — CrossEncoder
    lives alongside SentenceTransformer in the same library)
"""

from sentence_transformers import CrossEncoder

from bm25_retrieval import BM25Retriever, CHUNKS_PATH, load_chunks
from vector_retrieval import VectorRetriever
from rrf_fusion import rrf_fuse

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-12-v2"


class Reranker:
    """Loaded once, reused across queries — model load is the expensive
    part, .rerank() itself is one batched predict() call."""

    def __init__(self, model_name: str = MODEL_NAME):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, results: list[dict], top_k: int = 5) -> list[dict]:
        if not results:
            return []

        pairs = [[query, r["text"]] for r in results]
        scores = self.model.predict(pairs)

        # keep the RRF rank each chunk arrived with (as "prerank_rank")
        # so you can see how much the cross-encoder actually reshuffled
        # things — that's the interesting number for your report, not
        # just the final order
        for r, score in zip(results, scores):
            r["prerank_rank"] = r.get("rank")
            r["rerank_score"] = float(score)

        reranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)[:top_k]
        for rank, item in enumerate(reranked, start=1):
            item["rank"] = rank  # now reflects post-rerank order

        return reranked


def preview_results(query: str, results: list[dict]) -> None:
    print(f"\nQuery: {query}")
    print("-" * 60)
    for r in results:
        moved = r["prerank_rank"] - r["rank"]
        arrow = f"(was #{r['prerank_rank']}, {'+' if moved >= 0 else ''}{moved})"
        print(f"[{r['rank']}] rerank_score={r['rerank_score']:.3f} {arrow} "
              f"origin={r['origin']} ({r['source_filename']} #{r['chunk_index']})")
        print(f"    {r['text'][:150].strip()}...")


if __name__ == "__main__":
    chunks = load_chunks(CHUNKS_PATH)

    bm25 = BM25Retriever(chunks)
    vector = VectorRetriever(chunks)
    reranker = Reranker()

    test_query = ("What is the ASR WER for Pashto?")

    bm25_results = bm25.retrieve(test_query, top_k=20)
    vector_results = vector.retrieve(test_query, top_k=20)
    fused = rrf_fuse(bm25_results, vector_results, top_k=20)

    reranked = reranker.rerank(test_query, fused, top_k=5)
    preview_results(test_query, reranked)