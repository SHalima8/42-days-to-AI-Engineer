"""
Day 19 — Pipeline Assembly
Wires the individual retrieval modules into the four named pipelines used
for the Day 19 comparison:

    simple_rag(query)     -> rewrite + vector only
    bm25_only(query)      -> rewrite + BM25 only
    hybrid_rrf(query)     -> rewrite + BM25 + vector + RRF
    hybrid_rerank(query)  -> hybrid_rrf (top-20) + cross-encoder re-rank

All four start with the same query rewrite step, so any difference in
results reflects the retrieval method itself, not inconsistent rewriting
between methods.
"""

from bm25_retrieval import BM25Retriever, CHUNKS_PATH, load_chunks
from vector_retrieval import VectorRetriever
from query_rewriter import QueryRewriter
from rrf_fusion import rrf_fuse
from reranker import Reranker

DEFAULT_TOP_K = 5


class RAGPipelines:
    """Loads every index/model exactly once. Instantiate one of these per
    script run (e.g. once in compare_pipelines.py), not once per query —
    building BM25 + vector + rewriter + reranker is the expensive part,
    and there's no reason to pay that cost 20 times over for 20 questions."""

    def __init__(self, chunks_path: str = CHUNKS_PATH):
        chunks = load_chunks(chunks_path)
        self.bm25 = BM25Retriever(chunks)
        self.vector = VectorRetriever(chunks)
        self.rewriter = QueryRewriter()
        self.reranker = Reranker()

    def simple_rag(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
        rewritten = self.rewriter.rewrite(query)
        return self.vector.retrieve(rewritten, top_k=top_k)

    def bm25_only(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
        rewritten = self.rewriter.rewrite(query)
        return self.bm25.retrieve(rewritten, top_k=top_k)

    def hybrid_rrf(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
        rewritten = self.rewriter.rewrite(query)
        bm25_results = self.bm25.retrieve(rewritten, top_k=20)
        vector_results = self.vector.retrieve(rewritten, top_k=20)
        return rrf_fuse(bm25_results, vector_results, top_k=top_k)

    def hybrid_rerank(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
        rewritten = self.rewriter.rewrite(query)
        bm25_results = self.bm25.retrieve(rewritten, top_k=20)
        vector_results = self.vector.retrieve(rewritten, top_k=20)
        fused = rrf_fuse(bm25_results, vector_results, top_k=20)
        return self.reranker.rerank(rewritten, fused, top_k=top_k)


def preview_results(method_name: str, query: str, results: list[dict]) -> None:
    print(f"\n[{method_name}] {query}")
    print("-" * 60)
    for r in results:
        print(f"  #{r.get('rank')} ({r['source_filename']} #{r['chunk_index']})")
        print(f"    {r['text'][:120].strip()}...")


if __name__ == "__main__":
    pipelines = RAGPipelines()

    test_query = ("What is the ASR WER for Pashto?")

    for name, fn in [
        ("simple_rag", pipelines.simple_rag),
        ("bm25_only", pipelines.bm25_only),
        ("hybrid_rrf", pipelines.hybrid_rrf),
        ("hybrid_rerank", pipelines.hybrid_rerank),
    ]:
        results = fn(test_query, top_k=5)
        preview_results(name, test_query, results)