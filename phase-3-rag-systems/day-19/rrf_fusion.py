"""
Day 19 — Reciprocal Rank Fusion (RRF)
Merges BM25 and vector-retrieval rankings into a single unified list, so
a chunk that ranks well in *either* method gets pulled up top instead of
the final list being biased toward whichever method runs last/first.

RRF score for a chunk = sum, over each ranker it appears in, of
    1 / (k + rank_in_that_ranker)

k=60 is the standard constant from the original RRF paper — it dampens
how much rank 1 vs rank 2 matters, so one method's rank-1 pick doesn't
completely dominate a chunk that's consistently rank 3-4 in both.
A chunk that only shows up in one list (BM25 or vector, not both) still
gets scored fine using just that one rank — no need to invent a fake
rank in the list it's absent from.

Default top_k=20 to match the reranker step, which re-scores the top-20
fused chunks with the cross-encoder.
"""

from bm25_retrieval import BM25Retriever, CHUNKS_PATH, load_chunks
from vector_retrieval import VectorRetriever

RRF_K = 60


def rrf_fuse(
    bm25_results: list[dict],
    vector_results: list[dict],
    k: int = RRF_K,
    top_k: int = 20,
) -> list[dict]:
    scores: dict[tuple, float] = {}
    chunk_info: dict[tuple, dict] = {}
    origins: dict[tuple, set] = {}

    for results, origin in [(bm25_results, "bm25"), (vector_results, "vector")]:
        for r in results:
            key = (r["source_filename"], r["chunk_index"])
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + r["rank"])
            chunk_info.setdefault(key, r)
            origins.setdefault(key, set()).add(origin)

    fused = []
    for key, score in scores.items():
        info = chunk_info[key]
        fused.append({
            "score": score,
            "text": info["text"],
            "source_filename": info["source_filename"],
            "chunk_index": info["chunk_index"],
            "section_headings": info["section_headings"],
            # ["bm25"], ["vector"], or ["bm25", "vector"] — feeds
            # retrieval_origin_log.json directly
            "origin": sorted(origins[key]),
        })

    fused.sort(key=lambda x: x["score"], reverse=True)
    fused = fused[:top_k]
    for rank, item in enumerate(fused, start=1):
        item["rank"] = rank

    return fused


def preview_results(query: str, results: list[dict], limit: int = 10) -> None:
    print(f"\nQuery: {query}")
    print("-" * 60)
    for r in results[:limit]:
        print(f"[{r['rank']}] score={r['score']:.4f} origin={r['origin']} "
              f"({r['source_filename']} #{r['chunk_index']})")
        print(f"    {r['text'][:150].strip()}...")


if __name__ == "__main__":
    chunks = load_chunks(CHUNKS_PATH)

    bm25 = BM25Retriever(chunks)
    vector = VectorRetriever(chunks)

    test_query = ("What is the ASR WER for Pashto?")

    # Pull 20 from each so the fusion has real overlap/disagreement to
    # work with, not just each method's top-5
    bm25_results = bm25.retrieve(test_query, top_k=20)
    vector_results = vector.retrieve(test_query, top_k=20)

    fused = rrf_fuse(bm25_results, vector_results, top_k=20)

    # Print top 10 for readability; full 20 is what feeds the reranker
    preview_results(test_query, fused, limit=10)

    both = sum(1 for r in fused if len(r["origin"]) == 2)
    print(f"\n{both}/{len(fused)} chunks in top-20 were surfaced by both methods")