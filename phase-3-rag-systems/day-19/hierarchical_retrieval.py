"""
Day 19 — Hierarchical Retrieval (child → parent window expansion)
Retrieval (BM25/vector/RRF/rerank) runs on small chunks so matching stays
precise, but small chunks alone are often too little context for the LLM
to actually answer from. This module takes each retrieved "child" chunk
and expands it to a "parent window" — that chunk plus its immediate
neighbors from the same source document, stitched back into reading
order using chunk_index.

Needs the FULL chunk list (not just the retrieved subset) so it can look
up each chunk's neighbors, whether or not they were retrieved themselves.
"""

from bm25_retrieval import CHUNKS_PATH, load_chunks


def build_chunk_lookup(all_chunks: list[dict]) -> dict[tuple, dict]:
    """(source_filename, chunk_index) -> chunk record, for O(1) neighbor lookups."""
    return {
        (c["metadata"]["source_filename"], c["metadata"]["chunk_index"]): c
        for c in all_chunks
    }


class HierarchicalExpander:
    def __init__(self, all_chunks: list[dict], window: int = 1):
        """window=1 means "one chunk before, one chunk after" the retrieved
        child, i.e. a 3-chunk parent window. Bump to 2 for wider context if
        chunks are small; a chunk with no neighbor at the start/end of a
        document is just skipped rather than erroring."""
        self.window = window
        self.lookup = build_chunk_lookup(all_chunks)

    def expand(self, results: list[dict]) -> list[dict]:
        expanded = []
        for r in results:
            source = r["source_filename"]
            center = r["chunk_index"]

            neighbor_indices = range(center - self.window, center + self.window + 1)
            parent_chunks = [
                self.lookup[(source, idx)]
                for idx in neighbor_indices
                if (source, idx) in self.lookup
            ]
            parent_chunks.sort(key=lambda c: c["metadata"]["chunk_index"])

            parent_text = "\n\n".join(c["text"] for c in parent_chunks)
            parent_indices = [c["metadata"]["chunk_index"] for c in parent_chunks]

            expanded.append({
                **r,
                # keep the precise child text separately — useful for
                # debugging exactly what matched vs what got sent to the LLM
                "child_text": r["text"],
                "child_chunk_index": center,
                # "text" now holds the expanded window — downstream code
                # (pipelines.py building the LLM context) can keep reading
                # from "text" without needing to know hierarchical
                # retrieval happened at all
                "text": parent_text,
                "parent_chunk_indices": parent_indices,
            })
        return expanded


def preview_results(query: str, results: list[dict]) -> None:
    print(f"\nQuery: {query}")
    print("-" * 60)
    for r in results:
        print(f"[{r.get('rank')}] child #{r['child_chunk_index']} -> "
              f"parent window {r['parent_chunk_indices']} "
              f"({r['source_filename']})")
        print(f"    child : {r['child_text'][:100].strip()}...")
        print(f"    parent: {r['text'][:150].strip()}...")


if __name__ == "__main__":
    from vector_retrieval import VectorRetriever
    from bm25_retrieval import BM25Retriever
    from rrf_fusion import rrf_fuse
    from reranker import Reranker

    all_chunks = load_chunks(CHUNKS_PATH)

    bm25 = BM25Retriever(all_chunks)
    vector = VectorRetriever(all_chunks)
    reranker = Reranker()
    expander = HierarchicalExpander(all_chunks, window=1)

    test_query = ("What is the ASR WER for Pashto?")

    bm25_results = bm25.retrieve(test_query, top_k=20)
    vector_results = vector.retrieve(test_query, top_k=20)
    fused = rrf_fuse(bm25_results, vector_results, top_k=20)
    reranked = reranker.rerank(test_query, fused, top_k=5)

    expanded = expander.expand(reranked)
    preview_results(test_query, expanded)