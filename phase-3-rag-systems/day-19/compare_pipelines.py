"""
Day 19 — Pipeline Comparison
Runs every question in test_questions.py through all four pipelines.py
methods and writes out everything your report needs:

    outputs/bm25_vs_vector.json       - step 1: raw BM25 vs vector top-5 only
    outputs/pipeline_comparison.json  - full 4-way raw results, every question
    outputs/comparison_table.md       - question x method hit/miss/review table
    outputs/retrieval_origin_log.json - for hybrid methods, which chunks came
                                         from bm25/vector/both, pre-rerank

Precision is computed automatically only for questions where
expected_chunk_index is filled in (test_questions.py). Everything else is
marked "review" so it doesn't silently get counted as a miss — an unknown
ground truth isn't the same as a wrong answer.
"""

import json
import os

from pipelines import RAGPipelines
from test_questions import TEST_QUESTIONS

OUTPUT_DIR = "outputs"
TOP_K = 5
METHODS = ["simple_rag", "bm25_only", "hybrid_rrf", "hybrid_rerank"]


def is_hit(results: list[dict], expected_source, expected_chunk_index) -> bool | None:
    if expected_chunk_index is None:
        return None  # ground truth not filled in yet -> needs manual review
    return any(
        r["source_filename"] == expected_source and r["chunk_index"] == expected_chunk_index
        for r in results
    )


def run_comparison():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pipelines = RAGPipelines()  # loaded once, reused across all 20 x 4 runs

    full_results = []
    bm25_vs_vector = []
    origin_log = []

    for q in TEST_QUESTIONS:
        query = q["query"]
        row = {"id": q["id"], "query": query, "category": q["category"], "methods": {}}
        method_outputs = {}

        for method_name in METHODS:
            fn = getattr(pipelines, method_name)
            results = fn(query, top_k=TOP_K)
            method_outputs[method_name] = results

            hit = is_hit(results, q["expected_source"], q["expected_chunk_index"])
            row["methods"][method_name] = {
                "hit": hit,
                "top_result": (
                    {"source_filename": results[0]["source_filename"],
                     "chunk_index": results[0]["chunk_index"]}
                    if results else None
                ),
                "results": results,
            }

        full_results.append(row)

        bm25_vs_vector.append({
            "id": q["id"],
            "query": query,
            "bm25_top5": method_outputs["bm25_only"],
            "vector_top5": method_outputs["simple_rag"],
        })

        for method_name in ["hybrid_rrf", "hybrid_rerank"]:
            for r in method_outputs[method_name]:
                origin_log.append({
                    "question_id": q["id"],
                    "method": method_name,
                    "source_filename": r["source_filename"],
                    "chunk_index": r["chunk_index"],
                    "origin": r.get("origin", []),
                })

    with open(f"{OUTPUT_DIR}/pipeline_comparison.json", "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False)

    with open(f"{OUTPUT_DIR}/bm25_vs_vector.json", "w", encoding="utf-8") as f:
        json.dump(bm25_vs_vector, f, indent=2, ensure_ascii=False)

    with open(f"{OUTPUT_DIR}/retrieval_origin_log.json", "w", encoding="utf-8") as f:
        json.dump(origin_log, f, indent=2, ensure_ascii=False)

    write_comparison_table(full_results)
    print_precision_summary(full_results)
    print(f"\nWrote outputs to {OUTPUT_DIR}/")


def write_comparison_table(full_results: list[dict]) -> None:
    header = ["| # | Question | Category | " + " | ".join(METHODS) + " |",
              "|---|----------|----------|" + "|".join(["---"] * len(METHODS)) + "|"]
    lines = list(header)

    for row in full_results:
        cells = []
        for method_name in METHODS:
            hit = row["methods"][method_name]["hit"]
            cells.append("Hit" if hit is True else "Miss" if hit is False else "Review")
        short_q = row["query"] if len(row["query"]) <= 60 else row["query"][:57] + "..."
        lines.append(f"| {row['id']} | {short_q} | {row['category']} | " + " | ".join(cells) + " |")

    with open(f"{OUTPUT_DIR}/comparison_table.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def print_precision_summary(full_results: list[dict]) -> None:
    print("\nPrecision summary (only over questions with a known expected_chunk_index):")
    for method_name in METHODS:
        hits = [row["methods"][method_name]["hit"] for row in full_results]
        known = [h for h in hits if h is not None]
        if known:
            precision = sum(known) / len(known) * 100
            print(f"  {method_name}: {precision:.0f}% ({sum(known)}/{len(known)} known)")
        else:
            print(f"  {method_name}: no ground truth available yet")

    unresolved = sum(1 for q in TEST_QUESTIONS if q["expected_chunk_index"] is None)
    if unresolved:
        print(f"\n{unresolved}/{len(TEST_QUESTIONS)} questions still need "
              f"expected_chunk_index filled in for automatic scoring — "
              f"see find_chunk.py to help locate them.")


if __name__ == "__main__":
    run_comparison()