# day-17/run_benchmark.py

import json
import time
import csv
from embeddings.embedder_factory import get_embedder
from vector_stores.chroma_store import ChromaStore
from vector_stores.faiss_store import FaissStore

EVAL_QUESTIONS_PATH = "data/eval_questions.json"
MODELS = ["minilm", "mpnet", "bge"]
STORES = ["chroma", "faiss"]
TOP_K = 3

OUTPUT_RAW = "outputs/benchmark_results.csv"
OUTPUT_SUMMARY = "outputs/summary_table.csv"


def load_eval_questions(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def precision_at_k(retrieved_ids, relevant_ids, k=3):
    top_k = retrieved_ids[:k]
    hits = len(set(top_k) & set(relevant_ids))
    return hits / k


def get_store(store_type, model_name):
    if store_type == "chroma":
        return ChromaStore(model_name=model_name)
    else:
        return FaissStore(model_name=model_name)


def run_single_combo(model_name, store_type, embedder, store, questions):
    results = []
    for q in questions:
        t0 = time.time()
        query_vec = embedder.embed_query(q["query"])
        retrieved = store.query(query_vec, top_k=TOP_K)
        latency_ms = (time.time() - t0) * 1000

        retrieved_ids = [r[0] for r in retrieved]  # r = (chunk_id, text, score/distance)
        p_at_3 = precision_at_k(retrieved_ids, q["relevant_chunk_ids"], k=TOP_K)

        results.append({
            "model": model_name,
            "store": store_type,
            "query": q["query"],
            "retrieved_ids": ", ".join(retrieved_ids),
            "relevant_ids": ", ".join(q["relevant_chunk_ids"]),
            "precision_at_3": round(p_at_3, 3),
            "latency_ms": round(latency_ms, 2)
        })
    return results


def main():
    questions = load_eval_questions(EVAL_QUESTIONS_PATH)
    print(f"Loaded {len(questions)} eval questions.\n")

    all_results = []

    for model_name in MODELS:
        print(f"=== Loading embedder: {model_name} ===")
        embedder = get_embedder(model_name)

        for store_type in STORES:
            print(f"  -- Benchmarking store: {store_type} --")
            store = get_store(store_type, model_name)
            combo_results = run_single_combo(model_name, store_type, embedder, store, questions)
            all_results.extend(combo_results)

    # --- Save raw per-question results ---
    with open(OUTPUT_RAW, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nRaw results saved to {OUTPUT_RAW}")

    # --- Aggregate into summary table (avg precision + avg latency per model x store) ---
    summary = {}
    for r in all_results:
        key = (r["model"], r["store"])
        if key not in summary:
            summary[key] = {"precisions": [], "latencies": []}
        summary[key]["precisions"].append(r["precision_at_3"])
        summary[key]["latencies"].append(r["latency_ms"])

    with open(OUTPUT_SUMMARY, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "store", "avg_precision_at_3", "avg_latency_ms"])
        for (model_name, store_type), vals in summary.items():
            avg_p = sum(vals["precisions"]) / len(vals["precisions"])
            avg_l = sum(vals["latencies"]) / len(vals["latencies"])
            writer.writerow([model_name, store_type, round(avg_p, 3), round(avg_l, 2)])

    print(f"Summary table saved to {OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()