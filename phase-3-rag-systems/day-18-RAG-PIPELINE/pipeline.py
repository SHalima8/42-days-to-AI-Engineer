"""
pipeline.py
Orchestrates: question -> retrieve (with table resolution) -> augment -> generate.
cli.py stays dumb and just calls run().
"""

from src.retriever import retrieve
from src.prompt_builder import build_augmented_prompt
from src.llm_client import generate_answer


def run(question: str, top_k: int = None, verbose: bool = False) -> dict:
    chunks = retrieve(question, top_k=top_k)
    prompt = build_augmented_prompt(question, chunks)
    result = generate_answer(prompt)

    if verbose:
        print("\n--- Retrieved Chunks ---")
        for c in chunks:
            tag = "[TABLE]" if c["element_type"] == "table" else "[TEXT] "
            print(f"  {tag} [{c['confidence']}] {c['source']}: {c['text'][:80]}...")
        print("\n--- Augmented Prompt ---")
        print(prompt)

    return {
        "question": question,
        "chunks": chunks,
        "prompt": prompt,
        "answer": result.get("answer"),
        "error": result.get("error"),
        "input_tokens": result.get("input_tokens"),
        "output_tokens": result.get("output_tokens"),
    }