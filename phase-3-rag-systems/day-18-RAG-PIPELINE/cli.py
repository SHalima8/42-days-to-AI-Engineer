"""
cli.py
Entry point. Just an input loop — no retrieval/prompt/generation logic here.
"""

from pipeline import run


def main():
    print("Day 18 — Simple RAG Pipeline")
    print("Type your question, or 'exit' to quit. Add '-v' after your question for verbose mode.\n")

    while True:
        raw = input("You: ").strip()
        if raw.lower() in ("exit", "quit"):
            break
        if not raw:
            continue

        verbose = raw.endswith("-v")
        question = raw[:-2].strip() if verbose else raw

        result = run(question, verbose=verbose)

        if result["error"]:
            print(f"\n[ERROR] {result['error']}\n")
            continue

        print(f"\nAssistant: {result['answer']}")
        if result["input_tokens"] is not None:
            print(f"(tokens: {result['input_tokens']} in / {result['output_tokens']} out)")
        print()


if __name__ == "__main__":
    main()