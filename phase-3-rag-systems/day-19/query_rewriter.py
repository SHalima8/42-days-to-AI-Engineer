"""
Day 19 — Query Rewriter (Groq version)
Uses Groq to rewrite a raw/conversational user question into a clean,
keyword-dense retrieval query before it hits BM25 / vector search.
Falls back silently to the original query if the API call fails, is
empty, or rate-limited — a rewrite failure should never break the
retrieval pipeline downstream.

Install:
    pip install groq python-dotenv

.env:
    GROQ_API_KEY=your_key_here
"""

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# Fast + free-tier friendly; swap to "llama-3.1-8b-instant" if this one
# is ever deprecated or rate-limited on your account
MODEL_NAME = "llama-3.3-70b-versatile"

client = Groq(api_key=GROQ_API_KEY)

REWRITE_PROMPT = """You are a query rewriting module in a RAG pipeline. \
Rewrite the following user question into a clear, keyword-rich search \
query optimized for retrieval over internal documents. Preserve every \
named entity, resolve pronouns using the conversation history if given, \
and remove filler or conversational phrasing. Do NOT change the meaning \
or add information that isn't implied by the question.

Return ONLY the rewritten query on a single line — no preamble, no \
quotes, no explanation.

Conversation history (may be empty):
{history}

User question:
{query}

Rewritten query:"""


class QueryRewriter:
    """Same call shape convention as the retrievers: build once, call
    per-query. .rewrite(query) -> str, always returns something usable."""

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name

    def rewrite(self, query: str, history: list[str] | None = None) -> str:
        history_text = "\n".join(history) if history else "(none)"
        prompt = REWRITE_PROMPT.format(history=history_text, query=query)

        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=100,
            )
            rewritten = response.choices[0].message.content.strip().strip('"').strip("'")
            return rewritten if rewritten else query
        except Exception as e:
            print(f"[query_rewriter] Falling back to original query — {e}")
            return query


def preview_rewrite(original: str, rewritten: str) -> None:
    print(f"\nOriginal : {original}")
    print(f"Rewritten: {rewritten}")


if __name__ == "__main__":
    rewriter = QueryRewriter()

    test_queries = [
        "What is the ASR WER for Pashto?",
        "What is the ASR WER for Dari?",
    ]

    for q in test_queries:
        rewritten = rewriter.rewrite(q)
        preview_rewrite(q, rewritten)