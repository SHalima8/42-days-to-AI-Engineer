"""
prompt_builder.py
Builds the augmented prompt. Table chunks are labeled explicitly so the LLM
knows it's reading structured data, not prose — this noticeably improves
answers to "what is the value of X" style questions against tables.
"""


def build_augmented_prompt(query: str, retrieved_chunks: list[dict]) -> str:
    if not retrieved_chunks:
        context_block = "(No relevant context was retrieved from the document collection.)"
    else:
        blocks = []
        for c in retrieved_chunks:
            label = "TABLE" if c["element_type"] == "table" else "TEXT"
            location = f"{c['source']}"
            if c.get("section_heading"):
                location += f", section: {c['section_heading']}"

            blocks.append(
                f"[{label} | Source: {location} | Confidence: {c['confidence']}]\n{c['text']}"
            )
        context_block = "\n\n".join(blocks)

    prompt = f"""Context from document collection:
{context_block}

Question: {query}

Instructions:
- Answer ONLY using the context above.
- If the context does not contain enough information to answer, say so explicitly — do not guess or use outside knowledge.
- When you use a TABLE block, cite specific values from it rather than paraphrasing loosely.
- Mention which source (and section, if given) your answer is grounded in.

Answer:"""

    return prompt