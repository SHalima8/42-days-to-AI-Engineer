# Day 18 — Simple RAG Failure Log

## Case 1: Table content not retrieved for a table-answerable question
**Question:** What is the ASR WER for Pashto?
**Type:** Factual lookup
**Involved a table?** Yes -- answer lives in Table 4.1, but the table was
never retrieved.
**Retrieved chunks:** 4 paragraph-type chunks (confidence 0.764, 0.740,
0.714, 0.693), all describing Pashto's WER qualitatively. No table-type
chunk appeared in the top-4.
**Actual answer:** Model correctly stated Pashto has the highest WER and
cited the reason, then explicitly said the exact percentage was not
present in the retrieved context. No hallucination.
**Failure mode:** Missed context (retrieval miss).
**Root cause:** Table chunks are embedded from their placeholder string
("[Table 4.1 -- see table registry, id: ...]"), which is short and
generic. Its embedding is semantically weaker than a paragraph that
literally contains the words "WER" and "Pashto," so the paragraph wins
the vector similarity search even though the table holds the real answer.

---

## Case 2: Table placeholder leaking into a non-table chunk
**Question:** Same as Case 1 (What is the ASR WER for Pashto?)
**Type:** Factual lookup
**Involved a table?** Yes, indirectly.
**Retrieved chunks:** Verbose mode showed a raw, unresolved placeholder --
"[Table 4.2 -- see table registry, id: multilingual_speech_pipeline_table_0001]"
-- glued onto the end of a chunk tagged element_type: paragraph, not table.
**Actual answer:** N/A -- this is a context-quality issue, not a
generation-quality issue, but it does mean junk text reached the LLM.
**Failure mode:** Placeholder leak.
**Root cause:** The Day 16 chunker merged a table pointer into its
surrounding paragraph instead of keeping it as an isolated chunk. Since
retriever.py's table resolution only checks whole-chunk element_type,
a placeholder buried mid-paragraph is never caught or resolved.

---

## Case 3: Correct answer, wrong section cited
**Question:** How did the Q1 Punjabi incident change how script detection
works going forward?
**Type:** Inference / cross-section
**Involved a table?** No.
**Expected:** Answer should cite Section 10 (Roadmap), where the fix
("move content-based script detection upstream of all ingestion paths")
is actually listed.
**Actual answer:** Model gave the correct answer content, but attributed
the quote to Section 6.2 (Script Ambiguity in Punjabi) -- which only
describes the underlying problem, not the fix.
**Failure mode:** Citation misattribution. Content was grounded; the
citation was not.
**Root cause:** Two semantically adjacent chunks (one from 6.2 describing
the problem, one from Section 10 describing the fix) were both retrieved.
The model appears to have attributed the quote to the more prominent
nearby heading rather than tracking which specific chunk the sentence
came from.

---

## Summary
- No hallucinated facts observed on any tested query this session --
  when the exact answer wasn't retrieved, the model said so rather than
  guessing.
- Two distinct retrieval-layer bugs found (Cases 1 & 2), both tracing
  back to how tables are represented as chunks during Day 16 ingestion.
- One generation-layer bug found (Case 3): citation accuracy breaks down
  when multiple retrieved chunks discuss related content under different
  headings.
- Suggested fix for V2: embed tables using their resolved summary text
  (from tables_summary.md) instead of the placeholder string, so table
  chunks can actually compete in vector search on their real content.