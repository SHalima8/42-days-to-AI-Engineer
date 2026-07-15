# Day 18 -- Simple RAG Failure Log

## Case 1: Table content not retrieved for a table-answerable question
**Status: FIXED**

**Question:** What is the ASR WER for Pashto?
**Type:** Factual lookup
**Involved a table?** Yes -- answer lives in Table 4.1, but the table was
never retrieved.

**Before fix -- Retrieved chunks:** 4 paragraph-type chunks (confidence
0.764, 0.740, 0.714, 0.693), all describing Pashto's WER qualitatively.
No table-type chunk appeared in the top-4.
**Before fix -- Answer:** Model correctly stated Pashto has the highest
WER and cited the reason, then explicitly said the exact percentage was
not present in the retrieved context. No hallucination, but the pipeline
still failed to answer a question it should have been able to answer.

**Root cause:** Table chunks were embedded from their placeholder string
("[Table 4.1 -- see table registry, id: ...]"), which is short and
generic. Its embedding was semantically weaker than a paragraph that
literally contains the words "WER" and "Pashto," so the paragraph won
the vector similarity search even though the table held the real answer.
The embedding never represented the table's actual content.

**Fix implemented:** `reembed_tables.py` -- a one-time script that reads
`tables_lookup.json` for the list of tables, pulls each one's real
summary text from `tables_summary.md`, embeds that text (prefixed with
caption + section heading for extra semantic anchoring), and upserts it
into the `chunks_minilm` collection, replacing the old placeholder-based
entry for that `table_id`.

**After fix -- Retrieved chunks:** Table 4.1 now appears at position [2]
with confidence 0.743, carrying its real summary text.
**After fix -- Answer:** Model answered directly with 21.7%, correctly
sourced to Table 4.1.

---

## Case 2: Table placeholder leaking into a non-table chunk
**Status: FIXED (defensive patch)**

**Question:** What are the error types for Pashto?
**Type:** Factual lookup
**Involved a table?** Yes, indirectly.

**Before fix:** Verbose mode showed a raw, unresolved placeholder --
"[Table 4.2 -- see table registry, id: multilingual_speech_pipeline_table_0001]"
-- glued onto the end of a chunk tagged `element_type: paragraph`, not
`table`. The original table_lookup resolution logic only checked whole
chunks tagged `table`, so this buried placeholder passed through
untouched and reached the LLM as junk context.

**Root cause:** The Day 16 chunker (docx ingestion path) merged a table
pointer into its surrounding paragraph instead of keeping it as an
isolated chunk. This is a source-level (Day 16) bug -- not fixed at the
source for this exercise, since it would mean re-running ingestion.

**Fix implemented:** A defensive regex-based patch in `retriever.py`.
Every retrieved chunk -- not just ones tagged `table` -- is scanned for
a stray `[... id: <table_id>]` pattern. If found, the match is resolved
inline via `table_lookup.get_table_summary_only()`, regardless of which
chunk type the placeholder was hiding inside. This does not fix the
Day 16 chunking bug itself, but guarantees no raw placeholder text ever
reaches the LLM.

**After fix -- Answer:** Asked "What are the error types for Pashto?",
the model returned exact, correctly-sourced numbers (Substitution 12.4%,
Deletion 5.8%, Insertion 3.5%, all cited to Table 4.2 / Section 4.1) with
no placeholder artifacts or garbled references anywhere in the response.

---

## Case 3: Correct answer, wrong section cited
**Status: NOT FIXED (documented, not addressed this session)**

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

**Root cause:** Two semantically adjacent chunks (one from 6.2 describing
the problem, one from Section 10 describing the fix) were both retrieved.
The model appears to have attributed the quote to the more prominent
nearby heading rather than tracking which specific chunk the sentence
came from. This is a generation-layer issue, not a retrieval issue --
out of scope for the fixes applied this session.

---

## Bonus finding: cross-document naming collision handled correctly
While testing Case 2's fix, a second, unrelated table also captioned
"Table 4.2" (source: `scanned_document.pdf`, containing no actual Pashto
data) was retrieved alongside the real Table 4.2. The model correctly
distinguished between the two sources and explicitly noted the
`scanned_document.pdf` table had no data points for Pashto, rather than
blending the two same-named tables into a confused answer. Grounding
instructions held up well even under a same-caption collision across
source documents.

---

## Summary
- 2 of 3 identified failure modes fixed and verified with before/after
  evidence this session (Cases 1 and 2).
- Case 3 remains open -- a generation-layer citation issue, not
  addressed by either fix applied here.
- No hallucinated facts observed on any tested query across the whole
  session, before or after fixes -- when the exact answer wasn't
  retrievable, the model said so rather than guessing.
- Fix for Case 1 (re-embedding tables using real summary text instead of
  placeholder text) directly validates the V2 roadmap suggestion from
  the original report.