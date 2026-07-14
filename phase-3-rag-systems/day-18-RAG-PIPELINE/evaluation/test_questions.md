# Day 18 — Test Questions

Questions written against the ingested `multilingual_speech_pipeline.docx`,
covering all 4 required question types.

## 1. Factual lookup
- Q: What is the ASR WER for Pashto?
- Expected answer: 21.7% (Table 4.1)
- Actual: Model correctly stated Pashto has the highest WER of the three
  languages, cited the reason (limited training hours, dialectal variation),
  but explicitly said the exact percentage was not present in the retrieved
  context -- it did not fabricate a number. See failure_log.md, Case 1.

## 2. Inference required
- Q: How did the Q1 Punjabi incident change how script detection works going forward?
- Expected answer: Should connect Section 9.1 (what happened -- Shahmukhi
  data mistagged as Gurmukhi due to a file-naming convention change) with
  Section 10 (the fix -- move to content-based script detection upstream
  of all ingestion paths).
- Actual: Model gave the correct answer content, but cited the wrong
  section (Section 6.2 instead of Section 10). See failure_log.md, Case 3.

## 3. Cross-document / cross-section
- Q: Which subteam would need to be involved if a change to the shared
  front end caused a repeat of the Q2 latency incident?
- Expected answer: Needs Section 12.1 (component ownership -- shared front
  end requires sign-off from both modeling and data engineering subteams)
  combined with Section 9.2 (what the Q2 incident actually was).
- Actual: Not yet tested -- pending run.

## 4. No answer in docs
- Q: What GPU utilization percentage did the Sindhi model achieve last quarter?
- Expected answer: Sindhi is only in early data collection (Section 1),
  no training/serving numbers exist yet -- model should say "not found."
- Actual: Retrieved chunks came back with visibly lower confidence scores
  (0.691 -> 0.615) than the in-scope Pashto WER query (0.764 -> 0.693),
  even though 4 chunks were still returned. Confidence spread itself
  is a usable signal for whether a question is answerable from the corpus.