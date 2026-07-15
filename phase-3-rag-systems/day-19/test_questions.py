"""
Day 19 — Test Questions
The 20 test questions reused from Day 15-17, in one place so
compare_pipelines.py (and anything else) can import them instead of each
script hand-copying a list.

Each question is a dict, not just a bare string, so you can tag it with
the kind of question it is (exact/keyword-heavy vs. conceptual/paraphrased
vs. multi-hop) — that categorization is what lets your report say
*why* one method won on a given question, not just that it did.

REPLACE the placeholder entries below with your actual 20 questions from
Day 15-17. "expected_source" and "expected_chunk_index" are optional but
worth filling in wherever you know the ground-truth answer chunk — that
turns your precision judging in compare_pipelines.py from "read the
output and decide" into "check if expected_chunk_index is in the top-k",
which is faster and more consistent across 20 questions x 4 methods.
"""

TEST_QUESTIONS = [
    {
        "id": 1,
        "query": ("Which subteam would need to be involved if a change to "
                  "the shared front end caused a repeat of the Q2 latency incident?"),
        "category": "multi-hop",  # connects 12.1 ownership + 9.2 incident history
        "expected_source": "multilingual_speech_pipeline.docx",
        "expected_chunk_index": 33,  # confirmed via manual run — see run log
    },
    {
    "id": 2,
    "query": "What percentage of crowd-sourced audio gets rejected during quality filtering?",
    "category": "exact",
    "expected_source": "multilingual_speech_pipeline.docx",
    "expected_chunk_index": 4,   # was None
},
    {
        "id": 3,
        "query": "What's the end-to-end latency target for streaming ASR?",
        "category": "exact",  # section 5.2, "800ms"
        "expected_source": "multilingual_speech_pipeline.docx",
        "expected_chunk_index": None,
    },
    {
        "id": 4,
        "query": "Why does the shared front end work better for Pashto than for Punjabi?",
        "category": "conceptual",  # section 3.1, tonal distinctions being lost
        "expected_source": "multilingual_speech_pipeline.docx",
        "expected_chunk_index": None,
    },
    {
        "id": 5,
        "query": "How many native-speaker raters are used per language for MOS evaluation?",
        "category": "exact",  # section 4.2, "at least twelve"
        "expected_source": "multilingual_speech_pipeline.docx",
        "expected_chunk_index": None,
    },
    {
        "id": 6,
        "query": ("If Punjabi TTS quality quietly starts degrading, what would catch it "
                  "and roughly how long could it take?"),
        "category": "multi-hop",  # section 8.2 drift detection + its historical lead time
        "expected_source": "multilingual_speech_pipeline.docx",
        "expected_chunk_index": None,
    },
    {
        "id": 7,
        "query": "Why did the team split training data by speaker instead of by utterance?",
        "category": "conceptual",  # section 7.1
        "expected_source": "multilingual_speech_pipeline.docx",
        "expected_chunk_index": None,
    },
    {
        "id": 8,
        "query": "What instance type serves the Urdu acoustic model?",
        "category": "exact",  # table 5.1, g5.xlarge
        "expected_source": "multilingual_speech_pipeline.docx",
        "expected_chunk_index": None,
    },
    {
        "id": 9,
        "query": "What caused the Q1 incident, and what changed afterward to prevent a repeat?",
        "category": "multi-hop",  # section 9.1 cause + resolution
        "expected_source": "multilingual_speech_pipeline.docx",
        "expected_chunk_index": None,
    },
    {
        "id": 10,
        "query": "Why is text-to-speech round-tripping used cautiously as an augmentation technique?",
        "category": "conceptual",  # section 7.3, risk of reinforcing TTS bias
        "expected_source": "multilingual_speech_pipeline.docx",
        "expected_chunk_index": None,
    },
    {
        "id": 11,
        "query": "How long is raw audio kept before it's deleted?",
        "category": "exact",  # section 11.1, eighteen months
        "expected_source": "multilingual_speech_pipeline.docx",
        "expected_chunk_index": None,
    },
    {
        "id": 12,
        "query": ("Who gets paged if a shared front-end change causes elevated latency "
                  "in production right now?"),
        "category": "multi-hop",  # 12.1 ownership boundary + 12.2 escalation via on-call
        "expected_source": "multilingual_speech_pipeline.docx",
        "expected_chunk_index": None,
    },
    {
        "id": 13,
        "query": "Why are monitoring alerts configured per language instead of one shared threshold?",
        "category": "conceptual",  # section 8.1, differing baseline confidence distributions
        "expected_source": "multilingual_speech_pipeline.docx",
        "expected_chunk_index": None,
    },
    {
        "id": 14,
        "query": "What is the ASR word error rate for Pashto?",
        "category": "exact",  # table 4.1, 21.7%
        "expected_source": "multilingual_speech_pipeline.docx",
        "expected_chunk_index": None,
    },
    {
        "id": 15,
        "query": ("If a mistagged Punjabi batch is found during routine review, not during "
                  "a live incident, what's the correct way to report it?"),
        "category": "multi-hop",  # 12.2 escalation distinction, mirrors the Q1 case in 9.1
        "expected_source": "multilingual_speech_pipeline.docx",
        "expected_chunk_index": None,
    },
    {
        "id": 16,
        "query": "Why is Urdu fine-tuned before Punjabi and Pashto in the training curriculum?",
        "category": "conceptual",  # section 7.2, resistance to catastrophic forgetting
        "expected_source": "multilingual_speech_pipeline.docx",
        "expected_chunk_index": None,
    },
    {
        "id": 17,
        "query": "What order are the three production languages fine-tuned in?",
        "category": "exact",  # section 7.2, Urdu, Punjabi, Pashto
        "expected_source": "multilingual_speech_pipeline.docx",
        "expected_chunk_index": None,
    },
    {
        "id": 18,
        "query": "Is Sindhi included in the current evaluation benchmarks?",
        "category": "multi-hop",  # section 1 (Sindhi not yet covered) + section 4 (eval scope)
        "expected_source": "multilingual_speech_pipeline.docx",
        "expected_chunk_index": None,
    },
    {
        "id": 19,
        "query": "How does the team currently handle Punjabi data from Pakistani vs. Indian sources?",
        "category": "conceptual",  # section 11.2, tagged separately but not technically separated
        "expected_source": "multilingual_speech_pipeline.docx",
        "expected_chunk_index": None,
    },
    {
        "id": 20,
        "query": "What is the TTS mean opinion score for Urdu?",
        "category": "exact",  # table 4.1, 4.1
        "expected_source": "multilingual_speech_pipeline.docx",
        "expected_chunk_index": None,
    },
]


def get_questions(category: str | None = None) -> list[dict]:
    """All 20 by default, or filter to just one category (e.g. only the
    'exact' ones) if you want to check a specific hypothesis like
    'does BM25 win on exact-keyword questions specifically'."""
    if category is None:
        return TEST_QUESTIONS
    return [q for q in TEST_QUESTIONS if q["category"] == category]


if __name__ == "__main__":
    print(f"Loaded {len(TEST_QUESTIONS)} test questions")
    by_category = {}
    for q in TEST_QUESTIONS:
        by_category.setdefault(q["category"], 0)
        by_category[q["category"]] += 1
    for cat, count in by_category.items():
        print(f"  {cat}: {count}")