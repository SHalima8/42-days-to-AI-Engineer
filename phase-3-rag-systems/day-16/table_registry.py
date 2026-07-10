"""
table_registry.py

Shared table registry used by every ingestion module (pdf_ingest, docx_ingest,
ocr_ingest) so tables never get inlined into the main text/chunking flow.

WHY THIS EXISTS
    Before: a table's full content lived inline inside an ExtractedElement's
    text, mixed in with paragraphs. Chunkers had no way to know "this piece
    is a table, don't split it" — a table could get fragmented across chunk
    boundaries, or absorbed into a chunk of unrelated surrounding prose.
    There was also no way to answer a direct question like "explain table
    2.13" without hoping vector search happened to retrieve the right chunk.

    After: every table detected during ingestion is:
      1. assigned a stable table_id
      2. summarized once via an LLM call (Gemini) into clean prose
      3. registered in TWO separate files:
           tables_lookup.json  -> the INDEX  (caption -> table_id -> location)
           tables_summary.md   -> the CONTENT (table_id -> LLM summary + raw table)
      4. replaced, in the main ingestion output (pdf_text.json etc.), with a
         short POINTER element — just the table_id + a placeholder sentence,
         never the full table content.

    This happens at INGESTION time, before any chunking runs, so chunkers
    never see raw table content and can't fragment it.

DIRECT-LOOKUP FLOW THIS ENABLES  ("explain table 2.13")
    1. lookup_table_by_caption("2.13") searches tables_lookup.json by plain
       substring match — NOT vector search, NOT chunking.
    2. That returns the table_id, source file, page number, section.
    3. get_table_summary(table_id) pulls the matching "## <table_id>"
       section straight out of tables_summary.md.
    No embeddings involved, no risk of the table being split or missed —
    it's a direct, deterministic lookup for a precise reference.
"""

import os
import re
import json
from pathlib import Path
from typing import Optional

# Load .env into os.environ. Without this, GEMINI_API_KEY set in a .env
# file is invisible to os.environ.get() below — this was silently causing
# every table to fall back to raw markdown, with no error printed, because
# the code never actually attempted the API call in the first place.
from dotenv import load_dotenv
load_dotenv()

TABLES_LOOKUP_PATH = "data/extracted/tables_lookup.json"
TABLES_SUMMARY_PATH = "data/extracted/tables_summary.md"

GEMINI_MODEL = "gemini-2.5-flash"

# Matches things like "Table 2.13", "Table 4", "Figure 3.1" near a table
_CAPTION_PATTERN = re.compile(r"\b(Table|Fig(?:ure)?)\s+([0-9]+(?:\.[0-9]+)*)\b", re.IGNORECASE)


def detect_caption(nearby_text: Optional[str]) -> Optional[str]:
    """
    Looks for a caption-like pattern in text found near a table — usually
    the paragraph immediately before it, since that's where captions
    conventionally sit. Returns the matched caption string, or None if
    nothing looks like one (not every table has an explicit caption).
    """
    if not nearby_text:
        return None
    match = _CAPTION_PATTERN.search(nearby_text)
    return match.group(0).strip() if match else None


def generate_table_id(source_filename: str, index: int) -> str:
    """
    Builds a stable, human-readable table_id, e.g. 'who_report_table_0007',
    so it's traceable back to its source file just by reading the ID.
    """
    stem = Path(source_filename).stem.replace(" ", "_")
    return f"{stem}_table_{index:04d}"


def _load_lookup() -> list[dict]:
    if os.path.exists(TABLES_LOOKUP_PATH):
        with open(TABLES_LOOKUP_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def _save_lookup(entries: list[dict]) -> None:
    os.makedirs(os.path.dirname(TABLES_LOOKUP_PATH), exist_ok=True)
    with open(TABLES_LOOKUP_PATH, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)


def summarize_table_with_llm(markdown_table: str) -> str:
    """
    Shared summarization helper — used by every ingestion module, so PDF,
    DOCX, and OCR tables are all summarized with the same prompt instead of
    duplicating this logic three times. Falls back to the raw markdown
    table if no GEMINI_API_KEY is set, or if the call fails for any reason
    (rate limit, quota, network) — a summarization hiccup should never
    crash ingestion.
    """
    if not os.environ.get("GEMINI_API_KEY"):
        print("[table_registry] GEMINI_API_KEY not found in environment "
              "(check your .env file and that it's in the project root) — "
              "using raw markdown table instead of an LLM summary.")
        return markdown_table
    try:
        from google import genai
        client = genai.Client()
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=(
                "Convert the following table into a short, plain-prose "
                "summary paragraph that preserves every data point and "
                "row/column relationship. Do not omit any numbers or "
                "labels. Do not add commentary or interpretation beyond "
                "what the table states.\n\n" + markdown_table
            ),
        )
        return response.text.strip()
    except Exception as e:
        print(f"[table_registry] LLM summarization failed ({e}); falling back to raw markdown table.")
        return markdown_table


def register_table(
    table_id: str,
    source_filename: str,
    page_number: Optional[int],
    section_heading: Optional[str],
    caption: Optional[str],
    raw_markdown: str,
    summary: str,
) -> None:
    """
    Writes one table into BOTH registry files — an index entry appended to
    tables_lookup.json, and a summary section appended to tables_summary.md.
    Called once per detected table, from any ingestion module.
    """
    entries = _load_lookup()
    entries.append({
        "table_id": table_id,
        "caption": caption,
        "source_filename": source_filename,
        "page_number": page_number,
        "section_heading": section_heading,
    })
    _save_lookup(entries)

    os.makedirs(os.path.dirname(TABLES_SUMMARY_PATH), exist_ok=True)
    with open(TABLES_SUMMARY_PATH, "a", encoding="utf-8") as f:
        f.write(f"## {table_id}\n\n")
        if caption:
            f.write(f"**Caption:** {caption}\n\n")
        f.write(f"**Source:** {source_filename}")
        if page_number is not None:
            f.write(f", page {page_number}")
        if section_heading:
            f.write(f', under "{section_heading}"')
        f.write("\n\n")
        f.write(f"**Summary:**\n\n{summary}\n\n")
        f.write(f"<details><summary>Raw table (markdown)</summary>\n\n{raw_markdown}\n\n</details>\n\n")
        f.write("---\n\n")


def reset_registry() -> None:
    """
    Clears both registry files. Call this ONCE at the start of a full
    ingestion run across all your documents — otherwise re-running any
    ingestion script keeps appending and you get duplicate entries.
    """
    if os.path.exists(TABLES_LOOKUP_PATH):
        os.remove(TABLES_LOOKUP_PATH)
    if os.path.exists(TABLES_SUMMARY_PATH):
        os.remove(TABLES_SUMMARY_PATH)


def lookup_table_by_caption(query: str) -> list[dict]:
    """
    Plain substring lookup against tables_lookup.json — this is the
    non-vector lookup the "explain table 2.13" flow uses. Returns a list
    since a loose query like "2" could legitimately match several tables.
    """
    entries = _load_lookup()
    query_lower = query.lower()
    return [e for e in entries if e.get("caption") and query_lower in e["caption"].lower()]


def get_table_summary(table_id: str) -> Optional[str]:
    """Pulls just the Summary text for one table_id out of tables_summary.md."""
    if not os.path.exists(TABLES_SUMMARY_PATH):
        return None
    with open(TABLES_SUMMARY_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    marker = f"## {table_id}\n"
    start = content.find(marker)
    if start == -1:
        return None
    end = content.find("\n---\n", start)
    section = content[start:end if end != -1 else None]

    summary_marker = "**Summary:**\n\n"
    s_start = section.find(summary_marker)
    if s_start == -1:
        return None
    s_start += len(summary_marker)
    s_end = section.find("\n\n<details>", s_start)
    return section[s_start:s_end if s_end != -1 else None].strip()