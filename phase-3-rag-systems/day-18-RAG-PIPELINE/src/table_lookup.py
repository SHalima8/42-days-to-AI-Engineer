"""
table_lookup.py
Resolves a table_id (from a TABLE-type chunk's placeholder) into the actual
table content — pulled from tables_summary.md, since tables_lookup.json only
holds metadata (caption, source, section) not the summary/raw markdown itself.

tables_summary.md structure per entry:
    ## {table_id}
    **Caption:** ...
    **Source:** ...
    **Summary:**
    <text>
    <details>...</details>
    ---
"""

import re
from functools import lru_cache
from src import config


@lru_cache(maxsize=1)
def _load_summary_file() -> str:
    """Cached so we only read/parse the .md file once per process, not per lookup."""
    with open(config.TABLES_SUMMARY_PATH, "r", encoding="utf-8") as f:
        return f.read()


def get_table_block(table_id: str) -> str:
    """
    Returns the full markdown block (caption, source, summary, raw table)
    for a given table_id. Returns a fallback string if not found, rather
    than raising — a missing table shouldn't crash the whole pipeline.
    """
    content = _load_summary_file()

    # Match from "## {table_id}" up to the next "---" (or end of file)
    pattern = rf"## {re.escape(table_id)}\n(.*?)(?=\n---|\Z)"
    match = re.search(pattern, content, re.DOTALL)

    if not match:
        return f"[Table {table_id} referenced but not found in tables_summary.md]"

    return match.group(1).strip()


def get_table_summary_only(table_id: str) -> str:
    """
    Same as get_table_block but strips the raw markdown <details> block —
    useful if you want just the natural-language summary injected into the
    prompt (cheaper on tokens) rather than the full raw table too.
    """
    block = get_table_block(table_id)
    # cut off everything from the <details> tag onward
    return block.split("<details>")[0].strip()