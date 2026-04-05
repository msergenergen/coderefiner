"""
Format conversion, statistics, and problem formatting.
"""
from collections import Counter
from dataset.loader import load_lcb


def format_problem(row: dict) -> str:
    """
    Converts an LCB question to the format added to the LLM codegen system prompt.

    Only public_tests examples are shown — private tests remain hidden.
    """
    parts = [
        f"## Problem: {row['title']}",
        f"Difficulty: {row['difficulty']}",
        "",
        row["content"],
    ]

    # Starter code is rarely populated; include if present
    if row.get("starter_code", "").strip():
        parts += ["", "## Starter Code",
                  "```python", row["starter_code"], "```"]

    # Only public examples are shown
    pub_in  = row["public_tests"].get("input",  [])
    pub_out = row["public_tests"].get("output", [])

    if pub_in:
        parts += ["", "## Examples"]
        for i, (inp, out) in enumerate(zip(pub_in, pub_out), 1):
            parts += [f"**Example {i}:**",
                      f"Input:\n{inp}",
                      f"Output:\n{out}", ""]

    parts += [
        "## Output Format",
        "Read input from stdin. Write answer to stdout.",
        "Output only Python code inside ```python ... ``` markers.",
    ]

    return "\n".join(parts)


def dataset_stats(rows: list) -> dict:
    counts = Counter(r["difficulty"] for r in rows)
    return {
        "total":  len(rows),
        "easy":   counts.get("easy",   0),
        "medium": counts.get("medium", 0),
        "hard":   counts.get("hard",   0),
    }
