"""
LiveCodeBench loading, cleaning, and test case merging module.

Dataset column format (livecodebench/code_generation_lite):
  public_test_cases  — JSON string → list of {input, output, testtype}
  private_test_cases — base64 → zlib → pickle → JSON string → same format
"""
import base64
import json
import pickle
import random
import re
import zlib

from datasets import load_dataset


def load_lcb(
    split: str = "test",
    difficulty: str = "all",
    sample_size: int = None,
) -> list:
    """
    Loads, cleans, and merges test cases from the LCB dataset.

    Args:
        split:       "test" | "val"
        difficulty:  "all" | "easy" | "medium" | "hard"
        sample_size: None = all, int = first N examples

    Returns:
        List of dicts with keys:
        id, title, content, difficulty, starter_code,
        public_tests, private_tests, all_tests
        Each tests dict: {"input": [str, ...], "output": [str, ...]}
    """
    ds = load_dataset(
        "livecodebench/code_generation_lite",
        split=split,
        trust_remote_code=True,
    )

    rows = []
    for item in ds:
        if difficulty != "all" and item["difficulty"] != difficulty:
            continue

        public  = _parse_public_tests(item.get("public_test_cases",  "[]"))
        private = _parse_private_tests(item.get("private_test_cases", ""))

        all_inputs  = public["input"]  + private["input"]
        all_outputs = public["output"] + private["output"]

        rows.append({
            "id":           item["question_id"],
            "title":        item["question_title"],
            "content":      _clean_html(item["question_content"]),
            "difficulty":   item["difficulty"],
            "starter_code": item.get("starter_code", ""),
            "public_tests":  public,
            "private_tests": private,
            "all_tests": {
                "input":  all_inputs,
                "output": all_outputs,
            },
        })

    if sample_size is not None:
        rng  = random.Random(42)
        rows = rng.sample(rows, min(sample_size, len(rows)))

    return rows


def _parse_public_tests(raw: str) -> dict:
    """
    Parses public_test_cases column.
    Format: JSON string → list of {input, output, testtype}
    Returns: {"input": [str, ...], "output": [str, ...]}
    """
    try:
        cases = json.loads(raw) if raw else []
        return {
            "input":  [c["input"]  for c in cases if "input"  in c],
            "output": [c["output"] for c in cases if "output" in c],
        }
    except Exception:
        return {"input": [], "output": []}


def _parse_private_tests(raw: str) -> dict:
    """
    Parses private_test_cases column.
    Format: base64 → zlib → pickle → JSON string → list of {input, output, testtype}
    Returns: {"input": [str, ...], "output": [str, ...]}
    """
    if not raw:
        return {"input": [], "output": []}
    try:
        decompressed = zlib.decompress(base64.b64decode(raw))
        json_str     = pickle.loads(decompressed)
        cases        = json.loads(json_str)
        return {
            "input":  [c["input"]  for c in cases if "input"  in c],
            "output": [c["output"] for c in cases if "output" in c],
        }
    except Exception:
        return {"input": [], "output": []}


def _clean_html(text: str) -> str:
    """Removes HTML tags and converts HTML entities to plain text."""
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"&nbsp;",  " ",  text)
    text = re.sub(r"&lt;",    "<",  text)
    text = re.sub(r"&gt;",    ">",  text)
    text = re.sub(r"&amp;",   "&",  text)
    text = re.sub(r"\n{3,}",  "\n\n", text)
    return text.strip()
