"""
Faz 1 — Dataset Builder

Scans all iteration JSON files under data.source_dir, applies quality
filters, formats each example as a Prometheus-style chat prompt, splits
by question_id (no leakage), and saves HuggingFace Dataset splits.

Usage:
    python build_dataset.py
    python build_dataset.py --source-dir ../outputs/results --output-dir ./data
    python build_dataset.py --config config.yaml
"""

import argparse
import json
import os
import random
from glob import glob
from pathlib import Path
from typing import Optional

import yaml
from datasets import Dataset, DatasetDict


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path: str = "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Prompt template ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a specialized code judge for Python competitive programming solutions. "
    "Evaluate the given code using the rubric and reference tests, then respond with "
    "a single valid JSON object only — no markdown, no explanation."
)

RUBRIC = """\
- accuracy (0–25): Does the code correctly solve all reference test cases?
- efficiency (0–20): Is time/space complexity appropriate for the problem constraints?
- quality (0–20): Is the code readable, well-structured, and maintainable?
- security (0–15): Are there dangerous imports, eval/exec, or unsafe patterns?
- algorithm_choice (0–10): Does the code use the right algorithm for the given constraints?
- error_handling (0–10): Are edge cases and invalid inputs handled correctly?"""


def format_prompt(example: dict) -> str:
    """Returns the full user-turn string for the chat template."""
    all_tests_str = json.dumps(example["all_tests"], ensure_ascii=False, indent=2)
    return (
        f"[PROBLEM]\n{example['problem']}\n\n"
        f"[CODE]\n```python\n{example['stripped_code']}\n```\n\n"
        f"[RUBRIC]\n{RUBRIC}\n\n"
        f"[REFERENCE_TESTS]\n{all_tests_str}"
    )


def format_completion(example: dict) -> str:
    """Returns the assistant-turn JSON string."""
    output = {
        "breakdown":       example["breakdown"],
        "total_score":     example["total_score"],
        "pass":            example["pass"],
        "failure_reasons": example["failure_reasons"],
        "improved_prompt": example["improved_prompt"],
    }
    return json.dumps(output, ensure_ascii=False, indent=2)


def format_chat(example: dict, tokenizer=None) -> str:
    """
    Formats one example as a ChatML string.
    If tokenizer is provided, uses apply_chat_template for model-specific format.
    """
    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": format_prompt(example)},
        {"role": "assistant", "content": format_completion(example)},
    ]
    if tokenizer is not None:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    # Fallback: manual ChatML
    parts = []
    for m in messages:
        parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
    return "\n".join(parts) + "\n"


# ── JSON loading + filtering ──────────────────────────────────────────────────

def _has_json_parse_error(failure_reasons: list) -> bool:
    return any("json_parse_error" in str(r).lower() for r in failure_reasons)


def load_and_filter(source_dir: str, filters: dict) -> list[dict]:
    """
    Walks source_dir for all *.json files, loads each, applies quality filters.
    Returns a list of raw example dicts ready for formatting.
    """
    pattern = os.path.join(source_dir, "**", "*.json")
    paths = glob(pattern, recursive=True)

    examples = []
    skipped = {"no_training_block": 0, "empty_code": 0, "empty_prompt": 0,
               "json_parse_error": 0, "stop_high": 0, "score_floor": 0,
               "not_json": 0}

    for path in paths:
        # Skip non-iteration files (run_info.json etc.)
        if "iterations" not in path:
            continue

        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            skipped["not_json"] += 1
            continue

        training = data.get("training")
        if not training:
            skipped["no_training_block"] += 1
            continue

        status = data.get("status", "")
        node3 = data.get("node3", {})
        failure_reasons = node3.get("failure_reasons", [])

        if filters.get("exclude_stop_high") and status == "STOP_HIGH":
            skipped["stop_high"] += 1
            continue

        if filters.get("require_stripped_code") and not training.get("stripped_code", "").strip():
            skipped["empty_code"] += 1
            continue

        if filters.get("require_improved_prompt") and not node3.get("improved_prompt", "").strip():
            skipped["empty_prompt"] += 1
            continue

        if filters.get("exclude_json_parse_error") and _has_json_parse_error(failure_reasons):
            skipped["json_parse_error"] += 1
            continue

        min_score = filters.get("min_total_score", 0.0)
        if node3.get("total_score", 0.0) < min_score:
            skipped["score_floor"] += 1
            continue

        examples.append({
            "question_id":   data.get("question_id", "unknown"),
            "difficulty":    data.get("difficulty", "unknown"),
            "model":         data.get("model", "unknown"),
            "iteration":     data.get("iter", 0),
            "status":        status,
            "problem":       training.get("problem", ""),
            "stripped_code": training.get("stripped_code", ""),
            "all_tests":     training.get("all_tests", []),
            "breakdown":     node3.get("breakdown", {}),
            "total_score":   node3.get("total_score", 0.0),
            "pass":          node3.get("pass", False),
            "failure_reasons": failure_reasons,
            "improved_prompt": node3.get("improved_prompt", ""),
            "node2_pass_rate": data.get("node2", {}).get("pass_rate", 0.0),
        })

    total = len(examples) + sum(skipped.values())
    print(f"Scanned {total} files → kept {len(examples)} examples")
    print(f"  Skipped: {skipped}")
    return examples


# ── Split by question_id ──────────────────────────────────────────────────────

def split_by_question_id(examples: list[dict], cfg: dict, seed: int = 42) -> dict[str, list]:
    q_ids = sorted(set(e["question_id"] for e in examples))
    random.seed(seed)
    random.shuffle(q_ids)

    n = len(q_ids)
    n_val  = max(1, int(n * cfg["val"]))
    n_test = max(1, int(n * cfg["test"]))

    test_ids  = set(q_ids[:n_test])
    val_ids   = set(q_ids[n_test:n_test + n_val])
    train_ids = set(q_ids[n_test + n_val:])

    splits = {"train": [], "val": [], "test": []}
    for e in examples:
        if e["question_id"] in test_ids:
            splits["test"].append(e)
        elif e["question_id"] in val_ids:
            splits["val"].append(e)
        else:
            splits["train"].append(e)

    for name, items in splits.items():
        print(f"  {name:5s}: {len(items):5d} examples ({len(set(e['question_id'] for e in items))} q_ids)")

    return splits


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build judge model SFT dataset from iteration JSONs")
    parser.add_argument("--config",     default="config.yaml")
    parser.add_argument("--source-dir", default=None, help="Override data.source_dir")
    parser.add_argument("--output-dir", default=None, help="Override data.output_dir")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]

    source_dir = args.source_dir or data_cfg["source_dir"]
    output_dir = args.output_dir or data_cfg["output_dir"]
    source_dir = os.path.abspath(source_dir)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")

    examples = load_and_filter(source_dir, data_cfg["quality_filters"])
    if not examples:
        print("ERROR: No examples passed the quality filters. Run cloud pipeline first (Faz 0).")
        return

    print(f"\nSplitting {len(examples)} examples by question_id …")
    splits = split_by_question_id(examples, data_cfg["split"], seed=data_cfg["split"]["seed"])

    # Format chat strings and build HuggingFace datasets
    hf_splits = {}
    for split_name, items in splits.items():
        rows = [{"text": format_chat(e), **e} for e in items]
        hf_splits[split_name] = Dataset.from_list(rows)

    dataset = DatasetDict(hf_splits)
    dataset.save_to_disk(output_dir)
    print(f"\nDataset saved → {output_dir}")

    # Save a small sample for inspection
    sample_path = os.path.join(output_dir, "sample.jsonl")
    with open(sample_path, "w", encoding="utf-8") as f:
        for e in splits["train"][:3]:
            f.write(json.dumps({
                "prompt": format_prompt(e),
                "completion": format_completion(e),
            }, ensure_ascii=False) + "\n")
    print(f"Sample (3 train examples) → {sample_path}")


if __name__ == "__main__":
    main()
