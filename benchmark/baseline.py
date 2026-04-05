"""
Single-shot baseline runner — V0 variant.

Generates code once per model without any pipeline.
Results are used for ablation comparison.

Usage:
    python benchmark/baseline.py --model M1 --sample 50 --tag v0_M1
    python benchmark/baseline.py --model M2 --sample 50 --tag v0_M2
    python benchmark/baseline.py --model M3 --sample 50 --tag v0_M3
"""
import argparse
import json
import os
import re
import time
from datetime import datetime

from loguru import logger

from config import settings, model_registry
from dataset.loader import load_lcb
from dataset.splitter import format_problem
from dataset.edge_cases import augment_row
from models.client import get_client
from sandbox.lcb_eval import lcb_evaluate
from memory.logger import write_run_info


def _extract_code(text: str) -> str:
    """
    Extracts ```python ... ``` block from LLM output.
    Falls back to the full text if no block is found.
    """
    # Take the last ```python...``` block (guard against lcb_runner bug)
    matches = re.findall(r"```python\s*(.*?)```", text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    # Backtick block without language tag
    matches = re.findall(r"```\s*(.*?)```", text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return text.strip()


def run_single_shot(
    problem_row: dict,
    model_id: str,
    run_id: str,
) -> dict:
    """
    Generates code once for a single model without any pipeline feedback.
    Evaluates correctness via lcb_runner or fallback.
    """
    aug_row    = augment_row(problem_row)
    edge_cases = aug_row.get("edge_cases", [])
    prompt     = format_problem(problem_row)

    cfg    = model_registry.get(model_id)
    client = get_client(model_id)

    t0 = time.perf_counter()

    try:
        response = client.chat.completions.create(
            model=cfg.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=cfg.temperature,
            max_tokens=2048,
        )
        code   = _extract_code(response.choices[0].message.content)
        tokens = {
            "prompt":     response.usage.prompt_tokens     if response.usage else 0,
            "completion": response.usage.completion_tokens if response.usage else 0,
        }
    except Exception as e:
        logger.warning(f"[{model_id}] Codegen error: {e}")
        code   = ""
        tokens = {"prompt": 0, "completion": 0}

    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    # Correctness evaluation — lcb_runner or subprocess fallback
    eval_result = lcb_evaluate(
        code=code,
        problem_row=aug_row,
        edge_cases=edge_cases,
        timeout=settings.sandbox_timeout_sec,
    )

    passed = eval_result["pass_rate"] == 1.0 and eval_result["total_count"] > 0

    return {
        "run_id":       run_id,
        "model":        model_id,
        "model_name":   cfg.model_name,
        "question_id":  problem_row.get("question_id", problem_row.get("id", "unknown")),
        "difficulty":   problem_row.get("difficulty", "medium"),
        "variant":      "V0",
        "passed":       passed,
        "pass_rate":    eval_result["pass_rate"],
        "passed_count": eval_result["passed_count"],
        "total_count":  eval_result["total_count"],
        "elapsed_ms":   elapsed_ms,
        "tokens":       tokens,
        "code":         code,
        "failed_cases": eval_result["failed_cases"],
    }


def run_baseline(
    model_id: str,
    difficulty: str = "all",
    sample_size: int = 50,
    output_tag: str = "",
) -> dict:
    """
    Runs the full V0 baseline for a model over the given sample.
    Writes results to outputs/results/baseline_<tag>_<ts>.json.
    """
    run_id = write_run_info()
    rows   = load_lcb(
        split=settings.lcb_split,
        difficulty=difficulty,
        sample_size=sample_size,
    )

    logger.info(
        f"Baseline | run_id={run_id} | model={model_id} | "
        f"{len(rows)} problems | difficulty={difficulty}"
    )

    results      = []
    pass_counts  = {"easy": 0, "medium": 0, "hard": 0}
    total_counts = {"easy": 0, "medium": 0, "hard": 0}

    for i, row in enumerate(rows):
        diff = row.get("difficulty", "medium")
        logger.info(f"[{i+1}/{len(rows)}] {row.get('title', '')} ({diff})")
        result = run_single_shot(row, model_id, run_id)
        results.append(result)

        total_counts[diff] = total_counts.get(diff, 0) + 1
        if result["passed"]:
            pass_counts[diff] = pass_counts.get(diff, 0) + 1

    summary = {
        "run_id":    run_id,
        "variant":   "V0",
        "model":     model_id,
        "timestamp": datetime.now().isoformat(),
        "total":     len(rows),
        "tsr": {
            d: round(pass_counts.get(d, 0) / total_counts[d], 3)
            if total_counts.get(d, 0) > 0 else 0.0
            for d in ("easy", "medium", "hard")
        },
        "results": results,
    }

    date_dir   = datetime.now().strftime("%Y-%m-%d")
    output_dir = os.path.join(settings.output_dir, date_dir)
    os.makedirs(output_dir, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag  = f"_{output_tag}" if output_tag else ""
    path = os.path.join(output_dir, f"baseline{tag}_{ts}.json")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(f"Done → {path}")
    logger.info(f"TSR: {summary['tsr']}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM-as-a-Judge — V0 Single-Shot Baseline")
    parser.add_argument("--model",      required=True, choices=["M1", "M2", "M3"])
    parser.add_argument("--difficulty", default="all")
    parser.add_argument("--sample",     type=int, default=50)
    parser.add_argument("--tag",        default="")
    args = parser.parse_args()

    run_baseline(
        model_id    = args.model,
        difficulty  = args.difficulty,
        sample_size = args.sample,
        output_tag  = args.tag,
    )
