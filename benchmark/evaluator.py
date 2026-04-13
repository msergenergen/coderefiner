"""
Ablation variant runner — V1/V2/V3.

Runs the full pipeline for multiple LCB problems with a given variant.
Results are written to outputs/results/benchmark_<variant>_<tag>_<ts>.json.

Usage:
    python -m benchmark.evaluator --variant V3 --sample 50 --tag v3_full
    python -m benchmark.evaluator --variant V2 --sample 50 --tag v2
    python -m benchmark.evaluator --variant V1 --difficulty easy --sample 25
"""
import argparse
import json
import os
from datetime import datetime

from dotenv import load_dotenv
from loguru import logger

from config import settings
from dataset.loader import load_lcb
from memory.logger import write_run_info
from pipeline.graph import build_graph
from pipeline.state import PipelineState
from dataset.splitter import format_problem
from dataset.edge_cases import augment_row

load_dotenv()

_EMPTY_TIMING = {
    "codegen_ms": 0, "node1_ms": 0,
    "node2_ms": 0,   "node3_ms": 0, "total_ms": 0,
}
_EMPTY_TOKENS = {
    "codegen_prompt": 0, "codegen_completion": 0,
    "judgement_prompt": 0, "judgement_completion": 0, "total": 0,
}


def run_variant_for_problem(
    problem_row: dict,
    run_id: str,
    variant: str,
) -> dict:
    """
    Runs the M1→M2→M3 pipeline for a single problem with the given variant.
    Returns a summary dict with per-model status and score.
    """
    graph = build_graph(variant=variant)

    aug_row      = augment_row(problem_row)
    edge_cases   = aug_row.get("edge_cases", [])
    pub_inputs   = problem_row.get("public_tests", {}).get("input",  [])
    pub_outputs  = problem_row.get("public_tests", {}).get("output", [])
    problem_text = format_problem(problem_row)
    question_id  = problem_row.get("question_id", problem_row.get("id", "unknown"))
    difficulty   = problem_row.get("difficulty", "medium")

    all_results = []

    for model_id in settings.model_sequence:
        logger.info(f"[{variant}][{model_id}] starting — {problem_row.get('title', question_id)}")

        initial_state: PipelineState = {
            "run_id":         run_id,
            "active_model":   model_id,
            "variant":        variant,
            "iteration":      0,
            "max_iterations": settings.max_iterations,
            "problem":        problem_text,
            "problem_row":    aug_row,
            "question_id":    question_id,
            "difficulty":     difficulty,
            "test_inputs":    pub_inputs,
            "test_outputs":   pub_outputs,
            "edge_cases":     edge_cases,
            "generated_code": "",
            "stripped_code":  "",
            "node1_cc":                  0.0,
            "node1_mi":                  100.0,
            "node1_halstead":            0.0,
            "node1_bandit_high":         False,
            "node1_bandit_medium_count": 0,
            "node1_severity":            "OK",
            "node1_issues":              [],
            "node2_pass_rate":           0.0,
            "node2_passed_count":        0,
            "node2_total_count":         0,
            "node2_failed_cases":        [],
            "node2_crash_count":         0,
            "node2_test_diversity_score": 1.0,
            "node2_lcb_available":       False,
            "node2_cpu_ms":    0.0,
            "node2_ram_mb":    0.0,
            "node2_timed_out": False,
            "node3_total_score":     0.0,
            "node3_pass":            False,
            "node3_score_breakdown": {},
            "node3_failure_reasons": [],
            "node3_improved_prompt": "",
            "timing": dict(_EMPTY_TIMING),
            "tokens": dict(_EMPTY_TOKENS),
            "memory_log":   [],
            "final_status": "PENDING",
        }

        final_state = graph.invoke(initial_state)
        status      = final_state["final_status"]
        score       = final_state.get("node3_total_score", 0.0)

        final_pass_rate = final_state.get("node2_pass_rate", 0.0)
        logger.info(
            f"[{variant}][{model_id}] done — status={status} "
            f"score={score:.1f} pass_rate={final_pass_rate:.2f}"
        )
        all_results.append({
            "model":           model_id,
            "status":          status,
            "score":           score,
            "final_pass_rate": final_pass_rate,   # son iterasyonun lcb pass_rate'i
        })

    return {
        "problem_id": question_id,
        "difficulty": difficulty,
        "results":    all_results,
    }


def run_evaluator(
    variant: str = "V3",
    difficulty: str = "all",
    sample_size: int = 50,
    output_tag: str = "",
) -> dict:
    """
    Runs the full ablation variant over the given sample.
    Writes results to outputs/results/benchmark_<variant>_<tag>_<ts>.json.
    """
    run_id = write_run_info()
    rows   = load_lcb(
        split=settings.lcb_split,
        difficulty=difficulty,
        sample_size=sample_size,
    )

    logger.info(
        f"Evaluator | run_id={run_id} | variant={variant} | "
        f"{len(rows)} problems | difficulty={difficulty}"
    )

    results      = []
    pass_counts  = {"easy": 0, "medium": 0, "hard": 0}
    total_counts = {"easy": 0, "medium": 0, "hard": 0}

    for i, row in enumerate(rows):
        diff = row.get("difficulty", "medium")
        logger.info(f"[{i+1}/{len(rows)}] {row.get('title', '')} ({diff})")
        result = run_variant_for_problem(row, run_id, variant)
        results.append(result)

        total_counts[diff] = total_counts.get(diff, 0) + 1
        # Pipeline PASS: at least one model got PASS
        if any(m["status"] == "PASS" for m in result["results"]):
            pass_counts[diff] = pass_counts.get(diff, 0) + 1

    summary = {
        "run_id":    run_id,
        "variant":   variant,
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
    path = os.path.join(output_dir, f"benchmark_{variant}{tag}_{ts}.json")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(f"Done → {path}")
    logger.info(f"TSR: {summary['tsr']}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM-as-a-Judge — Ablation Variant Evaluator")
    parser.add_argument(
        "--variant",
        default="V3",
        choices=["V1", "V2", "V3"],
        help=(
            "V1: Execution + Judgement (Node 2 + Node 3)  |  "
            "V2: Static Analysis + Judgement (Node 1 + Node 3)  |  "
            "V3: Full pipeline (Node 1 + Node 2 + Node 3)"
        ),
    )
    parser.add_argument("--difficulty", default="all")
    parser.add_argument("--sample",     type=int, default=50)
    parser.add_argument("--tag",        default="")
    args = parser.parse_args()

    run_evaluator(
        variant     = args.variant,
        difficulty  = args.difficulty,
        sample_size = args.sample,
        output_tag  = args.tag,
    )
