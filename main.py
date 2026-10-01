"""
Pipeline entry point.
Runs M1 → M2 → M3 sequentially for each LCB problem.
"""
import os

from dotenv import load_dotenv
from loguru import logger

from pipeline.graph   import build_graph
from pipeline.state   import PipelineState
from config           import settings
from dataset.loader   import load_lcb
from dataset.splitter import format_problem
from dataset.edge_cases import augment_row
from memory.logger    import write_run_info

load_dotenv()

_EMPTY_TIMING = {
    "codegen_ms": 0,
    "node1_ms":   0,
    "node2_ms":   0,
    "node3_ms":   0,
    "total_ms":   0,
}
_EMPTY_TOKENS = {
    "codegen_prompt":      0,
    "codegen_completion":  0,
    "judgement_prompt":     0,
    "judgement_completion": 0,
    "total":               0,
}


def run_pipeline_for_problem(problem_row: dict, run_id: str) -> dict:
    """
    Runs the full M1→M2→M3 pipeline for a single LCB problem.
    Each model gets its own LangGraph invocation with a fresh initial state.

    Returns a summary dict with per-model status and score.
    """
    graph = build_graph()

    aug_row      = augment_row(problem_row)
    edge_cases   = aug_row.get("edge_cases", [])
    pub_inputs   = problem_row.get("public_tests", {}).get("input",  [])
    pub_outputs  = problem_row.get("public_tests", {}).get("output", [])
    problem_text = format_problem(problem_row)
    question_id  = problem_row.get("question_id", problem_row.get("id", "unknown"))
    difficulty   = problem_row.get("difficulty", "medium")

    all_results = []

    for model_id in settings.model_sequence:
        logger.info(f"[{model_id}] starting — {problem_row.get('title', question_id)}")

        initial_state: PipelineState = {
            # Run metadata
            "run_id":         run_id,
            "active_model":   model_id,
            "variant":        "V3",
            "iteration":      0,
            "max_iterations": settings.max_iterations,

            # Problem input
            "problem":      problem_text,
            "problem_row":  aug_row,
            "question_id":  question_id,
            "difficulty":   difficulty,
            "test_inputs":  pub_inputs,
            "test_outputs": pub_outputs,
            "edge_cases":   edge_cases,

            # Code (empty until codegen runs)
            "generated_code": "",
            "stripped_code":  "",

            # Node 1 defaults
            "node1_cc":                  0.0,
            "node1_mi":                  100.0,
            "node1_halstead":            0.0,
            "node1_bandit_high":         False,
            "node1_bandit_medium_count": 0,
            "node1_severity":            "OK",
            "node1_issues":              [],

            # Node 2 defaults
            "node2_pass_rate":           0.0,
            "node2_passed_count":        0,
            "node2_total_count":         0,
            "node2_failed_cases":        [],
            "node2_crash_count":         0,
            "node2_test_diversity_score": 1.0,
            "node2_lcb_available":       False,
            "node2_cpu_ms":              0.0,
            "node2_ram_mb":              0.0,
            "node2_timed_out":           False,

            # Node 3 defaults
            "node3_total_score":     0.0,
            "node3_pass":            False,
            "node3_score_breakdown": {},
            "node3_failure_reasons": [],
            "node3_improved_prompt": "",

            # Performance metrics
            "timing": dict(_EMPTY_TIMING),
            "tokens": dict(_EMPTY_TOKENS),

            # Log
            "memory_log":   [],
            "final_status": "PENDING",
        }

        final_state = graph.invoke(initial_state)
        status      = final_state["final_status"]
        score       = final_state.get("node3_total_score", 0.0)

        logger.info(f"[{model_id}] done — status={status} score={score:.1f}")
        all_results.append({"model": model_id, "status": status, "score": score})

        # On PASS, remaining models still run per the sequential design
        # (each model is independent; M1 PASS does not skip M2)

    return {
        "problem_id": question_id,
        "difficulty": difficulty,
        "results":    all_results,
    }


def main():
    logger.info("Pipeline starting...")

    run_id = write_run_info()
    logger.info(f"run_id: {run_id}")

    rows = load_lcb(
        split=settings.lcb_split,
        difficulty=settings.lcb_difficulty,
        sample_size=settings.lcb_sample_size,
    )
    logger.info(f"{len(rows)} problems loaded.")

    # Optional sharding so N processes can share one llama-server started with
    # --parallel N. Slicing happens after the seeded sample, so the union of
    # all shards is exactly the same problem set as a single-process run.
    shard_count = int(os.getenv("SHARD_COUNT", "1"))
    shard_index = int(os.getenv("SHARD_INDEX", "0"))
    if shard_count > 1:
        rows = rows[shard_index::shard_count]
        logger.info(f"Shard {shard_index}/{shard_count}: {len(rows)} problems.")

    # Optional resume: skip problems already finished in an earlier run
    # (comma-separated question_ids). Order of the seeded sample is kept.
    skip = {q.strip() for q in os.getenv("SKIP_QIDS", "").split(",") if q.strip()}
    if skip:
        rows = [r for r in rows if r.get("question_id", r.get("id")) not in skip]
        logger.info(f"Skipping {len(skip)} finished problems; {len(rows)} remain.")

    # Optional cap: first N problems of the seeded sample (e.g. a 50-problem run).
    max_problems = int(os.getenv("MAX_PROBLEMS", "0"))
    if max_problems > 0:
        rows = rows[:max_problems]
        logger.info(f"Limiting to the first {len(rows)} problems of the sample.")

    all_results = []
    for row in rows:
        result = run_pipeline_for_problem(row, run_id)
        logger.info(f"Problem done: {result}")
        all_results.append(result)

    # ── Summary ───────────────────────────────────────────────────────────────
    status_counts = {"PASS": 0, "STOP_MAX": 0, "STOP_HIGH": 0}
    scores = []
    for res in all_results:
        for m in res["results"]:
            s = m["status"]
            status_counts[s] = status_counts.get(s, 0) + 1
            if m["score"] > 0:
                scores.append(m["score"])

    total_runs = sum(status_counts.values())
    avg_score  = round(sum(scores) / len(scores), 1) if scores else 0.0

    logger.info("─" * 42)
    logger.info("Pipeline complete.")
    logger.info(f"  Problems  : {len(all_results)}")
    logger.info(f"  Total runs: {total_runs}  (models × problems)")
    for status, count in status_counts.items():
        pct = round(count / total_runs * 100, 1) if total_runs else 0.0
        logger.info(f"  {status:<10}: {count}  ({pct}%)")
    logger.info(f"  Avg score : {avg_score}")
    logger.info("─" * 42)


if __name__ == "__main__":
    main()
