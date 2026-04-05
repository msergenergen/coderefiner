"""
LangGraph Node 2: Sandbox & Execution

Phase 1: lcb_runner correctness (all_tests)
Phase 2: Own runner runtime metrics (public_tests)
"""
import time
from pipeline.state import PipelineState
from sandbox.lcb_eval import lcb_evaluate
from sandbox.runner  import measure_runtime
from dataset.edge_cases import augment_row


def sandbox_node(state: PipelineState) -> PipelineState:
    """
    Node 2: Two-phase evaluation.

    Reads from state:
        generated_code, problem_row, test_inputs (public)

    Writes to state:
        node2_pass_rate, node2_passed_count, node2_total_count,
        node2_cpu_ms, node2_ram_mb, node2_timed_out,
        node2_crash_count, node2_failed_cases,
        node2_lcb_available
    """
    t0   = time.perf_counter()
    code = state["generated_code"]

    problem_row = state.get("problem_row", {})
    edge_cases  = state.get("edge_cases", [])
    pub_inputs  = state.get("test_inputs", [])   # public_tests["input"]

    # ── Phase 1: lcb_runner correctness ──────────────────────────────────
    faz1 = lcb_evaluate(
        code=code,
        problem_row=problem_row,
        edge_cases=edge_cases,
        timeout=10,
    )

    # ── Phase 2: Runtime metrics (public tests) ──────────────────────────
    faz2 = {"cpu_ms": 0.0, "ram_mb": 0.0, "timed_out": False, "error": None}

    if pub_inputs:
        faz2 = measure_runtime(
            code=code,
            test_inputs=pub_inputs,
        )

    node2_ms = int((time.perf_counter() - t0) * 1000)

    # Update timing in state
    timing = dict(state.get("timing", {}))
    timing["node2_ms"] = node2_ms
    timing["total_ms"] = sum(
        timing.get(k, 0)
        for k in ("codegen_ms", "node1_ms", "node2_ms", "node3_ms")
    )

    return {
        **state,
        # Correctness (Phase 1)
        "node2_pass_rate":            faz1["pass_rate"],
        "node2_passed_count":         faz1["passed_count"],
        "node2_total_count":          faz1["total_count"],
        "node2_failed_cases":         faz1["failed_cases"],
        "node2_crash_count":          faz1["crash_count"],
        "node2_test_diversity_score": faz1["test_diversity_score"],
        "node2_lcb_available":        faz1["lcb_available"],
        # Runtime (Phase 2)
        "node2_cpu_ms":    faz2["cpu_ms"],
        "node2_ram_mb":    faz2["ram_mb"],
        "node2_timed_out": faz2["timed_out"],
        # Timing
        "timing": timing,
    }
