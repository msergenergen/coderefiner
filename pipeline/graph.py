"""
LangGraph StateGraph definition.

Each iteration is logged via log_iteration_node → memory.md + JSON:
  LOOP      — after judgement, threshold not met
  PASS      — after judgement, threshold met
  STOP_MAX  — after judgement, max_iter reached
  STOP_HIGH — severity HIGH, pipeline stopped early

Each node writes its own timing to state["timing"].
Codegen and judgement nodes write token counts to state["tokens"].
"""
from langgraph.graph import StateGraph, END

from pipeline.state  import PipelineState
from pipeline.gates  import severity_gate, iteration_gate
from pipeline.router import advance_model

from nodes.codegen         import codegen_node
from nodes.static_analysis import static_analysis_node
from nodes.sandbox         import sandbox_node
from nodes.cleaner         import strip_comments_node
from nodes.judgement        import judgement_node
from memory.logger         import log_iteration_node, log_severity_stop_node


def _write_memory_final_node(state: PipelineState) -> PipelineState:
    """PASS: update final_status. Row was already written by log_iteration_node."""
    return {**state, "final_status": "PASS"}


def _stop_max_node(state: PipelineState) -> PipelineState:
    """STOP_MAX: update final_status. Row was already written by log_iteration_node."""
    return {**state, "final_status": "STOP_MAX"}


def _next_model_node(state: PipelineState) -> PipelineState:
    """Advance to the next model after PASS."""
    return advance_model(state)


def _strip_before_judgement(state: PipelineState) -> PipelineState:
    """
    Used when sandbox is skipped (LOW severity in V3, or entire V2 variant).
    Resets node2 fields to safe defaults then strips comments.

    Must assign literal defaults, not `state.get(key, default)` — state is
    the same dict threaded across iterations within one model/problem run,
    so `.get()` silently returns the PREVIOUS iteration's real sandbox
    result instead of a default whenever the key already exists (which it
    always does after iteration 1). Concretely: if iteration N-1 passed 80%
    of tests and iteration N is skipped to LOW (e.g. a syntax-error
    fallback — see nodes/static_analysis.py), the Judge would otherwise be
    shown iteration N-1's 80% pass rate for code that never even ran.
    """
    updated = {
        **state,
        "node2_pass_rate":            0.0,
        "node2_passed_count":         0,
        "node2_total_count":          0,
        "node2_failed_cases":         [],
        "node2_crash_count":          0,
        "node2_test_diversity_score": 1.0,
        "node2_lcb_available":        False,
        "node2_cpu_ms":               0.0,
        "node2_ram_mb":               0.0,
        "node2_timed_out":            False,
    }
    return strip_comments_node(updated)


def build_graph(variant: str = "V3"):
    """
    Builds and compiles the LangGraph state graph.

    Variant matrix:
        V1 — Node 2 + Node 3        (Execution + Judgement, no static analysis)
        V2 — Node 1 + Node 3        (Static Analysis + Judgement, no execution)
        V3 — Node 1 + Node 2 + Node 3  (Full pipeline)

    PASS criterion (all variants):
        node3_total_score >= threshold AND node3_pass
    """
    use_node1 = variant in ("V2", "V3")
    use_node2 = variant in ("V1", "V3")

    g = StateGraph(PipelineState)

    # ── Core nodes — always present ───────────────────────────────────
    g.add_node("codegen",            codegen_node)
    g.add_node("judgement",           judgement_node)
    g.add_node("log_iteration",      log_iteration_node)
    g.add_node("write_memory_final", _write_memory_final_node)
    g.add_node("stop_max",           _stop_max_node)
    g.add_node("next_model",         _next_model_node)

    # ── Conditional nodes ─────────────────────────────────────────────
    if use_node1:
        g.add_node("static_analysis",    static_analysis_node)
        g.add_node("strip_before_judge", _strip_before_judgement)
        g.add_node("log_severity_stop",  log_severity_stop_node)

    if use_node2:
        g.add_node("sandbox",        sandbox_node)
        g.add_node("strip_comments", strip_comments_node)

    # ── Entry point ──────────────────────────────────────────────────
    g.set_entry_point("codegen")

    # ── Routing ──────────────────────────────────────────────────────
    if use_node1:
        g.add_edge("codegen", "static_analysis")

        if use_node2:
            # V3: OK → sandbox, LOW → strip_before_judge, HIGH → stop
            g.add_conditional_edges(
                "static_analysis",
                severity_gate,
                {
                    "to_node2":  "sandbox",
                    "to_node3":  "strip_before_judge",
                    "stop_high": "log_severity_stop",
                },
            )
            g.add_edge("sandbox",            "strip_comments")
            g.add_edge("strip_comments",     "judgement")
        else:
            # V2: OK and LOW both → strip_before_judge (no sandbox)
            g.add_conditional_edges(
                "static_analysis",
                severity_gate,
                {
                    "to_node2":  "strip_before_judge",  # OK → skip sandbox
                    "to_node3":  "strip_before_judge",  # LOW → skip sandbox
                    "stop_high": "log_severity_stop",
                },
            )

        g.add_edge("strip_before_judge", "judgement")

    else:
        # V1: codegen → sandbox → strip_comments → judgement
        g.add_edge("codegen",        "sandbox")
        g.add_edge("sandbox",        "strip_comments")
        g.add_edge("strip_comments", "judgement")

    # ── Judgement → iteration gate ────────────────────────────────────
    g.add_edge("judgement", "log_iteration")

    g.add_conditional_edges(
        "log_iteration",
        iteration_gate,
        {
            "write_memory_final": "write_memory_final",
            "stop_max":           "stop_max",
            "loop_to_codegen":    "codegen",
        },
    )

    # ── Terminal edges ────────────────────────────────────────────────
    g.add_edge("write_memory_final", "next_model")
    g.add_edge("stop_max",           END)
    g.add_edge("next_model",         END)

    if use_node1:
        g.add_edge("log_severity_stop", END)

    return g.compile()
