"""
Conditional edge functions.
iteration_gate runs AFTER log_iteration_node —
logging is already complete, only the routing decision is made here.
"""
from config import thresholds
from pipeline.state import PipelineState

PASS_THRESHOLDS = thresholds.node3.as_dict()


def severity_gate(state: PipelineState) -> str:
    """
    Routing after Node 1.

    "to_node2"  → OK  — run sandbox
    "to_node3"  → LOW — skip sandbox, go directly to judgment
    "stop_high" → HIGH — go to log_severity_stop_node
    """
    sev = state["node1_severity"]
    if sev == "HIGH":
        return "stop_high"
    elif sev == "LOW":
        return "to_node3"
    return "to_node2"


def iteration_gate(state: PipelineState) -> str:
    """
    Routing after log_iteration_node.
    The row is already written to memory.md and JSON.

    "write_memory_final" → PASS     — update final_status, advance to next model
    "stop_max"           → STOP_MAX — update final_status, END
    "loop_to_codegen"    → LOOP     — retry with same model
    """
    threshold = PASS_THRESHOLDS[state["difficulty"]]
    passed    = state["node3_total_score"] >= threshold and state["node3_pass"]

    if passed:
        return "write_memory_final"
    elif state["iteration"] >= state["max_iterations"]:
        return "stop_max"
    return "loop_to_codegen"
