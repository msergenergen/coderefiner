"""
Conditional edge functions.
iteration_gate runs AFTER log_iteration_node —
logging is already complete, only the routing decision is made here.
"""
from config import thresholds, settings
from pipeline.state import PipelineState

PASS_THRESHOLDS = thresholds.node3.as_dict()


def severity_gate(state: PipelineState) -> str:
    """
    Routing after Node 1.

    "to_node2"  → OK  — run sandbox
    "to_node3"  → LOW — skip sandbox, go directly to judgement
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
    "loop_to_codegen"    → LOOP     — retry with same model (or trend extension)
    """
    threshold = PASS_THRESHOLDS[state["difficulty"]]
    passed    = state["node3_total_score"] >= threshold and state["node3_pass"]

    if passed:
        return "write_memory_final"

    if state["iteration"] >= state["max_iterations"]:
        if settings.trend_extension_enabled and _should_extend(state, threshold):
            return "loop_to_codegen"
        return "stop_max"

    return "loop_to_codegen"


# ── Trend-aware extension helpers ─────────────────────────────────────────────

def _compute_ema(values: list, alpha: float) -> list:
    """
    Returns EMA-smoothed version of values.
    alpha=0.5 gives recent values 2× weight over older ones.
    """
    if not values:
        return []
    ema = [values[0]]
    for v in values[1:]:
        ema.append(alpha * v + (1 - alpha) * ema[-1])
    return ema


def _linear_slope(values: list) -> float:
    """
    Returns the slope of a least-squares line fit through values.
    Uses the last min(len(values), 3) points for recency.
    Returns 0.0 for fewer than 2 points.
    """
    pts = values[-3:]
    n = len(pts)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2
    y_mean = sum(pts) / n
    num = sum((i - x_mean) * (pts[i] - y_mean) for i in range(n))
    den = sum((i - x_mean) ** 2 for i in range(n))
    return num / den if den else 0.0


def _should_extend(state: PipelineState, threshold: float) -> bool:
    """
    Returns True when all activation conditions are met:

      1. accuracy > 0           — at least one test passing (hard gate)
      2. pass_rate_slope > 0    — pass rate trending upward (last 3 iters)
      3. score_slope >= min     — EMA-smoothed score genuinely improving
      4. expected_gain >= gap   — projected gain can close the distance to threshold
      5. extension_count < max  — extension budget not exhausted
    """
    # 1. Accuracy hard gate
    accuracy = state.get("node3_score_breakdown", {}).get("accuracy", 0.0)
    if accuracy <= 0.0:
        return False

    # 5. Extension budget
    extension_count = state.get("extension_count", 0)
    if extension_count >= settings.max_extension_iters:
        return False

    score_history     = state.get("score_history", [])
    pass_rate_history = state.get("pass_rate_history", [])

    # Need at least 3 data points for a meaningful trend
    if len(score_history) < 3:
        return False

    # 2. Pass rate must be trending upward
    pr_slope = _linear_slope(pass_rate_history)
    if pr_slope <= 0.0:
        return False

    # 3. EMA-smoothed score slope
    smoothed     = _compute_ema(score_history, alpha=settings.trend_ema_alpha)
    score_slope  = _linear_slope(smoothed)
    if score_slope < settings.trend_min_slope:
        return False

    # 4. Expected gain must be sufficient to potentially reach threshold.
    # Uses a fixed projection window (trend_projection_window) — independent of
    # the extension budget — so increasing max_extension_iters does not
    # artificially inflate the gain estimate.
    expected_gain = score_slope * settings.trend_projection_window
    gap           = threshold - state.get("node3_total_score", 0.0)
    if expected_gain < gap:
        return False

    return True
