"""
Sequential model router.
Runs M1 → M2 → M3 in order.
Advances to the next model after one completes.
"""
from pipeline.state import PipelineState

MODEL_SEQUENCE = ["M1", "M2", "M3"]


def get_next_model(current: str):
    """
    Returns the model after the current one.
    Returns None after M3 (all models completed).
    """
    try:
        idx = MODEL_SEQUENCE.index(current)
    except ValueError:
        return None
    if idx + 1 < len(MODEL_SEQUENCE):
        return MODEL_SEQUENCE[idx + 1]
    return None


def advance_model(state: PipelineState) -> PipelineState:
    """
    Advances to the next model and resets the iteration counter.
    """
    next_model = get_next_model(state["active_model"])

    if next_model is None:
        # All models completed
        return {**state, "final_status": "PASS"}

    return {
        **state,
        "active_model":          next_model,
        "iteration":             0,
        "node3_improved_prompt": "",   # Reset for next model
        "final_status":          "PENDING",
    }
