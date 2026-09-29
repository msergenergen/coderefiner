"""
Sequential model router.
Runs models in the order defined by PIPELINE_MODELS env var (default: M1,M2).
Advances to the next model after one completes.
"""
from config import settings
from pipeline.state import PipelineState


def get_next_model(current: str):
    """
    Returns the model after the current one in settings.model_sequence.
    Returns None when all models are completed.
    """
    sequence = settings.model_sequence
    try:
        idx = sequence.index(current)
    except ValueError:
        return None
    if idx + 1 < len(sequence):
        return sequence[idx + 1]
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
