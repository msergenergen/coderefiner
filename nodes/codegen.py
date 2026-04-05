"""
LangGraph Node 0: Code Generation

Sends the problem to the active model, parses the generated code,
and writes it to state.
"""
import re
import time
from pipeline.state import PipelineState
from models.client import get_client, get_model_name, get_temperature, health_check
from models.prompts import build_system_prompt, build_user_prompt


def codegen_node(state: PipelineState) -> PipelineState:
    """
    Generates code from the active model.

    Reads from state: active_model, problem, node3_improved_prompt
    Writes to state:  generated_code, iteration (incremented), timing, tokens
    """
    model_id = state["active_model"]
    t0 = time.perf_counter()

    # Server health check
    if not health_check(model_id):
        raise RuntimeError(
            f"Cannot reach {model_id} server. "
            f"Make sure the llama.cpp server is running."
        )

    # Build prompt
    system_prompt = build_system_prompt(
        improved_prompt=state.get("node3_improved_prompt")
    )
    user_prompt = build_user_prompt(state["problem"])

    # Generate code
    from config import model_registry
    cfg    = model_registry.get(model_id)
    client = get_client(model_id)

    response = client.chat.completions.create(
        model=cfg.model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=cfg.temperature,
        max_tokens=2048,
    )

    raw_response = response.choices[0].message.content
    code = _extract_code(raw_response)

    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    # Update timing
    timing = dict(state.get("timing", {}))
    timing["codegen_ms"] = elapsed_ms
    timing["total_ms"] = sum(
        timing.get(k, 0)
        for k in ("codegen_ms", "node1_ms", "node2_ms", "node3_ms")
    )

    # Update token counts
    tokens = dict(state.get("tokens", {}))
    if response.usage:
        tokens["codegen_prompt"]     = response.usage.prompt_tokens
        tokens["codegen_completion"] = response.usage.completion_tokens
        tokens["total"] = (
            tokens.get("codegen_prompt", 0) + tokens.get("codegen_completion", 0) +
            tokens.get("judgement_prompt", 0) + tokens.get("judgement_completion", 0)
        )

    return {
        **state,
        "generated_code": code,
        "iteration":      state.get("iteration", 0) + 1,
        "timing":         timing,
        "tokens":         tokens,
    }


def _extract_code(response: str) -> str:
    """
    Extracts the ```python ... ``` block from the model's response.
    Returns the full response if no block is found.
    """
    # Take the last ```python...``` block (guard against lcb_runner bug)
    matches = re.findall(r"```python\s*(.*?)```", response, re.DOTALL)
    if matches:
        return matches[-1].strip()
    # Backtick block without language tag
    matches = re.findall(r"```\s*(.*?)```", response, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return response.strip()
