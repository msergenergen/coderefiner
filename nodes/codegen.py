"""
LangGraph Node 0: Code Generation

Sends the problem to the active model, parses the generated code,
and writes it to state.
"""
import re
import time
from pipeline.state import PipelineState
from models.client import chat, health_check
from models.prompts import build_system_prompt, build_user_prompt
from config import settings


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
    # On a repair iteration the generator sees its own previous code, which
    # the judge's feedback refers to (no history beyond that one attempt).
    previous = state.get("generated_code") if state.get("node3_improved_prompt") else None
    user_prompt = build_user_prompt(state["problem"], previous_code=previous)

    # Generate code
    res = chat(
        model_id,
        [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        # 4096 was too tight for a reasoning model: thinking consumed the
        # budget before the closing ```python fence, so the fallback
        # extraction returned the whole raw (unparseable) response as "code".
        max_tokens=settings.codegen_max_tokens,
        tag={"question_id": state.get("question_id"), "iteration": state.get("iteration", 0) + 1,
             "run_id": state.get("run_id")},
    )
    # Reasoning models (e.g. qwen3.5-9b) emit chain-of-thought in a separate
    # reasoning field and can leave `content` empty if max_tokens is
    # exhausted during "thinking" — chat() then returns the reasoning so a
    # response is still extracted rather than silently returning nothing.
    raw_response = res.text
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
    if res.prompt_tokens or res.completion_tokens:
        tokens["codegen_prompt"]     = res.prompt_tokens
        tokens["codegen_completion"] = res.completion_tokens
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
