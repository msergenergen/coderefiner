"""
OpenAI-compatible client for models running on llama.cpp or OpenRouter.
Model config is managed via config/models.py.

chat() is the single entry point the nodes use. For a paid backend
(ENV=openrouter) it streams the response (long reasoning generations stay
alive and a mid-stream provider error is detected), retries transient
failures with backoff, and records every call's tokens, cost and serving
provider in the budget ledger, refusing calls that could pass the cap.
"""
import time
from dataclasses import dataclass

import httpx
import openai
from loguru import logger
from openai import OpenAI

from config import model_registry, settings
from models import budget, tracing


@dataclass
class ChatResult:
    text:              str     # content, or reasoning if content is empty
    content:           str
    reasoning:         str
    finish_reason:     str
    prompt_tokens:     int
    completion_tokens: int
    reasoning_tokens:  int
    cost_usd:          float
    provider:          str


class _StreamError(RuntimeError):
    """Provider error delivered inside an HTTP-200 stream."""


_RETRYABLE = (
    openai.APIConnectionError, openai.APITimeoutError, openai.RateLimitError,
    openai.InternalServerError, httpx.TransportError, _StreamError,
)


def _is_retryable(e: Exception) -> bool:
    if isinstance(e, _RETRYABLE):
        return True
    # The SDK raises a bare APIError (no HTTP status) for an error event inside
    # an HTTP-200 stream, e.g. "Internal Server Error" from the provider.
    # Status errors other than 429/5xx (bad request, auth, 402 no credit) are final.
    return isinstance(e, openai.APIError) and not isinstance(e, openai.APIStatusError)


def get_client(model_id: str) -> OpenAI:
    """
    Returns an OpenAI client for the given model ID.

    Args:
        model_id: "M1" | "M2" | "QWEN"

    Returns:
        OpenAI client (base_url points to llama.cpp server)
    """
    cfg = model_registry.get(model_id)
    if model_registry.paid:
        return OpenAI(
            base_url=cfg.base_url,
            api_key=model_registry.api_key(),
            timeout=httpx.Timeout(settings.api_timeout_sec, connect=30.0),
            max_retries=settings.api_max_retries,
            default_headers={"X-Title": "llm-code-refinement"},
        )
    return OpenAI(base_url=cfg.base_url, api_key=model_registry.api_key())


def _reasoning_of(obj) -> str:
    # OpenRouter: "reasoning"; llama.cpp: "reasoning_content".
    extra = getattr(obj, "model_extra", None) or {}
    return (getattr(obj, "reasoning", None) or extra.get("reasoning")
            or getattr(obj, "reasoning_content", None) or extra.get("reasoning_content") or "")


def _usage_fields(usage) -> tuple[int, int, int, float | None]:
    if not usage:
        return 0, 0, 0, None
    extra   = getattr(usage, "model_extra", None) or {}
    details = getattr(usage, "completion_tokens_details", None)
    r_tok   = getattr(details, "reasoning_tokens", 0) or 0 if details else 0
    cost    = getattr(usage, "cost", None)
    if cost is None:
        cost = extra.get("cost")
    return (usage.prompt_tokens or 0, usage.completion_tokens or 0, r_tok,
            float(cost) if cost is not None else None)


def _local_chat(model_id: str, messages: list, max_tokens: int) -> ChatResult:
    cfg = model_registry.get(model_id)
    response = get_client(model_id).chat.completions.create(
        model=cfg.model_name, messages=messages,
        temperature=cfg.temperature, max_tokens=max_tokens,
    )
    msg       = response.choices[0].message
    content   = msg.content or ""
    reasoning = _reasoning_of(msg)
    p, c, r, _ = _usage_fields(response.usage)
    return ChatResult(content or reasoning, content, reasoning,
                      response.choices[0].finish_reason or "", p, c, r, 0.0, "local")


def _stream_once(client: OpenAI, cfg, messages: list, max_tokens: int) -> ChatResult:
    stream = client.chat.completions.create(
        model=cfg.model_name, messages=messages,
        temperature=cfg.temperature, max_tokens=max_tokens,
        top_p=cfg.top_p if cfg.top_p is not None else openai.NOT_GIVEN,
        stream=True, stream_options={"include_usage": True},
        extra_body=cfg.extra_body,
    )
    content, reasoning = [], []
    finish, provider, usage = "", "", None
    for chunk in stream:
        extra = chunk.model_extra or {}
        if extra.get("error"):
            raise _StreamError(str(extra["error"])[:300])
        provider = extra.get("provider") or provider
        if chunk.choices:
            choice = chunk.choices[0]
            if choice.delta.content:
                content.append(choice.delta.content)
            r = _reasoning_of(choice.delta)
            if r:
                reasoning.append(r)
            if choice.finish_reason:
                finish = choice.finish_reason
        if chunk.usage:
            usage = chunk.usage
    if finish == "error":
        raise _StreamError("finish_reason=error")
    p, c, r, cost = _usage_fields(usage)
    if cost is None:
        cost = (p * cfg.price_in + c * cfg.price_out) / 1e6
    text = "".join(content)
    return ChatResult(text or "".join(reasoning), text, "".join(reasoning),
                      finish, p, c, r, cost, provider)


def chat(model_id: str, messages: list, max_tokens: int, tag: dict | None = None) -> ChatResult:
    """
    Sends one chat completion and returns text plus usage.

    Args:
        model_id:   registry key ("M1" | "M2" | "QWEN")
        messages:   OpenAI-style message list
        max_tokens: output budget (reasoning + answer)
        tag:        context written to the ledger (e.g. question_id, iteration)
    """
    cfg = model_registry.get(model_id)
    gen = tracing.generation(cfg.role or model_id, cfg, messages, max_tokens, tag)
    try:
        if not model_registry.paid:
            res, attempt = _local_chat(model_id, messages, max_tokens), 1
        else:
            res, attempt = _paid_chat(model_id, cfg, messages, max_tokens, tag)
    except Exception as e:
        gen.error(e)
        raise
    gen.done(res, attempt)
    return res


def _paid_chat(model_id: str, cfg, messages: list, max_tokens: int,
               tag: dict | None) -> tuple[ChatResult, int]:
    client = get_client(model_id)
    prompt_chars = sum(len(m.get("content", "")) for m in messages)
    # Worst case: ~3 chars/token prompt and the full output budget used.
    budget.check((prompt_chars / 3 * cfg.price_in + max_tokens * cfg.price_out) / 1e6)

    attempts = settings.api_max_retries + 1
    for attempt in range(1, attempts + 1):
        t0 = time.perf_counter()
        try:
            res = _stream_once(client, cfg, messages, max_tokens)
        except Exception as e:
            if not _is_retryable(e):
                raise
            # A dropped stream may still be billed; we cannot see the
            # partial usage, so the reserve above already covered it.
            if attempt == attempts:
                raise
            wait = min(120, 5 * 2 ** (attempt - 1))
            logger.warning("API call failed ({}/{}), retrying in {}s | {} | {}: {}",
                           attempt, attempts, wait, cfg.model_name,
                           type(e).__name__, str(e)[:200])
            time.sleep(wait)
            continue
        budget.record({
            "model_id": model_id, "model": cfg.model_name, "role": cfg.role,
            "provider": res.provider, "finish_reason": res.finish_reason,
            "prompt_tokens": res.prompt_tokens, "completion_tokens": res.completion_tokens,
            "reasoning_tokens": res.reasoning_tokens, "cost_usd": res.cost_usd,
            "elapsed_s": round(time.perf_counter() - t0, 1), "attempt": attempt,
            **(tag or {}),
        })
        return res, attempt


def get_model_name(model_id: str) -> str:
    return model_registry.get(model_id).model_name


def get_temperature(model_id: str) -> float:
    return model_registry.get(model_id).temperature


def generate(
    model_id: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 2048,
) -> str:
    """
    Sends a completion request to the given model.

    Args:
        model_id:      "M1" | "M2" | "QWEN"
        system_prompt: System prompt text
        user_prompt:   User message (problem + requirements)
        max_tokens:    Maximum tokens to generate

    Returns:
        Raw text response from the model
    """
    client      = get_client(model_id)
    model_name  = get_model_name(model_id)
    temperature = get_temperature(model_id)

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return response.choices[0].message.content


def health_check(model_id: str) -> bool:
    """
    Checks whether the model server is responding.

    Returns:
        True: server responded
        False: server unreachable
    """
    if model_registry.paid:
        # Hosted API: a missing key fails here; outages surface (with
        # retries) on the first real call instead of an extra request.
        model_registry.api_key()
        return True
    try:
        client = get_client(model_id)
        client.models.list()
        return True
    except Exception:
        return False
