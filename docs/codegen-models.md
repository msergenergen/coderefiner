# codegen-models.md — Code Generation Models

## 1. Model Overview

| Setting | Generator | Sampling | Output budget | Notes |
|---|---|---|---|---|
| Main (`ENV=openrouter`) | DeepSeek-V3 (`deepseek/deepseek-chat`, DeepInfra fp4) | T 0.7 (model card demo; no generation_config.json), top_p/top_k not sent | 8192 | non-reasoning; released 2024-12-26; open-weight (DeepSeek Model License) |
| Secondary (`ENV=local`) | Qwen3.5-9B UD-Q4_K_XL, llama.cpp :8001 | T 0.6, top_p 0.95, top_k 20, min_p 0 | 6144 | reasoning model; `--jinja --cache-ram 0` |
| Optional (`ENV=cloud`) | Qwen3.5-9B; M2 = Qwen2.5-Coder-1.5B | as above | 32768 | `CLOUD_README.md` §B |

Sampling values come from each model's card / `generation_config.json`. All backends are
OpenAI-compatible; nodes call `models/client.chat()` (below). Only `M1` runs by default
(`PIPELINE_MODELS=M1`).

---

## 2. models/client.py

```python
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
from models import budget


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
    if not model_registry.paid:
        return _local_chat(model_id, messages, max_tokens)

    cfg    = model_registry.get(model_id)
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
        return res


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
```

---

## 3. models/prompts.py

```python
"""
System prompt templates for codegen and judgement models.

Codegen prompt consists of four layers:
  1. CODEGEN_IDENTITY        — who it is
  2. CODEGEN_CONTEXT         — where it is in the pipeline
  3. CODEGEN_CONSTRAINTS     — behavior rules
  4. CODEGEN_FEEDBACK_RULES  — how to use feedback (only on iteration > 1)

First iteration:  layers 1 + 2 + 3
Subsequent:       layers 1 + 2 + 3 + 4 + improved_prompt
"""

# ── Layer 1: Identity ─────────────────────────────────────────────────────────

CODEGEN_IDENTITY = """\
You are an expert Python programmer specializing in competitive programming.
Your sole responsibility in this pipeline is to produce correct, efficient,
and secure Python solutions to algorithmic problems.\
"""

# ── Layer 2: Pipeline context ─────────────────────────────────────────────────

CODEGEN_CONTEXT = """\
You are operating as the code generation node of an automated iterative \
refinement pipeline. After you produce a solution, it passes through \
three evaluation stages:

  1. Static Analyzer  — checks complexity (CC, MI) and security (Bandit)
  2. Sandbox Executor — runs your code against public and private test cases
  3. Judge LLM        — scores your solution 0–100 on a structured rubric

If your solution scores below the acceptance threshold, you will receive
specific improvement feedback and be asked to produce a new solution.
This process repeats up to 10 times per model.\
"""

# ── Layer 3: Behavior rules ───────────────────────────────────────────────────

CODEGEN_CONSTRAINTS = """\
Rules you must follow in every solution:

Output format:
- Output ONLY Python code inside ```python ... ``` markers
- Do not include any text, explanation, or comments outside the markers

I/O format:
- If the problem gives Starter Code (a class with a method signature),
  implement exactly that class and method: it is called directly with the
  test arguments and its return value is checked, so do not read stdin
  and do not print the result
- Otherwise read all input from stdin using input() or sys.stdin and
  write all output to stdout using print()
- Do not include any print() calls for debugging or examples

Correctness:
- Handle ALL edge cases: empty input, negative numbers, zero, large N
- Do not assume input is always well-formed

Security (enforced by static analyzer):
- Allowed imports only: os, sys, math, random, time, datetime,
  collections, itertools, functools, heapq, bisect, string,
  re, json, copy, typing, dataclasses
- Do NOT use: subprocess, socket, requests, exec(), eval(), __import__()

Code quality:
- Keep Cyclomatic Complexity below 10 per function
- Prefer clear variable names over single letters (except loop counters)
- Split functions exceeding 30 lines into smaller helpers\
"""

# ── Layer 4: Feedback integration (only on iteration > 1) ────────────────────

CODEGEN_FEEDBACK_RULES = """\
You are receiving improvement feedback from the previous iteration.

How to use this feedback:
- Read every point carefully — each one corresponds to a real failure
- Write a COMPLETELY NEW solution — do not patch or modify the old one
- Address EVERY issue mentioned — skipping any point will cause the same failure
- Reference specific failing test cases in your logic
- The feedback already tells you what went wrong — trust it and fix it

What NOT to do:
- Do not copy the previous solution with minor edits
- Do not add print() statements for the failing cases
- Do not explain what you changed — just produce the fixed code\
"""


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_system_prompt(improved_prompt: str = None) -> str:
    """
    Builds the codegen system prompt from four layers.

    Args:
        improved_prompt: Improvement instructions produced by Node 3.
                         None on the first iteration — feedback layer is not added.

    Returns:
        Full system prompt text.

    Layer structure:
        First iteration      → IDENTITY + CONTEXT + CONSTRAINTS
        Subsequent iterations → IDENTITY + CONTEXT + CONSTRAINTS
                               + FEEDBACK_RULES + improved_prompt
    """
    base = "\n\n---\n\n".join([
        CODEGEN_IDENTITY,
        CODEGEN_CONTEXT,
        CODEGEN_CONSTRAINTS,
    ])

    if not improved_prompt:
        return base

    return "\n\n---\n\n".join([
        CODEGEN_IDENTITY,
        CODEGEN_CONTEXT,
        CODEGEN_CONSTRAINTS,
        CODEGEN_FEEDBACK_RULES,
        f"## Improvement Notes from Previous Attempt\n\n{improved_prompt}",
    ])


def build_user_prompt(formatted_problem: str) -> str:
    """
    Converts an LCB question to user prompt format.
    Does not repeat the system prompt — only provides the problem.
    """
    return (
        f"Solve the following problem:\n\n{formatted_problem}\n\n"
        "Write your complete Python solution:"
    )
```

---

## 3.1 Layer Structure — Visual Overview

```
İlk iterasyon (improved_prompt = None):
┌─────────────────────────┐
│   CODEGEN_IDENTITY      │  "You are an expert Python programmer..."
├─────────────────────────┤
│   CODEGEN_CONTEXT       │  "You operate inside a pipeline..."
├─────────────────────────┤
│   CODEGEN_CONSTRAINTS   │  "Rules: stdin/stdout, whitelist, edge cases..."
└─────────────────────────┘

Sonraki iterasyonlar (improved_prompt doluysa):
┌─────────────────────────┐
│   CODEGEN_IDENTITY      │
├─────────────────────────┤
│   CODEGEN_CONTEXT       │
├─────────────────────────┤
│   CODEGEN_CONSTRAINTS   │
├─────────────────────────┤
│   CODEGEN_FEEDBACK_RULES│  "Write completely new — address every point..."
├─────────────────────────┤
│   improved_prompt       │  Node 3'ün ürettiği spesifik talimatlar
└─────────────────────────┘
```

---

## 4. nodes/codegen.py

```python
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
    user_prompt = build_user_prompt(state["problem"])

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
```

**2026-09-30 düzeltme — `max_tokens` 4096 → 6144:** 100-problem pilot koşusunda
(`run_20260930_000556`) 27 iterasyondan 5'i (~%19) codegen tamamlamasını tam
olarak 4096 token sınırında bitirdi — reasoning model olan qwen3.5-9b önce
`reasoning_content` (chain-of-thought) üretip sonra kodu yazdığı için, sınıra
takılan üretimler kapanan ` ```python ` fence'ini hiç göremeden kesiliyordu.
`_extract_code()` kapanmamış bloğu bulamayınca ham (parse edilemeyen) metni
"kod" olarak döndürüyor, bu da Node 1'de bir `SyntaxError` sentinel'ine ve —
düzeltilmeden önce — kalıcı `STOP_HIGH`'a yol açıyordu (bkz. problem 3298,
`docs/node1.md`'deki not). `max_tokens`'ın artırılması kesilme sıklığını
azaltır; kalıcı durdurma riski ise `docs/node1.md`'deki severity düzeltmesiyle
ayrıca giderildi.

---

### Output budget

`codegen_node` uses `max_tokens=settings.codegen_max_tokens` (`CODEGEN_MAX_TOKENS`, default 6144).
Locally ~46% of generation calls hit this cap (73% on hard problems), which is why it is
configurable for cloud runs; see `docs/config.md`.

### `chat()` — single call path (2026-10-02)

`codegen_node` and the judge call `models/client.chat(model_id, messages, max_tokens, tag)`,
which returns a `ChatResult` (`text`, `content`, `reasoning`, `finish_reason`, token counts,
`cost_usd`, `provider`). `text` is `content`, or the reasoning when `content` is empty (budget
spent while thinking) — the same fallback as before, now covering OpenRouter's `reasoning`
field as well as llama.cpp's `reasoning_content`.

- **llama.cpp (`ENV=local|cloud`)**: one non-streamed request, as before.
- **OpenRouter (`ENV=openrouter`)**:
  1. `budget.check()` refuses the call if ledger spend + worst case
     (prompt chars / 3 × input price + `max_tokens` × output price) exceeds `API_BUDGET_USD`
     → `BudgetExceeded`, which stops the process (the judge re-raises it instead of using its
     rule-based fallback).
  2. The response is streamed (`stream_options.include_usage`), so long reasoning generations
     keep the connection alive and an error chunk inside an HTTP-200 stream is detected.
  3. Transient failures (429, 5xx, timeouts, dropped streams, in-stream errors — including the
     SDK's bare `APIError` raised for an error event inside an HTTP-200 stream) are retried with
     exponential backoff (5 s doubling, max 120 s, `API_MAX_RETRIES`). Other status errors
     (400, 401, 402 out of credit) fail immediately.
  4. Each successful call is appended to the ledger with the provider that actually served it,
     tokens (incl. reasoning tokens), reported cost (price-table estimate if absent), latency,
     attempt number, and `question_id` / `iteration` / `run_id`.

Context management: each call is stateless — the system prompt (with the judge's latest
`improved_prompt`, rewritten every iteration and capped at 2000 chars) plus the problem. No
conversation history accumulates across iterations, so prompt size stays flat (~2–4k tokens)
and the output budget is the only large term.

## 5. pipeline/router.py

```python
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
```

---

## 6. Verification

### Model Server Testi

```bash
# While M1 server is running:
source .venv/bin/activate
python -c "
from models.client import generate, health_check

# Health check
print('M1 sağlık:', health_check('M1'))

# Simple code generation
code = generate(
    model_id='M1',
    system_prompt='You are a Python expert. Output only code.',
    user_prompt='Write a function that returns the sum of a list.',
    max_tokens=200,
)
print('Üretilen kod:')
print(code)
"
```

### Expected Output

```
M1 sağlık: True
Üretilen kod:
def sum_list(numbers):
    return sum(numbers)
```

### Codegen Node Testi

```bash
python -c "
from pipeline.state import PipelineState
from nodes.codegen import codegen_node

state: PipelineState = {
    'active_model': 'M1',
    'iteration': 0,
    'max_iterations': 10,
    'problem': '## Problem: Two Sum\nDifficulty: easy\nReturn indices of two numbers that add up to target.',
    'difficulty': 'easy',
    'generated_code': '',
    'node3_improved_prompt': '',
    'memory_log': [],
    'final_status': 'PENDING',
}

result = codegen_node(state)
print('Üretilen kod:')
print(result['generated_code'])
print('İterasyon:', result['iteration'])
"
```


## Repair prompt with the previous solution (`FEEDBACK_INCLUDE_CODE`, 2026-10-02)

Until 10:10 on 2026-10-02 a repair iteration gave the generator the judge's `improved_prompt`
(system prompt, layer 4 `CODEGEN_FEEDBACK_RULES`: "write a COMPLETELY NEW solution") and the
problem, but not the code being judged. The judge's feedback ("your code returns 4 for input
3 2 5 7") therefore referred to code the generator could not see. In the first half of the main
runs only 2 of ~76 problems went from wrong at iteration 1 to correct at the end.

With `FEEDBACK_INCLUDE_CODE=true` (default) a repair iteration follows LiveCodeBench's
self-repair prompt, which gives the question, the code to be corrected and the error feedback
(Jain et al., arXiv:2403.07974, App. C.3; also Olausson et al., ICLR 2024, arXiv:2306.09896):

- system prompt: layers 1–3 + `CODEGEN_REPAIR_RULES` (write a complete corrected solution; keep
  what the feedback does not criticise; change approach if the feedback says it is wrong) +
  the judge's notes;
- user prompt: the problem, then `## Your Previous Solution (rejected)` with the last generated
  code (≤ 8000 chars), then "Write your complete corrected Python solution:".

The judge remains the only feedback channel: raw failing tests reach the generator only through
the judge's notes. Only the last attempt is shown; no conversation history accumulates.
`FEEDBACK_INCLUDE_CODE=false` restores the earlier prompts exactly (runs `run_20261002_0918*`).
The flag is recorded in `run_info_<run_id>.json` → `pipeline_config.feedback_include_code`.


## Langfuse tracing (`models/tracing.py`, 2026-10-02)

Optional observability layer on top of the ledger and the iteration JSON files (those stay the
record of truth). Langfuse Python SDK v4 (OpenTelemetry based, `langfuse==4.16.0`).

**Switch.** On only when `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set
(`LANGFUSE_BASE_URL`, default `https://cloud.langfuse.com`; `LANGFUSE_ENABLED=false` forces
off). For OpenRouter runs the keys go in the git-ignored `.env.openrouter`;
`scripts/run_openrouter.sh` exports them and `scripts/run_openrouter.sh langfuse-check` tests
them. Without keys every helper is a no-op. Any tracing error is logged once and switches
tracing off for the process; it never stops a run.

| Langfuse object | Created by | Content |
|---|---|---|
| session | `run_id` | one pipeline shard or baseline process |
| trace `"<qid>/<model>"` | `main.run_pipeline_for_problem`, `sample_baseline._sample` | one problem × generator (baseline: one sample, model `V0-k`); tags: ENV, difficulty, judge decision; id = `create_trace_id(seed="run_id/qid/model")` |
| generation `codegen` / `judgement` | `models/client.chat()` | messages, output, model, sampling parameters, input/output/reasoning tokens, cost, provider, finish reason, attempts; failed calls as level ERROR |
| event `iteration` | `memory/logger.py` | iteration, status, severity, CC, public pass rate, rubric total, verdict, verdict source and reason |
| scores | logger (end of problem) | `status`, `iterations`, `public_pass_rate`, `rubric_total`, `judge_verdict` (reason as comment) |
| scores | `benchmark/openrouter_results.py --langfuse` | `hidden_pass`, `hidden_pass_first` (offline hidden tests; score id = trace id + name, so re-running replaces them) |

Because the trace id is derived from `run_id/question_id/model`, hidden-test results computed
later land on the same trace: in Langfuse, filtering `judge_verdict = ACCEPT` and
`hidden_pass = 0` lists exactly the judge's false acceptances with the full prompts.

The LangChain/LangGraph callback handler is not used: it would upload the whole graph state at
every node, including all hidden tests of the problem.
