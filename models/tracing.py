"""
Optional Langfuse tracing (SDK v4, OpenTelemetry based).

Enabled only when LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set (host:
LANGFUSE_BASE_URL, default https://cloud.langfuse.com) and LANGFUSE_ENABLED is
not "false". Without keys every helper is a no-op, so tests and runs without
an account behave exactly as before. A tracing failure is logged once and
never interrupts a run: the ledger (models/budget.py) and the iteration JSON
files stay the record of truth; Langfuse is a viewer on top of them.

Trace layout
  session  = run_id                       (one pipeline / baseline process)
  trace    = one problem × generator       (id derived from run_id/qid/model,
                                             so offline hidden-test scores can
                                             be attached later: trace_id_for)
    generation  codegen / judgement       (every chat() call: messages, output,
                                             tokens, cost, provider, attempts)
    event       iteration                 (compact summary per iteration)
  scores    judge_verdict, rubric_total, public_pass_rate, status
            (+ hidden_pass from benchmark/openrouter_results.py --langfuse)
"""
from __future__ import annotations

import contextlib
import os
from typing import Any, Iterator, Optional

from loguru import logger

_client = None          # Langfuse instance, created lazily
_disabled = False       # set after the first failure or when keys are missing


def enabled() -> bool:
    global _client, _disabled
    if _disabled:
        return False
    if _client is not None:
        return True
    if os.getenv("LANGFUSE_ENABLED", "true").lower() == "false" or not (
            os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        _disabled = True
        return False
    try:
        from langfuse import Langfuse
        _client = Langfuse(environment=os.getenv("ENV", "local"))
        return True
    except Exception as e:                      # missing package, bad config
        _fail(e)
        return False


def _fail(e: Exception) -> None:
    global _disabled
    if not _disabled:
        logger.warning(f"Langfuse tracing disabled: {type(e).__name__}: {str(e)[:200]}")
    _disabled = True


def trace_id_for(run_id: str, question_id: str, model_id: str = "M1") -> Optional[str]:
    """Deterministic trace id of one problem × generator in one run."""
    if not enabled():
        return None
    return _client.create_trace_id(seed=f"{run_id}/{question_id}/{model_id}")


def _ascii(meta: dict) -> dict:
    return {k: str(v)[:200] for k, v in meta.items() if v is not None}


@contextlib.contextmanager
def problem_trace(run_id: str, question_id: str, model_id: str,
                  difficulty: str, **meta: Any) -> Iterator[None]:
    """Root span for one problem × generator; children inherit session/tags."""
    if not enabled():
        yield
        return
    try:
        from langfuse import propagate_attributes
        cm_attr = propagate_attributes(
            session_id=run_id, trace_name=f"{question_id}/{model_id}",
            tags=[t for t in (os.getenv("ENV", "local"), difficulty,
                              meta.get("judge_decision")) if t],
            metadata=_ascii({"run_id": run_id, "question_id": question_id,
                             "model_id": model_id, "difficulty": difficulty, **meta}))
        cm_span = _client.start_as_current_observation(
            name="problem", as_type="chain",
            trace_context={"trace_id": trace_id_for(run_id, question_id, model_id)},
            input={"question_id": question_id, "difficulty": difficulty})
        cm_attr.__enter__()
        cm_span.__enter__()
    except Exception as e:
        _fail(e)
        yield
        return
    try:
        yield
    finally:
        for cm in (cm_span, cm_attr):
            try:
                cm.__exit__(None, None, None)
            except Exception as e:
                _fail(e)


class _Generation:
    """Handle returned by generation(); call done() or error() once."""

    def __init__(self, obs=None):
        self._obs = obs

    def done(self, res, attempts: int) -> None:
        if self._obs is None:
            return
        try:
            self._obs.update(
                output=res.text,
                usage_details={"input": res.prompt_tokens, "output": res.completion_tokens,
                               "reasoning": res.reasoning_tokens},
                cost_details={"total": res.cost_usd} if res.cost_usd else None,
                metadata={"finish_reason": res.finish_reason, "provider": res.provider,
                          "attempts": attempts})
            self._obs.end()
        except Exception as e:
            _fail(e)

    def error(self, e: Exception) -> None:
        if self._obs is None:
            return
        try:
            self._obs.update(level="ERROR", status_message=f"{type(e).__name__}: {str(e)[:300]}")
            self._obs.end()
        except Exception as err:
            _fail(err)


def generation(name: str, cfg, messages: list, max_tokens: int,
               tag: Optional[dict] = None) -> _Generation:
    """Starts a generation observation for one chat() call."""
    if not enabled():
        return _Generation()
    try:
        params = {"temperature": cfg.temperature, "max_tokens": max_tokens}
        if getattr(cfg, "top_p", None) is not None:
            params["top_p"] = cfg.top_p
        obs = _client.start_observation(
            name=name, as_type="generation", model=cfg.model_name,
            input=messages, model_parameters=params, metadata=_ascii(tag or {}))
        return _Generation(obs)
    except Exception as e:
        _fail(e)
        return _Generation()


def event(name: str, output: dict) -> None:
    """Point-in-time event inside the current trace (e.g. one iteration)."""
    if not enabled():
        return
    try:
        _client.create_event(name=name, output=output)
    except Exception as e:
        _fail(e)


def score(name: str, value, comment: Optional[str] = None,
          data_type: Optional[str] = None, trace_id: Optional[str] = None) -> None:
    """Score the current trace, or the given trace_id (offline scores; the
    score id is derived from trace and name, so re-running replaces it)."""
    if not enabled():
        return
    try:
        if trace_id:
            _client.create_score(name=name, value=value, trace_id=trace_id,
                                 score_id=f"{trace_id}-{name}",
                                 comment=comment, data_type=data_type)
        else:
            _client.score_current_trace(name=name, value=value, comment=comment,
                                        data_type=data_type)
    except Exception as e:
        _fail(e)


def flush() -> None:
    if _client is None:
        return
    try:
        _client.flush()
    except Exception as e:
        _fail(e)
