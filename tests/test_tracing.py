"""
Unit tests: models/tracing.py (optional Langfuse tracing).

No network. Spans go through the real Langfuse SDK into an in-memory
OpenTelemetry exporter; scores (sent over HTTP by the SDK) are checked on a
recording stub.
Covered:
  - no keys → disabled, chat() unchanged, nothing recorded
  - chat() → one generation span with model, usage, cost and run metadata
  - problem_trace → deterministic trace id (run_id/question_id/model) shared by
    its generations, so offline hidden-test scores land on the same trace
  - a failing call is recorded as an ERROR generation and still raised
  - tracing errors never propagate into the pipeline
"""
import dataclasses
import uuid

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import models.budget as budget
import models.client as client
import models.tracing as tracing
from config.models import _openrouter_registry
from tests.test_api_client import MSG, _FakeClient, _chunk, _usage


@pytest.fixture
def no_tracing(monkeypatch):
    monkeypatch.setattr(tracing, "_client", None)
    monkeypatch.setattr(tracing, "_disabled", False)
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)


@pytest.fixture
def lf(monkeypatch):
    """Real Langfuse client exporting spans to memory; scores recorded."""
    from langfuse import Langfuse
    exporter = InMemorySpanExporter()
    # The SDK keeps one client per public key, so each test gets its own key.
    c = Langfuse(public_key=f"pk-lf-test-{uuid.uuid4().hex}", secret_key="sk-lf-test",
                 base_url="http://127.0.0.1:9", tracer_provider=TracerProvider(),
                 span_exporter=exporter)
    scores = []
    monkeypatch.setattr(c, "score_current_trace", lambda **kw: scores.append(kw))
    monkeypatch.setattr(c, "create_score", lambda **kw: scores.append(kw))
    monkeypatch.setattr(tracing, "_client", c)
    monkeypatch.setattr(tracing, "_disabled", False)
    return exporter, scores


@pytest.fixture
def paid(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(client, "model_registry", _openrouter_registry())
    s = dataclasses.replace(client.settings, api_ledger_path=str(tmp_path / "l.jsonl"),
                            api_budget_usd=1.0, api_max_retries=1)
    monkeypatch.setattr(client, "settings", s)
    monkeypatch.setattr(budget, "settings", s)
    monkeypatch.setattr(client.time, "sleep", lambda _s: None)

    def install(streams):
        fake = _FakeClient(streams)
        monkeypatch.setattr(client, "get_client", lambda _m: fake)
        return fake
    return install


def _ok():
    return [[_chunk(content="ok", finish="stop"), _chunk(usage=_usage(p=100, c=50, cost=0.002))]]


def test_disabled_without_keys(no_tracing, paid):
    paid(_ok())
    assert client.chat("M1", MSG, max_tokens=10).text == "ok"
    assert tracing.enabled() is False and tracing.trace_id_for("r", "q") is None
    with tracing.problem_trace("r", "q", "M1", "easy"):
        tracing.score("x", 1.0)                       # no-op, no error


def test_generation_span_records_model_usage_cost(lf, paid):
    exporter, _ = lf
    paid(_ok())
    with tracing.problem_trace("run_1", "3771", "M1", "medium", judge_decision="verdict"):
        client.chat("M1", MSG, max_tokens=10, tag={"question_id": "3771", "iteration": 2})
    tracing._client.flush()
    spans = {s.name: s for s in exporter.get_finished_spans()}
    gen, root = spans["codegen"], spans["problem"]
    attrs = dict(gen.attributes)
    assert attrs["langfuse.observation.type"] == "generation"
    assert attrs["langfuse.observation.model.name"] == "deepseek/deepseek-chat"
    assert '"input": 100' in attrs["langfuse.observation.usage_details"]
    assert "0.002" in attrs["langfuse.observation.cost_details"]
    assert gen.context.trace_id == root.context.trace_id
    expected = tracing.trace_id_for("run_1", "3771", "M1")
    assert format(root.context.trace_id, "032x") == expected
    assert dict(root.attributes)["session.id"] == "run_1"


def test_failed_call_is_an_error_generation_and_still_raises(lf, paid):
    import httpx, openai
    exporter, _ = lf
    req = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    paid([openai.BadRequestError("bad", response=httpx.Response(400, request=req), body=None)])
    with pytest.raises(openai.BadRequestError):
        client.chat("QWEN", MSG, max_tokens=10)
    tracing._client.flush()
    (gen,) = [s for s in exporter.get_finished_spans() if s.name == "judgement"]
    assert dict(gen.attributes)["langfuse.observation.level"] == "ERROR"


def test_scores_and_hidden_upload(lf):
    _, scores = lf
    tracing.score("judge_verdict", "ACCEPT", data_type="CATEGORICAL")
    from benchmark.openrouter_results import push_hidden_scores
    probs = {"Y": {"q1": [{"run_id": "r1", "model": "M1", "iter": 1, "_hidden": 0.4},
                          {"run_id": "r1", "model": "M1", "iter": 2, "_hidden": 1.0}]}}
    assert push_hidden_scores(probs) == 1
    tid = tracing.trace_id_for("r1", "q1", "M1")
    by = {s["name"]: s for s in scores}
    assert by["judge_verdict"]["value"] == "ACCEPT"
    assert by["hidden_pass"]["trace_id"] == tid and by["hidden_pass"]["value"] == 1.0
    assert by["hidden_pass_first"]["value"] == 0.0
    assert by["hidden_pass"]["score_id"] == f"{tid}-hidden_pass"


def test_tracing_errors_do_not_break_calls(monkeypatch, paid):
    class Broken:
        def start_observation(self, **kw):
            raise RuntimeError("collector down")
    monkeypatch.setattr(tracing, "_client", Broken())
    monkeypatch.setattr(tracing, "_disabled", False)
    paid(_ok())
    assert client.chat("M1", MSG, max_tokens=10).text == "ok"
    assert tracing._disabled is True                 # switched off after the first failure
