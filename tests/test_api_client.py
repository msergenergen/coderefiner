"""
Unit tests: models/client.py chat() on a paid backend + models/budget.py.

No network: the OpenAI client is replaced by a fake that yields stream chunks.
Covered:
  - streamed content/reasoning are assembled; reasoning is the fallback text
  - reported cost is recorded in the ledger (with tag fields)
  - missing cost falls back to the configured per-token prices
  - the cap blocks a call before it is sent
  - an in-stream provider error is retried, then succeeds
"""
import dataclasses
from types import SimpleNamespace as NS

import openai
import pytest

import models.budget as budget
import models.client as client
from config.models import _openrouter_registry


def _chunk(content=None, reasoning=None, finish=None, usage=None, extra=None):
    delta = NS(content=content, reasoning=reasoning, model_extra={})
    choices = [NS(delta=delta, finish_reason=finish)] if (content or reasoning or finish) else []
    return NS(choices=choices, usage=usage, model_extra=extra or {"provider": "DeepInfra"})


def _usage(p=100, c=50, cost=0.001):
    return NS(prompt_tokens=p, completion_tokens=c, cost=cost, model_extra={},
              completion_tokens_details=NS(reasoning_tokens=30))


class _FakeClient:
    def __init__(self, streams):
        self.streams = list(streams)
        self.calls = []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        s = self.streams.pop(0)
        if isinstance(s, Exception):
            raise s
        return iter(s)


@pytest.fixture
def paid(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(client, "model_registry", _openrouter_registry())
    s = dataclasses.replace(client.settings, api_ledger_path=str(tmp_path / "ledger.jsonl"),
                            api_budget_usd=1.0, api_max_retries=2)
    monkeypatch.setattr(client, "settings", s)
    monkeypatch.setattr(budget, "settings", s)
    monkeypatch.setattr(client.time, "sleep", lambda _s: None)

    def install(streams):
        fake = _FakeClient(streams)
        monkeypatch.setattr(client, "get_client", lambda _m: fake)
        return fake
    return install


MSG = [{"role": "user", "content": "hi"}]


def test_stream_assembles_and_records_cost(paid):
    fake = paid([[_chunk(reasoning="think "), _chunk(content="```python\nx=1\n```", finish="stop"),
                  _chunk(usage=_usage())]])
    res = client.chat("M1", MSG, max_tokens=1000, tag={"question_id": "q1", "iteration": 2})
    assert res.text.startswith("```python")
    assert res.reasoning == "think "
    assert res.provider == "DeepInfra" and res.reasoning_tokens == 30
    assert budget.spent_usd() == pytest.approx(0.001)
    kw = fake.calls[0]
    assert kw["extra_body"]["provider"]["allow_fallbacks"] is False
    assert kw["stream"] is True and kw["temperature"] == 0.7
    assert kw["top_p"] is openai.NOT_GIVEN and "top_k" not in kw["extra_body"]   # DeepSeek-V3: no top_p/top_k set
    assert kw["extra_body"]["provider"]["order"] == ["deepinfra/fp4"]


def test_judge_uses_qwen72b_generation_config(paid):
    fake = paid([[_chunk(content="ok", finish="stop"), _chunk(usage=_usage())]])
    client.chat("QWEN", MSG, max_tokens=100)
    kw = fake.calls[0]
    assert kw["top_p"] == 0.8 and kw["extra_body"]["top_k"] == 20
    assert kw["extra_body"]["repetition_penalty"] == 1.05
    assert kw["extra_body"]["provider"]["order"] == ["novita/bf16"]


def test_reasoning_is_fallback_when_content_empty(paid):
    paid([[_chunk(reasoning="only thoughts", finish="length"), _chunk(usage=_usage())]])
    res = client.chat("M1", MSG, max_tokens=1000)
    assert res.text == "only thoughts" and res.finish_reason == "length"


def test_missing_cost_uses_price_table(paid):
    paid([[_chunk(content="ok", finish="stop"), _chunk(usage=_usage(p=1_000_000, c=0, cost=None))]])
    res = client.chat("QWEN", MSG, max_tokens=10)
    assert res.cost_usd == pytest.approx(0.38)


def test_cap_blocks_before_sending(paid):
    fake = paid([])
    budget.record({"cost_usd": 0.999})
    with pytest.raises(budget.BudgetExceeded):
        client.chat("M1", MSG, max_tokens=8192)    # worst case ~$0.008 > $0.001 left
    assert fake.calls == []


def test_stream_error_is_retried(paid):
    fake = paid([[_chunk(extra={"error": {"code": 502, "message": "upstream"}})],
                 [_chunk(content="ok", finish="stop"), _chunk(usage=_usage())]])
    res = client.chat("M1", MSG, max_tokens=100)
    assert res.text == "ok" and len(fake.calls) == 2


def test_missing_key_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        _openrouter_registry().api_key()


def test_unknown_model_provider_pair_is_rejected(monkeypatch):
    monkeypatch.setenv("OPENROUTER_CODEGEN_PROVIDER", "somewhere/fp4")
    with pytest.raises(ValueError, match="No OpenRouter settings"):
        _openrouter_registry()


def test_pass_at_k_and_filtered_summary():
    from benchmark.sample_baseline import pass_at_k, summarise
    assert pass_at_k(5, 0, 1) == 0.0 and pass_at_k(5, 5, 1) == 1.0
    assert pass_at_k(5, 1, 1) == pytest.approx(0.2)
    assert pass_at_k(5, 1, 5) == 1.0
    recs = [  # q1: sample 0 passes public but not hidden, sample 1 is correct
        {"question_id": "q1", "difficulty": "easy", "sample": 0, "public": 1.0, "hidden": 0.5},
        {"question_id": "q1", "difficulty": "easy", "sample": 1, "public": 1.0, "hidden": 1.0},
    ]
    out = summarise(recs, n=2)["all"]
    assert out["pass@1"] == 50.0 and out["pass@2"] == 100.0
    assert out["filt@2"] == 0.0     # first public-passing sample is submitted


def test_feedback_code_leak_detection():
    from benchmark.openrouter_results import has_code
    mandated = "Fix the off-by-one. Output only Python code inside ```python ... ``` markers."
    assert not has_code(mandated)
    assert not has_code("Use a prefix sum; for each query return the range total in O(1).")
    assert has_code(mandated + "\n```python\nprint(1)\n```")
    assert has_code("Rewrite as:\n```python\ndef solve(n):\n    return n * 2\n```")
    # format instructions and empty skeletons are not solution code
    assert not has_code("Wrap code in ```python ... ``` markers.")
    assert not has_code("Required structure:\n```python\nclass Solution:\n"
                        "    def f(self, n: int) -> int:\n        # implement\n        pass\n```")


def test_baseline_metrics_and_budget_match():
    from benchmark.openrouter_results import baseline_metrics, budget_matched
    def s(q, k, pub, hid, d="easy"):
        return {"question_id": q, "sample": k, "public": pub, "hidden": hid, "difficulty": d}
    base = {"q1": [s("q1", 0, 0.0, 0.0), s("q1", 1, 1.0, 1.0)],
            "q2": [s("q2", 0, 1.0, 0.0), s("q2", 1, 1.0, 1.0)]}
    v0 = baseline_metrics(base, ["q1", "q2"], n=2)
    assert v0["all"]["pass@1"] == 0.5 and v0["all"]["pass@2"] == 1.0
    assert v0["all"]["filt@2"] == 0.5   # q1 picks sample 1 (correct), q2 picks sample 0 (wrong)
    bm = budget_matched({"all": {"mean_generations": 1.5, "v3": 1.0}}, v0, n=2)
    assert bm["filt@1"] == 0.0 and bm["filt@2"] == 0.5 and bm["pass@2"] == 1.0


def test_bare_stream_apierror_is_retried_but_bad_request_is_not(paid):
    import httpx, openai
    req = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    fake = paid([openai.APIError("Internal Server Error", req, body=None),
                 [_chunk(content="ok", finish="stop"), _chunk(usage=_usage())]])
    assert client.chat("M1", MSG, max_tokens=100).text == "ok" and len(fake.calls) == 2

    bad = openai.BadRequestError("bad", response=httpx.Response(400, request=req), body=None)
    fake = paid([bad])
    with pytest.raises(openai.BadRequestError):
        client.chat("M1", MSG, max_tokens=100)
    assert len(fake.calls) == 1


def test_hidden_grading_stops_at_first_failure_like_lcb_runner():
    from sandbox.lcb_eval import _fallback_evaluate
    row = {"all_tests": {"input": ["1\n", "2\n", "3\n"], "output": ["1", "5", "3"]},
           "starter_code": ""}
    full = _fallback_evaluate("print(input())", row, 10)
    early = _fallback_evaluate("print(input())", row, 10, stop_on_fail=True)
    assert full["passed_count"] == 2 and early["passed_count"] == 1
    assert early["total_count"] == 3 and early["pass_rate"] < 1.0   # verdict unchanged: fail
    ok = {"all_tests": {"input": ["1\n", "2\n"], "output": ["1", "2"]}, "starter_code": ""}
    assert _fallback_evaluate("print(input())", ok, 10, stop_on_fail=True)["pass_rate"] == 1.0


def test_judge_prompt_frames_metrics_as_evidence():
    from nodes.judgement import _build_judgement_prompt as build
    scores = {"accuracy": 25.0, "efficiency": 20.0, "quality": 10.0, "security": 15.0,
              "numeric_total": 70.0}
    p = build(scores, "print(1)", "problem", ["Maintainability Index 50 — threshold 65."],
              [], 90.0, False, "medium", 1)
    assert "required fixes" not in p and "not grounds for REJECT" in p
    assert "quality findings alone mean ACCEPT" in p
    failing = build({**scores, "accuracy": 12.5}, "print(1)", "problem", [],
                    [{"input": "1", "expected": "2", "got": "1"}], 90.0, False, "medium", 1)
    assert "verdict must be REJECT" in failing


def test_cost_separates_baseline_and_pipeline_with_same_run_id(tmp_path):
    import json
    from benchmark.openrouter_results import cost
    led = tmp_path / "l.jsonl"
    led.write_text("\n".join(json.dumps(e) for e in [
        {"run_id": "r1", "phase": "baseline", "cost_usd": 1.0},
        {"run_id": "r1", "cost_usd": 2.0},                       # pipeline shard, same second
        {"run_id": "r2", "cost_usd": 4.0}]))
    out = cost(str(led), {"baseline": {"r1"}, "Y": {"r1", "r2"}})
    assert out["baseline"]["usd"] == 1.0 and out["Y"]["usd"] == 6.0



def test_compare_rows_counts_discordant_pairs():
    from benchmark.openrouter_results import compare_rows
    this = {"Y": {"a": True, "b": True, "c": False, "d": True}}
    other = {"Y": {"a": True, "b": False, "c": True, "d": False}}
    out = compare_rows(this, other)["Y"]
    assert out["n"] == 4 and out["b"] == 2 and out["c"] == 1
    assert out["this"] == 0.75 and out["other"] == 0.5



def test_write_latex_macros_and_table(tmp_path):
    from benchmark.openrouter_results import write_latex
    pr = lambda: {k: {"n_problems": 2, "pass@1": 0.25, "pass@1_ci": [0.1, 0.5], "pass@2": 0.5,
                      "pass@3": 0.5, "pass@5": 0.5, "filt@2": 0.5, "filt@3": 0.5}
                  for k in ("all", "easy", "medium", "hard")}
    des = lambda v: {k: {"v3": v, "v3_ci": [0.1, 0.9], "delta_vs_v0": 0.05, "delta_ci": [-0.01, 0.1],
                         "wilcoxon_p": 0.2, "iter1": 0.25,
                         "iter1_vs_final_mcnemar": {"b": 1, "c": 0, "p": 1.0},
                         "mean_generations": 2.5, "first_stop": 0.5}
                     for k in ("all", "easy", "medium", "hard")}
    jd = {"public_passing": 4, "rejected": 1, "rejected_and_hidden_wrong": 1,
          "rejected_but_hidden_correct": 0, "accepted_precision": 0.6, "verdict_from_judge": 1.0,
          "feedback_with_code": 0, "feedback_total": 3}
    rep = {"n": 2, "V0": pr(), "Y": des(0.3), "P": des(0.4), "budget": {"Y": {}, "P": {}},
           "judge": {"Y": jd, "P": jd}, "Y_vs_P": {"b": 0, "c": 1, "p": 1.0},
           "cost": {"Y": {"usd": 0.5}, "P": {"usd": 0.4}, "baseline": {"usd": 0.3}}}
    write_latex(rep, str(tmp_path))
    mac = (tmp_path / "veri" / "ana_sonuclar.tex").read_text()
    assert "\\newcommand{\\aVo}{25{,}0}" in mac and "\\newcommand{\\aVoEn}{25.0}" in mac
    assert "\\newcommand{\\aYW}{$=$~0{,}200}" in mac
    tab = (tmp_path / "cizelgeler" / "cizelge-ana.tex").read_text()
    # V0, filt@2 (budget-matched), V3_Y, V3_P; the row's best value in bold
    assert "\\textbf{Toplam} & 2 & 25{,}0 & \\textbf{50{,}0} & 30{,}0 & 40{,}0" in tab
    proc = (tmp_path / "cizelgeler" / "cizelge-surec.tex").read_text()
    assert "Düzeltilen / bozulan problem & 1/0 & 1/0 \\\\" in proc
    assert "Kod içeren geri bildirim & 0/3 & 0/3" in proc
    # the budget figure needs the pipeline curves
    assert not (tmp_path / "sekiller" / "sekil2-butce-egrisi.pdf").exists()
    for d, v in (("Y", 0.3), ("P", 0.4)):
        rep[d]["curve"] = {"all": [v - 0.1, v]}
    for k in (1, 4, 5):
        rep["V0"]["all"].update({f"filt@{k}": 0.5, f"pass@{k}": 0.5})
    write_latex(rep, str(tmp_path))
    assert (tmp_path / "sekiller" / "sekil2-butce-egrisi.pdf").stat().st_size > 0
    # a further Y-design analysis with its own judge: table row, \e<tag> macros, extra curve
    import copy
    er = copy.deepcopy(rep)
    er["Y"]["all"].update({"v3": 0.45, "wilcoxon_p": 0.0001,
                           "iter1_vs_final_mcnemar": {"b": 7, "c": 1, "p": 0.03}})
    write_latex(rep, str(tmp_path), [("Gen", "Judge", "G", er)])
    mac = (tmp_path / "veri" / "ana_sonuclar.tex").read_text()
    assert "\\newcommand{\\eGY}{45{,}0}" in mac and "\\newcommand{\\eGW}{$<$~0{,}001}" in mac
    assert "\\newcommand{\\aVo}{50{,}0}" in mac          # main macros still there (pass@1 set to 0.5 above)
    yt = (tmp_path / "cizelgeler" / "cizelge-yargic.tex").read_text()
    assert "Gen & Judge & 50{,}0 & \\textbf{45{,}0} & $<$0{,}001 & 7/1 & 60{,}0 & 0{,}50 \\\\" in yt
    assert "DeepSeek-V3 & Qwen2.5-72B & 50{,}0 & \\textbf{30{,}0} & 0{,}200 & 1/0" in yt
