"""Offline judge replay (benchmark/judge_replay.py): attempt selection, prompt
reconstruction and the decision statistics, without any API call."""
import json

from benchmark import judge_replay as jr


def _iteration(run_id="run_x", qid="q1", it=1, pass_rate=1.0, total=3, verdict="ACCEPT",
               passed=True, judged=True):
    return {
        "run_id": run_id, "question_id": qid, "iter": it, "difficulty": "easy",
        "status": "PASS" if passed else "LOOP",
        "timing": {"codegen_ms": 1000, "node1_ms": 100, "node2_ms": 100,
                   "node3_ms": 20000 if judged else 0, "total_ms": 21200 if judged else 1200},
        "tokens": {"judgement_prompt": 2000 if judged else 0},
        "node1": {"cc": 3, "mi": 70.0, "halstead": 50.0, "bandit_high": False,
                  "bandit_medium_count": 0, "issues": []},
        "node2": {"pass_rate": pass_rate, "total_count": total, "cpu_ms": 10.0,
                  "ram_mb": 0.0, "timed_out": False, "failed_cases": []},
        "node3": {"verdict": verdict, "pass": passed},
        "training": {"generated_code": "print(1)\n", "stripped_code": "print(1)\n",
                     "problem": "## Problem: demo\nPrint one.\n"},
    }


def _write(tmp_path, items, hidden):
    root = tmp_path / "results"
    for d in items:
        p = root / "2026-10-02" / "iterations" / d["question_id"]
        p.mkdir(parents=True, exist_ok=True)
        (p / f"M1_iter{d['iter']}_{d['run_id']}.json").write_text(json.dumps(d))
    cache = root / "hidden_eval_cache.json"
    cache.write_text(json.dumps(hidden))
    return str(root), str(cache)


def test_load_attempts_keeps_public_passing_with_hidden_result(tmp_path):
    items = [_iteration(qid="q1"), _iteration(qid="q2", pass_rate=0.5),
             _iteration(qid="q3"), _iteration(run_id="other", qid="q4")]
    root, cache = _write(tmp_path, items, {"run_x/q1/1": 1.0, "run_x/q2/1": 0.0,
                                           "other/q4/1": 1.0})
    got = jr.load_attempts({"run_x"}, root, cache)
    # q2 failed a public test, q3 has no hidden result, q4 is another run
    assert [a["question_id"] for a in got] == ["q1"]
    assert got[0]["_hidden_ok"] is True


def test_judge_messages_match_pipeline_prompt():
    msgs = jr.judge_messages(_iteration())
    assert msgs[0]["content"] == jr.JUDGEMENT_SYSTEM_PROMPT
    assert "All visible tests passed" in msgs[1]["content"]
    assert "print(1)" in msgs[1]["content"]


def test_missing_verdict_counts_as_rubric_fallback_accept():
    assert jr.decision({"verdict": "NONE", "hidden_ok": True}) == "ACCEPT"
    assert jr.decision({"verdict": "ERROR", "hidden_ok": True}) == "ACCEPT"
    assert jr.decision({"verdict": "REJECT", "hidden_ok": True}) == "REJECT"
    # the pipeline's own rows carry the decision it actually took
    assert jr.decision({"verdict": "NONE", "decision": "REJECT"}) == "REJECT"


def test_decision_stats_precision_and_caught():
    rows = [{"verdict": "ACCEPT", "hidden_ok": True}, {"verdict": "ACCEPT", "hidden_ok": False},
            {"verdict": "REJECT", "hidden_ok": False}, {"verdict": "REJECT", "hidden_ok": True},
            {"verdict": "NONE", "hidden_ok": False}]
    s = jr.decision_stats(rows)
    assert (s["accept"], s["reject"], s["no_verdict"]) == (3, 2, 1)
    assert s["precision"] == 1 / 3
    assert s["wrong_caught"] == 1 / 3
    assert s["right_kept"] == 1 / 2


def test_paired_labels_follow_mcnemar_orientation():
    base = [{"key": "a", "verdict": "ACCEPT", "hidden_ok": False},   # original wrong
            {"key": "b", "verdict": "ACCEPT", "hidden_ok": True}]    # both right
    cand = [{"key": "a", "verdict": "REJECT", "hidden_ok": False},   # candidate right
            {"key": "b", "verdict": "ACCEPT", "hidden_ok": True}]
    p = jr.paired(base, cand)
    assert (p["only_candidate_right"], p["only_original_right"]) == (1, 0)
    assert p["verdict_changed"] == 1


def test_full_run_estimate_uses_slowest_shard():
    runs = {"s0": [_iteration(), _iteration(it=2)], "s1": [_iteration(judged=False)]}
    est = jr.full_run_estimate(runs, {"latency_mean_s": 60.0, "cost_usd": 0.2, "n": 100})
    assert est["judge_calls"] == 2
    assert est["estimated_wall_min"] == round((2 * 1.2 + 2 * 60) / 60, 1)
    assert est["estimated_judge_cost_usd"] == round(2 * 0.002, 3)
