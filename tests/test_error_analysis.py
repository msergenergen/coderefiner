"""benchmark/error_analysis.py — categories, judge-decision analysis and the scans."""
import json
import math

from benchmark.error_analysis import (FEEDBACK, REASONS, Grader, accepts, auc, category, coverage,
                                      feedback, outcomes, parses, rejects, static_auc, transitions)


def _it(qid, i, status, *, hidden=None, pub=True, verdict="", reason="", code="print(1)",
        diff="hard", fb="", cc=5, mi=70.0, hal=100.0):
    return {"run_id": "r", "question_id": qid, "iter": i, "status": status, "difficulty": diff,
            "training": {"generated_code": code},
            "node1": {"cc": cc, "mi": mi, "halstead": hal},
            "node2": {"total_count": 2, "pass_rate": 1.0 if pub else 0.5},
            "node3": {"verdict": verdict, "verdict_reason": reason, "improved_prompt": fb}}


def _grader(cache, official=None):
    return Grader(cache, official or {})


def test_category_order():
    assert category(True, "print(1)", None) == "correct"
    assert category(True, "print(1)", None, stopped=True) == "stopped"
    assert category(False, "def f(:\n", -4) == "unparseable"       # lcb_runner calls it -4
    assert category(False, "", -2) == "unparseable"
    assert category(False, "print(1)", -2) == "wrong_answer"
    assert category(False, "print(1)", -3) == "time_limit"
    assert category(False, "print(1)", -4) == "runtime_error"
    assert category(False, "print(1)", -5) == "other"
    assert parses("x = 1") and not parses("x = (")


def test_auc_counts_ties_as_half():
    assert auc([3, 4], [1, 2]) == 1.0
    assert auc([1], [1]) == 0.5
    assert auc([1, 3], [2]) == 0.5
    assert math.isnan(auc([], [1]))


def test_grader_reads_cache_official_and_stop():
    g = _grader({"r/q/1": 1.0, "r/q/2": 0.5}, {"r/q/2": -3})
    assert g.attempt(_it("q", 1, "LOOP")) == "correct"
    assert g.attempt(_it("q", 2, "STOP_MAX")) == "time_limit"
    assert g.attempt(_it("q", 3, "STOP_HIGH")) == "stopped"
    s = {"question_id": "q", "sample": 0, "code": "print(1)", "hidden": 0.0}
    assert _grader({}, {"b/q/s0": -2}).sample("b", s) == "wrong_answer"


def test_outcomes_transitions_and_accepts():
    probs = {
        "a": [_it("a", 1, "LOOP", pub=False), _it("a", 2, "PASS", verdict="ACCEPT")],   # fixed
        "b": [_it("b", 1, "PASS", verdict="ACCEPT")],                                  # false accept
        "c": [_it("c", 1, "STOP_HIGH")],
    }
    official = {"r/a/1": -2, "r/b/1": -3}
    g = _grader({"r/a/1": 0.0, "r/a/2": 1.0, "r/b/1": 0.9}, official)
    o = outcomes(probs, g)
    assert o["all"] == {"correct": 1, "time_limit": 1, "stopped": 1}
    assert o["wrong_public"] == {"public_pass": 1, "stopped": 1}
    assert o["wrong_end"] == {"PASS": 1, "STOP_HIGH": 1}
    t = transitions(probs, g)
    assert t["fixed_from"] == {"wrong_answer": 1} and t["broken"] == 0
    assert t["ever_correct"] == 1 and t["ever_but_not_delivered"] == 0
    a = accepts(probs, g)
    assert a["accepted"] == 2 and a["wrong"] == 1 and a["wrong_by_category"] == {"time_limit": 1}


def test_rejects_of_public_passing_code():
    probs = {
        # wrong code rightly rejected for its complexity, then fixed
        "a": [_it("a", 1, "LOOP", verdict="REJECT", reason="O(n^2) is too slow for n = 2e5"),
              _it("a", 2, "PASS", verdict="ACCEPT")],
        # correct code rejected naming an input; the problem still ends wrong
        "b": [_it("b", 1, "LOOP", verdict="REJECT", reason="fails for input [1, 2]"),
              _it("b", 2, "STOP_MAX", pub=False, verdict="REJECT")],
        # a REJECT of code that failed public tests is not counted
        "c": [_it("c", 1, "STOP_MAX", pub=False, verdict="REJECT", reason="wrong")],
    }
    g = _grader({"r/a/1": 0.2, "r/a/2": 1.0, "r/b/1": 1.0, "r/b/2": 0.0}, {"r/a/1": -3})
    r = rejects(probs, g)
    assert r["n"] == 2 and r["hidden_wrong"] == 1 and r["hidden_correct"] == 1
    assert r["classes"]["complexity"] == 1 and r["classes"]["input"] == 1
    assert r["classes"]["input_or_complexity"] == 2 and r["classes"]["none"] == 0
    assert r["complexity_reason_and_time_limit"] == 1
    assert r["wrong_problems_fixed"] == 1 and r["correct_problems_final_wrong"] == 1


def test_reason_and_feedback_scans():
    assert REASONS["static"].search("Cyclomatic complexity is 25")
    assert not REASONS["input"].search("The approach is fine")
    assert FEEDBACK["algorithm"].search("Use a Fenwick tree") and FEEDBACK["algorithm"].search("try DP")
    assert FEEDBACK["formula"].search("dp[i] = min(dp[i-1], x)")
    assert FEEDBACK["complexity"].search("aim for O(n log n)")
    assert not FEEDBACK["complexity"].search("Optimize the loop")
    probs = {"a": [_it("a", 1, "LOOP", fb="Use binary search; O(log n) per query"),
                   _it("a", 2, "STOP_HIGH", fb="ignored"), _it("a", 3, "PASS", fb="")]}
    f = feedback(probs)
    assert f == {"n": 1, "algorithm": 1, "formula": 0, "complexity": 1, "algorithm_or_formula": 1}


def test_static_auc_uses_first_public_passing_attempt():
    probs = {"a": [_it("a", 1, "LOOP", pub=False, cc=99), _it("a", 2, "PASS", cc=2)],
             "b": [_it("b", 1, "PASS", cc=8)]}
    g = _grader({"r/a/2": 1.0, "r/b/1": 0.0}, {"r/b/1": -2})
    s = static_auc([probs], g)
    assert s["hard"]["n"] == 2 and s["hard"]["correct"] == 1 and s["hard"]["cc"] == 0.0


def test_coverage_pools_loop_and_baseline(tmp_path):
    designs = {"Y": {"a": [_it("a", 1, "PASS")], "b": [_it("b", 1, "STOP_MAX")]}}
    baselines = {"base": {"b": [{"question_id": "b", "sample": 0, "hidden": 0.0, "difficulty": "hard"}],
                          "c": [{"question_id": "c", "sample": 0, "hidden": 1.0, "difficulty": "easy"}]}}
    g = _grader({"r/a/1": 1.0, "r/b/1": 0.0})
    c = coverage(designs, baselines, g)
    assert c == {"problems": 3, "solved_by_none": 1, "solved_by_none_by_difficulty": {"hard": 1},
                 "only_loop": 1, "only_baseline": 1}
    json.dumps(c)
