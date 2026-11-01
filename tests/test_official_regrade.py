"""benchmark/official_regrade.py — the parts that do not need lcb_runner."""
import json

from benchmark.official_regrade import collect_items, official_pass, summarise


def test_official_pass_follows_lcb_runner():
    assert official_pass([True, True])
    assert not official_pass([True, False])
    assert not official_pass([True, -1])          # timeout / runtime error
    assert not official_pass([-2])                # compile error
    assert not official_pass([])
    assert official_pass([1, 1]) and not official_pass([1, 0])


def test_collect_items_reads_baseline_and_cached_pipeline_attempts(tmp_path):
    base = tmp_path / "run_b.regraded.jsonl"
    base.write_text("\n".join(json.dumps(r) for r in [
        {"question_id": "q1", "sample": 0, "code": "print(1)", "hidden": 1.0},
        {"question_id": "q1", "sample": 1, "code": "print(2)", "hidden": 0.5}]))
    root = tmp_path / "results"
    d = root / "2026-10-02" / "iterations" / "q1"
    d.mkdir(parents=True)
    def it(i, status, code):
        return {"run_id": "run_x", "question_id": "q1", "iter": i, "status": status,
                "difficulty": "easy", "training": {"generated_code": code}}
    (d / "M1_iter1_run_x.json").write_text(json.dumps(it(1, "LOOP", "a")))
    (d / "M1_iter2_run_x.json").write_text(json.dumps(it(2, "STOP_HIGH", "b")))
    (root / "hidden_eval_cache.json").write_text(json.dumps({"run_x/q1/1": 1.0}))
    items = collect_items([str(base)], ["run_x"], str(root))
    keys = {i["key"]: i for i in items}
    assert set(keys) == {"run_b/q1/s0", "run_b/q1/s1", "run_x/q1/1"}   # STOP_HIGH never graded
    assert keys["run_b/q1/s0"]["local"] and not keys["run_b/q1/s1"]["local"]
    assert keys["run_x/q1/1"]["local"] and keys["run_x/q1/1"]["code"] == "a"
    assert keys["run_x/q1/1"]["group"] == "run:run_x"


def test_collect_items_keeps_every_run_of_a_shared_problem(tmp_path):
    # designs Y, P, C and D solve the same problems: each run's attempts must be kept
    root = tmp_path / "results"
    d = root / "2026-10-02" / "iterations" / "q1"
    d.mkdir(parents=True)
    cache = {}
    for rid in ("run_y", "run_p"):
        (d / f"M1_iter1_{rid}.json").write_text(json.dumps(
            {"run_id": rid, "question_id": "q1", "iter": 1, "status": "PASS",
             "difficulty": "easy", "training": {"generated_code": rid}}))
        cache[f"{rid}/q1/1"] = 1.0
    (root / "hidden_eval_cache.json").write_text(json.dumps(cache))
    items = collect_items([], ["run_y", "run_p"], str(root))
    assert {i["key"]: i["code"] for i in items} == {"run_y/q1/1": "run_y", "run_p/q1/1": "run_p"}


def test_summarise_counts_agreement_per_group():
    rows = [{"key": "a", "group": "g1", "local": True, "official": True, "error_code": None},
            {"key": "b", "group": "g1", "local": False, "official": False, "error_code": -2},
            {"key": "c", "group": "g2", "local": True, "official": False, "error_code": -3},
            {"key": "d", "group": "g2", "local": False, "official": True, "error_code": None}]
    s = summarise(rows)
    assert s["all"]["agree"] == 2 and s["all"]["local_only"] == 1 and s["all"]["official_only"] == 1
    assert s["groups"]["g1"]["agreement"] == 1.0 and s["groups"]["g2"]["agreement"] == 0.0
    assert [d["key"] for d in s["disagreements"]] == ["c", "d"]
