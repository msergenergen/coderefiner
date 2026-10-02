"""
Offline judge replay (main experiment, docs/analysis.md).

Re-judges every recorded attempt that passed all public tests with a candidate
judge and scores each verdict against the hidden tests. No code is generated:
the candidate receives the pipeline's own judge prompt (nodes/judgement.py)
rebuilt from the iteration JSON, i.e. the same problem excerpt, comment-stripped
code and Node 1/2 evidence the original judge saw. Only these attempts are
replayed because a failing public test forces REJECT whatever the judge says.

What it measures: the candidate's decision accuracy on the exact decisions that
determined the delivered code. What it cannot measure: how the loop would have
continued after a different verdict (that needs a full pipeline run); the
report gives a time/cost estimate for that run from the replay's latencies.

Usage (key and settings loaded by the wrapper):
  scripts/run_openrouter.sh judge-replay <model> <provider> <name> [--run-ids ...]
  python -m benchmark.judge_replay --name gpt-oss-120b --run-ids run_A run_B ...
  python -m benchmark.judge_replay --report outputs/openrouter/judge_replay

The candidate is the registry's judge slot (OPENROUTER_JUDGE_MODEL /
OPENROUTER_JUDGE_PROVIDER); every call is in the API ledger with
phase=judge_replay, so pipeline cost figures are unaffected.
"""
import argparse
import glob
import json
import os
import statistics
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from loguru import logger

from benchmark.paper_results import mcnemar_p
from config import model_registry, settings
from metrics.scoring import compute_numeric_scores
from models.budget import BudgetExceeded
from models.client import chat
from nodes.judgement import (JUDGEMENT_SYSTEM_PROMPT, _build_judgement_prompt,
                             _parse_judgement, _parses)

ORIGINAL = "original"


def load_attempts(run_ids: set, results_root: str, cache_path: str) -> list:
    """Public-passing attempts of the given runs with their hidden-test result."""
    with open(cache_path) as f:
        hidden = json.load(f)
    out, missing = [], 0
    for path in sorted(glob.glob(os.path.join(results_root, "*", "iterations", "*", "*.json"))):
        with open(path) as f:
            it = json.load(f)
        if it.get("run_id") not in run_ids:
            continue
        n2 = it.get("node2", {})
        if not (n2.get("total_count", 0) > 0 and n2.get("pass_rate", 0.0) >= 1.0):
            continue
        key = f"{it['run_id']}/{it['question_id']}/{it['iter']}"
        if key not in hidden:
            missing += 1
            continue
        it["_key"] = key
        it["_hidden_ok"] = hidden[key] >= 1.0
        out.append(it)
    if missing:
        logger.warning("{} public-passing attempts have no hidden result; skipped", missing)
    return out


def judge_messages(it: dict) -> list:
    """The judge prompt exactly as judgement_node builds it."""
    n1, n2, tr = it["node1"], it["node2"], it["training"]
    numeric = compute_numeric_scores(
        pass_rate=n2.get("pass_rate", 0.0), cpu_ms=n2.get("cpu_ms", 0.0),
        ram_mb=n2.get("ram_mb", 0.0), mi=n1.get("mi", 100.0), cc=n1.get("cc", 0.0),
        bandit_high=n1.get("bandit_high", False),
        bandit_medium_count=n1.get("bandit_medium_count", 0),
        executed=n2.get("total_count", 0) > 0, parsed=_parses(tr.get("generated_code", "")),
    )
    user = _build_judgement_prompt(
        numeric_scores=numeric,
        code=tr.get("stripped_code") or tr.get("generated_code", ""),
        problem=tr["problem"], node1_issues=n1.get("issues", []),
        failed_cases=n2.get("failed_cases", []), halstead=n1.get("halstead", 0.0),
        timed_out=n2.get("timed_out", False), difficulty=it["difficulty"],
        iteration=it["iter"],
    )
    return [{"role": "system", "content": JUDGEMENT_SYSTEM_PROMPT},
            {"role": "user", "content": user}]


def replay_one(it: dict, name: str) -> dict:
    t0 = time.perf_counter()
    row = {"key": it["_key"], "question_id": it["question_id"], "iter": it["iter"],
           "source_run": it["run_id"], "difficulty": it["difficulty"],
           "hidden_ok": it["_hidden_ok"], "judge": name}
    try:
        res = chat("QWEN", judge_messages(it), max_tokens=settings.judge_max_tokens,
                   tag={"phase": "judge_replay", "replay": name, "run_id": f"replay_{name}",
                        "question_id": it["question_id"], "iteration": it["iter"],
                        "source_run": it["run_id"]})
    except BudgetExceeded:
        raise
    except Exception as e:                       # recorded, never a fake verdict
        row.update(verdict="ERROR", reason=f"{type(e).__name__}: {str(e)[:200]}")
    else:
        parsed = _parse_judgement(res.text)
        verdict = parsed.get("verdict", "")
        row.update(verdict=verdict if verdict in ("ACCEPT", "REJECT") else "NONE",
                   reason=parsed.get("verdict_reason", "")[:400],
                   prompt_tokens=res.prompt_tokens, completion_tokens=res.completion_tokens,
                   reasoning_tokens=res.reasoning_tokens, cost_usd=res.cost_usd,
                   finish_reason=res.finish_reason)
    row["elapsed_s"] = round(time.perf_counter() - t0, 2)
    return row


def replay(attempts: list, name: str, out_path: str, workers: int) -> None:
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            done = {json.loads(line)["key"] for line in f if line.strip()}
    todo = [it for it in attempts if it["_key"] not in done]
    logger.info("judge replay {}: {} attempts ({} already done) with {}",
                name, len(todo), len(done), model_registry.get("QWEN").model_name)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as pool, open(out_path, "a") as out:
        futures = [pool.submit(replay_one, it, name) for it in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            out.write(json.dumps(fut.result(), ensure_ascii=False) + "\n")
            out.flush()
            if i % 20 == 0:
                logger.info("{}/{} replayed", i, len(todo))


def original_rows(attempts: list) -> list:
    """The pipeline's own decisions on the same attempts (node3_pass, so a
    missing verdict already carries its rubric fallback)."""
    return [{"key": it["_key"], "question_id": it["question_id"], "iter": it["iter"],
             "source_run": it["run_id"], "difficulty": it["difficulty"],
             "hidden_ok": it["_hidden_ok"], "judge": ORIGINAL,
             "verdict": it["node3"].get("verdict") or "NONE",
             "decision": "ACCEPT" if it["node3"].get("pass") else "REJECT",
             "elapsed_s": it.get("timing", {}).get("node3_ms", 0) / 1000}
            for it in attempts]


def decision(r: dict) -> str:
    """Effective pipeline decision. Without a usable verdict the pipeline falls
    back to the rubric threshold, which in the main runs never rejected
    public-passing code (docs/main-results.md), so it counts as ACCEPT."""
    if "decision" in r:
        return r["decision"]
    return r["verdict"] if r["verdict"] in ("ACCEPT", "REJECT") else "ACCEPT"


def decision_stats(rows: list) -> dict:
    """Confusion counts of the effective decision against the hidden tests."""
    acc_ok = sum(decision(r) == "ACCEPT" and r["hidden_ok"] for r in rows)
    acc_bad = sum(decision(r) == "ACCEPT" and not r["hidden_ok"] for r in rows)
    rej_ok = sum(decision(r) == "REJECT" and r["hidden_ok"] for r in rows)
    rej_bad = sum(decision(r) == "REJECT" and not r["hidden_ok"] for r in rows)
    n_ok = sum(r["hidden_ok"] for r in rows)
    lat = [r["elapsed_s"] for r in rows if r.get("elapsed_s")]
    return {
        "n": len(rows), "hidden_ok": n_ok,
        "accept": acc_ok + acc_bad, "reject": rej_ok + rej_bad,
        "no_verdict": sum(r["verdict"] not in ("ACCEPT", "REJECT") for r in rows),
        "accept_correct": acc_ok, "accept_wrong": acc_bad,
        "reject_correct_code": rej_ok, "reject_wrong_code": rej_bad,
        # share of accepted code that is right (what the loop delivers)
        "precision": acc_ok / (acc_ok + acc_bad) if acc_ok + acc_bad else None,
        # share of wrong code the judge stops (what the loop gets to repair)
        "wrong_caught": rej_bad / (len(rows) - n_ok) if len(rows) > n_ok else None,
        # share of right code it lets through
        "right_kept": acc_ok / n_ok if n_ok else None,
        "accuracy": (acc_ok + rej_bad) / len(rows) if rows else None,
        "latency_mean_s": round(statistics.mean(lat), 2) if lat else None,
        "latency_p90_s": round(sorted(lat)[int(0.9 * (len(lat) - 1))], 2) if lat else None,
        "cost_usd": round(sum(r.get("cost_usd") or 0.0 for r in rows), 4),
        "completion_tokens_mean": (round(statistics.mean(
            r["completion_tokens"] for r in rows if "completion_tokens" in r))
            if any("completion_tokens" in r for r in rows) else None),
    }


def paired(base: list, cand: list) -> dict:
    """Exact McNemar on 'decision agrees with the hidden tests', same attempts."""
    b = {r["key"]: r for r in base}
    pairs = []
    for r in cand:
        if r["key"] in b:
            right = lambda x: (decision(x) == "ACCEPT") == x["hidden_ok"]   # noqa: E731
            pairs.append((right(b[r["key"]]), right(r)))
    # mcnemar_p(pairs of (first, second)) -> (only second right, only first right, p)
    only_cand, only_orig, p = mcnemar_p(pairs)
    flips = sum(decision(b[r["key"]]) != decision(r) for r in cand if r["key"] in b)
    return {"n": len(pairs), "only_original_right": only_orig, "only_candidate_right": only_cand,
            "p": p, "verdict_changed": flips}


def full_run_estimate(all_iters: dict, cand: dict) -> dict:
    """Wall time and cost of one full Y run with the candidate as judge, assuming
    the same iterations (a different judge changes them: rough estimate only).
    Shards run in parallel, so the slowest shard sets the wall time."""
    per_shard = []
    judge_calls = 0
    for run_id, iters in all_iters.items():
        t = 0.0
        for it in iters:
            tm = it.get("timing", {})
            judged = it.get("tokens", {}).get("judgement_prompt", 0) > 0
            judge_calls += judged
            t += (tm.get("codegen_ms", 0) + tm.get("node1_ms", 0) + tm.get("node2_ms", 0)) / 1000
            t += cand["latency_mean_s"] if judged else 0.0
        per_shard.append(t)
    orig_shard = [sum(it.get("timing", {}).get("total_ms", 0) for it in iters) / 1000
                  for iters in all_iters.values()]
    cost_per_call = cand["cost_usd"] / cand["n"] if cand["n"] else 0.0
    return {"shards": len(all_iters), "judge_calls": judge_calls,
            "original_wall_min": round(max(orig_shard) / 60, 1) if orig_shard else None,
            "estimated_wall_min": round(max(per_shard) / 60, 1) if per_shard else None,
            "estimated_judge_cost_usd": round(judge_calls * cost_per_call, 3)}


def load_rows(path: str) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def report(out_dir: str, attempts: list, groups: dict, full_runs: dict) -> dict:
    orig = original_rows(attempts)
    rep = {"groups": {}, "candidates": {}}
    for gname, ids in groups.items():
        rep["groups"][gname] = {ORIGINAL: decision_stats([r for r in orig if r["source_run"] in ids])}
    for path in sorted(glob.glob(os.path.join(out_dir, "*.jsonl"))):
        name = os.path.basename(path)[:-len(".jsonl")]
        rows = load_rows(path)
        cand = {"all": decision_stats(rows), "vs_original": paired(orig, rows)}
        for gname, ids in groups.items():
            g = [r for r in rows if r["source_run"] in ids]
            rep["groups"][gname][name] = decision_stats(g)
            rep["groups"][gname][name]["vs_original"] = paired(
                [r for r in orig if r["source_run"] in ids], g)
        if full_runs:
            cand["full_run_estimate"] = full_run_estimate(full_runs, cand["all"])
        rep["candidates"][name] = cand
    rep["original_all"] = decision_stats(orig)
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    return rep


def iterations_of(run_ids: list, results_root: str) -> dict:
    runs = defaultdict(list)
    for path in glob.glob(os.path.join(results_root, "*", "iterations", "*", "*.json")):
        with open(path) as f:
            it = json.load(f)
        if it.get("run_id") in run_ids:
            runs[it["run_id"]].append(it)
    return dict(runs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-ids", nargs="+", required=True,
                    help="pipeline runs whose public-passing attempts are replayed")
    ap.add_argument("--groups", nargs="*", default=[],
                    help="NAME=run1,run2 subsets reported separately (e.g. the main design)")
    ap.add_argument("--name", help="candidate label (output file stem); omit with --report")
    ap.add_argument("--results-root", default="outputs/openrouter/results")
    ap.add_argument("--out", default="outputs/openrouter/judge_replay")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="first N attempts only (smoke test)")
    ap.add_argument("--estimate-runs", nargs="*", default=[],
                    help="runs of one full Y design (one per shard) for the time/cost estimate")
    ap.add_argument("--report", action="store_true", help="only (re)write report.json")
    args = ap.parse_args()

    cache = os.path.join(args.results_root, "hidden_eval_cache.json")
    attempts = load_attempts(set(args.run_ids), args.results_root, cache)
    if args.limit:
        attempts = attempts[:args.limit]
    if not args.report:
        if not args.name:
            ap.error("--name is required unless --report")
        replay(attempts, args.name, os.path.join(args.out, f"{args.name}.jsonl"), args.workers)
    groups = {"all": set(args.run_ids)}
    for g in args.groups:
        gname, ids = g.split("=", 1)
        groups[gname] = set(ids.split(","))
    full = iterations_of(args.estimate_runs, args.results_root) if args.estimate_runs else {}
    rep = report(args.out, attempts, groups, full)
    print(json.dumps(rep, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
