"""
Single-shot baseline (V0) with n independent samples per problem.

Each sample uses exactly the prompt of the pipeline's first iteration
(build_system_prompt() with no feedback + build_user_prompt(format_problem)),
so the only difference to the pipeline is the feedback loop. Every sample is
graded on the public tests and on all (public + private) tests with the same
local LCB-compatible evaluator the paper results use.

Metrics (docs/ablation.md §7):
  pass@1  — mean over problems of c/n (unbiased estimator, Chen et al. 2021)
  pass@k  — unbiased estimator 1 - C(n-c, k)/C(n, k)   (oracle upper bound)
  filt@k  — first of the first k samples that passes the public tests is
            submitted (else the k-th); realistic, no hidden-test access

Usage:
  python -m benchmark.sample_baseline --n 5 [--workers 8] [--resume FILE]
Problem set comes from the usual settings (LCB_DATE_FROM/TO, LCB_SAMPLE_SIZE,
MAX_PROBLEMS). Output: outputs/results/baseline_samples/<run_id>.jsonl
"""
import argparse
import concurrent.futures as cf
import json
import os
import threading
from collections import defaultdict
from datetime import datetime
from math import comb

from loguru import logger

from config import settings
from dataset.loader import load_lcb
from dataset.splitter import format_problem
from models.budget import BudgetExceeded
from models import tracing
from models.client import chat
from models.prompts import build_system_prompt, build_user_prompt
from nodes.codegen import _extract_code
from sandbox.lcb_eval import CORRECTNESS_TIMEOUT_SEC, _fallback_evaluate

OUT_DIR = os.path.join(settings.output_dir, "baseline_samples")


def _grade(code: str, tests: dict, starter: str, timeout: int, stop_on_fail: bool = False) -> float:
    if not code.strip() or not tests.get("input"):
        return 0.0
    row = {"all_tests": tests, "starter_code": starter}
    return _fallback_evaluate(code, row, timeout=timeout, stop_on_fail=stop_on_fail)["pass_rate"]


def _sample_untraced(row: dict, k: int, run_id: str) -> dict:
    qid = row["id"]
    res = chat(
        "M1",
        [
            {"role": "system", "content": build_system_prompt(None)},
            {"role": "user",   "content": build_user_prompt(format_problem(row))},
        ],
        max_tokens=settings.codegen_max_tokens,
        tag={"question_id": qid, "sample": k, "run_id": run_id, "phase": "baseline"},
    )
    code = _extract_code(res.text)
    return {
        "question_id":   qid,
        "difficulty":    row["difficulty"],
        "contest_date":  row.get("contest_date", ""),
        "sample":        k,
        "code":          code,
        "finish_reason": res.finish_reason,
        "completion_tokens": res.completion_tokens,
        "cost_usd":      res.cost_usd,
        # public: same limit as the pipeline's sandbox (Node 2); hidden: the
        # offline limit used for pipeline outputs too, stopping at the first
        # failure as lcb_runner does.
        "public":        _grade(code, row["public_tests"], row.get("starter_code", ""),
                                CORRECTNESS_TIMEOUT_SEC),
        "hidden":        _grade(code, row["all_tests"],    row.get("starter_code", ""),
                                settings.hidden_timeout_sec, stop_on_fail=True),
    }


def _sample(row: dict, k: int, run_id: str) -> dict:
    """One V0 sample; with Langfuse on, one trace per sample (session = run_id)."""
    with tracing.problem_trace(run_id, row["id"], f"V0-{k}", row["difficulty"],
                               phase="baseline"):
        rec = _sample_untraced(row, k, run_id)
        tracing.score("public_pass_rate", float(rec["public"]))
        tracing.score("hidden_pass", float(rec["hidden"] >= 1.0), data_type="BOOLEAN")
        return rec


def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def summarise(records: list, n: int) -> dict:
    by_q = defaultdict(list)
    for r in records:
        by_q[r["question_id"]].append(r)
    out = defaultdict(lambda: defaultdict(list))
    for qid, rs in by_q.items():
        rs = sorted(rs, key=lambda r: r["sample"])[:n]
        if len(rs) < n:
            continue
        c = sum(r["hidden"] >= 1.0 for r in rs)
        for group in ("all", rs[0]["difficulty"]):
            # pass@1 comes from the loop (pass_at_k(n, c, 1) == c / n)
            for k in range(1, n + 1):
                out[group][f"pass@{k}"].append(pass_at_k(n, c, k))
                first = next((r for r in rs[:k] if r["public"] >= 1.0), rs[k - 1])
                out[group][f"filt@{k}"].append(float(first["hidden"] >= 1.0))
    return {g: {m: round(100 * sum(v) / len(v), 1) for m, v in d.items()} | {"n_problems": len(d["pass@1"])}
            for g, d in out.items()}


def regrade(path: str, rows: dict, workers: int, n: int) -> None:
    """Re-grades stored samples (public and hidden) with the current rules.
    Used once on 2026-10-02: the first main baseline started with a 15 s
    public-test limit instead of Node 2's 10 s."""
    with open(path) as f:
        recs = [json.loads(line) for line in f]

    def one(r):
        row = rows[r["question_id"]]
        return {**r,
                "public": _grade(r["code"], row["public_tests"], row.get("starter_code", ""),
                                 CORRECTNESS_TIMEOUT_SEC),
                "hidden": _grade(r["code"], row["all_tests"], row.get("starter_code", ""),
                                 settings.hidden_timeout_sec, stop_on_fail=True)}

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        new = list(ex.map(one, recs))
    changed = sum((a["public"] >= 1.0) != (b["public"] >= 1.0) or
                  (a["hidden"] >= 1.0) != (b["hidden"] >= 1.0) for a, b in zip(recs, new))
    out = path.replace(".jsonl", ".regraded.jsonl")
    with open(out, "w") as f:
        for r in new:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"re-graded {len(new)} samples, {changed} pass/fail flips → {out}")
    print(json.dumps(summarise(new, n), indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--resume", help="existing JSONL to continue")
    ap.add_argument("--regrade", help="existing JSONL: re-grade every sample with the "
                    "current grading rules (no API calls) → <name>.regraded.jsonl")
    args = ap.parse_args()

    rows = load_lcb(split=settings.lcb_split, difficulty=settings.lcb_difficulty,
                    sample_size=settings.lcb_sample_size,
                    date_from=settings.lcb_date_from, date_to=settings.lcb_date_to)
    max_problems = int(os.getenv("MAX_PROBLEMS", "0"))
    if max_problems > 0:
        rows = rows[:max_problems]

    if args.regrade:
        regrade(args.regrade, {r["id"]: r for r in rows}, args.workers, args.n)
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    path = args.resume or os.path.join(
        OUT_DIR, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")
    run_id = os.path.splitext(os.path.basename(path))[0]
    done, records = set(), []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                records.append(r)
                done.add((r["question_id"], r["sample"]))

    jobs = [(row, k) for row in rows for k in range(args.n) if (row["id"], k) not in done]
    logger.info(f"{len(rows)} problems × n={args.n}: {len(jobs)} samples to run → {path}")

    lock = threading.Lock()
    with open(path, "a") as f, cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_sample, row, k, run_id) for row, k in jobs]
        for i, fut in enumerate(cf.as_completed(futs), 1):
            try:
                r = fut.result()
            except BudgetExceeded as e:
                logger.error(str(e))
                for fu in futs:
                    fu.cancel()
                break
            except Exception as e:
                # Missing samples leave the problem out of the summary;
                # rerun with --resume to fill them in.
                logger.warning(f"sample failed: {type(e).__name__}: {str(e)[:200]}")
                continue
            with lock:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
            records.append(r)
            if i % 20 == 0:
                logger.info(f"{i}/{len(jobs)} samples done")

    tracing.flush()
    print(json.dumps(summarise(records, args.n), indent=2))


if __name__ == "__main__":
    main()
