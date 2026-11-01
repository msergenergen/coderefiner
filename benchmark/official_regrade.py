"""
Re-grade recorded code with LiveCodeBench's official evaluator (lcb_runner) and compare
with the pinned local evaluator (sandbox/lcb_eval.py, LCB_EVALUATOR=local) that graded
every number of the paper (docs/analysis.md §12).

Tests are built exactly as lcb_runner does (CodeGenerationProblem.get_evaluation_sample:
public + private tests, fn_name from the metadata) from the same cached dataset the
pipeline used; each program is run with lcb_runner.evaluation.compute_code_generation_metrics
.check_correctness (per-test timeout, first failure stops). Correct = every test result > 0,
as in lcb_runner's pass@k.

Graded items
  --baseline  JSONL files of benchmark/sample_baseline.py: every sample (local = "hidden")
  --run-ids   pipeline runs: every attempt the local evaluator graded (local result from
              <results-root>/hidden_eval_cache.json; STOP_HIGH attempts were never graded)

Usage (lcb_runner is not pip-installed; point to a clone of the official repository)
  git clone https://github.com/LiveCodeBench/LiveCodeBench <dir>
  python -m benchmark.official_regrade --lcb-repo <dir> \
      --baseline outputs/openrouter/results/baseline_samples/run_20261002_091834.regraded.jsonl \
      --run-ids run_20261002_100638 ... --out outputs/openrouter/official_regrade
Results are appended to <out>/official.jsonl (resumable) and summarised in <out>/summary.json.
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed


def official_pass(res: list) -> bool:
    """lcb_runner counts a program correct when every per-test result is > 0
    (True); False, -1 (timeout/runtime error), -2 (compile error) fail."""
    return bool(res) and all((r is True) or (not isinstance(r, bool) and r > 0) for r in res)


def _grade(item: dict, sample: dict, timeout: int) -> dict:
    from lcb_runner.evaluation.compute_code_generation_metrics import check_correctness
    try:
        res, meta = check_correctness(sample, item["code"], timeout=timeout, debug=False)
        res = [r.item() if hasattr(r, "item") else r for r in res]
        out = {"official": official_pass(res), "error_code": meta.get("error_code")}
    except Exception as e:   # lcb_runner's own runner treats this as a failure too
        out = {"official": False, "error_code": -5, "error": repr(e)[:200]}
    return {**{k: item[k] for k in ("key", "group", "question_id", "local")}, **out}


def collect_items(baselines: list, run_ids: list, results_root: str) -> list:
    items = []
    for path in baselines:
        stem = os.path.basename(path).split(".")[0]
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                items.append({"key": f"{stem}/{r['question_id']}/s{r['sample']}",
                              "group": f"baseline:{stem}", "question_id": r["question_id"],
                              "code": r.get("code") or "", "local": r["hidden"] >= 1.0})
    if run_ids:
        sys.path.insert(0, os.getcwd())
        from benchmark.paper_results import load_problems
        with open(os.path.join(results_root, "hidden_eval_cache.json")) as f:
            cache = json.load(f)
        # one run at a time: load_problems keys by question_id and the designs' runs share
        # the same problems, so loading them together keeps only one run per problem
        for rid in run_ids:
            for qid, iters in load_problems({rid}, results_root).items():
                for it in iters:
                    key = f"{it['run_id']}/{qid}/{it['iter']}"
                    if it["status"] == "STOP_HIGH" or key not in cache:
                        continue
                    code = it["training"].get("generated_code") or it["training"].get("stripped_code", "")
                    items.append({"key": key, "group": f"run:{it['run_id']}", "question_id": qid,
                                  "code": code, "local": cache[key] >= 1.0})
    return items


def summarise(rows: list) -> dict:
    """Agreement of the two evaluators per group and overall: both pass, both fail,
    local only, official only; pass rates under each."""
    def block(sel):
        c = Counter((r["local"], r["official"]) for r in sel)
        n = len(sel)
        return {"n": n, "agree": c[(True, True)] + c[(False, False)],
                "both_pass": c[(True, True)], "both_fail": c[(False, False)],
                "local_only": c[(True, False)], "official_only": c[(False, True)],
                "agreement": (c[(True, True)] + c[(False, False)]) / n if n else 0.0,
                "local_rate": sum(r["local"] for r in sel) / n if n else 0.0,
                "official_rate": sum(r["official"] for r in sel) / n if n else 0.0}
    groups = defaultdict(list)
    for r in rows:
        groups[r["group"]].append(r)
    return {"all": block(rows), "groups": {g: block(s) for g, s in sorted(groups.items())},
            "disagreements": [{k: r[k] for k in ("key", "local", "official", "error_code")}
                              for r in rows if r["local"] != r["official"]]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lcb-repo", required=True, help="clone of github.com/LiveCodeBench/LiveCodeBench")
    ap.add_argument("--baseline", nargs="*", default=[])
    ap.add_argument("--run-ids", nargs="*", default=[])
    ap.add_argument("--results-root", default="outputs/openrouter/results")
    ap.add_argument("--out", default="outputs/openrouter/official_regrade")
    ap.add_argument("--timeout", type=int, default=6, help="per-test timeout (lcb_runner default 6 s)")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    repo = os.path.abspath(args.lcb_repo)
    sys.path.insert(0, repo)
    # spawned workers (and lcb_runner's own per-program process) import lcb_runner too
    os.environ["PYTHONPATH"] = os.pathsep.join([repo, os.getcwd(), os.environ.get("PYTHONPATH", "")])
    from datasets import load_dataset
    from lcb_runner.benchmarks.code_generation import CodeGenerationProblem

    items = collect_items(args.baseline, args.run_ids, args.results_root)
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "official.jsonl")
    done = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                r = json.loads(line)
                done[r["key"]] = r
    todo = [it for it in items if it["key"] not in done]
    print(f"{len(items)} programs, {len(done)} already graded, {len(todo)} to grade")

    # Same dataset (and cache) as the pipeline's loader; tests built the official way.
    need = {it["question_id"] for it in todo}
    samples = {}
    for row in load_dataset("livecodebench/code_generation_lite", split="test", trust_remote_code=True):
        if row["question_id"] in need:
            samples[row["question_id"]] = CodeGenerationProblem(**row).get_evaluation_sample()
    missing = need - set(samples)
    if missing:
        raise SystemExit(f"problems missing from the dataset: {sorted(missing)[:5]}")

    with open(out_path, "a") as f, ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_grade, it, samples[it["question_id"]], args.timeout) for it in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            done[r["key"]] = r
            f.write(json.dumps(r) + "\n")
            f.flush()
            if i % 100 == 0:
                print(f"  {i}/{len(todo)}", flush=True)

    rows = [done[it["key"]] for it in items]
    summary = summarise(rows)
    summary["timeout"] = args.timeout
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    a = summary["all"]
    print(f"agreement {a['agree']}/{a['n']} ({100 * a['agreement']:.1f}%); local only {a['local_only']}, "
          f"official only {a['official_only']}")
    for g, b in summary["groups"].items():
        print(f"  {g:32s} {b['agree']:4d}/{b['n']:4d}  local {100 * b['local_rate']:.1f}%  "
              f"official {100 * b['official_rate']:.1f}%")


if __name__ == "__main__":
    main()
