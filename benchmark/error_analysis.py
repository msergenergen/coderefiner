"""
Error analysis of the main experiment (docs/analysis.md §13): why the delivered code
fails, which failures the loop repairs, and where the judge's decisions go wrong.

No model calls and no re-execution. Every input is already on disk:
  - iteration logs (<results-root>/<date>/iterations/<qid>/*.json),
  - the local evaluator's hidden-test results (<results-root>/hidden_eval_cache.json,
    the baseline JSONL's "hidden" field) — correctness, as in every other number,
  - lcb_runner's error_code of the first failing hidden test for every graded program
    (official_regrade/official.jsonl, docs/analysis.md §12): -2 wrong answer, -3 time
    limit exceeded (6 s per test), -4 runtime error. Every program the local evaluator
    fails also fails there (official_only = 0), so each local failure has a code.
    lcb_runner reports a syntax error as -4; code that ast.parse rejects (mostly output
    cut off at the token limit) is counted separately as "unparseable".

Inputs
  --design NAME RUN_ID...   repeatable; the shard runs of one design (loaded together)
  --baseline NAME JSONL     repeatable; benchmark/sample_baseline.py output (TÖ samples)
  --auc-designs NAME...     designs whose first public-passing attempts feed static_auc
  --coverage-designs/--coverage-baselines NAME...   sources pooled in coverage (default all)
  --official                official_regrade/official.jsonl

Reported (<out>/error_analysis.json, <out>/report.md)
  outcomes     delivered code per design × difficulty: correct / wrong_answer /
               time_limit / runtime_error / unparseable / stopped (Severity Gate);
               wrong delivered code by public-test status and by how the loop ended
  baseline     the same categories over every baseline sample
  transitions  category of the first attempt → category of the delivered code
  accepts      loop-ending accepts (status PASS) whose code fails the hidden tests,
               by category and difficulty (judge ACCEPT in Y, rubric threshold in P)
  rejects      judge REJECTs of public-passing code: reason classes (keyword scan),
               hidden outcome, and whether the problem still ended correct
  feedback     share of judge feedback naming an algorithm / data structure, a
               formula, or a complexity bound (keyword scan)
  static_auc   within-difficulty AUC (correct > wrong) of CC, MI and Halstead volume
               over each problem's first public-passing attempt
  coverage     problems no attempt of any source solved; solved only by the loop
               or only by baseline samples
  timing       wall time per attempt and per problem

Usage
  python -m benchmark.error_analysis \
      --design B_Y run_20261002_100638 run_20261002_100640 run_20261002_100642 run_20261002_100644 \
      --design B_P run_20261002_100646 ... --baseline deepseek <jsonl> \
      --auc-designs B_Y B_P --out outputs/openrouter/error_analysis
"""
import argparse
import ast
import json
import os
import re
from collections import Counter, defaultdict

from benchmark.openrouter_results import _STATIC
from benchmark.paper_results import DIFFS, load_problems

CATEGORIES = ("correct", "wrong_answer", "time_limit", "runtime_error", "unparseable", "stopped", "other")
_ERROR_CODE = {-2: "wrong_answer", -3: "time_limit", -4: "runtime_error"}

# Reasons of a REJECT of public-passing code. The judge prompt requires such a REJECT
# to name a concrete input or a complexity violation (docs/node3.md).
REASONS = {
    "input": re.compile(r"\binputs?\b|counter-?example|edge case|corner case|e\.g\.|for example|"
                        r"\bwhen\s+\w+\s*(=|is|==)|\[\s*-?\d", re.I),
    "complexity": re.compile(r"complexit|\bO\(|too slow|time limit|\bTLE\b|time out|timeout|"
                             r"efficien|exceed|constraint|perform", re.I),
    "static": _STATIC,
}

# Judge feedback that points to a solution: algorithm / data-structure names, formulas,
# complexity bounds. The paper's one-off scan (§11 of docs/main-results.md) used a
# narrower, unrecorded list; these three groups replace it.
FEEDBACK = {
    "algorithm": re.compile(
        r"dynamic programming|\bDP\b|dp\[|memoi[sz]|segment tree|fenwick|binary indexed|dijkstra|"
        r"binary search|two[- ]pointers?|sliding window|prefix sums?|union[- ]find|disjoint[- ]set|"
        r"\bBFS\b|\bDFS\b|breadth[- ]first|depth[- ]first|topological|greedy|heap|priority queue|"
        r"sieve|\btrie\b|monotonic (stack|queue|deque)|bitmask|\bKMP\b", re.I),
    "formula": re.compile(r"=\s*(min|max)\(|recurrence|formula|modular inverse|factorial|binomial|\bnCr\b",
                          re.I),
    "complexity": re.compile(r"\bO\([^)]*\)"),
}


def parses(code: str) -> bool:
    if not code.strip():
        return False
    try:
        ast.parse(code)
    except (SyntaxError, ValueError):
        return False
    return True


def category(hidden_ok: bool, code: str, error_code, stopped: bool = False) -> str:
    """Outcome of one program on the hidden tests."""
    if stopped:
        return "stopped"
    if hidden_ok:
        return "correct"
    if not parses(code):
        return "unparseable"
    return _ERROR_CODE.get(error_code, "other")


def auc(pos: list, neg: list) -> float:
    """P(a correct attempt's value > a wrong one's), ties count one half."""
    if not pos or not neg:
        return float("nan")
    s = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return s / (len(pos) * len(neg))


def code_of(it: dict) -> str:
    return it["training"].get("generated_code") or it["training"].get("stripped_code", "")


def public_pass(it: dict) -> bool:
    n2 = it.get("node2") or {}
    return n2.get("total_count", 0) > 0 and n2.get("pass_rate", 0.0) >= 1.0


def verdict(it: dict) -> str:
    return (it["node3"].get("verdict") or "").upper()


def platform(qid: str) -> str:
    return "leetcode" if qid.isdigit() else "atcoder"


class Grader:
    """Hidden-test category of a pipeline attempt or a baseline sample."""

    def __init__(self, cache: dict, official: dict):
        self.cache, self.official = cache, official

    def attempt(self, it: dict) -> str:
        key = f"{it['run_id']}/{it['question_id']}/{it['iter']}"
        if it["status"] == "STOP_HIGH":
            return "stopped"
        return category(self.cache.get(key, 0.0) >= 1.0, code_of(it), self.official.get(key))

    def sample(self, stem: str, r: dict) -> str:
        key = f"{stem}/{r['question_id']}/s{r['sample']}"
        return category(r["hidden"] >= 1.0, r.get("code") or "", self.official.get(key))


def by_difficulty(rows: list, cat_of) -> dict:
    out = {"all": dict(Counter(cat_of(r) for r in rows))}
    for d, _ in DIFFS:
        out[d] = dict(Counter(cat_of(r) for r in rows if r["difficulty"] == d))
    return out


def outcomes(probs: dict, g: Grader) -> dict:
    last = [iters[-1] for iters in probs.values()]
    res = by_difficulty(last, g.attempt)
    wrong = [it for it in last if g.attempt(it) != "correct"]
    res["wrong_public"] = dict(Counter("stopped" if it["status"] == "STOP_HIGH" else
                                       ("public_pass" if public_pass(it) else "public_fail")
                                       for it in wrong))
    res["wrong_end"] = dict(Counter(it["status"] for it in wrong))
    res["by_platform"] = {p: dict(Counter(g.attempt(it) for it in last if platform(it["question_id"]) == p))
                          for p in ("atcoder", "leetcode")}
    return res


def transitions(probs: dict, g: Grader) -> dict:
    pairs = Counter((g.attempt(iters[0]), g.attempt(iters[-1])) for iters in probs.values())
    fixed = Counter(a for (a, b), n in pairs.items() for _ in range(n) if a != "correct" and b == "correct")
    broken = sum(n for (a, b), n in pairs.items() if a == "correct" and b != "correct")
    ever = sum(any(g.attempt(it) == "correct" for it in iters) for iters in probs.values())
    return {"pairs": {f"{a}->{b}": n for (a, b), n in sorted(pairs.items())},
            "fixed_from": dict(fixed), "broken": broken,
            "ever_correct": ever,
            "ever_but_not_delivered": sum(any(g.attempt(it) == "correct" for it in iters)
                                          and g.attempt(iters[-1]) != "correct"
                                          for iters in probs.values())}


def accepts(probs: dict, g: Grader) -> dict:
    """Loop-ending accepts (status PASS) and how the wrong ones fail."""
    acc = [it for iters in probs.values() for it in iters if it["status"] == "PASS"]
    wrong = [it for it in acc if g.attempt(it) != "correct"]
    return {"accepted": len(acc), "wrong": len(wrong),
            "wrong_by_category": dict(Counter(g.attempt(it) for it in wrong)),
            "wrong_by_difficulty": dict(Counter(it["difficulty"] for it in wrong)),
            "accepted_by_difficulty": dict(Counter(it["difficulty"] for it in acc))}


def rejects(probs: dict, g: Grader) -> dict:
    """Judge REJECTs of code that passed every public test (verdict design)."""
    rows = []
    for iters in probs.values():
        final_ok = g.attempt(iters[-1]) == "correct"
        for it in iters:
            if it["status"] == "STOP_HIGH" or not public_pass(it) or verdict(it) != "REJECT":
                continue
            reason = it["node3"].get("verdict_reason") or ""
            rows.append({"qid": it["question_id"], "cat": g.attempt(it), "final_ok": final_ok,
                         "classes": [k for k, rx in REASONS.items() if rx.search(reason)]})
    wrong = [r for r in rows if r["cat"] != "correct"]
    right = [r for r in rows if r["cat"] == "correct"]

    def cls(sel):
        c = Counter(k for r in sel for k in r["classes"])
        c["input_or_complexity"] = sum(bool({"input", "complexity"} & set(r["classes"])) for r in sel)
        c["none"] = sum(not r["classes"] for r in sel)
        return dict(c)
    return {
        "n": len(rows),
        "classes": cls(rows),
        "hidden_wrong": len(wrong), "hidden_correct": len(right),
        "hidden_wrong_by_category": dict(Counter(r["cat"] for r in wrong)),
        "complexity_reason_and_time_limit": sum("complexity" in r["classes"] and r["cat"] == "time_limit"
                                                for r in wrong),
        "complexity_reason_wrong": sum("complexity" in r["classes"] for r in wrong),
        "classes_hidden_correct": cls(right),
        # problem-level fate: a problem can be rejected several times
        "wrong_problems": len({r["qid"] for r in wrong}),
        "wrong_problems_fixed": len({r["qid"] for r in wrong if r["final_ok"]}),
        "correct_problems": len({r["qid"] for r in right}),
        "correct_problems_final_wrong": len({r["qid"] for r in right if not r["final_ok"]}),
    }


def feedback(probs: dict) -> dict:
    texts = [it["node3"]["improved_prompt"] for iters in probs.values() for it in iters
             if it["status"] != "STOP_HIGH" and (it["node3"].get("improved_prompt") or "").strip()]
    out = {"n": len(texts)}
    for k, rx in FEEDBACK.items():
        out[k] = sum(bool(rx.search(t)) for t in texts)
    out["algorithm_or_formula"] = sum(bool(FEEDBACK["algorithm"].search(t) or FEEDBACK["formula"].search(t))
                                      for t in texts)
    return out


def static_auc(designs: list, g: Grader) -> dict:
    """First public-passing attempt per problem and design; does a static metric
    separate hidden-correct from hidden-wrong code within a difficulty level?"""
    pts = []
    for probs in designs:
        for iters in probs.values():
            first = next((it for it in iters if it["status"] != "STOP_HIGH" and public_pass(it)), None)
            if first:
                pts.append((first["difficulty"], g.attempt(first) == "correct", first["node1"]))
    out = {}
    for d in ["all"] + [d for d, _ in DIFFS]:
        sel = [p for p in pts if d == "all" or p[0] == d]
        ok = [n1 for _, c, n1 in sel if c]
        bad = [n1 for _, c, n1 in sel if not c]
        out[d] = {"n": len(sel), "correct": len(ok), "wrong": len(bad),
                  **{m: auc([float(x[m]) for x in ok], [float(x[m]) for x in bad])
                     for m in ("cc", "mi", "halstead")}}
    return out


def coverage(designs: dict, baselines: dict, g: Grader) -> dict:
    solved_loop, solved_base, diff = defaultdict(bool), defaultdict(bool), {}
    for probs in designs.values():
        for q, iters in probs.items():
            diff[q] = iters[0]["difficulty"]
            solved_loop[q] |= any(g.attempt(it) == "correct" for it in iters)
    for stem, by_q in baselines.items():
        for q, rs in by_q.items():
            diff.setdefault(q, rs[0]["difficulty"])
            solved_base[q] |= any(g.sample(stem, r) == "correct" for r in rs)
    qs = sorted(diff)
    none = [q for q in qs if not solved_loop[q] and not solved_base[q]]
    return {"problems": len(qs),
            "solved_by_none": len(none),
            "solved_by_none_by_difficulty": dict(Counter(diff[q] for q in none)),
            "only_loop": sum(solved_loop[q] and not solved_base[q] for q in qs),
            "only_baseline": sum(solved_base[q] and not solved_loop[q] for q in qs)}


def timing(probs: dict) -> dict:
    its = [it for iters in probs.values() for it in iters]
    t = lambda k: [float(it["timing"].get(k, 0)) / 1000 for it in its]  # noqa: E731
    total = t("total_ms")
    return {"attempts": len(its),
            "attempt_mean_s": sum(total) / len(its),
            "codegen_mean_s": sum(t("codegen_ms")) / len(its),
            "judge_mean_s": sum(t("node3_ms")) / len(its),
            "problem_mean_s": sum(total) / len(probs)}


def load_baseline(path: str) -> dict:
    by_q = defaultdict(list)
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            by_q[r["question_id"]].append(r)
    return dict(by_q)


def report_md(res: dict) -> str:
    def pct(k, n):
        return f"{100 * k / n:.1f}" if n else "–"
    lines = ["# Error analysis", ""]
    lines += ["## Delivered code (problems)", "",
              "| Design | n | " + " | ".join(CATEGORIES) + " |",
              "|---|---|" + "---|" * len(CATEGORIES)]
    for name, o in res["outcomes"].items():
        a = o["all"]
        n = sum(a.values())
        lines.append(f"| {name} | {n} | " + " | ".join(str(a.get(c, 0)) for c in CATEGORIES) + " |")
    lines += ["", "## Baseline samples", "",
              "| Source | n | " + " | ".join(CATEGORIES) + " |", "|---|---|" + "---|" * len(CATEGORIES)]
    for name, o in res["baseline"].items():
        a = o["all"]
        n = sum(a.values())
        lines.append(f"| {name} | {n} | " + " | ".join(str(a.get(c, 0)) for c in CATEGORIES) + " |")
    lines += ["", "## Loop-ending accepts that fail the hidden tests", ""]
    for name, a in res["accepts"].items():
        lines.append(f"- {name}: {a['wrong']}/{a['accepted']} wrong ({pct(a['wrong'], a['accepted'])} %); "
                     f"{a['wrong_by_category']}")
    lines += ["", "## Judge REJECTs of public-passing code", ""]
    for name, r in res["rejects"].items():
        lines.append(f"- {name}: n {r['n']}, hidden wrong {r['hidden_wrong']} {r['hidden_wrong_by_category']}, "
                     f"classes {r['classes']}; wrong problems fixed {r['wrong_problems_fixed']}/{r['wrong_problems']}; "
                     f"correct-code problems ending wrong {r['correct_problems_final_wrong']}/{r['correct_problems']}")
    lines += ["", "## Feedback naming a solution element", ""]
    for name, f in res["feedback"].items():
        lines.append(f"- {name}: n {f['n']}; algorithm {pct(f['algorithm'], f['n'])} %, formula "
                     f"{pct(f['formula'], f['n'])} %, complexity bound {pct(f['complexity'], f['n'])} %")
    lines += ["", "## Static metrics vs hidden correctness (AUC, correct > wrong)", ""]
    for d, a in res["static_auc"].items():
        lines.append(f"- {d}: n {a['n']} ({a['correct']}/{a['wrong']}); CC {a['cc']:.2f}, MI {a['mi']:.2f}, "
                     f"Halstead {a['halstead']:.2f}")
    c = res["coverage"]
    lines += ["", "## Coverage", "",
              f"- {c['solved_by_none']}/{c['problems']} problems solved by no attempt "
              f"{c['solved_by_none_by_difficulty']}; only loop {c['only_loop']}, only baseline {c['only_baseline']}"]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", nargs="+", action="append", default=[], metavar=("NAME", "RUN_ID"))
    ap.add_argument("--baseline", nargs=2, action="append", default=[], metavar=("NAME", "JSONL"))
    ap.add_argument("--auc-designs", nargs="*", default=[])
    ap.add_argument("--coverage-designs", nargs="*", default=None, help="default: every design")
    ap.add_argument("--coverage-baselines", nargs="*", default=None, help="default: every baseline")
    ap.add_argument("--official", default="outputs/openrouter/official_regrade/official.jsonl")
    ap.add_argument("--results-root", default="outputs/openrouter/results")
    ap.add_argument("--out", default="outputs/openrouter/error_analysis")
    args = ap.parse_args()

    with open(os.path.join(args.results_root, "hidden_eval_cache.json")) as f:
        cache = json.load(f)
    official = {}
    with open(args.official) as f:
        for line in f:
            r = json.loads(line)
            official[r["key"]] = r.get("error_code")
    g = Grader(cache, official)

    designs = {d[0]: load_problems(set(d[1:]), args.results_root) for d in args.design}
    baselines = {}
    for name, path in args.baseline:
        baselines[name] = (os.path.basename(path).split(".")[0], load_baseline(path))

    res = {
        "outcomes": {n: outcomes(p, g) for n, p in designs.items()},
        "baseline": {n: by_difficulty([r for rs in by_q.values() for r in rs],
                                      lambda r, s=stem: g.sample(s, r))
                     for n, (stem, by_q) in baselines.items()},
        "transitions": {n: transitions(p, g) for n, p in designs.items()},
        "accepts": {n: accepts(p, g) for n, p in designs.items()},
        "rejects": {n: rejects(p, g) for n, p in designs.items()},
        "feedback": {n: feedback(p) for n, p in designs.items()},
        "static_auc": static_auc([designs[n] for n in args.auc_designs], g),
        "coverage": coverage({n: p for n, p in designs.items()
                              if args.coverage_designs is None or n in args.coverage_designs},
                             {s: q for n, (s, q) in baselines.items()
                              if args.coverage_baselines is None or n in args.coverage_baselines}, g),
        "timing": {n: timing(p) for n, p in designs.items()},
    }
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "error_analysis.json"), "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    with open(os.path.join(args.out, "report.md"), "w") as f:
        f.write(report_md(res))
    print(report_md(res))


if __name__ == "__main__":
    main()
