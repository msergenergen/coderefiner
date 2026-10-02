"""
Results of the main, contamination-free experiment (docs/ablation.md §7).

Inputs
  --baseline   JSONL from benchmark/sample_baseline.py (V0: n samples/problem)
  --y-run-ids  pipeline runs with JUDGE_DECISION=verdict  (design Y, proposed)
  --p-run-ids  pipeline runs with JUDGE_DECISION=rubric   (design P, comparison)
  --ledger     API ledger (cost per phase/run)

Every comparison uses only the problems present in all given sources, so it
is paired. Correctness = all public + private tests pass, graded offline with
the same local evaluator for baseline samples and pipeline code.

Reported
  V0 pass@1 (mean c/n), pass@k and filt@k for k=1..n           (sample_baseline)
  V3 strict accuracy of the delivered code, Wilson 95% CI       (per design)
  V3 − V0: mean paired difference, bootstrap 95% CI, Wilcoxon p
  Budget-matched: V3 vs filt@k̄ / pass@k̄, k̄ = mean generations per problem
  Y vs P: exact McNemar on delivered code
  Judge: rejections of public-passing code and whether hidden tests agree;
         code fragments in judge feedback (solution-leak check)
  Cost: $ per phase and per solved problem (from the ledger)

Usage
  python -m benchmark.openrouter_results --baseline <jsonl> \
      --y-run-ids run_A run_B --p-run-ids run_C run_D \
      [--results-root outputs/openrouter/results] [--out outputs/openrouter/report]
"""
import argparse
import json
import math
import os
import random
import re
from collections import defaultdict

from scipy.stats import wilcoxon

from benchmark.paper_results import (DIFFS, analyse, hidden_evaluate, lcb_meta,
                                     load_problems, mcnemar_p, num, pval_rel, wilson)
from benchmark.paper_results import pct as tr_pct
from benchmark.sample_baseline import pass_at_k
from config.pipeline import settings

# Inline mentions of the answer format ("wrap code in ```python ... ```")
# are instructions, not code.
_INLINE_FENCE = re.compile(r"```\w*\s*\.\.\.\s*```")
_BLOCK = re.compile(r"```[\w+-]*\n(.*?)```", re.S)
# Lines that only shape a skeleton (signatures, imports, comments, pass).
_SKELETON = re.compile(r"^\s*(#.*|pass|\.\.\.|import .*|from \S+ import .*|class \w+.*:|def .*:|if __name__.*:|\w+\(\))?\s*$")
_LOGIC_LINE = re.compile(r"^\s*(return |for .+:\s*$|while .+:\s*$|\w+(\[.*\])?\s*[+\-*/]?=\s*\S)", re.M)


def has_code(text: str) -> bool:
    """True if judge feedback contains solution logic, not just a format
    instruction or an empty skeleton (signature + comments + pass)."""
    text = _INLINE_FENCE.sub("", text or "")
    for block in _BLOCK.findall(text):
        if any(not _SKELETON.match(line) for line in block.splitlines()):
            return True
    outside = _BLOCK.sub("", text)
    return len(_LOGIC_LINE.findall(outside)) >= 2


def load_baseline(path: str) -> dict:
    by_q = defaultdict(list)
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            by_q[r["question_id"]].append(r)
    return {q: sorted(rs, key=lambda r: r["sample"]) for q, rs in by_q.items()}


def bootstrap_ci(values: list, reps: int = 10000, seed: int = 42) -> tuple:
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choice(values) for _ in range(n)) / n for _ in range(reps))
    return means[int(0.025 * reps)], means[int(0.975 * reps) - 1]


def baseline_metrics(base: dict, qids: list, n: int) -> dict:
    out = {}
    for key, sel in [("all", qids)] + [(d, [q for q in qids if base[q][0]["difficulty"] == d])
                                       for d, _ in DIFFS]:
        if not sel:
            continue
        m = {"n_problems": len(sel)}
        cs = [sum(r["hidden"] >= 1.0 for r in base[q][:n]) for q in sel]
        for k in range(1, n + 1):
            m[f"pass@{k}"] = sum(pass_at_k(n, c, k) for c in cs) / len(sel)
            filt = []
            for q in sel:
                rs = base[q][:k]
                first = next((r for r in rs if r["public"] >= 1.0), rs[-1])
                filt.append(first["hidden"] >= 1.0)
            m[f"filt@{k}"] = sum(filt) / len(sel)
        m["pass@1_ci"] = bootstrap_ci([c / n for c in cs])
        out[key] = m
    return out


def design_metrics(res: dict, base: dict, n: int) -> dict:
    rows = {r["qid"]: r for r in res["rows"]}
    qids = sorted(rows)
    out = {}
    for key, sel in [("all", qids)] + [(d, [q for q in qids if rows[q]["diff"] == d]) for d, _ in DIFFS]:
        if not sel:
            continue
        k3 = sum(rows[q]["v3"] for q in sel)
        diffs = [rows[q]["v3"] - sum(r["hidden"] >= 1.0 for r in base[q][:n]) / n for q in sel]
        nz = [d for d in diffs if d != 0]
        b, c, p = mcnemar_p([(rows[q]["v0"], rows[q]["v3"]) for q in sel])
        out[key] = {
            "n_problems": len(sel), "v3": k3 / len(sel), "v3_ci": wilson(k3, len(sel)),
            "iter1": sum(rows[q]["v0"] for q in sel) / len(sel),
            "iter1_vs_final_mcnemar": {"b": b, "c": c, "p": p},
            "delta_vs_v0": sum(diffs) / len(sel), "delta_ci": bootstrap_ci(diffs),
            "wilcoxon_p": wilcoxon(nz).pvalue if len(nz) >= 1 else 1.0,
            "mean_generations": sum(rows[q]["n_iter"] for q in sel) / len(sel),
        }
    return out


def budget_matched(design: dict, v0: dict, n: int) -> dict:
    """V3 against baseline selections that use about as many generations."""
    kbar = design["all"]["mean_generations"]
    lo, hi = min(n, max(1, math.floor(kbar))), min(n, max(1, math.ceil(kbar)))
    a = v0["all"]
    return {"k_bar": kbar, "v3": design["all"]["v3"],
            f"filt@{lo}": a[f"filt@{lo}"], f"filt@{hi}": a[f"filt@{hi}"],
            f"pass@{lo}": a[f"pass@{lo}"], f"pass@{hi}": a[f"pass@{hi}"]}


def judge_metrics(res: dict, problems: dict) -> dict:
    pr = res["proc"]
    judged = [it for its in problems.values() for it in its if it["status"] != "STOP_HIGH"]
    leaks = [it for it in judged if has_code(it["node3"].get("improved_prompt", ""))]
    return {
        "public_passing": pr["pub_n"], "rejected": pr["pub_rej"],
        "rejected_and_hidden_wrong": pr["pub_rej_good"], "rejected_but_hidden_correct": pr["pub_rej_bad"],
        "accepted_precision": pr["rub_prec"], "verdict_from_judge": pr["verdict_judge"],
        "judge_llm_error": pr["judge_err"], "json_error": pr["json_err"],
        "feedback_with_code": len(leaks), "feedback_total": len(judged),
    }


def cost(ledger: str, groups: dict) -> dict:
    spend = defaultdict(float)
    calls = defaultdict(int)
    if os.path.exists(ledger):
        with open(ledger) as f:
            for line in f:
                e = json.loads(line)
                # run ids are second-stamped, so a baseline and a pipeline
                # process started in the same second share one (happened on
                # 2026-10-02); the "phase" tag tells them apart.
                is_base = e.get("phase") == "baseline"
                for name, ids in groups.items():
                    if e.get("run_id") in ids and (name == "baseline") == is_base:
                        spend[name] += e.get("cost_usd", 0.0)
                        calls[name] += 1
    return {name: {"usd": round(spend[name], 4), "calls": calls[name]} for name in groups}


def push_hidden_scores(designs: dict) -> int:
    """Langfuse: hidden-test outcome of the first and the delivered code of
    every problem, on the trace the pipeline wrote (models/tracing.py)."""
    from models import tracing
    n = 0
    for probs in designs.values():
        for qid, iters in probs.items():
            first, last = iters[0], iters[-1]
            tid = tracing.trace_id_for(last["run_id"], qid, last.get("model", "M1"))
            if tid is None:
                return 0                      # Langfuse not configured
            tracing.score("hidden_pass", float(last["_hidden"] >= 1.0), data_type="BOOLEAN",
                          trace_id=tid, comment=f"delivered code, iteration {last['iter']}")
            tracing.score("hidden_pass_first", float(first["_hidden"] >= 1.0),
                          data_type="BOOLEAN", trace_id=tid)
            n += 1
    tracing.flush()
    return n


def write_latex(rep: dict, out_dir: str) -> None:
    """Paper macros (prefix \\a…) and the main results table, Turkish number
    format, from one report. Writes <out_dir>/veri/ana_sonuclar.tex and
    <out_dir>/cizelgeler/cizelge-ana.tex (both overwritten; do not hand-edit)."""
    v0, Y, P, bud, jd = rep["V0"], rep["Y"], rep["P"], rep["budget"], rep["judge"]
    m = {"aN": rep["n"], "aNkolay": v0["easy"]["n_problems"],
         "aNorta": v0["medium"]["n_problems"], "aNzor": v0["hard"]["n_problems"],
         "aVo": tr_pct(v0["all"]["pass@1"]), "aVoLo": tr_pct(v0["all"]["pass@1_ci"][0]),
         "aVoHi": tr_pct(v0["all"]["pass@1_ci"][1]), "aPassFive": tr_pct(v0["all"]["pass@5"])}
    for d, D in (("Y", Y), ("P", P)):
        a = D["all"]
        m.update({f"a{d}": tr_pct(a["v3"]), f"a{d}Lo": tr_pct(a["v3_ci"][0]), f"a{d}Hi": tr_pct(a["v3_ci"][1]),
                  f"a{d}Gain": num(100 * a["delta_vs_v0"]),
                  f"a{d}GainLo": num(100 * a["delta_ci"][0]), f"a{d}GainHi": num(100 * a["delta_ci"][1]),
                  f"a{d}W": pval_rel(a["wilcoxon_p"]), f"a{d}First": tr_pct(a["iter1"]),
                  f"a{d}B": a["iter1_vs_final_mcnemar"]["b"], f"a{d}C": a["iter1_vs_final_mcnemar"]["c"],
                  f"a{d}Mp": pval_rel(a["iter1_vs_final_mcnemar"]["p"]),
                  f"a{d}Gen": num(a["mean_generations"], 2),
                  f"a{d}Prec": tr_pct(jd[d]["accepted_precision"]),
                  f"a{d}Cost": num(rep["cost"][d]["usd"], 2)})
    kb = bud["Y"]
    m.update({"aFiltTwo": tr_pct(rep["V0"]["all"]["filt@2"]), "aFiltThree": tr_pct(rep["V0"]["all"]["filt@3"]),
              "aPassTwo": tr_pct(rep["V0"]["all"]["pass@2"]), "aPassThree": tr_pct(rep["V0"]["all"]["pass@3"]),
              "aYPub": jd["Y"]["public_passing"], "aYRej": jd["Y"]["rejected"],
              "aYRejGood": jd["Y"]["rejected_and_hidden_wrong"],
              "aYRejBad": jd["Y"]["rejected_but_hidden_correct"],
              "aYJudge": tr_pct(jd["Y"]["verdict_from_judge"]),
              "aLeak": jd["Y"]["feedback_with_code"] + jd["P"]["feedback_with_code"],
              "aFeedback": jd["Y"]["feedback_total"] + jd["P"]["feedback_total"],
              "aBaseCost": num(rep["cost"]["baseline"]["usd"], 2)})
    if "Y_vs_P" in rep:
        m.update({"aDb": rep["Y_vs_P"]["b"], "aDc": rep["Y_vs_P"]["c"], "aDp": pval_rel(rep["Y_vs_P"]["p"])})
    for d, c in rep.get("vs_compare", {}).items():
        m.update({f"aAbl{d}": tr_pct(c["other"]), f"aAbl{d}B": c["b"], f"aAbl{d}C": c["c"],
                  f"aAbl{d}p": pval_rel(c["p"])})
    # English abstract: decimal point
    en = lambda x: f"{100 * x:.1f}"
    m.update({"aVoEn": en(v0["all"]["pass@1"]), "aYEn": en(Y["all"]["v3"]), "aPEn": en(P["all"]["v3"]),
              "aYPrecEn": en(jd["Y"]["accepted_precision"]), "aPPrecEn": en(jd["P"]["accepted_precision"]),
              "aFiltTwoEn": en(rep["V0"]["all"]["filt@2"]),
              "aPWEn": "$<$~0.001" if P["all"]["wilcoxon_p"] < 0.001 else f"$=$~{P['all']['wilcoxon_p']:.3f}",
              "aYWEn": "$<$~0.001" if Y["all"]["wilcoxon_p"] < 0.001 else f"$=$~{Y['all']['wilcoxon_p']:.3f}"})
    os.makedirs(os.path.join(out_dir, "veri"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "cizelgeler"), exist_ok=True)
    with open(os.path.join(out_dir, "veri", "ana_sonuclar.tex"), "w") as f:
        f.write("% Auto-generated by benchmark/openrouter_results.py --latex; do not edit by hand.\n")
        for k, v in m.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    rows = [("Kolay", "easy"), ("Orta", "medium"), ("Zor", "hard"), ("\\textbf{Toplam}", "all")]
    L = ["% Auto-generated by benchmark/openrouter_results.py --latex; do not edit by hand.",
         "\\begin{tabular}{|l|c|c|c|c|}", "\\hline",
         "\\textbf{Zorluk} & \\textbf{n} & \\textbf{V0} & \\textbf{V3$_Y$} & \\textbf{V3$_P$} \\\\ \\hline"]
    for name, k in rows:
        L.append(f"{name} & {v0[k]['n_problems']} & {tr_pct(v0[k]['pass@1'])} & {tr_pct(Y[k]['v3'])} & "
                 f"{tr_pct(P[k]['v3'])} \\\\ \\hline")
    L.append("\\end{tabular}")
    with open(os.path.join(out_dir, "cizelgeler", "cizelge-ana.tex"), "w") as f:
        f.write("\n".join(L) + "\n")


def compare_rows(rows: dict, other: dict) -> dict:
    """Paired comparison of delivered-code correctness with another analysis
    (same design, common problems). Pairs are (other, this), so b = only
    this run correct, c = only the other run correct."""
    out = {}
    for name, cur in rows.items():
        prev = other.get(name, {})
        common = sorted(set(cur) & set(prev))
        if not common:
            continue
        b, c, p = mcnemar_p([(prev[q], cur[q]) for q in common])
        out[name] = {"n": len(common), "this": sum(bool(cur[q]) for q in common) / len(common),
                     "other": sum(bool(prev[q]) for q in common) / len(common),
                     "b": b, "c": c, "p": p}
    return out


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def write_report(rep: dict, path: str) -> None:
    v0, L = rep["V0"], []
    L.append(f"# Ana deney sonuçları (n = {rep['n']} problem, eşleştirilmiş)\n")
    L.append("| Zorluk | n | V0 pass@1 | V3 (Y) | V3 (P) |")
    L.append("|---|---|---|---|---|")
    for key, name in [("all", "Toplam")] + [(d, tr) for d, tr in DIFFS]:
        if key not in v0:
            continue
        y = rep.get("Y", {}).get(key)
        p = rep.get("P", {}).get(key)
        L.append(f"| {name} | {v0[key]['n_problems']} | {pct(v0[key]['pass@1'])} | "
                 f"{pct(y['v3']) if y else '–'} | {pct(p['v3']) if p else '–'} |")
    for d in ("Y", "P"):
        if d in rep:
            a = rep[d]["all"]
            L.append(f"\n**{d}:** V3 − V0 = {100*a['delta_vs_v0']:+.1f} puan "
                     f"(%95 GA {100*a['delta_ci'][0]:+.1f} … {100*a['delta_ci'][1]:+.1f}; Wilcoxon p = {a['wilcoxon_p']:.3g}); "
                     f"ilk iterasyon → nihai McNemar b/c = {a['iter1_vs_final_mcnemar']['b']}/{a['iter1_vs_final_mcnemar']['c']}, "
                     f"p = {a['iter1_vs_final_mcnemar']['p']:.3g}; ortalama üretim {a['mean_generations']:.2f}.")
            L.append(f"Eşit bütçe: {json.dumps({k: round(v, 3) for k, v in rep['budget'][d].items()})}")
            L.append(f"Yargıç: {json.dumps(rep['judge'][d])}")
    if "Y_vs_P" in rep:
        c = rep["Y_vs_P"]
        L.append(f"\n**Y vs P** (teslim edilen kod): b/c = {c['b']}/{c['c']}, p = {c['p']:.3g}")
    for name, c in rep.get("vs_compare", {}).items():
        L.append(f"\n**{name}, karşılaştırılan analize göre** (n = {c['n']}): {pct(c['this'])} vs "
                 f"{pct(c['other'])}; yalnız bu koşuda doğru / yalnız ötekinde doğru = "
                 f"{c['b']}/{c['c']}, McNemar p = {c['p']:.3g}")
    L.append(f"\n**Maliyet:** {json.dumps(rep['cost'])}")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--y-run-ids", nargs="+", default=[])
    ap.add_argument("--p-run-ids", nargs="+", default=[])
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--results-root", default="outputs/openrouter/results")
    ap.add_argument("--ledger", default="outputs/api_usage.jsonl")
    ap.add_argument("--out", default="outputs/openrouter/report")
    ap.add_argument("--latex", help="paper directory: write veri/ana_sonuclar.tex and "
                    "cizelgeler/cizelge-ana.tex (needs both designs)")
    ap.add_argument("--langfuse", action="store_true",
                    help="attach hidden-test results as scores to the runs' Langfuse traces")
    ap.add_argument("--compare", help="results.json of another analysis (e.g. an ablation): "
                    "paired McNemar per design on the delivered code")
    args = ap.parse_args()

    base = {q: rs for q, rs in load_baseline(args.baseline).items() if len(rs) >= args.n}
    meta = lcb_meta()
    cache = os.path.join(args.results_root, "hidden_eval_cache.json")
    designs = {}
    for name, ids in (("Y", args.y_run_ids), ("P", args.p_run_ids)):
        if ids:
            probs = load_problems(set(ids), args.results_root)
            hidden_evaluate(probs, meta, cache, timeout=settings.hidden_timeout_sec,
                            stop_on_fail=True)
            designs[name] = probs

    if args.langfuse:
        push_hidden_scores(designs)

    qids = set(base)
    for probs in designs.values():
        qids &= set(probs)
    qids = sorted(qids)
    if not qids:
        raise SystemExit("No problems common to the baseline and the given runs.")

    rep = {"n": len(qids), "V0": baseline_metrics(base, qids, args.n),
           "budget": {}, "judge": {}}
    analysed = {}
    for name, probs in designs.items():
        probs = {q: probs[q] for q in qids}
        res = analyse(probs, meta)
        analysed[name] = res
        rep[name] = design_metrics(res, base, args.n)
        rep["budget"][name] = budget_matched(rep[name], rep["V0"], args.n)
        rep["judge"][name] = judge_metrics(res, probs)
    rep["rows"] = {name: {r["qid"]: r["v3"] for r in res["rows"]} for name, res in analysed.items()}
    if args.compare:
        rep["vs_compare"] = compare_rows(rep["rows"], json.load(open(args.compare)).get("rows", {}))
    if "Y" in analysed and "P" in analysed:
        rp = {r["qid"]: r for r in analysed["P"]["rows"]}
        b, c, p = mcnemar_p([(rp[r["qid"]]["v3"], r["v3"]) for r in analysed["Y"]["rows"]])
        rep["Y_vs_P"] = {"b": b, "c": c, "p": p}
    rep["cost"] = cost(args.ledger, {
        "baseline": {os.path.basename(args.baseline).split(".")[0]},   # also *.regraded.jsonl
        "Y": set(args.y_run_ids), "P": set(args.p_run_ids)})

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False, default=float)
    write_report(rep, os.path.join(args.out, "report.md"))
    if args.latex:
        write_latex(rep, args.latex)
    print(open(os.path.join(args.out, "report.md")).read())


if __name__ == "__main__":
    main()
