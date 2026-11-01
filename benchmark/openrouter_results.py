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
  Node 1: severity of every attempt; Severity-Gate stops by reason and whether
         the stopped code would pass the hidden tests; rubric threshold below
         which public-passing code fell; judge REJECTs and feedback that name a
         static metric (keyword scan)
  Noise floor: pass@1 of each single V0 sample; Δ↑/Δ↓ (fixed/broken vs. the
         first attempt, in % of problems) for the literature table
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
from collections import Counter, defaultdict

from scipy.stats import wilcoxon

from benchmark.paper_results import (DIFFS, analyse, hidden_evaluate, lcb_meta,
                                     load_problems, mcnemar_p, num, pval_rel, strict, wilson)
from benchmark.paper_results import pct as tr_pct
from benchmark.sample_baseline import pass_at_k
from config.pipeline import settings
from config.thresholds import thresholds

# Judge text that names a Node-1 tool or metric (keyword scan; time complexity is not static).
_STATIC = re.compile(r"cyclomatic|maintainab|halstead|bandit|mypy|ruff|static analy|\bCC\b|\bMI\b", re.I)

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
        # pass@1 of each single sample: the spread of a one-sample run (noise floor)
        m["per_sample"] = [sum(base[q][s]["hidden"] >= 1.0 for q in sel) / len(sel) for s in range(n)]
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


def stop_reason(it: dict) -> str:
    n1 = it["node1"]
    if n1.get("cc", 0) > thresholds.node1.cc_hard_limit:
        return "cc"
    if any("Forbidden import" in s for s in n1.get("issues", [])):
        return "import"
    return "bandit" if n1.get("bandit_high") else "other"


def stopped_hidden(meta: dict, timeout: int):
    """Hidden pass rate of a STOP_HIGH attempt's code. The delivered result of
    such a problem stays a failure; this only says what the gate cost."""
    from sandbox.lcb_eval import _fallback_evaluate

    def run(it):
        code = it["training"].get("generated_code") or ""
        if "all_tests" not in it["training"]:
            return None   # public log archive: tests stripped (scripts/export_release_logs.py)
        row = {"all_tests": it["training"]["all_tests"],
               "starter_code": meta.get(it["question_id"], ("", ""))[1]}
        if not code.strip() or not row["all_tests"].get("input"):
            return 0.0
        return _fallback_evaluate(code, row, timeout=timeout, stop_on_fail=True)["pass_rate"]
    return run


def static_evidence(problems: dict, stopped_pass=None) -> dict:
    """What Node 1 did in one design's runs: severity of every attempt, Severity-Gate
    stops by reason and difficulty (and, given stopped_pass(it) -> hidden pass rate,
    how many stopped codes were correct; None without stopped_pass or tests),
    public-passing attempts under the rubric
    threshold, judge REJECTs of public-passing code whose reason names a static
    metric, and judge feedback that names one."""
    its = [it for iters in problems.values() for it in iters]
    stops = [it for it in its if it["status"] == "STOP_HIGH"]
    judged = [it for it in its if it["status"] != "STOP_HIGH"]
    pub = [it for it in judged if it["node2"].get("pass_rate", 0) == 1.0]
    rej = [it for it in pub if (it["node3"].get("verdict") or "").upper() == "REJECT"]
    rej_static = [it for it in rej if _STATIC.search(it["node3"].get("verdict_reason") or "")]
    fb = [it["node3"]["improved_prompt"] for it in judged if (it["node3"].get("improved_prompt") or "").strip()]
    count = lambda xs: dict(sorted(Counter(xs).items()))
    passed = [stopped_pass(it) for it in stops] if stopped_pass else [None]
    return {
        "attempts": len(its),
        "severity": count([it["node1"]["severity"] for it in its]),
        "stops": len(stops),
        "stops_by_reason": count([stop_reason(it) for it in stops]),
        "stops_by_difficulty": count([it["difficulty"] for it in stops]),
        "stops_hidden_correct": (None if None in passed else sum(p == 1.0 for p in passed)),
        "public_passing": len(pub),
        "public_below_threshold": sum(it["node3"].get("total_score", 0)
                                      < thresholds.node3.for_difficulty(it["difficulty"]) for it in pub),
        "public_rejected": len(rej),
        "public_rejected_static": len(rej_static),
        "public_rejected_static_hidden_correct": sum(strict(it) for it in rej_static),
        "feedback": len(fb),
        "feedback_static": sum(bool(_STATIC.search(t)) for t in fb),
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


def write_latex(rep: dict, out_dir: str, extras: list = ()) -> None:
    """Paper macros (prefix \\a…) and the main results table, Turkish number
    format, from one report. Writes <out_dir>/veri/ana_sonuclar.tex and
    <out_dir>/cizelgeler/cizelge-ana.tex (both overwritten; do not hand-edit).
    extras: (generator, judge, tag, report) of further Y-design analyses; they
    add \\e<tag>… macros, rows of cizelgeler/cizelge-yargic.tex and, when they
    share the main baseline, a curve in the budget figure."""
    v0, Y, P, jd = rep["V0"], rep["Y"], rep["P"], rep["judge"]
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
                  f"a{d}Cost": num(rep["cost"][d]["usd"], 2),
                  # Δ↑/Δ↓ (CTRL, Xie et al. 2025): % of problems fixed / broken vs. the first attempt
                  f"a{d}Up": tr_pct(a["iter1_vs_final_mcnemar"]["b"] / rep["n"]),
                  f"a{d}Down": tr_pct(a["iter1_vs_final_mcnemar"]["c"] / rep["n"])})
    if "per_sample" in v0["all"]:
        m.update({"aVoMin": tr_pct(min(v0["all"]["per_sample"])),
                  "aVoMax": tr_pct(max(v0["all"]["per_sample"]))})
    if "static" in rep:
        sy, sp = rep["static"]["Y"], rep["static"]["P"]
        m.update({"aStops": sy["stops"] + sp["stops"],
                  "aStopsCC": sy["stops_by_reason"].get("cc", 0) + sp["stops_by_reason"].get("cc", 0),
                  "aStopsOk": ("?" if None in (sy["stops_hidden_correct"], sp["stops_hidden_correct"])
                               else sy["stops_hidden_correct"] + sp["stops_hidden_correct"]),
                  "aYStatRej": sy["public_rejected_static"],
                  "aYStatRejOk": sy["public_rejected_static_hidden_correct"]})
        for d, s in (("Y", sy), ("P", sp)):
            m[f"a{d}StatFb"] = tr_pct(s["feedback_static"] / s["feedback"]) if s["feedback"] else "0{,}0"
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
    def en(x):
        return f"{100 * x:.1f}"
    m.update({"aVoEn": en(v0["all"]["pass@1"]), "aYEn": en(Y["all"]["v3"]), "aPEn": en(P["all"]["v3"]),
              "aYPrecEn": en(jd["Y"]["accepted_precision"]), "aPPrecEn": en(jd["P"]["accepted_precision"]),
              "aFiltTwoEn": en(rep["V0"]["all"]["filt@2"]),
              "aYGainEn": en(Y["all"]["delta_vs_v0"]), "aPGainEn": en(P["all"]["delta_vs_v0"]),
              "aPWEn": "$<$~0.001" if P["all"]["wilcoxon_p"] < 0.001 else f"$=$~{P['all']['wilcoxon_p']:.3f}",
              "aYWEn": "$<$~0.001" if Y["all"]["wilcoxon_p"] < 0.001 else f"$=$~{Y['all']['wilcoxon_p']:.3f}"})
    os.makedirs(os.path.join(out_dir, "veri"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "cizelgeler"), exist_ok=True)
    with open(os.path.join(out_dir, "veri", "ana_sonuclar.tex"), "w") as f:
        f.write("% Auto-generated by benchmark/openrouter_results.py --latex; do not edit by hand.\n")
        for k, v in m.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    with open(os.path.join(out_dir, "cizelgeler", "cizelge-ana.tex"), "w") as f:
        f.write(main_table(rep))
    with open(os.path.join(out_dir, "cizelgeler", "cizelge-surec.tex"), "w") as f:
        f.write(process_table(rep))
    if extras:
        for gen, judge, tag, er in extras:
            m.update(extra_macros(er, tag))
        with open(os.path.join(out_dir, "veri", "ana_sonuclar.tex"), "a") as f:
            for k, v in m.items():
                if k.startswith("e"):
                    f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
        with open(os.path.join(out_dir, "cizelgeler", "cizelge-yargic.tex"), "w") as f:
            f.write(judge_table(rep, extras))
    os.makedirs(os.path.join(out_dir, "sekiller"), exist_ok=True)
    if "curve" in Y and "curve" in P:
        same = [(judge, tag, er) for _, judge, tag, er in extras
                if abs(er["V0"]["all"]["pass@1"] - v0["all"]["pass@1"]) < 1e-9 and "curve" in er["Y"]]
        plot_budget(rep, os.path.join(out_dir, "sekiller", "sekil2-butce-egrisi.pdf"), same)


def extra_macros(er: dict, tag: str) -> dict:
    """\\e<tag>… macros of one further Y-design analysis (own baseline)."""
    a, jd, v0 = er["Y"]["all"], er["judge"]["Y"], er["V0"]["all"]
    mc = a["iter1_vs_final_mcnemar"]
    out = {"Vo": tr_pct(v0["pass@1"]), "Y": tr_pct(a["v3"]), "Gain": num(100 * a["delta_vs_v0"]),
           "GainLo": num(100 * a["delta_ci"][0]), "GainHi": num(100 * a["delta_ci"][1]),
           "W": pval_rel(a["wilcoxon_p"]), "B": mc["b"], "C": mc["c"], "Mp": pval_rel(mc["p"]),
           "Gen": num(a["mean_generations"], 2), "Prec": tr_pct(jd["accepted_precision"]),
           "Pub": jd["public_passing"], "Rej": jd["rejected"],
           "RejGood": jd["rejected_and_hidden_wrong"], "RejBad": jd["rejected_but_hidden_correct"],
           "Leak": jd["feedback_with_code"], "Feedback": jd["feedback_total"],
           "Cost": num(er["cost"]["Y"]["usd"], 2), "BaseCost": num(er["cost"]["baseline"]["usd"], 2),
           "FiltTwo": tr_pct(v0["filt@2"]), "FiltThree": tr_pct(v0["filt@3"]),
           "PassFive": tr_pct(v0["pass@5"]),
           "YEn": f"{100 * a['v3']:.1f}", "VoEn": f"{100 * v0['pass@1']:.1f}",
           "GainEn": f"{100 * a['delta_vs_v0']:.1f}",
           "Up": tr_pct(mc["b"] / er["n"]), "Down": tr_pct(mc["c"] / er["n"])}
    if "static" in er:
        s = er["static"]["Y"]
        out.update({"Stops": s["stops"],
                    "StopsOk": "?" if s["stops_hidden_correct"] is None else s["stops_hidden_correct"],
                    "StatRej": s["public_rejected_static"],
                    "StatFb": tr_pct(s["feedback_static"] / s["feedback"]) if s["feedback"] else "0{,}0"})
    for d in ("easy", "medium", "hard"):
        key = {"easy": "Kolay", "medium": "Orta", "hard": "Zor"}[d]
        out[f"Vo{key}"] = tr_pct(er["V0"][d]["pass@1"])
        out[f"Y{key}"] = tr_pct(er["Y"][d]["v3"])
    for name, c in er.get("vs_compare", {}).items():
        out.update({"VsOther": tr_pct(c["other"]), "VsB": c["b"], "VsC": c["c"], "VsP": pval_rel(c["p"])})
    return {f"e{tag}{k}": v for k, v in out.items()}


def judge_table(rep: dict, extras: list) -> str:
    """Generator x judge configurations under the judge's verdict (Y): single-sample
    TÖ, delivered code İ_Y, Wilcoxon p of the paired gain, fixed/broken problems
    against the run's own first attempt, acceptance precision, API cost."""
    def row(gen, judge, D, jd, v0, usd):
        a = D["all"]
        mc = a["iter1_vs_final_mcnemar"]
        return (f"{gen} & {judge} & {tr_pct(v0['pass@1'])} & \\textbf{{{tr_pct(a['v3'])}}} & "
                f"{'$<$0{,}001' if a['wilcoxon_p'] < 0.001 else num(a['wilcoxon_p'], 3)} & {mc['b']}/{mc['c']} & {tr_pct(jd['accepted_precision'])} & "
                f"{num(usd, 2)} \\\\ \\hline")
    L = [AUTOGEN, "\\begin{tabular}{|l|l|c|c|c|c|c|c|}", "\\hline",
         "\\textbf{Üretici} & \\textbf{Yargıç} & \\textbf{TÖ} & \\textbf{İ$_Y$} & \\textbf{p} & "
         "\\textbf{D/B} & \\textbf{İs.} & \\textbf{\\$} \\\\ \\hline",
         row("DeepSeek-V3", "Qwen2.5-72B", rep["Y"], rep["judge"]["Y"], rep["V0"]["all"], rep["cost"]["Y"]["usd"])]
    for gen, judge, tag, er in extras:
        L.append(row(gen, judge, er["Y"], er["judge"]["Y"], er["V0"]["all"], er["cost"]["Y"]["usd"]))
    L.append("\\end{tabular}")
    return "\n".join(L) + "\n"


AUTOGEN = "% Auto-generated by benchmark/openrouter_results.py --latex; do not edit by hand."


def main_table(rep: dict) -> str:
    """Difficulty x {TÖ, filt@2, İ_Y, İ_P}; the best value of each row in bold
    (ties too). Paper names (docs/analysis.md): TÖ = tek-örnekli, the single-sample
    baseline (V0); İ_Y^Q / İ_P = iyileştirme, the delivered code under the judge's
    verdict / the score threshold (V3), superscript = judge (Q: Qwen2.5-72B; extras
    use their --extra TAG, e.g. G: gpt-oss-120b); filt@2 is the
    budget-matched resampling baseline."""
    v0, Y, P = rep["V0"], rep["Y"], rep["P"]
    L = [AUTOGEN, "\\begin{tabular}{|l|c|c|c|c|c|}", "\\hline",
         "\\textbf{Zorluk} & \\textbf{n} & \\textbf{TÖ} & \\textbf{filt@2} & "
         "\\textbf{İ$_Y^{Q}$} & \\textbf{İ$_P$} \\\\ \\hline"]
    for name, k in [("Kolay", "easy"), ("Orta", "medium"), ("Zor", "hard"), ("\\textbf{Toplam}", "all")]:
        vals = [v0[k]["pass@1"], v0[k]["filt@2"], Y[k]["v3"], P[k]["v3"]]
        best = max(round(v, 3) for v in vals)
        cells = [f"\\textbf{{{tr_pct(v)}}}" if round(v, 3) == best else tr_pct(v) for v in vals]
        L.append(f"{name} & {v0[k]['n_problems']} & " + " & ".join(cells) + " \\\\ \\hline")
    L.append("\\end{tabular}")
    return "\n".join(L) + "\n"


def process_table(rep: dict) -> str:
    """Per-design process summary of the main runs (Y and P)."""
    Y, P, jd = rep["Y"]["all"], rep["P"]["all"], rep["judge"]

    def bc(d):
        m = d["iter1_vs_final_mcnemar"]
        return f"{m['b']}/{m['c']}"
    rows = [
        ("Ortalama üretim sayısı", num(Y["mean_generations"], 2), num(P["mean_generations"], 2)),
        ("İlk denemede biten (\\%)", tr_pct(Y["first_stop"]), tr_pct(P["first_stop"])),
        ("Düzeltilen / bozulan problem", bc(Y), bc(P)),
        ("Açık testleri geçen deneme", str(jd["Y"]["public_passing"]), str(jd["P"]["public_passing"])),
        ("Geçerli yargıç kararı (\\%)", tr_pct(jd["Y"]["verdict_from_judge"]), "--"),
        ("Kod içeren geri bildirim", f"{jd['Y']['feedback_with_code']}/{jd['Y']['feedback_total']}",
         f"{jd['P']['feedback_with_code']}/{jd['P']['feedback_total']}"),
        ("API maliyeti (ABD \\$)", num(rep["cost"]["Y"]["usd"], 2), num(rep["cost"]["P"]["usd"], 2)),
    ]
    L = [AUTOGEN, "\\begin{tabular}{|l|c|c|}", "\\hline",
         "\\textbf{Ölçüt} & \\textbf{Y} & \\textbf{P} \\\\ \\hline"]
    for r in rows:
        L.append(" & ".join(r) + " \\\\ \\hline")
    L.append("\\end{tabular}")
    return "\n".join(L) + "\n"


def plot_budget(rep: dict, path: str, extra: list = ()) -> None:
    """Accuracy against the generation budget k: the pipeline's delivered code
    after at most k iterations (Y, P) next to resampling the V0 samples with
    the same k (filt@k: first public-passing sample; pass@k: any correct
    sample, an upper bound that needs the hidden tests)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        # Math labels (İ$_Y^{Q}$, filt@$k$) in the same Times New Roman as the paper text.
        "mathtext.fontset": "custom", "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic", "mathtext.bf": "Times New Roman:bold",
        "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42,
    })
    v0 = rep["V0"]["all"]
    ks = [k for k in range(1, 6) if f"filt@{k}" in v0]
    lines = [
        ("İ$_P$ (puan eşiği)", [rep["P"]["curve"]["all"][min(k, len(rep["P"]["curve"]["all"])) - 1] for k in ks],
         "#1f4e9a", "-", "o", 1.8),
        ("İ$_Y^{Q}$ (Qwen2.5-72B yargıç)", [rep["Y"]["curve"]["all"][min(k, len(rep["Y"]["curve"]["all"])) - 1] for k in ks],
         "#d9822b", "-", "s", 1.8),
    ]
    for (judge, tag, er), color in zip(extra, ("#2a8a4a", "#8a2a6a")):
        cv = er["Y"]["curve"]["all"]
        lines.append((f"İ$_Y^{{{tag}}}$ ({judge} yargıç)", [cv[min(k, len(cv)) - 1] for k in ks],
                      color, "-", "D", 1.8))
    lines += [
        ("filt@k (yeniden örnekleme)", [v0[f"filt@{k}"] for k in ks], "0.35", "--", "^", 1.1),
        ("pass@k (gizli testle seçim)", [v0[f"pass@{k}"] for k in ks], "0.6", ":", "v", 1.1),
    ]
    fig, ax = plt.subplots(figsize=(3.3, 2.15))
    # baselines drawn above the pipeline curves so overlapping points stay visible
    for label, ys, color, ls, mk, lw in lines:
        z = 3 if ls != "-" else 2
        ax.plot(ks, [100 * y for y in ys], color=color, linestyle=ls, marker=mk,
                markersize=3.6, linewidth=lw, label=label, zorder=z)
    ax.axhline(100 * v0["pass@1"], color="0.75", linewidth=0.8)
    ax.text(ks[-1], 100 * v0["pass@1"] - 0.25, "tek-örnekli (TÖ)", ha="right", va="top", fontsize=6.5,
            color="0.45")
    ax.set_xlabel("Problem başına üretim bütçesi (deneme sayısı)")
    ax.set_ylabel("pass@1 (%)")
    ax.set_xticks(ks)
    ax.grid(axis="y", linewidth=0.3, alpha=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if extra:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2, frameon=False,
                  handlelength=2.2, columnspacing=1.0, fontsize=6.5)
    else:
        ax.legend(loc="lower right", frameon=False, handlelength=2.4, borderaxespad=0.2)
    lo = min(min(100 * y for y in ys) for _, ys, *_ in lines)
    ax.set_ylim(lo - 4.5, None)
    fig.tight_layout(pad=0.3)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


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
    if "per_sample" in v0["all"]:
        L.append(f"\n**V0 tek örnekler:** pass@1 = {', '.join(pct(x) for x in v0['all']['per_sample'])}")
    for d, s in rep.get("static", {}).items():
        L.append(f"\n**Düğüm 1 ({d}):** {s['attempts']} deneme, önem {s['severity']}; "
                 f"durdurma {s['stops']} {s['stops_by_reason']} {s['stops_by_difficulty']}, "
                 f"gizli testte doğru olan durdurulmuş kod {s['stops_hidden_correct']}; "
                 f"açık testleri geçen {s['public_passing']} denemeden rubrik eşiğinin altında "
                 f"{s['public_below_threshold']}; bunların reddi {s['public_rejected']}, gerekçesi statik "
                 f"ölçüt adlandıran {s['public_rejected_static']} (gizli testte doğru "
                 f"{s['public_rejected_static_hidden_correct']}); statik ölçüt adlandıran geri bildirim "
                 f"{s['feedback_static']}/{s['feedback']}.")
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
    ap.add_argument("--extra", nargs=4, action="append", default=[],
                    metavar=("GENERATOR", "JUDGE", "TAG", "RESULTS_JSON"),
                    help="with --latex: a further Y-design analysis for the generator x judge "
                    "table (cizelge-yargic.tex) and \\e<TAG>… macros; TAG letters only")
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
           "budget": {}, "judge": {}, "static": {}}
    analysed = {}
    for name, probs in designs.items():
        probs = {q: probs[q] for q in qids}
        rep["static"][name] = static_evidence(probs, stopped_hidden(meta, settings.hidden_timeout_sec))
        res = analyse(probs, meta)
        analysed[name] = res
        rep[name] = design_metrics(res, base, args.n)
        rep[name]["curve"] = res["curve"]
        rep[name]["all"]["first_stop"] = sum(r["n_iter"] == 1 for r in res["rows"]) / len(res["rows"])
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
        extras = [(g, j, t, json.load(open(p))) for g, j, t, p in args.extra]
        write_latex(rep, args.latex, extras)
    print(open(os.path.join(args.out, "report.md")).read())


if __name__ == "__main__":
    main()
