"""
Builds every number, table and result figure of the ELECO 2026 paper
directly from the iteration JSONs of one (possibly sharded) pipeline run.

V0 is the first iteration of the same run: identical system/user prompt and
token budget, no judge feedback yet. V3 is the code the pipeline finally
delivers (last iteration). Pairing both on the same sample is what makes the
exact McNemar test valid.

"Strict" success = the code passes every public + private LCB test when
re-executed offline here. The loop itself only ever saw the public examples
(nodes/sandbox.py), so this is a held-out measurement. "Rubric" success =
the pipeline's own PASS decision (score >= threshold, public accuracy > 0).

Usage:
    python -m benchmark.paper_results --run-ids run_A run_B run_C \
        --out ~/Desktop/eleco2026-bildiri
"""
import argparse
import concurrent.futures as cf
import glob
import json
import os
import sys
from collections import Counter, defaultdict

from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.proportion import proportion_confint

DIFFS = [("easy", "Kolay"), ("medium", "Orta"), ("hard", "Zor")]
TERMINAL = {"PASS", "STOP_MAX", "STOP_HIGH"}
MAX_ITER = 5


def load_problems(run_ids: set, results_root: str) -> dict:
    by_q = defaultdict(list)
    for path in glob.glob(os.path.join(results_root, "*", "iterations", "*", "*.json")):
        with open(path) as f:
            d = json.load(f)
        if d.get("run_id") in run_ids:
            by_q[d["question_id"]].append(d)
    problems = {}
    for qid, iters in by_q.items():
        iters.sort(key=lambda x: x["iter"])
        if iters[-1]["status"] in TERMINAL:
            problems[qid] = iters
    return problems


def lcb_meta() -> dict:
    from datasets import load_dataset
    ds = load_dataset("livecodebench/code_generation_lite", split="test",
                      trust_remote_code=True)
    return {q: (p, s) for q, p, s in zip(ds["question_id"], ds["platform"], ds["starter_code"])}


def hidden_evaluate(problems: dict, meta: dict, cache_path: str) -> None:
    """Re-executes every iteration's code on all public + private tests and
    stores the pass rate on the iteration dict under "_hidden"."""
    sys.path.insert(0, os.getcwd())
    from sandbox.lcb_eval import _fallback_evaluate

    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)

    def key(it):
        return f"{it['run_id']}/{it['question_id']}/{it['iter']}"

    todo = []
    for qid, iters in problems.items():
        for it in iters:
            if it["status"] == "STOP_HIGH":
                it["_hidden"] = 0.0
            elif key(it) in cache:
                it["_hidden"] = cache[key(it)]
            else:
                todo.append(it)

    def run(it):
        code = it["training"].get("generated_code") or it["training"].get("stripped_code", "")
        row = {"all_tests": it["training"]["all_tests"],
               "starter_code": meta.get(it["question_id"], ("", ""))[1]}
        if not code.strip() or not row["all_tests"].get("input"):
            return it, 0.0
        return it, _fallback_evaluate(code, row, timeout=10)["pass_rate"]

    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        for it, pr in ex.map(run, todo):
            it["_hidden"] = pr
            cache[key(it)] = pr
    with open(cache_path, "w") as f:
        json.dump(cache, f)


def strict(it: dict) -> bool:
    return it["_hidden"] == 1.0


def pass_rate(it: dict) -> float:
    return it["_hidden"]


def wilson(k: int, n: int):
    if n == 0:
        return (0.0, 0.0)
    return proportion_confint(k, n, alpha=0.05, method="wilson")


def mcnemar_p(pairs: list) -> tuple:
    b = sum(1 for v0, v3 in pairs if not v0 and v3)
    c = sum(1 for v0, v3 in pairs if v0 and not v3)
    if b + c == 0:
        return b, c, 1.0
    a = sum(1 for v0, v3 in pairs if v0 and v3)
    d = sum(1 for v0, v3 in pairs if not v0 and not v3)
    return b, c, float(mcnemar([[a, c], [b, d]], exact=True).pvalue)


# Turkish number formatting (decimal comma)

def pct(x: float, nd: int = 1) -> str:
    return f"{100 * x:.{nd}f}".replace(".", "{,}")


def num(x: float, nd: int = 1) -> str:
    if abs(x) < 0.5 * 10 ** -nd:
        x = 0.0  # avoid "-0,0"
    return f"{x:.{nd}f}".replace(".", "{,}").replace("-", "$-$")


def pval(p: float) -> str:
    return "$<$0{,}001" if p < 0.001 else num(p, 3)


def pval_rel(p: float) -> str:
    """For running text: "p \\rP" reads "p = 0,012" or "p < 0,001"."""
    return "$<$~0{,}001" if p < 0.001 else f"$=$~{num(p, 3)}"


def analyse(problems: dict, meta: dict) -> dict:
    rows = []
    for qid, iters in problems.items():
        first, last = iters[0], iters[-1]
        rows.append({
            "qid": qid,
            "diff": first["difficulty"],
            "platform": meta.get(qid, ("unknown", ""))[0],
            "n_iter": len(iters),
            "v0": strict(first), "v3": strict(last),
            "v0_rub": first["status"] == "PASS", "v3_rub": last["status"] == "PASS",
            "pr0": pass_rate(first), "pr3": pass_rate(last),
            "ever": any(strict(it) for it in iters),
            "regressed": any(strict(it) for it in iters) and not strict(last),
            "tok_first": first["tokens"].get("total", 0),
            "tok_all": sum(it["tokens"].get("total", 0) for it in iters),
            "iters": iters,
        })

    out = {"rows": rows}

    def group(sel):
        n = len(sel)
        k0 = sum(r["v0"] for r in sel)
        k3 = sum(r["v3"] for r in sel)
        b, c, p = mcnemar_p([(r["v0"], r["v3"]) for r in sel])
        return {
            "n": n, "k0": k0, "k3": k3,
            "p0": k0 / n if n else 0.0, "p3": k3 / n if n else 0.0,
            "ci0": wilson(k0, n), "ci3": wilson(k3, n),
            "b": b, "c": c, "p": p,
            "r0": sum(r["v0_rub"] for r in sel) / n if n else 0.0,
            "r3": sum(r["v3_rub"] for r in sel) / n if n else 0.0,
            "pr0": sum(r["pr0"] for r in sel) / n if n else 0.0,
            "pr3": sum(r["pr3"] for r in sel) / n if n else 0.0,
        }

    out["all"] = group(rows)
    out["by_diff"] = {d: group([r for r in rows if r["diff"] == d]) for d, _ in DIFFS}
    out["by_platform"] = {p: group([r for r in rows if r["platform"] == p])
                          for p in ("atcoder", "leetcode")}

    # success@k: state of the delivered code after a budget of k iterations
    kmax = max(r["n_iter"] for r in rows)
    curve = {}
    groups = [("all", rows)] + [(d, [r for r in rows if r["diff"] == d]) for d, _ in DIFFS]
    for key, sel in groups:
        pts = []
        for k in range(1, kmax + 1):
            ok = sum(strict(r["iters"][min(k, r["n_iter"]) - 1]) for r in sel)
            pts.append(ok / len(sel) if sel else 0.0)
        curve[key] = pts
    out["curve"] = curve
    out["kmax"] = kmax

    all_iters = [it for r in rows for it in r["iters"]]
    sev = Counter(it["node1"]["severity"] for it in all_iters)
    syntax = sum(any("failed to parse" in s for s in it["node1"].get("issues", []))
                 for it in all_iters)
    judged = [it for it in all_iters if it["status"] != "STOP_HIGH"]
    judge_err = sum("judgement_llm_error" in (it["node3"].get("failure_reasons") or [])
                    for it in judged)
    json_err = sum("json_parse_error" in (it["node3"].get("failure_reasons") or [])
                   for it in judged)
    truncated = sum(
        it["tokens"].get("codegen_completion", 0) >= 6144
        or it["tokens"].get("codegen_prompt", 0) + it["tokens"].get("codegen_completion", 0) >= 8190
        for it in all_iters)

    # rubric PASS (decided on public tests) vs held-out strict correctness
    executed = [it for it in all_iters if it["status"] != "STOP_HIGH"]
    tp = sum(it["status"] == "PASS" and strict(it) for it in executed)
    fp = sum(it["status"] == "PASS" and not strict(it) for it in executed)
    fn = sum(it["status"] != "PASS" and strict(it) for it in executed)

    out["proc"] = {
        "n_iter_total": len(all_iters),
        "mean_iter": len(all_iters) / len(rows),
        "first_iter_stop": sum(r["n_iter"] == 1 for r in rows) / len(rows),
        "extended": sum(r["n_iter"] > MAX_ITER for r in rows),
        "sev_ok": sev.get("OK", 0) / len(all_iters),
        "sev_low": sev.get("LOW", 0) / len(all_iters),
        "sev_high": sev.get("HIGH", 0) / len(all_iters),
        "syntax": syntax / len(all_iters),
        "judge_err": judge_err / len(judged) if judged else 0.0,
        "json_err": json_err / len(judged) if judged else 0.0,
        "truncated": truncated / len(all_iters),
        "rub_prec": tp / (tp + fp) if tp + fp else 0.0,
        "rub_rec": tp / (tp + fn) if tp + fn else 0.0,
        "rub_fp": fp, "rub_tp": tp, "rub_fn": fn,
        "regressed": sum(r["regressed"] for r in rows),
        "ever": sum(r["ever"] for r in rows) / len(rows),
        "tok_ratio": sum(r["tok_all"] for r in rows) / max(1, sum(r["tok_first"] for r in rows)),
        "tok_mean_all": sum(r["tok_all"] for r in rows) / len(rows),
        "tok_mean_first": sum(r["tok_first"] for r in rows) / len(rows),
    }
    out["diff_counts"] = Counter(r["diff"] for r in rows)
    out["plat_counts"] = Counter(r["platform"] for r in rows)
    return out


def write_tex(res: dict, path: str, table_path: str) -> None:
    a, pr = res["all"], res["proc"]
    m = {
        "rN": str(a["n"]),
        "rNkolay": str(res["diff_counts"]["easy"]),
        "rNorta": str(res["diff_counts"]["medium"]),
        "rNzor": str(res["diff_counts"]["hard"]),
        "rNatcoder": str(res["plat_counts"].get("atcoder", 0)),
        "rNleetcode": str(res["plat_counts"].get("leetcode", 0)),
        "rNcodeforces": str(res["plat_counts"].get("codeforces", 0)),
        "rVo": pct(a["p0"]), "rVt": pct(a["p3"]),
        "rKo": str(a["k0"]), "rKt": str(a["k3"]),
        "rGain": num(100 * (a["p3"] - a["p0"])),
        "rVoLo": pct(a["ci0"][0]), "rVoHi": pct(a["ci0"][1]),
        "rVtLo": pct(a["ci3"][0]), "rVtHi": pct(a["ci3"][1]),
        "rB": str(a["b"]), "rC": str(a["c"]), "rP": pval_rel(a["p"]),
        "rRubVo": pct(a["r0"]), "rRubVt": pct(a["r3"]),
        "rPrVo": pct(a["pr0"]), "rPrVt": pct(a["pr3"]),
        "rMeanIter": num(pr["mean_iter"], 2),
        "rFirstStop": pct(pr["first_iter_stop"]),
        "rExtended": str(pr["extended"]),
        "rIterTotal": str(pr["n_iter_total"]),
        "rSevOk": pct(pr["sev_ok"]), "rSevLow": pct(pr["sev_low"]), "rSevHigh": pct(pr["sev_high"]),
        "rSyntax": pct(pr["syntax"]),
        "rJudgeErr": pct(pr["judge_err"]), "rJsonErr": pct(pr["json_err"]),
        "rTrunc": pct(pr["truncated"]),
        "rRubPrec": pct(pr["rub_prec"]), "rRubRec": pct(pr["rub_rec"]),
        "rRubFp": str(pr["rub_fp"]), "rRubAcc": str(pr["rub_tp"] + pr["rub_fp"]), "rRubTp": str(pr["rub_tp"]), "rRubFn": str(pr["rub_fn"]),
        "rRegressed": str(pr["regressed"]),
        "rEver": pct(pr["ever"]),
        "rTokRatio": num(pr["tok_ratio"], 1),
        "rTokFirst": f"{pr['tok_mean_first']:,.0f}".replace(",", "."),
        "rTokAll": f"{pr['tok_mean_all']:,.0f}".replace(",", "."),
        "rKmax": str(res["kmax"]),
    }
    for d, tr in DIFFS:
        g = res["by_diff"][d]
        m[f"rVo{tr}"] = pct(g["p0"])
        m[f"rVt{tr}"] = pct(g["p3"])
        m[f"rGain{tr}"] = num(100 * (g["p3"] - g["p0"]))
        m[f"rP{tr}"] = pval_rel(g["p"])
    for p, tr in (("atcoder", "Atcoder"), ("leetcode", "Leetcode")):
        g = res["by_platform"][p]
        m[f"rVo{tr}"] = pct(g["p0"])
        m[f"rVt{tr}"] = pct(g["p3"])
        m[f"rP{tr}"] = pval_rel(g["p"])

    lines = ["% Auto-generated by benchmark/paper_results.py; do not edit by hand."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in m.items()]
    # English-abstract variants: decimal point instead of the Turkish comma.
    lines += [f"\\newcommand{{\\{k}En}}{{{v.replace('{,}', '.')}}}"
              for k, v in m.items() if k in ("rVo", "rVt", "rRubPrec", "rP")]

    body = []
    for d, tr in DIFFS + [("all", "Toplam")]:
        g = res["all"] if d == "all" else res["by_diff"][d]
        name = f"\\textbf{{{tr}}}" if d == "all" else tr
        diff = 100 * (g["p3"] - g["p0"])
        sign = "$+$" if diff >= 0 else "$-$"
        body.append(
            f"{name} & {g['n']} & {pct(g['p0'])} & {pct(g['p3'])} & "
            f"{sign}{num(abs(diff))} & {g['b']}/{g['c']} & {pval(g['p'])} \\\\ \\hline")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")

    # Tabular body of the results table; caption/label live in the section file.
    table = [
        "% Auto-generated by benchmark/paper_results.py; do not edit by hand.",
        "\\begin{tabular}{|l|c|c|c|c|c|c|}",
        "\\hline",
        "\\textbf{Zorluk} & \\textbf{n} & \\textbf{V0} & \\textbf{V3} & \\textbf{Fark} & \\textbf{b/c} & \\textbf{p} \\\\ \\hline",
        *body,
        "\\end{tabular}",
    ]
    with open(table_path, "w") as f:
        f.write("\n".join(table) + "\n")


def plot_curve(res: dict, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "pdf.fonttype": 42,
    })
    ks = list(range(1, res["kmax"] + 1))
    fig, ax = plt.subplots(figsize=(3.25, 2.05))
    styles = [("all", "Tümü", "k", "-", "o", 1.6),
              ("easy", "Kolay", "0.35", "--", "s", 1.0),
              ("medium", "Orta", "0.35", "-.", "^", 1.0),
              ("hard", "Zor", "0.35", ":", "v", 1.0)]
    for key, label, color, ls, mk, lw in styles:
        ys = [100 * y for y in res["curve"][key]]
        n = res["all"]["n"] if key == "all" else res["by_diff"][key]["n"]
        ax.plot(ks, ys, color=color, linestyle=ls, marker=mk, markersize=3.2,
                linewidth=lw, label=f"{label} (n={n})")
    ax.axvline(MAX_ITER, color="0.6", linewidth=0.6, linestyle=":")
    ax.set_xlabel("Yineleme bütçesi (k)")
    ax.set_ylabel("Katı doğruluk (%)")
    ax.set_xticks(ks)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="-", linewidth=0.3, alpha=0.5)
    ax.legend(loc="upper left", ncol=2, frameon=False, handlelength=2.2,
              columnspacing=0.8, borderaxespad=0.2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(pad=0.3)
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-ids", nargs="+", required=True)
    ap.add_argument("--results-root", default="outputs/results")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    problems = load_problems(set(args.run_ids), args.results_root)
    if not problems:
        raise SystemExit("No completed problems found for the given run ids.")
    meta = lcb_meta()
    out = os.path.expanduser(args.out)
    hidden_evaluate(problems, meta, os.path.join(args.results_root, "hidden_eval_cache.json"))
    res = analyse(problems, meta)

    # Overleaf project layout (see docs/analysis.md §9).
    for sub in ("veri", "cizelgeler", "sekiller"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    write_tex(res, os.path.join(out, "veri", "sonuclar.tex"),
              os.path.join(out, "cizelgeler", "cizelge2-sonuclar.tex"))
    plot_curve(res, os.path.join(out, "sekiller", "sekil2-yineleme-butcesi.pdf"))

    a, pr = res["all"], res["proc"]
    print(f"problems={a['n']}  diff={dict(res['diff_counts'])}  plat={dict(res['plat_counts'])}")
    print(f"strict V0={a['p0']:.3f} {a['ci0']}  V3={a['p3']:.3f} {a['ci3']}  "
          f"b/c={a['b']}/{a['c']} p={a['p']:.4g}")
    for d, _ in DIFFS:
        g = res["by_diff"][d]
        print(f"  {d:6s} n={g['n']:3d} V0={g['p0']:.3f} V3={g['p3']:.3f} b/c={g['b']}/{g['c']} p={g['p']:.4g}")
    for p in ("atcoder", "leetcode"):
        g = res["by_platform"][p]
        print(f"  {p:8s} n={g['n']:3d} V0={g['p0']:.3f} V3={g['p3']:.3f} b/c={g['b']}/{g['c']} p={g['p']:.4g}")
    print(f"rubric PASS V0={a['r0']:.3f} V3={a['r3']:.3f} | mean pass_rate V0={a['pr0']:.3f} V3={a['pr3']:.3f}")
    print(f"curve(all)={[round(x, 3) for x in res['curve']['all']]}")
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in pr.items()}))


if __name__ == "__main__":
    main()
