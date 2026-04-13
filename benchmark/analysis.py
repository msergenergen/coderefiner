"""
Results analysis: TSR computation, McNemar test, ablation comparison,
hypothesis validation.

Two TSR metrics are reported for pipeline variants (V1–V3/V5):
  TSR_quality     — node3_score ≥ threshold AND node3_pass (LLM-judge criterion)
  TSR_correctness — final_pass_rate == 1.0 (lcb execution criterion, comparable to V0)

V0 baseline uses only TSR_correctness (lcb pass_rate == 1.0).
Primary McNemar test and H1–H4 comparisons use TSR_correctness.

Usage:
    python benchmark/analysis.py --pipeline outputs/results/benchmark_V3_*.json
                                  --baseline  outputs/results/baseline_v0_M1_*.json
"""
import argparse
import json
import os
from pathlib import Path

from scipy.stats import chi2_contingency
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.proportion import proportion_confint


# ── TSR computation ───────────────────────────────────────────────────────────

def _pipeline_quality_pass(problem_result: dict) -> bool:
    """Pipeline PASS via LLM-judge criterion (node3_score ≥ threshold AND node3_pass)."""
    return any(m["status"] == "PASS" for m in problem_result["results"])


def _pipeline_correctness_pass(problem_result: dict) -> bool:
    """Pipeline PASS via lcb execution criterion (final_pass_rate == 1.0)."""
    return any(m.get("final_pass_rate", 0.0) == 1.0 for m in problem_result["results"])


def compute_tsr(results: list) -> dict:
    """
    TSR_quality: pipeline PASS based on LLM-judge score (node3_pass).
    For pipeline variants (results have a 'results' key per model).
    For baseline V0, falls back to r['passed'] (lcb correctness).

    Returns:
        {"easy": 0.72, "medium": 0.45, "hard": 0.20, "overall": 0.55}
    """
    counts = {"easy": [0, 0], "medium": [0, 0], "hard": [0, 0]}

    for r in results:
        diff = r.get("difficulty", "medium")
        # Pipeline variant: did at least one model get PASS (quality)?
        if "results" in r:
            passed = _pipeline_quality_pass(r)
        # Baseline variant: single model
        else:
            passed = r.get("passed", False)

        if diff not in counts:
            counts[diff] = [0, 0]
        counts[diff][1] += 1
        if passed:
            counts[diff][0] += 1

    tsr = {}
    total_pass  = sum(v[0] for v in counts.values())
    total_count = sum(v[1] for v in counts.values())

    for diff, (p, t) in counts.items():
        tsr[diff] = round(p / t, 3) if t > 0 else 0.0

    tsr["overall"] = round(total_pass / total_count, 3) if total_count > 0 else 0.0
    return tsr


def compute_tsr_correctness(results: list) -> dict:
    """
    TSR_correctness: PASS when final_pass_rate == 1.0 (all lcb tests pass).
    Directly comparable to V0 baseline (which uses the same lcb criterion).
    For pipeline variants, at least one model must have final_pass_rate == 1.0.
    For baseline V0, uses r['passed'] (which is already pass_rate == 1.0).

    Returns:
        {"easy": 0.60, "medium": 0.35, "hard": 0.15, "overall": 0.42}
    """
    counts = {"easy": [0, 0], "medium": [0, 0], "hard": [0, 0]}

    for r in results:
        diff = r.get("difficulty", "medium")
        # Pipeline variant
        if "results" in r:
            passed = _pipeline_correctness_pass(r)
        # Baseline V0
        else:
            passed = r.get("passed", False)

        if diff not in counts:
            counts[diff] = [0, 0]
        counts[diff][1] += 1
        if passed:
            counts[diff][0] += 1

    tsr = {}
    total_pass  = sum(v[0] for v in counts.values())
    total_count = sum(v[1] for v in counts.values())

    for diff, (p, t) in counts.items():
        tsr[diff] = round(p / t, 3) if t > 0 else 0.0

    tsr["overall"] = round(total_pass / total_count, 3) if total_count > 0 else 0.0
    return tsr


def compute_tsr_with_ci(passed: int, total: int) -> dict:
    """
    Computes TSR and 95% Wilson confidence interval.
    """
    if total == 0:
        return {"tsr": 0.0, "ci_lo": 0.0, "ci_hi": 0.0}

    tsr = passed / total
    lo, hi = proportion_confint(passed, total, alpha=0.05, method="wilson")
    return {
        "tsr":   round(tsr, 3),
        "ci_lo": round(lo, 3),
        "ci_hi": round(hi, 3),
    }


# ── Statistical tests ─────────────────────────────────────────────────────────

def mcnemar_test(pipeline_results: list, baseline_results: list,
                 criterion: str = "correctness") -> dict:
    """
    McNemar test for pipeline vs baseline, paired binary outcomes.

    Args:
        pipeline_results: benchmark JSON results list (V1–V3/V5)
        baseline_results: V0 baseline JSON results list
        criterion: "correctness" (final_pass_rate==1.0, default, comparable to V0)
                   "quality"     (node3_pass, LLM-judge criterion)

    Returns:
        {"statistic", "p_value", "b", "c", "significant", "interpretation", "criterion"}
    """
    # Match by question_id
    pipeline_map = {r["problem_id"]: r for r in pipeline_results}
    baseline_map = {r["question_id"]: r for r in baseline_results}

    common_ids = set(pipeline_map) & set(baseline_map)

    a = b = c = d = 0   # contingency table cells

    for qid in common_ids:
        if criterion == "quality":
            p_pass = _pipeline_quality_pass(pipeline_map[qid])
        else:
            p_pass = _pipeline_correctness_pass(pipeline_map[qid])
        b_pass = baseline_map[qid].get("passed", False)

        if     p_pass and     b_pass: a += 1  # both pass
        if     p_pass and not b_pass: b += 1  # pipeline wins
        if not p_pass and     b_pass: c += 1  # baseline wins
        if not p_pass and not b_pass: d += 1  # both fail

    result = mcnemar([[a, b], [c, d]], exact=(b + c < 25))

    interpretation = (
        f"Pipeline better on {b} questions, worse on {c}. "
        f"{'Statistically significant (p<0.05).' if result.pvalue < 0.05 else 'Not statistically significant (p≥0.05).'}"
    )

    return {
        "criterion":      criterion,
        "statistic":      round(result.statistic, 4),
        "p_value":        round(result.pvalue, 4),
        "a_both_pass":    a,
        "b_pipeline_win": b,
        "c_baseline_win": c,
        "d_both_fail":    d,
        "n_compared":     len(common_ids),
        "significant":    result.pvalue < 0.05,
        "interpretation": interpretation,
    }


# ── Ablation comparison ───────────────────────────────────────────────────────

def compare_variants(variant_results: dict) -> dict:
    """
    Computes both TSR_quality and TSR_correctness for V0–V5.

    Args:
        variant_results: {"V0": [...], "V1": [...], ..., "V5": [...]}

    Returns:
        Per-variant dict with both TSR layers + delta vs V0 (correctness-based).
        V0 has no TSR_quality (no judge); its TSR_correctness == TSR_quality.
    """
    v0_correctness = compute_tsr_correctness(variant_results.get("V0", []))["overall"]
    comparison = {}

    for variant, results in variant_results.items():
        tsr_q = compute_tsr(results)             # quality (judge-based)
        tsr_c = compute_tsr_correctness(results)  # correctness (lcb-based)

        comparison[variant] = {
            # Quality (judge-based)
            "tsr_quality_overall": tsr_q["overall"],
            "tsr_quality_easy":    tsr_q.get("easy", 0.0),
            "tsr_quality_medium":  tsr_q.get("medium", 0.0),
            "tsr_quality_hard":    tsr_q.get("hard", 0.0),
            # Correctness (lcb-based, comparable to V0)
            "tsr_correctness_overall": tsr_c["overall"],
            "tsr_correctness_easy":    tsr_c.get("easy", 0.0),
            "tsr_correctness_medium":  tsr_c.get("medium", 0.0),
            "tsr_correctness_hard":    tsr_c.get("hard", 0.0),
            # Delta vs V0 (correctness-based — apples to apples)
            "delta_vs_v0": round(tsr_c["overall"] - v0_correctness, 3),
        }

    return comparison


# ── Result tables ─────────────────────────────────────────────────────────────

def print_ablation_table(comparison: dict) -> None:
    """
    Prints ablation results as two markdown tables:
      Table 1 — TSR_correctness (lcb-based, comparable to V0)
      Table 2 — TSR_quality (judge-based, pipeline's optimisation target)
    """
    descriptions = {
        "V0": "Single-shot baseline",
        "V1": "Execution + Judgement",
        "V2": "Static Analysis + Judgement",
        "V3": "Full pipeline",
        "V5": "Full pipeline + Trend Extension",
    }
    variants = [v for v in ["V0", "V1", "V2", "V3", "V5"] if v in comparison]

    print("\n### TSR_correctness (lcb execution — primary, comparable to V0)\n")
    print("| Variant | Description | Easy | Medium | Hard | Overall | ΔV0 |")
    print("|---------|-------------|------|--------|------|---------|-----|")
    for v in variants:
        r   = comparison[v]
        dsc = descriptions.get(v, "")
        delta = f"+{r['delta_vs_v0']:.3f}" if r["delta_vs_v0"] >= 0 \
                else f"{r['delta_vs_v0']:.3f}"
        print(
            f"| {v} | {dsc} "
            f"| {r['tsr_correctness_easy']:.3f} "
            f"| {r['tsr_correctness_medium']:.3f} "
            f"| {r['tsr_correctness_hard']:.3f} "
            f"| {r['tsr_correctness_overall']:.3f} "
            f"| {delta} |"
        )

    print("\n### TSR_quality (LLM-judge score — secondary, optimisation target)\n")
    print("| Variant | Description | Easy | Medium | Hard | Overall | Quality−Correctness |")
    print("|---------|-------------|------|--------|------|---------|---------------------|")
    for v in variants:
        r   = comparison[v]
        dsc = descriptions.get(v, "")
        gap = round(r["tsr_quality_overall"] - r["tsr_correctness_overall"], 3)
        gap_str = f"+{gap:.3f}" if gap >= 0 else f"{gap:.3f}"
        print(
            f"| {v} | {dsc} "
            f"| {r['tsr_quality_easy']:.3f} "
            f"| {r['tsr_quality_medium']:.3f} "
            f"| {r['tsr_quality_hard']:.3f} "
            f"| {r['tsr_quality_overall']:.3f} "
            f"| {gap_str} |"
        )


def print_model_table(results_by_model: dict) -> None:
    """Prints V0 vs V4 comparison per model."""
    print("| Model | V0 TSR | V4 TSR | Δ | V4 Avg iter | STOP_HIGH% |")
    print("|-------|--------|--------|---|-------------|------------|")

    for model, data in results_by_model.items():
        v0  = data.get("v0_tsr", 0.0)
        v4  = data.get("v4_tsr", 0.0)
        avg = data.get("avg_iter", 0.0)
        sh  = data.get("stop_high_rate", 0.0)
        delta = f"+{v4-v0:.3f}" if v4 >= v0 else f"{v4-v0:.3f}"
        print(f"| {model} | {v0:.3f} | {v4:.3f} | {delta} | {avg:.1f} | {sh:.1%} |")


# ── Hypothesis validation ─────────────────────────────────────────────────────

def evaluate_hypotheses(comparison: dict, mcnemar_result: dict,
                         model_data: dict) -> dict:
    """
    Evaluates H1–H4 hypotheses based on results.
    """
    # All hypothesis comparisons use TSR_correctness (apples-to-apples with V0)
    v0_easy = comparison.get("V0", {}).get("tsr_correctness_easy",    0)
    v0_hard = comparison.get("V0", {}).get("tsr_correctness_hard",    0)
    v0_tsr  = comparison.get("V0", {}).get("tsr_correctness_overall", 0)
    v1_tsr  = comparison.get("V1", {}).get("tsr_correctness_overall", 0)
    v2_tsr  = comparison.get("V2", {}).get("tsr_correctness_overall", 0)
    v3_easy = comparison.get("V3", {}).get("tsr_correctness_easy",    0)
    v3_hard = comparison.get("V3", {}).get("tsr_correctness_hard",    0)
    v3_tsr  = comparison.get("V3", {}).get("tsr_correctness_overall", 0)

    results = {}

    # H1: V3 easy TSR ≥ V0 easy TSR + 0.15 (full pipeline improves baseline)
    h1_delta = v3_easy - v0_easy
    results["H1"] = {
        "hypothesis": "V3 easy TSR ≥ V0 + 15 points",
        "result":     h1_delta,
        "supported":  h1_delta >= 0.15,
        "note":       f"Actual delta: {h1_delta:+.3f}",
    }

    # H2: V1 TSR > V0 TSR (execution + judgement improves baseline)
    h2_delta = v1_tsr - v0_tsr
    results["H2"] = {
        "hypothesis": "V1 TSR > V0 TSR (Execution+Judgement improves baseline)",
        "result":     h2_delta,
        "supported":  h2_delta > 0,
        "note":       f"V1-V0 delta: {h2_delta:+.3f}",
    }

    # H3: V2 TSR > V0 TSR (static analysis + judgement improves baseline)
    h3_delta = v2_tsr - v0_tsr
    results["H3"] = {
        "hypothesis": "V2 TSR > V0 TSR (Static+Judgement improves baseline)",
        "result":     h3_delta,
        "supported":  h3_delta > 0,
        "note":       f"V2-V0 delta: {h3_delta:+.3f}",
    }

    # H4: V3 TSR > V1 TSR and V3 TSR > V2 TSR (full pipeline beats ablations)
    h4_v1 = v3_tsr - v1_tsr
    h4_v2 = v3_tsr - v2_tsr
    results["H4"] = {
        "hypothesis": "V3 TSR > V1 and V3 TSR > V2 (full pipeline is best)",
        "result":     {"v3_vs_v1": h4_v1, "v3_vs_v2": h4_v2},
        "supported":  h4_v1 > 0 and h4_v2 > 0,
        "note":       f"V3-V1: {h4_v1:+.3f}, V3-V2: {h4_v2:+.3f}",
    }

    # H5: Easy gain < Hard gain (ceiling effect)
    easy_gain = v3_easy - v0_easy
    hard_gain = v3_hard - v0_hard
    results["H5"] = {
        "hypothesis": "Pipeline gain: easy < hard (ceiling effect)",
        "result":     {"easy_gain": easy_gain, "hard_gain": hard_gain},
        "supported":  easy_gain < hard_gain,
        "note":       f"Easy: {easy_gain:+.3f}, Hard: {hard_gain:+.3f}",
    }

    # H6: Pipeline gain lower in model with high STOP_HIGH rate
    sh_rates = {m: d.get("stop_high_rate", 0) for m, d in model_data.items()}
    gains    = {m: d.get("v3_tsr", 0) - d.get("v0_tsr", 0)
                for m, d in model_data.items()}
    if sh_rates and gains:
        max_sh_model   = max(sh_rates, key=sh_rates.get)
        min_gain_model = min(gains,   key=gains.get)
        results["H6"] = {
            "hypothesis": "High STOP_HIGH model has lower pipeline gain",
            "supported":  max_sh_model == min_gain_model,
            "note": (
                f"Highest STOP_HIGH: {max_sh_model} ({sh_rates[max_sh_model]:.1%}), "
                f"Lowest gain: {min_gain_model} ({gains[min_gain_model]:+.3f})"
            ),
        }

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def _load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM-as-a-Judge — Results Analysis")
    parser.add_argument("--pipeline", required=True, help="Pipeline benchmark JSON path")
    parser.add_argument("--baseline", required=True, help="Baseline V0 JSON path")
    args = parser.parse_args()

    pipeline_data = _load_json(args.pipeline)
    baseline_data = _load_json(args.baseline)

    pipeline_results = pipeline_data["results"]
    baseline_results = baseline_data["results"]

    # TSR — both layers
    p_tsr_quality     = compute_tsr(pipeline_results)
    p_tsr_correctness = compute_tsr_correctness(pipeline_results)
    b_tsr             = compute_tsr_correctness(baseline_results)

    print(f"Pipeline TSR_correctness (lcb): {p_tsr_correctness}")
    print(f"Pipeline TSR_quality (judge):   {p_tsr_quality}")
    print(f"Baseline  TSR_correctness (V0): {b_tsr}")

    gap = round(p_tsr_quality["overall"] - p_tsr_correctness["overall"], 3)
    print(f"\nQuality−Correctness gap (pipeline): {gap:+.3f}")
    print("  (positive gap = judge credits code that doesn't pass all lcb tests)")

    # McNemar — primary: correctness criterion
    print(f"\nMcNemar Test (correctness criterion — primary, comparable to V0):")
    mc_c = mcnemar_test(pipeline_results, baseline_results, criterion="correctness")
    print(f"  p-value: {mc_c['p_value']}")
    print(f"  {mc_c['interpretation']}")

    # McNemar — secondary: quality criterion
    print(f"\nMcNemar Test (quality criterion — secondary):")
    mc_q = mcnemar_test(pipeline_results, baseline_results, criterion="quality")
    print(f"  p-value: {mc_q['p_value']}")
    print(f"  {mc_q['interpretation']}")
