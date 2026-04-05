"""
Results analysis: TSR computation, McNemar test, ablation comparison,
hypothesis validation.

Usage:
    python benchmark/analysis.py --pipeline outputs/results/benchmark_V4_*.json
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

def compute_tsr(results: list) -> dict:
    """
    Computes Task Success Rate by difficulty level.

    Args:
        results: list from benchmark JSON's results field

    Returns:
        {"easy": 0.72, "medium": 0.45, "hard": 0.20, "overall": 0.55}
    """
    counts = {"easy": [0, 0], "medium": [0, 0], "hard": [0, 0]}

    for r in results:
        diff = r.get("difficulty", "medium")
        # Pipeline variant: did at least one model get PASS?
        if "results" in r:
            passed = any(m["status"] == "PASS" for m in r["results"])
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

def mcnemar_test(pipeline_results: list, baseline_results: list) -> dict:
    """
    McNemar test for pipeline vs baseline, paired binary outcomes.

    Args:
        pipeline_results: V4 benchmark JSON results list
        baseline_results: V0 baseline JSON results list

    Returns:
        {"statistic", "p_value", "b", "c", "significant", "interpretation"}
    """
    # Match by question_id
    pipeline_map = {r["problem_id"]: r for r in pipeline_results}
    baseline_map = {r["question_id"]: r for r in baseline_results}

    common_ids = set(pipeline_map) & set(baseline_map)

    a = b = c = d = 0   # contingency table cells

    for qid in common_ids:
        p_pass = any(m["status"] == "PASS"
                     for m in pipeline_map[qid]["results"])
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
    Computes TSR difference between V0–V4.

    Args:
        variant_results: {"V0": [...], "V1": [...], ..., "V4": [...]}

    Returns:
        Per-variant TSR + delta vs V0
    """
    v0_tsr = compute_tsr(variant_results.get("V0", []))["overall"]
    comparison = {}

    for variant, results in variant_results.items():
        tsr = compute_tsr(results)
        comparison[variant] = {
            "tsr_overall": tsr["overall"],
            "tsr_easy":    tsr.get("easy", 0.0),
            "tsr_medium":  tsr.get("medium", 0.0),
            "tsr_hard":    tsr.get("hard", 0.0),
            "delta_vs_v0": round(tsr["overall"] - v0_tsr, 3),
        }

    return comparison


# ── Result tables ─────────────────────────────────────────────────────────────

def print_ablation_table(comparison: dict) -> None:
    """Prints ablation results as a markdown table."""
    print("| Variant | Description | Easy TSR | Med TSR | Hard TSR | Overall | ΔV0 |")
    print("|---------|-------------|----------|---------|----------|---------|-----|")

    descriptions = {
        "V0": "Single-shot baseline",
        "V1": "Execution + Judgment",
        "V2": "Static Analysis + Judgment",
        "V3": "Full pipeline",
    }

    for v in ["V0", "V1", "V2", "V3"]:
        if v not in comparison:
            continue
        r   = comparison[v]
        dsc = descriptions.get(v, "")
        delta = f"+{r['delta_vs_v0']:.3f}" if r["delta_vs_v0"] >= 0 \
                else f"{r['delta_vs_v0']:.3f}"
        print(
            f"| {v} | {dsc} "
            f"| {r['tsr_easy']:.3f} "
            f"| {r['tsr_medium']:.3f} "
            f"| {r['tsr_hard']:.3f} "
            f"| {r['tsr_overall']:.3f} "
            f"| {delta} |"
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
    v0_easy = comparison.get("V0", {}).get("tsr_easy",    0)
    v0_hard = comparison.get("V0", {}).get("tsr_hard",    0)
    v0_tsr  = comparison.get("V0", {}).get("tsr_overall", 0)
    v1_tsr  = comparison.get("V1", {}).get("tsr_overall", 0)
    v2_tsr  = comparison.get("V2", {}).get("tsr_overall", 0)
    v3_easy = comparison.get("V3", {}).get("tsr_easy",    0)
    v3_hard = comparison.get("V3", {}).get("tsr_hard",    0)
    v3_tsr  = comparison.get("V3", {}).get("tsr_overall", 0)

    results = {}

    # H1: V3 easy TSR ≥ V0 easy TSR + 0.15 (full pipeline improves baseline)
    h1_delta = v3_easy - v0_easy
    results["H1"] = {
        "hypothesis": "V3 easy TSR ≥ V0 + 15 points",
        "result":     h1_delta,
        "supported":  h1_delta >= 0.15,
        "note":       f"Actual delta: {h1_delta:+.3f}",
    }

    # H2: V1 TSR > V0 TSR (execution + judgment improves baseline)
    h2_delta = v1_tsr - v0_tsr
    results["H2"] = {
        "hypothesis": "V1 TSR > V0 TSR (Execution+Judgment improves baseline)",
        "result":     h2_delta,
        "supported":  h2_delta > 0,
        "note":       f"V1-V0 delta: {h2_delta:+.3f}",
    }

    # H3: V2 TSR > V0 TSR (static analysis + judgment improves baseline)
    h3_delta = v2_tsr - v0_tsr
    results["H3"] = {
        "hypothesis": "V2 TSR > V0 TSR (Static+Judgment improves baseline)",
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

    # TSR
    p_tsr = compute_tsr(pipeline_results)
    b_tsr = compute_tsr(baseline_results)
    print(f"Pipeline TSR: {p_tsr}")
    print(f"Baseline TSR: {b_tsr}")

    # McNemar
    mc = mcnemar_test(pipeline_results, baseline_results)
    print(f"\nMcNemar Test:")
    print(f"  p-value: {mc['p_value']}")
    print(f"  {mc['interpretation']}")
