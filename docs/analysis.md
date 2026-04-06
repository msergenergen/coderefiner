# analysis.md — Results Analysis

## 1. Temel Metrikler

### Task Success Rate (TSR)

A question is considered successful when:
- **V0 baseline:** The model's single-shot output passes all tests
- **V1–V4 pipeline:** At least one model (M1, M2, or M3) received PASS

```python
# benchmark/analysis.py

import json
import os
from pathlib import Path
from scipy.stats import chi2_contingency
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.proportion import proportion_confint


def compute_tsr(results: list[dict]) -> dict:
    """
    Zorluk seviyesine göre Task Success Rate hesaplar.

    Args:
        results: benchmark JSON'ındaki results listesi

    Returns:
        {"easy": 0.72, "medium": 0.45, "hard": 0.20, "overall": 0.55}
    """
    counts = {"easy": [0, 0], "medium": [0, 0], "hard": [0, 0]}

    for r in results:
        diff = r["difficulty"]
        # Pipeline variant: did at least one model get PASS?
        if "results" in r:
            passed = any(m["status"] == "PASS" for m in r["results"])
        # Baseline variant: single model
        else:
            passed = r.get("passed", False)

        counts[diff][1] += 1        # toplam
        if passed:
            counts[diff][0] += 1    # başarılı

    tsr = {}
    total_pass  = sum(v[0] for v in counts.values())
    total_count = sum(v[1] for v in counts.values())

    for diff, (p, t) in counts.items():
        tsr[diff] = round(p / t, 3) if t > 0 else 0.0

    tsr["overall"] = round(total_pass / total_count, 3) if total_count > 0 else 0.0
    return tsr


def compute_tsr_with_ci(passed: int, total: int) -> dict:
    """
    TSR ve %95 Wilson güven aralığı hesaplar.
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
```

---

## 2. Statistical Tests

### McNemar Testi

For two-condition (pipeline vs baseline) paired binary outcomes.
Pair per question: (pipeline PASS?, baseline PASS?)

```python
def mcnemar_test(pipeline_results: list[dict],
                 baseline_results: list[dict]) -> dict:
    """
    Pipeline vs baseline için McNemar testi.

    Args:
        pipeline_results: V4 benchmark JSON results
        baseline_results: V0 baseline JSON results

    Returns:
        {"statistic": float, "p_value": float, "b": int, "c": int,
         "significant": bool, "interpretation": str}
    """
    # Match by question_id
    pipeline_map = {r["problem_id"]: r for r in pipeline_results}
    baseline_map = {r["question_id"]: r for r in baseline_results}

    common_ids = set(pipeline_map) & set(baseline_map)

    a = b = c = d = 0   # contingency table

    for qid in common_ids:
        p_pass = any(m["status"] == "PASS"
                     for m in pipeline_map[qid]["results"])
        b_pass = baseline_map[qid].get("passed", False)

        if     p_pass and     b_pass: a += 1  # her ikisi başarılı
        if     p_pass and not b_pass: b += 1  # pipeline kazandı
        if not p_pass and     b_pass: c += 1  # baseline kazandı
        if not p_pass and not b_pass: d += 1  # her ikisi başarısız

    result = mcnemar([[a, b], [c, d]], exact=(b + c < 25))

    interpretation = (
        f"Pipeline {b} soruda baseline'dan iyi, "
        f"{c} soruda kötü. "
        f"{'Fark istatistiksel olarak anlamlı (p<0.05).' if result.pvalue < 0.05 else 'Fark istatistiksel olarak anlamlı değil (p≥0.05).'}"
    )

    return {
        "statistic":     round(result.statistic, 4),
        "p_value":       round(result.pvalue, 4),
        "a_both_pass":   a,
        "b_pipeline_win": b,
        "c_baseline_win": c,
        "d_both_fail":   d,
        "n_compared":    len(common_ids),
        "significant":   result.pvalue < 0.05,
        "interpretation": interpretation,
    }
```

### Ablation Comparison

```python
def compare_variants(variant_results: dict[str, list[dict]]) -> dict:
    """
    V0–V4 arasındaki TSR farkını hesaplar.

    Args:
        variant_results: {"V0": [...], "V1": [...], ..., "V4": [...]}

    Returns:
        Her varyant için TSR + V0'a göre delta
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
```

---

## 3. Result Tables

### 3.1 Ana Ablasyon Tablosu

```python
def print_ablation_table(comparison: dict) -> None:
    """
    Ablasyon sonuçlarını markdown tablo olarak yazdırır.
    """
    print("| Varyant | Açıklama | Easy TSR | Med TSR | Hard TSR | Overall | ΔV0 |")
    print("|---------|----------|----------|---------|----------|---------|-----|")

    descriptions = {
        "V0": "Single-shot baseline",
        "V1": "Execution + Judgement",
        "V2": "Static Analysis + Judgement",
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
```

**Example output (placeholder — will be filled with cloud results):**

```
| Varyant | Açıklama                    | Easy TSR | Med TSR | Hard TSR | Overall | ΔV0    |
|---------|-----------------------------|----------|---------|----------|---------|--------|
| V0      | Single-shot baseline        | ?        | ?       | ?        | ?       | +0.000 |
| V1      | Execution + Judgement       | ?        | ?       | ?        | ?       | ?      |
| V2      | Static Analysis + Judgement | ?        | ?       | ?        | ?       | ?      |
| V3      | Full pipeline               | ?        | ?       | ?        | ?       | ?      |
```

### 3.2 Per-Model Table

```python
def print_model_table(results_by_model: dict) -> None:
    """
    Her model için V0 vs V3 karşılaştırması.
    """
    print("| Model | V0 TSR | V3 TSR | Δ | V3 Ort. iter | STOP_HIGH% |")
    print("|-------|--------|--------|---|-------------|------------|")

    for model, data in results_by_model.items():
        v0  = data.get("v0_tsr", 0.0)
        v4  = data.get("v4_tsr", 0.0)
        avg = data.get("avg_iter", 0.0)
        sh  = data.get("stop_high_rate", 0.0)
        delta = f"+{v4-v0:.3f}" if v4 >= v0 else f"{v4-v0:.3f}"
        print(f"| {model} | {v0:.3f} | {v4:.3f} | {delta} | {avg:.1f} | {sh:.1%} |")
```

---

## 4. Hypothesis Validation

```python
def evaluate_hypotheses(comparison: dict, mcnemar_result: dict,
                         model_data: dict) -> dict:
    """
    H1–H4 hipotezlerini sonuçlara göre değerlendirir.
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

    # H1: V3 easy TSR ≥ V0 easy TSR + 0.15
    h1_delta = v3_easy - v0_easy
    results["H1"] = {
        "hypothesis": "V3 easy TSR ≥ V0 + 15 puan",
        "result":     h1_delta,
        "supported":  h1_delta >= 0.15,
        "note":       f"Gerçekleşen delta: {h1_delta:+.3f}",
    }

    # H2: V1 TSR > V0 TSR (Execution + Judgement improves baseline)
    h2_delta = v1_tsr - v0_tsr
    results["H2"] = {
        "hypothesis": "V1 TSR > V0 TSR (Execution+Judgement > baseline)",
        "result":     h2_delta,
        "supported":  h2_delta > 0,
        "note":       f"V1-V0 delta: {h2_delta:+.3f}",
    }

    # H3: V3 > V1 and V3 > V2 (full pipeline is best)
    h3_v1 = v3_tsr - v1_tsr
    h3_v2 = v3_tsr - v2_tsr
    results["H3"] = {
        "hypothesis": "V3 TSR > V1 ve V3 TSR > V2 (tam pipeline en iyi)",
        "result":     {"v3_vs_v1": h3_v1, "v3_vs_v2": h3_v2},
        "supported":  h3_v1 > 0 and h3_v2 > 0,
        "note":       f"V3-V1: {h3_v1:+.3f}, V3-V2: {h3_v2:+.3f}",
    }

    # H4: Easy gain < Hard gain (ceiling effect)
    easy_gain = v3_easy - v0_easy
    hard_gain = v3_hard - v0_hard
    results["H4"] = {
        "hypothesis": "Pipeline katkısı: easy < hard (tavan etkisi)",
        "result":     {"easy_gain": easy_gain, "hard_gain": hard_gain},
        "supported":  easy_gain < hard_gain,
        "note":       f"Easy: {easy_gain:+.3f}, Hard: {hard_gain:+.3f}",
    }

    # H5: Pipeline gain lower in model with high STOP_HIGH rate
    sh_rates = {m: d.get("stop_high_rate", 0) for m, d in model_data.items()}
    gains    = {m: d.get("v3_tsr", 0) - d.get("v0_tsr", 0)
                for m, d in model_data.items()}
    if sh_rates and gains:
        max_sh_model  = max(sh_rates, key=sh_rates.get)
        min_gain_model = min(gains,   key=gains.get)
        results["H5"] = {
            "hypothesis": "STOP_HIGH yüksek modelde pipeline katkısı düşük",
            "supported":  max_sh_model == min_gain_model,
            "note": (
                f"En yüksek STOP_HIGH: {max_sh_model} ({sh_rates[max_sh_model]:.1%}), "
                f"En düşük gain: {min_gain_model} ({gains[min_gain_model]:+.3f})"
            ),
        }

    return results
```

---

## 5. benchmark/analysis.py — Full Run

```python
"""
Tüm benchmark sonuçlarını analiz eder.

Kullanım:
    python benchmark/analysis.py --results-dir outputs/results/
"""
import argparse
import glob
import json
import os
from loguru import logger


def load_results(results_dir: str) -> dict:
    """
    outputs/results/ altındaki tüm JSON dosyalarını yükler.
    Dosya adındaki tag'e göre gruplar.
    """
    files = glob.glob(os.path.join(results_dir, "*.json"))
    loaded = {}
    for f in files:
        name = os.path.basename(f).replace(".json", "")
        with open(f) as fp:
            data = json.load(fp)
        loaded[name] = data
    return loaded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="outputs/results/")
    args = parser.parse_args()

    all_files = load_results(args.results_dir)

    # Group results by variant
    variant_results = {}
    baseline_results = {}

    for name, data in all_files.items():
        variant = data.get("variant", "")
        if variant == "V0":
            model = data.get("model", "unknown")
            baseline_results[model] = data.get("results", [])
        elif variant in ("V1", "V2", "V3", "V4"):
            variant_results[variant] = data.get("results", [])

    # TSR hesapla
    comparison = compare_variants(variant_results)
    print("\n=== Ablasyon Tablosu ===")
    print_ablation_table(comparison)

    # Per-model analysis
    if baseline_results and "V3" in variant_results:
        print("\n=== Model Bazlı Karşılaştırma ===")
        # Match V0 vs V3 results per model
        # ...

    # McNemar testi (V0 M1 vs V3)
    if "M1" in baseline_results and "V3" in variant_results:
        mc = mcnemar_test(variant_results["V3"], baseline_results["M1"])
        print(f"\n=== McNemar (V3 vs V0-M1) ===")
        print(f"p-value: {mc['p_value']}")
        print(f"Anlamlı: {mc['significant']}")
        print(mc["interpretation"])

    # Hypothesis evaluation
    hyp = evaluate_hypotheses(comparison, {}, {})
    print("\n=== Hipotez Sonuçları ===")
    for h, r in hyp.items():
        status = "✓ DESTEKLENDİ" if r["supported"] else "✗ DESTEKLENMEDİ"
        print(f"{h}: {status} — {r['note']}")


if __name__ == "__main__":
    main()
```

---

## 6. Hata Analizi

### STOP_MAX Inspection

```python
from memory.logger import read_memory

rows = read_memory()
stop_max = [r for r in rows if r["status"] == "STOP_MAX"]

print(f"STOP_MAX sayısı: {len(stop_max)}")
print(f"Zorluk dağılımı:")
from collections import Counter
print(Counter(r["difficulty"] for r in stop_max))

# Questions that stalled with the lowest score
stop_max.sort(key=lambda r: r["score"])
for r in stop_max[:5]:
    print(f"  {r['q_id']} | {r['difficulty']} | score={r['score']} | iter={r['iter']}")
```

### improved_prompt Quality Check

Select 20 samples, inspect manually:

```python
import json, os, random

iter_dir = "outputs/iterations"
all_jsons = []
for root, _, files in os.walk(iter_dir):
    for f in files:
        if f.endswith(".json"):
            all_jsons.append(os.path.join(root, f))

sample = random.sample(all_jsons, min(20, len(all_jsons)))

for path in sample:
    data = json.load(open(path))
    prompt = data["node3"].get("improved_prompt", "")
    print(f"\n{'='*50}")
    print(f"Q: {data['question_id']} | Model: {data['model']} | Status: {data['status']}")
    print(f"Score: {data['node3']['total_score']}")
    print(f"Prompt (ilk 300 karakter):\n{prompt[:300]}")
    print()
    # Manual evaluation: 1-5 score (actionable? specific? self-contained?))
```

### Per-Node Contribution Analysis

```python
rows = read_memory()

# STOP_HIGH rate — Node 1's effect
total    = len(rows)
stop_hi  = sum(1 for r in rows if r["status"] == "STOP_HIGH")
print(f"Node 1 STOP_HIGH oranı: {stop_hi/total:.1%} ({stop_hi}/{total})")

# Average iteration count (PASS cases)
pass_rows = [r for r in rows if r["status"] == "PASS"]
if pass_rows:
    avg_iter = sum(r["iter"] for r in pass_rows) / len(pass_rows)
    print(f"Ortalama PASS iterasyonu: {avg_iter:.1f}")

# Score improvement (LOOP → PASS)
# Compare first and last scores per question-model pair
from itertools import groupby

rows_sorted = sorted(rows, key=lambda r: (r["q_id"], r["model"], r["iter"]))
for (q, m), group in groupby(rows_sorted, key=lambda r: (r["q_id"], r["model"])):
    grp = list(group)
    if len(grp) > 1:
        first_score = grp[0]["score"]
        last_score  = grp[-1]["score"]
        delta = last_score - first_score
        if delta != 0:
            print(f"{q} | {m} | Δscore: {delta:+.1f} ({first_score:.1f} → {last_score:.1f})")
```
