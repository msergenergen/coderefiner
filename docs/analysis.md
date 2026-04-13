# analysis.md — Results Analysis

## 1. TSR Tanımı — İki Katman

Pipeline varyantları (V1–V3/V5) ile V0 baseline arasında metodolojik olarak
doğru bir karşılaştırma yapabilmek için **iki ayrı TSR** raporlanır:

| Metrik | Kriter | Ne Zaman Kullanılır |
|--------|--------|---------------------|
| **TSR_correctness** | `final_pass_rate == 1.0` (lcb execution) | V0 ile birincil karşılaştırma; McNemar; H1–H4 |
| **TSR_quality** | `node3_pass == True` (LLM-judge skoru ≥ eşik) | Pipeline'ın optimizasyon hedefi; ikincil analiz |

**Neden iki katman?**
- V0, `lcb pass_rate == 1.0` (tüm testler geçer) ile PASS sayar.
- V1–V3, `node3_score ≥ eşik AND node3_pass` ile PASS sayar.
- Farklı kriterler doğrudan karşılaştırılamaz; TSR_correctness bu boşluğu kapatır.
- `Quality − Correctness gap` (pozitifse) judge'ın lcb'nin geçmediği koda puan
  verdiğini gösterir — tez için ayrı bir bulgu değeri taşır.

```
A question is PASS in TSR_correctness  ←  any model's final_pass_rate == 1.0
A question is PASS in TSR_quality      ←  any model's status == "PASS"
A question is PASS in V0               ←  lcb pass_rate == 1.0  (=TSR_correctness)
```

---

## 2. TSR Hesaplama

### 2.1 TSR_quality — `compute_tsr()`

```python
# benchmark/analysis.py

def _pipeline_quality_pass(problem_result: dict) -> bool:
    """Pipeline PASS via LLM-judge criterion (node3_score ≥ threshold AND node3_pass)."""
    return any(m["status"] == "PASS" for m in problem_result["results"])


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
        if "results" in r:
            passed = _pipeline_quality_pass(r)
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
```

### 2.2 TSR_correctness — `compute_tsr_correctness()`

```python
def _pipeline_correctness_pass(problem_result: dict) -> bool:
    """Pipeline PASS via lcb execution criterion (final_pass_rate == 1.0)."""
    return any(m.get("final_pass_rate", 0.0) == 1.0 for m in problem_result["results"])


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
        if "results" in r:
            passed = _pipeline_correctness_pass(r)
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
```

### 2.3 Wilson CI — `compute_tsr_with_ci()`

```python
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

## 3. Statistical Tests

### McNemar Testi

İki koşullu (pipeline vs baseline) eşleştirilmiş ikili sonuçlar için.
Her soru başına bir çift: (pipeline PASS?, baseline PASS?)

`criterion` parametresi ile hangi TSR katmanının kullanılacağı seçilir.
**Birincil test:** `criterion="correctness"` (V0 ile aynı kriter, apples-to-apples).
**İkincil test:** `criterion="quality"` (judge-bazlı).

```python
def mcnemar_test(pipeline_results: list, baseline_results: list,
                 criterion: str = "correctness") -> dict:
    """
    Pipeline vs baseline için McNemar testi.

    Args:
        pipeline_results: benchmark JSON results (V1–V3/V5)
        baseline_results: V0 baseline JSON results
        criterion: "correctness" (final_pass_rate==1.0, default)
                   "quality"     (node3_pass, LLM-judge)

    Returns:
        {"criterion", "statistic", "p_value", "a_both_pass", "b_pipeline_win",
         "c_baseline_win", "d_both_fail", "n_compared", "significant", "interpretation"}
    """
    pipeline_map = {r["problem_id"]: r for r in pipeline_results}
    baseline_map = {r["question_id"]: r for r in baseline_results}
    common_ids = set(pipeline_map) & set(baseline_map)

    a = b = c = d = 0

    for qid in common_ids:
        if criterion == "quality":
            p_pass = _pipeline_quality_pass(pipeline_map[qid])
        else:
            p_pass = _pipeline_correctness_pass(pipeline_map[qid])
        b_pass = baseline_map[qid].get("passed", False)

        if     p_pass and     b_pass: a += 1
        if     p_pass and not b_pass: b += 1
        if not p_pass and     b_pass: c += 1
        if not p_pass and not b_pass: d += 1

    result = mcnemar([[a, b], [c, d]], exact=(b + c < 25))
    # ...
    return {
        "criterion": criterion,
        "p_value":   round(result.pvalue, 4),
        # b + c cells decide significance
        ...
    }
```

**Tablo:**
```
               Baseline PASS   Baseline FAIL
Pipeline PASS       a               b
Pipeline FAIL       c               d

McNemar: b ve c hücrelerine dayanır
p < 0.05 → fark istatistiksel olarak anlamlı
```

---

## 4. Ablation Comparison

### `compare_variants()`

Her varyant için her iki TSR katmanını hesaplar.
`delta_vs_v0` her zaman `TSR_correctness` bazlıdır (apples-to-apples).

```python
def compare_variants(variant_results: dict) -> dict:
    """
    Computes both TSR_quality and TSR_correctness for V0–V5.

    Returns per-variant dict:
        tsr_quality_{overall,easy,medium,hard}
        tsr_correctness_{overall,easy,medium,hard}
        delta_vs_v0  ← correctness-based, comparable to V0
    """
    v0_correctness = compute_tsr_correctness(variant_results.get("V0", []))["overall"]
    comparison = {}
    for variant, results in variant_results.items():
        tsr_q = compute_tsr(results)
        tsr_c = compute_tsr_correctness(results)
        comparison[variant] = {
            "tsr_quality_overall":     tsr_q["overall"],
            "tsr_quality_easy":        tsr_q.get("easy", 0.0),
            "tsr_quality_medium":      tsr_q.get("medium", 0.0),
            "tsr_quality_hard":        tsr_q.get("hard", 0.0),
            "tsr_correctness_overall": tsr_c["overall"],
            "tsr_correctness_easy":    tsr_c.get("easy", 0.0),
            "tsr_correctness_medium":  tsr_c.get("medium", 0.0),
            "tsr_correctness_hard":    tsr_c.get("hard", 0.0),
            "delta_vs_v0": round(tsr_c["overall"] - v0_correctness, 3),
        }
    return comparison
```

---

## 5. Result Tables

### 5.1 Ana Ablasyon Tablosu — İki Tablo

`print_ablation_table()` iki tablo üretir:

**Tablo 1 — TSR_correctness (birincil, V0 ile karşılaştırılabilir):**

```
### TSR_correctness (lcb execution — primary, comparable to V0)

| Variant | Description                 | Easy | Medium | Hard | Overall | ΔV0 |
|---------|-----------------------------|------|--------|------|---------|-----|
| V0      | Single-shot baseline        | ?    | ?      | ?    | ?       | +0.000 |
| V1      | Execution + Judgement       | ?    | ?      | ?    | ?       | ?   |
| V2      | Static Analysis + Judgement | ?    | ?      | ?    | ?       | ?   |
| V3      | Full pipeline               | ?    | ?      | ?    | ?       | ?   |
| V5      | Full pipeline + Trend Ext.  | ?    | ?      | ?    | ?       | ?   |
```

**Tablo 2 — TSR_quality (ikincil, pipeline'ın optimizasyon hedefi):**

```
### TSR_quality (LLM-judge score — secondary, optimisation target)

| Variant | Description                 | Easy | Medium | Hard | Overall | Quality−Correctness |
|---------|-----------------------------|------|--------|------|---------|---------------------|
| V0      | Single-shot baseline        | ?    | ?      | ?    | ?       | +0.000 |
| V1      | Execution + Judgement       | ?    | ?      | ?    | ?       | ?      |
| V2      | Static Analysis + Judgement | ?    | ?      | ?    | ?       | ?      |
| V3      | Full pipeline               | ?    | ?      | ?    | ?       | ?      |
| V5      | Full pipeline + Trend Ext.  | ?    | ?      | ?    | ?       | ?      |
```

`Quality − Correctness` sütunu: pozitifse judge, lcb'nin geçmediği koda puan veriyor.

### 5.2 Per-Model Table

```python
def print_model_table(results_by_model: dict) -> None:
    """Her model için V0 vs V3 karşılaştırması."""
    print("| Model | V0 TSR | V3 TSR | Δ | V3 Ort. iter | STOP_HIGH% |")
    print("|-------|--------|--------|---|-------------|------------|")
    for model, data in results_by_model.items():
        v0  = data.get("v0_tsr", 0.0)
        v3  = data.get("v3_tsr", 0.0)
        avg = data.get("avg_iter", 0.0)
        sh  = data.get("stop_high_rate", 0.0)
        delta = f"+{v3-v0:.3f}" if v3 >= v0 else f"{v3-v0:.3f}"
        print(f"| {model} | {v0:.3f} | {v3:.3f} | {delta} | {avg:.1f} | {sh:.1%} |")
```

---

## 6. Hypothesis Validation

Tüm hipotezler **TSR_correctness** bazlıdır (V0 ile aynı kriter).

```python
def evaluate_hypotheses(comparison: dict, mcnemar_result: dict,
                         model_data: dict) -> dict:
    # All hypothesis comparisons use TSR_correctness (apples-to-apples with V0)
    v0_easy = comparison.get("V0", {}).get("tsr_correctness_easy",    0)
    v0_hard = comparison.get("V0", {}).get("tsr_correctness_hard",    0)
    v0_tsr  = comparison.get("V0", {}).get("tsr_correctness_overall", 0)
    v1_tsr  = comparison.get("V1", {}).get("tsr_correctness_overall", 0)
    v2_tsr  = comparison.get("V2", {}).get("tsr_correctness_overall", 0)
    v3_easy = comparison.get("V3", {}).get("tsr_correctness_easy",    0)
    v3_hard = comparison.get("V3", {}).get("tsr_correctness_hard",    0)
    v3_tsr  = comparison.get("V3", {}).get("tsr_correctness_overall", 0)
    ...
```

| Hipotez | Kriter |
|---------|--------|
| H1: V3 easy TSR_correctness ≥ V0 + 15 puan | TSR_correctness |
| H2: V1 TSR_correctness > V0 TSR_correctness | TSR_correctness |
| H3: V2 TSR_correctness > V0 TSR_correctness | TSR_correctness |
| H4: V3 TSR_correctness > V1 ve > V2 | TSR_correctness |
| H5: Easy gain < Hard gain (ceiling etkisi) | TSR_correctness |
| H6: STOP_HIGH yüksek modelde pipeline katkısı düşük | model bazlı |

---

## 7. CLI Çalıştırma

```bash
python benchmark/analysis.py \
    --pipeline outputs/results/benchmark_V3_v3_full_<ts>.json \
    --baseline outputs/results/baseline_v0_M1_<ts>.json
```

**Çıktı:**
```
Pipeline TSR_correctness (lcb): {"easy": ?, "medium": ?, "hard": ?, "overall": ?}
Pipeline TSR_quality (judge):   {"easy": ?, "medium": ?, "hard": ?, "overall": ?}
Baseline  TSR_correctness (V0): {"easy": ?, "medium": ?, "hard": ?, "overall": ?}

Quality−Correctness gap (pipeline): +X.XXX
  (positive gap = judge credits code that doesn't pass all lcb tests)

McNemar Test (correctness criterion — primary, comparable to V0):
  p-value: X.XXXX
  Pipeline better on N questions, worse on M. ...

McNemar Test (quality criterion — secondary):
  p-value: X.XXXX
  Pipeline better on N questions, worse on M. ...
```

---

## 8. Hata Analizi

### STOP_MAX İnceleme

```python
from memory.logger import read_memory

rows = read_memory()
stop_max = [r for r in rows if r["status"] == "STOP_MAX"]

print(f"STOP_MAX sayısı: {len(stop_max)}")
from collections import Counter
print(Counter(r["difficulty"] for r in stop_max))

stop_max.sort(key=lambda r: r["score"])
for r in stop_max[:5]:
    print(f"  {r['q_id']} | {r['difficulty']} | score={r['score']} | iter={r['iter']}")
```

### improved_prompt Kalite Kontrolü

20 örnek üzerinde manuel inceleme:

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
    # Manuel değerlendirme: 1-5 (actionable? specific? self-contained?)
```

### Per-Node Katkı Analizi

```python
rows = read_memory()

# STOP_HIGH oranı — Node 1 etkisi
total   = len(rows)
stop_hi = sum(1 for r in rows if r["status"] == "STOP_HIGH")
print(f"Node 1 STOP_HIGH oranı: {stop_hi/total:.1%} ({stop_hi}/{total})")

# Ortalama iterasyon (PASS durumları)
pass_rows = [r for r in rows if r["status"] == "PASS"]
if pass_rows:
    avg_iter = sum(r["iter"] for r in pass_rows) / len(pass_rows)
    print(f"Ortalama PASS iterasyonu: {avg_iter:.1f}")

# Skor iyileşmesi (LOOP → PASS)
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
