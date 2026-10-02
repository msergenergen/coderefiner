# analysis.md — Results Analysis

> **Durum (2026-10-02).** §1–§8 eski `benchmark/analysis.py` araç takımını anlatır; bu kod
> `benchmark/legacy/analysis.py` altına taşındı. Güncel analiz araçları: §9
> (`paper_results.py`, ELECO bildirisi) ve §10 (`openrouter_results.py`, ana deney).
> TSR_correctness = pass@1 (teslim edilen kod, gizli testler); bkz. `docs/ablation.md` §1 notu.

## 1. TSR Tanımı — İki Katman

Pipeline varyantları (V1–V3) ile V0 baseline arasında metodolojik olarak
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
        pipeline_results: benchmark JSON results (V1–V3)
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
    Computes both TSR_quality and TSR_correctness for V0–V3.

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

---

## 9. Paper Results — `benchmark/paper_results.py`

Produces every number, the results table and the budget curve used in the ELECO paper, from the
iteration JSONs of one or more runs.

```bash
python -m benchmark.paper_results --run-ids run_20260930_222527 --out ~/Desktop/eleco2026-bildiri
```

**Multiple runs.** `--run-ids` accepts several runs; if a problem appears in more than one, the
chronologically latest run supersedes the earlier attempt entirely (even while still in progress).

**Comparing two judge-decision designs.**
`--baseline-run-ids` analyses a second set of runs on the *same* problems (restricted to the
main run's completed problems), writes its macros with a `q` prefix (`\qVt`, `\qRubPrec`, …) and a
paired McNemar on the delivered code of the two designs (`\rDn`, `\rDb` = only main correct,
`\rDc` = only baseline correct, `\rDp`). The results table then shows `V3_P` (rubric/score
decision) and `V3_Y` (judge verdict) side by side. `--qid-file` restricts both to a fixed problem
list. Judge-decision metrics on iterations that passed every public test: `\rPubN`, `\rPubRej`
(rejected), `\rPubRejGood` (rejected and actually wrong on hidden tests), `\rPubRejBad` (rejected
but correct), `\rPubWrong`, mean judge points for hidden-correct vs. hidden-wrong code
(`\rLlmOk`, `\rLlmBad`), share of decisions made by the judge itself (`\rVerdictJudge`).

```bash
python -m benchmark.paper_results --run-ids run_20261001_102445 \
    --baseline-run-ids run_20260930_222527 run_20261001_070529 \
    --qid-file benchmark/sample50_qids.txt --out ~/Desktop/eleco2026-bildiri
```

**Protocol (leakage-free).** The refinement loop only ever sees the public example tests
(`nodes/sandbox.py` Phase 1). `paper_results.py` re-grades the code of *every* iteration offline on the
full hidden test set (public + private) with `sandbox/lcb_eval._fallback_evaluate`
(8 threads, cached in `outputs/results/hidden_eval_cache.json`; `STOP_HIGH` iterations count as 0).

- **Strict accuracy** = hidden `pass_rate == 1.0`.
- **V0** = iteration 1 of the same run (single-shot baseline under identical prompt/token budget);
  **V3** = the code delivered at the last iteration.
- **Paired test:** exact (binomial) McNemar on the discordant counts b (only V3 correct) and c (only V0 correct);
  **95% CIs:** Wilson.
- **success@k:** strict accuracy of the code delivered with a budget of k iterations
  (carry-forward of the last iteration for problems that stopped earlier).
- **Rubric precision / recall:** judge `PASS` decisions versus hidden-test correctness, per iteration.
- **Gain decomposition:** of the problems only V3 solves (b), how many had an unparseable (cut-off) first attempt (`\rBTrunc`) vs. runnable-but-wrong code (`\rBFix`).
- **Truncation by difficulty:** `\rTruncKolay/Orta/Zor` — share of codegen calls hitting the 6144-token output cap or the 8192 context.
- **Process stats:** severity-gate distribution, syntax-error share, truncated generations,
  judge-call failures, JSON fallbacks, token ratio (all iterations / first iteration).

**Outputs** (in `--out`, which is the Overleaf project root):
- `veri/sonuclar.tex` — `\newcommand` macros (`\rN`, `\rVo`, `\rVt`, `\rP`, `\rRubPrec`, per-difficulty and
  per-platform variants, `...En` variants with a decimal point for the English abstract).
  Numbers use the Turkish decimal comma (`{,}`) and a math minus.
- `cizelgeler/cizelge2-sonuclar.tex` — tabular body of the results table (caption/label are in
  `bolumler/05-bulgular.tex`).
- `sekiller/sekil2-yineleme-butcesi.pdf` — grayscale success@k curve (overall and per difficulty).

Paper project layout: `main.tex` (preamble + `\input` order), `bolumler/00-…07-*.tex` (sections),
`sekiller/`, `cizelgeler/`, `veri/`, `kaynaklar.tex`, `OKUBENI.md`.

Because the sample order is seeded-random, a completed prefix of the 150-problem run is itself a
random subsample; the paper reports `\rN` completed problems.

---

## 10. Main experiment — `benchmark/openrouter_results.py` (2026-10-02)

Inputs: the baseline JSONL from `benchmark/sample_baseline.py` (V0, n = 5 samples per problem),
the pipeline run ids of design Y (`JUDGE_DECISION=verdict`) and P (`rubric`), and the API ledger.
Only problems present in every source are used, so all comparisons are paired. Hidden-test
grading reuses `paper_results.hidden_evaluate` (cache: `<results-root>/hidden_eval_cache.json`)
with `timeout=HIDDEN_TIMEOUT_SEC` (10 s) and `stop_on_fail=True` — the same rules as the V0
baseline, stopping at the first failing test as lcb_runner does (`docs/node2.md`). Only
"all tests pass" is reported, so the early stop does not change any metric. `paper_results`
itself keeps its defaults (10 s, no early stop) so the ELECO numbers and cache are unchanged.
Cost is summed from the ledger by run id; baseline and pipeline entries are told apart by the
`phase` tag, because run ids are second-stamped and the main baseline shared its id
(`run_20261002_091834`) with pipeline shard Y-0 that started in the same second.

| Output | Definition |
|---|---|
| V0 pass@1 | mean over problems of c/n; bootstrap 95% CI over problems |
| pass@k | unbiased estimator 1 − C(n−c, k)/C(n, k) (Chen et al. 2021); needs hidden tests to pick → upper bound |
| filt@k | first of the first k samples that passes all public tests is submitted (else the k-th); no hidden-test access |
| V3 | strict accuracy of the delivered code, Wilson 95% CI |
| V3 − V0 | mean paired difference (V3_i − c_i/n), bootstrap 95% CI, Wilcoxon signed-rank p |
| iter 1 → final | exact McNemar on the pipeline's own first attempt vs. delivered code |
| budget-matched | V3 vs filt@k and pass@k at k = ⌊k̄⌋, ⌈k̄⌉, k̄ = mean generations per problem |
| Y vs P | exact McNemar on delivered code |
| `rows` (results.json) | per design, question_id → delivered code passes all hidden tests |
| `--langfuse` | uploads `hidden_pass` / `hidden_pass_first` scores to the runs' Langfuse traces (`docs/codegen-models.md`) |
| `--latex <paper dir>` | writes `veri/ana_sonuclar.tex` (`\a…` macros: Turkish decimal comma, plus `…En` variants with a decimal point for the Abstract) and `cizelgeler/cizelge-ana.tex` (difficulty × V0/V3_Y/V3_P table) from `results.json`; the paper never hand-copies a number |
| `--compare <results.json>` | per design, exact McNemar against another analysis on common problems (ablation: feedback only vs feedback + previous solution) |
| judge | public-passing attempts, rejections and whether hidden tests agree, acceptance precision, share of verdicts from the judge, JSON/LLM errors |
| feedback leaks | judge feedback containing solution code (fenced block with logic lines, or ≥ 2 logic lines outside fences; format instructions and empty skeletons excluded) |
| cost | USD and calls per phase (baseline file stem / run ids in the ledger) |

```bash
python -m benchmark.openrouter_results \
    --baseline outputs/openrouter/results/baseline_samples/<run_id>.jsonl \
    --y-run-ids <run ids> --p-run-ids <run ids> \
    [--results-root outputs/openrouter/results] [--ledger outputs/api_usage.jsonl] \
    [--out outputs/openrouter/report]
```

Writes `results.json` and a Turkish `report.md`. A dry run on the ELECO local runs (n = 1,
first iteration as V0) reproduced the paper's paired numbers and found solution code in 7/97
(Y) and 9/109 (P) judge feedbacks — the reason for the no-code rule (`docs/node3.md` §4.3).


---

## 11. Offline judge replay — `benchmark/judge_replay.py` (2026-10-02)

Re-judges every recorded attempt that passed all public tests with a candidate judge and
scores the verdict against the hidden tests. Only these attempts are replayed: a failing
public test forces REJECT whatever the judge says (`_public_precondition`), so these are the
decisions that determine the delivered code.

- **Same input.** The candidate receives `JUDGEMENT_SYSTEM_PROMPT` and the user prompt that
  `_build_judgement_prompt` builds from the iteration JSON (problem excerpt, comment-stripped
  code, Node 1 findings, Node 2 results), parsed with `_parse_judgement`.
- **Candidate.** The registry's judge slot, selected with `OPENROUTER_JUDGE_MODEL` /
  `OPENROUTER_JUDGE_PROVIDER` (entries in `config/models.py`). Reasoning models get
  `JUDGE_MAX_TOKENS=8192` because reasoning tokens count toward the output budget.
- **Decision.** The pipeline's own rows use `node3.pass` (a missing verdict already carries its
  rubric fallback). A candidate's missing verdict (`NONE`/`ERROR`) counts as ACCEPT, because the
  rubric fallback never rejected public-passing code in the main runs; `no_verdict` reports how
  often that happened.
- **Ledger.** Calls carry `phase=judge_replay` and `run_id=replay_<name>`, so pipeline cost
  figures (`cost()` matches run ids) are unaffected; the spend still counts toward the cap.
- **Limit.** It measures decision accuracy only. How the loop would have continued after a
  different verdict needs a full pipeline run; `full_run_estimate` gives its wall time
  (slowest shard, candidate's mean latency per judge call, same iterations) and judge cost.

| Field (`report.json`) | Meaning |
|---|---|
| `precision` | hidden-correct share of accepted attempts (the paper's acceptance precision) |
| `wrong_caught` | rejected share of hidden-wrong attempts (what the loop gets to repair) |
| `right_kept` | accepted share of hidden-correct attempts |
| `accuracy` | decision agrees with the hidden tests |
| `vs_original` | same attempts, exact McNemar on "decision agrees with hidden tests" against the pipeline's judge; `verdict_changed` counts flipped decisions |
| `full_run_estimate` | one full Y run with this judge: judge calls, original and estimated wall minutes, judge cost |

```bash
scripts/run_openrouter.sh judge-replay openai/gpt-oss-120b deepinfra/bf16 gpt-oss-120b \
    --run-ids <Y-A and Y-B run ids> --groups A_feedback_only=<ids,…> B_main=<ids,…> \
    --estimate-runs <Y-B run ids> [--workers 8] [--limit N]
python -m benchmark.judge_replay --run-ids … --groups … --estimate-runs … --report
```

Output: `outputs/openrouter/judge_replay/<name>.jsonl` (one row per attempt, resumable) and
`report.json`. Results: `docs/main-results.md`.
