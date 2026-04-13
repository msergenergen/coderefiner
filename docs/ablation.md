# ablation.md — Academic Framework and Ablation Experiments

## 1. Research Question and Terminology

### Research Question

> Can a multi-stage iterative refinement pipeline that integrates static code analysis,
> sandboxed execution, and LLM-based judgement mechanisms
> meaningfully improve the LiveCodeBench task success rate of
> selected open-weight models compared to a single-shot baseline?

### Terminology Decisions

| Term Used | Why | Term Avoided |
|-----------------|-------|-----------------|
| Task Success Rate (TSR) | Pipeline runs iteratively — not a single-shot metric | pass@1 |
| Iterative guided repair | Node feedback is used — not blind retry | self-repair |
| Baseline (single-shot) | Model output without pipeline, single attempt | vanilla |

**Why not pass@1:**
LCB pass@1 measures the rate at which a model gives the correct answer in one attempt.
The pipeline allows up to 10 iterations per model and generates an improved_prompt
with Node 1/2/3 feedback at each iteration. This is test-time compute
with guided repair and is not directly comparable to pass@1.

**İki Katmanlı TSR:**

| Metrik | Kriter | Kullanım Yeri |
|--------|--------|---------------|
| **TSR_correctness** | `final_pass_rate == 1.0` (lcb execution) | V0 ile birincil karşılaştırma; McNemar; H1–H4 |
| **TSR_quality** | `node3_pass == True` (LLM-judge skoru ≥ eşik) | Pipeline optimizasyon hedefi; ikincil analiz |

```
Pipeline TSR_correctness = (final_pass_rate==1.0 olan sorular) / (Toplam sorular)
Pipeline TSR_quality     = (node3_pass==True olan sorular) / (Toplam sorular)
Baseline TSR             = (lcb pass_rate==1.0 olan sorular) / (Toplam sorular)  ← = TSR_correctness
```

`Quality − Correctness gap` pozitifse: judge, lcb'nin geçmediği koda puan veriyor.
Bu fark hem metodolojik şeffaflık hem de tez bulgusu olarak raporlanır.

---

## 2. Ablation Variants

Each variant runs independently. Separate results are obtained for each model per variant.

| Variant | Node 1 | Node 2 | Node 3 | Trend Ext. | Description |
|---------|--------|--------|--------|------------|-------------|
| **V0** — Single-shot baseline | ✗ | ✗ | ✗ | ✗ | Pipeline yok, tek deneme |
| **V1** — Execution + Judgement | ✗ | ✓ | ✓ | ✗ | Test sonuçları + Judgement LLM, statik analiz yok |
| **V2** — Static Analysis + Judgement | ✓ | ✗ | ✓ | ✗ | Statik analiz + Judgement LLM, execution yok |
| **V3** — Full Pipeline | ✓ | ✓ | ✓ | ✗ | Tam sistem, sabit max_iterations |
| **V5** — Full Pipeline + Trend Extension | ✓ | ✓ | ✓ | ✓ | V3 + EMA trend analizi ile ek iterasyon |

> **Not:** V4 numarası gelecekteki bir variant için rezerve edilmiştir.

### What Each Variant Answers

```
V0 → Modelin doğal yeteneği nedir? (referans taban)
V1 → Execution feedback + Judgement yeterli mi? (Node 1 olmadan)
V2 → Statik analiz + Judgement yeterli mi? (Node 2 olmadan)
V3 → Tüm bileşenler birlikte ne kadar katkı sağlıyor?
V5 → Trend-aware extension STOP_MAX vakalarını kurtarıyor mu?

V3 - V1 farkı → Node 1'in (statik analiz) marjinal katkısı
V3 - V2 farkı → Node 2'nin (execution) marjinal katkısı
V1 - V0 farkı → Execution + Judgement'ın baseline'a katkısı
V2 - V0 farkı → Static + Judgement'ın baseline'a katkısı
V5 - V3 farkı → Trend extension'ın marjinal katkısı
```

### V5 Aktivasyon

V5, `TREND_EXTENSION_ENABLED=true` ile etkinleştirilir.
`iteration_gate` max_iterations'a ulaşıldığında stop_max yerine
EMA trend analizini çalıştırır; **tüm 5 koşul sağlanırsa** ek iterasyon verir:

1. `accuracy > 0` — en az bir test geçiyor (hard gate)
2. `pass_rate_slope > 0` — pass rate son 3 iterasyonda artıyor
3. EMA-smoothed `score_slope >= TREND_MIN_SLOPE` — skor gerçekten iyileşiyor
4. `expected_gain = slope × remaining >= gap` — kalan extension'larla eşiğe ulaşılabilir
5. `extension_count < MAX_EXTENSION_ITERS` — bütçe tükenmemiş

```bash
# V5 çalıştırma
TREND_EXTENSION_ENABLED=true python -m benchmark.evaluator --variant V3 --sample 50 --tag v5
```

---

## 3. Hipotezler

```
H1: V3 easy TSR ≥ V0 easy TSR + 15 puan
    Gerekçe: Execution + statik analiz + Judgement feedback
             en az 2 iterasyonda yanlış kodu düzeltir

H2: V1 TSR > V0 TSR  (Execution + Judgement baseline'ı geçiyor)
    Gerekçe: failed_cases + improved_prompt, kör yeniden denemeden
             daha fazla bilgi taşır

H3: V2 TSR > V0 TSR  (Static Analysis + Judgement baseline'ı geçiyor)
    Gerekçe: Statik sorunlar düzeltildiğinde kod kalitesi artar
             ve Judgement daha iyi improved_prompt üretir

H4: V3 TSR > V1 TSR  ve  V3 TSR > V2 TSR  (tam pipeline ablasyonları geçiyor)
    Gerekçe: Her iki node'un birlikte çalışması daha zengin feedback sağlar
```

### Statistical Test

**McNemar test** — for paired binary outcomes.
A pair is formed per question: (pipeline PASS?, baseline PASS?)

```python
from statsmodels.stats.contingency_tables import mcnemar

# Tablo:
#         Baseline PASS  Baseline FAIL
# Pipeline PASS    a           b
# Pipeline FAIL    c           d

# McNemar: based on b and c cells
# p < 0.05 → difference is statistically significant

result = mcnemar([[a, b], [c, d]], exact=True)
print(f"p-value: {result.pvalue:.4f}")
print(f"Odds Ratio: {b/c:.2f}" if c > 0 else "OR: inf")
```

**95% Confidence Interval** — Wilson score interval for TSR:

```python
from statsmodels.stats.proportion import proportion_confint

lo, hi = proportion_confint(
    count=passed, nobs=total, alpha=0.05, method="wilson"
)
print(f"TSR: {passed/total:.3f} [{lo:.3f}, {hi:.3f}]")
```

---

## 4. benchmark/baseline.py

```python
"""
Single-shot baseline çalıştırma — V0 varyantı.

Her model için pipeline olmadan, tek denemeli kod üretimi yapar.
Sonuçlar ablasyon karşılaştırması için kullanılır.

Kullanım:
    python benchmark/baseline.py --model M1 --sample 50 --tag v0_M1
    python benchmark/baseline.py --model M2 --sample 50 --tag v0_M2
    python benchmark/baseline.py --model M3 --sample 50 --tag v0_M3
"""
import argparse
import json
import os
import time
from datetime import datetime
from loguru import logger

from config import settings, model_registry
from dataset.loader import load_lcb
from dataset.splitter import format_problem
from dataset.edge_cases import augment_row
from models.client import get_client, get_model_name, get_temperature
from sandbox.lcb_eval import lcb_evaluate
from memory.logger import write_run_info


def run_single_shot(
    problem_row: dict,
    model_id: str,
    run_id: str,
) -> dict:
    """
    Tek model için pipeline olmadan tek denemeli kod üretir.
    """
    aug_row    = augment_row(problem_row)
    edge_cases = aug_row.get("edge_cases", [])
    prompt     = format_problem(problem_row)

    cfg    = model_registry.get(model_id)
    client = get_client(model_id)

    t0 = time.perf_counter()

    try:
        response = client.chat.completions.create(
            model=cfg.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=cfg.temperature,
            max_tokens=2048,
        )
        code = _extract_code(response.choices[0].message.content)
        tokens = {
            "prompt":     response.usage.prompt_tokens     if response.usage else 0,
            "completion": response.usage.completion_tokens if response.usage else 0,
        }
    except Exception as e:
        logger.warning(f"[{model_id}] Codegen hatası: {e}")
        code   = ""
        tokens = {"prompt": 0, "completion": 0}

    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    # Correctness evaluation — lcb_runner or fallback
    eval_result = lcb_evaluate(
        code=code,
        problem_row=aug_row,
        edge_cases=edge_cases,
        timeout=settings.sandbox_timeout_sec,
    )

    passed = eval_result["pass_rate"] == 1.0 and eval_result["total_count"] > 0

    return {
        "run_id":      run_id,
        "model":       model_id,
        "model_name":  cfg.model_name,
        "question_id": problem_row["id"],
        "difficulty":  problem_row["difficulty"],
        "variant":     "V0",
        "passed":      passed,
        "pass_rate":   eval_result["pass_rate"],
        "passed_count": eval_result["passed_count"],
        "total_count":  eval_result["total_count"],
        "elapsed_ms":  elapsed_ms,
        "tokens":      tokens,
        "code":        code,
        "failed_cases": eval_result["failed_cases"],
    }


def _extract_code(text: str) -> str:
    """
    LLM çıktısından ```python ... ``` bloğunu çıkarır.
    Bulunamazsa tüm metni döner.
    """
    import re
    # Take the last ```python...``` block (guard against lcb_runner bug)
    matches = re.findall(r"```python\s*(.*?)```", text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    # Backtick block only (no python tag)
    matches = re.findall(r"```\s*(.*?)```", text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return text.strip()


def run_baseline(
    model_id: str,
    difficulty: str = "all",
    sample_size: int = 50,
    output_tag: str = "",
) -> dict:

    run_id = write_run_info()
    rows   = load_lcb(
        split       = settings.lcb_split,
        difficulty  = difficulty,
        sample_size = sample_size,
    )

    logger.info(
        f"Baseline | run_id={run_id} | model={model_id} | "
        f"{len(rows)} soru | difficulty={difficulty}"
    )

    results      = []
    pass_counts  = {"easy": 0, "medium": 0, "hard": 0}
    total_counts = {"easy": 0, "medium": 0, "hard": 0}

    for i, row in enumerate(rows):
        logger.info(f"[{i+1}/{len(rows)}] {row['title']} ({row['difficulty']})")
        result = run_single_shot(row, model_id, run_id)
        results.append(result)

        diff = row["difficulty"]
        total_counts[diff] += 1
        if result["passed"]:
            pass_counts[diff] += 1

    summary = {
        "run_id":    run_id,
        "variant":   "V0",
        "model":     model_id,
        "timestamp": datetime.now().isoformat(),
        "total":     len(rows),
        "tsr": {
            d: round(pass_counts[d] / total_counts[d], 3)
            if total_counts[d] else 0.0
            for d in ("easy", "medium", "hard")
        },
        "results": results,
    }

    output_dir = settings.output_dir
    os.makedirs(output_dir, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag  = f"_{output_tag}" if output_tag else ""
    path = os.path.join(output_dir, f"baseline{tag}_{ts}.json")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(f"Tamamlandı → {path}")
    logger.info(f"TSR: {summary['tsr']}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM-as-a-Judge — V0 Single-Shot Baseline")
    parser.add_argument("--model",      required=True, choices=["M1", "M2", "M3"])
    parser.add_argument("--difficulty", default="all")
    parser.add_argument("--sample",     type=int, default=50)
    parser.add_argument("--tag",        default="")
    args = parser.parse_args()

    run_baseline(
        model_id   = args.model,
        difficulty = args.difficulty,
        sample_size= args.sample,
        output_tag = args.tag,
    )
```

---

## 5. benchmark/evaluator.py — Variant Support

The difference between V1–V3 is managed via the `build_graph(variant=...)` call.

```python
# benchmark/evaluator.py — CLI argument

parser.add_argument(
    "--variant",
    default="V3",
    choices=["V1", "V2", "V3"],
    help=(
        "V1: Execution + Judgement (Node 2 + Node 3)  |  "
        "V2: Static Analysis + Judgement (Node 1 + Node 3)  |  "
        "V3: Full pipeline (Node 1 + Node 2 + Node 3)"
    ),
)
```

```python
# pipeline/graph.py — build_graph() variant support

def build_graph(variant: str = "V3") -> StateGraph:
    """
    variant:
        V1 — Node 2 + Node 3  (Execution + Judgement, no static analysis)
        V2 — Node 1 + Node 3  (Static Analysis + Judgement, no execution)
        V3 — Full pipeline     (Node 1 + Node 2 + Node 3)
    """
    use_node1 = variant in ("V2", "V3")
    use_node2 = variant in ("V1", "V3")

    # Core nodes always present: codegen, judgement, log_iteration, etc.
    # Node 1 (static_analysis, severity_gate) added if use_node1
    # Node 2 (sandbox, strip_comments)        added if use_node2
    # V2 routing: codegen → static_analysis → strip_before_judge → judgement
    # V1 routing: codegen → sandbox → strip_comments → judgement
    # V3 routing: codegen → static_analysis → sandbox/strip_before_judge → judgement
```

PASS criterion is identical across all variants: `node3_total_score ≥ threshold`
(Judgement always runs — V0 is the only variant without it, and V0 uses `baseline.py`).

### benchmark_V*.json — Output Format

Each evaluator run writes a JSON file with two TSR layers:

```json
{
  "run_id":    "run_20250413_143000",
  "variant":   "V3",
  "timestamp": "2025-04-13T14:30:00",
  "total":     50,
  "tsr_quality": {
    "easy":   0.880,
    "medium": 0.720,
    "hard":   0.600
  },
  "tsr_correctness": {
    "easy":   0.800,
    "medium": 0.640,
    "hard":   0.520
  },
  "results": [
    {
      "problem_id": "q_001",
      "difficulty": "easy",
      "results": [
        { "model": "M1", "status": "PASS", "score": 84.0, "final_pass_rate": 0.95 },
        { "model": "M2", "status": "PASS", "score": 91.0, "final_pass_rate": 1.00 }
      ]
    }
  ]
}
```

- **`tsr_quality`** — `node3_pass == True` (LLM-judge criterion). Pipeline optimization target.
- **`tsr_correctness`** — `final_pass_rate == 1.0` (lcb execution criterion). Primary comparison metric; apples-to-apples with V0 `"tsr"`.
- V0 baseline JSON uses `"tsr"` (correctness-only, single model, no judge).

---

## 6. Phase 4 — Experiment Run Commands

### Order Matters: Baseline First, Then Pipeline

```bash
# ── V0 Baseline — separately for each model ───────────────────────────
python benchmark/baseline.py --model M1 --sample 50 --tag v0_M1
python benchmark/baseline.py --model M2 --sample 50 --tag v0_M2
python benchmark/baseline.py --model M3 --sample 50 --tag v0_M3

# ── Ablation Variants ──────────────────────────────────────────────────
python -m benchmark.evaluator --variant V1 --sample 50 --tag v1
python -m benchmark.evaluator --variant V2 --sample 50 --tag v2
python -m benchmark.evaluator --variant V3 --sample 50 --tag v3_full

# ── Difficulty breakdown (V3 full pipeline) ────────────────────────────
python -m benchmark.evaluator --variant V3 --difficulty easy   --sample 25 --tag v3_easy
python -m benchmark.evaluator --variant V3 --difficulty medium --sample 15 --tag v3_medium
python -m benchmark.evaluator --variant V3 --difficulty hard   --sample 10 --tag v3_hard
```

> `.env` symlink → `.env.cloud` olduğu için `ENV=cloud` prefix'i gerekmez.

### Check Before Each Run

```bash
# Previous outputs saved to local machine?
ls outputs/results/

# All 4 servers running?
for port in 8001 8002 8003 8004; do
    curl -s http://localhost:$port/v1/models | python -m json.tool | grep '"id"' || echo "port $port: NOT running"
done

# Disk dolu mu? (~99 GB model + outputs)
df -h /workspace
```

### Output Files

```
outputs/
├── run_info.json                        ← Her run için ortam bilgisi
├── memory.md                            ← Tüm iterasyonlar (append)
└── results/
    └── YYYY-MM-DD/
        ├── baseline_v0_M1_<ts>.json
        ├── baseline_v0_M2_<ts>.json
        ├── baseline_v0_M3_<ts>.json
        ├── benchmark_V1_v1_<ts>.json
        ├── benchmark_V2_v2_<ts>.json
        ├── benchmark_V3_v3_full_<ts>.json
        └── iterations/
            └── <question_id>/
                └── M1_iter1.json
```

> Her variant başlamadan önce `outputs/memory.md` sıfırlanmalı
> ya da `MEMORY_LOG_PATH=outputs/memory_v1.md` şeklinde yönlendirilmeli.
