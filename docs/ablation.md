# ablation.md — Academic Framework and Ablation Experiments

## 1. Research Question and Terminology

### Research Question

> Can a multi-stage iterative refinement pipeline that integrates static code analysis,
> sandboxed execution, and LLM-based judgment mechanisms
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

**Correct comparison:**
Pipeline TSR = (Pipeline ile PASS alan sorular) / (Toplam sorular)
Baseline TSR = (Questions correct in single-shot) / (Total questions)

---

## 2. Ablation Variants

Each variant runs independently. Separate results are obtained for each model per variant.

| Variant | Node 1 | Node 2 | Node 3 | Description |
|---------|--------|--------|--------|----------|
| **V0** — Single-shot baseline | ✗ | ✗ | ✗ | Pipeline yok, tek deneme |
| **V1** — +Execution | ✗ | ✓ | ✗ | Feedback from test results only |
| **V2** — +Static+Execution | ✓ | ✓ | ✗ | Statik analiz + test, judgment yok |
| **V3** — +Execution+Judgment | ✗ | ✓ | ✓ | Test + judgment, statik analiz yok |
| **V4** — Full Pipeline | ✓ | ✓ | ✓ | Tam sistem |

### What Each Variant Answers

```
V0 → Modelin doğal yeteneği nedir? (tavan referansı)
V1 → Sadece "doğru cevap bu" bilgisi yeterli mi?
V2 → Statik analiz feedback'e ek katkı sağlıyor mu?
V3 → Judgment LLM olmadan execution yeterli mi?
V4 → Tüm bileşenler birlikte ne kadar katkı sağlıyor?

V2 - V1 farkı → Node 1'in marjinal katkısı
V4 - V3 farkı → Node 1'in V4 içindeki katkısı
V3 - V1 farkı → Node 3'ün (Judgment) marjinal katkısı
```

---

## 3. Hipotezler

```
H1: V4 TSR ≥ V0 TSR + 15 puan (easy sorularda)
    Gerekçe: Execution feedback en az 2 iterasyonda yanlış kodu düzeltir

H2: V3 TSR > V2 TSR (Judgment katkısı > Statik analiz katkısı)
    Gerekçe: failed_cases + improved_prompt daha actionable

H3: Easy sorularda pipeline katkısı < Hard sorularda katkı
    Gerekçe: Easy'de modeller zaten yüksek başarıya yakın (tavan etkisi)

H4: STOP_HIGH oranı yüksek modellerde pipeline TSR katkısı düşer
    Gerekçe: Node 1'de çok soru ellenirse iterasyon şansı kalmaz
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

Mevcut `evaluator.py`'e `--variant` flag'i eklenir.
The difference between V1–V4 is managed via the `build_graph(variant=...)` call:

```python
# benchmark/evaluator.py — additional arguments

parser.add_argument(
    "--variant",
    default="V4",
    choices=["V1", "V2", "V3", "V4"],
    help=(
        "V1: sadece Node 2  |  "
        "V2: Node 1 + Node 2  |  "
        "V3: Node 2 + Node 3  |  "
        "V4: tam pipeline"
    ),
)
```

```python
# pipeline/graph.py — build_graph() variant support

def build_graph(variant: str = "V4") -> StateGraph:
    """
    variant:
        V1 — Node 1 yok, Node 3 yok; sadece execution feedback
        V2 — Node 1 var, Node 3 yok
        V3 — Node 1 yok, Node 3 var
        V4 — tam pipeline (varsayılan)
    """
    use_node1 = variant in ("V2", "V4")
    use_node3 = variant in ("V3", "V4")

    g = StateGraph(PipelineState)

    g.add_node("codegen",           codegen_node)
    g.add_node("static_analysis",   static_analysis_node)
    g.add_node("sandbox",           sandbox_node)
    g.add_node("strip_comments",    strip_comments_node)
    g.add_node("judgment",          judgment_node)
    g.add_node("log_iteration",     log_iteration_node)
    g.add_node("log_severity_stop", log_severity_stop_node)
    g.add_node("write_memory_final",_write_memory_final_node)
    g.add_node("stop_max",          _stop_max_node)
    g.add_node("next_model",        _next_model_node)

    g.set_entry_point("codegen")

    # Node 1 yok → codegen'den direkt sandbox'a git
    if use_node1:
        g.add_edge("codegen", "static_analysis")
        g.add_conditional_edges(
            "static_analysis", severity_gate,
            {"to_node2": "sandbox", "to_node3": "judgment",
             "stop_high": "log_severity_stop"},
        )
    else:
        # severity_gate olmadan direkt sandbox
        g.add_edge("codegen", "sandbox")

    g.add_edge("sandbox", "strip_comments")

    # Node 3 yok → execution sonucu direkt log'a git
    if use_node3:
        g.add_edge("strip_comments", "judgment")
        g.add_edge("judgment",       "log_iteration")
    else:
        # without judgment: PASS if pass_rate meets threshold
        # Handle this via _no_judgment_gate inside log_iteration_node
        g.add_edge("strip_comments", "log_iteration")

    g.add_conditional_edges(
        "log_iteration", iteration_gate,
        {"write_memory_final": "write_memory_final",
         "stop_max":           "stop_max",
         "loop_to_codegen":    "codegen"},
    )

    g.add_edge("write_memory_final", "next_model")
    g.add_edge("log_severity_stop",  END)
    g.add_edge("stop_max",           END)
    g.add_edge("next_model",         END)

    return g.compile()
```

> **Note:** When Node 3 is absent in V1 and V3, `iteration_gate` uses only
> `node2_pass_rate` as the threshold. `node3_total_score` and `node3_pass`
> remain 0/False in state; `iteration_gate` evaluates this via `node2_pass_rate`.
> This behavior should be documented in `gates.py`.

---

## 6. Phase 4 — Experiment Run Commands

### Order Matters: Baseline First, Then Pipeline

```bash
# ── V0 Baseline — separately for each model ────────────────────────────
ENV=cloud python benchmark/baseline.py --model M1 --sample 50 --tag v0_M1
ENV=cloud python benchmark/baseline.py --model M2 --sample 50 --tag v0_M2
ENV=cloud python benchmark/baseline.py --model M3 --sample 50 --tag v0_M3

# ── Ablation Variants ───────────────────────────────────────────────
ENV=cloud python -m benchmark.evaluator --variant V1 --sample 50 --tag v1
ENV=cloud python -m benchmark.evaluator --variant V2 --sample 50 --tag v2
ENV=cloud python -m benchmark.evaluator --variant V3 --sample 50 --tag v3
ENV=cloud python -m benchmark.evaluator --variant V4 --sample 50 --tag v4_full

# ── Difficulty-Based (V4 full pipeline) ───────────────────────────────────
ENV=cloud python -m benchmark.evaluator --variant V4 --difficulty easy   --sample 25 --tag v4_easy
ENV=cloud python -m benchmark.evaluator --variant V4 --difficulty medium --sample 15 --tag v4_medium
ENV=cloud python -m benchmark.evaluator --variant V4 --difficulty hard   --sample 10 --tag v4_hard
```

### Check Before Each Run

```bash
# Were the previous run's outputs copied to local machine?
ls outputs/results/

# Is the server running?
curl -s http://localhost:8001/v1/models | python -m json.tool

# Disk dolu mu? (100GB+ model var)
df -h /workspace
```

### Output Files

```
outputs/
├── run_info.json                    ← Her run için ortam bilgisi
├── memory.md                        ← Tüm iterasyonlar (append)
├── results/
│   ├── baseline_v0_M1_20250115.json
│   ├── baseline_v0_M2_20250115.json
│   ├── baseline_v0_M3_20250115.json
│   ├── benchmark_v1_20250116.json
│   ├── benchmark_v2_20250116.json
│   ├── benchmark_v3_20250116.json
│   └── benchmark_v4_full_20250117.json
└── iterations/
    └── q_xxx/
        └── M1_iter1.json
```

> Before each variant starts, `outputs/memory.md` must be reset
> or redirected to a separate file such as `MEMORY_LOG_PATH=outputs/memory_v1.md`.
