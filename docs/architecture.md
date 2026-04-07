# architecture.md — Pipeline Mimarisi

## 1. Overview

The pipeline is a cyclic graph structure managed by LangGraph.
Three LLMs run sequentially; each model's output passes through three analysis nodes.
Failing outputs are retried with the same model up to `max_iterations`.

---

## 2. Pipeline Flow

```
START
  │
  ▼
┌─────────────────────────────┐
│   Sequential Model Router   │  M1 → M2 → M3 (sırayla)
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│      Code Generation        │  ← improved_prompt (loop'tan)
│  aktif model + system prompt│
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│   NODE 1: Static Analysis   │
│  CC · MI · Bandit · Halstead│
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│       Severity Gate         │
└──┬──────────┬──────────┬────┘
   │          │          │
 HIGH        LOW         OK
   │          │          │
   ▼          │          ▼
 STOP         │   ┌──────────────────┐
  END         │   │ NODE 2: Sandbox  │
              │   │ CPU·RAM·test pass│
              │   └────────┬─────────┘
              │            │
              └──────┬─────┘
                     │
                     ▼
          ┌──────────────────────┐
          │  NODE 3: Judgement LLM│
          │  100-puan rubric     │
          └──────────┬───────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │    Iteration Gate    │
          └──┬───────────┬───┬───┘
             │           │   │
           PASS      FAIL     FAIL
             │      <max      =max
             │         │         │
             ▼         ▼         ▼
        memory.md    loop      STOP
             │      codegen     END
             ▼
      Next Model / END
```

---

## 3. Node Details

### Node 1 — Statik Analiz

Purpose: Detect structural and security issues before executing the code.

| Metric | Tool | Threshold | Severity |
|--------|------|-----------|----------|
| Cyclomatic Complexity (CC) | `radon cc` | ≤ 10/fonksiyon | CC > 20 → HIGH |
| Maintainability Index (MI) | `radon mi` | ≥ 65 (A/B grade) | MI < 40 → LOW |
| Cognitive Complexity | `lizard` | ≤ 15 | — |
| Halstead Volume | `radon hal` | ≤ 1000 | > 2000 → LOW |
| Bandit Security Score | `bandit` | HIGH severity = 0 | HIGH → EXIT |
| Import Security | custom AST | non-whitelisted = BLOCK | → HIGH |

**Severity decision:**
- `HIGH`: Bandit HIGH finding OR CC > 20 OR non-whitelisted import → STOP
- `LOW`: Warnings present but not critical → skip Node 2, go to Node 3
- `OK`: All thresholds passed → continue to Node 2

---

### Node 2 — Sandbox & Execution

Purpose: Execute the code in an isolated environment and measure correctness and performance metrics.

**A) Correctness Metrics**

| Metrik | Threshold |
|--------|-----------|
| Test Pass Rate (pytest + LCB test case'leri) | Easy ≥ %95 / Medium ≥ %80 / Hard ≥ %60 |
| Edge Case Coverage | ≥ %70 |
| Output Determinism (5 tekrar) | %100 |
| Exception Handling Coverage | ≥ %80 |

**B) Performans Metrikleri**

| Metric | Tool | Threshold |
|--------|------|-----------|
| Peak Memory (RSS) | `tracemalloc` + `psutil` | ≤ referans × 2 |
| CPU Time | `time.process_time()` | ≤ referans × 1.5 |
| Wall Clock Time | `time.perf_counter()` | LCB limit × %80 |
| Memory Leak | RSS delta after N iterations | Δ ≤ 5% |

**C) Reliability**

| Metrik | Hedef |
|--------|-------|
| Flakiness Rate (5x tekrar) | %0 |
| Timeout behavior | Graceful exit |
| Sandbox escape attempt | 0 tolerans |

**Sandbox security:**
- AST-based import whitelist check (before execution)
- `subprocess` ile izole process
- `resource.setrlimit` ile RAM limiti (512 MB)
- Timeout via `subprocess.TimeoutExpired` (10 seconds)

---

### Node 3 — Judgement LLM

Purpose: Synthesize Node 1 and Node 2 metrics to produce a 100-point score
and write an improved system prompt on failure.

**Rubric (100 puan)**

| Dimension | Weight | Source |
|-------|---------|--------|
| Accuracy | 25p | Node 2: test pass rate normalized |
| Efficiency | 20p | Node 2: CPU + Memory composite |
| Quality | 20p | Node 1: MI + CC composite |
| Security | 15p | Node 1: Bandit severity weighted |
| Algorithm Choice | 10p | Judgement LLM evaluation |
| Error Handling | 10p | Judgement LLM evaluation |

**Node 3 Output Schema (JSON)**

```json
{
  "total_score": 74.5,
  "pass": false,
  "score_breakdown": {
    "accuracy": 18.75,
    "efficiency": 16.0,
    "quality": 14.0,
    "security": 15.0,
    "algorithm_choice": 6,
    "error_handling": 5
  },
  "failure_reasons": ["low_pass_rate", "cc_too_high"],
  "improved_system_prompt": "..."
}
```

`improved_system_prompt` completely replaces the previous system prompt — no accumulation.
The model does not change; the same active model runs again.

---

## 4. State Schema

```python
class PipelineState(TypedDict):
    # Routing
    active_model: Literal["M1", "M2", "M3"]
    iteration: int
    max_iterations: int          # = 10

    # Input
    problem: str
    difficulty: Literal["easy", "medium", "hard"]
    generated_code: str

    # Node 1
    node1_cc: float
    node1_mi: float
    node1_bandit_high: bool
    node1_halstead: float
    node1_severity: Literal["LOW", "OK", "HIGH"]

    # Node 2
    node2_pass_rate: float
    node2_cpu_ms: float
    node2_ram_mb: float
    node2_test_diversity_score: float

    # Node 3
    node3_total_score: float
    node3_pass: bool
    node3_failure_reasons: list[str]
    node3_improved_prompt: str

    # Log
    memory_log: list[dict]
    final_status: Literal["PASS", "STOP_HIGH", "STOP_MAX", "PENDING"]
```

---

## 5. Conditional Edge Logic

```python
# Severity Gate (after Node 1)
def severity_gate(state) -> str:
    if state["node1_severity"] == "HIGH":
        return "stop_end"
    elif state["node1_severity"] == "LOW":
        return "to_node3"      # Node 2 atlanır
    return "to_node2"

# Iteration Gate (after Node 3)
PASS_THRESHOLDS = {"easy": 80, "medium": 70, "hard": 60}

def iteration_gate(state) -> str:
    threshold = PASS_THRESHOLDS[state["difficulty"]]
    if state["node3_total_score"] >= threshold and state["node3_pass"]:
        return "write_memory"
    elif state["iteration"] >= state["max_iterations"]:
        return "stop_end"
    return "loop_to_codegen"
```

---

## 6. memory.md Format

```markdown
| iter | model | model_name | q_id | diff | cc  | mi   | severity | pass_rt | score | status    | total_ms | total_tok | json |
|------|-------|------------|------|------|-----|------|----------|---------|-------|-----------|----------|-----------|------|
|    1 | M1    | gpt-oss-20b | q001 | easy | 3.0 | 72.1 | OK       |    0.85 |  68.0 | LOOP      |     7245 |      2550 | ...  |
|    2 | M1    | gpt-oss-20b | q001 | easy | 2.0 | 81.4 | OK       |    1.00 |  84.0 | PASS      |     6100 |      2300 | ...  |
|    1 | M1    | gpt-oss-20b | q002 | hard | 8.0 | 45.0 | HIGH     |    0.00 |   0.0 | STOP_HIGH |     3142 |       730 | ...  |
```

Rows are independent per model. The iteration counter resets at the start of each model.

---

## 7. Hardware Requirements

The three models are not held in memory simultaneously; they are loaded and run sequentially.
**Dominant model: Qwen3-Coder-Next (~46 GB Q4)** — determines hardware selection.

| Model | Boyut (GGUF Q4) | Min. VRAM | Notlar |
|-------|----------------|-----------|--------|
| gpt-oss-20b (M1) | ~11 GB | 24 GB | Runs even on RTX 4090 |
| Qwen3-Coder-Next (M2) | ~46 GB | **80 GB** | Dominant; requires A100 80GB |
| GLM-4.7-Flash (M3) | ~18 GB | 24 GB | RTX 4090'da 110 tok/s |
| Qwen3.5-35B-A3B (Judgement) | ~24 GB | 24 GB | Independent; runs every iteration |

**Vast.ai instance selection:** A100 80GB, 64 GB RAM, 300 GB SSD — ~$0.80/hr.
50 soru benchmark tahmini: ~$11 toplam.

---

## 8. Technology Stack

| Katman | Teknoloji |
|--------|-----------|
| Orchestration | LangGraph 0.2+ |
| LLM Inference | llama.cpp (OpenAI-compat. server) |
| LLM Client | openai Python SDK |
| Codegen M1 | gpt-oss-20b Unsloth GGUF (Apache 2.0) |
| Codegen M2 | Qwen3-Coder-Next Unsloth GGUF |
| Codegen M3 | GLM-4.7-Flash Unsloth GGUF |
| Judgement | Qwen3.5-35B-A3B Unsloth GGUF |
| Statik Analiz | radon, lizard, bandit |
| Execution | subprocess + tracemalloc + psutil |
| Test Runner | pytest |
| Veri Seti | LiveCodeBench (HuggingFace datasets) |
| Loglama | loguru |
| Ortam | Python 3.11, Ubuntu 22.04 (Vast.ai A100 80GB) |
