# architecture.md — Pipeline Mimarisi

> Son güncelleme: 2026-10-02. Ayrıntılar: `graph.md` (graf, durum, kapılar), `node1.md`,
> `node2.md`, `node3.md`, `codegen-models.md`, `config.md`.

## 1. Overview

The pipeline is a cyclic LangGraph state graph. For each problem, the active generator writes
code; Node 1 and Node 2 produce deterministic evidence; Node 3 — the LLM judge — makes the final
ACCEPT/REJECT decision and writes the instructions for the next attempt. A rejected attempt is
retried with the same generator, up to `MAX_ITERATIONS` (5). The loop sees only the public
example tests; delivered code is graded offline on all hidden tests.

Generators run once per problem in the order of `PIPELINE_MODELS` (default: only `M1`).

---

## 2. Pipeline Flow

```
START
  │
  ▼
codegen ── system prompt (+ judge's improved_prompt) + problem (+ previous solution on repair)
  │
  ▼
NODE 1: Static analysis ── CC · MI · Halstead · Bandit · mypy/ruff · import whitelist
  │
  ▼
Severity gate ──── HIGH ──→ log_severity_stop ──→ END (STOP_HIGH)
  │        └────── LOW  ──→ strip_comments ──┐   (Node 2 skipped)
  │ OK                                       │
  ▼                                          │
NODE 2: Sandbox on public tests              │
  pass rate · failed cases · cpu_ms · ram_mb │
  │                                          │
  ▼                                          │
strip_comments ◄─────────────────────────────┘
  │
  ▼
NODE 3: Judge ── rubric + tests as evidence → verdict (ACCEPT/REJECT) + improved_prompt
  │
  ▼
log_iteration ── memory.md row + iteration JSON
  │
  ▼
Iteration gate ── PASS (node3_pass)          → write_memory_final → next generator / END
               ── STOP_MAX (iteration = max) → END
               ── LOOP                       → codegen (same generator)
```

---

## 3. Node Details

### Node 1 — Static analysis (`node1.md`)

| Metric | Tool | Issue reported | Severity |
|---|---|---|---|
| Cyclomatic complexity | radon | CC > 10 | CC > 20 → HIGH |
| Maintainability index | radon | MI < 65 | MI < 40 → LOW |
| Halstead volume | radon | > 1000 | > 2000 → LOW |
| Bandit | bandit | medium findings | any HIGH → HIGH |
| Imports | AST whitelist | — | non-whitelisted → HIGH |
| Types / lint | mypy, ruff | findings listed | — |
| Parse | ast | "failed to parse" | unparseable → LOW |

HIGH stops the attempt (security risk or excessive complexity). Unparseable code is LOW, not
HIGH: it is usually a cut-off generation, so it goes to the judge and the model gets another try.

### Node 2 — Sandbox (`node2.md`)

Each public test runs in its own subprocess (timeout `SANDBOX_TIMEOUT_SEC`). Grading follows
LCB's rules with the pinned local evaluator (`LCB_EVALUATOR=local`): stdin problems compare
stdout line by line (numeric tolerance), LeetCode problems call the starter method with JSON
arguments. Up to `SANDBOX_MAX_FAILED_CASES` failures (input, expected, got, error) go to the
judge; CPU time and peak RAM are measured. Memory limit via `resource.setrlimit` (Linux only).

### Node 3 — Judge (`node3.md`)

Evidence: 100-point rubric — 80 points deterministic (accuracy 25 from public pass rate,
efficiency 20 from CPU/RAM, quality 20 from MI/CC, security 15 from Bandit) and 20 from the judge
(algorithm choice 10, error handling 10) — plus failed test cases, metric issues and the
problem's constraints section.

Decision (`JUDGE_DECISION=verdict`, default):
- precondition: every public test passes; otherwise the attempt cannot pass;
- the judge's ACCEPT/REJECT is final; a REJECT of public-passing code must cite a concrete input
  allowed by the constraints or an infeasible complexity — style or metrics alone are not grounds;
- no usable verdict → rubric threshold (Easy 80 / Medium 70 / Hard 60).

`JUDGE_DECISION=rubric` (design P, comparison) uses the threshold only.

Feedback: `improved_prompt` is rewritten from scratch every iteration (≤ 2000 chars), cites
failing tests and metric values, and must not contain code or pseudo-code.

---

## 4. State and Gates

Full `PipelineState` field list: `graph.md`. Gate logic (`pipeline/gates.py`):

```python
def severity_gate(state):           # after Node 1
    if state["node1_severity"] == "HIGH": return "stop_high"   # → log_severity_stop
    if state["node1_severity"] == "LOW":  return "to_node3"    # → strip_before_judge
    return "to_node2"                                          # → sandbox

def iteration_gate(state):          # after log_iteration
    if state["node3_pass"]:                       return "write_memory_final"
    if state["iteration"] >= state["max_iterations"]:
        if settings.trend_extension_enabled and _should_extend(state, threshold):
            return "loop_to_codegen"              # off by default
        return "stop_max"
    return "loop_to_codegen"
```

`graph.md` has the full wiring, including variants V1 (no Node 1) and V2 (no Node 2).

---

## 5. Backends

| `ENV` | Generator | Judge | Notes |
|---|---|---|---|
| `openrouter` (main) | DeepSeek-V3, DeepInfra fp4 | Qwen2.5-72B-Instruct, Novita bf16 | Streamed, retried, provider-pinned, $5 cap |
| `local` (ELECO paper) | Qwen3.5-9B UD-Q4_K_XL, llama.cpp :8001 | same server | Mac mini M4 16 GB, ~16 tok/s |
| `cloud` (optional) | Qwen3.5-9B on a rented GPU | same / Qwen3.5-35B-A3B | `CLOUD_README.md` §B |

All are OpenAI-compatible; nodes call `models/client.chat()`.

---

## 6. Technology Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph |
| Inference | OpenRouter API; llama.cpp server (local/cloud) |
| Client | `openai` Python SDK (`models/client.py`), spend ledger (`models/budget.py`) |
| Static analysis | radon, lizard, bandit, mypy, ruff |
| Execution | subprocess, psutil, `resource` |
| Data | LiveCodeBench `code_generation_lite` (release_v6), `datasets<3.0` |
| Statistics | statsmodels (McNemar, Wilson), scipy (Wilcoxon) |
| Logging | loguru; memory.md + per-iteration JSON |
| Runtime | Python 3.11 (macOS / Ubuntu) |
