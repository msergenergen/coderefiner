# CLAUDE.md — LLM-as-a-Judge for Iterative Code Refinement

This file is written for every AI tool (Claude, Copilot, etc.) involved in the project
to quickly understand it. Refer to this file when navigating the codebase.

---

## Project Purpose

A LangGraph-based pipeline that routes Easy / Medium / Hard Python problems
from the LiveCodeBench (LCB) dataset to three LLMs (M1, M2, M3) sequentially,
passes generated code through three analysis nodes, and automatically
improves failing outputs.

This system is being developed as part of a **master's thesis**.
For empirical evaluation, all run results are written to `outputs/memory.md`
and compared against a single-shot baseline.

---

## Environment

Active environment is determined by the `.env` symlink in the project root.

| Environment | Hardware | Backend | `.env` symlink |
|-------------|----------|---------|----------------|
| `local` | Mac Mini M4 16GB | Apple Metal | `ln -sf .env.local .env` |
| `cloud` | Vast.ai A100 80GB | CUDA | `ln -sf .env.cloud .env` |

**Local** (ENV=local): Qwen2.5-Coder-7B on port 8001 — used for all roles (M1/M2/M3 + judgment).
**Cloud** (ENV=cloud): Full model set on separate ports (see Models section).

The `.env` file is never committed to git. Create the appropriate env file and symlink
as the first setup step (see `docs/setup.md` section A6 for local, B6 for cloud).

---

## Models

| ID | Model | Format | Port | Min. Memory | Notes |
|----|-------|--------|------|-------------|-------|
| M1 | gpt-oss-20b (Unsloth GGUF) | GGUF | 8001 | ~11 GB | Apache 2.0; `--jinja` required |
| M2 | Qwen3-Coder-Next (Unsloth GGUF) | GGUF | 8002 | ~46 GB | Dominant model; determines hardware |
| M3 | GLM-4.7-Flash (Unsloth GGUF) | GGUF | 8003 | ~18 GB | `--repeat-penalty 1.0` required |
| QWEN | Qwen3.5-35B-A3B (Unsloth GGUF) | GGUF | 8004 | ~24 GB | Judgment only; independent model |

All models expose an **OpenAI-compatible API** (`/v1/chat/completions`).
The pipeline connects to all via the `openai` Python library.
Codegen order is fixed: M1 → M2 → M3. Judgment calls QWEN every iteration.

In local mode, all roles (M1/M2/M3/QWEN) route to port 8001 (Qwen2.5-Coder-7B).

---

## Pipeline Summary

```
START
  ↓
Sequential Router (M1 → M2 → M3)
  ↓
Code Generation      ← improved_prompt fed back here on loop
  ↓
Node 1: Static Analysis  (CC · MI · Bandit · Halstead)
  ↓
Severity Gate
  ├─ HIGH  → log_severity_stop → END
  ├─ LOW   → strip_comments → Node 3  (Node 2 skipped)
  └─ OK    ↓
Node 2: Sandbox & Execution  (lcb_runner correctness + cpu_ms · ram_mb)
  ↓
strip_comments          ← removes comments before judgment (bias prevention)
  ↓
Node 3: Judgment LLM  (100-point rubric + improved_prompt)
  ↓
log_iteration           ← writes to memory.md + iterations/JSON
  ↓
Iteration Gate
  ├─ PASS      → write_memory_final → next model → END
  ├─ STOP_MAX  → stop_max → END
  └─ LOOP      → Code Generation (same model, next iteration)
```

---

## Directory Structure

```
project/
├── CLAUDE.md                  ← this file
├── docs/                      ← full phase documentation
│   ├── project-plan.md        ← thesis work plan + status indicators
│   ├── architecture.md        ← system overview
│   ├── setup.md               ← Mac Mini + Vast.ai installation
│   ├── config.md              ← ENV flag, model registry, thresholds
│   ├── dataset.md             ← LCB loading, all_tests, edge cases
│   ├── codegen-models.md      ← models/client.py, prompts.py
│   ├── node1.md               ← static analysis: CC, MI, Halstead, Bandit
│   ├── node2.md               ← two-phase sandbox: lcb_runner + runtime
│   ├── node3.md               ← judgment LLM, rubric, improved_prompt
│   ├── graph.md               ← LangGraph graph, PipelineState, gates, cleaner
│   ├── memory-logging.md      ← run_info.json, memory.md, iteration JSON
│   ├── testing.md             ← 10 test files, pilot checklist, benchmark
│   ├── ablation.md            ← V0–V4 variants, baseline.py, analysis
│   ├── analysis.md            ← TSR, McNemar, hypothesis validation
│   ├── future-work.md         ← research directions
│   ├── cloud-comparison.md    ← Vast.ai vs RunPod vs Modal
│   └── unsloth-model-comparison.md  ← LCB scores, memory, cost table
├── pipeline/
│   ├── state.py               ← PipelineState TypedDict
│   ├── graph.py               ← LangGraph graph definition
│   ├── router.py              ← sequential model router
│   └── gates.py               ← severity_gate, iteration_gate
├── nodes/
│   ├── codegen.py             ← code generation node
│   ├── static_analysis.py     ← Node 1
│   ├── cleaner.py             ← strip_comments_node
│   ├── sandbox.py             ← Node 2
│   └── judgment.py            ← Node 3
├── metrics/
│   ├── complexity.py          ← radon + lizard
│   ├── security.py            ← bandit
│   └── scoring.py             ← 100-point rubric
├── sandbox/
│   ├── lcb_eval.py            ← lcb_runner wrapper + fallback
│   ├── runner.py              ← runtime metrics (cpu_ms, ram_mb)
│   └── whitelist.py           ← allowed imports
├── models/
│   ├── client.py              ← OpenAI-compatible client
│   └── prompts.py             ← four-layer codegen system prompt
├── dataset/
│   ├── loader.py              ← LCB loading + all_tests merge
│   ├── splitter.py            ← format_problem, dataset_stats
│   ├── edge_cases.py          ← rule-based edge case generation
│   └── verify.py              ← installation verification script
├── memory/
│   └── logger.py              ← write_run_info, log_iteration_node, read_memory
├── benchmark/
│   ├── evaluator.py           ← full benchmark runner (V1–V4 variants)
│   ├── baseline.py            ← single-shot baseline runner (V0)
│   └── analysis.py            ← TSR, McNemar, hypothesis validation
├── tests/
│   ├── conftest.py
│   ├── test_cleaner.py
│   ├── test_metrics.py
│   ├── test_gates.py
│   ├── test_lcb_eval.py
│   ├── test_runtime.py
│   ├── test_edge_cases.py
│   ├── test_dataset.py
│   ├── test_logger.py
│   ├── test_nodes.py
│   ├── test_prompts.py
│   └── test_baseline.py
├── outputs/
│   ├── run_info.json          ← auto-generated per run
│   ├── memory.md              ← auto-generated, one row per iteration
│   └── results/               ← JSON benchmark results
├── scripts/
│   └── start_model.sh         ← start a single llama.cpp server
├── config/
│   ├── __init__.py            ← from config import settings, thresholds, model_registry
│   ├── pipeline.py            ← max_iter, sandbox, LCB settings
│   ├── thresholds.py          ← Node 1/2/3 thresholds
│   └── models.py              ← model endpoint, name, temperature
├── main.py                    ← pipeline entry point
├── requirements.txt           ← Python dependencies
├── .env                       ← symlink → .env.local or .env.cloud
├── .env.local                 ← Mac Mini M4 config (never commit)
└── .env.cloud                 ← Vast.ai config (never commit)
```

---

## Key Architecture Decisions

1. **Sequential model execution**: M1 → M2 → M3 run in order. When a model gets PASS,
   results are written to memory.md and the next model starts. No parallel execution.
   Dominant model: Qwen3-Coder-Next (~46 GB) — minimum A100 80GB required on cloud.

2. **Early exit — Severity Gate**: If Node 1 detects Bandit HIGH severity or CC > 20,
   the code is stopped before reaching the sandbox. On LOW severity (minor warnings),
   Node 2 is skipped, going directly to Node 3.

3. **Loop target — active model**: The `improved_prompt` produced by Node 3
   is sent back to the currently active model (M1, M2, or M3). The model does not change.

4. **Termination criteria**:
   - `max_iterations = 10` (independent counter per model)
   - PASS threshold: Easy ≥ 80 / Medium ≥ 70 / Hard ≥ 60 (out of 100)

5. **Environment via symlink** — no env flag needed at runtime:
   - `ln -sf .env.local .env` → Mac Mini M4 local debug (done once)
   - `ln -sf .env.cloud .env`  → Vast.ai A100 production (done once)
   - Then just: `python main.py`

6. **Model-specific required parameters**:
   - gpt-oss-20b (M1): `--jinja` — required for Harmony chat template
   - GLM-4.7-Flash (M3): `--repeat-penalty 1.0` — otherwise causes repetition loops

---

## Important Notes

- All models run locally on llama.cpp. Internet is only needed for model download.
- Never commit `.env`, `.env.local`, or `.env.cloud` to git.
- Copy `outputs/` to your local machine regularly when running on Vast.ai.
- Node 2 sandbox uses `resource.setrlimit` — Linux only (Vast.ai runs Ubuntu).
- Each phase has a dedicated `docs/*.md` file. Read the relevant doc if stuck.
- gpt-oss-20b is not a split GGUF; it downloads as a single file.
- Minimum A100 80GB required on Vast.ai for Qwen3-Coder-Next (~46 GB).
