# CLAUDE.md — LLM-as-a-Judge for Iterative Code Refinement

This file is written for every AI tool (Claude, Copilot, etc.) involved in the project
to quickly understand it. Refer to this file when navigating the codebase.

---

## Project Purpose

A LangGraph-based pipeline that routes Easy / Medium / Hard Python problems
from the LiveCodeBench (LCB) dataset to two LLMs (M1, M2) sequentially,
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
| `cloud` | Vast.ai GPU (≥32GB; A100 80GB no longer confirmed necessary — see Models section) | CUDA | `ln -sf .env.cloud .env` |

**Local** (ENV=local): qwen3.5-9b (Unsloth UD-Q4_K_XL GGUF, ~6 GB) on port 8001 — used for all roles (M1/M2 + judgement).
**Cloud** (ENV=cloud): Full model set on separate ports (see Models section).

The `.env` file is never committed to git. Create the appropriate env file and symlink
as the first setup step (see `docs/setup.md` section A6 for local, B6 for cloud).

---

## Models

| ID | Model | Format | Port | Min. Memory | Notes |
|----|-------|--------|------|-------------|-------|
| M1 | Qwen3.5-9B (Unsloth GGUF) | GGUF | 8001 | ~6 GB | Medium-scale anchor; reasoning model; `--jinja --cache-ram 0` required |
| M2 | Qwen2.5-Coder-1.5B-Instruct (GGUF) | GGUF | 8002 | ~1 GB | Small-scale anchor; standard quant (no UD variant) |
| QWEN | Qwen3.5-35B-A3B (Unsloth GGUF) | GGUF | 8004 | ~24 GB | Judgement only; independent model |

All models expose an **OpenAI-compatible API** (`/v1/chat/completions`).
The pipeline connects to all via the `openai` Python library.
Codegen order is fixed: M1 → M2. Judgement calls QWEN every iteration.

In local mode, all roles (M1/M2/QWEN) route to port 8001 (qwen3.5-9b). Note: qwen3.5-9b returns reasoning_content (chain-of-thought) which the client falls back to if content is empty.

---

## Pipeline Summary

```
START
  ↓
Sequential Router (M1 → M2)
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
strip_comments          ← removes comments before judgement (bias prevention)
  ↓
Node 3: Judgement LLM  (rubric as evidence + ACCEPT/REJECT verdict + improved_prompt)
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
│   ├── node3.md               ← judgement LLM, rubric, improved_prompt
│   ├── graph.md               ← LangGraph graph, PipelineState, gates, cleaner
│   ├── memory-logging.md      ← run_info.json, memory.md, iteration JSON
│   ├── testing.md             ← 10 test files, pilot checklist, benchmark
│   ├── ablation.md            ← V0–V4 variants, baseline.py, analysis
│   ├── sensitivity.md         ← post-hoc sensitivity analysis plan (run after cloud)
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
│   └── judgement.py            ← Node 3
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

1. **Sequential model execution**: M1 → M2 run in order. When a model gets PASS,
   results are written to memory.md and the next model starts. No parallel execution.
   Both are same-family Qwen models isolating model size as the controlled variable.
   Largest concurrent memory footprint: ~M1 (6 GB) + QWEN (24 GB) ≈ 30 GB; likely reduces hardware requirement — may allow smaller than A100 80GB, pending verification with actual Vast.ai pricing.

2. **Early exit — Severity Gate**: If Node 1 detects Bandit HIGH severity or CC > 20,
   the code is stopped before reaching the sandbox. On LOW severity (minor warnings),
   Node 2 is skipped, going directly to Node 3.

3. **Loop target — active model**: The `improved_prompt` produced by Node 3
   is sent back to the currently active model (M1 or M2). The model does not change.

4. **Termination criteria**:
   - `max_iterations = 10` (independent counter per model)
   - PASS threshold: Easy ≥ 80 / Medium ≥ 70 / Hard ≥ 60 (out of 100)

5. **Environment via symlink** — no env flag needed at runtime:
   - `ln -sf .env.local .env` → Mac Mini M4 local debug (done once)
   - `ln -sf .env.cloud .env`  → Vast.ai production (done once)
   - Then just: `python main.py`

6. **Model-specific required parameters**:
   - Qwen3.5-9B (M1, cloud + local): `--jinja --cache-ram 0` — Qwen3.5-9B is a reasoning model that emits reasoning_content before content (client falls back if content is empty); the default 8 GiB prompt cache causes Metal GPU OOM when combined with the model's resident weights on local hardware.
   - Qwen2.5-Coder-1.5B-Instruct (M2, cloud): No special flags required; only standard GGUF quants available (no Unsloth UD variant).

---

## Docs Update Rule

**When modifying code, always update the corresponding `docs/*.md` file in the same commit.**

| Modified file | Docs file to update |
|---|---|
| `memory/logger.py` | `docs/memory-logging.md` |
| `benchmark/evaluator.py` | `docs/ablation.md` |
| `benchmark/analysis.py` | `docs/analysis.md` |
| `benchmark/baseline.py` | `docs/ablation.md` |
| `pipeline/graph.py`, `pipeline/gates.py` | `docs/graph.md` |
| `nodes/codegen.py`, `models/prompts.py` | `docs/codegen-models.md` |
| `nodes/static_analysis.py`, `metrics/` | `docs/node1.md` |
| `nodes/sandbox.py`, `sandbox/` | `docs/node2.md` |
| `nodes/judgement.py`, `metrics/scoring.py` | `docs/node3.md` |
| `config/` | `docs/config.md` |
| `dataset/` | `docs/dataset.md` |
| `pipeline/state.py` | `docs/graph.md` |

Do not ask for permission — update docs automatically as part of the same change.
If the docs update would overwrite thesis-authored prose (design rationale, methodology explanation),
preserve the existing text and only add/update the affected code snippet or field list.

---

## Important Notes

- All models run locally on llama.cpp. Internet is only needed for model download.
- Never commit `.env`, `.env.local`, or `.env.cloud` to git.
- Copy `outputs/` to your local machine regularly when running on Vast.ai.
- Node 2 sandbox uses `resource.setrlimit` — Linux only (Vast.ai runs Ubuntu).
- Each phase has a dedicated `docs/*.md` file. Read the relevant doc if stuck.
- Hardware requirement on Vast.ai: largest concurrent memory ≈ 30 GB (M1 ~6 GB + QWEN ~24 GB), likely no longer requires A100 80GB but re-verification with current Vast.ai pricing is needed before finalizing a specific tier.
