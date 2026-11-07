# CLAUDE.md — LLM-as-a-Judge for Iterative Code Refinement

This file is written for every AI tool (Claude, Copilot, etc.) involved in the project
to quickly understand it. Refer to this file when navigating the codebase.

---

## Project Purpose

A LangGraph pipeline that solves LiveCodeBench (LCB) Python problems iteratively: a generator
writes code, three analysis nodes produce evidence (static analysis, sandboxed execution on the
public tests, rubric), and an **LLM judge makes the final ACCEPT/REJECT decision** and writes the
feedback (`improved_prompt`) for the next attempt. Delivered code is graded offline on the
hidden tests. The judge-as-decider premise is central to the thesis: deterministic metrics are
evidence and a hallucination guard, never the decision itself.

Developed for a **master's thesis**; an ELECO 2026 paper (Turkish, 5 pages) reports the
OpenRouter main experiment and the gpt-oss-120b judge runs. Paper sources: `~/Desktop/workspace/eleco2026-bildiri/` (outside this repo).

---

## Experiments

| Setting | Generator | Judge | Problems | Backend |
|---|---|---|---|---|
| **Main (contamination-free)** | DeepSeek-V3 (2024-12, DeepInfra fp4) | Qwen2.5-72B-Instruct (2024-09, Novita bf16) | LCB v6 2025-01..2025-04, seed-42 sample of 150 (of 182) | OpenRouter |
| Judge effect (2026-10-03, Y only) | DeepSeek-V3 / Llama-3.1-8B (2024-07, DeepInfra fp8) | gpt-oss-120b (2025-08, DeepInfra) | same 150 | OpenRouter |
| Secondary (local, not in the paper) | Qwen3.5-9B (2026-02) | same model | seeded LCB sample of 150, first 50 reported | Mac mini M4, llama.cpp |

- All models are open-weight (DeepSeek Model License, Qwen License, Apache-2.0 for Qwen3.5); state this in the paper.
- LCB stopped at v6 (problems up to 2025-04). Newer models may have trained on it, so the main
  setting follows LCB's post-cutoff protocol; the contamination risk of newer models must
  always be stated (paper + docs). Details: `docs/dataset.md` §10.
- Main protocol (`docs/ablation.md` §7): ≤ 5 generations, no trend extension; V0 = 5 samples per
  problem with the first-iteration prompt (`benchmark/sample_baseline.py`); metric = pass@1 of
  the delivered code; budget-matched baselines (filt@k, pass@k), cost, judge accuracy and
  feedback-leak counts (`benchmark/openrouter_results.py`). Designs: Y (`JUDGE_DECISION=verdict`,
  proposed) and P (`rubric`, comparison). Budget cap $5 (`API_BUDGET_USD`).

---

## Environments

| `ENV` | Hardware | Backend | Settings |
|---|---|---|---|
| `openrouter` | none (API) | OpenRouter | `config/openrouter.env` (tracked) + key from `.env.openrouter` (git-ignored), loaded by `scripts/run_openrouter.sh` |
| `local` | Mac mini M4 16 GB | llama.cpp, Metal | `.env` → `.env.local`; one server on :8001 (qwen3.5-9b) for all roles |
| `cloud` | rented GPU (Vast.ai), optional | llama.cpp, CUDA | `.env` → `.env.cloud`; `CLOUD_README.md` §B |

All backends are OpenAI-compatible; nodes call `models/client.chat()`. With `ENV=openrouter`
every call is streamed, retried with backoff, pinned to one provider endpoint
(`allow_fallbacks=false`, `max_price`), and logged to `outputs/api_usage.jsonl`; `models/budget.py`
refuses a call that could push total spend past the cap. Optional Langfuse tracing
(`models/tracing.py`) turns on when `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` are set (keys only in
`.env.openrouter`); it records every call, iteration events and judge/hidden-test scores.

Local model flags: Qwen3.5-9B needs `--jinja --cache-ram 0` (reasoning model; the default 8 GiB
prompt cache causes Metal OOM). The client falls back to the reasoning text if `content` is empty.

---

## Pipeline Summary

```
START → codegen (active model; on loop: judge's improved_prompt + its own previous code)
  → Node 1: Static Analysis (CC · MI · Halstead · Bandit · mypy/ruff · import whitelist)
  → Severity Gate ── HIGH → log_severity_stop → END
                  ── LOW  → strip_comments → Node 3   (Node 2 skipped)
                  ── OK   ↓
  → Node 2: Sandbox on public tests (local LCB-compatible evaluator; cpu_ms · ram_mb)
  → strip_comments (bias prevention)
  → Node 3: Judge — rubric + tests as evidence → ACCEPT/REJECT verdict + improved_prompt
  → log_iteration (memory.md + iteration JSON)
  → Iteration Gate ── PASS     → write_memory_final → next generator → END
                   ── STOP_MAX → END
                   ── LOOP     → codegen (same model, next iteration)
```

PASS = `node3_pass`: all public tests pass AND judge ACCEPT (verdict design); without a usable
verdict, rubric threshold Easy ≥ 80 / Medium ≥ 70 / Hard ≥ 60. The judge must not put code in
its feedback.

---

## Directory Structure

```
project/
├── LICENSE                    ← MIT license (Copyright 2026 Mehmet Sergen Ergen)
├── CITATION.cff               ← Citation metadata (software + conference paper)
├── CLAUDE.md / README.md / CLOUD_README.md (English public README; Turkish runbook: §A, §B)
├── main.py                    ← pipeline entry (MAX_PROBLEMS → SKIP_QIDS → SHARD_* filters)
├── config/
│   ├── pipeline.py            ← settings (all env vars), thresholds.py, models.py (registries)
│   └── openrouter.env         ← main-experiment settings (no secrets)
├── pipeline/                  ← state.py, graph.py (V1/V2/V3), gates.py, router.py
├── nodes/                     ← codegen, static_analysis, sandbox, cleaner, judgement
├── metrics/                   ← complexity, security, linting, scoring (100-point rubric)
├── sandbox/                   ← lcb_eval.py (local evaluator), runner.py, whitelist.py
├── models/                    ← client.py (chat), budget.py (spend ledger), prompts.py, tracing.py (optional Langfuse)
├── dataset/                   ← loader.py (date window, seed 42 sample), splitter, edge_cases, verify
├── memory/logger.py           ← memory.md, iteration JSON, run_info_<run_id>.json
├── benchmark/
│   ├── sample_baseline.py     ← V0, n samples/problem (main experiment)
│   ├── openrouter_results.py  ← main-experiment analysis
│   ├── judge_replay.py        ← offline re-judging of recorded attempts with another judge
│   ├── official_regrade.py    ← re-grade recorded code with LCB's official lcb_runner, agreement with the local evaluator
│   ├── error_analysis.py      ← failure categories of delivered code, judge accept/reject analysis (no calls, no re-execution)
│   ├── paper_results.py       ← ELECO paper numbers/tables/figure (local runs)
│   ├── evaluator.py           ← ablation runner (V1/V2/V3)
│   ├── sample50_qids.txt      ← first 50 qids of the local sample
│   └── legacy/                ← old V0–V3 JSON toolkit (not comparable; kept for reference)
├── scripts/
│   ├── run_openrouter.sh      ← API-based runner (smoke, baseline, pipeline)
│   ├── start_model.sh         ← Local model server launcher
│   ├── local_setup.sh         ← Local environment setup
│   ├── cloud_setup.sh         ← Vast.ai cloud GPU setup
│   └── export_release_logs.py ← public log archive release/…zip (strips LCB content; docs/memory-logging.md)
├── tests/                     ← 21 test files + conftest.py (pytest -q: 301 passed, 1 skipped)
├── docs/                      ← see index below
├── judge_training/            ← future work: fine-tuned judge
├── release/ (git-ignored)     ← coderefiner-eleco2026-logs.zip (built by export_release_logs.py)
└── outputs/ (git-ignored)     ← results/ (local), openrouter/ (main), api_usage.jsonl
```

### Docs index

| Doc | Status |
|---|---|
| `project-plan.md` | Work plan and current status |
| `architecture.md`, `graph.md` | System overview; graph, state, gates |
| `config.md`, `setup.md` | All settings; installation |
| `dataset.md` | LCB loading, date window, contamination (§10) |
| `codegen-models.md`, `node1.md`, `node2.md`, `node3.md` | Components (chat client, judge verdict, no-code rule) |
| `memory-logging.md` | Output files |
| `ablation.md` (§7 main protocol), `analysis.md` | Experiment design and analysis |
| `main-results.md` | Main-experiment results (2026-10-02): V0 vs Y/P, refinement gain, budget match, judge accuracy, ablation; §11 gpt-oss-120b judge runs (2026-10-03) |
| `testing.md` | Test suite |
| `future-work.md`, `judge-model-training.md`, `sensitivity.md` | Plans |
| `metrics.md` | Metric thresholds and rubric reference |
| `paper.md`, `cloud-comparison.md`, `unsloth-model-comparison.md` | Historical (superseded; see banner in each) |

---

## Key Architecture Decisions

1. **Judge decides.** The judge's justified ACCEPT/REJECT is final; passing every public test is
   a precondition. A REJECT of public-passing code must name a concrete input or a complexity
   violation. Design P (rubric threshold) exists only as the comparison.
2. **Leakage-free evaluation.** The loop sees public tests only; hidden tests are used offline.
   One pinned evaluator (`LCB_EVALUATOR=local`) grades everything.
3. **Early exit — Severity Gate.** Bandit HIGH, non-whitelisted import or CC > 20 → stop.
   Unparseable code or poor MI/Halstead → LOW: skip the sandbox, go to the judge.
4. **Loop target — active model.** `improved_prompt` (rewritten each iteration, ≤ 2000 chars) goes
   back to the same generator together with its previous solution (LCB self-repair format,
   `FEEDBACK_INCLUDE_CODE=true`); no history beyond that one attempt accumulates.
5. **Termination.** `MAX_ITERATIONS=5`; trend extension off by default (the ELECO local runs used
   5 + up to 3).
6. **Generator and judge from different families** in the main setting (self-preference bias),
   and the judge may not write code (solution leak).

---

## Docs Update Rule

**When modifying code, always update the corresponding `docs/*.md` file in the same commit.**

| Modified file | Docs file to update |
|---|---|
| `memory/logger.py` | `docs/memory-logging.md` |
| `benchmark/sample_baseline.py`, `benchmark/evaluator.py` | `docs/ablation.md` |
| `benchmark/openrouter_results.py`, `benchmark/paper_results.py`, `benchmark/judge_replay.py`, `benchmark/official_regrade.py`, `benchmark/error_analysis.py` | `docs/analysis.md` |
| `pipeline/graph.py`, `pipeline/gates.py`, `pipeline/state.py` | `docs/graph.md` |
| `nodes/codegen.py`, `models/prompts.py`, `models/client.py`, `models/budget.py`, `models/tracing.py` | `docs/codegen-models.md` |
| `nodes/static_analysis.py`, `metrics/` | `docs/node1.md` |
| `nodes/sandbox.py`, `sandbox/` | `docs/node2.md` |
| `nodes/judgement.py`, `metrics/scoring.py` | `docs/node3.md` |
| `config/` (incl. `openrouter.env`) | `docs/config.md` |
| `dataset/` | `docs/dataset.md` |
| `scripts/run_openrouter.sh`, `scripts/*setup.sh` | `CLOUD_README.md`, `docs/setup.md` |
| `scripts/export_release_logs.py` | `docs/memory-logging.md` (Release archive), `README.md` |
| `tests/` | `docs/testing.md` |

Do not ask for permission — update docs automatically as part of the same change.
If the docs update would overwrite thesis-authored prose (design rationale, methodology explanation),
preserve the existing text and only add/update the affected code snippet or field list.

---

## Important Notes

- Never commit `.env`, `.env.local`, `.env.cloud` or `.env.openrouter` (the API key lives only in `.env.openrouter`).
- Paid runs: check spend with `scripts/run_openrouter.sh status`; never raise `API_BUDGET_USD` without the user's approval.
- Every reference in the paper/thesis must be verified against a primary source (no ghost references).
- Node 2 memory limit uses `resource.setrlimit` — Linux only; skipped on macOS.
- `datasets<3.0` is required; radon/bandit are invoked via `sys.executable -m`.
