# LLM-as-a-Judge for Iterative Code Refinement

A LangGraph pipeline that generates Python solutions for LiveCodeBench (LCB) problems,
checks each attempt with static analysis and sandboxed execution on the **public** tests, and
lets an LLM judge make the final ACCEPT/REJECT decision and write the feedback for the next
attempt. Delivered code is graded offline on the **hidden** tests.

Master's thesis project (Kocaeli University); an ELECO 2026 paper is based on it.

---

## Experiments at a glance

| Setting | Generator | Judge | Problems | Where |
|---|---|---|---|---|
| **Main (contamination-free)** | DeepSeek-V3 (2024-12) | Qwen2.5-72B-Instruct (2024-09) | LCB v6, 2025-01..2025-04: seeded sample of 150 of 182 | OpenRouter API |
| Secondary (modern reasoning model) | Qwen3.5-9B (2026-02) | same model | LCB v6, seeded sample of 150 (first 50 reported) | Mac mini M4, llama.cpp |

Both main-setting models are open-weight (DeepSeek-V3: MIT code + DeepSeek Model License;
Qwen2.5-72B: Qwen License), as is the secondary Qwen3.5-9B.

Why two settings: LCB's last release (v6) ends with April 2025 problems, so newer models may
have been trained on them. LCB's own protocol evaluates a model only on problems released after
its cutoff; the main setting follows it. See [docs/dataset.md §10](docs/dataset.md).

**Protocol (main):** at most 5 generations per problem; V0 = 5 independent single-shot samples
with the pipeline's first-iteration prompt; metric = pass@1 of the delivered code (all hidden
tests pass); budget-matched baselines and cost are reported alongside. Full protocol:
[docs/ablation.md §7](docs/ablation.md). Results: [docs/main-results.md](docs/main-results.md).

---

## Pipeline

```
START → codegen (prompt + judge's improved_prompt and the previous solution on repair)
  → Node 1: static analysis (CC · MI · Halstead · Bandit · mypy/ruff · import whitelist)
  → severity gate ── HIGH → stop
                  ── LOW  → skip Node 2
                  ── OK   → Node 2: sandbox on public tests (pass rate, cpu_ms, ram_mb)
  → strip comments (bias prevention)
  → Node 3: judge — rubric + test results as evidence → ACCEPT / REJECT + improved_prompt
  → log iteration
  → iteration gate ── PASS (judge ACCEPT, all public tests pass) → end
                   ── STOP_MAX (5 attempts)                      → end
                   ── LOOP                                       → codegen
```

- The judge's verdict is final (`JUDGE_DECISION=verdict`). Precondition: every public test passes.
  A REJECT of public-passing code must cite a concrete input or a complexity violation; style or
  metrics alone are not grounds. Design P (`JUDGE_DECISION=rubric`) uses the 100-point rubric
  threshold instead and is the comparison design.
- The judge must not write code in its feedback (solution-leak guard; leaks are counted).
- The loop never sees hidden tests.

---

## Quick start

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m dataset.verify          # expects "Window 2025-01..2025-04: 182 problems"
pytest -q                         # 272 passed, 1 skipped
```

### Main experiment (OpenRouter)

```bash
cp .env.openrouter.example .env.openrouter     # paste OPENROUTER_API_KEY=...  (git-ignored)
scripts/run_openrouter.sh smoke                 # pipeline, 2 problems
scripts/run_openrouter.sh pilot 20              # baseline, 20 problems × 5 samples (floor check)
scripts/run_openrouter.sh baseline              # baseline, 150 × 5
scripts/run_openrouter.sh run                   # pipeline, 150 × {verdict, rubric}
scripts/run_openrouter.sh status                # progress + spend
python -m benchmark.openrouter_results --baseline <jsonl> --y-run-ids ... --p-run-ids ...
```

Optional tracing: put `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` in `.env.openrouter` and every
model call, iteration and judge verdict appears in Langfuse (`scripts/run_openrouter.sh langfuse-check`).

Settings: [config/openrouter.env](config/openrouter.env). Every API call is streamed, retried,
and logged to `outputs/api_usage.jsonl`; a call that could push total spend past
`API_BUDGET_USD` ($5) is never sent. Runbook: [CLOUD_README.md](CLOUD_README.md) §A.

### Secondary setting (local, Mac mini M4 16 GB)

```bash
bash scripts/local_setup.sh        # llama.cpp (Metal), venv, .env.local, Qwen3.5-9B GGUF (~6 GB)
bash scripts/start_model.sh        # qwen3.5-9b on :8001 (--jinja --cache-ram 0)
python main.py
python -m benchmark.paper_results --run-ids <run ids> --out <overleaf dir>
```

A rented-GPU (Vast.ai) variant with the full token budget is described in
[CLOUD_README.md](CLOUD_README.md) §B; it is optional.

---

## Configuration

Environment is chosen by `ENV`: `local` (one llama.cpp server), `cloud` (llama.cpp servers per
role), `openrouter` (hosted APIs). Local/cloud read `.env` (symlink to `.env.local` /
`.env.cloud`); OpenRouter runs read `config/openrouter.env` through the run script. All
variables: [docs/config.md](docs/config.md).

| Variable | Default | Meaning |
|---|---|---|
| `JUDGE_DECISION` | `verdict` | `verdict` (judge decides) / `rubric` (score threshold) |
| `MAX_ITERATIONS` | `5` | Generations per problem |
| `TREND_EXTENSION_ENABLED` | `false` | Up to 3 extra attempts on an upward trend (ELECO local runs: on) |
| `PIPELINE_MODELS` | `M1` | Generators to run per problem |
| `LCB_DATE_FROM` / `LCB_DATE_TO` | empty | Contest-date window (`YYYY-MM`) |
| `CODEGEN_MAX_TOKENS` / `JUDGE_MAX_TOKENS` | 6144 / 4096 | Output budgets (OpenRouter: 8192 / 4096) |
| `LCB_EVALUATOR` | `local` | Pinned LCB-compatible evaluator for all reported results |

---

## Analysis tools

| Script | Use |
|---|---|
| `benchmark/sample_baseline.py` | V0: n samples/problem, graded on public + hidden tests |
| `benchmark/openrouter_results.py` | Main experiment: V0 pass@1/pass@k/filt@k vs V3 (Y, P), paired tests, budget-matched comparison, judge accuracy, feedback leaks, cost |
| `benchmark/paper_results.py` | ELECO paper numbers, tables and figure from local runs |
| `benchmark/evaluator.py` | Ablation runner for variants V1 (no static analysis) / V2 (no execution) / V3 |
| `benchmark/legacy/` | Old V0–V3 JSON toolkit (different V0 prompt; not comparable) |

---

## Outputs

```
outputs/
├── api_usage.jsonl                      ← every paid API call (cost, tokens, provider, run_id)
├── results/                             ← local runs
└── openrouter/
    ├── results/<YYYY-MM-DD>/
    │   ├── run_info_<run_id>.json       ← models, endpoints, settings of one process
    │   └── iterations/<question_id>/<model>_iter<n>_<run_id>.json
    ├── results/baseline_samples/<run_id>.jsonl
    ├── logs/                            ← one log per shard
    └── memory.md                        ← one row per iteration
```

Status values: `LOOP`, `PASS`, `STOP_MAX`, `STOP_HIGH` (Node 1 stopped the attempt).

---

## Project structure

```
main.py                 pipeline entry point (MAX_PROBLEMS → SKIP_QIDS → SHARD_* filters)
config/                 settings, thresholds, model registry (local / cloud / openrouter)
pipeline/               LangGraph graph, state, gates, router
nodes/                  codegen, static_analysis, sandbox, cleaner, judgement
metrics/                radon / lizard / bandit / mypy+ruff, 100-point rubric
sandbox/                local LCB-compatible evaluator, runtime measurement, import whitelist
models/                 chat() client (llama.cpp or OpenRouter), budget ledger, prompts
dataset/                LCB loader (date window, seeded sample), formatter, edge cases, verify
memory/                 iteration logger (memory.md, iteration JSON, run_info)
benchmark/              sample_baseline, openrouter_results, paper_results, evaluator, legacy/
scripts/                run_openrouter.sh, start_model.sh, local_setup.sh, cloud_setup.sh
tests/                  18 test files + conftest.py
docs/                   one document per module; see CLAUDE.md for the index
judge_training/         future work: fine-tuning a judge model
```

---

## Notes

- Never commit `.env`, `.env.local`, `.env.cloud`, `.env.openrouter`.
- `datasets<3.0` is required (LCB's loading script needs `trust_remote_code`).
- radon and bandit are called as `sys.executable -m ...` so the venv install is used.
- The sandbox memory limit (`resource.setrlimit`) applies on Linux only; skipped on macOS.
- Code and code comments are in English; the runbook, thesis-facing docs and the paper are in Turkish.
