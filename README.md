# CodeRefiner: LLM-as-a-Judge for Iterative Code Refinement

CodeRefiner is a LangGraph pipeline that solves LiveCodeBench (LCB) Python problems iteratively.
A generator writes code; static analysis and sandboxed execution on the **public** tests produce
evidence; an **LLM judge makes the final ACCEPT/REJECT decision** and writes the feedback for the
next attempt. The delivered code is graded offline on the **hidden** tests, which the loop never
sees. Deterministic metrics are evidence and a guard, never the decision itself.

The project is part of a master's thesis at Kocaeli University and accompanies the ELECO 2026
paper (in Turkish):

> M. S. Ergen and A. Sayar, "Yinelemeli Kod İyileştirmede Yargıç Olarak Büyük Dil Modelleri: Çok
> Aşamalı Bir Yaklaşım" (LLM-as-a-Judge for Iterative Code Refinement: A Multi-Stage Approach),
> ELECO 2026, Bursa, Türkiye, 26–28 November 2026.

Citation metadata: [CITATION.cff](CITATION.cff).

---

## Results

150 LCB problems released 2025-01..2025-04 (seed-42 sample of 182; 40 easy, 38 medium, 72 hard).
Metric: pass@1 of the single delivered code on all hidden tests. Gains are in percentage points
over the same generator's single-sample baseline (TÖ; 5 independent samples per problem).

| Generator | Judge | Design | Single-sample | CodeRefiner | Gain | Wilcoxon p |
|---|---|---|---|---|---|---|
| DeepSeek-V3 | Qwen2.5-72B-Instruct | Y: judge verdict decides | 34.1% | 36.7% | +2.5 | 0.175 |
| DeepSeek-V3 | Qwen2.5-72B-Instruct | P: rubric threshold decides | 34.1% | 40.0% | +5.9 | 0.015 |
| DeepSeek-V3 | gpt-oss-120b | Y | 34.1% | 50.0% | +15.9 | < 0.001 |
| Llama-3.1-8B-Instruct | gpt-oss-120b | Y | 13.2% | 30.0% | +16.8 | < 0.001 |

- At most 5 generations per problem. The loop fixed problems the first attempt got wrong and broke
  at most one correct first attempt per run (5/0, 5/0, 27/1, 23/1).
- Budget-matched baseline: submitting the first of two single samples that passes all public tests
  (filt@2) gives 38.7% for DeepSeek-V3.
- **Contamination.** DeepSeek-V3 (2024-12) and Qwen2.5-72B (2024-09) were released before the
  problems (LCB's post-cutoff protocol). gpt-oss-120b was released in 2025-08, after the problems,
  so memorisation cannot be ruled out for the two gpt-oss runs; they complement the main
  experiment rather than replace it.
- **Evaluator check.** All numbers come from the pinned local LCB-compatible evaluator. Re-grading the
  3,227 recorded programs with LiveCodeBench's official `lcb_runner` agreed on 99.6% of them (99.75%
  after re-running four load-induced timeouts alone). In the main experiment only design P changes
  (40.0% → 39.3%, gain +5.2 points, p = 0.026); see [docs/main-results.md](docs/main-results.md) §14.

Details, ablations, judge accuracy and the comparison with published methods:
[docs/main-results.md](docs/main-results.md).

---

## Pipeline

```
START → codegen (active model; on loop: the judge's feedback + its own previous code)
  → Node 1: static analysis (cyclomatic complexity · maintainability index · Halstead · Bandit · mypy/ruff · import whitelist)
  → Severity Gate ── HIGH → stop                       (Bandit HIGH, non-whitelisted import or CC > 20)
                  ── LOW  → strip comments → Node 3     (unparseable code or poor MI/Halstead; sandbox skipped)
                  ── OK   ↓
  → Node 2: sandbox on the public tests (pass rate, CPU time, memory)
  → strip comments (bias prevention)
  → Node 3: judge — rubric + test results as evidence → ACCEPT/REJECT + feedback (no code allowed)
  → log iteration
  → Iteration Gate ── PASS (all public tests pass AND judge ACCEPT) → END
                   ── STOP_MAX (5 generations)                     → END
                   ── LOOP                                         → codegen
```

- **Design Y** (`JUDGE_DECISION=verdict`, proposed): the judge's justified verdict is final; passing
  all public tests is a precondition, and a REJECT of public-passing code must name a concrete
  failing input or a time-complexity problem.
- **Design P** (`JUDGE_DECISION=rubric`, comparison): accept when all public tests pass and the
  100-point rubric score reaches the difficulty threshold (easy 80, medium 70, hard 60).
- Generator and judge come from different model families, and the judge may not write code
  (solution-leak check in the analysis).

Architecture and node details: [docs/architecture.md](docs/architecture.md),
[docs/graph.md](docs/graph.md), [docs/node1.md](docs/node1.md), [docs/node2.md](docs/node2.md),
[docs/node3.md](docs/node3.md).

---

## Models and providers

| Role | Model (release) | License | Access |
|---|---|---|---|
| Generator (main) | DeepSeek-V3 (2024-12) | DeepSeek Model License | OpenRouter → DeepInfra (fp4) |
| Judge (main) | Qwen2.5-72B-Instruct (2024-09) | Qwen License | OpenRouter → Novita (bf16) |
| Judge (additional runs) | gpt-oss-120b (2025-08) | Apache-2.0 | OpenRouter → DeepInfra |
| Generator (additional run) | Llama-3.1-8B-Instruct (2024-07) | Llama 3.1 Community License | OpenRouter → DeepInfra (fp8) |
| Secondary local experiment (not in the paper) | Qwen3.5-9B (2026-02) | Apache-2.0 | llama.cpp on a Mac mini M4 |

With `ENV=openrouter` every call is streamed, retried with backoff, pinned to one provider
endpoint (`allow_fallbacks=false`, `max_price`) and logged to `outputs/api_usage.jsonl`;
`models/budget.py` refuses a call that could push total spend past `API_BUDGET_USD`. Total OpenRouter
spend for the project, trial runs included, was 4.41 USD.

---

## Setup

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt          # note: datasets<3.0 is required for the LCB loader
python -m dataset.verify                 # loads LiveCodeBench and checks the environment
pytest -q                                # 293 passed, 1 skipped
```

Environments (OpenRouter, local llama.cpp, rented GPU): [docs/setup.md](docs/setup.md) and the
runbook [CLOUD_README.md](CLOUD_README.md) (Turkish; §A OpenRouter, §B Vast.ai). The Node 2 memory
limit uses `resource.setrlimit` and is skipped on macOS.

---

## Reproducing the paper

### From the released logs (no API calls)

1. Download `coderefiner-eleco2026-logs.zip` from the GitHub release and unzip it into `outputs/`
   (it creates `outputs/openrouter/...`). The archive holds every attempt's code, metrics, judge
   verdict and feedback, the baseline samples, the hidden-test results (`hidden_eval_cache.json`),
   the analyses and the API ledger; LiveCodeBench problem statements and tests are stripped (see
   `DATA_README.md` inside the archive for how to re-join them by `question_id`).
2. Run the analysis commands in [docs/main-results.md](docs/main-results.md) §11 ("Komut"): first
   analyses C and D, then the main analysis B, which also writes the paper's macros, tables and the
   budget figure with `--latex <paper dir>`. Hidden-test grading is read from the cache; the
   LiveCodeBench dataset is still downloaded once for problem metadata. With stripped logs the
   Severity-Gate check "would the stopped code have passed" cannot be recomputed and is reported
   as unknown.

Tool reference: [docs/analysis.md](docs/analysis.md) §10 (`benchmark/openrouter_results.py`) and
§11 (`benchmark/judge_replay.py`) and §12 (`benchmark/official_regrade.py`).

### Re-running the experiment (paid API calls)

```bash
cp .env.openrouter.example .env.openrouter   # add OPENROUTER_API_KEY (never commit this file)
scripts/run_openrouter.sh smoke              # pipeline on 2 problems
scripts/run_openrouter.sh baseline           # single-sample baseline, 5 samples per problem
scripts/run_openrouter.sh run                # pipeline, designs Y and P, 4 shards each
scripts/run_openrouter.sh status             # progress and spend
```

Settings live in [config/openrouter.env](config/openrouter.env); every option is described in
[docs/config.md](docs/config.md). New runs get new run ids; pass them to the analysis commands.

---

## Experiment logs

`scripts/export_release_logs.py` builds `release/coderefiner-eleco2026-logs.zip` (about 6 MB)
from `outputs/` without modifying it, for the paper's runs only (run ids listed in the script and
in `DATA_README.md`). The logs are released under CC BY 4.0. They contain outputs of DeepSeek-V3,
Qwen2.5-72B-Instruct, gpt-oss-120b and Llama-3.1-8B-Instruct, whose licenses may apply to those
outputs. Details: [docs/memory-logging.md](docs/memory-logging.md).

---

## Repository layout

```
main.py              pipeline entry
config/              settings (pipeline.py), thresholds, model registry, openrouter.env
pipeline/            LangGraph graph, state, gates, router
nodes/               codegen, static analysis, sandbox, comment stripping, judge
metrics/             complexity, security, linting, 100-point rubric
sandbox/             local LCB-compatible evaluator, runner, import whitelist
models/              chat client, budget ledger, prompts, optional Langfuse tracing
dataset/             LCB loader (date window, seed-42 sample), formatting, checks
memory/              iteration logs and run info
benchmark/           baseline sampling, analyses (openrouter_results, judge_replay, paper_results), legacy tools
scripts/             run_openrouter.sh, local/cloud setup, export_release_logs.py
tests/               pytest suite
docs/                design and results documentation (index in CLAUDE.md)
judge_training/      future work: fine-tuned judge
outputs/, release/   git-ignored
```

Most documentation under `docs/` is written in Turkish, with code identifiers in English.

---

## License

- Code: MIT ([LICENSE](LICENSE)).
- Experiment logs: CC BY 4.0.
- LiveCodeBench problems and tests are not redistributed; they are loaded from
  `livecodebench/code_generation_lite` on Hugging Face.
