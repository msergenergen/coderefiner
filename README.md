# LLM-as-a-Judge for Iterative Code Refinement

A LangGraph-based pipeline that routes LiveCodeBench (LCB) problems through two LLMs
sequentially, evaluates each generated solution with three analysis nodes, and automatically
improves failing outputs using LLM-generated feedback.

Built as a master's thesis project.

---

## Table of Contents

- [Overview](#overview)
- [Pipeline Architecture](#pipeline-architecture)
- [Requirements](#requirements)
- [Environment Setup](#environment-setup)
  - [Local (Mac Mini M4)](#local-mac-mini-m4)
  - [Cloud (Vast.ai A100)](#cloud-vastai-a100)
- [Configuration](#configuration)
- [Running the Pipeline](#running-the-pipeline)
- [Scripts Reference](#scripts-reference)
- [Testing](#testing)
- [Benchmark & Ablation](#benchmark--ablation)
- [Output Files](#output-files)
- [Project Structure](#project-structure)

---

## Overview

The pipeline:
1. Loads Python coding problems from LiveCodeBench
2. Routes each problem through two LLMs in sequence: **M1 → M2**
3. For each model, iteratively refines the generated code (up to 10 iterations) using feedback from:
   - **Node 1 – Static Analysis**: Cyclomatic complexity, Maintainability Index, Halstead volume, Bandit security scan
   - **Node 2 – Sandbox Execution**: Runs test cases via lcb_runner, measures CPU/RAM usage
   - **Node 3 – Judgement LLM**: Scores code on a 100-point rubric, produces an `improved_prompt`
4. Writes iteration results to `outputs/memory.md` and per-iteration JSON files
5. Compares against a single-shot baseline (V0) for ablation analysis

**Ablation variants:**

| Variant | Nodes Active | Description |
|---------|-------------|-------------|
| V0 | — | Single-shot baseline (no pipeline) |
| V1 | Node 2 + Node 3 | Execution + Judgement (no static analysis) |
| V2 | Node 1 + Node 3 | Static Analysis + Judgement (no execution) |
| V3 | Node 1 + Node 2 + Node 3 | Full pipeline |

---

## Pipeline Architecture

```
START
  ↓
codegen (M1 / M2)   ← improved_prompt fed back on loop
  ↓
Node 1: Static Analysis   (CC · MI · Halstead · Bandit)
  ↓
Severity Gate
  ├─ HIGH  → log_severity_stop → END
  ├─ LOW   → Node 3 (Node 2 skipped)
  └─ OK    ↓
Node 2: Sandbox           (lcb_runner correctness + cpu_ms · ram_mb)
  ↓
strip_comments            (removes comments before judgement — bias prevention)
  ↓
Node 3: Judgement LLM      (100-point rubric → improved_prompt)
  ↓
log_iteration             (writes to memory.md + JSON)
  ↓
Iteration Gate
  ├─ PASS     → write_memory_final → next model → END
  ├─ STOP_MAX → END
  └─ LOOP     → codegen (same model, next iteration)
```

**PASS thresholds:** Easy ≥ 80 / Medium ≥ 70 / Hard ≥ 60 (out of 100)

---

## Requirements

### System

| Tool | Version | Purpose |
|------|---------|---------|
| macOS (Apple Silicon) or Ubuntu 22.04 | — | Local or cloud host |
| Python | 3.11 | Runtime |
| cmake | 3.x | Build llama.cpp |
| git | 2.x | Clone repo |

### Python Packages

See [requirements.txt](requirements.txt). Key dependencies:

```
langgraph>=0.2.0          # Pipeline graph
openai>=1.0.0             # Model API client (OpenAI-compatible)
python-dotenv             # .env loading
loguru                    # Logging
radon>=6.0.0              # CC / MI / Halstead metrics
bandit>=1.7.0             # Security analysis
lizard>=1.17.0            # Cognitive complexity
psutil>=5.9.0             # RAM measurement
datasets>=2.20.0,<3.0.0   # LiveCodeBench (must be <3.0 — see note below)
huggingface_hub>=0.23.0   # Model download
statsmodels, scipy        # McNemar test, Wilson CI
pytest, pytest-timeout, pytest-cov  # Testing
```

> **Note on `datasets` version:** LiveCodeBench uses a custom loading script
> (`trust_remote_code=True`) that is incompatible with `datasets>=3.0`.
> The version is pinned to `<3.0.0` in `requirements.txt`.

### Models (llama.cpp GGUF)

| ID | Model | Size | Port | Required flag |
|----|-------|------|------|---------------|
| M1 | Qwen3.5-9B (Unsloth GGUF) | ~6 GB | 8001 | `--jinja --cache-ram 0` |
| M2 | Qwen2.5-Coder-1.5B-Instruct (GGUF) | ~1 GB | 8002 | — |
| QWEN | Qwen3.5-35B-A3B (Unsloth GGUF) | ~24 GB | 8004 | — |

**Local (debug):** All roles use a single qwen3.5-9b (Unsloth UD-Q4_K_XL, ~6 GB) on port 8001.  
**Cloud (benchmark):** Each model runs on a separate port. Concurrent memory ~30 GB (M1+QWEN); likely no longer requires A100 80GB—re-verify with actual Vast.ai pricing.

---

## Environment Setup

### Local (Mac Mini M4)

#### 1. Install system tools

```bash
# Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
eval "$(/opt/homebrew/bin/brew shellenv)"

# Tools
brew install python@3.11 cmake git curl wget
```

#### 2. Build llama.cpp (Apple Metal — no CUDA needed)

```bash
cd ~
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp

cmake -B build -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=OFF
cmake --build build --config Release -j$(sysctl -n hw.logicalcpu) \
      --target llama-cli llama-server

cp build/bin/llama-* .
./llama-server --version   # verify
```

#### 3. Clone project and create virtual environment

```bash
git clone git@github.com:msergenergen/llm-code-refinement.git
cd llm-code-refinement

python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

#### 4. Create `.env.local` and activate it

```bash
cat > .env.local << 'EOF'
ENV=local
LOCAL_BASE_URL=http://localhost:8001/v1
LOCAL_CODEGEN_MODEL=qwen3.5-9b
LOCAL_JUDGEMENT_MODEL=qwen3.5-9b
LLAMA_API_KEY=local
MAX_ITERATIONS=3
SANDBOX_TIMEOUT_SEC=15
LCB_SPLIT=test
LCB_DIFFICULTY=easy
LCB_SAMPLE_SIZE=5
OUTPUT_DIR=outputs/results
MEMORY_LOG_PATH=outputs/memory.md
EOF

ln -sf .env.local .env
mkdir -p outputs/results
```

#### 5. Download the local model (~6 GB)

```bash
hf download unsloth/Qwen3.5-9B-GGUF \
    --include "Qwen3.5-9B-UD-Q4_K_XL.gguf" \
    --local-dir ./models/qwen3.5-9b
```

#### 6. Start the model server

In a **separate terminal** (or tmux window):

```bash
source .venv/bin/activate
bash scripts/start_model.sh      # no argument needed in local mode
```

You should see: `llama server listening at http://0.0.0.0:8001`

#### 7. Verify setup

```bash
source .venv/bin/activate

# Config check
python -c "
from config import settings, model_registry
print('ENV:       ', settings.env)           # local
print('M1 model:  ', model_registry.M1.model_name)   # qwen3.5-9b
print('Local mode:', model_registry.is_local())       # True
"

# Dataset check
python dataset/verify.py

# Graph compiles
python -c "from pipeline.graph import build_graph; build_graph(); print('OK')"

# Run tests
pytest tests/ -v --timeout=30

# Server check
curl -s http://localhost:8001/v1/models | python -m json.tool
```

---

### Cloud (Vast.ai A100)

#### 1. Select instance

Requirements for concurrent M1 (Qwen3.5-9B, ~6 GB) + QWEN judge (~24 GB) ≈ 30 GB:

```
GPU VRAM:  ≥ 32 GB (A100 80GB likely no longer required; smaller tier may suffice)
RAM:       ≥ 64 GB
Disk:      ≥ 300 GB
Image:     nvidia/cuda:12.1.0-base-ubuntu22.04
```

> Re-verification of actual Vast.ai pricing needed before finalizing specific GPU tier recommendation.

#### 2. Connect and set up tmux

```bash
ssh -p <PORT> root@<HOST>
tmux new -s main
```

#### 3. Install system packages

```bash
apt-get update && apt-get install -y \
    git curl wget build-essential cmake python3.11 python3.11-venv
```

#### 4. Build llama.cpp (CUDA)

```bash
cd /workspace
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON
cmake --build build --config Release -j$(nproc) --target llama-cli llama-server
cp build/bin/llama-* .
```

#### 5. Clone project, create venv, create `.env.cloud`

```bash
cd /workspace
git clone git@github.com:msergenergen/llm-code-refinement.git
cd llm-code-refinement

python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cat > .env.cloud << 'EOF'
ENV=cloud
M1_BASE_URL=http://localhost:8001/v1
M2_BASE_URL=http://localhost:8002/v1
QWEN_BASE_URL=http://localhost:8004/v1
LLAMA_API_KEY=local
MAX_ITERATIONS=10
PIPELINE_MODELS=M1,M2
SANDBOX_TIMEOUT_SEC=10
SANDBOX_MEMORY_LIMIT_MB=512
SANDBOX_FLAKINESS_RUNS=5
SANDBOX_MAX_FAILED_CASES=5
LCB_SPLIT=test
LCB_DIFFICULTY=all
LCB_SAMPLE_SIZE=50
OUTPUT_DIR=outputs/results
MEMORY_LOG_PATH=outputs/memory.md
PIPELINE_LOG_PATH=outputs/pipeline.log
EOF

ln -sf .env.cloud .env
mkdir -p outputs/results
```

#### 6. Download models (run in parallel tmux windows)

```bash
# M1 (~6 GB)
huggingface-cli download unsloth/Qwen3.5-9B-GGUF --include "*UD-Q4_K_XL*" --local-dir /workspace/models/qwen3.5-9b

# M2 (~1 GB)
huggingface-cli download unsloth/Qwen2.5-Coder-1.5B-Instruct-GGUF --include "*Q4_K_M*" --local-dir /workspace/models/qwen2.5-coder-1.5b

# Judgement (~24 GB)
huggingface-cli download unsloth/Qwen3.5-35B-A3B-GGUF --include "*UD-Q4_K_XL*" --local-dir /workspace/models/qwen3.5-35b
```

#### 7. Start model servers

Each model in its own tmux window:

```bash
bash scripts/start_model.sh M1    # port 8001
bash scripts/start_model.sh M2    # port 8002
bash scripts/start_model.sh QWEN  # port 8004
```

---

## Configuration

All configuration is via the `.env` file (symlinked to `.env.local` or `.env.cloud`).

| Variable | Default (local) | Description |
|----------|----------------|-------------|
| `ENV` | `local` | `local` or `cloud` |
| `LOCAL_BASE_URL` | `http://localhost:8001/v1` | Single server URL (local mode) |
| `MAX_ITERATIONS` | `3` (local) / `10` (cloud) | Max iterations per model |
| `SANDBOX_TIMEOUT_SEC` | `15` | Code execution timeout |
| `LCB_SPLIT` | `test` | Dataset split |
| `LCB_DIFFICULTY` | `easy` | `easy` / `medium` / `hard` / `all` |
| `LCB_SAMPLE_SIZE` | `5` | Number of problems to run |
| `OUTPUT_DIR` | `outputs/results` | JSON results directory |
| `MEMORY_LOG_PATH` | `outputs/memory.md` | Iteration log file |

Switch environments with:

```bash
ln -sf .env.local .env   # Mac Mini
ln -sf .env.cloud .env   # Vast.ai
```

---

## Running the Pipeline

```bash
source .venv/bin/activate

# Run the full pipeline (reads LCB_SAMPLE_SIZE problems from .env)
python main.py

# Watch live output
tail -f outputs/memory.md
```

---

## Scripts Reference

### `scripts/start_model.sh`

Starts a llama.cpp model server. In local mode, always starts port 8001
with the downloaded qwen3.5-9b (`--jinja` applied automatically). In cloud
mode, starts the model matching the given role ID on its dedicated port.

```bash
# Local mode — argument is optional (ENV=local detected from .env)
bash scripts/start_model.sh         # starts qwen3.5-9b on port 8001
bash scripts/start_model.sh M1      # same result in local mode

# Cloud mode — argument required
bash scripts/start_model.sh M1      # Qwen3.5-9B         → port 8001
bash scripts/start_model.sh M2      # Qwen2.5-Coder-1.5B → port 8002
bash scripts/start_model.sh QWEN    # Qwen3.5-35B-A3B    → port 8004
```

**Model-specific flags applied automatically:**
- `M1 (Qwen3.5-9B)`: `--jinja --cache-ram 0` — reasoning model; cache-ram 0 prevents Metal GPU OOM
- `M2 (Qwen2.5-Coder-1.5B)`: standard flags

### `benchmark/baseline.py`

Runs V0 single-shot baseline: one attempt per problem, no pipeline feedback.
Used to measure the model's natural ability for ablation comparison.

```bash
python benchmark/baseline.py --model M1 --sample 50 --tag v0_M1
python benchmark/baseline.py --model M2 --sample 50 --tag v0_M2

# Options:
#   --model      M1 | M2
#   --difficulty easy | medium | hard | all  (default: all)
#   --sample     N problems  (default: 50)
#   --tag        filename suffix for output JSON
```

Output: `outputs/results/baseline_<tag>_<timestamp>.json`

### `benchmark/evaluator.py`

Runs V1–V4 ablation variants of the full pipeline.

```bash
python -m benchmark.evaluator --variant V3 --sample 50 --tag v3_full
python -m benchmark.evaluator --variant V1 --sample 50 --tag v1
python -m benchmark.evaluator --variant V2 --difficulty easy --sample 25

# Options:
#   --variant    V1 | V2 | V3  (default: V3)
#   --difficulty easy | medium | hard | all  (default: all)
#   --sample     N problems
#   --tag        filename suffix
```

Output: `outputs/results/benchmark_<variant>_<tag>_<timestamp>.json`

### `benchmark/analysis.py`

Computes Task Success Rate (TSR), McNemar test, and ablation comparison
between a pipeline run and a baseline run.

```bash
python benchmark/analysis.py \
    --pipeline outputs/results/benchmark_V4_v4_full_20250115.json \
    --baseline  outputs/results/baseline_v0_M1_20250115.json
```

### `dataset/verify.py`

Verifies the LiveCodeBench dataset loads correctly and prints statistics.

```bash
python dataset/verify.py
# Expected output:
#   Load OK — 10 examples
#   Distribution: {'total': 10, 'easy': 5, 'medium': 2, 'hard': 3}
```

---

## Testing

```bash
source .venv/bin/activate

# All tests
pytest tests/ -v --timeout=30

# Fast tests only (skip sandbox execution tests)
pytest tests/ -v --timeout=30 \
    --ignore=tests/test_runtime.py \
    --ignore=tests/test_lcb_eval.py

# Single test file
pytest tests/test_gates.py -v
pytest tests/test_metrics.py -v

# Coverage report
pytest tests/ --cov=. --cov-report=term-missing \
    --cov-omit="benchmark/*,tests/*,.venv/*"
```

**Expected result:** 128 passed, 1 skipped (~4 seconds)

| Test File | What It Tests |
|-----------|---------------|
| `test_cleaner.py` | Comment stripping (tokenize-based) |
| `test_metrics.py` | CC, MI, Halstead, Bandit, severity decisions |
| `test_gates.py` | severity_gate and iteration_gate routing |
| `test_lcb_eval.py` | lcb_runner diversity score and fallback |
| `test_runtime.py` | Subprocess-based runtime measurement |
| `test_edge_cases.py` | Rule-based edge case generation |
| `test_dataset.py` | HTML cleaning, problem formatting |
| `test_logger.py` | memory.md writing and iteration JSON |
| `test_nodes.py` | Integration: static_analysis_node, cleaner_node |
| `test_prompts.py` | Four-layer codegen prompt structure |
| `test_baseline.py` | Code block extraction from LLM output |

---

## Benchmark & Ablation

### Recommended run order

```bash
# 1. V0 Baseline — separately for each model (always run first)
python benchmark/baseline.py --model M1 --sample 50 --tag v0_M1
python benchmark/baseline.py --model M2 --sample 50 --tag v0_M2

# 2. Ablation variants
python -m benchmark.evaluator --variant V1 --sample 50 --tag v1
python -m benchmark.evaluator --variant V2 --sample 50 --tag v2
python -m benchmark.evaluator --variant V3 --sample 50 --tag v3_full

# 3. Difficulty breakdown (V3 full pipeline)
python -m benchmark.evaluator --variant V3 --difficulty easy   --sample 25 --tag v3_easy
python -m benchmark.evaluator --variant V3 --difficulty medium --sample 15 --tag v3_medium
python -m benchmark.evaluator --variant V3 --difficulty hard   --sample 10 --tag v3_hard
```

> Reset `outputs/memory.md` or set a different `MEMORY_LOG_PATH` before each variant run.

### Before each run — checklist

```bash
# Previous outputs saved?
ls outputs/results/

# Server running?
curl -s http://localhost:8001/v1/models | python -m json.tool

# Disk space (cloud)?
df -h /workspace
```

---

## Output Files

```
outputs/
├── run_info.json              ← Hardware info, model names, timestamp (one per run)
├── memory.md                  ← Iteration log table (appended across all runs)
└── results/
    ├── baseline_v0_M1_<ts>.json   ← V0 baseline results
    ├── benchmark_V4_<tag>_<ts>.json  ← Pipeline variant results
    └── iterations/
        └── <question_id>/
            └── M1_iter1.json      ← Per-iteration detail (metrics, scores, improved_prompt)
```

**`memory.md` columns:** `iter | model | model_name | q_id | difficulty | cc | mi | severity | pass_rate | score | status | total_ms | total_tok | json_path`

**Status values:**
- `LOOP` — judgement threshold not met, will retry
- `PASS` — threshold met, moving to next model
- `STOP_MAX` — max iterations reached
- `STOP_HIGH` — Node 1 HIGH severity, pipeline stopped early

---

## Project Structure

```
llm-code-refinement/
├── main.py                    ← Pipeline entry point
├── requirements.txt
├── pytest.ini
├── .env                       ← Symlink → .env.local or .env.cloud
├── scripts/
│   └── start_model.sh         ← Model server launcher
├── config/                    ← Settings, thresholds, model registry
├── pipeline/                  ← LangGraph graph, state, gates, router
├── nodes/                     ← codegen, static_analysis, sandbox, cleaner, judgement
├── metrics/                   ← CC/MI/Halstead (radon), security (bandit)
├── sandbox/                   ← lcb_runner wrapper, runtime measurer, import whitelist
├── models/                    ← OpenAI-compatible client, codegen prompts
├── dataset/                   ← LCB loader, problem formatter, edge case generator
├── memory/                    ← Iteration logger (memory.md + JSON)
├── benchmark/                 ← baseline.py (V0), evaluator.py (V1–V4), analysis.py
├── tests/                     ← 11 test files + conftest.py
├── docs/                      ← Phase documentation (one .md per module)
└── models/qwen3.5-9b/        ← Downloaded GGUF model (local mode)
```

---

## Important Notes

- **Never commit** `.env`, `.env.local`, `.env.cloud` — they are in `.gitignore`
- **Node 2 sandbox** uses `resource.setrlimit` for memory limits — Linux only (Vast.ai). On macOS the limit is skipped automatically
- **`datasets<3.0`** is required — newer versions dropped `trust_remote_code` support needed by LiveCodeBench
- **radon and bandit** are called via `sys.executable -m radon/bandit` (not bare CLI) to ensure the venv installation is used
- All code comments and documentation are in **English**
- Copy `outputs/` to your local machine regularly when running on Vast.ai
