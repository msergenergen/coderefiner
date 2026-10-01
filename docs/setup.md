# setup.md — Kurulum Rehberi

Two environments are defined:

| Environment | Purpose | Hardware | ENV flag |
|-------|------|---------|----------|
| **local** | Debug, pipeline validation | Mac Mini M4 16GB | `ENV=local` |
| **cloud** | Thesis benchmark | Vast.ai GPU ~30GB | `ENV=cloud` |

Both environments follow the same installation sequence:
`System tools → llama.cpp → Project + Python env → Model download → Server start → Verification`

---

## ORTAM A — Mac Mini M4 16GB (Local Debug)

### A1. Xcode Command Line Tools

Required for llama.cpp and cmake.

```bash
xcode-select --install
# Click "Install" in the dialog, takes a few minutes

# Verify installation
xcode-select -p
# Should output /Library/Developer/CommandLineTools
```

### A2. Homebrew

```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Add to PATH for ARM Mac (skip if already added)
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"
```

### A3. System Tools

```bash
brew install python@3.11 cmake git curl wget

# Verify
python3.11 --version   # Python 3.11.x
cmake --version        # cmake 3.x.x
git --version          # git 2.x.x
```

### A4. llama.cpp — Metal ile Derleme

Metal uses the Apple Silicon GPU directly — CUDA is not required.

```bash
cd ~
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp

# Metal otomatik aktif; CUDA=OFF
cmake -B build \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_CUDA=OFF

cmake --build build \
    --config Release \
    -j$(sysctl -n hw.logicalcpu) \
    --target llama-cli llama-server

cp build/bin/llama-* .

# Verify
./llama-cli --version
./llama-server --version
```

### A5. Proje Kurulumu

```bash
cd ~
git clone https://github.com/<kullanici>/llm-judge-refinement.git
cd llm-judge-refinement

# Python virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

# lcb_runner — official LCB evaluation engine (not on PyPI, not in
# requirements.txt). Without this, sandbox/lcb_eval.py silently falls back
# to a basic subprocess evaluator and node2_lcb_available stays False —
# confirmed happening on every local pilot run to date (see docs/node2.md).
git clone https://github.com/LiveCodeBench/LiveCodeBench.git /tmp/LiveCodeBench
pip install -e /tmp/LiveCodeBench

# Verify
python -c "import langgraph; print('LangGraph OK')"
python -c "import radon, bandit, lizard; print('Metrics OK')"
python -c "from lcb_runner.evaluation.utils_execute import check_correctness; print('lcb_runner OK')"
```

### A6. Create .env.local

```bash
cat > .env.local << 'EOF'
# ── Environment ───────────────────────────────────────────────────
ENV=local

# ── Single server — all roles go here ────────────────────────────
LOCAL_BASE_URL=http://localhost:8001/v1
LOCAL_CODEGEN_MODEL=qwen3.5-9b
LOCAL_JUDGEMENT_MODEL=qwen3.5-9b
LLAMA_API_KEY=local

# ── Pipeline — keep short for debug ──────────────────────────────
MAX_ITERATIONS=3
SANDBOX_TIMEOUT_SEC=15
SANDBOX_MEMORY_LIMIT_MB=256
SANDBOX_FLAKINESS_RUNS=2
SANDBOX_MAX_FAILED_CASES=3

# ── LiveCodeBench ─────────────────────────────────────────────────
LCB_SPLIT=test
LCB_DIFFICULTY=easy
LCB_SAMPLE_SIZE=5

# ── Output ────────────────────────────────────────────────────────
OUTPUT_DIR=outputs/results
MEMORY_LOG_PATH=outputs/memory.md
PIPELINE_LOG_PATH=outputs/pipeline.log
EOF

# Symlink — .env always points to the active environment file
ln -sf .env.local .env

mkdir -p outputs/results

# .gitignore — never commit env files
grep -qxF '.env'       .gitignore 2>/dev/null || echo '.env'       >> .gitignore
grep -qxF '.env.local' .gitignore 2>/dev/null || echo '.env.local' >> .gitignore
grep -qxF '.env.cloud' .gitignore 2>/dev/null || echo '.env.cloud' >> .gitignore
```

### A7. Model Download

```bash
# In project root directory
mkdir -p models

# qwen3.5-9b (Unsloth UD-Q4_K_XL) — for both codegen and judgement (~6 GB)
hf download unsloth/Qwen3.5-9B-GGUF \
    --include "Qwen3.5-9B-UD-Q4_K_XL.gguf" \
    --local-dir ./models/qwen3.5-9b
```

> Unsloth Dynamic (UD) Q4_K_XL gives best quality/size for qwen3.5-9b.
> Toplam indirme: ~6 GB. Fits comfortably on 16GB Mac Mini M4.

### A8. Start llama.cpp Server (Local)

Single server; all roles (M1/M2/QWEN) route to this port:

```bash
# Run in a separate terminal or tmux window in project root
~/llama.cpp/llama-server \
    --model ./models/qwen3.5-9b/*.gguf \
    --port 8001 \
    --host 0.0.0.0 \
    --ctx-size 8192 \
    --n-gpu-layers 99 \
    --parallel 1 \    # single slot — 3 default slots trigger Metal GPU OOM; also: --cache-ram 0
    --jinja \         # required — qwen3.5-9b uses jinja-based chat template
    --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0  # Qwen3 thinking-mode settings

# You should see this in the output:
# llama server listening at http://0.0.0.0:8001
```

> **Paralel slot (`LOCAL_PARALLEL`)** — `scripts/start_model.sh` reads
> `PARALLEL=${LOCAL_PARALLEL:-1}` and sets `--ctx-size $((8192 * PARALLEL)) --parallel $PARALLEL`
> so every slot keeps an 8192-token window. On the Mac Mini M4 (16 GB), 2–3 slots gave
> no throughput gain (~20 t/s total vs. ~16 t/s single) and pushed the machine into swap
> (0.4–1 t/s decode). Keep the default of 1 locally.

### A9. Local Verification

```bash
source .venv/bin/activate

# Is the server running?
curl -s http://localhost:8001/v1/models | python -m json.tool

# ENV verification
ENV=local python -c "
from config import settings, model_registry
print('ENV:       ', settings.env)
print('M1 model:  ', model_registry.M1.model_name)   # qwen3.5-9b
print('QWEN model:', model_registry.QWEN.model_name)  # qwen3.5-9b
print('Local mod: ', model_registry.is_local())       # True
"

# Dataset verification
python dataset/verify.py

# Run pipeline (5 easy questions)
ENV=local python main.py
```

---

## ORTAM B — Vast.ai GPU (Cloud)

> **Güncel ve önerilen yol: `CLOUD_README.md`** (tek model, yargıç = M1, token bütçesi, paralel parçalar,
> sonuçların Mac'te değerlendirilmesi). Aşağıdaki bölüm ilk üç modelli plan içindir. `lcb_runner`
> kurulsa bile `LCB_EVALUATOR=local` varsayılanı yerel değerlendiriciyi kullanır.

### B1. Vast.ai Instance Selection

**Instance requirements** — Concurrent largest models: Qwen3.5-9B (M1, ~6 GB) + Qwen3.5-35B-A3B (judge, ~24 GB) ≈ 30 GB.
A100 80GB is likely no longer required; may allow smaller/cheaper GPU tier. Re-verification with actual Vast.ai pricing is needed before finalizing specific tier recommendation:

| Parameter | Minimum | Recommended |
|-----------|---------|----------|
| GPU VRAM | ~32 GB (likely allows smaller tier than A100 80GB) | 48–80 GB |
| System RAM | 64 GB | 128 GB |
| Disk | 300 GB | 500 GB |
| CPU | 8 vCPU | 16 vCPU |
| Network | 500 Mbps | 1 Gbps |

**Filters in the Vast.ai interface:**

```
Disk Space:  ≥ 300 GB
RAM:         ≥ 64 GB
GPU RAM:     ≥ 80 GB
Reliability: ≥ 0.95
CUDA:        ≥ 12.1
Image:       nvidia/cuda:12.1.0-base-ubuntu22.04
```

**Add SSH key:** Vast.ai → Account → SSH Keys → paste your local public key:

```bash
cat ~/.ssh/id_rsa.pub
```

### B2. SSH Connection

```bash
# Copy the SSH command from the Vast.ai console, example:
ssh -p 12345 root@ssh6.vast.ai

# First step: start tmux (session persists if connection drops)
tmux new -s main
# To reconnect: tmux attach -t main
```

### B3. Sistem Paketleri

```bash
apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    htop \
    tmux \
    tree \
    build-essential \
    cmake \
    ninja-build \
    libssl-dev \
    libffi-dev \
    libcurl4-openssl-dev \
    pciutils \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip

# Verify
python3.11 --version   # Python 3.11.x
cmake --version        # cmake 3.x.x
nvidia-smi             # A100 görünmeli
```

### B4. llama.cpp — CUDA ile Derleme

```bash
cd /workspace

git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp

cmake -B build \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_CUDA=ON

cmake --build build \
    --config Release \
    -j$(nproc) \
    --clean-first \
    --target llama-cli llama-server

cp build/bin/llama-* .

# Verify
./llama-cli --version
./llama-server --version
```

### B5. Proje Kurulumu

```bash
cd /workspace
git clone git@github.com:msergenergen/llm-code-refinement.git
cd llm-code-refinement

# Python virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

# lcb_runner — official LCB evaluation engine (not on PyPI, not in
# requirements.txt). Without this, sandbox/lcb_eval.py silently falls back
# to a basic subprocess evaluator and node2_lcb_available stays False —
# using the official grader for the final thesis results is a methodological
# requirement (see docs/node2.md), so this step must not be skipped on cloud.
git clone https://github.com/LiveCodeBench/LiveCodeBench.git /workspace/LiveCodeBench
pip install -e /workspace/LiveCodeBench

# Verify
python -c "import langgraph; print('LangGraph OK')"
python -c "import radon, bandit, lizard; print('Metrics OK')"
python -c "import mypy, ruff; print('Linting OK')" 2>/dev/null || \
    python -c "import subprocess, sys; subprocess.run([sys.executable, '-m', 'mypy', '--version']); print('mypy OK')"
python -c "from lcb_runner.evaluation.utils_execute import check_correctness; print('lcb_runner OK')"
```

### B6. Create .env.cloud

```bash
cat > .env.cloud << 'EOF'
# ── Environment ───────────────────────────────────────────────────
ENV=cloud

# ── Model server URLs ─────────────────────────────────────────────
M1_BASE_URL=http://localhost:8001/v1
M2_BASE_URL=http://localhost:8002/v1
QWEN_BASE_URL=http://localhost:8004/v1
LLAMA_API_KEY=local

# ── Pipeline ──────────────────────────────────────────────────────
MAX_ITERATIONS=10
PIPELINE_MODELS=M1,M2
SANDBOX_TIMEOUT_SEC=10
SANDBOX_MEMORY_LIMIT_MB=512
SANDBOX_FLAKINESS_RUNS=5
SANDBOX_MAX_FAILED_CASES=5

# ── LiveCodeBench ─────────────────────────────────────────────────
LCB_SPLIT=test
LCB_DIFFICULTY=all
LCB_SAMPLE_SIZE=50

# ── Output ────────────────────────────────────────────────────────
OUTPUT_DIR=outputs/results
MEMORY_LOG_PATH=outputs/memory.md
PIPELINE_LOG_PATH=outputs/pipeline.log
EOF

# Symlink — .env always points to the active environment file
ln -sf .env.cloud .env

mkdir -p outputs/results

# .gitignore — never commit env files
grep -qxF '.env'       .gitignore 2>/dev/null || echo '.env'       >> .gitignore
grep -qxF '.env.local' .gitignore 2>/dev/null || echo '.env.local' >> .gitignore
grep -qxF '.env.cloud' .gitignore 2>/dev/null || echo '.env.cloud' >> .gitignore
```

### B7. Model Download

`huggingface-cli` is installed inside the venv — activate it first.

```bash
source /workspace/llm-code-refinement/.venv/bin/activate

mkdir -p /workspace/models
cd /workspace/models
```

Download each model in a separate tmux window — takes time, can run in parallel:

```bash
# M1 — Qwen3.5-9B (~6 GB)
tmux new-window -n "dl-m1"
huggingface-cli download unsloth/Qwen3.5-9B-GGUF \
    --include "*UD-Q4_K_XL*" \
    --local-dir ./qwen3.5-9b

# M2 — Qwen2.5-Coder-1.5B-Instruct (~1 GB)
tmux new-window -n "dl-m2"
huggingface-cli download unsloth/Qwen2.5-Coder-1.5B-Instruct-GGUF \
    --include "*Q4_K_M*" \
    --local-dir ./qwen2.5-coder-1.5b

# Judgement — Qwen3.5-35B-A3B (~24 GB)
tmux new-window -n "dl-qwen"
huggingface-cli download unsloth/Qwen3.5-35B-A3B-GGUF \
    --include "*UD-Q4_K_XL*" \
    --local-dir ./qwen3.5-35b
```

Verify after download completes:

```bash
du -sh /workspace/models/*/   # ~31 GB toplam
ls /workspace/models/qwen3.5-9b/*.gguf
ls /workspace/models/qwen2.5-coder-1.5b/*.gguf
ls /workspace/models/qwen3.5-35b/*.gguf
```

> If the download is interrupted, run the same `huggingface-cli download` command to resume.

### B8. Create scripts/start_model.sh

The script is already in the repository — no need to recreate it on the cloud instance.
Just make it executable after cloning:

```bash
chmod +x /workspace/llm-code-refinement/scripts/start_model.sh
```

Key behaviours built into the script:
- Detects `ENV` from `.env` automatically (local vs cloud)
- Uses `sort | head -1` to select only the first GGUF shard — Unsloth UD quants may be split across multiple files; llama.cpp discovers the remaining shards automatically
- Prints the resolved model path before starting so you can verify which file was picked
- Exits with a clear error if no `.gguf` file is found in the model directory

### B9. Cloud Verification

Verify in order — each step must be green before proceeding to the next:

```bash
cd /workspace/llm-code-refinement
source .venv/bin/activate

# 1. Python dependencies
python -c "import langgraph, radon, bandit, lizard, openai; print('Deps OK')"

# 2. Linting tools (mypy + ruff)
python -m mypy --version && python -m ruff --version && echo 'Linting OK'

# 3. ENV verification
python -c "
from config import settings, model_registry, thresholds
assert settings.env == 'cloud', 'ENV hatalı!'
assert model_registry.M1.model_name   == 'qwen3.5-9b',                'M1 model hatalı'
assert model_registry.M2.model_name   == 'qwen2.5-coder-1.5b-instruct', 'M2 model hatalı'
assert model_registry.QWEN.model_name == 'qwen3.5-35b-a3b',             'QWEN model hatalı'
assert thresholds.node3.easy   == 80,  'easy threshold hatalı'
assert thresholds.node3.medium == 70,  'medium threshold hatalı'
assert thresholds.node3.hard   == 60,  'hard threshold hatalı'
assert thresholds.node2.cpu_limit_ms == 2000.0, 'cpu_limit_ms hatalı'
print('Config OK — ENV=cloud, modeller ve eşikler doğru')
"

# 4. Ablation variant smoke-test (graph builds without error)
python -c "
from pipeline.graph import build_graph
for v in ('V1', 'V2', 'V3'):
    build_graph(v)
    print(f'  {v} graph OK')
print('Ablation variants OK')
"

# 5. Dataset
python dataset/verify.py

# 6. Are model files present?
for d in qwen3.5-9b qwen2.5-coder-1.5b qwen3.5-35b; do
    count=$(ls /workspace/models/$d/*.gguf 2>/dev/null | wc -l)
    echo "$d: $count GGUF files"
done
# Each line should show ≥ 1

# 7. M1 server test (first start: run start_model.sh M1 in new tmux window)
bash scripts/start_model.sh M1 &
sleep 15   # server starts

curl -s http://localhost:8001/v1/models | python -m json.tool
curl -s http://localhost:8001/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"qwen3.5-9b","messages":[{"role":"user","content":"print(42)"}],"max_tokens":30}' \
    | python -c "import sys,json; r=json.load(sys.stdin); print('M1 OK:', r['choices'][0]['message']['content'][:50])"
```

---

## Development → Production Migration Order

```
1. ENV=local (Mac Mini)
   5 easy questions → pipeline flow, gates, memory.md verified

2. ENV=cloud, M1 only (Qwen3.5-9B)
   5 questions → Vast.ai + llama.cpp setup verified

3. ENV=cloud, M1 + M2
   10 questions → full two-model codegen sequence verified

4. ENV=cloud, tam pipeline (M1+M2+QWEN)
   50 questions → thesis main benchmark

5. Ablation
   Single model / node disable variants
```

---

## Dikkat Edilecekler

**For both environments:**
- The `.env` file must not be committed to `git` — add it to `.gitignore`
- Copy the `outputs/` directory to your local machine regularly

**Local (Mac Mini):**
- Single server (port 8001) runs; the pipeline looks for no other port — `is_local()` verifies this
- M4 GPU is used via Metal offload (`--n-gpu-layers 99`); removing it dramatically reduces speed

**Cloud (Vast.ai):**
- **Stop** the instance when not in use — billed hourly
- Instance may freeze when disk is full: monitor with `df -h`
- If download is interrupted, the same `huggingface-cli download` command resumes it
- `QWEN_BASE_URL` points to the judgement model — must not be the same port as the M1 server
