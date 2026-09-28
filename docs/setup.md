# setup.md — Kurulum Rehberi

Two environments are defined:

| Environment | Purpose | Hardware | ENV flag |
|-------|------|---------|----------|
| **local** | Debug, pipeline validation | Mac Mini M4 16GB | `ENV=local` |
| **cloud** | Thesis benchmark | Vast.ai A100 80GB | `ENV=cloud` |

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

# Verify
python -c "import langgraph; print('LangGraph OK')"
python -c "import radon, bandit, lizard; print('Metrics OK')"
```

### A6. Create .env.local

```bash
cat > .env.local << 'EOF'
# ── Environment ───────────────────────────────────────────────────
ENV=local

# ── Single server — all roles go here ────────────────────────────
LOCAL_BASE_URL=http://localhost:8001/v1
LOCAL_CODEGEN_MODEL=gpt-oss-20b
LOCAL_JUDGEMENT_MODEL=gpt-oss-20b
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

# gpt-oss-20b (Unsloth UD-Q4_K_XL) — for both codegen and judgement (~11.9 GB)
hf download unsloth/gpt-oss-20b-GGUF \
    --include "gpt-oss-20b-UD-Q4_K_XL.gguf" \
    --local-dir ./models/gpt-oss-20b
```

> Unsloth Dynamic (UD) quants give the best quality/size tradeoff for gpt-oss —
> all quants land near ~11-13 GB since the MoE weights are already native MXFP4.
> Toplam indirme: ~11.9 GB.

### A8. Start llama.cpp Server (Local)

Single server; all roles (M1/M2/M3/QWEN) route to this port:

```bash
# Run in a separate terminal or tmux window in project root
~/llama.cpp/llama-server \
    --model ./models/gpt-oss-20b/*.gguf \
    --port 8001 \
    --host 0.0.0.0 \
    --ctx-size 8192 \
    --n-gpu-layers 99 \
    --jinja   # required — gpt-oss-20b uses the Harmony chat template

# You should see this in the output:
# llama server listening at http://0.0.0.0:8001
```

### A9. Local Verification

```bash
source .venv/bin/activate

# Is the server running?
curl -s http://localhost:8001/v1/models | python -m json.tool

# ENV verification
ENV=local python -c "
from config import settings, model_registry
print('ENV:       ', settings.env)
print('M1 model:  ', model_registry.M1.model_name)   # gpt-oss-20b
print('QWEN model:', model_registry.QWEN.model_name)  # gpt-oss-20b
print('Local mod: ', model_registry.is_local())       # True
"

# Dataset verification
python dataset/verify.py

# Run pipeline (5 easy questions)
ENV=local python main.py
```

---

## ORTAM B — Vast.ai A100 80GB (Cloud)

### B1. Vast.ai Instance Selection

**Instance requirements** — dominant model Qwen3-Coder-Next (~46 GB Q4):

| Parameter | Minimum | Recommended |
|-----------|---------|----------|
| GPU VRAM | 80 GB (A100 80GB) | 80 GB |
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

# Verify
python -c "import langgraph; print('LangGraph OK')"
python -c "import radon, bandit, lizard; print('Metrics OK')"
python -c "import mypy, ruff; print('Linting OK')" 2>/dev/null || \
    python -c "import subprocess, sys; subprocess.run([sys.executable, '-m', 'mypy', '--version']); print('mypy OK')"
```

### B6. Create .env.cloud

```bash
cat > .env.cloud << 'EOF'
# ── Environment ───────────────────────────────────────────────────
ENV=cloud

# ── Model server URLs ─────────────────────────────────────────────
M1_BASE_URL=http://localhost:8001/v1
M2_BASE_URL=http://localhost:8002/v1
M3_BASE_URL=http://localhost:8003/v1
QWEN_BASE_URL=http://localhost:8004/v1
LLAMA_API_KEY=local

# ── Pipeline ──────────────────────────────────────────────────────
MAX_ITERATIONS=10
PIPELINE_MODELS=M1,M2,M3
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
# M1 — gpt-oss-20b (~11 GB)
tmux new-window -n "dl-m1"
huggingface-cli download unsloth/gpt-oss-20b-GGUF \
    --include "*UD-Q4_K_XL*" \
    --local-dir ./gpt-oss-20b

# M2 — Qwen3-Coder-Next (~46 GB) — dominant model
tmux new-window -n "dl-m2"
huggingface-cli download unsloth/Qwen3-Coder-Next-GGUF \
    --include "*UD-Q4_K_XL*" \
    --local-dir ./qwen3-coder-next

# M3 — GLM-4.7-Flash (~18 GB)
tmux new-window -n "dl-m3"
huggingface-cli download unsloth/GLM-4.7-Flash-GGUF \
    --include "*UD-Q4_K_XL*" \
    --local-dir ./glm-4.7-flash

# Judgement — Qwen3.5-35B-A3B (~24 GB)
tmux new-window -n "dl-qwen"
huggingface-cli download unsloth/Qwen3.5-35B-A3B-GGUF \
    --include "*UD-Q4_K_XL*" \
    --local-dir ./qwen3.5-35b
```

Verify after download completes:

```bash
du -sh /workspace/models/*/   # ~99 GB toplam
ls /workspace/models/gpt-oss-20b/*.gguf
ls /workspace/models/qwen3-coder-next/*.gguf
ls /workspace/models/glm-4.7-flash/*.gguf
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
assert model_registry.M1.model_name   == 'gpt-oss-20b',      'M1 model hatalı'
assert model_registry.M2.model_name   == 'qwen3-coder-next',  'M2 model hatalı'
assert model_registry.M3.model_name   == 'glm-4.7-flash',     'M3 model hatalı'
assert model_registry.QWEN.model_name == 'qwen3.5-35b-a3b',   'QWEN model hatalı'
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
for d in gpt-oss-20b qwen3-coder-next glm-4.7-flash qwen3.5-35b; do
    count=$(ls /workspace/models/$d/*.gguf 2>/dev/null | wc -l)
    echo "$d: $count GGUF dosyası"
done
# Each line should show ≥ 1

# 7. M1 server test (first start: run start_model.sh M1 in new tmux window)
bash scripts/start_model.sh M1 &
sleep 15   # server yüklensin

curl -s http://localhost:8001/v1/models | python -m json.tool
curl -s http://localhost:8001/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"gpt-oss-20b","messages":[{"role":"user","content":"print(42)"}],"max_tokens":30}' \
    | python -c "import sys,json; r=json.load(sys.stdin); print('M1 OK:', r['choices'][0]['message']['content'][:50])"
```

---

## Development → Production Migration Order

```
1. ENV=local (Mac Mini)
   5 easy soru → pipeline akışı, gate'ler, memory.md doğrulandı

2. ENV=cloud, sadece M3 (GLM-4.7-Flash)
   5 soru → Vast.ai + llama.cpp kurulumu doğrulandı

3. ENV=cloud, M1 + M3
   10 soru → gpt-oss-20b Harmony format (--jinja) doğrulandı

4. ENV=cloud, tam pipeline (M1+M2+M3+QWEN)
   50 soru → tez ana benchmark

5. Ablation
   Tek model / node kapatma varyantları
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
