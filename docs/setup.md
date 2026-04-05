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
LOCAL_CODEGEN_MODEL=qwen2.5-coder-7b
LOCAL_JUDGMENT_MODEL=qwen2.5-coder-7b
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
cd models

# Qwen2.5-Coder-7B — for both codegen and judgment (~4 GB)
huggingface-cli download bartowski/Qwen2.5-Coder-7B-Instruct-GGUF \
    --include "*Q4_K_M.gguf" \
    --local-dir ./qwen2.5-coder-7b
```

> `bartowski` quants are tested with Mac Metal and work smoothly with llama.cpp.
> Toplam indirme: ~4 GB.

### A8. Start llama.cpp Server (Local)

Single server; all roles (M1/M2/M3/QWEN) route to this port:

```bash
# Run in a separate terminal or tmux window in project root
~/llama.cpp/llama-server \
    --model ./models/qwen2.5-coder-7b/*.gguf \
    --port 8001 \
    --host 0.0.0.0 \
    --ctx-size 8192 \
    --n-gpu-layers 99   # Metal — tüm katmanlar GPU'da

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
print('M1 model:  ', model_registry.M1.model_name)   # qwen2.5-coder-7b
print('QWEN model:', model_registry.QWEN.model_name)  # qwen2.5-coder-7b
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
source /workspace/llm-judge-refinement/.venv/bin/activate

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

# Judgment — Qwen3.5-35B-A3B (~24 GB)
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

```bash
cat > /workspace/llm-judge-refinement/scripts/start_model.sh << 'EOF'
#!/bin/bash
# Starts a single model — stops the previous one.
# Usage: bash scripts/start_model.sh M1|M2|M3|QWEN

set -e
MODEL=${1:?"\nKullanım: $0 M1|M2|M3|QWEN"}
LLAMA=/workspace/llama.cpp/llama-server
MODELS=/workspace/models

stop_server() {
    pkill -f "llama-server" 2>/dev/null || true
    sleep 3
}

case $MODEL in
M1)
    echo "→ gpt-oss-20b başlatılıyor (port 8001)..."
    stop_server
    $LLAMA \
        --model $MODELS/gpt-oss-20b/*.gguf \
        --port 8001 --host 0.0.0.0 \
        --ctx-size 16384 --n-gpu-layers 99 \
        --jinja \
        --temp 1.0 --top-p 1.0 --min-p 0.0 --top-k 0
    ;;
M2)
    echo "→ Qwen3-Coder-Next başlatılıyor (port 8002)..."
    stop_server
    $LLAMA \
        --model $MODELS/qwen3-coder-next/*.gguf \
        --port 8002 --host 0.0.0.0 \
        --ctx-size 16384 --n-gpu-layers 99 \
        --temp 1.0 --top-p 0.95 --min-p 0.01 --top-k 40
    ;;
M3)
    echo "→ GLM-4.7-Flash başlatılıyor (port 8003)..."
    stop_server
    $LLAMA \
        --model $MODELS/glm-4.7-flash/*.gguf \
        --port 8003 --host 0.0.0.0 \
        --ctx-size 16384 --n-gpu-layers 99 \
        --repeat-penalty 1.0 \
        --temp 1.0 --top-p 0.95 --min-p 0.01
    ;;
QWEN)
    echo "→ Qwen3.5-35B-A3B başlatılıyor (port 8004)..."
    stop_server
    $LLAMA \
        --model $MODELS/qwen3.5-35b/*.gguf \
        --port 8004 --host 0.0.0.0 \
        --ctx-size 8192 --n-gpu-layers 99 \
        --temp 0.3 --top-p 0.95 --min-p 0.01
    ;;
*)
    echo "Geçersiz model: '$MODEL'. M1|M2|M3|QWEN olmalı."
    exit 1
    ;;
esac
EOF

chmod +x /workspace/llm-judge-refinement/scripts/start_model.sh
```

### B9. Cloud Verification

Verify in order — each step must be green before proceeding to the next:

```bash
cd /workspace/llm-judge-refinement
source .venv/bin/activate

# 1. Python dependencies
python -c "import langgraph, radon, bandit, lizard, openai; print('Deps OK')"

# 2. ENV verification
ENV=cloud python -c "
from config import settings, model_registry
assert settings.env == 'cloud', 'ENV hatalı!'
assert model_registry.M1.model_name == 'gpt-oss-20b'
assert model_registry.M2.model_name == 'qwen3-coder-next'
assert model_registry.M3.model_name == 'glm-4.7-flash'
assert model_registry.QWEN.model_name == 'qwen3.5-35b-a3b'
print('Config OK — ENV=cloud, modeller doğru')
"

# 3. Dataset
python dataset/verify.py

# 4. Are model files present?
for d in gpt-oss-20b qwen3-coder-next glm-4.7-flash qwen3.5-35b; do
    count=$(ls /workspace/models/$d/*.gguf 2>/dev/null | wc -l)
    echo "$d: $count GGUF dosyası"
done
# Each line should show ≥ 1

# 5. M1 server test (first start: run start_model.sh M1 in new tmux window)
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
- `QWEN_BASE_URL` points to the judgment model — must not be the same port as the M1 server
