# setup.md — Kurulum Rehberi

> Son güncelleme: 2026-10-02.

| Ortam | Amaç | Donanım | Nasıl seçilir |
|-------|------|---------|----------|
| **openrouter** | Ana deney (sızıntısız) | Yok — API | `scripts/run_openrouter.sh` (`config/openrouter.env` + `.env.openrouter` anahtarı) |
| **local** | ELECO bildirisi deneyi, geliştirme | Mac mini M4 16 GB | `ln -sf .env.local .env` |
| **cloud** | İsteğe bağlı: yerel deneyin büyük bütçeli tekrarı | Kiralık GPU (Vast.ai) | `ln -sf .env.cloud .env` |

Ortak adımlar: Python ortamı (A5) her yol için gerekir. OpenRouter için ek kurulum yoktur:
anahtar + `CLOUD_README.md` §A. Yerel yol: `Sistem araçları → llama.cpp → Proje → Model → Sunucu → Doğrulama`
(A1–A9; tek komut: `bash scripts/local_setup.sh`).

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
git clone https://github.com/<kullanici>/llm-code-refinement.git
cd llm-code-refinement

# Python virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

# lcb_runner is NOT needed: every reported result uses the pinned local
# LCB-compatible evaluator (LCB_EVALUATOR=local, sandbox/lcb_eval.py).

# Verify
python -c "import langgraph; print('LangGraph OK')"
python -c "import radon, bandit, lizard; print('Metrics OK')"
python -m dataset.verify     # "Window 2025-01..2025-04: 182 problems"
pytest -q                    # 272 passed, 1 skipped
```

### A6. Create .env.local

```bash
# Exact contents used for the ELECO paper runs: docs/config.md §6
# (or let scripts/local_setup.sh write a template).
cat > .env.local << 'EOF'
ENV=local
LOCAL_BASE_URL=http://localhost:8001/v1
LOCAL_CODEGEN_MODEL=qwen3.5-9b
LOCAL_JUDGEMENT_MODEL=qwen3.5-9b
LLAMA_API_KEY=local
PIPELINE_MODELS=M1
JUDGE_MODEL=QWEN
MAX_ITERATIONS=5
TREND_EXTENSION_ENABLED=false
SANDBOX_TIMEOUT_SEC=15
SANDBOX_MEMORY_LIMIT_MB=256
SANDBOX_FLAKINESS_RUNS=2
SANDBOX_MAX_FAILED_CASES=3
LCB_SPLIT=test
LCB_DIFFICULTY=all
LCB_SAMPLE_SIZE=150
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

# Run pipeline on 2 problems
MAX_PROBLEMS=2 python main.py
```

---

## ORTAM B — Kiralık GPU (isteğe bağlı)

Adım adım rehber: `CLOUD_README.md` §B (makine seçimi, CUDA ile llama.cpp, `.env.cloud`,
model indirme, paralel parçalar, sonuçların Mac'te değerlendirilmesi). Tek komutluk kurulum:
`bash scripts/cloud_setup.sh` (yalnızca Qwen3.5-9B'yi indirir; M2 ve ayrı yargıç için
`MODELS_EXTRA=1`). Eski üç modelli plan (Qwen3.5-35B-A3B yargıç, M1 → M2 sırası) artık
kullanılmıyor.

---

## Çalıştırma sırası

```
1. pytest -q + python -m dataset.verify             (her ortam)
2. OpenRouter: smoke → pilot 20 → baseline → run    (ana deney, CLOUD_README.md §A)
3. Yerel: MAX_PROBLEMS=2 python main.py → tam koşu  (ek düzen)
```

---

## Dikkat Edilecekler

**For both environments:**
- `.env*` files (except `.env.openrouter.example`) must not be committed — they are in `.gitignore`
- Copy the `outputs/` directory to your local machine regularly

**Local (Mac Mini):**
- Single server (port 8001) runs; the pipeline looks for no other port — `is_local()` verifies this
- M4 GPU is used via Metal offload (`--n-gpu-layers 99`); removing it dramatically reduces speed

**Cloud (Vast.ai):**
- **Stop** the instance when not in use — billed hourly
- Instance may freeze when disk is full: monitor with `df -h`
- If download is interrupted, the same `huggingface-cli download` command resumes it
- With `JUDGE_MODEL=M1` the judge uses the M1 server; set `QWEN_BASE_URL` only for a separate judge server

**OpenRouter:**
- The key lives only in `.env.openrouter`; never paste it into chats or commit it
- `scripts/run_openrouter.sh status` shows spend against `API_BUDGET_USD`
- `LOG_TAG=_x scripts/run_openrouter.sh run` gives new runs distinct log names while older runs are still writing theirs
- `scripts/run_openrouter.sh judge-replay MODEL PROVIDER NAME --run-ids …` re-judges the recorded public-passing attempts with another judge, without generating code (`docs/analysis.md` §11)
- Optional Langfuse tracing: add `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` (and `LANGFUSE_BASE_URL`) to `.env.openrouter`; check with `scripts/run_openrouter.sh langfuse-check` (`langfuse` is in `requirements.txt`)
