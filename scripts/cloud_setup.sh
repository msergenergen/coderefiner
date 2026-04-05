#!/bin/bash
# cloud_setup.sh — Vast.ai A100 one-shot setup script
#
# Run this script on the cloud instance after SSH connection:
#   tmux new -s main
#   bash /path/to/cloud_setup.sh [--skip-models] [--skip-llamacpp]
#
# Phases:
#   1. System packages
#   2. llama.cpp CUDA build         (skip with --skip-llamacpp)
#   3. Project clone + venv + deps
#   4. .env.cloud creation
#   5. Model downloads              (skip with --skip-models)
#   6. Verification
#
# Resume behaviour:
#   Each phase is guarded — already-done steps are skipped automatically.
#   Re-running the script is safe.

set -euo pipefail

# ── Argument parsing ──────────────────────────────────────────────────────────
SKIP_MODELS=0
SKIP_LLAMACPP=0
for arg in "$@"; do
    case $arg in
        --skip-models)   SKIP_MODELS=1 ;;
        --skip-llamacpp) SKIP_LLAMACPP=1 ;;
        --help)
            echo "Usage: bash cloud_setup.sh [--skip-models] [--skip-llamacpp]"
            echo "  --skip-llamacpp  Skip llama.cpp build (already built)"
            echo "  --skip-models    Skip model downloads (already downloaded)"
            exit 0
            ;;
    esac
done

# ── Paths ─────────────────────────────────────────────────────────────────────
WORKSPACE=/workspace
PROJECT=$WORKSPACE/llm-code-refinement
LLAMACPP=$WORKSPACE/llama.cpp
MODELS=$WORKSPACE/models

# ── Colour helpers ────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
ok()   { echo -e "${GREEN}✓ $*${NC}"; }
warn() { echo -e "${YELLOW}⚠ $*${NC}"; }
die()  { echo -e "${RED}✗ $*${NC}"; exit 1; }
section() { echo -e "\n${YELLOW}══ $* ══${NC}"; }

# ── Phase 1 — System packages ─────────────────────────────────────────────────
section "Phase 1 — System packages"

apt-get update -qq
apt-get install -y \
    git curl wget htop tmux tree \
    build-essential cmake ninja-build \
    libssl-dev libffi-dev libcurl4-openssl-dev \
    pciutils \
    python3.11 python3.11-venv python3.11-dev python3-pip \
    > /dev/null

python3.11 --version | grep -q "3.11" || die "Python 3.11 kurulumu başarısız"
cmake --version | head -1
nvidia-smi | head -3
ok "System packages OK"

# ── Phase 2 — llama.cpp CUDA build ───────────────────────────────────────────
section "Phase 2 — llama.cpp (CUDA)"

if [ "$SKIP_LLAMACPP" = "1" ]; then
    warn "llama.cpp atlandı (--skip-llamacpp)"
elif [ -x "$LLAMACPP/llama-server" ]; then
    ok "llama.cpp zaten mevcut — atlanıyor"
else
    cd "$WORKSPACE"
    [ -d "$LLAMACPP" ] || git clone https://github.com/ggml-org/llama.cpp

    cd "$LLAMACPP"
    cmake -B build \
        -DBUILD_SHARED_LIBS=OFF \
        -DGGML_CUDA=ON \
        -DCMAKE_BUILD_TYPE=Release \
        > /tmp/cmake_configure.log 2>&1 \
        || die "cmake configure hatası — /tmp/cmake_configure.log"

    cmake --build build \
        --config Release \
        -j"$(nproc)" \
        --clean-first \
        --target llama-cli llama-server \
        > /tmp/cmake_build.log 2>&1 \
        || die "cmake build hatası — /tmp/cmake_build.log"

    cp build/bin/llama-* .
    ok "llama.cpp build tamamlandı"
fi

"$LLAMACPP/llama-server" --version > /dev/null || die "llama-server çalışmıyor"
ok "llama-server OK"

# ── Phase 3 — Project clone + venv ───────────────────────────────────────────
section "Phase 3 — Project + Python env"

mkdir -p "$WORKSPACE"
cd "$WORKSPACE"

if [ -d "$PROJECT/.git" ]; then
    ok "Repo zaten mevcut — git pull"
    cd "$PROJECT" && git pull --ff-only
else
    git clone git@github.com:msergenergen/llm-code-refinement.git
fi

cd "$PROJECT"

if [ ! -d ".venv" ]; then
    python3.11 -m venv .venv
    ok "Venv oluşturuldu"
fi

source .venv/bin/activate
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
ok "Python dependencies kuruldu"

python -c "import langgraph, radon, bandit, lizard, openai; print('  Core deps OK')"
python -m mypy --version > /dev/null && python -m ruff --version > /dev/null
ok "mypy + ruff OK"

chmod +x scripts/start_model.sh

# ── Phase 4 — .env.cloud ─────────────────────────────────────────────────────
section "Phase 4 — .env.cloud"

if [ -f "$PROJECT/.env.cloud" ]; then
    ok ".env.cloud zaten mevcut — atlanıyor"
else
    cat > "$PROJECT/.env.cloud" << 'EOF'
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
    ok ".env.cloud oluşturuldu"
fi

ln -sf .env.cloud .env
mkdir -p outputs/results
ok ".env → .env.cloud symlink aktif"

# ── Phase 5 — Model downloads ─────────────────────────────────────────────────
section "Phase 5 — Model downloads"

mkdir -p "$MODELS"

if [ "$SKIP_MODELS" = "1" ]; then
    warn "Model indirme atlandı (--skip-models)"
else
    source "$PROJECT/.venv/bin/activate"

    # Each model is downloaded sequentially.
    # huggingface-cli resumes automatically if interrupted.
    declare -A REPOS=(
        [gpt-oss-20b]="unsloth/gpt-oss-20b-GGUF"
        [qwen3-coder-next]="unsloth/Qwen3-Coder-Next-GGUF"
        [glm-4.7-flash]="unsloth/GLM-4.7-Flash-GGUF"
        [qwen3.5-35b]="unsloth/Qwen3.5-35B-A3B-GGUF"
    )
    declare -A SIZES=(
        [gpt-oss-20b]="~11 GB"
        [qwen3-coder-next]="~46 GB"
        [glm-4.7-flash]="~18 GB"
        [qwen3.5-35b]="~24 GB"
    )

    for DIR in gpt-oss-20b qwen3-coder-next glm-4.7-flash qwen3.5-35b; do
        REPO="${REPOS[$DIR]}"
        SIZE="${SIZES[$DIR]}"
        TARGET="$MODELS/$DIR"

        existing=$(ls "$TARGET/"*.gguf 2>/dev/null | wc -l)
        if [ "$existing" -ge 1 ]; then
            ok "$DIR — zaten mevcut ($existing shard), atlanıyor"
        else
            echo "  İndiriliyor: $REPO  ($SIZE)"
            huggingface-cli download "$REPO" \
                --include "*UD-Q4_K_XL*" \
                --local-dir "$TARGET"
            ok "$DIR indirme tamamlandı"
        fi
    done

    echo ""
    echo "Disk kullanımı:"
    du -sh "$MODELS"/*/
fi

# ── Phase 6 — Verification ───────────────────────────────────────────────────
section "Phase 6 — Verification"

cd "$PROJECT"
source .venv/bin/activate

# 1. Core Python deps
python -c "import langgraph, radon, bandit, lizard, openai; print('  [1/6] Deps OK')"

# 2. Linting
python -m mypy --version > /dev/null
python -m ruff --version > /dev/null
echo "  [2/6] mypy + ruff OK"

# 3. Config + thresholds
python - << 'PYEOF'
from config import settings, model_registry, thresholds
assert settings.env == 'cloud',                            'ENV != cloud'
assert model_registry.M1.model_name   == 'gpt-oss-20b',   'M1 model hatalı'
assert model_registry.M2.model_name   == 'qwen3-coder-next', 'M2 model hatalı'
assert model_registry.M3.model_name   == 'glm-4.7-flash',  'M3 model hatalı'
assert model_registry.QWEN.model_name == 'qwen3.5-35b-a3b','QWEN model hatalı'
assert thresholds.node3.easy   == 80
assert thresholds.node3.medium == 70
assert thresholds.node3.hard   == 60
assert thresholds.node2.cpu_limit_ms == 2000.0
print('  [3/6] Config OK — ENV=cloud, modeller ve eşikler doğru')
PYEOF

# 4. Ablation variant smoke-test
python - << 'PYEOF'
from pipeline.graph import build_graph
for v in ('V1', 'V2', 'V3'):
    build_graph(v)
print('  [4/6] Ablation variants OK (V1/V2/V3)')
PYEOF

# 5. Dataset
python dataset/verify.py | sed 's/^/  /'
echo "  [5/6] Dataset OK"

# 6. Model files present
echo "  [6/6] Model shards:"
all_ok=1
for d in gpt-oss-20b qwen3-coder-next glm-4.7-flash qwen3.5-35b; do
    count=$(ls "$MODELS/$d/"*.gguf 2>/dev/null | wc -l)
    if [ "$count" -ge 1 ]; then
        echo "    $d: $count shard ✓"
    else
        echo "    $d: BULUNAMADI ✗"
        all_ok=0
    fi
done
[ "$all_ok" = "1" ] || warn "Bazı modeller eksik — Phase 5'i çalıştırın"

# ── Done ──────────────────────────────────────────────────────────────────────
section "Kurulum tamamlandı"

cat << 'MSG'

Sonraki adımlar — her model için ayrı tmux penceresi:

  tmux new-window -n m1   && bash scripts/start_model.sh M1
  tmux new-window -n m2   && bash scripts/start_model.sh M2
  tmux new-window -n m3   && bash scripts/start_model.sh M3
  tmux new-window -n qwen && bash scripts/start_model.sh QWEN

Serverlar ayağa kalktıktan sonra pipeline çalıştırma:

  source .venv/bin/activate
  python main.py

Ablation benchmark:

  python benchmark/baseline.py --model M1 --sample 50 --tag v0_M1
  python -m benchmark.evaluator --variant V3 --sample 50 --tag v3_full
MSG
