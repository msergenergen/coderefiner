#!/bin/bash
# local_setup.sh — Mac Mini M4 one-shot setup script
#
# Run from anywhere on the Mac:
#   bash ~/Desktop/llm-code-refinement/scripts/local_setup.sh [--skip-model] [--skip-llamacpp]
#
# Phases:
#   1. Xcode Command Line Tools
#   2. Homebrew + system tools (python@3.11, cmake, git)
#   3. llama.cpp Metal build         (skip with --skip-llamacpp)
#   4. Project clone + venv + deps
#   5. .env.local creation
#   6. Model download (~11.9 GB)     (skip with --skip-model)
#   7. Verification
#
# Resume behaviour:
#   Each phase is guarded — already-done steps are skipped automatically.
#   Re-running the script is safe.

set -euo pipefail

# ── Argument parsing ──────────────────────────────────────────────────────────
SKIP_MODEL=0
SKIP_LLAMACPP=0
for arg in "$@"; do
    case $arg in
        --skip-model)    SKIP_MODEL=1 ;;
        --skip-llamacpp) SKIP_LLAMACPP=1 ;;
        --help)
            echo "Usage: bash local_setup.sh [--skip-model] [--skip-llamacpp]"
            echo "  --skip-llamacpp  Skip llama.cpp build (already built)"
            echo "  --skip-model     Skip model download (already downloaded)"
            exit 0
            ;;
    esac
done

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT="$HOME/Desktop/llm-code-refinement"
LLAMACPP="$HOME/llama.cpp"
MODEL_DIR="$PROJECT/models/qwen3.5-9b"

# ── Colour helpers ────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
ok()      { echo -e "${GREEN}✓ $*${NC}"; }
warn()    { echo -e "${YELLOW}⚠ $*${NC}"; }
die()     { echo -e "${RED}✗ $*${NC}"; exit 1; }
section() { echo -e "\n${YELLOW}══ $* ══${NC}"; }

# macOS only
[[ "$(uname)" == "Darwin" ]] || die "Bu betik yalnızca macOS içindir"

# ── Phase 1 — Xcode Command Line Tools ───────────────────────────────────────
section "Phase 1 — Xcode Command Line Tools"

if xcode-select -p &>/dev/null; then
    ok "Xcode CLT zaten kurulu: $(xcode-select -p)"
else
    warn "Xcode CLT kurulumu başlatılıyor — açılan diyalogda 'Install'a tıklayın"
    xcode-select --install
    # Wait until the installation finishes
    until xcode-select -p &>/dev/null; do sleep 5; done
    ok "Xcode CLT kurulumu tamamlandı"
fi

# ── Phase 2 — Homebrew + system tools ────────────────────────────────────────
section "Phase 2 — Homebrew + system tools"

if ! command -v brew &>/dev/null; then
    warn "Homebrew bulunamadı — kuruluyor..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# ARM Mac PATH
if [[ -f /opt/homebrew/bin/brew ]]; then
    eval "$(/opt/homebrew/bin/brew shellenv)"
    # Persist to shell profile if not already there
    grep -qF 'brew shellenv' ~/.zprofile 2>/dev/null || \
        echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
fi

ok "Homebrew: $(brew --version | head -1)"

for pkg in python@3.11 cmake git curl wget; do
    if brew list "$pkg" &>/dev/null; then
        ok "$pkg zaten kurulu"
    else
        echo "  $pkg kuruluyor..."
        brew install "$pkg" --quiet
        ok "$pkg kuruldu"
    fi
done

python3.11 --version | grep -q "3.11" || die "python3.11 PATH'de bulunamadı"
ok "python3.11: $(python3.11 --version)"
ok "cmake: $(cmake --version | head -1)"

# ── Phase 3 — llama.cpp Metal build ──────────────────────────────────────────
section "Phase 3 — llama.cpp (Apple Metal)"

if [ "$SKIP_LLAMACPP" = "1" ]; then
    warn "llama.cpp atlandı (--skip-llamacpp)"
elif [ -x "$LLAMACPP/llama-server" ]; then
    ok "llama.cpp zaten mevcut — atlanıyor ($LLAMACPP/llama-server)"
else
    cd "$HOME"
    [ -d "$LLAMACPP" ] || git clone https://github.com/ggml-org/llama.cpp

    cd "$LLAMACPP"
    cmake -B build \
        -DBUILD_SHARED_LIBS=OFF \
        -DGGML_CUDA=OFF \
        -DCMAKE_BUILD_TYPE=Release \
        > /tmp/cmake_configure.log 2>&1 \
        || die "cmake configure hatası — /tmp/cmake_configure.log"

    cmake --build build \
        --config Release \
        -j"$(sysctl -n hw.logicalcpu)" \
        --target llama-cli llama-server \
        > /tmp/cmake_build.log 2>&1 \
        || die "cmake build hatası — /tmp/cmake_build.log"

    cp build/bin/llama-* .
    ok "llama.cpp Metal build tamamlandı"
fi

"$LLAMACPP/llama-server" --version > /dev/null || die "llama-server çalışmıyor"
ok "llama-server OK"

# ── Phase 4 — Project clone + venv + deps ────────────────────────────────────
section "Phase 4 — Project + Python env"

if [ -d "$PROJECT/.git" ]; then
    ok "Repo zaten mevcut — git pull"
    cd "$PROJECT" && git pull --ff-only
else
    cd "$HOME/Desktop"
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

chmod +x scripts/start_model.sh scripts/cloud_setup.sh

# ── Phase 5 — .env.local ─────────────────────────────────────────────────────
section "Phase 5 — .env.local"

if [ -f "$PROJECT/.env.local" ]; then
    ok ".env.local zaten mevcut — atlanıyor"
else
    cat > "$PROJECT/.env.local" << 'EOF'
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
    ok ".env.local oluşturuldu"
fi

ln -sf .env.local .env
mkdir -p outputs/results
ok ".env → .env.local symlink aktif"

# ── Phase 6 — Model download (~6 GB) ──────────────────────────────────────
section "Phase 6 — Model download (qwen3.5-9b)"

if [ "$SKIP_MODEL" = "1" ]; then
    warn "Model indirme atlandı (--skip-model)"
elif [ "$(ls "$MODEL_DIR/"*.gguf 2>/dev/null | wc -l)" -ge 1 ]; then
    ok "Model zaten mevcut — atlanıyor"
    ls "$MODEL_DIR/"*.gguf
else
    source "$PROJECT/.venv/bin/activate"
    mkdir -p "$MODEL_DIR"
    echo "  İndiriliyor: unsloth/Qwen3.5-9B-GGUF UD-Q4_K_XL (~6 GB)"
    hf download unsloth/Qwen3.5-9B-GGUF \
        --include "Qwen3.5-9B-UD-Q4_K_XL.gguf" \
        --local-dir "$MODEL_DIR"
    ok "Model indirme tamamlandı"
    ls "$MODEL_DIR/"*.gguf
fi

# ── Phase 7 — Verification ───────────────────────────────────────────────────
section "Phase 7 — Verification"

cd "$PROJECT"
source .venv/bin/activate

# 1. Core Python deps
python -c "import langgraph, radon, bandit, lizard, openai; print('  [1/6] Deps OK')"

# 2. Linting
python -m mypy --version > /dev/null
python -m ruff --version > /dev/null
echo "  [2/6] mypy + ruff OK"

# 3. Config
python - << 'PYEOF'
from config import settings, model_registry, thresholds
assert settings.env == 'local',                                       'ENV != local'
assert model_registry.M1.model_name   == 'qwen3.5-9b',               'M1 model hatalı'
assert model_registry.QWEN.model_name == 'qwen3.5-9b',               'QWEN model hatalı'
assert model_registry.is_local() == True,                             'is_local() False'
assert thresholds.node3.easy   == 80
assert thresholds.node3.medium == 70
assert thresholds.node3.hard   == 60
print('  [3/6] Config OK — ENV=local, single server, eşikler doğru')
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

# 6. Model file present
count=$(ls "$MODEL_DIR/"*.gguf 2>/dev/null | wc -l)
if [ "$count" -ge 1 ]; then
    ok "[6/6] Model mevcut ($count dosya)"
else
    warn "[6/6] Model dosyası bulunamadı — Phase 6'yı çalıştırın"
fi

# ── Done ──────────────────────────────────────────────────────────────────────
section "Kurulum tamamlandı"

cat << MSG

Sonraki adım — model sunucusunu başlat (ayrı terminal/tmux):

  bash $PROJECT/scripts/start_model.sh

Sunucu ayağa kalktıktan sonra pipeline çalıştırma:

  cd $PROJECT
  source .venv/bin/activate
  python main.py

Test suite:

  pytest tests/ -q --timeout=30
MSG
