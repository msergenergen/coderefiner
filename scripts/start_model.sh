#!/bin/bash
# Starts a single model server — stops the previous one first.
#
# Local (Mac Mini):  argument is optional; always starts gpt-oss-20b on port 8001
#   bash scripts/start_model.sh
#   bash scripts/start_model.sh M1   # same result
#
# Cloud (Vast.ai):   argument required; each model has a dedicated port
#   bash scripts/start_model.sh M1|M2|M3|QWEN

set -e
MODEL=${1:-local}

# Always resolve paths relative to the project root (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Detect environment from .env in project root
if [ -f "$PROJECT_ROOT/.env" ]; then
    ENV_VAL=$(grep -E '^ENV=' "$PROJECT_ROOT/.env" | cut -d= -f2 | tr -d '[:space:]')
else
    ENV_VAL="local"
fi

# ── Local mode ────────────────────────────────────────────────────────────────
if [ "$ENV_VAL" = "local" ]; then
    LLAMA=~/llama.cpp/llama-server
    MODEL_PATH=$(ls "$PROJECT_ROOT/models/gpt-oss-20b/"*.gguf 2>/dev/null | head -1)

    if [ -z "$MODEL_PATH" ]; then
        echo "Model not found: $PROJECT_ROOT/models/gpt-oss-20b/*.gguf"
        echo "Run: hf download unsloth/gpt-oss-20b-GGUF --include \"gpt-oss-20b-UD-Q4_K_XL.gguf\" --local-dir ./models/gpt-oss-20b"
        exit 1
    fi

    echo "→ Local mode: starting gpt-oss-20b on port 8001 (Apple Metal)..."
    pkill -f "llama-server" 2>/dev/null || true
    sleep 2

    $LLAMA \
        --model "$MODEL_PATH" \
        --port 8001 \
        --host 0.0.0.0 \
        --ctx-size 8192 \
        --n-gpu-layers 99 \
        --jinja \
        --temp 1.0 --top-p 1.0 --min-p 0.0 --top-k 0

    exit 0
fi

# ── Cloud mode (Vast.ai A100) ─────────────────────────────────────────────────
LLAMA=/workspace/llama.cpp/llama-server
MODELS=/workspace/models

stop_server() {
    pkill -f "llama-server" 2>/dev/null || true
    sleep 3
}

# Unsloth UD quants may be split GGUFs (e.g. 00001-of-00003.gguf).
# llama.cpp needs only the first shard — it auto-discovers the rest.
# Using 'sort | head -1' instead of a bare glob prevents passing
# multiple filenames to --model when the model is split.
first_shard() {
    ls "$1/"*.gguf 2>/dev/null | sort | head -1
}

case $MODEL in
M1)
    MODEL_PATH=$(first_shard "$MODELS/gpt-oss-20b")
    [ -z "$MODEL_PATH" ] && { echo "Model not found: $MODELS/gpt-oss-20b/*.gguf"; exit 1; }
    echo "→ Starting gpt-oss-20b (port 8001) — $MODEL_PATH"
    stop_server
    $LLAMA \
        --model "$MODEL_PATH" \
        --port 8001 --host 0.0.0.0 \
        --ctx-size 16384 --n-gpu-layers 99 \
        --jinja \
        --temp 1.0 --top-p 1.0 --min-p 0.0 --top-k 0
    ;;
M2)
    MODEL_PATH=$(first_shard "$MODELS/qwen3-coder-next")
    [ -z "$MODEL_PATH" ] && { echo "Model not found: $MODELS/qwen3-coder-next/*.gguf"; exit 1; }
    echo "→ Starting Qwen3-Coder-Next (port 8002) — $MODEL_PATH"
    stop_server
    $LLAMA \
        --model "$MODEL_PATH" \
        --port 8002 --host 0.0.0.0 \
        --ctx-size 16384 --n-gpu-layers 99 \
        --temp 1.0 --top-p 0.95 --min-p 0.01 --top-k 40
    ;;
M3)
    MODEL_PATH=$(first_shard "$MODELS/glm-4.7-flash")
    [ -z "$MODEL_PATH" ] && { echo "Model not found: $MODELS/glm-4.7-flash/*.gguf"; exit 1; }
    echo "→ Starting GLM-4.7-Flash (port 8003) — $MODEL_PATH"
    stop_server
    $LLAMA \
        --model "$MODEL_PATH" \
        --port 8003 --host 0.0.0.0 \
        --ctx-size 16384 --n-gpu-layers 99 \
        --repeat-penalty 1.0 \
        --temp 1.0 --top-p 0.95 --min-p 0.01
    ;;
QWEN)
    MODEL_PATH=$(first_shard "$MODELS/qwen3.5-35b")
    [ -z "$MODEL_PATH" ] && { echo "Model not found: $MODELS/qwen3.5-35b/*.gguf"; exit 1; }
    echo "→ Starting Qwen3.5-35B-A3B (port 8004) — $MODEL_PATH"
    stop_server
    $LLAMA \
        --model "$MODEL_PATH" \
        --port 8004 --host 0.0.0.0 \
        --ctx-size 8192 --n-gpu-layers 99 \
        --temp 0.3 --top-p 0.95 --min-p 0.01
    ;;
*)
    echo "Cloud mode requires a model argument: M1 | M2 | M3 | QWEN"
    echo "Usage: bash scripts/start_model.sh M1"
    exit 1
    ;;
esac
