#!/bin/bash
# Starts a single model server — stops the previous one first.
#
# Local (Mac Mini):  argument is optional; always starts Qwen2.5-Coder-7B on port 8001
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
    MODEL_PATH=$(ls "$PROJECT_ROOT/models/qwen2.5-coder-7b/"*.gguf 2>/dev/null | head -1)

    if [ -z "$MODEL_PATH" ]; then
        echo "Model not found: $PROJECT_ROOT/models/qwen2.5-coder-7b/*.gguf"
        echo "Run A7 first: huggingface-cli download bartowski/Qwen2.5-Coder-7B-Instruct-GGUF ..."
        exit 1
    fi

    echo "→ Local mode: starting Qwen2.5-Coder-7B on port 8001 (Apple Metal)..."
    pkill -f "llama-server" 2>/dev/null || true
    sleep 2

    $LLAMA \
        --model "$MODEL_PATH" \
        --port 8001 \
        --host 0.0.0.0 \
        --ctx-size 8192 \
        --n-gpu-layers 99 \
        --temp 0.7

    exit 0
fi

# ── Cloud mode (Vast.ai A100) ─────────────────────────────────────────────────
LLAMA=/workspace/llama.cpp/llama-server
MODELS=/workspace/models

stop_server() {
    pkill -f "llama-server" 2>/dev/null || true
    sleep 3
}

case $MODEL in
M1)
    echo "→ Starting gpt-oss-20b (port 8001)..."
    stop_server
    $LLAMA \
        --model "$MODELS/gpt-oss-20b/"*.gguf \
        --port 8001 --host 0.0.0.0 \
        --ctx-size 16384 --n-gpu-layers 99 \
        --jinja \
        --temp 1.0 --top-p 1.0 --min-p 0.0 --top-k 0
    ;;
M2)
    echo "→ Starting Qwen3-Coder-Next (port 8002)..."
    stop_server
    $LLAMA \
        --model "$MODELS/qwen3-coder-next/"*.gguf \
        --port 8002 --host 0.0.0.0 \
        --ctx-size 16384 --n-gpu-layers 99 \
        --temp 1.0 --top-p 0.95 --min-p 0.01 --top-k 40
    ;;
M3)
    echo "→ Starting GLM-4.7-Flash (port 8003)..."
    stop_server
    $LLAMA \
        --model "$MODELS/glm-4.7-flash/"*.gguf \
        --port 8003 --host 0.0.0.0 \
        --ctx-size 16384 --n-gpu-layers 99 \
        --repeat-penalty 1.0 \
        --temp 1.0 --top-p 0.95 --min-p 0.01
    ;;
QWEN)
    echo "→ Starting Qwen3.5-35B-A3B (port 8004)..."
    stop_server
    $LLAMA \
        --model "$MODELS/qwen3.5-35b/"*.gguf \
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
