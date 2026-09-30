#!/bin/bash
# Starts a single model server — stops the previous one first.
#
# Local (Mac Mini):  argument is optional; always starts gpt-oss-20b on port 8001
#   bash scripts/start_model.sh
#   bash scripts/start_model.sh M1   # same result
#
# Cloud (Vast.ai):   argument required; each model has a dedicated port
#   bash scripts/start_model.sh M1|M2|QWEN
#   M1 = Qwen3.5-9B (medium), M2 = Qwen2.5-Coder-1.5B-Instruct (small) —
#   same-family Qwen models at two size tiers, isolating model size as
#   the controlled variable (see config/models.py docstring).

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
    MODEL_PATH=$(ls "$PROJECT_ROOT/models/qwen3.5-9b/"*.gguf 2>/dev/null | head -1)

    if [ -z "$MODEL_PATH" ]; then
        echo "Model not found: $PROJECT_ROOT/models/qwen3.5-9b/*.gguf"
        echo "Run: hf download unsloth/Qwen3.5-9B-GGUF --include \"Qwen3.5-9B-UD-Q4_K_XL.gguf\" --local-dir ./models/qwen3.5-9b"
        exit 1
    fi

    # llama-server splits --ctx-size evenly across slots, so scale it with
    # the slot count to keep 8192 tokens per request. Pair LOCAL_PARALLEL=N
    # with N main.py processes (SHARD_COUNT=N, SHARD_INDEX=0..N-1).
    PARALLEL=${LOCAL_PARALLEL:-1}

    echo "→ Local mode: starting qwen3.5-9b on port 8001 (Apple Metal, $PARALLEL slot(s))..."
    pkill -f "llama-server" 2>/dev/null || true
    sleep 2

    $LLAMA \
        --model "$MODEL_PATH" \
        --port 8001 \
        --host 0.0.0.0 \
        --ctx-size $((8192 * PARALLEL)) \
        --n-gpu-layers 99 \
        --parallel "$PARALLEL" \
        --cache-ram 0 \
        --jinja \
        --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0

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
    MODEL_PATH=$(first_shard "$MODELS/qwen3.5-9b")
    [ -z "$MODEL_PATH" ] && { echo "Model not found: $MODELS/qwen3.5-9b/*.gguf"; exit 1; }
    echo "→ Starting Qwen3.5-9B — medium anchor (port 8001) — $MODEL_PATH"
    stop_server
    $LLAMA \
        --model "$MODEL_PATH" \
        --port 8001 --host 0.0.0.0 \
        --ctx-size 16384 --n-gpu-layers 99 \
        --cache-ram 0 \
        --jinja \
        --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0
    ;;
M2)
    MODEL_PATH=$(first_shard "$MODELS/qwen2.5-coder-1.5b")
    [ -z "$MODEL_PATH" ] && { echo "Model not found: $MODELS/qwen2.5-coder-1.5b/*.gguf"; exit 1; }
    echo "→ Starting Qwen2.5-Coder-1.5B-Instruct — small anchor (port 8002) — $MODEL_PATH"
    stop_server
    $LLAMA \
        --model "$MODEL_PATH" \
        --port 8002 --host 0.0.0.0 \
        --ctx-size 16384 --n-gpu-layers 99 \
        --cache-ram 0 \
        --jinja \
        --temp 0.7 --top-p 0.8 --top-k 20 --repeat-penalty 1.05
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
    echo "Cloud mode requires a model argument: M1 | M2 | QWEN"
    echo "Usage: bash scripts/start_model.sh M1"
    exit 1
    ;;
esac
