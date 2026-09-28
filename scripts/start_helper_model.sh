#!/bin/bash
# Starts a llama.cpp server for a local "orchestrator helper" model — used by
# Claude Code to draft code/docs/paper text, separate from the research
# pipeline's own M1/M2/M3/QWEN servers (see scripts/start_model.sh).
# Always stops any previous helper server on the same port first.
#
#   bash scripts/start_helper_model.sh              # defaults to qwen3.5-9b
#   bash scripts/start_helper_model.sh qwen3.5-9b    # same result

set -e

MODEL_KEY=${1:-qwen3.5-9b}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LLAMA=~/llama.cpp/llama-server

case "$MODEL_KEY" in
qwen3.5-9b)
    MODEL_DIR="$PROJECT_ROOT/models/qwen3.5-9b"
    PORT=8010
    CTX=8192
    ;;
*)
    echo "Bilinmeyen helper model: $MODEL_KEY"
    echo "Kullanım: bash scripts/start_helper_model.sh [qwen3.5-9b]"
    exit 1
    ;;
esac

MODEL_PATH=$(ls "$MODEL_DIR/"*.gguf 2>/dev/null | head -1)
if [ -z "$MODEL_PATH" ]; then
    echo "Model not found: $MODEL_DIR/*.gguf"
    exit 1
fi

echo "→ Helper mode: starting $MODEL_KEY on port $PORT (Apple Metal)..."
pkill -f "llama-server.*--port $PORT" 2>/dev/null || true
sleep 2

$LLAMA \
    --model "$MODEL_PATH" \
    --port "$PORT" \
    --host 0.0.0.0 \
    --ctx-size "$CTX" \
    --n-gpu-layers 99 \
    --jinja
