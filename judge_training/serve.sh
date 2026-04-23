#!/usr/bin/env bash
# Starts the fine-tuned judge model via llama.cpp.
#
# Usage:
#   bash serve.sh                          # uses config.yaml defaults
#   bash serve.sh --model ../models/judge-7b-code.gguf
#   bash serve.sh --port 8005 --cpu        # CPU-only (no GPU)
#   bash serve.sh --config config.yaml
#
# The server listens on an OpenAI-compatible API — the existing
# models/client.py and nodes/judgement.py need no changes.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG="$SCRIPT_DIR/config.yaml"
GGUF_PATH=""
PORT=""
N_GPU_LAYERS=""
CTX_SIZE=""
PARALLEL=""
CPU_ONLY=false

# ── Arg parsing ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)     CONFIG="$2";       shift 2 ;;
        --model)      GGUF_PATH="$2";    shift 2 ;;
        --port)       PORT="$2";         shift 2 ;;
        --ctx-size)   CTX_SIZE="$2";     shift 2 ;;
        --parallel)   PARALLEL="$2";     shift 2 ;;
        --cpu)        CPU_ONLY=true;     shift   ;;
        -h|--help)
            echo "Usage: bash serve.sh [--model PATH] [--port PORT] [--cpu] [--config FILE]"
            exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── Read config ───────────────────────────────────────────────────────────────
_cfg() {
    python3 -c "import yaml,os; c=yaml.safe_load(open('$CONFIG')); print($1)" 2>/dev/null
}

LLAMA_CPP_DIR=$(_cfg "os.path.expanduser(c['gguf']['llama_cpp_dir'])")
LLAMA="$LLAMA_CPP_DIR/llama-server"

[ -z "$GGUF_PATH"    ] && GGUF_PATH=$(_cfg "os.path.abspath(c['gguf']['output_path'])")
[ -z "$PORT"         ] && PORT=$(_cfg "c['serve']['port']")
[ -z "$CTX_SIZE"     ] && CTX_SIZE=$(_cfg "c['serve']['ctx_size']")
[ -z "$PARALLEL"     ] && PARALLEL=$(_cfg "c['serve']['parallel']")
[ -z "$N_GPU_LAYERS" ] && N_GPU_LAYERS=$(_cfg "c['serve']['n_gpu_layers']")

$CPU_ONLY && N_GPU_LAYERS=0

# ── Pre-flight checks ─────────────────────────────────────────────────────────
if [ ! -f "$LLAMA" ]; then
    echo "ERROR: llama-server not found at $LLAMA"
    echo "       Set gguf.llama_cpp_dir in $CONFIG"
    exit 1
fi

if [ ! -f "$GGUF_PATH" ]; then
    echo "ERROR: GGUF not found at $GGUF_PATH"
    echo "       Run: bash convert_to_gguf.sh ./checkpoints/<checkpoint>"
    exit 1
fi

GGUF_SIZE=$(du -sh "$GGUF_PATH" | cut -f1)

echo "════════════════════════════════════════════════════"
echo " Judge Model Server"
echo "════════════════════════════════════════════════════"
echo " GGUF        : $GGUF_PATH ($GGUF_SIZE)"
echo " Port        : $PORT"
echo " ctx-size    : $CTX_SIZE"
echo " n-gpu-layers: $N_GPU_LAYERS"
echo " parallel    : $PARALLEL"
echo "════════════════════════════════════════════════════"
echo " Health check (after startup):"
echo "   python test_judge.py --port $PORT"
echo ""
echo " To use in the pipeline, set in .env.local / .env.cloud:"
echo "   LOCAL_JUDGEMENT_MODEL=$(basename "$GGUF_PATH" .gguf)"
echo "════════════════════════════════════════════════════"
echo ""

# Stop any server already on this port
if lsof -ti :"$PORT" &>/dev/null; then
    echo "Stopping existing process on port $PORT …"
    lsof -ti :"$PORT" | xargs kill -9 2>/dev/null || true
    sleep 2
fi

exec "$LLAMA" \
    --model        "$GGUF_PATH" \
    --port         "$PORT" \
    --host         0.0.0.0 \
    --ctx-size     "$CTX_SIZE" \
    --n-gpu-layers "$N_GPU_LAYERS" \
    --parallel     "$PARALLEL" \
    --temp         0.1 \
    --repeat-penalty 1.0
