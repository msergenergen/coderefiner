#!/usr/bin/env bash
# Faz 5 — GGUF Conversion
#
# Merges LoRA adapter into base model, converts to GGUF, quantizes,
# and prints the .env.local update needed to switch the pipeline to the new judge.
#
# Usage:
#   bash convert_to_gguf.sh <checkpoint_dir> [config.yaml]
#
# Example:
#   bash convert_to_gguf.sh ./checkpoints/checkpoint-600
#   bash convert_to_gguf.sh ./checkpoints/checkpoint-600 config.yaml

set -euo pipefail

CHECKPOINT="${1:?Usage: bash convert_to_gguf.sh <checkpoint_dir> [config.yaml]}"
CONFIG="${2:-config.yaml}"

# ── Read config values ────────────────────────────────────────────────────────
BASE_MODEL=$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c['base_model'])")
LLAMA_CPP_DIR=$(python3 -c "import yaml,os; c=yaml.safe_load(open('$CONFIG')); print(os.path.expanduser(c['gguf']['llama_cpp_dir']))")
GGUF_OUTPUT=$(python3 -c "import yaml,os; c=yaml.safe_load(open('$CONFIG')); print(os.path.abspath(c['gguf']['output_path']))")
QUANT_TYPE=$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c['gguf']['quant_type'])")
SERVE_PORT=$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c['serve']['port'])")

MERGED_DIR="$(dirname "$CHECKPOINT")/merged"
GGUF_FP16="${GGUF_OUTPUT%.gguf}-fp16.gguf"

echo "════════════════════════════════════════════════════"
echo " GGUF Conversion"
echo "════════════════════════════════════════════════════"
echo " Checkpoint : $CHECKPOINT"
echo " Base model : $BASE_MODEL"
echo " llama.cpp  : $LLAMA_CPP_DIR"
echo " Output     : $GGUF_OUTPUT  ($QUANT_TYPE)"
echo "════════════════════════════════════════════════════"
echo ""

# ── Step 1: Merge LoRA into base model ───────────────────────────────────────
echo "[1/4] Merging LoRA adapter into base model …"
python3 - <<PYEOF
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, os

checkpoint = "$CHECKPOINT"
base_model = "$BASE_MODEL"
merged_dir = "$MERGED_DIR"

os.makedirs(merged_dir, exist_ok=True)

print(f"  Loading base: {base_model}")
model = AutoModelForCausalLM.from_pretrained(
    base_model, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)

print(f"  Applying adapter: {checkpoint}")
model = PeftModel.from_pretrained(model, checkpoint)
model = model.merge_and_unload()

print(f"  Saving merged model → {merged_dir}")
model.save_pretrained(merged_dir)
tokenizer.save_pretrained(merged_dir)
print("  Done.")
PYEOF

# ── Step 2: Convert to GGUF (fp16) ───────────────────────────────────────────
echo ""
echo "[2/4] Converting to GGUF (fp16) …"
mkdir -p "$(dirname "$GGUF_OUTPUT")"
python3 "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" "$MERGED_DIR" \
    --outfile "$GGUF_FP16" \
    --outtype f16

# ── Step 3: Quantize ─────────────────────────────────────────────────────────
echo ""
echo "[3/4] Quantizing to $QUANT_TYPE …"
"$LLAMA_CPP_DIR/llama-quantize" "$GGUF_FP16" "$GGUF_OUTPUT" "$QUANT_TYPE"

# ── Step 4: Summary ───────────────────────────────────────────────────────────
echo ""
echo "[4/4] Done."
echo ""
echo "════════════════════════════════════════════════════"
echo " Output GGUF : $GGUF_OUTPUT"
GGUF_SIZE=$(du -sh "$GGUF_OUTPUT" 2>/dev/null | cut -f1 || echo "?")
echo " File size   : $GGUF_SIZE"
echo ""
echo " To use in the pipeline, update .env.local / .env.cloud:"
echo ""
MODEL_NAME=$(basename "$GGUF_OUTPUT" .gguf)
echo "   LOCAL_JUDGEMENT_MODEL=$MODEL_NAME"
echo "   # (and add JUDGE_MODEL=CUSTOM to your env if needed)"
echo ""
echo " To serve the model manually:"
echo ""
echo "   ~/llama.cpp/llama-server \\"
echo "     --model $GGUF_OUTPUT \\"
echo "     --port $SERVE_PORT \\"
echo "     --ctx-size $(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c['serve']['ctx_size'])") \\"
echo "     --n-gpu-layers $(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c['serve']['n_gpu_layers'])") \\"
echo "     --parallel $(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c['serve']['parallel'])")"
echo "════════════════════════════════════════════════════"
