"""
Faz 2 — QLoRA SFT Training

Trains a LoRA adapter on top of the base model using the dataset
produced by build_dataset.py. Reads all hyperparameters from config.yaml.

Usage:
    python train.py
    python train.py --config config.yaml
    python train.py --model Qwen/Qwen2.5-7B-Instruct --epochs 5 --batch-size 2
    python train.py --load-in-4bit          # Mac Mini M4 mode
"""

import argparse
import os

import torch
import yaml
from datasets import load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer, SFTConfig


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_bnb_config(q_cfg: dict) -> BitsAndBytesConfig | None:
    if not q_cfg.get("load_in_4bit", False):
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=getattr(torch, q_cfg.get("bnb_4bit_compute_dtype", "bfloat16")),
        bnb_4bit_quant_type=q_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=True,
    )


def build_lora_config(l_cfg: dict) -> LoraConfig:
    return LoraConfig(
        r=l_cfg["r"],
        lora_alpha=l_cfg["alpha"],
        lora_dropout=l_cfg["dropout"],
        target_modules=l_cfg["target_modules"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="QLoRA SFT training for judge model")
    parser.add_argument("--config",       default="config.yaml")
    parser.add_argument("--model",        default=None,  help="Override base_model")
    parser.add_argument("--data-dir",     default=None,  help="Override data.output_dir")
    parser.add_argument("--output-dir",   default=None,  help="Override training.output_dir")
    parser.add_argument("--epochs",       default=None,  type=int)
    parser.add_argument("--batch-size",   default=None,  type=int)
    parser.add_argument("--lr",           default=None,  type=float)
    parser.add_argument("--load-in-4bit", action="store_true", help="Enable 4-bit quantization (Mac mode)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    model_id  = args.model      or cfg["base_model"]
    data_dir  = args.data_dir   or cfg["data"]["output_dir"]
    out_dir   = args.output_dir or cfg["training"]["output_dir"]
    t_cfg     = cfg["training"]
    l_cfg     = cfg["lora"]
    q_cfg     = cfg["quantization"]

    # CLI overrides
    if args.epochs:     t_cfg["num_train_epochs"] = args.epochs
    if args.batch_size: t_cfg["per_device_train_batch_size"] = args.batch_size
    if args.lr:         t_cfg["learning_rate"] = args.lr
    if args.load_in_4bit: q_cfg["load_in_4bit"] = True

    data_dir = os.path.abspath(data_dir)
    out_dir  = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Base model : {model_id}")
    print(f"Dataset    : {data_dir}")
    print(f"Output     : {out_dir}")
    print(f"LoRA r={l_cfg['r']}  alpha={l_cfg['alpha']}  4bit={q_cfg['load_in_4bit']}")

    # ── Load dataset ─────────────────────────────────────────────────────────
    dataset = load_from_disk(data_dir)
    train_data = dataset["train"]
    val_data   = dataset["val"]
    print(f"Train: {len(train_data)}  Val: {len(val_data)}")

    # ── Tokenizer ────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Model ────────────────────────────────────────────────────────────────
    bnb_config = build_bnb_config(q_cfg)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16 if not q_cfg.get("load_in_4bit") else None,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    # ── LoRA ─────────────────────────────────────────────────────────────────
    lora_config = build_lora_config(l_cfg)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Training args ─────────────────────────────────────────────────────────
    sft_cfg = SFTConfig(
        output_dir=out_dir,
        num_train_epochs=t_cfg["num_train_epochs"],
        per_device_train_batch_size=t_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=t_cfg["gradient_accumulation_steps"],
        learning_rate=t_cfg["learning_rate"],
        lr_scheduler_type=t_cfg["lr_scheduler_type"],
        warmup_ratio=t_cfg["warmup_ratio"],
        bf16=t_cfg["bf16"],
        tf32=t_cfg.get("tf32", False),
        max_seq_length=t_cfg["max_seq_length"],
        logging_steps=t_cfg["logging_steps"],
        eval_strategy="steps",
        eval_steps=t_cfg["eval_steps"],
        save_strategy="steps",
        save_steps=t_cfg["save_steps"],
        save_total_limit=t_cfg["save_total_limit"],
        load_best_model_at_end=t_cfg["load_best_model_at_end"],
        metric_for_best_model=t_cfg["metric_for_best_model"],
        report_to=t_cfg.get("report_to", "none"),
        dataset_text_field="text",
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"\nAdapter saved → {out_dir}")
    print("Next: python evaluate.py --checkpoint", out_dir)
    print("Next: bash convert_to_gguf.sh", out_dir)


if __name__ == "__main__":
    main()
