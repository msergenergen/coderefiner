"""
Faz 4 — Evaluation

Runs the fine-tuned judge on the test split and computes:
  1. Pearson correlation between predicted total_score and node2_pass_rate
  2. Score consistency (3 runs on same input, measure std)
  3. JSON parse success rate (model must emit valid JSON)
  4. Breakdown category MAE vs QWEN-labeled ground truth

Usage:
    python evaluate.py --checkpoint ./checkpoints
    python evaluate.py --checkpoint ./checkpoints --data-dir ./data --consistency-runs 3
    python evaluate.py --checkpoint ./checkpoints --split val
"""

import argparse
import json
import os
import re
import statistics

import torch
import yaml
from datasets import load_from_disk
from peft import PeftModel
from scipy.stats import pearsonr
from transformers import AutoModelForCausalLM, AutoTokenizer

from build_dataset import SYSTEM_PROMPT, format_prompt


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Inference ────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(checkpoint: str, base_model: str):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, checkpoint)
    model.eval()
    return model, tokenizer


def predict(model, tokenizer, example: dict, max_new_tokens: int = 512) -> str:
    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": format_prompt(example)},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def parse_json_output(raw: str) -> dict | None:
    # Try direct parse first, then extract first JSON block
    for candidate in [raw, re.search(r"\{.*\}", raw, re.DOTALL)]:
        if isinstance(candidate, re.Match):
            candidate = candidate.group(0)
        if candidate:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
    return None


# ── Metrics ───────────────────────────────────────────────────────────────────

def evaluate_split(model, tokenizer, examples: list[dict], consistency_runs: int) -> dict:
    predicted_scores  = []
    ground_truth_pass = []
    parse_failures    = 0
    consistency_stds  = []
    breakdown_errors  = {k: [] for k in ["accuracy", "efficiency", "quality",
                                          "security", "algorithm_choice", "error_handling"]}

    for i, ex in enumerate(examples):
        raw = predict(model, tokenizer, ex)
        parsed = parse_json_output(raw)

        if parsed is None:
            parse_failures += 1
            print(f"  [{i+1}/{len(examples)}] PARSE FAIL: {raw[:80]!r}")
            continue

        pred_score = float(parsed.get("total_score", 0.0))
        predicted_scores.append(pred_score)
        ground_truth_pass.append(ex["node2_pass_rate"])

        # Breakdown MAE
        pred_bd = parsed.get("breakdown", {})
        true_bd = ex.get("breakdown", {})
        for k in breakdown_errors:
            if k in pred_bd and k in true_bd:
                breakdown_errors[k].append(abs(float(pred_bd[k]) - float(true_bd[k])))

        # Consistency: re-run same input N times with temperature > 0
        if consistency_runs > 1:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": format_prompt(ex)},
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            scores_this = [pred_score]
            for _ in range(consistency_runs - 1):
                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=512,
                        do_sample=True,
                        temperature=0.3,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                decoded = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                p = parse_json_output(decoded.strip())
                if p:
                    scores_this.append(float(p.get("total_score", pred_score)))
            if len(scores_this) > 1:
                consistency_stds.append(statistics.stdev(scores_this))

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(examples)} evaluated …")

    n = len(predicted_scores)
    results = {
        "n_examples":          len(examples),
        "n_parsed":            n,
        "json_parse_rate":     n / len(examples) if examples else 0.0,
    }

    if n >= 2:
        r, p_val = pearsonr(predicted_scores, ground_truth_pass)
        results["pearson_r"]  = round(r, 4)
        results["pearson_p"]  = round(p_val, 6)
    else:
        results["pearson_r"] = None
        results["pearson_p"] = None

    if consistency_stds:
        results["consistency_std_mean"] = round(statistics.mean(consistency_stds), 3)

    results["breakdown_mae"] = {
        k: round(statistics.mean(v), 2) if v else None
        for k, v in breakdown_errors.items()
    }

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned judge model")
    parser.add_argument("--config",            default="config.yaml")
    parser.add_argument("--checkpoint",        required=True, help="Path to LoRA adapter checkpoint")
    parser.add_argument("--data-dir",          default=None)
    parser.add_argument("--split",             default="test", choices=["train", "val", "test"])
    parser.add_argument("--consistency-runs",  default=3, type=int)
    parser.add_argument("--max-examples",      default=None, type=int,
                        help="Limit number of examples (for quick sanity check)")
    args = parser.parse_args()

    cfg      = load_config(args.config)
    data_dir = os.path.abspath(args.data_dir or cfg["data"]["output_dir"])

    print(f"Checkpoint : {args.checkpoint}")
    print(f"Data dir   : {data_dir}  (split: {args.split})")
    print(f"Base model : {cfg['base_model']}")

    model, tokenizer = load_model_and_tokenizer(args.checkpoint, cfg["base_model"])

    dataset = load_from_disk(data_dir)
    examples = list(dataset[args.split])
    if args.max_examples:
        examples = examples[:args.max_examples]

    print(f"Evaluating {len(examples)} examples (consistency_runs={args.consistency_runs}) …\n")
    results = evaluate_split(model, tokenizer, examples, args.consistency_runs)

    print("\n── Results ─────────────────────────────────────────────────────")
    print(f"  JSON parse rate     : {results['json_parse_rate']:.2%}  ({results['n_parsed']}/{results['n_examples']})")
    if results["pearson_r"] is not None:
        print(f"  Pearson r (score vs pass_rate): {results['pearson_r']}  p={results['pearson_p']}")
    if "consistency_std_mean" in results:
        print(f"  Consistency std (mean)        : {results['consistency_std_mean']} pts")
    print(f"  Breakdown MAE per category:")
    for k, v in results["breakdown_mae"].items():
        print(f"    {k:20s}: {v}")
    print()

    out_path = os.path.join(os.path.dirname(args.checkpoint), f"eval_{args.split}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved → {out_path}")


if __name__ == "__main__":
    main()
