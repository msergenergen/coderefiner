"""
Judge model smoke test + interactive runner.

Modes:
  1. Smoke test — sends a synthetic example, validates JSON output
  2. File mode  — reads a real iteration JSON, sends it to the server
  3. Dataset mode — runs N examples from the test split, prints a summary

Usage:
    python test_judge.py                          # smoke test against default port
    python test_judge.py --port 8005
    python test_judge.py --file ../outputs/results/.../M1_iter1.json
    python test_judge.py --dataset ./data --n 10
    python test_judge.py --host http://some-server:8004
"""

import argparse
import json
import os
import sys
import time

import yaml

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: pip install openai")
    sys.exit(1)

try:
    from datasets import load_from_disk
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path: str = "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Prompt — imported from build_dataset.py so the rubric can't drift out of
# sync between training-data generation and this evaluation script ──────────

from build_dataset import SYSTEM_PROMPT, RUBRIC  # noqa: E402


def build_user_message(problem: str, code: str, all_tests) -> str:
    tests_str = json.dumps(all_tests, ensure_ascii=False, indent=2)
    return (
        f"[PROBLEM]\n{problem}\n\n"
        f"[CODE]\n```python\n{code}\n```\n\n"
        f"[RUBRIC]\n{RUBRIC}\n\n"
        f"[REFERENCE_TESTS]\n{tests_str}"
    )


# ── Inference ─────────────────────────────────────────────────────────────────

def call_judge(client: OpenAI, problem: str, code: str, all_tests,
               model: str = "judge") -> tuple[str, float]:
    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_user_message(problem, code, all_tests)},
        ],
        temperature=0.1,
        max_tokens=1024,
    )
    elapsed = time.perf_counter() - t0
    return response.choices[0].message.content.strip(), elapsed


def parse_output(raw: str) -> dict | None:
    import re
    for candidate in [raw, (m := re.search(r"\{.*\}", raw, re.DOTALL)) and m.group(0)]:
        if candidate:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
    return None


def print_result(raw: str, parsed: dict | None, elapsed: float,
                 ground_truth_score: float | None = None,
                 ground_truth_pass_rate: float | None = None) -> bool:
    ok = parsed is not None
    status = "OK" if ok else "PARSE FAIL"

    print(f"\n── Result [{status}]  {elapsed:.1f}s ─────────────────────────────")
    if ok:
        print(f"  total_score : {parsed.get('total_score', '?')}")
        print(f"  pass        : {parsed.get('pass', '?')}")
        bd = parsed.get("breakdown", {})
        if bd:
            cats = ["accuracy", "efficiency", "quality", "security",
                    "algorithm_choice", "error_handling"]
            print(f"  breakdown   : " + "  ".join(f"{k[:3]}={bd.get(k,'?')}" for k in cats))
        reasons = parsed.get("failure_reasons", [])
        if reasons:
            print(f"  reasons     : {reasons[0]}" + (f" (+{len(reasons)-1} more)" if len(reasons) > 1 else ""))
        prompt_preview = (parsed.get("improved_prompt") or "")[:80]
        if prompt_preview:
            print(f"  improved_prompt: {prompt_preview!r}…")
        if ground_truth_score is not None:
            diff = abs(parsed.get("total_score", 0) - ground_truth_score)
            print(f"  Δ score vs ground truth: {diff:.1f} pts (GT={ground_truth_score})")
        if ground_truth_pass_rate is not None:
            print(f"  Ground truth pass_rate : {ground_truth_pass_rate:.2f}")
    else:
        print(f"  Raw output: {raw[:200]!r}")
    return ok


# ── Smoke test ────────────────────────────────────────────────────────────────

SMOKE_PROBLEM = """\
Given a list of integers, return their sum. If the list is empty, return 0.

Constraints: 1 ≤ n ≤ 10^5, -10^9 ≤ nums[i] ≤ 10^9

Input: A single line containing space-separated integers.
Output: A single integer."""

SMOKE_CODE = """\
def solve(nums):
    return sum(nums)"""

SMOKE_TESTS = [
    {"input": "1 2 3", "output": "6"},
    {"input": "0",     "output": "0"},
    {"input": "-1 1",  "output": "0"},
]


def run_smoke(client: OpenAI, model: str):
    print("Mode: smoke test (synthetic example)")
    raw, elapsed = call_judge(client, SMOKE_PROBLEM, SMOKE_CODE, SMOKE_TESTS, model)
    parsed = parse_output(raw)
    ok = print_result(raw, parsed, elapsed)
    if ok:
        print("\n✓ Server is responding with valid JSON judge output.")
    else:
        print("\n✗ Server responded but output was not valid JSON.")
        sys.exit(1)


# ── File mode ─────────────────────────────────────────────────────────────────

def run_file(client: OpenAI, model: str, path: str):
    print(f"Mode: file  ({path})")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    training = data.get("training", {})
    problem  = training.get("problem", "")
    code     = training.get("stripped_code", "")
    tests    = training.get("all_tests", [])

    if not code:
        print("ERROR: training.stripped_code is empty in this file.")
        print("       This file predates the training block — re-run the pipeline.")
        sys.exit(1)

    gt_score     = data.get("node3", {}).get("total_score")
    gt_pass_rate = data.get("node2", {}).get("pass_rate")

    print(f"  q_id: {data.get('question_id')}  iter: {data.get('iter')}  model: {data.get('model')}")
    raw, elapsed = call_judge(client, problem, code, tests, model)
    parsed = parse_output(raw)
    print_result(raw, parsed, elapsed, gt_score, gt_pass_rate)


# ── Dataset mode ──────────────────────────────────────────────────────────────

def run_dataset(client: OpenAI, model: str, data_dir: str, n: int, split: str):
    if not HF_AVAILABLE:
        print("ERROR: pip install datasets")
        sys.exit(1)

    print(f"Mode: dataset  ({data_dir}  split={split}  n={n})")
    dataset = load_from_disk(data_dir)
    examples = list(dataset[split])[:n]

    parse_ok = 0
    score_diffs = []

    for i, ex in enumerate(examples):
        raw, elapsed = call_judge(client, ex["problem"], ex["stripped_code"],
                                  ex["all_tests"], model)
        parsed = parse_output(raw)
        ok = parsed is not None
        if ok:
            parse_ok += 1
            diff = abs(parsed.get("total_score", 0) - ex["total_score"])
            score_diffs.append(diff)

        gt_info = f"GT={ex['total_score']:.0f}  pass_rate={ex['node2_pass_rate']:.2f}"
        status  = f"{'OK' if ok else 'FAIL':4s}  {elapsed:.1f}s  {gt_info}"
        pred_score = parsed.get("total_score", "?") if ok else "?"
        print(f"  [{i+1:3d}/{n}]  pred={pred_score}  {status}")

    print(f"\n── Summary ──────────────────────────────────────────────────")
    print(f"  JSON parse rate : {parse_ok}/{n}  ({parse_ok/n:.0%})")
    if score_diffs:
        import statistics
        print(f"  Score MAE vs GT : {statistics.mean(score_diffs):.1f} pts")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Test the running judge model server")
    parser.add_argument("--config",  default="config.yaml")
    parser.add_argument("--host",    default=None,    help="Server base URL, e.g. http://localhost:8004/v1")
    parser.add_argument("--port",    default=None,    type=int)
    parser.add_argument("--model",   default="judge", help="Model name sent in API request")
    parser.add_argument("--file",    default=None,    help="Path to a single iteration JSON")
    parser.add_argument("--dataset", default=None,    help="Path to HF dataset dir")
    parser.add_argument("--split",   default="test",  choices=["train", "val", "test"])
    parser.add_argument("--n",       default=20,      type=int, help="Max examples in dataset mode")
    args = parser.parse_args()

    cfg  = load_config(args.config)
    port = args.port or cfg["serve"]["port"]
    base_url = args.host or f"http://localhost:{port}/v1"

    print(f"Connecting to: {base_url}")
    client = OpenAI(base_url=base_url, api_key="local")

    # Quick connectivity check
    try:
        client.models.list()
    except Exception as e:
        print(f"ERROR: Cannot reach {base_url}")
        print(f"       Start the server first: bash serve.sh")
        print(f"       ({e})")
        sys.exit(1)

    if args.file:
        run_file(client, args.model, args.file)
    elif args.dataset:
        run_dataset(client, args.model, os.path.abspath(args.dataset), args.n, args.split)
    else:
        run_smoke(client, args.model)


if __name__ == "__main__":
    main()
