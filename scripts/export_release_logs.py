#!/usr/bin/env python
"""
Build release archive: coderefiner-eleco2026-logs.zip

Collects iteration JSONs, baseline samples, analysis results, and API logs for
the main experiment (2026-10-02, DeepSeek-V3 + Qwen2.5-72B judge) and related trials.

Strips LiveCodeBench problem statements from iteration JSONs to respect content licensing.
Filters API usage log to the included runs.

No modifications to outputs/ — only reads and builds the zip file.
"""

import json
import os
import sys
import zipfile
from pathlib import Path
from typing import Set

# Main experiment run IDs (from docs/main-results.md §11 command block and §2-11 text)
# Format: design_description: [run_ids]
RUN_IDS_BY_SETTING = {
    # V0 baseline (5 samples/problem; .regraded = re-graded offline with the pinned local evaluator)
    "baseline": ["run_20261002_091834.regraded"],

    # §2-8: Main experiment (Y = judge-verdict, P = rubric score)
    "main_Y": [
        "run_20261002_100638", "run_20261002_100640",
        "run_20261002_100642", "run_20261002_100644",
    ],
    "main_P": [
        "run_20261002_100646", "run_20261002_100648",
        "run_20261002_100650", "run_20261002_100652",
    ],

    # §6: Ablation A, feedback only (docs/ablation.md "Koşu kayıtları"); shard Y-0 shares
    # its second-stamped id with the V0 baseline file
    "ablation_A_Y": [
        "run_20261002_091834", "run_20261002_091836",
        "run_20261002_091838", "run_20261002_091840",
    ],
    "ablation_A_P": [
        "run_20261002_091842", "run_20261002_091844",
        "run_20261002_091846", "run_20261002_091848",
    ],

    # §11: Judge-effect trials (gpt-oss-120b judge, different generators)
    "trial_C_gptoss": [
        "run_20261003_093242", "run_20261003_093244",
        "run_20261003_093246", "run_20261003_093248",
    ],
    "trial_D_llama_gptoss": [
        "run_20261003_100935", "run_20261003_100937",
        "run_20261003_100939", "run_20261003_100941",
    ],

    # Baseline for trial D (Llama-3.1-8B single-sample)
    "baseline_D": ["run_20261003_093410"],
}

# Flatten to a single set for filtering
ALL_RUN_IDS = set()
for run_list in RUN_IDS_BY_SETTING.values():
    ALL_RUN_IDS.update(run_list)


def strip_lcb_content(iteration_data: dict) -> dict:
    """
    Remove LiveCodeBench problem statements and test definitions.

    Strips:
    - training.problem (problem statement)
    - training.all_tests (all test cases)

    Keeps:
    - training.generated_code, training.stripped_code
    - node1/2/3 metrics and verdicts
    - question_id, model, status, tokens, timing
    - judge feedback (improved_prompt, verdict, verdict_reason)
    """
    if "training" in iteration_data and isinstance(iteration_data["training"], dict):
        training = iteration_data["training"]
        training.pop("problem", None)
        training.pop("all_tests", None)
    return iteration_data


def collect_iteration_jsons(outputs_dir: Path, run_ids: Set[str]) -> list:
    """Find all iteration JSONs matching the run_ids."""
    iteration_dir = outputs_dir / "openrouter" / "results"
    results = []

    if not iteration_dir.exists():
        print(f"Warning: iterations dir not found: {iteration_dir}")
        return results

    # Iterate over date directories (e.g., 2026-10-02, 2026-10-03)
    for date_dir in iteration_dir.glob("20*"):
        if not date_dir.is_dir():
            continue

        iterations_dir = date_dir / "iterations"
        if not iterations_dir.exists():
            continue

        # For each question_id directory
        for qid_dir in iterations_dir.glob("*"):
            if not qid_dir.is_dir():
                continue

            # For each iteration JSON matching a run_id
            for json_file in qid_dir.glob("*.json"):
                filename = json_file.name
                # Pattern: <model>_iter<n>_<run_id>.json
                for run_id in run_ids:
                    if run_id in filename:
                        results.append(json_file)
                        break

    return sorted(results)


def collect_baseline_jsonls(outputs_dir: Path, run_ids: Set[str]) -> list:
    """Find baseline JSONL files matching the run_ids."""
    baseline_dir = outputs_dir / "openrouter" / "results" / "baseline_samples"
    results = []

    if not baseline_dir.exists():
        print(f"Warning: baseline dir not found: {baseline_dir}")
        return results

    for jsonl_file in baseline_dir.glob("*.jsonl"):
        filename = jsonl_file.name
        # Pattern: <run_id>.jsonl or <run_id>.regraded.jsonl
        for run_id in run_ids:
            if run_id in filename:
                results.append(jsonl_file)
                break

    return sorted(results)


def filter_api_usage(api_usage_file: Path, run_ids: Set[str]) -> str:
    """Filter API usage log to only the included run_ids. Return as string."""
    if not api_usage_file.exists():
        print(f"Warning: api_usage.jsonl not found: {api_usage_file}")
        return ""

    filtered_lines = []
    with open(api_usage_file, "r") as f:
        for line in f:
            try:
                record = json.loads(line)
                if record.get("run_id") in run_ids:
                    filtered_lines.append(line)
            except json.JSONDecodeError:
                continue

    return "".join(filtered_lines)


def filter_hidden_cache(cache_file: Path, run_ids: Set[str]) -> dict:
    """Hidden-test pass rates (key run_id/question_id/iter) of the included runs."""
    if not cache_file.exists():
        print(f"Warning: hidden_eval_cache.json not found: {cache_file}")
        return {}
    with open(cache_file) as f:
        cache = json.load(f)
    return {k: v for k, v in cache.items() if k.split("/", 1)[0] in run_ids}


def collect_analysis_files(outputs_dir: Path) -> dict:
    """Collect analysis results.json and report.md from each analysis_* directory."""
    analysis_dir = outputs_dir / "openrouter"
    results = {}

    if not analysis_dir.exists():
        return results

    for subdir in analysis_dir.glob("analysis_*"):
        if not subdir.is_dir():
            continue

        # results.json
        results_json = subdir / "results.json"
        if results_json.exists():
            results[str(results_json.relative_to(outputs_dir))] = results_json

        # report.md
        report_md = subdir / "report.md"
        if report_md.exists():
            results[str(report_md.relative_to(outputs_dir))] = report_md

    return results


def collect_judge_replay(outputs_dir: Path) -> list:
    """Collect judge_replay/*.jsonl and report.json files."""
    judge_replay_dir = outputs_dir / "openrouter" / "judge_replay"
    results = []

    if not judge_replay_dir.exists():
        return results

    for file in judge_replay_dir.glob("*"):
        if file.suffix in [".jsonl", ".json"]:
            results.append(file)

    return sorted(results)


def find_run_info_files(outputs_dir: Path, run_ids: Set[str]) -> list:
    """Find run_info_*.json files matching the run_ids."""
    results_dir = outputs_dir / "openrouter" / "results"
    results = []

    if not results_dir.exists():
        return results

    for date_dir in results_dir.glob("20*"):
        for run_info in date_dir.glob("run_info_*.json"):
            filename = run_info.name
            # Pattern: run_info_<run_id>.json
            for run_id in run_ids:
                if run_id in filename:
                    results.append(run_info)
                    break

    return sorted(results)


def create_data_readme() -> str:
    """Create a DATA_README.md for the archive."""
    return """# CodeRefiner — ELECO 2026 experiment logs

Logs behind the paper "LLM-as-a-Judge for Iterative Code Refinement: A Multi-Stage Approach"
(M. S. Ergen, A. Sayar; ELECO 2026, Bursa, 26-28 November 2026). Code:
https://github.com/msergenergen/coderefiner (see CITATION.cff there).

Problems: 150 LiveCodeBench problems released 2025-01..2025-04 (seed-42 sample of 182).
Correctness = all public + private tests pass, graded offline with the repository's local
LCB-compatible evaluator (`sandbox/lcb_eval.py`).

## Runs

| Setting | Generator | Judge | Run ids |
|---|---|---|---|
| V0 baseline (5 samples/problem) | DeepSeek-V3 | - | `run_20261002_091834` (`.regraded.jsonl` is the graded file) |
| B, design Y (judge verdict; main) | DeepSeek-V3 | Qwen2.5-72B-Instruct | `run_20261002_100638/40/42/44` |
| B, design P (rubric threshold) | DeepSeek-V3 | Qwen2.5-72B-Instruct | `run_20261002_100646/48/50/52` |
| A, feedback only (ablation), Y | DeepSeek-V3 | Qwen2.5-72B-Instruct | `run_20261002_091834/36/38/40` |
| A, feedback only (ablation), P | DeepSeek-V3 | Qwen2.5-72B-Instruct | `run_20261002_091842/44/46/48` |
| C, design Y | DeepSeek-V3 | gpt-oss-120b | `run_20261003_093242/44/46/48` |
| D baseline (5 samples/problem) | Llama-3.1-8B-Instruct | - | `run_20261003_093410` |
| D, design Y | Llama-3.1-8B-Instruct | gpt-oss-120b | `run_20261003_100935/37/39/41` |

Each design ran as 4 shards (one run id per shard). Run ids are second-stamped: shard Y-0 of
ablation A started in the same second as the V0 baseline and shares its id; the baseline is the
JSONL file, the shard is the iteration JSONs. gpt-oss-120b was released after the problems, so
contamination cannot be ruled out for C and D.

## Files

### `openrouter/results/<date>/iterations/<question_id>/M1_iter<N>_<run_id>.json`
One generation attempt of the pipeline.
- `run_id`, `iter` (1-5), `question_id` (LCB id, string, e.g. `3562`, `abc394_f`), `difficulty`,
  `model` / `model_name` (generator), `variant` (`V3` = full pipeline)
- `status`: `LOOP` (rejected, next iteration), `PASS` (accepted, delivered), `STOP_MAX`
  (5 generations used, last attempt delivered), `STOP_HIGH` (stopped by the Severity Gate)
- `timing`: `codegen_ms`, `node1_ms`, `node2_ms`, `node3_ms`, `total_ms`
- `tokens`: `codegen_prompt`, `codegen_completion`, `judgement_prompt`, `judgement_completion`, `total`
- `node1` (static analysis): `cc`, `mi`, `halstead`, `bandit_high`, `bandit_medium_count`,
  `issues`, `severity` (`OK` / `LOW` / `HIGH`)
- `node2` (sandbox, public tests only): `pass_rate`, `passed_count`, `total_count`,
  `failed_cases` (public example inputs, expected and actual outputs), `timed_out`,
  `crash_count`, `cpu_ms`, `ram_mb` (runtime fields are reliable only in C and D)
- `node3` (judge): `verdict` (`ACCEPT` / `REJECT`), `verdict_reason`, `verdict_source`,
  `improved_prompt` (feedback for the next attempt), `total_score` and `breakdown` (100-point
  rubric), `pass`, `failure_reasons`
- `training`: `generated_code`, `stripped_code` (comments removed, what the judge saw)
- `trend`: trend-extension bookkeeping (off in these runs)

Removed: `training.problem` (prompt text) and `training.all_tests` (public + hidden tests).

### `openrouter/results/baseline_samples/<run_id>[.regraded].jsonl`
One single-sample generation per line: `question_id`, `difficulty`, `contest_date`, `sample`
(0-4), `code`, `finish_reason`, `completion_tokens`, `cost_usd`, `public` and `hidden` (pass
rate on the public tests and on all tests, 0-1).

### `openrouter/results/hidden_eval_cache.json`
`<run_id>/<question_id>/<iter>` -> pass rate of that attempt's code on all tests (grading stops at
the first failing test, so only 1.0 is meaningful).

### `openrouter/results/<date>/run_info_<run_id>.json`
Run metadata: models, settings, hardware.

### `openrouter/analysis_<A|B|C|D>_*/results.json`, `report.md`
Output of `benchmark/openrouter_results.py` (definitions: `docs/analysis.md` section 10 of the
repository; numbers: `docs/main-results.md`).

### `openrouter/judge_replay/`
Offline re-judging of recorded public-passing attempts by other judges
(`benchmark/judge_replay.py`, `docs/main-results.md` section 10).

### `openrouter/api_usage.jsonl`
The API ledger restricted to the runs above: model, role, provider, tokens, cost, timing,
run id, question id. No prompts, responses or keys.

## Re-joining problems and tests

With the repository installed (`datasets<3.0`):

```python
from dataset.loader import load_lcb
rows = {r["id"]: r for r in load_lcb(date_from="2025-01", date_to="2025-04")}
row = rows[record["question_id"]]          # row["content"], row["all_tests"], row["starter_code"]
```

The dataset is `livecodebench/code_generation_lite` on Hugging Face (release_latest).

## License

The logs are released under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). Generated
code and judge texts were produced by DeepSeek-V3 (DeepSeek Model License), Qwen2.5-72B-Instruct
(Qwen License), gpt-oss-120b (Apache-2.0) and Llama-3.1-8B-Instruct (Llama 3.1 Community
License); their terms may apply to these outputs. LiveCodeBench problem statements and tests
are not included.
"""


def main():
    print(f"Including {len(ALL_RUN_IDS)} run IDs:")
    for run_id in sorted(ALL_RUN_IDS):
        print(f"  {run_id}")
    print()
    project_root = Path(__file__).parent.parent
    outputs_dir = project_root / "outputs"
    release_dir = project_root / "release"

    # Create release directory
    release_dir.mkdir(exist_ok=True)

    zip_path = release_dir / "coderefiner-eleco2026-logs.zip"

    print("=" * 70)
    print("CodeRefiner — Release Archive Builder")
    print("=" * 70)
    print()

    # Collect files
    print("Collecting files...")
    iteration_jsons = collect_iteration_jsons(outputs_dir, ALL_RUN_IDS)
    print(f"  Iteration JSONs: {len(iteration_jsons)}")

    baseline_jsonls = collect_baseline_jsonls(outputs_dir, ALL_RUN_IDS)
    print(f"  Baseline JSONL files: {len(baseline_jsonls)}")

    run_info_files = find_run_info_files(outputs_dir, ALL_RUN_IDS)
    print(f"  Run info files: {len(run_info_files)}")

    analysis_files = collect_analysis_files(outputs_dir)
    print(f"  Analysis files: {len(analysis_files)}")

    judge_replay_files = collect_judge_replay(outputs_dir)
    print(f"  Judge replay files: {len(judge_replay_files)}")

    api_usage_file = outputs_dir / "api_usage.jsonl"
    api_usage_filtered = filter_api_usage(api_usage_file, ALL_RUN_IDS)
    api_usage_lines = len(api_usage_filtered.split('\n')) - 1 if api_usage_filtered else 0
    print(f"  API usage records (filtered): {api_usage_lines}")

    hidden_cache = filter_hidden_cache(outputs_dir / "openrouter" / "results" / "hidden_eval_cache.json",
                                       ALL_RUN_IDS)
    print(f"  Hidden-test cache entries (filtered): {len(hidden_cache)}")

    print()

    # Build archive
    print(f"Building archive: {zip_path}")
    total_size = 0
    file_count = 0

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add DATA_README.md
        data_readme = create_data_readme()
        zf.writestr("DATA_README.md", data_readme)
        file_count += 1

        # Add iteration JSONs (stripped)
        for json_file in iteration_jsons:
            with open(json_file, "r") as f:
                data = json.load(f)
            stripped = strip_lcb_content(data)
            arcname = f"openrouter/results/{json_file.relative_to(outputs_dir / 'openrouter' / 'results')}"
            zf.writestr(arcname, json.dumps(stripped))
            file_count += 1
            total_size += len(json.dumps(stripped))

        # Add baseline JSONL files
        for jsonl_file in baseline_jsonls:
            with open(jsonl_file, "r") as f:
                content = f.read()
            arcname = f"openrouter/results/{jsonl_file.relative_to(outputs_dir / 'openrouter' / 'results')}"
            zf.writestr(arcname, content)
            file_count += 1
            total_size += len(content)

        # Add run_info files
        for run_info in run_info_files:
            with open(run_info, "r") as f:
                content = f.read()
            arcname = f"openrouter/results/{run_info.relative_to(outputs_dir / 'openrouter' / 'results')}"
            zf.writestr(arcname, content)
            file_count += 1
            total_size += len(content)

        # Add analysis files
        for arcname, filepath in analysis_files.items():
            with open(filepath, "r") as f:
                content = f.read()
            zf.writestr(arcname, content)
            file_count += 1
            total_size += len(content)

        # Add the hidden-test cache of the included runs
        content = json.dumps(hidden_cache, indent=0, sort_keys=True)
        zf.writestr("openrouter/results/hidden_eval_cache.json", content)
        file_count += 1
        total_size += len(content)

        # Add judge replay files
        for replay_file in judge_replay_files:
            with open(replay_file, "r") as f:
                content = f.read()
            arcname = f"openrouter/judge_replay/{replay_file.name}"
            zf.writestr(arcname, content)
            file_count += 1
            total_size += len(content)

        # Add filtered API usage
        if api_usage_filtered:
            zf.writestr("openrouter/api_usage.jsonl", api_usage_filtered)
            file_count += 1
            total_size += len(api_usage_filtered)

    # Print summary
    print()
    print("=" * 70)
    print("Archive created successfully!")
    print("=" * 70)
    print(f"Path: {zip_path}")
    print(f"Total files: {file_count}")
    print(f"Uncompressed size: {total_size / (1024*1024):.1f} MB")

    # Check actual zip size
    zip_size = zip_path.stat().st_size
    print(f"Compressed size: {zip_size / (1024*1024):.1f} MB")
    print()

    if zip_size > 100 * 1024 * 1024:
        print("WARNING: Archive exceeds 100 MB; redistribution may be slower.")

    print("Contents stripped:")
    print("  - training.problem (problem statement)")
    print("  - training.all_tests (test cases)")
    print()
    print("To rejoin problem content, use LiveCodeBench (date window 2025-01..2025-04).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
