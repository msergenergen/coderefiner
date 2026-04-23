"""
Two-layer logging module.

  memory.md          — summary table, one row per iteration
  run_info.json      — environment + model info, written once per run
  iterations/<q_id>/ — full detail JSON, one file per iteration
"""
import os
import json
import platform
import subprocess
from datetime import datetime
from typing import Optional

from pipeline.state import PipelineState
from config import settings, model_registry


# ── Hardware info collection ──────────────────────────────────────────────────

def _collect_hardware() -> dict:
    """Collects GPU, CPU, and RAM information."""
    info = {
        "gpu_name":    "unknown",
        "gpu_vram_gb": 0,
        "gpu_count":   0,
        "ram_gb":      0,
        "cpu_model":   platform.processor() or "unknown",
        "cpu_cores":   os.cpu_count() or 0,
        "os":          f"{platform.system()} {platform.release()}",
        "cuda":        "unknown",
        "llama_cpp":   "unknown",
    }

    # GPU — nvidia-smi
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,count",
             "--format=csv,noheader,nounits"],
            text=True, timeout=5
        ).strip().split("\n")
        if out:
            parts = out[0].split(", ")
            info["gpu_name"]    = parts[0].strip()
            info["gpu_vram_gb"] = round(int(parts[1]) / 1024)
            info["gpu_count"]   = len(out)

        cuda_out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True, timeout=5
        ).strip()
        info["cuda"] = cuda_out
    except Exception:
        pass  # nvidia-smi not available on local (Mac) — skip silently

    # RAM — psutil
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / (1024 ** 3))
    except ImportError:
        pass

    # llama.cpp version
    try:
        llama_path = os.path.expanduser("~/llama.cpp/llama-server")
        out = subprocess.check_output(
            [llama_path, "--version"],
            text=True, timeout=3, stderr=subprocess.STDOUT
        )
        info["llama_cpp"] = out.strip().split("\n")[0] if out.strip() else "unknown"
    except Exception:
        pass

    return info


def _dated_output_dir() -> str:
    """Returns settings.output_dir / YYYY-MM-DD, creating it if needed."""
    dated = os.path.join(settings.output_dir, datetime.now().strftime("%Y-%m-%d"))
    os.makedirs(dated, exist_ok=True)
    return dated


def write_run_info() -> str:
    """
    Called once at the start of a run.
    Writes environment + model info to run_info.json.
    Returns: run_id
    """
    run_id   = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir  = _dated_output_dir()
    path     = os.path.join(out_dir, "run_info.json")

    payload = {
        "run_id":    run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "env":       settings.env,
        "hardware":  _collect_hardware(),
        "models": {
            **{m: model_registry.get(m).model_name for m in settings.model_sequence},
            settings.judge_model: model_registry.get(settings.judge_model).model_name,
        },
        "pipeline_config": {
            "max_iterations":      settings.max_iterations,
            "lcb_split":           settings.lcb_split,
            "lcb_difficulty":      settings.lcb_difficulty,
            "lcb_sample_size":     settings.lcb_sample_size,
            "sandbox_timeout_sec": settings.sandbox_timeout_sec,
            "pass_threshold": {
                "easy":   80,
                "medium": 70,
                "hard":   60,
            },
            "trend_extension": {
                "enabled":          settings.trend_extension_enabled,
                "max_extension_iters": settings.max_extension_iters,
                "min_slope":        settings.trend_min_slope,
                "ema_alpha":        settings.trend_ema_alpha,
            },
        },
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return run_id


# ── memory.md ─────────────────────────────────────────────────────────────────

_MD_HEADER = (
    "| iter | model | model_name"
    " | q_id | diff | cc | mi | severity"
    " | pass_rt | score | status"
    " | total_ms | total_tok | json |\n"
    "|------|-------|-----------|"
    "------|------|-----|------|----------|"
    "---------|-------|-----------|"
    "----------|-----------|------|\n"
)


def _ensure_md_header(run_id: str) -> None:
    path = settings.memory_log_path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            date_str = datetime.now().strftime("%Y-%m-%d")
            f.write(f"# Pipeline Run Log — {run_id} — {date_str}\n\n")
            f.write(_MD_HEADER)


def _append_md_row(state: PipelineState, status: str,
                   run_id: str, json_path: str,
                   total_ms: int, total_tok: int) -> None:
    _ensure_md_header(run_id)

    model_name = model_registry.get(state["active_model"]).model_name
    q_id       = state.get("question_id", "unknown")
    # Use relative path for portability
    try:
        rel_path = os.path.relpath(
            json_path,
            start=os.path.dirname(settings.memory_log_path) or "."
        )
    except ValueError:
        rel_path = json_path

    row = (
        f"| {state['iteration']:4d} "
        f"| {state['active_model']:5s} "
        f"| {model_name:17s} "
        f"| {q_id:5s} "
        f"| {state['difficulty']:5s} "
        f"| {state.get('node1_cc', 0.0):3.1f} "
        f"| {state.get('node1_mi', 0.0):4.1f} "
        f"| {state.get('node1_severity', 'N/A'):8s} "
        f"| {state.get('node2_pass_rate', 0.0):7.2f} "
        f"| {state.get('node3_total_score', 0.0):5.1f} "
        f"| {status:9s} "
        f"| {total_ms:8d} "
        f"| {total_tok:9d} "
        f"| {rel_path} |\n"
    )

    with open(settings.memory_log_path, "a", encoding="utf-8") as f:
        f.write(row)


# ── Iteration JSON ────────────────────────────────────────────────────────────

def _write_iteration_json(state: PipelineState, status: str,
                           run_id: str, timing: dict, tokens: dict) -> str:
    """
    Writes the detail JSON for a single iteration.
    Returns: file path
    """
    q_id       = state.get("question_id", "unknown")
    model      = state["active_model"]
    iter_n     = state["iteration"]
    model_name = model_registry.get(model).model_name

    dir_path = os.path.join(
        _dated_output_dir(), "iterations", q_id
    )
    os.makedirs(dir_path, exist_ok=True)
    file_path = os.path.join(dir_path, f"{model}_iter{iter_n}.json")

    payload = {
        "run_id":      run_id,
        "iter":        iter_n,
        "model":       model,
        "model_name":  model_name,
        "question_id": q_id,
        "difficulty":  state["difficulty"],
        "variant":     state.get("variant", "V3"),
        "status":      status,

        "timing": timing,

        "tokens": tokens,

        "node1": {
            "cc":                  state.get("node1_cc", 0.0),
            "mi":                  state.get("node1_mi", 0.0),
            "halstead":            state.get("node1_halstead", 0.0),
            "severity":            state.get("node1_severity", "N/A"),
            "bandit_high":         state.get("node1_bandit_high", False),
            "bandit_medium_count": state.get("node1_bandit_medium_count", 0),
            "issues":              state.get("node1_issues", []),
        },

        "node2": {
            "pass_rate":            state.get("node2_pass_rate", 0.0),
            "passed_count":         state.get("node2_passed_count", 0),
            "total_count":          state.get("node2_total_count", 0),
            "crash_count":          state.get("node2_crash_count", 0),
            "lcb_available":        state.get("node2_lcb_available", False),
            "cpu_ms":               state.get("node2_cpu_ms", 0.0),
            "ram_mb":               state.get("node2_ram_mb", 0.0),
            "timed_out":            state.get("node2_timed_out", False),
            "test_diversity_score": state.get("node2_test_diversity_score", 0.0),
            "failed_cases":         state.get("node2_failed_cases", []),
        },

        "node3": {
            "total_score":     state.get("node3_total_score", 0.0),
            "pass":            state.get("node3_pass", False),
            "breakdown":       state.get("node3_score_breakdown", {}),
            "failure_reasons": state.get("node3_failure_reasons", []),
            "improved_prompt": state.get("node3_improved_prompt", ""),
        },

        # Fields required for judge model training (Prometheus-style SFT).
        # stripped_code is what the judge actually receives — training input.
        # problem is the formatted problem statement — training context.
        # all_tests are reference tests used as rubric ground truth.
        "training": {
            "stripped_code": state.get("stripped_code", ""),
            "problem":       state.get("problem", ""),
            "all_tests":     state.get("problem_row", {}).get("all_tests", []),
        },

        "trend": {
            "extension_count":  state.get("extension_count", 0),
            "score_history":    state.get("score_history", []),
            "pass_rate_history": state.get("pass_rate_history", []),
        },
    }

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return file_path


# ── Public node functions ─────────────────────────────────────────────────────

def log_iteration_node(state: PipelineState) -> PipelineState:
    """
    LangGraph node: logs each iteration after judgement.
    Writes LOOP / PASS / STOP_MAX statuses.
    Also maintains score_history and pass_rate_history for trend-aware extension.
    """
    from config import thresholds, settings
    threshold = thresholds.node3.for_difficulty(state["difficulty"])
    passed    = state["node3_total_score"] >= threshold and state["node3_pass"]

    if passed:
        status = "PASS"
    elif state["iteration"] >= state["max_iterations"]:
        status = "STOP_MAX"
    else:
        status = "LOOP"

    run_id  = state.get("run_id", "unknown")
    timing  = state.get("timing", {"codegen_ms": 0, "node1_ms": 0,
                                    "node2_ms": 0, "node3_ms": 0, "total_ms": 0})
    tokens  = state.get("tokens", {"codegen_prompt": 0, "codegen_completion": 0,
                                    "judgement_prompt": 0, "judgement_completion": 0,
                                    "total": 0})

    json_path  = _write_iteration_json(state, status, run_id, timing, tokens)
    total_ms   = timing.get("total_ms", 0)
    total_tok  = tokens.get("total", 0)

    _append_md_row(state, status, run_id, json_path, total_ms, total_tok)

    # Append current iteration metrics to history lists for trend-aware extension.
    # extension_count is incremented only when this iteration ran beyond max_iterations.
    score_history     = list(state.get("score_history",     []))
    pass_rate_history = list(state.get("pass_rate_history", []))
    score_history.append(state.get("node3_total_score", 0.0))
    pass_rate_history.append(state.get("node2_pass_rate", 0.0))

    extension_count = state.get("extension_count", 0)
    if state["iteration"] > state["max_iterations"]:
        extension_count += 1

    return {
        **state,
        "score_history":     score_history,
        "pass_rate_history": pass_rate_history,
        "extension_count":   extension_count,
    }


def log_severity_stop_node(state: PipelineState) -> PipelineState:
    """
    LangGraph node: logs on severity HIGH.
    Timing and tokens are partial since Node 2/3 did not run.
    """
    run_id = state.get("run_id", "unknown")
    timing = state.get("timing", {"codegen_ms": 0, "node1_ms": 0,
                                   "node2_ms": 0, "node3_ms": 0, "total_ms": 0})
    tokens = state.get("tokens", {"codegen_prompt": 0, "codegen_completion": 0,
                                   "judgement_prompt": 0, "judgement_completion": 0,
                                   "total": 0})

    json_path = _write_iteration_json(state, "STOP_HIGH", run_id, timing, tokens)
    total_ms  = timing.get("total_ms", 0)
    total_tok = tokens.get("total", 0)

    _append_md_row(state, "STOP_HIGH", run_id, json_path, total_ms, total_tok)
    return {**state, "final_status": "STOP_HIGH"}


# ── Read functions ────────────────────────────────────────────────────────────

def read_memory(path: Optional[str] = None) -> list:
    """
    Parses memory.md and returns a list of dicts.
    Used for analysis and ablation.
    """
    path = path or settings.memory_log_path
    if not os.path.exists(path):
        return []

    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("|") or "| iter |" in line or "---" in line:
                continue
            cells = [c.strip() for c in line.split("|")[1:-1]]
            if len(cells) < 14:
                continue
            try:
                rows.append({
                    "iter":       int(cells[0]),
                    "model":      cells[1],
                    "model_name": cells[2],
                    "q_id":       cells[3],
                    "difficulty": cells[4],
                    "cc":         float(cells[5]),
                    "mi":         float(cells[6]),
                    "severity":   cells[7],
                    "pass_rt":    float(cells[8]),
                    "score":      float(cells[9]),
                    "status":     cells[10],
                    "total_ms":   int(cells[11]),
                    "total_tok":  int(cells[12]),
                    "json":       cells[13],
                })
            except (ValueError, IndexError):
                continue
    return rows


def read_iteration(q_id: str, model: str, iter_n: int,
                   date: str = None) -> Optional[dict]:
    """
    Reads the JSON detail for a specific iteration.
    date: YYYY-MM-DD string; defaults to today.
    """
    date_dir = date or datetime.now().strftime("%Y-%m-%d")
    path = os.path.join(
        settings.output_dir, date_dir, "iterations", q_id, f"{model}_iter{iter_n}.json"
    )
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)
