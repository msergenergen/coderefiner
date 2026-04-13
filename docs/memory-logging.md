# memory-logging.md — Memory & Logging

## 1. Output Structure

```
outputs/
├── memory.md                        ← Özet tablo — her iterasyon bir satır
├── run_info.json                    ← Ortam + model bilgisi (run başına bir kez)
└── iterations/
    ├── q_001/
    │   ├── M1_iter1.json            ← Tam detay: node çıktıları, timing, token
    │   ├── M1_iter2.json
    │   └── M2_iter1.json
    └── q_002/
        └── M1_iter1.json
```

`memory.md` stays human-readable for quick comparison and thesis tables.
JSON files provide complete data for deep analysis, debugging, and ablation.

---

## 2. run_info.json Format

Written once at the start of each pipeline run.
Contains GPU model, LLM names used, and pipeline configuration.

```json
{
  "run_id":    "run_20250115_143022",
  "timestamp": "2025-01-15T14:30:22",
  "env":       "cloud",

  "hardware": {
    "gpu_name":    "NVIDIA A100-SXM4-80GB",
    "gpu_vram_gb": 80,
    "gpu_count":   1,
    "ram_gb":      128,
    "cpu_model":   "AMD EPYC 7543 32-Core Processor",
    "cpu_cores":   32,
    "os":          "Ubuntu 22.04.3 LTS",
    "cuda":        "12.1",
    "llama_cpp":   "b4567"
  },

  "models": {
    "M1":   "gpt-oss-20b",
    "M2":   "qwen3-coder-next",
    "M3":   "glm-4.7-flash",
    "QWEN": "qwen3.5-35b-a3b"
  },

  "pipeline_config": {
    "max_iterations":     10,
    "lcb_split":          "test",
    "lcb_difficulty":     "all",
    "lcb_sample_size":    50,
    "sandbox_timeout_sec": 10,
    "pass_threshold":     { "easy": 80, "medium": 70, "hard": 60 }
  }
}
```

---

## 3. memory.md Format

One row per iteration. Details are in JSON; the table holds numeric summary + reference.

```markdown
# Pipeline Run Log — run_20250115_143022

| iter | model | model_name        | q_id  | diff  | cc  | mi   | severity | pass_rt | score | status    | total_ms | total_tok | json                          |
|------|-------|-------------------|-------|-------|-----|------|----------|---------|-------|-----------|----------|-----------|-------------------------------|
|    1 | M1    | gpt-oss-20b       | q_001 | easy  | 3.0 | 72.1 | OK       |    0.85 |  68.0 | LOOP      |     7245 |      2550 | iterations/q_001/M1_iter1.json |
|    2 | M1    | gpt-oss-20b       | q_001 | easy  | 2.0 | 81.4 | OK       |    0.95 |  84.2 | PASS      |     6810 |      2340 | iterations/q_001/M1_iter2.json |
|    1 | M2    | qwen3-coder-next  | q_001 | easy  | 1.0 | 88.0 | OK       |    1.00 |  92.5 | PASS      |     5200 |      1980 | iterations/q_001/M2_iter1.json |
|    1 | M1    | gpt-oss-20b       | q_002 | hard  | 24.0| 35.0 | HIGH     |    0.00 |   0.0 | STOP_HIGH |      320 |       450 | iterations/q_002/M1_iter1.json |
|    1 | M1    | gpt-oss-20b       | q_003 | hard  | 5.0 | 55.3 | OK       |    0.50 |  48.0 | LOOP      |     8900 |      3100 | iterations/q_003/M1_iter1.json |
|   10 | M1    | gpt-oss-20b       | q_003 | hard  | 4.0 | 61.0 | OK       |    0.62 |  57.3 | STOP_MAX  |     7600 |      2890 | iterations/q_003/M1_iter10.json |
```

**Columns:**
- `iter` — independent iteration counter per model
- `model` — pipeline role (M1/M2/M3)
- `model_name` — actual model name sent to llama.cpp
- `q_id` — LCB soru ID'si
- `diff` — easy / medium / hard
- `cc` — Cyclomatic Complexity (Node 1)
- `mi` — Maintainability Index (Node 1)
- `severity` — LOW / OK / HIGH (Node 1)
- `pass_rt` — test pass rate (Node 2), 0.00 on STOP_HIGH
- `score` — Node 3 total score (out of 100), 0.0 on STOP_HIGH
- `status` — LOOP / PASS / STOP_HIGH / STOP_MAX
- `total_ms` — total duration of the iteration (all nodes)
- `total_tok` — total tokens consumed in iteration (codegen + judgement)
- `json` — relative path to the detail file

---

## 4. Iteration JSON Format

```json
{
  "run_id":      "run_20250115_143022",
  "iter":        1,
  "model":       "M1",
  "model_name":  "gpt-oss-20b",
  "question_id": "q_001",
  "difficulty":  "easy",
  "status":      "LOOP",

  "timing": {
    "codegen_ms":  3200,
    "node1_ms":      45,
    "node2_ms":    1200,
    "node3_ms":    2800,
    "total_ms":    7245
  },

  "tokens": {
    "codegen_prompt":      450,
    "codegen_completion":  320,
    "judgement_prompt":    1200,
    "judgement_completion": 580,
    "total":              2550
  },

  "node1": {
    "cc":                  3.0,
    "mi":                 72.1,
    "halstead":          150.0,
    "severity":           "OK",
    "bandit_high":        false,
    "bandit_medium_count": 1,
    "issues": [
      "Cyclomatic Complexity 12 — eşik 10. Fonksiyon daha küçük parçalara bölünmeli.",
      "Maintainability Index 72.1 — eşik 65. Kod yapısı iyileştirilmeli."
    ]
  },

  "node2": {
    "pass_rate":            0.85,
    "passed_count":         17,
    "total_count":          20,
    "crash_count":           1,
    "lcb_available":        true,
    "cpu_ms":             340.0,
    "ram_mb":             128.0,
    "timed_out":           false,
    "test_diversity_score": 0.72,
    "failed_cases": [
      { "input": "[1, 2, 3]", "expected": "6",    "got": "None" },
      { "input": "[0]",       "expected": "0",    "got": "Error: NoneType" }
    ]
  },

  "node3": {
    "total_score": 68.0,
    "pass":        false,
    "breakdown": {
      "accuracy":         15,
      "efficiency":       14,
      "quality":          12,
      "security":         10,
      "algorithm_choice":  8,
      "error_handling":    9
    },
    "failure_reasons": [
      "Test pass rate 0.85 — easy eşiği 0.90 altında.",
      "Hata yönetimi eksik: None döndürmek yerine ValueError fırlatılmalı."
    ],
    "improved_prompt": "Bir önceki çözümde şu sorunlar tespit edildi:\n1. Boş liste girişi kontrol edilmemiş — None yerine ValueError fırlatmalısın.\n2. Test case [1,2,3] için beklenen 6 ama None döndürdün...\n\nLütfen bu sorunları gidererek yeni bir çözüm üret:"
  },

  "trend": {
    "extension_count":   0,
    "score_history":     [52.0, 68.0],
    "pass_rate_history": [0.70, 0.85]
  }
}
```

**On STOP_HIGH**, node2 and node3 blocks are written with zero/empty values:

```json
{
  "iter": 1, "model": "M1", "model_name": "gpt-oss-20b",
  "question_id": "q_002", "difficulty": "hard", "status": "STOP_HIGH",

  "timing": { "codegen_ms": 3100, "node1_ms": 42, "node2_ms": 0, "node3_ms": 0, "total_ms": 3142 },
  "tokens": { "codegen_prompt": 420, "codegen_completion": 310, "judgement_prompt": 0, "judgement_completion": 0, "total": 730 },

  "node1": {
    "cc": 24.0, "mi": 35.0, "halstead": 2400.0,
    "severity": "HIGH", "bandit_high": true,
    "issues": ["Bandit HIGH: subprocess kullanımı — yasak import.", "CC 24 — kritik eşik 20 aşıldı."]
  },
  "node2": { "pass_rate": 0.0, "passed_count": 0, "total_count": 0, "crash_count": 0, "lcb_available": false, "cpu_ms": 0.0, "ram_mb": 0.0, "timed_out": false, "test_diversity_score": 0.0, "failed_cases": [] },
  "node3": { "total_score": 0.0, "pass": false, "breakdown": {}, "failure_reasons": ["STOP_HIGH — Node 2/3 çalışmadı."], "improved_prompt": "" },
  "trend": { "extension_count": 0, "score_history": [], "pass_rate_history": [] }
}
```

---

## 5. memory/logger.py

```python
"""
İki katmanlı loglama modülü.

  memory.md          — özet tablo, her iterasyon bir satır
  run_info.json      — ortam + model bilgisi, run başına bir kez
  iterations/<q_id>/ — tam detay JSON, her iterasyon ayrı dosya
"""
import os
import json
import platform
import subprocess
from datetime import datetime
from typing import Optional

from pipeline.state import PipelineState
from config import settings, model_registry


# ── Ortam bilgisi toplama ────────────────────────────────────────────────

def _collect_hardware() -> dict:
    """GPU, CPU, RAM bilgilerini toplar."""
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

    # RAM — /proc/meminfo (Linux) veya sysctl (macOS)
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / (1024 ** 3))
    except ImportError:
        pass

    # llama.cpp versiyon
    try:
        llama = settings.llama_cpp_path if hasattr(settings, "llama_cpp_path") else "llama-server"
        out = subprocess.check_output([llama, "--version"], text=True,
                                      timeout=3, stderr=subprocess.STDOUT)
        info["llama_cpp"] = out.strip().split()[1] if len(out.split()) > 1 else out.strip()
    except Exception:
        pass

    return info


def write_run_info() -> str:
    """
    Run başında bir kez çağrılır.
    Ortam + model bilgisini run_info.json'a yazar.
    Returns: run_id
    """
    run_id    = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    path      = os.path.join(settings.output_dir, "run_info.json")
    os.makedirs(settings.output_dir, exist_ok=True)

    payload = {
        "run_id":    run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "env":       settings.env,
        "hardware":  _collect_hardware(),
        "models": {
            "M1":   model_registry.M1.model_name,
            "M2":   model_registry.M2.model_name,
            "M3":   model_registry.M3.model_name,
            "QWEN": model_registry.QWEN.model_name,
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
        },
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return run_id


# ── memory.md ────────────────────────────────────────────────────────────

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
    os.makedirs(os.path.dirname(path), exist_ok=True)
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
    # JSON path: strip outputs/ prefix, use relative path
    rel_path = os.path.relpath(json_path, start=os.path.dirname(settings.memory_log_path))

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


# ── Iteration JSON ───────────────────────────────────────────────────────

def _write_iteration_json(state: PipelineState, status: str,
                           run_id: str, timing: dict, tokens: dict) -> str:
    """
    Tek iterasyon için detay JSON'ı yazar.
    Returns: dosya yolu
    """
    q_id       = state.get("question_id", "unknown")
    model      = state["active_model"]
    iter_n     = state["iteration"]
    model_name = model_registry.get(model).model_name

    dir_path = os.path.join(
        settings.output_dir, "iterations", q_id
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
        "status":      status,

        "timing": timing,   # codegen_ms, node1_ms, node2_ms, node3_ms, total_ms

        "tokens": tokens,   # codegen_prompt, codegen_completion,
                            # judgement_prompt, judgement_completion, total

        "node1": {
            "cc":          state.get("node1_cc", 0.0),
            "mi":          state.get("node1_mi", 0.0),
            "halstead":    state.get("node1_halstead", 0.0),
            "severity":    state.get("node1_severity", "N/A"),
            "bandit_high": state.get("node1_bandit_high", False),
            "issues":      state.get("node1_issues", []),
        },

        "node2": {
            "pass_rate":      state.get("node2_pass_rate", 0.0),
            "cpu_ms":         state.get("node2_cpu_ms", 0.0),
            "ram_mb":         state.get("node2_ram_mb", 0.0),
            "test_diversity_score":  state.get("node2_test_diversity_score", 0.0),
            "failed_cases":   state.get("node2_failed_cases", []),
        },

        "node3": {
            "total_score":      state.get("node3_total_score", 0.0),
            "pass":             state.get("node3_pass", False),
            "breakdown":        state.get("node3_score_breakdown", {}),
            "failure_reasons":  state.get("node3_failure_reasons", []),
            "improved_prompt":  state.get("node3_improved_prompt", ""),
        },
    }

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return file_path


# ── Public node functions ────────────────────────────────────────────────

def log_iteration_node(state: PipelineState) -> PipelineState:
    """
    LangGraph node: judgement sonrası her iterasyonu loglar.
    LOOP / PASS / STOP_MAX durumlarını yazar.
    """
    from config import thresholds
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

    return state


def log_severity_stop_node(state: PipelineState) -> PipelineState:
    """
    LangGraph node: severity HIGH durumunda loglar.
    Node 2/3 çalışmadığından timing ve token kısmi.
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


# ── Okuma ────────────────────────────────────────────────────────────────

def read_memory(path: Optional[str] = None) -> list[dict]:
    """
    memory.md'yi parse eder ve dict listesi döner.
    Analiz ve ablation için kullanılır.
    """
    path = path or settings.memory_log_path
    if not os.path.exists(path):
        return []

    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("|") or "iter" in line or "---" in line:
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


def read_iteration(q_id: str, model: str, iter_n: int) -> Optional[dict]:
    """
    Belirli bir iterasyonun JSON detayını okur.
    """
    path = os.path.join(
        settings.output_dir, "iterations", q_id, f"{model}_iter{iter_n}.json"
    )
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)
```

---

## 6. State'e Eklenmesi Gereken Alanlar

New fields must be added to `PipelineState` to collect timing and token data from nodes:

```python
# pipeline/state.py — ek alanlar

class PipelineState(TypedDict):
    # ... (mevcut alanlar) ...

    # ── Run bilgisi ───────────────────────────────────────────────────
    run_id: str   # Set in main.py by write_run_info()

    # ── Timing (each node writes its own duration) ────────────────────────
    timing: dict   # {"codegen_ms": 0, "node1_ms": 0, "node2_ms": 0,
                   #  "node3_ms": 0, "total_ms": 0}

    # ── Token consumption ────────────────────────────────────────────────
    tokens: dict   # {"codegen_prompt": 0, "codegen_completion": 0,
                   #  "judgement_prompt": 0, "judgement_completion": 0,
                   #  "total": 0}

    # ── Node 3 score breakdown ────────────────────────────────────────────
    node3_score_breakdown: dict  # {"accuracy": 0, "efficiency": 0, "quality": 0, "security": 0, "algorithm_choice": 0, "error_handling": 0}
```

---

## 7. Node'larda Timing ve Token Toplama

Each node measures its own duration and updates the `state["timing"]` dict.
Token counts are obtained from `response.usage` returned by the OpenAI client.

```python
# nodes/codegen.py — example

import time

def codegen_node(state: PipelineState) -> PipelineState:
    t0 = time.perf_counter()

    response = client.chat.completions.create(...)
    code     = response.choices[0].message.content

    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    # Mevcut timing'e ekle
    timing = dict(state.get("timing", {}))
    timing["codegen_ms"] = elapsed_ms

    # Token counts
    tokens = dict(state.get("tokens", {}))
    if response.usage:
        tokens["codegen_prompt"]     = response.usage.prompt_tokens
        tokens["codegen_completion"] = response.usage.completion_tokens

    # Update total_ms
    timing["total_ms"] = sum(
        timing.get(k, 0)
        for k in ("codegen_ms", "node1_ms", "node2_ms", "node3_ms")
    )
    tokens["total"] = (
        tokens.get("codegen_prompt", 0) + tokens.get("codegen_completion", 0) +
        tokens.get("judgement_prompt", 0) + tokens.get("judgement_completion", 0)
    )

    return {**state, "generated_code": code, "timing": timing, "tokens": tokens}
```

The judgement node follows the same pattern and fills `judgement_prompt` and `judgement_completion`.
Node 1 and Node 2 write timing but do not consume tokens (no LLM call).

---

## 8. Verification

### run_info.json Testi

```bash
python -c "
from memory.logger import write_run_info
run_id = write_run_info()
print('run_id:', run_id)

import json, os
path = os.path.join('outputs', 'run_info.json')
info = json.load(open(path))
print('env:    ', info['env'])
print('GPU:    ', info['hardware']['gpu_name'])
print('M1:     ', info['models']['M1'])
print('M2:     ', info['models']['M2'])
"
```

### Full Log Cycle

```bash
python -c "
import os
os.makedirs('outputs', exist_ok=True)
if os.path.exists('outputs/memory.md'): os.remove('outputs/memory.md')

from memory.logger import write_run_info, log_iteration_node, log_severity_stop_node, read_memory, read_iteration

run_id = write_run_info()

def s(q='q_001', status_hint='loop', **kw):
    base = {
        'run_id': run_id, 'active_model': 'M1', 'iteration': 1,
        'max_iterations': 10, 'question_id': q, 'difficulty': 'easy',
        'problem': '', 'test_inputs': [], 'test_outputs': [],
        'generated_code': 'def solve(): return None',
        'node1_cc': 3.0, 'node1_mi': 72.0, 'node1_halstead': 150.0,
        'node1_issues': ['CC yüksek'], 'node1_bandit_high': False, 'node1_severity': 'OK',
        'node2_pass_rate': 0.65, 'node2_cpu_ms': 500.0, 'node2_ram_mb': 120.0,
        'node2_test_diversity_score': 0.6, 'node2_failed_cases': [{'input':'[]','expected':'0','got':'None'}],
        'node3_total_score': 62.0, 'node3_pass': False,
        'node3_score_breakdown': {'accuracy':10,'efficiency':12,'quality':10,'security':10,'algorithm_choice':10,'error_handling':10},
        'node3_failure_reasons': ['pass_rate düşük'], 'node3_improved_prompt': 'Boş listeyi kontrol et.',
        'timing': {'codegen_ms': 3200, 'node1_ms': 45, 'node2_ms': 1200, 'node3_ms': 2800, 'total_ms': 7245},
        'tokens': {'codegen_prompt': 450, 'codegen_completion': 320, 'judgement_prompt': 1200, 'judgement_completion': 580, 'total': 2550},
        'memory_log': [], 'final_status': 'PENDING',
    }
    base.update(kw)
    return base

log_iteration_node(s('q_001', iteration=1, node3_total_score=62, node3_pass=False))
log_iteration_node(s('q_001', iteration=2, node3_total_score=85, node3_pass=True))
log_severity_stop_node(s('q_002', node1_severity='HIGH'))

rows = read_memory()
print(f'{len(rows)} satır yazıldı:', [r['status'] for r in rows])

detail = read_iteration('q_001', 'M1', 1)
print('iter1 improved_prompt:', detail['node3']['improved_prompt'][:40])
"
```

### Expected Output

```
run_id: run_20250115_143022
env:     cloud
GPU:     NVIDIA A100-SXM4-80GB
M1:      gpt-oss-20b
M2:      qwen3-coder-next
3 satır yazıldı: ['LOOP', 'PASS', 'STOP_HIGH']
iter1 improved_prompt: Boş listeyi kontrol et.
```
