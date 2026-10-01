# node2.md — Sandbox & Execution (Node 2)

## 1. Purpose

Evaluate the generated code in two independent phases:

| Phase | Tool | Measured |
|-----|------|---------|
| **Faz 1 — Correctness** | lcb_runner.evaluation | pass/fail, pass_rate, failed_cases |
| **Faz 2 — Runtime** | Kendi subprocess runner | cpu_ms, ram_mb, timeout |

The two phases run independently. Phase 1 measures correctness;
Phase 2 measures performance. Even if Phase 1 finds failing tests,
Phase 2 runs only on `public_tests` — for speed.

---

## 2. Faz 1 — lcb_runner Correctness Evaluation

### Neden lcb_runner?

lcb_runner is LCB's official evaluation engine. Using the official method is a methodological
necessity for the thesis claim that the pipeline improves LCB task success rates.
Runs on `all_tests` (public + private + edge_cases).

> **Verified gap:** every iteration JSON in the only pilot run so far
> (`outputs/results/2026-04-15/`) has `lcb_available: false` — `lcb_runner` was not
> installed and `_fallback_evaluate()` ran instead (exact-string-match; `crash_count`
> and `test_diversity_score` are hardcoded placeholders, not measurements — see
> `sandbox/lcb_eval.py:194-202`). `dataset/verify.py` now checks and warns on this.
> **Before any run whose results are meant to support the thesis claim above,
> confirm `lcb_runner: OK` in `dataset/verify.py` output — do not trust
> `crash_count`/`test_diversity_score` from a run where `lcb_available` is `false`.**

### How lcb_runner Works

The lcb_runner `check_correctness()` function runs the code inside a Process
using `exec()`. For each test case it:
- Compares against the expected output
- Enforces timeout
- Exception yakalar
- Returns per-test pass/fail

### Important: lcb_runner Bugs

There are documented code extraction bugs in lcb_runner:
- Truncates code after `###` tokens
- Treats empty backtick blocks as valid

Bu nedenle **kodu biz extract edip lcb_runner'a veriyoruz** —
We do not use lcb_runner's extraction pipeline — we only call
`check_correctness()` directly.
`codegen_node` + `strip_comments_node` already prepares clean code.

### sandbox/lcb_eval.py

```python
"""
lcb_runner evaluation katmanını saran adaptör.

lcb_runner tüm pipeline'ını değil, sadece check_correctness()
fonksiyonunu kullanır. Kodu biz sağlarız, lcb_runner test eder.
"""
import time
from typing import Optional

try:
    from lcb_runner.evaluation.utils_execute import check_correctness
    LCB_AVAILABLE = True
except ImportError:
    LCB_AVAILABLE = False


def lcb_evaluate(
    code: str,
    problem_row: dict,
    edge_cases: list[dict],
    timeout: int = 10,
) -> dict:
    """
    lcb_runner ile correctness değerlendirmesi yapar.

    Args:
        code:        Çalıştırılacak Python kodu (temiz, comment'siz)
        problem_row: load_lcb() çıktısı — all_tests içerir
        edge_cases:  dataset/edge_cases.py çıktısı (output=None olanlar)
        timeout:     Her test için saniye cinsinden limit

    Returns:
        {
            "pass_rate":     float,       # 0.0–1.0 (only tests with known expected output)
            "passed_count":  int,
            "total_count":   int,
            "failed_cases":  list[dict],  # max 5, {input, expected, got, error}
            "crash_count":          int,         # Runtime error count in output=None tests
            "test_diversity_score": float,       # 1 - crash_count/len(edge_cases)
            "lcb_available": bool,
        }
    """
    if not LCB_AVAILABLE:
        return _fallback_evaluate(code, problem_row, timeout)

    all_inputs  = problem_row["all_tests"]["input"]
    all_outputs = problem_row["all_tests"]["output"]

    results      = []
    failed_cases = []

    # ── Tests with known expected output (public + private) ─────────────────────
    for inp, expected in zip(all_inputs, all_outputs):
        result = _run_single_lcb(code, inp, expected, timeout)
        results.append(result)

        if not result["passed"] and len(failed_cases) < 5:
            failed_cases.append({
                "input":    inp,
                "expected": expected,
                "got":      result["got"],
                "error":    result["error"],
            })

    passed = sum(1 for r in results if r["passed"])
    total  = len(results)

    # ── Edge case'ler (output=None → sadece crash testi) ──────────────
    crash_count = 0
    for ec in edge_cases:
        result = _run_single_lcb(code, ec["input"], expected=None, timeout=timeout)
        if result.get("error"):
            crash_count += 1

    return {
        "pass_rate":            passed / total if total > 0 else 0.0,
        "passed_count":         passed,
        "total_count":          total,
        "failed_cases":         failed_cases,
        "crash_count":          crash_count,
        "test_diversity_score": _compute_diversity(crash_count, edge_cases),
        "lcb_available":        True,
    }


def _run_single_lcb(
    code: str,
    inp: str,
    expected: Optional[str],
    timeout: int,
) -> dict:
    """
    Tek test case için lcb_runner check_correctness() çağırır.

    lcb_runner'ın check_correctness() imzası sürüme göre değişebilir.
    Bu fonksiyon en yaygın imzayı hedefler; kurulum sonrası doğrula.
    """
    try:
        # lcb_runner check_correctness call
        # Signature may vary by version — check lcb_runner README
        result = check_correctness(
            problem={"input": inp, "output": expected or ""},
            completion=code,
            timeout=timeout,
        )

        # result genellikle: {"passed": bool, "result": str}
        passed = bool(result.get("passed", False))
        got    = result.get("result", "")
        error  = None if passed else result.get("result", "wrong_answer")

        return {"passed": passed, "got": got, "error": error}

    except Exception as e:
        return {"passed": False, "got": "", "error": str(e)[:200]}


def _compute_diversity(crash_count: int, edge_cases: list[dict]) -> float:
    """
    Edge case'lerin kaçının runtime error olmadan geçtiğini ölçer.

    Hesaplama:
        edge_cases boşsa → 1.0 (nötr)
        edge_cases doluysa → 1 - (crash_count / len(edge_cases))

    Yorumlama:
        1.0  → tüm edge case'ler çalıştı (crash yok)
        0.5  → yarısı crash etti
        0.0  → tümü crash etti

    Not: pass/fail değil, sadece "çalıştı mı?" ölçülüyor.
    Output=None case'lerde doğru cevap bilinmiyor;
    crash etmemesi yeterli koşul.
    """
    if not edge_cases:
        return 1.0
    score = 1.0 - (crash_count / len(edge_cases))
    return round(max(0.0, score), 3)


def _fallback_evaluate(
    code: str,
    problem_row: dict,
    timeout: int,
) -> dict:
    """
    lcb_runner kurulu değilse devreye giren basit subprocess evaluator.
    Tez ortamında bu fallback çalışmamalı; kurulum doğrulama bunu yakalar.
    """
    import subprocess, tempfile, os

    all_inputs  = problem_row["all_tests"]["input"]
    all_outputs = problem_row["all_tests"]["output"]

    results      = []
    failed_cases = []

    with tempfile.NamedTemporaryFile(
        suffix=".py", mode="w", delete=False
    ) as f:
        f.write(code)
        tmp = f.name

    try:
        for inp, expected in zip(all_inputs, all_outputs):
            try:
                proc = subprocess.run(
                    ["python3", tmp],
                    input=inp, capture_output=True,
                    text=True, timeout=timeout,
                )
                got    = proc.stdout.strip()
                passed = (got == expected.strip())
                error  = proc.stderr.strip() if proc.returncode != 0 else None
            except subprocess.TimeoutExpired:
                got, passed, error = "", False, "timeout"

            results.append({"passed": passed, "got": got, "error": error})
            if not passed and len(failed_cases) < 5:
                failed_cases.append({
                    "input": inp, "expected": expected,
                    "got": got, "error": error,
                })
    finally:
        os.unlink(tmp)

    passed = sum(1 for r in results if r["passed"])
    total  = len(results)

    return {
        "pass_rate":            passed / total if total > 0 else 0.0,
        "passed_count":         passed,
        "total_count":          total,
        "failed_cases":         failed_cases,
        "crash_count":          0,
        "test_diversity_score": 1.0,   # edge cases not run in fallback mode, neutral value
        "lcb_available":        False,
    }
```

---

## 3. Faz 2 — Runtime Metrics (Kendi Subprocess Runner)

Runs only on `public_tests` (2–3 tests).
Does not touch correctness — measures cpu_ms and ram_mb.
RAM limit enforced via `resource.setrlimit` (Linux only).

### sandbox/runner.py

```python
"""
Runtime metrik ölçüm modülü.

Faz 1 (lcb_runner) correctness'ı ölçtü.
Bu modül YALNIZCA cpu_ms ve ram_mb ölçer.
Public testlerde çalışır — 2–3 test yeterli.
"""
import os
import sys
import time
import subprocess
import tempfile
import psutil
from config import settings


def measure_runtime(
    code: str,
    test_inputs: list[str],
    timeout_sec: int | None = None,
) -> dict:
    """
    Kodu public testlerde çalıştırır, runtime metriklerini ölçer.

    Args:
        code:        Çalıştırılacak kod
        test_inputs: public_tests["input"] — 2–3 test yeterli
        timeout_sec: None → settings.sandbox_timeout_sec

    Returns:
        {
            "cpu_ms":    float,   # Average CPU time
            "ram_mb":    float,   # Peak RAM usage
            "timed_out": bool,
            "error":     str | None,
        }
    """
    timeout = timeout_sec or settings.sandbox_timeout_sec

    cpu_times = []
    peak_ram  = 0.0
    timed_out = False
    error     = None

    with tempfile.NamedTemporaryFile(
        suffix=".py", mode="w", delete=False
    ) as f:
        f.write(code)
        tmp = f.name

    try:
        for inp in test_inputs[:3]:   # At most 3 tests — speed is priority
            result = _run_single(tmp, inp, timeout)
            cpu_times.append(result["cpu_ms"])
            peak_ram = max(peak_ram, result["ram_mb"])
            if result["timed_out"]:
                timed_out = True
            if result["error"] and not error:
                error = result["error"]
    finally:
        os.unlink(tmp)

    avg_cpu = sum(cpu_times) / len(cpu_times) if cpu_times else 0.0

    return {
        "cpu_ms":    round(avg_cpu, 1),
        "ram_mb":    round(peak_ram, 1),
        "timed_out": timed_out,
        "error":     error,
    }


def _run_single(tmp_path: str, stdin_data: str, timeout: int) -> dict:
    """Tek test case için subprocess çalıştırır."""

    def _set_limits():
        """RAM limiti — sadece Linux."""
        if sys.platform.startswith("linux"):
            import resource
            limit = settings.sandbox_memory_limit_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

    try:
        t0 = time.perf_counter()

        proc = subprocess.Popen(
            ["python3", tmp_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=_set_limits,
        )

        try:
            _, stderr = proc.communicate(input=stdin_data, timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            return {
                "cpu_ms": timeout * 1000, "ram_mb": 0.0,
                "timed_out": True, "error": "timeout",
            }

        elapsed_ms = (time.perf_counter() - t0) * 1000

        ram_mb = 0.0
        try:
            p = psutil.Process(proc.pid)
            ram_mb = p.memory_info().rss / (1024 * 1024)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

        error = stderr.strip() if proc.returncode != 0 else None

        return {
            "cpu_ms": elapsed_ms, "ram_mb": ram_mb,
            "timed_out": False, "error": error,
        }

    except Exception as e:
        return {
            "cpu_ms": 0.0, "ram_mb": 0.0,
            "timed_out": False, "error": str(e)[:200],
        }
```

---

## 4. nodes/sandbox.py — Node That Combines Both Phases

```python
"""
LangGraph Node 2: Sandbox & Execution

Faz 1: lcb_runner ile correctness (all_tests)
Faz 2: Kendi runner ile runtime metrics (public_tests)
"""
import time
from pipeline.state import PipelineState
from sandbox.lcb_eval import lcb_evaluate
from sandbox.runner  import measure_runtime
from dataset.edge_cases import augment_row


def sandbox_node(state: PipelineState) -> PipelineState:
    """
    Node 2: İki fazlı evaluation.

    Giriş  (state'ten okur):
        generated_code, problem_row, test_inputs (public)

    Çıktı  (state'e yazar):
        node2_pass_rate, node2_passed_count, node2_total_count,
        node2_cpu_ms, node2_ram_mb, node2_timed_out,
        node2_crash_count, node2_failed_cases,
        node2_lcb_available
    """
    t0   = time.perf_counter()
    code = state["generated_code"]

    problem_row = state.get("problem_row", {})
    edge_cases  = state.get("edge_cases", [])
    pub_inputs  = state.get("test_inputs", [])   # public_tests["input"]

    # ── Faz 1: lcb_runner correctness ──────────────────────────────────
    faz1 = lcb_evaluate(
        code=code,
        problem_row=problem_row,
        edge_cases=edge_cases,
        timeout=10,
    )

    # ── Faz 2: Runtime metrikler (public testlerde) ────────────────────
    faz2 = {"cpu_ms": 0.0, "ram_mb": 0.0, "timed_out": False, "error": None}

    if pub_inputs:
        faz2 = measure_runtime(
            code=code,
            test_inputs=pub_inputs,
        )

    node2_ms = int((time.perf_counter() - t0) * 1000)

    # Update timing in state
    timing = dict(state.get("timing", {}))
    timing["node2_ms"]  = node2_ms
    timing["total_ms"]  = sum(
        timing.get(k, 0)
        for k in ("codegen_ms", "node1_ms", "node2_ms", "node3_ms")
    )

    return {
        **state,
        # Correctness (Faz 1)
        "node2_pass_rate":            faz1["pass_rate"],
        "node2_passed_count":         faz1["passed_count"],
        "node2_total_count":          faz1["total_count"],
        "node2_failed_cases":         faz1["failed_cases"],
        "node2_crash_count":          faz1["crash_count"],
        "node2_test_diversity_score": faz1["test_diversity_score"],
        "node2_lcb_available":        faz1["lcb_available"],
        # Runtime (Faz 2)
        "node2_cpu_ms":    faz2["cpu_ms"],
        "node2_ram_mb":    faz2["ram_mb"],
        "node2_timed_out": faz2["timed_out"],
        # Timing
        "timing": timing,
    }
```

---

## 5. State Update

`PipelineState`'e eklenmesi gereken yeni alanlar:

```python
# pipeline/state.py — new fields for Node 2

# Correctness (Faz 1 — lcb_runner)
node2_pass_rate:     float         # 0.0–1.0
node2_passed_count:  int           # Number of passed tests
node2_total_count:   int           # Total test count (public + private)
node2_failed_cases:  list[dict]    # Max 5, {input, expected, got, error}
node2_crash_count:            int           # Runtime error count in edge cases
node2_test_diversity_score:   float         # 1 - crash_count/len(edge_cases)
node2_lcb_available:          bool          # lcb_runner kurulu mu?

# Runtime (Faz 2 — subprocess)
node2_cpu_ms:    float             # Average CPU time (public testler)
node2_ram_mb:    float             # Peak RAM usage
node2_timed_out: bool              # Did a timeout occur?

# Problem context (main.py'de initial_state'e eklenir)
problem_row:  dict                 # load_lcb() output (includes all_tests)
edge_cases:   list[dict]           # augment_row() output
```

---

## 6. main.py Update

`initial_state`'e `problem_row` ve `edge_cases` eklenmeli:

```python
# main.py — inside run_pipeline_for_problem

from dataset.edge_cases import augment_row

def run_pipeline_for_problem(problem_row: dict, run_id: str) -> dict:
    # Generate edge cases — augment problem_row
    aug_row   = augment_row(problem_row)
    edge_cases = aug_row.get("edge_cases", [])

    pub_inputs  = problem_row["public_tests"].get("input",  [])
    pub_outputs = problem_row["public_tests"].get("output", [])

    for model_id in ["M1", "M2", "M3"]:
        initial_state = {
            ...
            "problem_row":  aug_row,          # all_tests dahil
            "edge_cases":   edge_cases,        # output=None case'ler
            "test_inputs":  pub_inputs,        # For Phase 2 (runtime metrics)
            "test_outputs": pub_outputs,
            ...
        }
```

---

## 7. Connection to Node 3

Node 2 outputs are passed to Node 3 as follows:

```python
# In node3's user prompt (nodes/judgement.py):

## Node 2 — Execution Results

| Metric              | Value       | Threshold  |
|---------------------|-------------|------------|
| Test pass rate      | 0.65        | ≥ 0.90     |
| Tests passed        | 13 / 20     |            |
| CPU time (ms)       | 340         | ≤ 2000     |
| RAM usage (MB)      | 128         | ≤ 256      |
| Crash tests failed  | 1           | 0          |
| lcb_runner used     | True        |            |

### Failing Test Cases (max 5)

| # | Input              | Expected | Got  | Error |
|---|--------------------|----------|------|-------|
| 1 | []                 | 0        | 12   | —     |
| 2 | [-2, -1, 0, 1, 2]  | 0        | 12   | —     |
```

---

## 8. Verification

### lcb_runner Installation Check

```bash
python -c "
from sandbox.lcb_eval import LCB_AVAILABLE
if LCB_AVAILABLE:
    print('lcb_runner: OK')
else:
    print('lcb_runner: KURULU DEĞİL — fallback runner çalışır')
    print('Kur: pip install -e LiveCodeBench/')
"
```

### Two-Phase Node Test

```bash
python -c "
import os
os.makedirs('outputs', exist_ok=True)

from sandbox.lcb_eval import lcb_evaluate
from sandbox.runner import measure_runtime

code = '''
n = int(input())
nums = list(map(int, input().split()))
print(sum(x for x in nums if x % 2 == 0))
'''

problem_row = {
    'all_tests': {
        'input':  ['6\n1 2 3 4 5 6', '5\n1 3 5 7 9', '4\n-2 -1 0 1'],
        'output': ['12',              '0',             '-2'],
    }
}

# Faz 1
faz1 = lcb_evaluate(code, problem_row, edge_cases=[], timeout=5)
print('Faz 1 — pass_rate:  ', faz1['pass_rate'])
print('Faz 1 — passed:     ', faz1['passed_count'], '/', faz1['total_count'])
print('Faz 1 — lcb_runner: ', faz1['lcb_available'])

# Faz 2
faz2 = measure_runtime(code, ['6\n1 2 3 4 5 6', '5\n1 3 5 7 9'])
print('Faz 2 — cpu_ms:     ', faz2['cpu_ms'])
print('Faz 2 — ram_mb:     ', faz2['ram_mb'])
print('Faz 2 — timed_out:  ', faz2['timed_out'])
"
```

### Expected Output

```
Faz 1 — pass_rate:   1.0
Faz 1 — passed:      3 / 3
Faz 1 — lcb_runner:  True        (False ise fallback çalışır)
Faz 2 — cpu_ms:      18.4
Faz 2 — ram_mb:      9.2
Faz 2 — timed_out:   False
```

---

## 9. Dikkat Edilecekler

**lcb_runner:**
- The `check_correctness()` signature may vary by LCB version.
  After installation, verify the call in `sandbox/lcb_eval.py` against the actual signature.
- lcb_runner is `exec()`-based — no process isolation.
  The Bandit + whitelist check in Node 1 mitigates this risk.
- `LCB_AVAILABLE = False` ise fallback runner devreye girer;
  this should not occur in the thesis environment.

**Runtime runner:**
- `resource.setrlimit` works only on Linux.
  On Mac Mini local debug, this line is skipped via a `sys.platform` check.
- Phase 2 is independent of Phase 1; Phase 2 runs even if Phase 1 has failures.
- If `public_tests` is empty, Phase 2 does not run and cpu_ms stays 0.0.

**Edge case'ler:**
- Cases with `output=None` are not included in `pass_rate` calculation.
- `crash_count` is reported separately in the Node 3 prompt.


---

## Evaluator selection (`LCB_EVALUATOR`)

`lcb_evaluate()` uses the LCB-compatible evaluator in `sandbox/lcb_eval.py`
(`_fallback_evaluate`) unless `LCB_EVALUATOR=lcb_runner` **and** `lcb_runner` is importable.
Default `local`: every reported result (Mac runs and the offline hidden-test grading in
`benchmark/paper_results.py`) used this evaluator, so a cloud machine that happens to have
`lcb_runner` installed must not silently switch graders. Correctness timeout stays 10 s per test
and at most 5 failing cases are passed to the judge (both fixed in code; `SANDBOX_TIMEOUT_SEC`
only affects the runtime measurement in `sandbox/runner.py`).
