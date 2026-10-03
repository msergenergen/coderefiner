"""
Runtime metrics measurement module (Node 2, Phase 2).

Phase 1 (sandbox/lcb_eval.py) measures correctness; this module only measures
how expensive the code is on the public tests:

  cpu_ms  — CPU time (user + system) of the child process, averaged over tests
  ram_mb  — peak resident memory of the child process (max over tests)

Both are read with os.wait4() when the child exits, so they describe that
process only. Until 2026-10-03 the module read memory with psutil *after* the
process had exited (always 0 MB) and reported wall-clock time; LeetCode-style
problems were also run as plain scripts, so the solution method was never
called and only interpreter start-up was timed. The main experiment of
2026-10-02 ran with that version (docs/node2.md).

Functional (LeetCode) problems use the same call driver as the correctness
evaluator: the starter method is called with the JSON arguments of each test.
Both numbers include interpreter start-up (~15 ms, ~10 MB).
"""
import json
import os
import subprocess
import sys
import tempfile
import time

from config import settings
from sandbox.lcb_eval import _CALL_DRIVER, _LCB_IMPORT_HEADER, _starter_method

# ru_maxrss is in bytes on macOS and in kilobytes on Linux
_MAXRSS_PER_MB = 1024 * 1024 if sys.platform == "darwin" else 1024


def measure_runtime(
    code: str,
    test_inputs: list,
    timeout_sec: int = None,
    starter_code: str = "",
) -> dict:
    """
    Runs code on public tests and measures runtime metrics.

    Args:
        code:         Code to execute
        test_inputs:  public_tests["input"] — 2–3 tests sufficient
        timeout_sec:  None → settings.sandbox_timeout_sec
        starter_code: LeetCode starter code; when it names a method the tests
                      are function calls, otherwise stdin programs

    Returns:
        {
            "cpu_ms":    float,   # Average CPU time (user + system)
            "ram_mb":    float,   # Peak resident memory
            "timed_out": bool,
            "error":     str | None,
        }
    """
    timeout = timeout_sec or settings.sandbox_timeout_sec
    method = _starter_method(starter_code)

    cpu_times = []
    peak_ram  = 0.0
    timed_out = False
    error     = None
    tmp_files = []

    def _tmp(content: str, suffix: str) -> str:
        with tempfile.NamedTemporaryFile(suffix=suffix, mode="w", delete=False) as f:
            f.write(content)
        tmp_files.append(f.name)
        return f.name

    try:
        src = _tmp((_LCB_IMPORT_HEADER + code) if method else code, ".py")
        driver = _tmp(_CALL_DRIVER, ".py") if method else None
        for inp in test_inputs[:3]:   # At most 3 tests — speed is priority
            if method:
                try:
                    args = _tmp(json.dumps([json.loads(line) for line in inp.split("\n")]), ".json")
                except ValueError as e:
                    error = error or f"bad test input: {str(e)[:100]}"
                    continue
                result = _run_single([sys.executable, driver, src, args, method], "", timeout)
            else:
                result = _run_single([sys.executable, src], inp, timeout)
            cpu_times.append(result["cpu_ms"])
            peak_ram = max(peak_ram, result["ram_mb"])
            if result["timed_out"]:
                timed_out = True
            if result["error"] and not error:
                error = result["error"]
    finally:
        for path in tmp_files:
            try:
                os.unlink(path)
            except OSError:
                pass

    avg_cpu = sum(cpu_times) / len(cpu_times) if cpu_times else 0.0

    return {
        "cpu_ms":    round(avg_cpu, 1),
        "ram_mb":    round(peak_ram, 1),
        "timed_out": timed_out,
        "error":     error,
    }


def _set_limits():
    """RAM limit — Linux only (RLIMIT_AS is not enforced on macOS)."""
    if sys.platform.startswith("linux"):
        import resource
        limit = settings.sandbox_memory_limit_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))


def _run_single(cmd: list, stdin_data: str, timeout: int) -> dict:
    """Runs one test in a child process and reads its resource usage on exit.

    stdin/stderr go through temporary files instead of pipes so that the
    child can be reaped with os.wait4(), which returns its own rusage;
    Popen.communicate() would reap it without exposing that."""
    try:
        with tempfile.TemporaryFile("w+") as fin, tempfile.TemporaryFile("w+") as ferr:
            fin.write(stdin_data)
            fin.seek(0)
            proc = subprocess.Popen(
                cmd, stdin=fin, stdout=subprocess.DEVNULL, stderr=ferr, text=True,
                preexec_fn=_set_limits if sys.platform.startswith("linux") else None,
            )
            deadline = time.perf_counter() + timeout
            while True:
                pid, status, usage = os.wait4(proc.pid, os.WNOHANG)
                if pid:
                    break
                if time.perf_counter() > deadline:
                    proc.kill()
                    os.wait4(proc.pid, 0)
                    proc.returncode = -9
                    return {"cpu_ms": timeout * 1000, "ram_mb": 0.0,
                            "timed_out": True, "error": "timeout"}
                time.sleep(0.002)
            proc.returncode = os.waitstatus_to_exitcode(status)
            ferr.seek(0)
            stderr = ferr.read()

        return {
            "cpu_ms": (usage.ru_utime + usage.ru_stime) * 1000,
            "ram_mb": usage.ru_maxrss / _MAXRSS_PER_MB,
            "timed_out": False,
            "error": stderr.strip()[-500:] if proc.returncode != 0 else None,
        }
    except Exception as e:
        return {
            "cpu_ms": 0.0, "ram_mb": 0.0,
            "timed_out": False, "error": str(e)[:200],
        }
