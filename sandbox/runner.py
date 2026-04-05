"""
Runtime metrics measurement module.

Phase 1 (lcb_runner) measures correctness.
This module ONLY measures cpu_ms and ram_mb.
Runs on public tests — 2–3 tests are sufficient.
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
    test_inputs: list,
    timeout_sec: int = None,
) -> dict:
    """
    Runs code on public tests and measures runtime metrics.

    Args:
        code:        Code to execute
        test_inputs: public_tests["input"] — 2–3 tests sufficient
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
        try:
            os.unlink(tmp)
        except OSError:
            pass

    avg_cpu = sum(cpu_times) / len(cpu_times) if cpu_times else 0.0

    return {
        "cpu_ms":    round(avg_cpu, 1),
        "ram_mb":    round(peak_ram, 1),
        "timed_out": timed_out,
        "error":     error,
    }


def _run_single(tmp_path: str, stdin_data: str, timeout: int) -> dict:
    """Runs a single test case in a subprocess."""

    def _set_limits():
        """RAM limit — Linux only."""
        if sys.platform.startswith("linux"):
            import resource
            limit = settings.sandbox_memory_limit_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

    try:
        t0 = time.perf_counter()

        proc = subprocess.Popen(
            [sys.executable, tmp_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=_set_limits if sys.platform.startswith("linux") else None,
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
