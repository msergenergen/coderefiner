"""
Adapter wrapping the lcb_runner evaluation layer.

Uses only check_correctness() from lcb_runner's pipeline — not the full pipeline.
We supply the code ourselves; lcb_runner runs the tests.
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
    edge_cases: list,
    timeout: int = 10,
) -> dict:
    """
    Correctness evaluation with lcb_runner.

    Args:
        code:        Python code to execute (clean, comment-free)
        problem_row: load_lcb() output — contains all_tests
        edge_cases:  dataset/edge_cases.py output (output=None cases)
        timeout:     Per-test time limit in seconds

    Returns:
        {
            "pass_rate":            float,     # 0.0–1.0 (only tests with known expected output)
            "passed_count":         int,
            "total_count":          int,
            "failed_cases":         list[dict], # max 5, {input, expected, got, error}
            "crash_count":          int,        # Runtime error count in output=None tests
            "test_diversity_score": float,      # 1 - crash_count/len(edge_cases)
            "lcb_available":        bool,
        }
    """
    from config import settings
    if not LCB_AVAILABLE or settings.lcb_evaluator != "lcb_runner":
        return _fallback_evaluate(code, problem_row, timeout)

    all_inputs  = problem_row["all_tests"]["input"]
    all_outputs = problem_row["all_tests"]["output"]

    results      = []
    failed_cases = []

    # ── Tests with known expected output (public + private) ───────────────
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

    # ── Edge cases (output=None → crash test only) ─────────────────────
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
    Calls lcb_runner check_correctness() for a single test case.

    The check_correctness() signature may vary by lcb_runner version.
    This function targets the most common signature; verify after installation.
    """
    try:
        result = check_correctness(
            problem={"input": inp, "output": expected or ""},
            completion=code,
            timeout=timeout,
        )

        # result typically: {"passed": bool, "result": str}
        passed = bool(result.get("passed", False))
        got    = result.get("result", "")
        error  = None if passed else result.get("result", "wrong_answer")

        return {"passed": passed, "got": got, "error": error}

    except Exception as e:
        return {"passed": False, "got": "", "error": str(e)[:200]}


def _compute_diversity(crash_count: int, edge_cases: list) -> float:
    """
    Measures what fraction of edge cases ran without a runtime error.

    Computation:
        edge_cases empty → 1.0 (neutral)
        edge_cases non-empty → 1 - (crash_count / len(edge_cases))

    Interpretation:
        1.0  → all edge cases ran (no crashes)
        0.5  → half crashed
        0.0  → all crashed

    Note: measures only "did it run?", not pass/fail.
    output=None cases have no known correct answer;
    not crashing is the only requirement.
    """
    if not edge_cases:
        return 1.0
    score = 1.0 - (crash_count / len(edge_cases))
    return round(max(0.0, score), 3)


# Mirrors lcb_runner's import header so starter-code signatures such as
# `List[int]` resolve even when the model omits `from typing import List`.
_LCB_IMPORT_HEADER = (
    "from string import *\nfrom re import *\nfrom datetime import *\n"
    "from collections import *\nfrom heapq import *\nfrom bisect import *\n"
    "from copy import *\nfrom math import *\nfrom random import *\n"
    "from statistics import *\nfrom itertools import *\nfrom functools import *\n"
    "from operator import *\nfrom io import *\nfrom sys import *\nfrom json import *\n"
    "from builtins import *\nfrom typing import *\n"
    "import string, re, datetime, collections, heapq, bisect, copy, math, random\n"
    "import statistics, itertools, functools, operator, io, sys, json\n"
    "sys.setrecursionlimit(50000)\n"
)

_RESULT_MARKER = "__LCB_RESULT__"

# Runs the candidate as a module (not __main__, so a stdin harness under
# `if __name__ == "__main__"` stays inert), then calls the starter-code
# method with the JSON-decoded arguments, as lcb_runner does for
# LeetCode-style ("functional") tests.
_CALL_DRIVER = r"""
import json, sys
src_path, args_path, method = sys.argv[1], sys.argv[2], sys.argv[3]
ns = {"__name__": "__lcb_candidate__"}
try:
    exec(compile(open(src_path).read(), "solution.py", "exec"), ns)
except SystemExit:
    pass
except BaseException:
    if "Solution" not in ns:
        raise
args = json.load(open(args_path))
pred = getattr(ns["Solution"](), method)(*args)
if isinstance(pred, tuple):
    pred = list(pred)
sys.stdout.write("\n" + MARKER + json.dumps(pred))
""".replace("MARKER", repr(_RESULT_MARKER))


def _stripped_lines(text: str) -> list:
    return [line.strip() for line in text.strip().split("\n")]


def _stdio_match(got: str, expected: str) -> bool:
    """lcb_runner's stdio grading: line-wise after strip, then numeric
    tokens compared as Decimals so that e.g. "1.0" matches "1"."""
    from decimal import Decimal, InvalidOperation
    got_lines, exp_lines = _stripped_lines(got), _stripped_lines(expected)
    if len(got_lines) != len(exp_lines):
        return False
    for g, e in zip(got_lines, exp_lines):
        if g == e:
            continue
        try:
            if [Decimal(t) for t in g.split()] != [Decimal(t) for t in e.split()]:
                return False
        except (InvalidOperation, ValueError):
            return False
    return True


# Per-test limit for correctness on the public tests (Node 2). The V0
# baseline grades its public tests with the same value so filt@k matches
# what the pipeline sees.
CORRECTNESS_TIMEOUT_SEC = 10


def _starter_method(starter_code: str) -> Optional[str]:
    import re
    match = re.search(r"def\s+(\w+)\s*\(\s*self", starter_code or "")
    return match.group(1) if match else None


def _fallback_evaluate(
    code: str,
    problem_row: dict,
    timeout: int,
    stop_on_fail: bool = False,
) -> dict:
    """
    Local LCB-compatible evaluator (used because lcb_runner is not installed).

    - LeetCode problems (starter code present, "functional" tests): the
      starter method is called directly with json.loads of each input line
      and the result is compared with json.loads(expected output).
    - AtCoder/Codeforces problems ("stdin" tests): the program reads the
      input from stdin; stdout is compared line by line as in lcb_runner.

    Earlier versions compared raw stdout strings for both kinds, so a
    correct LeetCode method failed unless the model also wrote a stdin
    harness whose printed output matched the JSON text exactly (Python's
    `True` vs JSON `true`, quoted strings, ...). In a 52-problem run that
    left LeetCode strict accuracy at 7.7% against 69.2% on AtCoder.

    stop_on_fail=True stops at the first failing test, as lcb_runner's
    grade_stdio / grade_call_based do. Pass/fail of the whole problem is
    unchanged; pass_rate is then a lower bound (passed so far / all tests).
    Used for offline hidden grading, where only "all tests pass" is reported.
    """
    import json
    import os
    import subprocess
    import sys
    import tempfile

    all_inputs  = problem_row["all_tests"]["input"]
    all_outputs = problem_row["all_tests"]["output"]
    method      = _starter_method(problem_row.get("starter_code", ""))

    results      = []
    failed_cases = []
    tmp_files    = []

    def _tmp(content: str, suffix: str) -> str:
        with tempfile.NamedTemporaryFile(suffix=suffix, mode="w", delete=False) as f:
            f.write(content)
        tmp_files.append(f.name)
        return f.name

    try:
        if method:
            src    = _tmp(_LCB_IMPORT_HEADER + code, ".py")
            driver = _tmp(_CALL_DRIVER, ".py")
        else:
            src = _tmp(code, ".py")

        for inp, expected in zip(all_inputs, all_outputs):
            got, error, passed = "", None, False
            try:
                if method:
                    args_path = _tmp(json.dumps([json.loads(l) for l in inp.split("\n")]), ".json")
                    proc = subprocess.run(
                        [sys.executable, driver, src, args_path, method],
                        stdin=subprocess.DEVNULL, capture_output=True,
                        text=True, timeout=timeout,
                    )
                    if _RESULT_MARKER in proc.stdout:
                        got = proc.stdout.rsplit(_RESULT_MARKER, 1)[1].strip()
                        passed = json.loads(got) == json.loads(expected)
                    else:
                        error = proc.stderr.strip()[-500:] or "no result"
                else:
                    proc = subprocess.run(
                        [sys.executable, src],
                        input=inp, capture_output=True,
                        text=True, timeout=timeout,
                    )
                    got    = proc.stdout.strip()
                    passed = _stdio_match(got, expected)
                    if proc.returncode != 0:
                        error = proc.stderr.strip()[-500:]
            except subprocess.TimeoutExpired:
                got, passed, error = "", False, "timeout"
            except (ValueError, OSError) as e:
                passed, error = False, str(e)[:200]

            results.append(passed)
            if not passed and len(failed_cases) < 5:
                failed_cases.append({
                    "input": inp, "expected": expected,
                    "got": got, "error": error,
                })
            if not passed and stop_on_fail:
                break
    finally:
        for path in tmp_files:
            try:
                os.unlink(path)
            except OSError:
                pass

    passed = sum(results)
    total  = len(all_inputs)

    return {
        "pass_rate":            passed / total if total > 0 else 0.0,
        "passed_count":         passed,
        "total_count":          total,
        "failed_cases":         failed_cases,
        "crash_count":          0,
        "test_diversity_score": 1.0,   # edge cases not run in fallback mode
        "lcb_available":        False,
    }
