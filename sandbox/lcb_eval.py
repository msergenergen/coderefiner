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
    if not LCB_AVAILABLE:
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


def _fallback_evaluate(
    code: str,
    problem_row: dict,
    timeout: int,
) -> dict:
    """
    Simple subprocess evaluator used when lcb_runner is not installed.
    This fallback should not run in the thesis environment;
    the setup verification script catches this case.
    """
    import subprocess
    import tempfile

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
        import os
        try:
            os.unlink(tmp)
        except OSError:
            pass

    passed = sum(1 for r in results if r["passed"])
    total  = len(results)

    return {
        "pass_rate":            passed / total if total > 0 else 0.0,
        "passed_count":         passed,
        "total_count":          total,
        "failed_cases":         failed_cases,
        "crash_count":          0,
        "test_diversity_score": 1.0,   # edge cases not run in fallback mode
        "lcb_available":        False,
    }
