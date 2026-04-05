"""
LangGraph Node 1: Static Analysis

Analyzes generated code without executing it and makes a severity decision.
"""
from pipeline.state import PipelineState
from metrics.complexity import (
    cyclomatic_complexity,
    maintainability_index,
    halstead_volume,
    cognitive_complexity,
)
from metrics.security import bandit_scan, check_imports
from metrics.linting  import mypy_check, ruff_check


def static_analysis_node(state: PipelineState) -> PipelineState:
    """
    Node 1: Runs static analysis and determines severity.

    Reads from state: generated_code
    Writes to state:  node1_cc, node1_mi, node1_halstead,
                      node1_bandit_high, node1_severity,
                      node1_issues  ← human-readable findings for Node 3
    """
    import time
    t0 = time.perf_counter()

    code = state["generated_code"]

    # Compute metrics
    cc        = cyclomatic_complexity(code)
    mi        = maintainability_index(code)
    halstead  = halstead_volume(code)
    cognitive = cognitive_complexity(code)
    bandit    = bandit_scan(code)
    forbidden = check_imports(code)
    mypy      = mypy_check(code)
    ruff      = ruff_check(code)

    # Severity decision
    severity = _decide_severity(
        cc=cc,
        mi=mi,
        halstead=halstead,
        bandit_high_count=bandit["high_count"],
        forbidden_imports=forbidden,
    )

    # Human-readable issue list for Node 3
    issues = _collect_issues(
        cc=cc, mi=mi, halstead=halstead,
        bandit=bandit, forbidden=forbidden,
        mypy=mypy, ruff=ruff,
    )

    node1_ms = int((time.perf_counter() - t0) * 1000)
    timing = dict(state.get("timing", {}))
    timing["node1_ms"] = node1_ms
    timing["total_ms"] = sum(
        timing.get(k, 0)
        for k in ("codegen_ms", "node1_ms", "node2_ms", "node3_ms")
    )

    return {
        **state,
        "node1_cc":                  cc,
        "node1_mi":                  mi,
        "node1_halstead":            halstead,
        "node1_bandit_high":         bandit["high_count"] > 0,
        "node1_bandit_medium_count": bandit["medium_count"],
        "node1_severity":            severity,
        "node1_issues":              issues,
        "timing":                    timing,
    }


def _collect_issues(
    cc: float,
    mi: float,
    halstead: float,
    bandit: dict,
    forbidden: list,
    mypy: dict,
    ruff: dict,
) -> list:
    """
    Produces human-readable issue sentences from numeric metrics.
    These are added directly to the Node 3 judgment prompt.
    """
    issues = []

    if cc > 10:
        issues.append(
            f"Cyclomatic Complexity {cc:.0f} — threshold 10. "
            "Function has too many branches; should be split into smaller parts."
        )
    if mi < 65:
        issues.append(
            f"Maintainability Index {mi:.1f} — threshold 65. "
            "Code is hard to maintain; improve variable naming or structure."
        )
    if halstead > 1000:
        issues.append(
            f"Halstead Volume {halstead:.0f} — threshold 1000. "
            "High operator/operand count; simplify expressions."
        )
    for issue in bandit.get("issues", []):
        sev  = issue.get("issue_severity", "")
        text = issue.get("issue_text", "")
        line = issue.get("line_number", "?")
        issues.append(f"Security [{sev}] line {line}: {text}")
    for imp in forbidden:
        issues.append(f"Forbidden import: `{imp}` — not in whitelist.")
    for err in mypy.get("errors", []):
        issues.append(f"Type error (mypy): {err}")
    for issue in ruff.get("issues", []):
        issues.append(f"Lint (ruff): {issue}")

    return issues


def _decide_severity(
    cc: float,
    mi: float,
    halstead: float,
    bandit_high_count: int,
    forbidden_imports: list,
) -> str:
    """
    Makes a severity decision based on thresholds.
    Priority: HIGH > LOW > OK
    """
    # HIGH — critical, do not send to sandbox
    if bandit_high_count > 0:
        return "HIGH"
    if forbidden_imports:
        return "HIGH"
    if cc > 20:
        return "HIGH"

    # LOW — warnings present, skip Node 2
    if mi < 40:
        return "LOW"
    if halstead > 2000:
        return "LOW"

    return "OK"
