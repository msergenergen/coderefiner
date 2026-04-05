"""
Rule-based edge case generation for LCB questions with few test cases.

Rule: Generate edge cases when total all_tests count is below EDGE_CASE_THRESHOLD.
Does not require an LLM — deterministic generation based on problem constraints.
"""
import re

EDGE_CASE_THRESHOLD = 8   # Add edge cases when below this count


def needs_edge_cases(row: dict) -> bool:
    """Returns True if all_tests total is below the threshold."""
    return len(row["all_tests"]["input"]) < EDGE_CASE_THRESHOLD


def generate_edge_cases(row: dict) -> list:
    """
    Generates rule-based edge cases from problem content.

    Returns:
        [{"input": str, "output": None}, ...]
        output=None → correct answer unknown; test only that code doesn't crash

    Note:
        output=None cases are not used for pass/fail scoring,
        only for runtime error detection.
        Node 2 counts these cases as "crash tests" separately.
    """
    cases = []
    content = row["content"].lower()

    # 1. Empty input — applicable to almost every problem
    cases.append({"input": "0\n", "output": None})

    # 2. Single-element list
    if "list" in content or "array" in content or "nums" in content:
        cases.append({"input": "1\n0\n",  "output": None})
        cases.append({"input": "1\n-1\n", "output": None})

    # 3. Large N — TLE test
    if re.search(r"10\^[456]|1000000|100000", content):
        n = 100_000
        cases.append({
            "input":  f"{n}\n" + " ".join(["1"] * n) + "\n",
            "output": None
        })

    # 4. Negative numbers
    if re.search(r"negative|integer|int", content):
        cases.append({"input": "3\n-1 -2 -3\n", "output": None})

    # 5. All zeros
    cases.append({"input": "3\n0 0 0\n", "output": None})

    return cases


def augment_row(row: dict) -> dict:
    """
    Adds edge cases to all_tests if needed.
    Does not modify public_tests or private_tests.

    Returns:
        Updated row (edge_cases field added)
    """
    if not needs_edge_cases(row):
        return {**row, "edge_cases": []}

    edge = generate_edge_cases(row)
    return {**row, "edge_cases": edge}
