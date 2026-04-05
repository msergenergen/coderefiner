"""
Shared fixtures — used by all test files.
"""
import pytest
from pipeline.state import PipelineState


# ── Code constants ────────────────────────────────────────────────────────

CLEAN_CODE = """\
def two_sum(nums, target):
    seen = {}
    for i, n in enumerate(nums):
        if target - n in seen:
            return [seen[target - n], i]
        seen[n] = i
    return []
"""

CLEAN_CODE_WITH_COMMENTS = """\
# Find the sum of two numbers
def two_sum(nums, target):
    seen = {}  # hash map
    for i, n in enumerate(nums):  # iterate each element
        if target - n in seen:
            return [seen[target - n], i]  # found
        seen[n] = i
    return []
"""

COMMENTED_WITH_DOCSTRING = """\
def two_sum(nums, target):
    \"\"\"İki sayının indekslerini döner.\"\"\"
    # hash map approach
    seen = {}
    return seen  # placeholder
"""

DANGEROUS_CODE = """\
import os
import subprocess
exec(input())
os.system("rm -rf /")
"""

COMPLEX_CODE = """\
def nightmare(a, b, c, d, e):
    if a:
        if b:
            if c:
                if d:
                    if e:
                        for i in range(10):
                            for j in range(10):
                                if i == j:
                                    while True:
                                        break
    return 0
"""

STDIN_CODE = """\
n = int(input())
nums = list(map(int, input().split()))
print(sum(x for x in nums if x % 2 == 0))
"""


# ── State fixture ────────────────────────────────────────────────────────

@pytest.fixture
def base_state():
    """Returns a minimal valid PipelineState. Can be overridden in tests."""
    def _make(**overrides) -> PipelineState:
        state = {
            "run_id":         "test_run",
            "active_model":   "M1",
            "iteration":      1,
            "max_iterations": 10,
            "problem":        "Test problem",
            "problem_row":    {
                "id": "q_test", "title": "Test", "difficulty": "easy",
                "content": "Test content", "starter_code": "",
                "public_tests":  {"input": ["3\n1 2 3"], "output": ["2"]},
                "private_tests": {"input": [],            "output": []},
                "all_tests":     {"input": ["3\n1 2 3"], "output": ["2"]},
            },
            "question_id":    "q_test",
            "difficulty":     "easy",
            "test_inputs":    ["3\n1 2 3"],
            "test_outputs":   ["2"],
            "edge_cases":     [],
            "generated_code": CLEAN_CODE,
            "stripped_code":  CLEAN_CODE,
            "node1_cc":          0.0,
            "node1_mi":          100.0,
            "node1_halstead":    0.0,
            "node1_bandit_high": False,
            "node1_severity":    "OK",
            "node1_issues":      [],
            "node2_pass_rate":            0.0,
            "node2_passed_count":         0,
            "node2_total_count":          0,
            "node2_failed_cases":         [],
            "node2_crash_count":          0,
            "node2_test_diversity_score": 1.0,
            "node2_lcb_available":        False,
            "node2_cpu_ms":    0.0,
            "node2_ram_mb":    0.0,
            "node2_timed_out": False,
            "node3_total_score":     0.0,
            "node3_pass":            False,
            "node3_score_breakdown": {},
            "node3_failure_reasons": [],
            "node3_improved_prompt": "",
            "timing": {
                "codegen_ms": 0, "node1_ms": 0,
                "node2_ms": 0,   "node3_ms": 0, "total_ms": 0
            },
            "tokens": {
                "codegen_prompt": 0, "codegen_completion": 0,
                "judgement_prompt": 0, "judgement_completion": 0, "total": 0
            },
            "memory_log":   [],
            "final_status": "PENDING",
        }
        state.update(overrides)
        return state
    return _make
