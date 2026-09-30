"""
Unit tests: sandbox/lcb_eval.py
_compute_diversity(), lcb_evaluate() fallback behavior.

Tests run without lcb_runner installed — fallback path is tested.
"""
import pytest
from sandbox.lcb_eval import _compute_diversity, LCB_AVAILABLE


class TestComputeDiversity:

    def test_no_edge_cases_returns_1(self):
        assert _compute_diversity(crash_count=0, edge_cases=[]) == 1.0

    def test_no_crashes_returns_1(self):
        edge = [{"input": "0\n", "output": None}] * 4
        assert _compute_diversity(crash_count=0, edge_cases=edge) == 1.0

    def test_all_crash_returns_0(self):
        edge = [{"input": "0\n", "output": None}] * 4
        assert _compute_diversity(crash_count=4, edge_cases=edge) == 0.0

    def test_half_crash_returns_05(self):
        edge = [{"input": "0\n", "output": None}] * 4
        score = _compute_diversity(crash_count=2, edge_cases=edge)
        assert score == 0.5

    def test_score_never_below_zero(self):
        edge = [{"input": "0\n", "output": None}] * 2
        # Score never goes below 0.0 even if crash_count > len(edge_cases)
        score = _compute_diversity(crash_count=10, edge_cases=edge)
        assert score >= 0.0


class TestFallbackBehavior:
    """Pipeline continues even without lcb_runner installed."""

    SIMPLE_CODE = (
        "n = int(input())\n"
        "nums = list(map(int, input().split()))\n"
        "print(sum(x for x in nums if x % 2 == 0))\n"
    )
    PROBLEM_ROW = {
        "all_tests": {
            "input":  ["6\n1 2 3 4 5 6", "5\n1 3 5 7 9"],
            "output": ["12",              "0"],
        }
    }

    def test_lcb_available_is_bool(self):
        assert isinstance(LCB_AVAILABLE, bool)

    @pytest.mark.skipif(LCB_AVAILABLE, reason="lcb_runner installed — fallback cannot be tested")
    def test_fallback_runs(self):
        from sandbox.lcb_eval import _fallback_evaluate
        result = _fallback_evaluate(
            code=self.SIMPLE_CODE,
            problem_row=self.PROBLEM_ROW,
            timeout=5,
        )
        assert "pass_rate"            in result
        assert "failed_cases"         in result
        assert "test_diversity_score" in result
        assert result["lcb_available"] is False

    @pytest.mark.skipif(not LCB_AVAILABLE, reason="lcb_runner not installed")
    def test_lcb_evaluate_runs(self):
        from sandbox.lcb_eval import lcb_evaluate
        result = lcb_evaluate(
            code=self.SIMPLE_CODE,
            problem_row=self.PROBLEM_ROW,
            edge_cases=[],
            timeout=5,
        )
        assert result["pass_rate"] == 1.0
        assert result["lcb_available"] is True


class TestLocalLcbCompatibleEvaluator:
    """The local evaluator must grade LeetCode-style problems by calling the
    starter method (as lcb_runner does), not by comparing stdout text."""

    STARTER = "class Solution:\n    def total(self, k: int, nums: List[int]) -> bool:\n        "

    def _row(self, inputs, outputs, starter=""):
        return {"all_tests": {"input": inputs, "output": outputs}, "starter_code": starter}

    def test_functional_method_called_with_json_args(self):
        from sandbox.lcb_eval import _fallback_evaluate
        code = (
            "class Solution:\n"
            "    def total(self, k: int, nums: List[int]) -> bool:\n"
            "        return sum(nums) == k\n"
        )
        row = self._row(["3\n[1, 2]", "4\n[1, 2]"], ["true", "false"], self.STARTER)
        res = _fallback_evaluate(code, row, timeout=10)
        assert res["pass_rate"] == 1.0, res["failed_cases"]

    def test_functional_ignores_unguarded_stdin_harness(self):
        # Top-level input() raises EOFError on the empty stdin, but Solution is
        # already defined by then, so the method must still be graded.
        from sandbox.lcb_eval import _fallback_evaluate
        code = (
            "class Solution:\n"
            "    def total(self, k, nums):\n"
            "        return sum(nums) == k\n"
            "k = int(input())\n"
        )
        row = self._row(["3\n[1, 2]"], ["true"], self.STARTER)
        assert _fallback_evaluate(code, row, timeout=10)["pass_rate"] == 1.0

    def test_functional_missing_class_fails_cleanly(self):
        from sandbox.lcb_eval import _fallback_evaluate
        row = self._row(["3\n[1, 2]"], ["true"], self.STARTER)
        res = _fallback_evaluate("print('true')\n", row, timeout=10)
        assert res["pass_rate"] == 0.0
        assert res["failed_cases"][0]["error"]

    def test_stdio_compares_lines_and_decimals(self):
        from sandbox.lcb_eval import _stdio_match
        assert _stdio_match("1\n2.00  \n", "1.0\n2")
        assert not _stdio_match("abc", "abd")
        assert not _stdio_match("1\n2", "1")
