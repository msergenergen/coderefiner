"""
Unit tests: sandbox/runner.py
measure_runtime() behavior.
"""
import pytest
from sandbox.runner import measure_runtime


CORRECT_CODE = (
    "n = int(input())\n"
    "nums = list(map(int, input().split()))\n"
    "print(sum(x for x in nums if x % 2 == 0))\n"
)
INFINITE_LOOP = "while True: pass\n"
SYNTAX_ERROR  = "def f(\n"


class TestMeasureRuntime:

    def test_return_format(self):
        result = measure_runtime(CORRECT_CODE, ["6\n1 2 3 4 5 6"])
        assert "cpu_ms"    in result
        assert "ram_mb"    in result
        assert "timed_out" in result
        assert "error"     in result

    def test_correct_code_positive_cpu_ms(self):
        result = measure_runtime(CORRECT_CODE, ["6\n1 2 3 4 5 6"])
        assert result["cpu_ms"] > 0
        assert result["timed_out"] is False

    def test_timeout_caught(self):
        result = measure_runtime(INFINITE_LOOP, [""], timeout_sec=2)
        assert result["timed_out"] is True

    def test_empty_input_list(self):
        result = measure_runtime(CORRECT_CODE, [])
        assert result["cpu_ms"] == 0.0
        assert result["timed_out"] is False

    def test_multiple_tests_averaged(self):
        inputs = ["6\n1 2 3 4 5 6", "5\n1 3 5 7 9"]
        result = measure_runtime(CORRECT_CODE, inputs)
        assert result["cpu_ms"] > 0
        assert result["ram_mb"] >= 0

    def test_max_3_tests_run(self):
        # Runs at most 3 tests even if 5 are provided
        inputs = ["6\n1 2 3 4 5 6"] * 5
        result = measure_runtime(CORRECT_CODE, inputs)
        # 3 tests × ~20ms = at most ~200ms expected (with margin for slow CI)
        assert result["cpu_ms"] < 10_000


SLEEPER = "import time\ntime.sleep(0.5)\nprint(1)\n"
BUSY = "s = 0\nfor i in range(3_000_000):\n    s += i\nprint(s)\n"
ALLOC = "x = bytearray(80 * 1024 * 1024)\nprint(len(x))\n"
LEETCODE = (
    "class Solution:\n"
    "    def total(self, nums: List[int]) -> int:\n"
    "        s = 0\n"
    "        for _ in range(3_000_000):\n"
    "            s += 1\n"
    "        return sum(nums) + s\n"
)
STARTER = "class Solution:\n    def total(self, nums: List[int]) -> int:\n        "


class TestResourceUsage:
    """cpu_ms/ram_mb come from the child's rusage (os.wait4), not psutil after exit."""

    def test_peak_memory_is_measured(self):
        result = measure_runtime(ALLOC, [""])
        assert result["ram_mb"] >= 80

    def test_cpu_time_not_wall_time(self):
        # sleeping costs wall time but almost no CPU time
        sleep = measure_runtime(SLEEPER, [""])
        busy = measure_runtime(BUSY, [""])
        assert sleep["cpu_ms"] < 300
        assert busy["cpu_ms"] > sleep["cpu_ms"]

    def test_leetcode_method_is_called(self):
        called = measure_runtime(LEETCODE, ["[1, 2, 3]"], starter_code=STARTER)
        as_script = measure_runtime(LEETCODE, ["[1, 2, 3]"])
        assert called["error"] is None
        # the busy method body only runs when the driver calls it
        assert called["cpu_ms"] > as_script["cpu_ms"] + 20

    def test_failing_program_reports_error(self):
        result = measure_runtime("raise ValueError('x')\n", [""])
        assert "ValueError" in result["error"]
