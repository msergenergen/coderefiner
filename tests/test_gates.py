"""
Unit tests: pipeline/gates.py
severity_gate and iteration_gate return values.

Critical: A wrong gate decision silently routes the pipeline in the wrong direction.
"""
import pytest
from pipeline.gates import severity_gate, iteration_gate


class TestSeverityGate:

    def test_high_goes_to_stop_high(self):
        assert severity_gate({"node1_severity": "HIGH"}) == "stop_high"

    def test_low_goes_to_node3(self):
        assert severity_gate({"node1_severity": "LOW"}) == "to_node3"

    def test_ok_goes_to_node2(self):
        assert severity_gate({"node1_severity": "OK"}) == "to_node2"


class TestIterationGate:
    """
    iteration_gate runs AFTER log_iteration_node.
    Return values: write_memory_final | stop_max | loop_to_codegen
    """

    def _state(self, score, passed, iteration,
               difficulty="easy", max_iter=10):
        return {
            "node3_total_score": score,
            "node3_pass":        passed,
            "iteration":         iteration,
            "max_iterations":    max_iter,
            "difficulty":        difficulty,
        }

    # ── Basic routing ────────────────────────────────────────────

    def test_pass_goes_to_write_memory_final(self):
        state = self._state(score=85, passed=True, iteration=3)
        assert iteration_gate(state) == "write_memory_final"

    def test_fail_goes_to_loop(self):
        state = self._state(score=60, passed=False, iteration=3)
        assert iteration_gate(state) == "loop_to_codegen"

    def test_max_iter_goes_to_stop_max(self):
        state = self._state(score=60, passed=False, iteration=10)
        assert iteration_gate(state) == "stop_max"

    # ── Difficulty thresholds — boundary values ────────────────────────────

    @pytest.mark.parametrize("difficulty,threshold", [
        ("easy",   80),
        ("medium", 70),
        ("hard",   60),
    ])
    def test_below_threshold_loops(self, difficulty, threshold):
        state = self._state(score=threshold - 1, passed=True,
                            iteration=3, difficulty=difficulty)
        assert iteration_gate(state) == "loop_to_codegen"

    @pytest.mark.parametrize("difficulty,threshold", [
        ("easy",   80),
        ("medium", 70),
        ("hard",   60),
    ])
    def test_at_threshold_passes(self, difficulty, threshold):
        state = self._state(score=threshold, passed=True,
                            iteration=3, difficulty=difficulty)
        assert iteration_gate(state) == "write_memory_final"

    # ── node3_pass False means no PASS ──────────────────────────────

    def test_sufficient_score_but_pass_false(self):
        # When node3_pass=False, PASS is not granted even with sufficient score
        state = self._state(score=90, passed=False, iteration=3)
        assert iteration_gate(state) == "loop_to_codegen"

    # ── max_iterations boundary values ───────────────────────────────

    def test_iter_9_max_10_continues(self):
        state = self._state(score=60, passed=False,
                            iteration=9, max_iter=10)
        assert iteration_gate(state) == "loop_to_codegen"

    def test_iter_10_max_10_stops(self):
        state = self._state(score=60, passed=False,
                            iteration=10, max_iter=10)
        assert iteration_gate(state) == "stop_max"
