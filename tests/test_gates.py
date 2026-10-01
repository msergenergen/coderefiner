"""
Unit tests: pipeline/gates.py
severity_gate and iteration_gate return values.

Critical: A wrong gate decision silently routes the pipeline in the wrong direction.
"""
import pytest
from unittest.mock import patch
from pipeline.gates import (
    severity_gate, iteration_gate,
    _compute_ema, _linear_slope, _should_extend,
)


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
    def test_gate_follows_node3_pass_not_score(self, difficulty, threshold):
        # The PASS decision is made once in judgement_node (judge verdict or
        # rubric); the gate must not re-apply the score threshold, or it would
        # override a judge ACCEPT on a below-threshold score.
        state = self._state(score=threshold - 1, passed=True,
                            iteration=3, difficulty=difficulty)
        assert iteration_gate(state) == "write_memory_final"
        state = self._state(score=threshold + 10, passed=False,
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


class TestComputeEma:

    def test_single_value_unchanged(self):
        assert _compute_ema([50.0], alpha=0.5) == [50.0]

    def test_constant_series_unchanged(self):
        result = _compute_ema([60.0, 60.0, 60.0], alpha=0.5)
        assert all(abs(v - 60.0) < 1e-9 for v in result)

    def test_increasing_series_smoothed(self):
        result = _compute_ema([60.0, 65.0, 70.0], alpha=0.5)
        # Each value should be strictly greater than the previous
        assert result[1] > result[0]
        assert result[2] > result[1]

    def test_empty_returns_empty(self):
        assert _compute_ema([], alpha=0.5) == []


class TestLinearSlope:

    def test_perfectly_increasing(self):
        slope = _linear_slope([60.0, 62.0, 64.0])
        assert abs(slope - 2.0) < 1e-9

    def test_perfectly_decreasing(self):
        slope = _linear_slope([64.0, 62.0, 60.0])
        assert abs(slope - (-2.0)) < 1e-9

    def test_flat_series(self):
        slope = _linear_slope([65.0, 65.0, 65.0])
        assert abs(slope) < 1e-9

    def test_single_value_returns_zero(self):
        assert _linear_slope([70.0]) == 0.0

    def test_uses_last_3_points(self):
        # Long history — slope should reflect only last 3 points
        slope_long = _linear_slope([10.0, 10.0, 10.0, 60.0, 62.0, 64.0])
        slope_short = _linear_slope([60.0, 62.0, 64.0])
        assert abs(slope_long - slope_short) < 1e-9


class TestShouldExtend:
    """
    _should_extend is tested with trend_extension_enabled=True patched in.
    Each condition is tested in isolation (all others satisfied).
    """

    def _good_state(self, threshold=80.0, score=74.0, extension_count=0):
        """Returns a state that satisfies all extension conditions."""
        return {
            "node3_total_score":   score,
            "node3_pass":          False,
            "node3_score_breakdown": {"accuracy": 10.0},   # > 0
            "node2_pass_rate":     0.3,
            "score_history":       [60.0, 63.0, 66.0, 69.0, 72.0],
            "pass_rate_history":   [0.1, 0.15, 0.2, 0.25, 0.3],
            "extension_count":     extension_count,
            "difficulty":          "easy",
            "max_iterations":      10,
            "iteration":           10,
        }

    def _settings(self, **overrides):
        defaults = dict(
            trend_extension_enabled=True,
            max_extension_iters=3,
            trend_projection_window=3,
            trend_min_slope=1.0,
            trend_ema_alpha=0.5,
        )
        defaults.update(overrides)
        return type("S", (), defaults)()

    def test_all_conditions_met_returns_true(self):
        with patch("pipeline.gates.settings", self._settings()):
            assert _should_extend(self._good_state(), threshold=80.0) is True

    def test_accuracy_zero_blocks_extension(self):
        state = self._good_state()
        state["node3_score_breakdown"]["accuracy"] = 0.0
        with patch("pipeline.gates.settings", self._settings()):
            assert _should_extend(state, threshold=80.0) is False

    def test_budget_exhausted_blocks_extension(self):
        state = self._good_state(extension_count=3)
        with patch("pipeline.gates.settings", self._settings(max_extension_iters=3)):
            assert _should_extend(state, threshold=80.0) is False

    def test_too_few_history_points_blocks_extension(self):
        state = self._good_state()
        state["score_history"] = [65.0, 68.0]       # < 3 points
        state["pass_rate_history"] = [0.1, 0.2]
        with patch("pipeline.gates.settings", self._settings()):
            assert _should_extend(state, threshold=80.0) is False

    def test_declining_pass_rate_blocks_extension(self):
        state = self._good_state()
        state["pass_rate_history"] = [0.3, 0.2, 0.1, 0.05, 0.0]
        with patch("pipeline.gates.settings", self._settings()):
            assert _should_extend(state, threshold=80.0) is False

    def test_flat_score_slope_blocks_extension(self):
        state = self._good_state()
        state["score_history"] = [65.0, 65.0, 65.0, 65.0, 65.0]
        with patch("pipeline.gates.settings", self._settings(trend_min_slope=1.0)):
            assert _should_extend(state, threshold=80.0) is False

    def test_gap_too_large_for_expected_gain_blocks(self):
        # score=50, threshold=80, gap=30; slope≈3, remaining=3 → expected=9 < 30
        state = self._good_state(score=50.0)
        state["score_history"] = [40.0, 43.0, 46.0, 48.0, 50.0]
        with patch("pipeline.gates.settings", self._settings(max_extension_iters=3)):
            assert _should_extend(state, threshold=80.0) is False

    def test_iteration_gate_extends_when_enabled(self):
        """iteration_gate routes to loop_to_codegen instead of stop_max on trend."""
        state = self._good_state()
        state["iteration"]     = 10
        state["max_iterations"] = 10
        with patch("pipeline.gates.settings", self._settings()):
            assert iteration_gate(state) == "loop_to_codegen"

    def test_iteration_gate_stops_when_disabled(self):
        """When feature is off, stop_max is always returned at max_iterations."""
        state = self._good_state()
        state["iteration"]      = 10
        state["max_iterations"] = 10
        with patch("pipeline.gates.settings", self._settings(trend_extension_enabled=False)):
            assert iteration_gate(state) == "stop_max"
