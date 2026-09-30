"""
Unit tests: metrics/scoring.py — the 100-point rubric's numeric dimensions.

Critical behaviors:
  - Limits are sourced from config.thresholds (cpu_limit_ms=2000, ram_limit_mb=512,
    cc_hard_limit=20) — tests pin these values so a threshold change is caught here too.
  - All four component scores clamp at 0.0, never go negative.
  - numeric_total is exactly the rounded sum of the four components.
"""
import pytest
from metrics.scoring import (
    compute_accuracy_score,
    compute_efficiency_score,
    compute_quality_score,
    compute_security_score,
    compute_numeric_scores,
)


class TestComputeAccuracyScore:

    @pytest.mark.parametrize("pass_rate,expected", [
        (0.0, 0.0),
        (0.5, 12.5),
        (1.0, 25.0),
    ])
    def test_scales_linearly_to_25(self, pass_rate, expected):
        assert compute_accuracy_score(pass_rate) == expected


class TestComputeEfficiencyScore:

    def test_zero_cost_is_max_score(self):
        assert compute_efficiency_score(cpu_ms=0.0, ram_mb=0.0) == 20.0

    def test_cpu_at_limit_zeroes_cpu_component(self):
        # cpu_ms == cpu_limit_ms (2000) -> cpu_score = 0, ram still perfect
        # (0 + 1) / 2 * 20 = 10.0
        assert compute_efficiency_score(cpu_ms=2000.0, ram_mb=0.0) == 10.0

    def test_cpu_over_limit_clamps_to_zero_not_negative(self):
        result = compute_efficiency_score(cpu_ms=4000.0, ram_mb=0.0)
        assert result == 10.0  # cpu_score clamped at 0, ram component still 20*0.5

    def test_ram_at_limit_zeroes_ram_component(self):
        # ram_mb == ram_limit_mb (512) -> ram_score = 0, cpu still perfect
        assert compute_efficiency_score(cpu_ms=0.0, ram_mb=512.0) == 10.0

    def test_ram_over_limit_clamps_to_zero_not_negative(self):
        result = compute_efficiency_score(cpu_ms=0.0, ram_mb=2048.0)
        assert result == 10.0

    def test_both_over_limit_is_zero(self):
        assert compute_efficiency_score(cpu_ms=999999.0, ram_mb=999999.0) == 0.0

    def test_valid_range(self):
        result = compute_efficiency_score(cpu_ms=1000.0, ram_mb=256.0)
        assert 0.0 <= result <= 20.0


class TestComputeQualityScore:

    def test_perfect_mi_and_cc_is_max_score(self):
        assert compute_quality_score(mi=100.0, cc=0.0) == 20.0

    def test_mi_over_100_clamps_via_min(self):
        # mi_score = min(1.0, mi/100) — mi=150 must not exceed the 1.0 cap
        assert compute_quality_score(mi=150.0, cc=0.0) == compute_quality_score(mi=100.0, cc=0.0)

    def test_cc_at_hard_limit_zeroes_cc_component(self):
        # cc == cc_hard_limit (20) -> cc_score = 0, mi still perfect
        # (1.0 + 0) / 2 * 20 = 10.0
        assert compute_quality_score(mi=100.0, cc=20.0) == 10.0

    def test_cc_over_hard_limit_clamps_to_zero_not_negative(self):
        result = compute_quality_score(mi=100.0, cc=40.0)
        assert result == 10.0

    def test_valid_range(self):
        result = compute_quality_score(mi=50.0, cc=10.0)
        assert 0.0 <= result <= 20.0


class TestComputeSecurityScore:

    def test_no_findings_is_max_score(self):
        assert compute_security_score(high_count=0, medium_count=0) == 15.0

    def test_one_high_zeroes_score(self):
        # 15 - (1*15) - 0 = 0
        assert compute_security_score(high_count=1, medium_count=0) == 0.0

    def test_one_medium_deducts_three(self):
        assert compute_security_score(high_count=0, medium_count=1) == 12.0

    def test_high_dominates_regardless_of_medium_count(self):
        # Any HIGH finding already zeroes the score; extra mediums can't go negative
        assert compute_security_score(high_count=1, medium_count=5) == 0.0

    def test_many_mediums_clamp_to_zero_not_negative(self):
        assert compute_security_score(high_count=0, medium_count=10) == 0.0


class TestComputeNumericScores:

    def test_returns_all_expected_keys(self):
        result = compute_numeric_scores(
            pass_rate=1.0, cpu_ms=0.0, ram_mb=0.0,
            mi=100.0, cc=0.0, bandit_high=False,
        )
        assert set(result.keys()) == {
            "accuracy", "efficiency", "quality", "security", "numeric_total",
        }

    def test_numeric_total_is_sum_of_components(self):
        result = compute_numeric_scores(
            pass_rate=0.8, cpu_ms=500.0, ram_mb=100.0,
            mi=70.0, cc=8.0, bandit_high=False, bandit_medium_count=1,
        )
        expected_total = round(
            result["accuracy"] + result["efficiency"]
            + result["quality"] + result["security"],
            2,
        )
        assert result["numeric_total"] == expected_total

    def test_perfect_solution_scores_80(self):
        # Full pass, no cost, perfect quality, no security findings -> 25+20+20+15 = 80
        result = compute_numeric_scores(
            pass_rate=1.0, cpu_ms=0.0, ram_mb=0.0,
            mi=100.0, cc=0.0, bandit_high=False,
        )
        assert result["numeric_total"] == 80.0

    def test_bandit_high_flag_is_translated_to_high_count(self):
        result = compute_numeric_scores(
            pass_rate=1.0, cpu_ms=0.0, ram_mb=0.0,
            mi=100.0, cc=0.0, bandit_high=True,
        )
        assert result["security"] == 0.0

    def test_worst_case_is_zero(self):
        result = compute_numeric_scores(
            pass_rate=0.0, cpu_ms=999999.0, ram_mb=999999.0,
            mi=0.0, cc=999.0, bandit_high=True, bandit_medium_count=10,
        )
        assert result["numeric_total"] == 0.0

    def test_unexecuted_code_gets_no_efficiency_credit(self):
        # Sandbox skipped → cpu_ms/ram_mb are 0 because nothing ran, not
        # because the code is fast; must not earn the full 20 points.
        result = compute_numeric_scores(
            pass_rate=0.0, cpu_ms=0.0, ram_mb=0.0,
            mi=80.0, cc=5.0, bandit_high=False, executed=False,
        )
        assert result["efficiency"] == 0.0
        assert result["quality"] > 0.0

    def test_unparseable_code_gets_no_quality_or_security_credit(self):
        # radon/bandit fall back to MI=100, CC=0, no findings on a
        # SyntaxError — those defaults must not become 20 + 15 points.
        result = compute_numeric_scores(
            pass_rate=0.0, cpu_ms=0.0, ram_mb=0.0,
            mi=100.0, cc=0.0, bandit_high=False,
            executed=False, parsed=False,
        )
        assert result["numeric_total"] == 0.0
