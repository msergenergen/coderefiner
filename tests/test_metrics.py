"""
Unit tests: metrics/complexity.py + metrics/security.py

Critical behaviors:
  - Threshold boundary values work correctly
  - _decide_severity priority order is correct
  - HIGH > LOW > OK
"""
import pytest
from metrics.complexity import (
    cyclomatic_complexity, maintainability_index,
    halstead_volume, cognitive_complexity,
)
from metrics.security import bandit_scan, check_imports
from nodes.static_analysis import _decide_severity
from tests.conftest import CLEAN_CODE, DANGEROUS_CODE, COMPLEX_CODE


class TestCyclomaticComplexity:

    def test_clean_code_low_cc(self):
        assert cyclomatic_complexity(CLEAN_CODE) <= 5

    def test_complex_code_high_cc(self):
        assert cyclomatic_complexity(COMPLEX_CODE) >= 10

    def test_single_line_function(self):
        assert cyclomatic_complexity("def f(): return 1") <= 2

    def test_empty_string(self):
        # Must not raise an exception
        result = cyclomatic_complexity("")
        assert isinstance(result, float)


class TestMaintainabilityIndex:

    def test_clean_code_high_mi(self):
        assert maintainability_index(CLEAN_CODE) >= 65

    def test_valid_range(self):
        mi = maintainability_index(CLEAN_CODE)
        assert 0.0 <= mi <= 100.0

    def test_syntax_error_no_exception(self):
        result = maintainability_index("def f(")
        assert isinstance(result, float)


class TestHalsteadVolume:

    def test_clean_code_reasonable_halstead(self):
        h = halstead_volume(CLEAN_CODE)
        assert h < 2000

    def test_empty_code(self):
        result = halstead_volume("")
        assert isinstance(result, float)


class TestBanditScan:

    def test_clean_code_no_high(self):
        result = bandit_scan(CLEAN_CODE)
        assert result["high_count"] == 0

    def test_dangerous_code_has_issues(self):
        # DANGEROUS_CODE uses exec and os.system — bandit flags at least MEDIUM
        result = bandit_scan(DANGEROUS_CODE)
        assert result["medium_count"] > 0 or result["high_count"] > 0

    def test_return_format(self):
        result = bandit_scan(CLEAN_CODE)
        assert "high_count"   in result
        assert "medium_count" in result
        assert "issues"       in result


class TestCheckImports:

    def test_clean_import_valid(self):
        code = "import math\nimport re"
        assert check_imports(code) == []

    def test_forbidden_import_detected(self):
        code = "import subprocess\nimport socket"
        forbidden = check_imports(code)
        assert len(forbidden) > 0
        assert any("subprocess" in f or "socket" in f for f in forbidden)

    def test_from_import_checked(self):
        code = "from os import path"
        result = check_imports(code)
        assert isinstance(result, list)

    def test_syntax_error_code(self):
        result = check_imports("def f(")
        assert isinstance(result, list)


class TestDecideSeverity:
    """
    _decide_severity priority order: HIGH > LOW > OK
    """

    def test_bandit_high_always_high(self):
        # Even if other metrics are good, bandit HIGH → severity HIGH
        sev = _decide_severity(cc=2.0, mi=90.0, halstead=100.0,
                               bandit_high_count=1, forbidden_imports=[])
        assert sev == "HIGH"

    def test_forbidden_import_high(self):
        sev = _decide_severity(cc=2.0, mi=90.0, halstead=100.0,
                               bandit_high_count=0, forbidden_imports=["subprocess"])
        assert sev == "HIGH"

    def test_cc_limit_exceeded_high(self):
        # CC > 20 → HIGH
        sev = _decide_severity(cc=21.0, mi=90.0, halstead=100.0,
                               bandit_high_count=0, forbidden_imports=[])
        assert sev == "HIGH"

    def test_cc_boundary_equal_not_high(self):
        # CC == 20 → not HIGH (condition is > 20)
        sev = _decide_severity(cc=20.0, mi=90.0, halstead=100.0,
                               bandit_high_count=0, forbidden_imports=[])
        assert sev != "HIGH"

    def test_mi_low_is_low(self):
        # MI < 40 → LOW
        sev = _decide_severity(cc=2.0, mi=35.0, halstead=100.0,
                               bandit_high_count=0, forbidden_imports=[])
        assert sev == "LOW"

    def test_halstead_high_is_low(self):
        # Halstead > 2000 → LOW
        sev = _decide_severity(cc=2.0, mi=90.0, halstead=2001.0,
                               bandit_high_count=0, forbidden_imports=[])
        assert sev == "LOW"

    def test_clean_code_ok(self):
        sev = _decide_severity(cc=2.0, mi=90.0, halstead=100.0,
                               bandit_high_count=0, forbidden_imports=[])
        assert sev == "OK"

    @pytest.mark.parametrize("cc,mi,halstead,expected", [
        (2.0,  90.0, 100.0,  "OK"),    # all good
        (2.0,  35.0, 100.0,  "LOW"),   # MI low
        (2.0,  90.0, 2001.0, "LOW"),   # Halstead high
        (21.0, 90.0, 100.0,  "HIGH"),  # CC exceeds limit
    ])
    def test_parametric_severity(self, cc, mi, halstead, expected):
        sev = _decide_severity(cc=cc, mi=mi, halstead=halstead,
                               bandit_high_count=0, forbidden_imports=[])
        assert sev == expected
