"""
Integration tests: Node state transformations.
Each node is called independently; state transformation is verified.
"""
import pytest
from nodes.static_analysis import static_analysis_node
from nodes.cleaner import strip_comments_node
from tests.conftest import CLEAN_CODE, DANGEROUS_CODE, CLEAN_CODE_WITH_COMMENTS


class TestStaticAnalysisNode:

    def test_clean_code_ok_severity(self, base_state):
        result = static_analysis_node(base_state(generated_code=CLEAN_CODE))
        assert result["node1_severity"]    == "OK"
        assert result["node1_bandit_high"] is False

    def test_dangerous_code_high_severity(self, base_state):
        result = static_analysis_node(base_state(generated_code=DANGEROUS_CODE))
        assert result["node1_severity"] == "HIGH"

    def test_state_fields_populated(self, base_state):
        result = static_analysis_node(base_state(generated_code=CLEAN_CODE))
        for field in ("node1_cc", "node1_mi", "node1_halstead",
                      "node1_bandit_high", "node1_severity", "node1_issues"):
            assert field in result, f"{field} missing from state"

    def test_other_state_fields_preserved(self, base_state):
        state  = base_state(generated_code=CLEAN_CODE, iteration=5)
        result = static_analysis_node(state)
        assert result["iteration"]     == 5
        assert result["active_model"]  == "M1"
        assert result["difficulty"]    == "easy"


class TestCleanerNode:

    def test_stripped_code_populated(self, base_state):
        state  = base_state(generated_code=CLEAN_CODE_WITH_COMMENTS)
        result = strip_comments_node(state)
        assert "stripped_code" in result
        assert "#" not in result["stripped_code"]

    def test_generated_code_preserved(self, base_state):
        state  = base_state(generated_code=CLEAN_CODE_WITH_COMMENTS)
        result = strip_comments_node(state)
        assert result["generated_code"] == CLEAN_CODE_WITH_COMMENTS
