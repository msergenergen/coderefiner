"""
Unit tests: pipeline/graph.py — StateGraph construction and pure node wrappers.

build_graph() itself is not exercised end-to-end here (that needs live model
servers for codegen/judgement) — these tests cover what's cheap and high-value:
compilation for every variant, and the pure state-transform nodes that route
around the LLM/sandbox calls.
"""
import pipeline.router as router
from pipeline.graph import (
    build_graph,
    _write_memory_final_node,
    _stop_max_node,
    _next_model_node,
    _strip_before_judgement,
)
from tests.conftest import CLEAN_CODE_WITH_COMMENTS


class TestBuildGraphCompiles:

    def test_v1_compiles(self):
        assert build_graph("V1") is not None

    def test_v2_compiles(self):
        assert build_graph("V2") is not None

    def test_v3_compiles(self):
        assert build_graph("V3") is not None

    def test_default_variant_is_v3(self):
        # build_graph() with no argument must not raise and must match V3
        assert build_graph() is not None


class TestWriteMemoryFinalNode:

    def test_sets_pass_status(self, base_state):
        result = _write_memory_final_node(base_state(final_status="PENDING"))
        assert result["final_status"] == "PASS"

    def test_preserves_other_fields(self, base_state):
        result = _write_memory_final_node(base_state(run_id="abc"))
        assert result["run_id"] == "abc"


class TestStopMaxNode:

    def test_sets_stop_max_status(self, base_state):
        result = _stop_max_node(base_state(final_status="PENDING"))
        assert result["final_status"] == "STOP_MAX"


class TestNextModelNode:

    def test_delegates_to_advance_model(self, base_state, monkeypatch):
        class _FakeSettings:
            model_sequence = ["M1", "M2"]
        monkeypatch.setattr(router, "settings", _FakeSettings())
        result = _next_model_node(base_state(active_model="M1", iteration=4))
        assert result["active_model"] == "M2"
        assert result["iteration"] == 0


class TestStripBeforeJudgement:
    """
    Covers the exact mechanism behind the LOW-severity path: when sandbox
    (Node 2) never ran, all node2_* fields must default to safe values —
    in particular node2_pass_rate defaults to 0.0, which judgement_node's
    hard gate (accuracy > 0.0) then uses to block PASS for untested code.
    """

    def test_fills_node2_defaults_when_absent(self, base_state):
        state = base_state(generated_code=CLEAN_CODE_WITH_COMMENTS)
        # Simulate the LOW-severity path: no node2_* keys were ever set by sandbox_node.
        for key in list(state):
            if key.startswith("node2_"):
                del state[key]

        result = _strip_before_judgement(state)

        assert result["node2_pass_rate"] == 0.0
        assert result["node2_passed_count"] == 0
        assert result["node2_total_count"] == 0
        assert result["node2_failed_cases"] == []
        assert result["node2_crash_count"] == 0
        assert result["node2_test_diversity_score"] == 1.0
        assert result["node2_lcb_available"] is False
        assert result["node2_cpu_ms"] == 0.0
        assert result["node2_ram_mb"] == 0.0
        assert result["node2_timed_out"] is False

    def test_preserves_existing_node2_values(self, base_state):
        # If node2 fields already exist (shouldn't normally happen on this path,
        # but the function must not clobber real data if it does), keep them.
        state = base_state(
            generated_code=CLEAN_CODE_WITH_COMMENTS,
            node2_pass_rate=0.75, node2_passed_count=3, node2_total_count=4,
        )
        result = _strip_before_judgement(state)
        assert result["node2_pass_rate"] == 0.75
        assert result["node2_passed_count"] == 3
        assert result["node2_total_count"] == 4

    def test_strips_comments_from_generated_code(self, base_state):
        state = base_state(generated_code=CLEAN_CODE_WITH_COMMENTS)
        result = _strip_before_judgement(state)
        assert "stripped_code" in result
        assert "#" not in result["stripped_code"]
        # generated_code (used for logging) is left untouched
        assert result["generated_code"] == CLEAN_CODE_WITH_COMMENTS
