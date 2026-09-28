"""
Unit tests: pipeline/router.py — sequential model router.
"""
import pipeline.router as router
from pipeline.router import get_next_model, advance_model


class _FakeSettings:
    def __init__(self, sequence):
        self.model_sequence = sequence


class TestGetNextModel:

    def test_returns_next_in_sequence(self, monkeypatch):
        monkeypatch.setattr(router, "settings", _FakeSettings(["M1", "M2", "M3"]))
        assert get_next_model("M1") == "M2"
        assert get_next_model("M2") == "M3"

    def test_last_model_returns_none(self, monkeypatch):
        monkeypatch.setattr(router, "settings", _FakeSettings(["M1", "M2", "M3"]))
        assert get_next_model("M3") is None

    def test_single_model_sequence_returns_none(self, monkeypatch):
        monkeypatch.setattr(router, "settings", _FakeSettings(["M1"]))
        assert get_next_model("M1") is None

    def test_unknown_model_returns_none(self, monkeypatch):
        monkeypatch.setattr(router, "settings", _FakeSettings(["M1", "M2", "M3"]))
        assert get_next_model("QWEN") is None


class TestAdvanceModel:

    def test_advances_to_next_model_and_resets_iteration(self, base_state, monkeypatch):
        monkeypatch.setattr(router, "settings", _FakeSettings(["M1", "M2", "M3"]))
        state = base_state(active_model="M1", iteration=5, node3_improved_prompt="old prompt")
        result = advance_model(state)
        assert result["active_model"] == "M2"
        assert result["iteration"] == 0
        assert result["node3_improved_prompt"] == ""
        assert result["final_status"] == "PENDING"

    def test_last_model_sets_final_pass(self, base_state, monkeypatch):
        monkeypatch.setattr(router, "settings", _FakeSettings(["M1", "M2", "M3"]))
        state = base_state(active_model="M3", iteration=3)
        result = advance_model(state)
        assert result["final_status"] == "PASS"
        # active_model / iteration untouched when no next model exists
        assert result["active_model"] == "M3"

    def test_other_state_fields_preserved(self, base_state, monkeypatch):
        monkeypatch.setattr(router, "settings", _FakeSettings(["M1", "M2"]))
        state = base_state(active_model="M1", run_id="xyz")
        result = advance_model(state)
        assert result["run_id"] == "xyz"
