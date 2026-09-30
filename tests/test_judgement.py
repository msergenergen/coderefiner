"""
Unit tests: nodes/judgement.py — Node 3 (Judgement LLM).

Critical behaviors covered:
  - Hard gate: accuracy == 0 (zero test pass rate) can never PASS, regardless
    of total score (nodes/judgement.py:137).
  - improved_prompt is truncated at 2000 chars.
  - _parse_judgement handles fenced JSON, bare JSON, and malformed output.
  - _get_llm_judgement falls back to a rule-based prompt on any LLM error.
  - _hint_for_issue matches known static-analysis issue patterns.
"""
import pytest
import nodes.judgement as judgement
from nodes.judgement import (
    _hint_for_issue,
    _parse_judgement,
    _fallback_prompt,
    _build_judgement_prompt,
    _get_llm_judgement,
    judgement_node,
)


class TestHintForIssue:

    def test_mypy_undefined_name_hint(self):
        hint = _hint_for_issue('Name "List" is not defined')
        assert hint is not None
        assert "from typing import List" in hint

    def test_ruff_f821_hint(self):
        hint = _hint_for_issue("[F821] line 3: Undefined name `Dict`")
        assert hint is not None
        assert "Dict" in hint

    def test_line_too_long_hint_includes_length(self):
        hint = _hint_for_issue("[E501] line 10: line too long (95 > 88 characters)")
        assert hint is not None
        assert "95" in hint

    def test_unknown_issue_returns_none(self):
        assert _hint_for_issue("some completely unrelated message") is None


class TestParseJudgement:

    def test_fenced_json_block(self):
        raw = (
            "Here is my evaluation:\n```json\n"
            '{"algorithm_choice": 8, "error_handling": 6, '
            '"failure_reasons": ["slow loop"], "improved_system_prompt": "fix it"}'
            "\n```"
        )
        result = _parse_judgement(raw)
        assert result["algorithm_choice"] == 8
        assert result["error_handling"] == 6
        assert result["failure_reasons"] == ["slow loop"]
        assert result["improved_system_prompt"] == "fix it"

    def test_bare_json_no_fences(self):
        raw = '{"algorithm_choice": 3, "error_handling": 2, "failure_reasons": [], "improved_system_prompt": "x"}'
        result = _parse_judgement(raw)
        assert result["algorithm_choice"] == 3
        assert result["error_handling"] == 2

    def test_malformed_json_returns_safe_defaults(self):
        result = _parse_judgement("not json at all {broken")
        assert result["algorithm_choice"] == 5
        assert result["error_handling"] == 5
        assert result["failure_reasons"] == ["json_parse_error"]
        assert result["improved_system_prompt"] == ""

    def test_out_of_range_scores_are_clamped(self):
        raw = '{"algorithm_choice": 99, "error_handling": -5, "failure_reasons": [], "improved_system_prompt": ""}'
        result = _parse_judgement(raw)
        assert result["algorithm_choice"] == 10
        assert result["error_handling"] == 0

    def test_missing_keys_default_to_five(self):
        result = _parse_judgement("{}")
        assert result["algorithm_choice"] == 5
        assert result["error_handling"] == 5


class TestFallbackPrompt:

    def test_includes_node1_issues(self):
        prompt = _fallback_prompt(node1_issues=["CC too high"], failed_cases=[], timed_out=False)
        assert "CC too high" in prompt

    def test_includes_failed_cases(self):
        cases = [{"input": "3", "expected": "6", "got": "5"}]
        prompt = _fallback_prompt(node1_issues=[], failed_cases=cases, timed_out=False)
        assert "expected" in prompt.lower()
        assert "'6'" in prompt

    def test_includes_timeout_note(self):
        prompt = _fallback_prompt(node1_issues=[], failed_cases=[], timed_out=True)
        assert "time limit" in prompt.lower()

    def test_clean_run_has_no_extra_sections(self):
        prompt = _fallback_prompt(node1_issues=[], failed_cases=[], timed_out=False)
        assert "Fix the following" not in prompt
        assert "failed these test cases" not in prompt


class TestBuildJudgementPrompt:

    def _numeric(self, **overrides):
        base = {"accuracy": 20.0, "efficiency": 15.0, "quality": 18.0,
                "security": 15.0, "numeric_total": 68.0}
        base.update(overrides)
        return base

    def test_includes_problem_and_code(self):
        prompt = _build_judgement_prompt(
            numeric_scores=self._numeric(), code="def f(): pass", problem="Solve X",
            node1_issues=[], failed_cases=[], halstead=10.0, timed_out=False,
            difficulty="easy", iteration=1,
        )
        assert "Solve X" in prompt
        assert "def f(): pass" in prompt

    def test_truncates_long_problem_and_code(self):
        prompt = _build_judgement_prompt(
            numeric_scores=self._numeric(), code="x = 1\n" * 5000, problem="P" * 5000,
            node1_issues=[], failed_cases=[], halstead=10.0, timed_out=False,
            difficulty="easy", iteration=1,
        )
        # problem truncated to 800 chars, code to 8000 chars (see source slices)
        assert "P" * 801 not in prompt
        assert len([c for c in prompt if c == "x"]) < 8100

    def test_failed_cases_rendered_with_fix_hints(self):
        prompt = _build_judgement_prompt(
            numeric_scores=self._numeric(), code="code", problem="prob",
            node1_issues=['Name "List" is not defined'],
            failed_cases=[{"input": "1", "expected": "2", "got": "3", "error": None}],
            halstead=10.0, timed_out=False, difficulty="medium", iteration=2,
        )
        assert "Case 1" in prompt
        assert "from typing import List" in prompt

    def test_no_issues_no_failed_cases(self):
        prompt = _build_judgement_prompt(
            numeric_scores=self._numeric(), code="code", problem="prob",
            node1_issues=[], failed_cases=[], halstead=10.0, timed_out=False,
            difficulty="hard", iteration=1,
        )
        assert "No issues detected." in prompt
        assert "None — all visible tests passed." in prompt


class TestGetLlmJudgementFallback:

    def test_client_error_triggers_fallback(self, monkeypatch):
        def _boom(model_id):
            raise ConnectionError("server unreachable")
        monkeypatch.setattr(judgement, "get_client", _boom)

        result = _get_llm_judgement(
            numeric_scores={"accuracy": 0.0, "efficiency": 0.0, "quality": 0.0,
                             "security": 0.0, "numeric_total": 0.0},
            code="code", problem="prob", node1_issues=["issue"], failed_cases=[],
            halstead=0.0, timed_out=False, difficulty="easy", iteration=1,
        )
        assert result["failure_reasons"] == ["judgement_llm_error"]
        assert result["algorithm_choice"] == 5
        assert result["error_handling"] == 5
        assert "issue" in result["improved_system_prompt"]


class TestJudgementNode:

    def _patch_llm(self, monkeypatch, **overrides):
        canned = {
            "algorithm_choice": 8,
            "error_handling": 8,
            "failure_reasons": [],
            "improved_system_prompt": "keep going",
            "_judgement_prompt_tokens": 100,
            "_judgement_completion_tokens": 50,
        }
        canned.update(overrides)
        monkeypatch.setattr(judgement, "_get_llm_judgement", lambda **kw: dict(canned))
        return canned

    def test_full_pass_scores_above_threshold(self, base_state, monkeypatch):
        self._patch_llm(monkeypatch)
        state = base_state(
            difficulty="easy",
            node1_mi=100.0, node1_cc=0.0, node1_bandit_high=False,
            node2_pass_rate=1.0, node2_cpu_ms=0.0, node2_ram_mb=0.0,
            node2_total_count=5,
        )
        result = judgement_node(state)
        # numeric_total(80) + algorithm_choice(8) + error_handling(8) = 96 >= easy(80)
        assert result["node3_total_score"] == 96.0
        assert result["node3_pass"] is True

    def test_zero_accuracy_hard_gates_fail_even_if_total_high(self, base_state, monkeypatch):
        # Sandbox ran but every test failed -> accuracy 0.0, while efficiency,
        # quality, security and the judge are all maxed. hard's threshold is
        # 60 — total(75) clears it, but accuracy==0 must still block PASS.
        self._patch_llm(monkeypatch, algorithm_choice=10, error_handling=10)
        state = base_state(
            difficulty="hard",
            node1_mi=100.0, node1_cc=0.0, node1_bandit_high=False,
            node2_pass_rate=0.0, node2_cpu_ms=0.0, node2_ram_mb=0.0,
            node2_total_count=5,
        )
        result = judgement_node(state)
        assert result["node3_total_score"] >= 60   # score alone would clear hard's threshold
        assert result["node3_pass"] is False        # hard gate still blocks it

    def test_below_threshold_fails(self, base_state, monkeypatch):
        self._patch_llm(monkeypatch, algorithm_choice=0, error_handling=0)
        state = base_state(
            difficulty="hard",
            node1_mi=50.0, node1_cc=15.0, node1_bandit_high=False,
            node2_pass_rate=0.2, node2_cpu_ms=1500.0, node2_ram_mb=400.0,
        )
        result = judgement_node(state)
        assert result["node3_pass"] is False

    def test_improved_prompt_truncated_at_2000_chars(self, base_state, monkeypatch):
        self._patch_llm(monkeypatch, improved_system_prompt="x" * 5000)
        result = judgement_node(base_state())
        assert len(result["node3_improved_prompt"]) == 2000

    def test_score_breakdown_has_six_dimensions(self, base_state, monkeypatch):
        self._patch_llm(monkeypatch)
        result = judgement_node(base_state())
        assert set(result["node3_score_breakdown"].keys()) == {
            "accuracy", "efficiency", "quality", "security",
            "algorithm_choice", "error_handling",
        }

    def test_token_bookkeeping_accumulates_codegen_and_judgement(self, base_state, monkeypatch):
        self._patch_llm(monkeypatch)
        state = base_state(tokens={
            "codegen_prompt": 200, "codegen_completion": 100,
            "judgement_prompt": 0, "judgement_completion": 0, "total": 300,
        })
        result = judgement_node(state)
        assert result["tokens"]["judgement_prompt"] == 100
        assert result["tokens"]["judgement_completion"] == 50
        assert result["tokens"]["total"] == 200 + 100 + 100 + 50

    def test_other_state_fields_preserved(self, base_state, monkeypatch):
        self._patch_llm(monkeypatch)
        state = base_state(run_id="abc123", active_model="M2")
        result = judgement_node(state)
        assert result["run_id"] == "abc123"
        assert result["active_model"] == "M2"


class TestClipAndParse:

    def test_clip_leaves_short_values_untouched(self):
        from nodes.judgement import _clip
        assert _clip("[1, 2, 3]") == "[1, 2, 3]"

    def test_clip_truncates_long_stress_inputs(self):
        from nodes.judgement import _clip
        out = _clip("9" * 7508)
        assert out.startswith("9" * 200)
        assert "7308 more chars truncated" in out
        assert len(out) < 300

    def test_parses_detects_syntax_errors(self):
        from nodes.judgement import _parses
        assert _parses("def f():\n    return 1\n")
        assert not _parses("def f(:\n")


class TestParseTripleQuotedPrompt:

    def test_python_triple_quoted_field_is_repaired(self):
        from nodes.judgement import _parse_judgement
        raw = (
            '{\n  "algorithm_choice": 4,\n  "error_handling": 5,\n'
            '  "failure_reasons": ["dp off by one"],\n'
            '  "improved_system_prompt": """Fix the DP.\n'
            'Case "leetscode" -> expected 1.\nUse {dict} carefully."""\n}'
        )
        out = _parse_judgement(raw)
        assert out["failure_reasons"] == ["dp off by one"]
        assert out["algorithm_choice"] == 4
        assert 'Case "leetscode"' in out["improved_system_prompt"]

    def test_raw_newline_inside_string_is_tolerated(self):
        from nodes.judgement import _parse_judgement
        raw = '{"algorithm_choice": 7, "error_handling": 6, "failure_reasons": [], "improved_system_prompt": "line one\nline two"}'
        assert _parse_judgement(raw)["algorithm_choice"] == 7
