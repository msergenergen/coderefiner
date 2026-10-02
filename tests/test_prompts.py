"""
Unit tests: models/prompts.py
Four-layer codegen prompt structure.

Critical behaviors:
  - No feedback layer on first iteration
  - Feedback layer present on subsequent iterations
  - improved_prompt content included in prompt
  - Layer separator (---) correct
  - Whitelist restrictions in CONSTRAINTS
  - Pipeline context in CONTEXT
"""
import pytest
from models.prompts import (
    build_system_prompt,
    CODEGEN_IDENTITY,
    CODEGEN_CONTEXT,
    CODEGEN_CONSTRAINTS,
    CODEGEN_FEEDBACK_RULES,
)


class TestBuildSystemPromptFirstIteration:
    """improved_prompt=None — first iteration"""

    def test_identity_included(self):
        prompt = build_system_prompt(None)
        assert "expert Python programmer" in prompt

    def test_context_included(self):
        prompt = build_system_prompt(None)
        assert "Static Analyzer" in prompt
        assert "Sandbox Executor" in prompt
        assert "Judge LLM" in prompt

    def test_constraints_included(self):
        prompt = build_system_prompt(None)
        assert "stdin" in prompt
        assert "stdout" in prompt
        assert "subprocess" in prompt   # in the banned imports list

    def test_no_feedback_layer(self):
        prompt = build_system_prompt(None)
        assert "COMPLETELY NEW" not in prompt

    def test_no_improved_prompt_content(self):
        prompt = build_system_prompt(None)
        assert "Improvement Notes" not in prompt


class TestBuildSystemPromptSubsequentIteration:
    """improved_prompt populated — iteration > 1"""

    SAMPLE_FEEDBACK = (
        "Your code returns None for empty list input. "
        "Add a guard: if not nums: return 0"
    )

    def test_feedback_rules_included(self):
        prompt = build_system_prompt(self.SAMPLE_FEEDBACK)
        assert "COMPLETE corrected solution" in prompt     # default: repair rules

    def test_improved_prompt_included(self):
        prompt = build_system_prompt(self.SAMPLE_FEEDBACK)
        assert self.SAMPLE_FEEDBACK in prompt

    def test_identity_still_present(self):
        prompt = build_system_prompt(self.SAMPLE_FEEDBACK)
        assert "expert Python programmer" in prompt

    def test_context_still_present(self):
        prompt = build_system_prompt(self.SAMPLE_FEEDBACK)
        assert "Static Analyzer" in prompt

    def test_constraints_still_present(self):
        prompt = build_system_prompt(self.SAMPLE_FEEDBACK)
        assert "stdin" in prompt

    def test_layer_order(self):
        prompt = build_system_prompt(self.SAMPLE_FEEDBACK)
        # IDENTITY → CONTEXT → CONSTRAINTS → FEEDBACK → improved_prompt
        idx_identity    = prompt.index("expert Python programmer")
        idx_context     = prompt.index("Static Analyzer")
        idx_constraints = prompt.index("stdin")
        idx_feedback    = prompt.index("COMPLETE corrected solution")
        idx_notes       = prompt.index(self.SAMPLE_FEEDBACK)

        assert idx_identity < idx_context < idx_constraints \
               < idx_feedback < idx_notes

    def test_empty_improved_prompt_same_as_none(self):
        prompt_none  = build_system_prompt(None)
        prompt_empty = build_system_prompt("")
        assert prompt_none == prompt_empty


class TestPromptConstants:
    """Constant content correct?"""

    def test_whitelist_in_constraints(self):
        assert "subprocess" in CODEGEN_CONSTRAINTS
        assert "collections" in CODEGEN_CONSTRAINTS
        assert "itertools"   in CODEGEN_CONSTRAINTS

    def test_pipeline_triple_in_context(self):
        assert "Static Analyzer"  in CODEGEN_CONTEXT
        assert "Sandbox Executor" in CODEGEN_CONTEXT
        assert "Judge LLM"        in CODEGEN_CONTEXT

    def test_max_iter_in_context(self):
        from config import settings
        assert f"at most {settings.max_iterations} attempts" in CODEGEN_CONTEXT

    def test_import_list_matches_whitelist(self):
        from sandbox.whitelist import ALLOWED_IMPORTS
        from models.prompts import CODEGEN_CONSTRAINTS
        for mod in ALLOWED_IMPORTS - {"unittest"}:
            assert mod in CODEGEN_CONSTRAINTS

    def test_feedback_rules_rewrite_emphasis(self):
        assert "COMPLETELY NEW" in CODEGEN_FEEDBACK_RULES
        assert "patch" in CODEGEN_FEEDBACK_RULES.lower() \
               or "old one" in CODEGEN_FEEDBACK_RULES



class TestRepairWithPreviousCode:
    """FEEDBACK_INCLUDE_CODE: previous solution shown on repair (LCB self-repair format)."""

    def test_previous_code_in_user_prompt(self):
        from models.prompts import build_user_prompt
        p = build_user_prompt("PROBLEM", previous_code="print(1)")
        assert "## Your Previous Solution (rejected)" in p and "print(1)" in p
        assert p.index("PROBLEM") < p.index("print(1)")
        assert "print(1)" not in build_user_prompt("PROBLEM")      # first iteration

    def test_flag_off_restores_feedback_only_prompts(self, monkeypatch):
        import dataclasses
        import models.prompts as mp
        monkeypatch.setattr(mp, "settings",
                            dataclasses.replace(mp.settings, feedback_include_code=False))
        assert "print(1)" not in mp.build_user_prompt("PROBLEM", previous_code="print(1)")
        assert "COMPLETELY NEW" in mp.build_system_prompt("fix it")

    def test_codegen_passes_previous_code_only_on_repair(self, monkeypatch):
        import nodes.codegen as cg
        from models.client import ChatResult
        seen = []
        monkeypatch.setattr(cg, "health_check", lambda _m: True)
        monkeypatch.setattr(cg, "chat", lambda m, msgs, **kw: seen.append(msgs) or
                            ChatResult("```python\nprint(2)\n```", "", "", "stop", 1, 1, 0, 0.0, "x"))
        base = {"active_model": "M1", "problem": "PROBLEM", "iteration": 0, "timing": {}, "tokens": {}}
        cg.codegen_node({**base, "generated_code": "", "node3_improved_prompt": None})
        cg.codegen_node({**base, "iteration": 1, "generated_code": "print(1)",
                         "node3_improved_prompt": "wrong for n=1"})
        assert "print(1)" not in seen[0][1]["content"]
        assert "print(1)" in seen[1][1]["content"] and "wrong for n=1" in seen[1][0]["content"]
