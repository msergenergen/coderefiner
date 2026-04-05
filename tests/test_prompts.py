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
        assert "COMPLETELY NEW" in prompt

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
        idx_feedback    = prompt.index("COMPLETELY NEW")
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
        assert "10" in CODEGEN_CONTEXT   # max 10 iterations

    def test_feedback_rules_rewrite_emphasis(self):
        assert "COMPLETELY NEW" in CODEGEN_FEEDBACK_RULES
        assert "patch" in CODEGEN_FEEDBACK_RULES.lower() \
               or "old one" in CODEGEN_FEEDBACK_RULES
