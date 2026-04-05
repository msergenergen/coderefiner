"""
Unit tests: nodes/cleaner.py
Module: strip_comments(), strip_comments_node()

Critical behaviors:
  - Line-start comment → removed
  - Inline comment → removed
  - Docstring (triple-quote) → PRESERVED
  - Syntax-error code → original returned, no exception
  - generated_code → unchanged
  - stripped_code → cleaned version
"""
import pytest
from nodes.cleaner import strip_comments, strip_comments_node
from tests.conftest import (
    CLEAN_CODE, CLEAN_CODE_WITH_COMMENTS, COMMENTED_WITH_DOCSTRING
)


class TestStripComments:

    def test_line_start_comment_removed(self):
        code = "# header\ndef f():\n    return 1"
        result = strip_comments(code)
        assert "#" not in result
        assert "def f():" in result
        assert "return 1" in result

    def test_inline_comment_removed(self):
        code = "x = a + b  # add"
        result = strip_comments(code)
        assert "# add" not in result
        assert "x = a + b" in result

    def test_docstring_preserved(self):
        result = strip_comments(COMMENTED_WITH_DOCSTRING)
        assert '"""İki sayının indekslerini döner."""' in result
        assert "# hash map approach" not in result
        assert "# placeholder" not in result

    def test_clean_code_unchanged(self):
        result = strip_comments(CLEAN_CODE)
        assert "def two_sum" in result
        assert "seen" in result

    def test_commented_code_cleaned(self):
        result = strip_comments(CLEAN_CODE_WITH_COMMENTS)
        assert "#" not in result
        assert "def two_sum" in result
        assert "seen" in result

    def test_syntax_error_returns_original(self):
        broken = "def f(\n    # missing closing paren"
        result = strip_comments(broken)
        assert isinstance(result, str)

    def test_empty_code(self):
        assert strip_comments("") == ""

    def test_comment_only_code(self):
        code = "# line 1\n# line 2\n# line 3"
        result = strip_comments(code)
        assert "#" not in result
        assert isinstance(result, str)

    def test_multi_line_comment_block(self):
        code = "# block start\n# continued\n# end\ndef f():\n    pass"
        result = strip_comments(code)
        assert "#" not in result
        assert "def f():" in result


class TestStripCommentsNode:

    def test_stripped_code_populated(self, base_state):
        state = base_state(generated_code=CLEAN_CODE_WITH_COMMENTS)
        result = strip_comments_node(state)
        assert "#" not in result["stripped_code"]

    def test_generated_code_unchanged(self, base_state):
        state = base_state(generated_code=CLEAN_CODE_WITH_COMMENTS)
        result = strip_comments_node(state)
        assert result["generated_code"] == CLEAN_CODE_WITH_COMMENTS

    def test_other_state_fields_preserved(self, base_state):
        state = base_state(generated_code=CLEAN_CODE_WITH_COMMENTS,
                           iteration=3, difficulty="hard")
        result = strip_comments_node(state)
        assert result["iteration"]  == 3
        assert result["difficulty"] == "hard"
