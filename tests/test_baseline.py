"""
Unit tests: benchmark/legacy/baseline.py (legacy V0 runner)
_extract_code() behavior — code extraction from LLM output.
"""
import pytest


def _extract_code(text: str) -> str:
    """Import from baseline module."""
    from benchmark.legacy.baseline import _extract_code as _ec
    return _ec(text)


class TestExtractCode:

    def test_python_block_extracted(self):
        text = "Here is the solution:\n```python\ndef f(): return 1\n```"
        assert "def f():" in _extract_code(text)

    def test_last_python_block_taken(self):
        text = "```python\ndef first(): pass\n```\n```python\ndef last(): pass\n```"
        result = _extract_code(text)
        assert "last" in result
        assert "first" not in result

    def test_plain_block_fallback(self):
        text = "```\ndef f(): return 1\n```"
        assert "def f():" in _extract_code(text)

    def test_no_block_returns_stripped(self):
        text = "def f(): return 1"
        assert "def f():" in _extract_code(text)

    def test_empty_string(self):
        assert _extract_code("") == ""

    def test_whitespace_stripped(self):
        text = "```python\n   def f(): pass\n   \n```"
        result = _extract_code(text)
        assert result == result.strip()
