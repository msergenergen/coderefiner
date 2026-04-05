"""
Unit tests: dataset/loader.py + dataset/splitter.py

Does not require lcb_runner connection — only string processing is tested.
"""
import pytest
from dataset.loader import _clean_html
from dataset.splitter import format_problem, dataset_stats


class TestCleanHtml:

    def test_html_tag_removed(self):
        assert _clean_html("<p>hello</p>") == "hello"

    def test_nbsp_becomes_space(self):
        assert " " in _clean_html("a&nbsp;b")

    def test_lt_gt_converted(self):
        result = _clean_html("&lt;br&gt;")
        assert "<" in result or ">" in result

    def test_multiple_blank_lines_reduced(self):
        result = _clean_html("a\n\n\n\nb")
        assert "\n\n\n" not in result

    def test_clean_text_unchanged(self):
        text = "Given a list of integers"
        assert _clean_html(text) == text


class TestFormatProblem:

    def _make_row(self, starter="", inputs=None, outputs=None):
        return {
            "title":        "Test Problem",
            "difficulty":   "easy",
            "content":      "Problem description here.",
            "starter_code": starter,
            "public_tests": {
                "input":  inputs  or ["3\n1 2 3"],
                "output": outputs or ["6"],
            },
        }

    def test_title_included(self):
        result = format_problem(self._make_row())
        assert "Test Problem" in result

    def test_difficulty_included(self):
        result = format_problem(self._make_row())
        assert "easy" in result

    def test_example_tests_shown(self):
        result = format_problem(self._make_row())
        assert "3\n1 2 3" in result
        assert "6" in result

    def test_starter_code_added_when_present(self):
        result = format_problem(self._make_row(starter="def solve(): pass"))
        assert "solve" in result

    def test_starter_code_absent_when_empty(self):
        result = format_problem(self._make_row(starter=""))
        assert "Starter Code" not in result

    def test_stdout_instruction_present(self):
        result = format_problem(self._make_row())
        assert "stdin" in result.lower() or "stdout" in result.lower()


class TestDatasetStats:

    def test_empty_list(self):
        stats = dataset_stats([])
        assert stats["total"] == 0

    def test_correct_counts(self):
        rows = [
            {"difficulty": "easy"},
            {"difficulty": "easy"},
            {"difficulty": "medium"},
            {"difficulty": "hard"},
        ]
        stats = dataset_stats(rows)
        assert stats["total"]  == 4
        assert stats["easy"]   == 2
        assert stats["medium"] == 1
        assert stats["hard"]   == 1
