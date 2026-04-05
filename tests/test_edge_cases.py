"""
Unit tests: dataset/edge_cases.py
needs_edge_cases(), generate_edge_cases(), augment_row()
"""
import pytest
from dataset.edge_cases import (
    needs_edge_cases, generate_edge_cases, augment_row,
    EDGE_CASE_THRESHOLD,
)


def make_row(n_tests: int, content: str = "Given a list of integers") -> dict:
    """Creates a minimal problem_row for testing."""
    inputs  = [f"test_{i}" for i in range(n_tests)]
    outputs = ["0"] * n_tests
    return {
        "id": "q_test", "title": "Test", "difficulty": "easy",
        "content": content, "starter_code": "",
        "public_tests":  {"input": inputs[:2], "output": outputs[:2]},
        "private_tests": {"input": inputs[2:], "output": outputs[2:]},
        "all_tests":     {"input": inputs,      "output": outputs},
    }


class TestNeedsEdgeCases:

    def test_few_tests_requires_edge_cases(self):
        row = make_row(n_tests=EDGE_CASE_THRESHOLD - 1)
        assert needs_edge_cases(row) is True

    def test_sufficient_tests_not_needed(self):
        row = make_row(n_tests=EDGE_CASE_THRESHOLD)
        assert needs_edge_cases(row) is False

    def test_many_tests_not_needed(self):
        row = make_row(n_tests=EDGE_CASE_THRESHOLD + 5)
        assert needs_edge_cases(row) is False


class TestGenerateEdgeCases:

    def test_at_least_one_case_generated(self):
        row = make_row(n_tests=3)
        cases = generate_edge_cases(row)
        assert len(cases) >= 1

    def test_each_case_has_input(self):
        row = make_row(n_tests=3)
        for case in generate_edge_cases(row):
            assert "input"  in case
            assert "output" in case

    def test_edge_case_output_is_none(self):
        # Edge case output is unknown — must be None
        row = make_row(n_tests=3)
        for case in generate_edge_cases(row):
            assert case["output"] is None

    def test_list_content_generates_cases(self):
        row = make_row(n_tests=3, content="Given a list of integers nums")
        cases = generate_edge_cases(row)
        assert len(cases) >= 1

    def test_input_is_string(self):
        row = make_row(n_tests=3)
        for case in generate_edge_cases(row):
            assert isinstance(case["input"], str)


class TestAugmentRow:

    def test_few_tests_adds_edge_cases(self):
        row = make_row(n_tests=3)
        aug = augment_row(row)
        assert "edge_cases" in aug
        assert len(aug["edge_cases"]) >= 1

    def test_sufficient_tests_empty_list(self):
        row = make_row(n_tests=EDGE_CASE_THRESHOLD + 2)
        aug = augment_row(row)
        assert aug["edge_cases"] == []

    def test_original_row_unchanged(self):
        row = make_row(n_tests=3)
        aug = augment_row(row)
        # all_tests must remain unchanged
        assert aug["all_tests"] == row["all_tests"]
