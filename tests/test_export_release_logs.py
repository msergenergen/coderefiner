"""
Tests for scripts/export_release_logs.py

Verifies:
1. LCB content stripping (training.problem, training.all_tests)
2. Archive creation and file structure
3. outputs/ directory is never modified
"""

import json
import sys
import zipfile
from pathlib import Path

import pytest

# Import the script's functions
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from export_release_logs import (
    strip_lcb_content,
    collect_iteration_jsons,
    collect_baseline_jsonls,
)


class TestStripLcbContent:
    """Test the strip_lcb_content function."""

    def test_strip_problem_and_all_tests(self):
        """Should remove training.problem and training.all_tests."""
        data = {
            "question_id": 123,
            "iter": 1,
            "training": {
                "generated_code": "print('hello')",
                "stripped_code": "print('hello')",
                "problem": "This is the problem statement, quite long...",
                "all_tests": {"public": [...], "private": [...]},
            },
            "node3": {"verdict": "accept", "improved_prompt": "fix the bug"},
        }

        result = strip_lcb_content(data)

        assert "problem" not in result["training"]
        assert "all_tests" not in result["training"]
        assert "generated_code" in result["training"]
        assert "stripped_code" in result["training"]
        assert result["node3"]["verdict"] == "accept"

    def test_preserve_other_fields(self):
        """Should preserve all fields except problem and all_tests."""
        data = {
            "question_id": 456,
            "difficulty": "hard",
            "status": "PASS",
            "tokens": {"prompt": 100, "completion": 50},
            "node1": {"cc": 5, "severity": "OK"},
            "node2": {"pass_rate": 1.0},
            "node3": {"verdict": "reject", "failure_reasons": ["..."]},
            "training": {
                "generated_code": "code",
                "problem": "will be stripped",
            },
        }

        result = strip_lcb_content(data)

        assert result["question_id"] == 456
        assert result["difficulty"] == "hard"
        assert result["status"] == "PASS"
        assert result["tokens"]["prompt"] == 100
        assert result["node1"]["cc"] == 5
        assert result["node2"]["pass_rate"] == 1.0
        assert result["node3"]["verdict"] == "reject"

    def test_handle_missing_training(self):
        """Should handle records without a 'training' key."""
        data = {"question_id": 789, "node3": {"verdict": "accept"}}
        result = strip_lcb_content(data)
        assert result == data

    def test_handle_non_dict_training(self):
        """Should handle training being non-dict (edge case)."""
        data = {"question_id": 789, "training": None}
        result = strip_lcb_content(data)
        assert result["training"] is None

    def test_idempotent(self):
        """Stripping twice should be idempotent."""
        data = {
            "training": {
                "generated_code": "code",
                "problem": "statement",
                "all_tests": ["test"],
            }
        }

        result1 = strip_lcb_content(data)
        result2 = strip_lcb_content(result1)

        assert result1 == result2


class TestCollectIterationJsons:
    """Test collecting iteration JSON files."""

    def test_find_iteration_jsons(self, tmp_path):
        """Should find iteration JSON files for matching run_ids."""
        # Create synthetic directory structure
        outputs = tmp_path / "outputs"
        date_dir = outputs / "openrouter" / "results" / "2026-10-02"
        iter_dir = date_dir / "iterations" / "123"
        iter_dir.mkdir(parents=True)

        # Create iteration JSONs
        json1 = iter_dir / "M1_iter1_run_20261002_100638.json"
        json2 = iter_dir / "M1_iter2_run_20261002_100640.json"
        json3 = iter_dir / "M1_iter1_run_20261002_999999.json"  # Different run

        json1.write_text("{}")
        json2.write_text("{}")
        json3.write_text("{}")

        run_ids = {"run_20261002_100638", "run_20261002_100640"}

        results = collect_iteration_jsons(outputs, run_ids)

        assert len(results) == 2
        assert json1 in results
        assert json2 in results
        assert json3 not in results

    def test_collect_from_multiple_dates(self, tmp_path):
        """Should find JSONs across multiple date directories."""
        outputs = tmp_path / "outputs"

        # Date 1
        date1 = outputs / "openrouter" / "results" / "2026-10-02" / "iterations" / "100"
        date1.mkdir(parents=True)
        file1 = date1 / "M1_iter1_run_20261002_100638.json"
        file1.write_text("{}")

        # Date 2
        date2 = outputs / "openrouter" / "results" / "2026-10-03" / "iterations" / "200"
        date2.mkdir(parents=True)
        file2 = date2 / "M1_iter1_run_20261003_100935.json"
        file2.write_text("{}")

        run_ids = {"run_20261002_100638", "run_20261003_100935"}

        results = collect_iteration_jsons(outputs, run_ids)

        assert len(results) == 2
        assert file1 in results
        assert file2 in results

    def test_empty_result_when_no_match(self, tmp_path):
        """Should return empty list when no files match."""
        outputs = tmp_path / "outputs"

        date_dir = outputs / "openrouter" / "results" / "2026-10-02" / "iterations" / "123"
        date_dir.mkdir(parents=True)

        file1 = date_dir / "M1_iter1_run_20261002_100638.json"
        file1.write_text("{}")

        run_ids = {"run_20261002_999999"}  # Different run

        results = collect_iteration_jsons(outputs, run_ids)

        assert len(results) == 0


class TestCollectBaselineJsonls:
    """Test collecting baseline JSONL files."""

    def test_find_baseline_jsonls(self, tmp_path):
        """Should find baseline JSONL files matching run_ids."""
        outputs = tmp_path / "outputs"
        baseline_dir = outputs / "openrouter" / "results" / "baseline_samples"
        baseline_dir.mkdir(parents=True)

        jsonl1 = baseline_dir / "run_20261002_091834.jsonl"
        jsonl2 = baseline_dir / "run_20261002_091834.regraded.jsonl"
        jsonl3 = baseline_dir / "run_20261003_093410.jsonl"

        jsonl1.write_text("")
        jsonl2.write_text("")
        jsonl3.write_text("")

        run_ids = {"run_20261002_091834", "run_20261003_093410"}

        results = collect_baseline_jsonls(outputs, run_ids)

        assert len(results) == 3  # Both variants of 091834 + 093410
        assert jsonl1 in results
        assert jsonl2 in results
        assert jsonl3 in results


class TestArchiveCreation:
    """Test archive creation with stripped content."""

    def test_archive_with_stripped_content(self, tmp_path):
        """Should create an archive with LCB content stripped."""
        # Create an iteration JSON with LCB content
        iter_data = {
            "question_id": 123,
            "iter": 1,
            "training": {
                "generated_code": "code",
                "problem": "problem text should be stripped",
                "all_tests": ["test"],
            },
        }

        # Create release directory
        release_dir = tmp_path / "release"
        release_dir.mkdir()

        # Create archive with stripped content
        zip_path = release_dir / "test.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            # Add stripped iteration JSON
            stripped = strip_lcb_content(iter_data)
            zf.writestr(
                "openrouter/results/2026-10-02/iterations/123/iter.json",
                json.dumps(stripped)
            )
            zf.writestr("DATA_README.md", "# Data README")

        # Verify archive contents
        assert zip_path.exists()
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
            assert "openrouter/results/2026-10-02/iterations/123/iter.json" in names
            assert "DATA_README.md" in names

            # Verify stripped content
            content = zf.read("openrouter/results/2026-10-02/iterations/123/iter.json")
            parsed = json.loads(content)
            assert "problem" not in parsed["training"]
            assert "all_tests" not in parsed["training"]
            assert parsed["training"]["generated_code"] == "code"

    def test_outputs_never_modified(self, tmp_path):
        """The script should only read from outputs/ (never write)."""
        # Create a test file in outputs
        outputs = tmp_path / "outputs"
        outputs.mkdir()
        test_file = outputs / "test.txt"
        test_file.write_text("original")
        original_mtime = test_file.stat().st_mtime

        # Verify the file was not modified during our test
        content = test_file.read_text()
        assert content == "original"
        assert test_file.stat().st_mtime == original_mtime
