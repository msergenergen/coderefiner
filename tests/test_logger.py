"""
Unit tests: memory/logger.py
_append_row(), read_memory(), JSON iteration writing.

File write tests use tmp_path for a temporary directory.
"""
import os
import json
import pytest
from unittest.mock import patch


class TestAppendRow:

    def test_memory_md_created(self, tmp_path, base_state):
        log_path = str(tmp_path / "memory.md")
        with patch("memory.logger.settings") as mock_settings:
            mock_settings.memory_log_path = log_path
            mock_settings.output_dir      = str(tmp_path)

            from memory.logger import log_iteration_node
            state = base_state(
                node3_total_score=62.0, node3_pass=False,
                run_id="test_run"
            )
            log_iteration_node(state)

        assert os.path.exists(log_path)

    def test_header_row_written(self, tmp_path, base_state):
        log_path = str(tmp_path / "memory.md")
        with patch("memory.logger.settings") as mock_settings:
            mock_settings.memory_log_path = log_path
            mock_settings.output_dir      = str(tmp_path)

            from memory.logger import log_iteration_node
            log_iteration_node(base_state(node3_total_score=62.0,
                                          node3_pass=False, run_id="r"))

        content = open(log_path).read()
        assert "Pipeline Run Log" in content
        assert "| iter |" in content

    def test_loop_status_written(self, tmp_path, base_state):
        log_path = str(tmp_path / "memory.md")
        with patch("memory.logger.settings") as mock_settings:
            mock_settings.memory_log_path = log_path
            mock_settings.output_dir      = str(tmp_path)

            from memory.logger import log_iteration_node
            log_iteration_node(base_state(
                node3_total_score=62.0, node3_pass=False,
                iteration=1, run_id="r"
            ))

        content = open(log_path).read()
        assert "LOOP" in content

    def test_pass_status_written(self, tmp_path, base_state):
        log_path = str(tmp_path / "memory.md")
        with patch("memory.logger.settings") as mock_settings:
            mock_settings.memory_log_path = log_path
            mock_settings.output_dir      = str(tmp_path)

            from memory.logger import log_iteration_node
            log_iteration_node(base_state(
                node3_total_score=85.0, node3_pass=True,
                iteration=2, run_id="r"
            ))

        content = open(log_path).read()
        assert "PASS" in content

    def test_stop_high_written(self, tmp_path, base_state):
        log_path = str(tmp_path / "memory.md")
        with patch("memory.logger.settings") as mock_settings:
            mock_settings.memory_log_path = log_path
            mock_settings.output_dir      = str(tmp_path)

            from memory.logger import log_severity_stop_node
            log_severity_stop_node(base_state(
                node1_severity="HIGH", run_id="r"
            ))

        content = open(log_path).read()
        assert "STOP_HIGH" in content


class TestReadMemory:

    def test_missing_file_returns_empty_list(self, tmp_path):
        from memory.logger import read_memory
        result = read_memory(str(tmp_path / "missing.md"))
        assert result == []

    def test_written_rows_readable(self, tmp_path, base_state):
        log_path = str(tmp_path / "memory.md")
        with patch("memory.logger.settings") as mock_settings:
            mock_settings.memory_log_path = log_path
            mock_settings.output_dir      = str(tmp_path)

            from memory.logger import log_iteration_node, read_memory
            log_iteration_node(base_state(
                node3_total_score=62.0, node3_pass=False, run_id="r"
            ))
            rows = read_memory(log_path)

        assert len(rows) == 1
        assert rows[0]["status"] == "LOOP"
        assert rows[0]["model"]  == "M1"


class TestIterationJson:

    def test_json_file_created(self, tmp_path, base_state):
        with patch("memory.logger.settings") as mock_settings, \
             patch("memory.logger._dated_output_dir", return_value=str(tmp_path)):
            mock_settings.memory_log_path = str(tmp_path / "memory.md")
            mock_settings.output_dir      = str(tmp_path)

            from memory.logger import log_iteration_node
            log_iteration_node(base_state(
                question_id="q_001", node3_total_score=62.0,
                node3_pass=False, run_id="r"
            ))

        json_path = tmp_path / "iterations" / "q_001" / "M1_iter1_r.json"
        assert json_path.exists()

    def test_json_content_correct(self, tmp_path, base_state):
        with patch("memory.logger.settings") as mock_settings, \
             patch("memory.logger._dated_output_dir", return_value=str(tmp_path)):
            mock_settings.memory_log_path = str(tmp_path / "memory.md")
            mock_settings.output_dir      = str(tmp_path)

            from memory.logger import log_iteration_node
            log_iteration_node(base_state(
                question_id="q_001", node3_total_score=62.0,
                node3_pass=False, run_id="r_42",
                node3_improved_prompt="Check for empty list."
            ))

        data = json.load(open(tmp_path / "iterations" / "q_001" / "M1_iter1_r_42.json"))
        assert data["run_id"]  == "r_42"
        assert data["status"]  == "LOOP"
        assert data["node3"]["improved_prompt"] == "Check for empty list."
        assert "node1" in data
        assert "node2" in data
        assert "timing" in data
        assert "tokens" in data
