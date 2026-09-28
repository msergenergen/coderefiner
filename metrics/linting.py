"""
Linting metrics: mypy (type checking) + ruff (style/undefined names).

Both run on a temp file via subprocess, following the same pattern as bandit_scan.
Results are added to node1_issues for Node 3 feedback — no severity change.
"""
import json
import os
import subprocess
import sys
import tempfile
from loguru import logger


def mypy_check(code: str) -> dict:
    """
    Runs mypy type checking on generated code.

    Returns:
        {
            "error_count": int,
            "errors": list[str]   # Up to 5 human-readable messages
        }
    """
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(code)
        tmp = f.name

    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "mypy", tmp,
                "--ignore-missing-imports",
                "--no-error-summary",
                "--no-pretty",
            ],
            capture_output=True, text=True, timeout=15,
        )
        errors = []
        for line in result.stdout.splitlines():
            if ": error:" in line:
                # Strip temp file path prefix for readability
                msg = line.split(tmp)[-1].lstrip(":").strip()
                errors.append(msg)

        return {
            "error_count": len(errors),
            "errors":      errors[:5],
        }
    except Exception as e:
        logger.warning(f"mypy_check: mypy failed ({e}) — defaulting to no errors")
        return {"error_count": 0, "errors": []}
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def ruff_check(code: str) -> dict:
    """
    Runs ruff linter on generated code.
    Targets: E/W (style), F (pyflakes — undefined names, unused imports).

    Returns:
        {
            "issue_count": int,
            "issues": list[str]   # Up to 5 human-readable messages
        }
    """
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(code)
        tmp = f.name

    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "ruff", "check", tmp,
                "--select", "E,W,F",
                "--output-format", "json",
                "--quiet",
            ],
            capture_output=True, text=True, timeout=15,
        )
        raw = result.stdout.strip()
        if not raw:
            return {"issue_count": 0, "issues": []}

        findings = json.loads(raw)
        messages = [
            f"[{f['code']}] line {f['location']['row']}: {f['message']}"
            for f in findings
        ]
        return {
            "issue_count": len(findings),
            "issues":      messages[:5],
        }
    except Exception as e:
        logger.warning(f"ruff_check: ruff failed ({e}) — defaulting to no issues")
        return {"issue_count": 0, "issues": []}
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass
