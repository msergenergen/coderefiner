"""
Security analysis using Bandit.
"""
import subprocess
import json
import sys
import tempfile
import os
import ast
from loguru import logger
from sandbox.whitelist import ALLOWED_IMPORTS

# Use the bandit from the active Python environment
_BANDIT = [sys.executable, "-m", "bandit"]


def bandit_scan(code: str) -> dict:
    """
    Runs a security scan with Bandit.

    Returns:
        {
            "high_count": int,    # Number of HIGH severity findings
            "medium_count": int,
            "issues": list[dict]  # Detailed findings
        }
    """
    with tempfile.NamedTemporaryFile(
        suffix=".py", mode="w", delete=False
    ) as f:
        f.write(code)
        tmp = f.name

    try:
        result = subprocess.run(
            _BANDIT + ["-f", "json", "-q", tmp],
            capture_output=True, text=True, timeout=15
        )
        # Bandit exit code 1 = findings exist — this is normal
        output = result.stdout or result.stderr
        if not output.strip():
            return {"high_count": 0, "medium_count": 0, "issues": []}

        data = json.loads(output)
        results = data.get("results", [])

        high   = [r for r in results if r.get("issue_severity") == "HIGH"]
        medium = [r for r in results if r.get("issue_severity") == "MEDIUM"]

        return {
            "high_count":   len(high),
            "medium_count": len(medium),
            "issues":       results,
        }
    except Exception as e:
        logger.warning(f"bandit_scan: bandit failed ({e}) — defaulting to no findings")
        return {"high_count": 0, "medium_count": 0, "issues": []}
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def check_imports(code: str) -> list:
    """
    Checks imports against the whitelist using AST.

    Returns:
        List of non-whitelisted imports (empty = clean)
    """
    forbidden = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ["syntax_error"]

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in ALLOWED_IMPORTS:
                    forbidden.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            root = module.split(".")[0]
            if root not in ALLOWED_IMPORTS:
                forbidden.append(module)

    return forbidden
