"""
Complexity metrics computation using radon and lizard.
"""
import subprocess
import json
import sys
import tempfile
import os
import lizard
from loguru import logger

# Use the radon from the active Python environment
_RADON = [sys.executable, "-m", "radon"]


def cyclomatic_complexity(code: str) -> float:
    """
    Returns the maximum Cyclomatic Complexity value for the code.
    Uses radon cc.
    """
    with tempfile.NamedTemporaryFile(
        suffix=".py", mode="w", delete=False
    ) as f:
        f.write(code)
        tmp = f.name

    try:
        result = subprocess.run(
            _RADON + ["cc", tmp, "-s", "-j"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0 or not result.stdout.strip():
            return 0.0

        data = json.loads(result.stdout)
        scores = []
        for blocks in data.values():
            for block in blocks:
                scores.append(block.get("complexity", 0))
        return max(scores) if scores else 0.0
    except Exception as e:
        logger.warning(f"cyclomatic_complexity: radon cc failed ({e}) — defaulting to 0.0")
        return 0.0
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def maintainability_index(code: str) -> float:
    """
    Returns the Maintainability Index value for the code (0–100).
    Uses radon mi.
    """
    with tempfile.NamedTemporaryFile(
        suffix=".py", mode="w", delete=False
    ) as f:
        f.write(code)
        tmp = f.name

    try:
        result = subprocess.run(
            _RADON + ["mi", tmp, "-j"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0 or not result.stdout.strip():
            return 100.0

        data = json.loads(result.stdout)
        scores = [v.get("mi", 100.0) for v in data.values()]
        return min(scores) if scores else 100.0
    except Exception as e:
        logger.warning(f"maintainability_index: radon mi failed ({e}) — defaulting to 100.0")
        return 100.0
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def halstead_volume(code: str) -> float:
    """
    Returns the Halstead Volume value for the code.
    Uses radon hal.
    """
    with tempfile.NamedTemporaryFile(
        suffix=".py", mode="w", delete=False
    ) as f:
        f.write(code)
        tmp = f.name

    try:
        result = subprocess.run(
            _RADON + ["hal", tmp, "-j"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0 or not result.stdout.strip():
            return 0.0

        data = json.loads(result.stdout)
        volumes = []
        for v in data.values():
            total = v.get("total", {})
            volumes.append(float(total.get("volume", 0.0)))
        return max(volumes) if volumes else 0.0
    except Exception as e:
        logger.warning(f"halstead_volume: radon hal failed ({e}) — defaulting to 0.0")
        return 0.0
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def cognitive_complexity(code: str) -> float:
    """
    Returns the maximum Cognitive Complexity value for the code.
    Uses lizard.
    """
    try:
        result = lizard.analyze_file.analyze_source_code("tmp.py", code)
        scores = [fn.cyclomatic_complexity for fn in result.function_list]
        return max(scores) if scores else 0.0
    except Exception as e:
        logger.warning(f"cognitive_complexity: lizard failed ({e}) — defaulting to 0.0")
        return 0.0
