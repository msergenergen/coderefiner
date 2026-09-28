"""
100-point rubric computation — numeric dimensions.
Algorithm Choice (10p) and Error Handling (10p) are filled by the Judge model.
"""
from config import thresholds


def compute_accuracy_score(pass_rate: float) -> float:
    """Accuracy score: 0–25 points."""
    return round(pass_rate * 25, 2)


def compute_efficiency_score(cpu_ms: float, ram_mb: float) -> float:
    """Efficiency score: 0–20 points.
    Limits sourced from config.thresholds.node2 (cpu_limit_ms, ram_limit_mb)
    so this formula can't silently desync from the threshold table shown to the judge.
    """
    n2 = thresholds.node2
    cpu_score = max(0.0, 1 - cpu_ms / n2.cpu_limit_ms)
    ram_score = max(0.0, 1 - ram_mb / n2.ram_limit_mb)
    return round((cpu_score + ram_score) / 2 * 20, 2)


def compute_quality_score(mi: float, cc: float) -> float:
    """Code quality score: 0–20 points."""
    mi_score = min(1.0, mi / 100)
    cc_score = max(0.0, 1 - cc / thresholds.node1.cc_hard_limit)
    return round((mi_score + cc_score) / 2 * 20, 2)


def compute_security_score(high_count: int, medium_count: int) -> float:
    """Security score: 0–15 points."""
    score = 15 - (high_count * 15) - (medium_count * 3)
    return round(max(0.0, float(score)), 2)


def compute_numeric_scores(
    pass_rate: float,
    cpu_ms: float,
    ram_mb: float,
    mi: float,
    cc: float,
    bandit_high: bool,
    bandit_medium_count: int = 0,
) -> dict:
    """
    Computes the 4 numerically calculable dimensions.

    Returns:
        {
            "accuracy":      float,   # 0–25
            "efficiency":    float,   # 0–20
            "quality":       float,   # 0–20
            "security":      float,   # 0–15
            "numeric_total": float    # 0–80
        }
    """
    acc  = compute_accuracy_score(pass_rate)
    eff  = compute_efficiency_score(cpu_ms, ram_mb)
    qual = compute_quality_score(mi, cc)
    sec  = compute_security_score(
        high_count=1 if bandit_high else 0,
        medium_count=bandit_medium_count,
    )
    return {
        "accuracy":      acc,
        "efficiency":    eff,
        "quality":       qual,
        "security":      sec,
        "numeric_total": round(acc + eff + qual + sec, 2),
    }
