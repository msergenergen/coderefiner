"""
All metric thresholds in one place.
Node 1, Node 2, Node 3, and the judgement prompt all read from this module.
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Node1Thresholds:
    cc_hard_limit:  float = 20.0    # CC > 20  → HIGH severity → STOP
    cc_warn:        float = 10.0    # CC > 10  → issue reported
    mi_low_limit:   float = 40.0    # MI < 40  → LOW severity
    mi_warn:        float = 65.0    # MI < 65  → issue reported
    halstead_warn:  float = 1000.0  # > 1000   → LOW severity
    cc_ideal:       float = 5.0
    mi_ideal:       float = 85.0
    halstead_ideal: float = 500.0


@dataclass(frozen=True)
class Node2Thresholds:
    cpu_limit_ms:         float = 2000.0
    ram_limit_mb:         float = 512.0
    pass_rate_easy:       float = 0.90
    pass_rate_medium:     float = 0.75
    pass_rate_hard:       float = 0.55
    timeout_rate_alarm:   float = 0.10
    nondeterminism_alarm: float = 0.03


@dataclass(frozen=True)
class Node3Thresholds:
    easy:   int = 80
    medium: int = 70
    hard:   int = 60

    def for_difficulty(self, difficulty: str) -> int:
        mapping = {"easy": self.easy, "medium": self.medium, "hard": self.hard}
        if difficulty not in mapping:
            raise ValueError(f"Invalid difficulty: {difficulty!r}")
        return mapping[difficulty]

    def as_dict(self) -> dict:
        return {"easy": self.easy, "medium": self.medium, "hard": self.hard}


@dataclass(frozen=True)
class AllThresholds:
    node1: Node1Thresholds = field(default_factory=Node1Thresholds)
    node2: Node2Thresholds = field(default_factory=Node2Thresholds)
    node3: Node3Thresholds = field(default_factory=Node3Thresholds)

    def judgement_context_table(self) -> str:
        """
        Generates the threshold table embedded in the judgement prompt.
        Hardcoded markdown string avoided — if values change, table updates automatically.
        Pass rate rows are split by difficulty because easy/medium/hard have different targets.
        """
        n1 = self.node1
        n2 = self.node2
        # "Poor" lower bounds: easy/medium drop by 20 pp; medium's floor equals the hard target
        poor_easy   = int(n2.pass_rate_easy   * 100) - 20  # 90 - 20 = 70%
        poor_medium = int(n2.pass_rate_hard   * 100)       # 55% (hard target = medium's poor floor)
        poor_hard   = int(n2.pass_rate_hard   * 100) - 20  # 55 - 20 = 35%
        return f"""## Metric Threshold Reference

| Metric | Good | Acceptable | Poor |
|--------|------|------------|------|
| Cyclomatic Complexity (CC) | ≤ {n1.cc_ideal:.0f} | {n1.cc_ideal+1:.0f}–{n1.cc_warn:.0f} | > {n1.cc_warn:.0f} |
| Maintainability Index (MI) | ≥ {n1.mi_ideal:.0f} | {n1.mi_warn:.0f}–{n1.mi_ideal:.0f} | < {n1.mi_warn:.0f} |
| Halstead Volume | ≤ {n1.halstead_ideal:.0f} | {n1.halstead_ideal:.0f}–{n1.halstead_warn:.0f} | > {n1.halstead_warn:.0f} |
| Test Pass Rate — Easy   | ≥ {n2.pass_rate_easy:.0%}  | {poor_easy}%–{int(n2.pass_rate_easy*100)}%   | < {poor_easy}%  |
| Test Pass Rate — Medium | ≥ {n2.pass_rate_medium:.0%}  | {poor_medium}%–{int(n2.pass_rate_medium*100)}%   | < {poor_medium}%  |
| Test Pass Rate — Hard   | ≥ {n2.pass_rate_hard:.0%}  | {poor_hard}%–{int(n2.pass_rate_hard*100)}%   | < {poor_hard}%  |
| Peak Memory | ≤ {n2.ram_limit_mb/2:.0f} MB | {n2.ram_limit_mb/2:.0f}–{n2.ram_limit_mb:.0f} MB | > {n2.ram_limit_mb:.0f} MB |
| CPU Time | ≤ {n2.cpu_limit_ms/2000:.0f} s | {n2.cpu_limit_ms/2000:.0f}–{n2.cpu_limit_ms/1000:.0f} s | > {n2.cpu_limit_ms/1000:.0f} s |

Bandit HIGH severity: zero tolerance.
MI < {n1.mi_warn:.0f} means the code is hard to maintain.
CC > {n1.cc_warn:.0f} means the function has too many branches.
IMPORTANT: Apply the pass rate row that matches the problem's difficulty level."""


thresholds = AllThresholds()
