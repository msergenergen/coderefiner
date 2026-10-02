"""
Spending guard for paid API backends (ENV=openrouter).

Every paid call is appended to a JSON-lines ledger. Before a call, the
cumulative spend plus the call's worst-case cost is compared with the cap;
if it would exceed the cap, BudgetExceeded is raised and the run stops.

The ledger is shared by all shard processes (append under an exclusive
file lock), so the cap holds across parallel shards and across runs.
Concurrent shards can each pass the check before any of them records, so
the overshoot is bounded by (shards × one call's worst-case cost).
"""
import fcntl
import json
import os
import time

from config import settings


class BudgetExceeded(RuntimeError):
    """Raised before a call that could push total spend past the cap."""


def _ledger_path() -> str:
    return settings.api_ledger_path


def spent_usd() -> float:
    path = _ledger_path()
    if not os.path.exists(path):
        return 0.0
    total = 0.0
    with open(path, encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_SH)
        for line in f:
            line = line.strip()
            if line:
                total += float(json.loads(line).get("cost_usd", 0.0))
    return total


def check(reserve_usd: float) -> None:
    """Raise BudgetExceeded if spend + reserve would exceed the cap."""
    cap = settings.api_budget_usd
    spent = spent_usd()
    if spent + reserve_usd > cap:
        raise BudgetExceeded(
            f"API budget reached: spent ${spent:.4f} + next call up to "
            f"${reserve_usd:.4f} > cap ${cap:.2f} (ledger: {_ledger_path()})"
        )


def record(entry: dict) -> None:
    path = _ledger_path()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    entry = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **entry}
    with open(path, "a", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        f.flush()
