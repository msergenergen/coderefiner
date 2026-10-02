# graph.md — LangGraph Graph Definition

## 1. Graph Flow

```
START
  ↓
codegen             ← timing["codegen_ms"] + tokens["codegen_*"] yazılır
  ↓
static_analysis     ← timing["node1_ms"] yazılır
  ↓
severity_gate ──── HIGH ──→ log_severity_stop_node → END
  │                              (STOP_HIGH: kısmi timing, sıfır token[judgement])
  │ LOW / OK
  ↓
sandbox             ← timing["node2_ms"] yazılır  (LOW → atlanır, 0 kalır)
  ↓
strip_comments      ← generated_code → stripped_code (yorumlar kaldırılır)
  ↓                   generated_code değişmez (sandbox + log bu kodu kullandı)
judgement            ← stripped_code üzerinden değerlendirir
  ↓                   timing["node3_ms"] + tokens["judgement_*"] yazılır
log_iteration_node  ← Her iterasyon: memory.md + JSON yazılır
  ↓                   timing["total_ms"] hesaplanır
iteration_gate
  ├─ PASS      → write_memory_final → next_model → END
  ├─ STOP_MAX  → stop_max → END          ← trend extension aktifse atlanabilir
  └─ LOOP      → codegen (sonraki iterasyon veya trend extension)
```

**PASS** = `node3_pass`: every public test passes AND the judge says ACCEPT (verdict design);
see `docs/node3.md` §4.2. `MAX_ITERATIONS` defaults to 5; trend extension is off by default.

**`strip_comments` neden burada?**
- Sandbox already ran `generated_code` — test results belong to the real code, must not change
- Judgement LLM must not be biased by comments: lines like `# This is very efficient` inflate scores
- `stripped_code` is for judgement only; log, memory.md, and JSON always store `generated_code`

---

## 2. pipeline/state.py

```python
"""
Central state carried between all nodes in the pipeline.
"""
from typing import TypedDict, Literal, Optional


class PipelineState(TypedDict):

    # ── Run and routing ────────────────────────────────────────────────
    run_id:         str          # Set in main.py by write_run_info()
    active_model:   Literal["M1", "M2"]
    variant:        Literal["V1", "V2", "V3"]  # Ablation variant
    iteration:      int
    max_iterations: int          # settings.max_iterations (default 5)

    # ── Input ─────────────────────────────────────────────────────────
    problem:        str          # LCB problem text (format_problem output)
    problem_row:    dict         # Raw row from load_lcb() — includes all_tests
    question_id:    str
    difficulty:     Literal["easy", "medium", "hard"]
    test_inputs:    list         # public_tests["input"] — used in Phase 2 runtime
    test_outputs:   list         # public_tests["output"]
    edge_cases:     list         # augment_row() output — output=None crash tests

    # ── Code generation ───────────────────────────────────────────────
    generated_code: str   # LLM output — used by sandbox and log
    stripped_code:  str   # Comment-stripped version — only judgement sees this

    # ── Node 1 outputs ──────────────────────────────────────────────
    node1_cc:                float
    node1_mi:                float
    node1_halstead:          float
    node1_bandit_high:       bool
    node1_bandit_medium_count: int
    node1_severity:          Literal["LOW", "OK", "HIGH"]
    node1_issues:            list

    # ── Node 2 outputs ──────────────────────────────────────────────
    # Correctness — Phase 1 (lcb_runner)
    node2_pass_rate:            float    # 0.0–1.0
    node2_passed_count:         int      # Number of passed tests
    node2_total_count:          int      # Total test count (public + private)
    node2_failed_cases:         list     # Max 5, {input, expected, got, error}
    node2_crash_count:          int      # Edge case runtime errors
    node2_test_diversity_score: float    # 1 - crash_count/len(edge_cases)
    node2_lcb_available:        bool     # lcb_runner installed?
    # Runtime — Phase 2 (subprocess)
    node2_cpu_ms:    float
    node2_ram_mb:    float
    node2_timed_out: bool

    # ── Node 3 outputs ──────────────────────────────────────────────
    node3_total_score:     float
    node3_pass:            bool
    node3_score_breakdown: dict        # {"accuracy": 0, "efficiency": 0, "quality": 0, "security": 0, "algorithm_choice": 0, "error_handling": 0}
    node3_failure_reasons: list
    node3_improved_prompt: str         # Completely rewritten every iteration
    node3_verdict:         str         # "ACCEPT" / "REJECT" from the judge ("" if absent)
    node3_verdict_reason:  str         # judge's justification for the verdict
    node3_verdict_source:  str         # "judge" | "fallback_rubric" | "rubric_mode"

    # ── Performance metrics (each node writes its own duration) ─────────
    timing: dict
    # {
    #   "codegen_ms": 0,   ← written by codegen_node
    #   "node1_ms":   0,   ← written by static_analysis_node
    #   "node2_ms":   0,   ← written by sandbox_node (stays 0 when LOW)
    #   "node3_ms":   0,   ← written by judgement_node
    #   "total_ms":   0    ← computed by log_iteration_node
    # }

    tokens: dict
    # {
    #   "codegen_prompt":      0,   ← written by codegen_node
    #   "codegen_completion":  0,   ← written by codegen_node
    #   "judgement_prompt":     0,   ← written by judgement_node (0 on STOP_HIGH)
    #   "judgement_completion": 0,   ← written by judgement_node (0 on STOP_HIGH)
    #   "total":               0    ← computed by log_iteration_node
    # }

    # ── Trend-aware extension ─────────────────────────────────────────
    score_history:     list   # [float] — node3_total_score appended each iteration
    pass_rate_history: list   # [float] — node2_pass_rate appended each iteration
    extension_count:   int    # number of trend extensions consumed (per model)

    # ── Log ───────────────────────────────────────────────────────────
    memory_log:   list
    final_status: Literal["PASS", "STOP_HIGH", "STOP_MAX", "PENDING"]
```

---

## 3. pipeline/gates.py

```python
"""
Conditional edge functions.
iteration_gate runs AFTER log_iteration_node —
logging is already complete, only the routing decision is made here.
"""
from config import thresholds, settings
from pipeline.state import PipelineState

PASS_THRESHOLDS = thresholds.node3.as_dict()


def severity_gate(state: PipelineState) -> str:
    """
    Routing after Node 1.

    "to_node2"  → OK  — run sandbox
    "to_node3"  → LOW — skip sandbox, go directly to judgement
    "stop_high" → HIGH — go to log_severity_stop_node
    """
    sev = state["node1_severity"]
    if sev == "HIGH":
        return "stop_high"
    elif sev == "LOW":
        return "to_node3"
    return "to_node2"


def iteration_gate(state: PipelineState) -> str:
    """
    Routing after log_iteration_node.
    The row is already written to memory.md and JSON.

    "write_memory_final" → PASS     — update final_status, advance to next model
    "stop_max"           → STOP_MAX — update final_status, END
    "loop_to_codegen"    → LOOP     — retry with same model (or trend extension)
    """
    threshold = PASS_THRESHOLDS[state["difficulty"]]
    # node3_pass is the single PASS decision (judge verdict or rubric; see
    # nodes/judgement.py). threshold is still used by the trend extension.
    passed    = state["node3_pass"]

    if passed:
        return "write_memory_final"

    if state["iteration"] >= state["max_iterations"]:
        if settings.trend_extension_enabled and _should_extend(state, threshold):
            return "loop_to_codegen"
        return "stop_max"

    return "loop_to_codegen"


# ── Trend-aware extension helpers ─────────────────────────────────────────────

def _compute_ema(values: list, alpha: float) -> list:
    """
    Returns EMA-smoothed version of values.
    alpha=0.5 gives recent values 2× weight over older ones.
    """
    if not values:
        return []
    ema = [values[0]]
    for v in values[1:]:
        ema.append(alpha * v + (1 - alpha) * ema[-1])
    return ema


def _linear_slope(values: list) -> float:
    """
    Returns the slope of a least-squares line fit through values.
    Uses the last min(len(values), 3) points for recency.
    Returns 0.0 for fewer than 2 points.
    """
    pts = values[-3:]
    n = len(pts)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2
    y_mean = sum(pts) / n
    num = sum((i - x_mean) * (pts[i] - y_mean) for i in range(n))
    den = sum((i - x_mean) ** 2 for i in range(n))
    return num / den if den else 0.0


def _should_extend(state: PipelineState, threshold: float) -> bool:
    """
    Returns True when all activation conditions are met:

      1. accuracy > 0           — at least one test passing (hard gate)
      2. pass_rate_slope > 0    — pass rate trending upward (last 3 iters)
      3. score_slope >= min     — EMA-smoothed score genuinely improving
      4. expected_gain >= gap   — projected gain can close the distance to threshold
      5. extension_count < max  — extension budget not exhausted
    """
    # 1. Accuracy hard gate
    accuracy = state.get("node3_score_breakdown", {}).get("accuracy", 0.0)
    if accuracy <= 0.0:
        return False

    # 5. Extension budget
    extension_count = state.get("extension_count", 0)
    if extension_count >= settings.max_extension_iters:
        return False

    score_history     = state.get("score_history", [])
    pass_rate_history = state.get("pass_rate_history", [])

    # Need at least 3 data points for a meaningful trend
    if len(score_history) < 3:
        return False

    # 2. Pass rate must be trending upward
    pr_slope = _linear_slope(pass_rate_history)
    if pr_slope <= 0.0:
        return False

    # 3. EMA-smoothed score slope
    smoothed     = _compute_ema(score_history, alpha=settings.trend_ema_alpha)
    score_slope  = _linear_slope(smoothed)
    if score_slope < settings.trend_min_slope:
        return False

    # 4. Expected gain must be sufficient to potentially reach threshold.
    # Uses a fixed projection window (trend_projection_window) — independent of
    # the extension budget — so increasing max_extension_iters does not
    # artificially inflate the gain estimate.
    expected_gain = score_slope * settings.trend_projection_window
    gap           = threshold - state.get("node3_total_score", 0.0)
    if expected_gain < gap:
        return False

    return True
```

---

## 4. pipeline/graph.py

**2026-09-30 düzeltme — `_strip_before_judgement`'ta stale state:** Bu node
(LOW severity'de sandbox atlandığında çağrılır) node2 alanlarını
`state.get(key, default)` ile "initialize" ediyordu — ama `state` bir
model/problem koşusu boyunca iterasyonlar arasında taşınan AYNI dict
olduğundan, anahtar zaten mevcutsa (iterasyon 2+'de her zaman mevcut)
`.get()` varsayılanı değil, ÖNCEKİ iterasyonun gerçek sandbox sonucunu
döndürüyordu. Örnek: iterasyon N-1 testlerin %80'ini geçmişken, iterasyon N
(ör. sözdizimi hatası nedeniyle) LOW'a yönlendirilirse, Judge'a o kodun hiç
çalışmadığı hâlde "%80 geçti" gösteriliyordu. Artık literal varsayılanlarla
(`0.0`, `0`, `[]`, `False`) sıfırlanıyor — kod için bkz. `pipeline/graph.py`.
Aynı desenin bir kardeşi `memory/logger.py::log_severity_stop_node`'da da
düzeltildi (bkz. `docs/memory-logging.md`).

```python
"""
LangGraph StateGraph definition.

Each iteration is logged via log_iteration_node → memory.md + JSON:
  LOOP      — after judgement, threshold not met
  PASS      — after judgement, threshold met
  STOP_MAX  — after judgement, max_iter reached
  STOP_HIGH — severity HIGH, pipeline stopped early

Each node writes its own timing to state["timing"].
Codegen and judgement nodes write token counts to state["tokens"].
"""
from langgraph.graph import StateGraph, END

from pipeline.state  import PipelineState
from pipeline.gates  import severity_gate, iteration_gate
from pipeline.router import advance_model

from nodes.codegen         import codegen_node
from nodes.static_analysis import static_analysis_node
from nodes.sandbox         import sandbox_node
from nodes.cleaner         import strip_comments_node
from nodes.judgement        import judgement_node
from memory.logger         import log_iteration_node, log_severity_stop_node


def _write_memory_final_node(state: PipelineState) -> PipelineState:
    """PASS: update final_status. Row was already written by log_iteration_node."""
    return {**state, "final_status": "PASS"}


def _stop_max_node(state: PipelineState) -> PipelineState:
    """STOP_MAX: update final_status. Row was already written by log_iteration_node."""
    return {**state, "final_status": "STOP_MAX"}


def _next_model_node(state: PipelineState) -> PipelineState:
    """Advance to the next model after PASS."""
    return advance_model(state)


def _strip_before_judgement(state: PipelineState) -> PipelineState:
    """
    Used when sandbox is skipped (LOW severity in V3, or entire V2 variant).
    Resets node2 fields to safe defaults then strips comments.

    Must assign literal defaults, not `state.get(key, default)` — state is
    the same dict threaded across iterations within one model/problem run,
    so `.get()` silently returns the PREVIOUS iteration's real sandbox
    result instead of a default whenever the key already exists (which it
    always does after iteration 1). Concretely: if iteration N-1 passed 80%
    of tests and iteration N is skipped to LOW (e.g. a syntax-error
    fallback — see nodes/static_analysis.py), the Judge would otherwise be
    shown iteration N-1's 80% pass rate for code that never even ran.
    """
    updated = {
        **state,
        "node2_pass_rate":            0.0,
        "node2_passed_count":         0,
        "node2_total_count":          0,
        "node2_failed_cases":         [],
        "node2_crash_count":          0,
        "node2_test_diversity_score": 1.0,
        "node2_lcb_available":        False,
        "node2_cpu_ms":               0.0,
        "node2_ram_mb":               0.0,
        "node2_timed_out":            False,
    }
    return strip_comments_node(updated)


def build_graph(variant: str = "V3"):
    """
    Builds and compiles the LangGraph state graph.

    Variant matrix:
        V1 — Node 2 + Node 3        (Execution + Judgement, no static analysis)
        V2 — Node 1 + Node 3        (Static Analysis + Judgement, no execution)
        V3 — Node 1 + Node 2 + Node 3  (Full pipeline)

    PASS criterion (all variants):
        node3_total_score >= threshold AND node3_pass
    """
    use_node1 = variant in ("V2", "V3")
    use_node2 = variant in ("V1", "V3")

    g = StateGraph(PipelineState)

    # ── Core nodes — always present ───────────────────────────────────
    g.add_node("codegen",            codegen_node)
    g.add_node("judgement",           judgement_node)
    g.add_node("log_iteration",      log_iteration_node)
    g.add_node("write_memory_final", _write_memory_final_node)
    g.add_node("stop_max",           _stop_max_node)
    g.add_node("next_model",         _next_model_node)

    # ── Conditional nodes ─────────────────────────────────────────────
    if use_node1:
        g.add_node("static_analysis",    static_analysis_node)
        g.add_node("strip_before_judge", _strip_before_judgement)
        g.add_node("log_severity_stop",  log_severity_stop_node)

    if use_node2:
        g.add_node("sandbox",        sandbox_node)
        g.add_node("strip_comments", strip_comments_node)

    # ── Entry point ──────────────────────────────────────────────────
    g.set_entry_point("codegen")

    # ── Routing ──────────────────────────────────────────────────────
    if use_node1:
        g.add_edge("codegen", "static_analysis")

        if use_node2:
            # V3: OK → sandbox, LOW → strip_before_judge, HIGH → stop
            g.add_conditional_edges(
                "static_analysis",
                severity_gate,
                {
                    "to_node2":  "sandbox",
                    "to_node3":  "strip_before_judge",
                    "stop_high": "log_severity_stop",
                },
            )
            g.add_edge("sandbox",            "strip_comments")
            g.add_edge("strip_comments",     "judgement")
        else:
            # V2: OK and LOW both → strip_before_judge (no sandbox)
            g.add_conditional_edges(
                "static_analysis",
                severity_gate,
                {
                    "to_node2":  "strip_before_judge",  # OK → skip sandbox
                    "to_node3":  "strip_before_judge",  # LOW → skip sandbox
                    "stop_high": "log_severity_stop",
                },
            )

        g.add_edge("strip_before_judge", "judgement")

    else:
        # V1: codegen → sandbox → strip_comments → judgement
        g.add_edge("codegen",        "sandbox")
        g.add_edge("sandbox",        "strip_comments")
        g.add_edge("strip_comments", "judgement")

    # ── Judgement → iteration gate ────────────────────────────────────
    g.add_edge("judgement", "log_iteration")

    g.add_conditional_edges(
        "log_iteration",
        iteration_gate,
        {
            "write_memory_final": "write_memory_final",
            "stop_max":           "stop_max",
            "loop_to_codegen":    "codegen",
        },
    )

    # ── Terminal edges ────────────────────────────────────────────────
    g.add_edge("write_memory_final", "next_model")
    g.add_edge("stop_max",           END)
    g.add_edge("next_model",         END)

    if use_node1:
        g.add_edge("log_severity_stop", END)

    return g.compile()
```

---

## 5. main.py

```python
"""
Pipeline entry point.
Runs the generators in PIPELINE_MODELS (default: M1 only) for each LCB problem.
"""
import os

from dotenv import load_dotenv
from loguru import logger

from pipeline.graph   import build_graph
from pipeline.state   import PipelineState
from config           import settings
from dataset.loader   import load_lcb
from dataset.splitter import format_problem
from dataset.edge_cases import augment_row
from memory.logger    import write_run_info

load_dotenv()

_EMPTY_TIMING = {
    "codegen_ms": 0,
    "node1_ms":   0,
    "node2_ms":   0,
    "node3_ms":   0,
    "total_ms":   0,
}
_EMPTY_TOKENS = {
    "codegen_prompt":      0,
    "codegen_completion":  0,
    "judgement_prompt":     0,
    "judgement_completion": 0,
    "total":               0,
}


def run_pipeline_for_problem(problem_row: dict, run_id: str) -> dict:
    """
    Runs the pipeline for a single LCB problem, once per generator in PIPELINE_MODELS.
    Each model gets its own LangGraph invocation with a fresh initial state.

    Returns a summary dict with per-model status and score.
    """
    graph = build_graph()

    aug_row      = augment_row(problem_row)
    edge_cases   = aug_row.get("edge_cases", [])
    pub_inputs   = problem_row.get("public_tests", {}).get("input",  [])
    pub_outputs  = problem_row.get("public_tests", {}).get("output", [])
    problem_text = format_problem(problem_row)
    question_id  = problem_row.get("question_id", problem_row.get("id", "unknown"))
    difficulty   = problem_row.get("difficulty", "medium")

    all_results = []

    for model_id in settings.model_sequence:
        logger.info(f"[{model_id}] starting — {problem_row.get('title', question_id)}")

        initial_state: PipelineState = {
            # Run metadata
            "run_id":         run_id,
            "active_model":   model_id,
            "variant":        "V3",
            "iteration":      0,
            "max_iterations": settings.max_iterations,

            # Problem input
            "problem":      problem_text,
            "problem_row":  aug_row,
            "question_id":  question_id,
            "difficulty":   difficulty,
            "test_inputs":  pub_inputs,
            "test_outputs": pub_outputs,
            "edge_cases":   edge_cases,

            # Code (empty until codegen runs)
            "generated_code": "",
            "stripped_code":  "",

            # Node 1 defaults
            "node1_cc":                  0.0,
            "node1_mi":                  100.0,
            "node1_halstead":            0.0,
            "node1_bandit_high":         False,
            "node1_bandit_medium_count": 0,
            "node1_severity":            "OK",
            "node1_issues":              [],

            # Node 2 defaults
            "node2_pass_rate":           0.0,
            "node2_passed_count":        0,
            "node2_total_count":         0,
            "node2_failed_cases":        [],
            "node2_crash_count":         0,
            "node2_test_diversity_score": 1.0,
            "node2_lcb_available":       False,
            "node2_cpu_ms":              0.0,
            "node2_ram_mb":              0.0,
            "node2_timed_out":           False,

            # Node 3 defaults
            "node3_total_score":     0.0,
            "node3_pass":            False,
            "node3_score_breakdown": {},
            "node3_failure_reasons": [],
            "node3_improved_prompt": "",

            # Performance metrics
            "timing": dict(_EMPTY_TIMING),
            "tokens": dict(_EMPTY_TOKENS),

            # Log
            "memory_log":   [],
            "final_status": "PENDING",
        }

        final_state = graph.invoke(initial_state)
        status      = final_state["final_status"]
        score       = final_state.get("node3_total_score", 0.0)

        logger.info(f"[{model_id}] done — status={status} score={score:.1f}")
        all_results.append({"model": model_id, "status": status, "score": score})

        # On PASS, remaining models still run per the sequential design
        # (each model is independent; M1 PASS does not skip M2)

    return {
        "problem_id": question_id,
        "difficulty": difficulty,
        "results":    all_results,
    }


def main():
    logger.info("Pipeline starting...")

    run_id = write_run_info()
    logger.info(f"run_id: {run_id}")

    rows = load_lcb(
        split=settings.lcb_split,
        difficulty=settings.lcb_difficulty,
        sample_size=settings.lcb_sample_size,
        date_from=settings.lcb_date_from,
        date_to=settings.lcb_date_to,
    )
    logger.info(f"{len(rows)} problems loaded.")

    # Optional cap: first N problems of the seeded sample (e.g. a 50-problem run).
    max_problems = int(os.getenv("MAX_PROBLEMS", "0"))
    if max_problems > 0:
        rows = rows[:max_problems]
        logger.info(f"Limiting to the first {len(rows)} problems of the sample.")

    # Optional resume: skip problems already finished in an earlier run
    # (comma-separated question_ids). Order of the seeded sample is kept.
    skip = {q.strip() for q in os.getenv("SKIP_QIDS", "").split(",") if q.strip()}
    if skip:
        rows = [r for r in rows if r.get("question_id", r.get("id")) not in skip]
        logger.info(f"Skipping {len(skip)} finished problems; {len(rows)} remain.")

    # Optional sharding so N processes can share one llama-server started with
    # --parallel N. Slicing happens last (after MAX_PROBLEMS / SKIP_QIDS), so
    # the union of all shards is exactly the single-process problem set.
    shard_count = int(os.getenv("SHARD_COUNT", "1"))
    shard_index = int(os.getenv("SHARD_INDEX", "0"))
    if shard_count > 1:
        rows = rows[shard_index::shard_count]
        logger.info(f"Shard {shard_index}/{shard_count}: {len(rows)} problems.")

    all_results = []
    for row in rows:
        result = run_pipeline_for_problem(row, run_id)
        logger.info(f"Problem done: {result}")
        all_results.append(result)

    # ── Summary ───────────────────────────────────────────────────────────────
    status_counts = {"PASS": 0, "STOP_MAX": 0, "STOP_HIGH": 0}
    scores = []
    for res in all_results:
        for m in res["results"]:
            s = m["status"]
            status_counts[s] = status_counts.get(s, 0) + 1
            if m["score"] > 0:
                scores.append(m["score"])

    total_runs = sum(status_counts.values())
    avg_score  = round(sum(scores) / len(scores), 1) if scores else 0.0

    logger.info("─" * 42)
    logger.info("Pipeline complete.")
    logger.info(f"  Problems  : {len(all_results)}")
    logger.info(f"  Total runs: {total_runs}  (models × problems)")
    for status, count in status_counts.items():
        pct = round(count / total_runs * 100, 1) if total_runs else 0.0
        logger.info(f"  {status:<10}: {count}  ({pct}%)")
    logger.info(f"  Avg score : {avg_score}")
    logger.info("─" * 42)


if __name__ == "__main__":
    main()
```

---

## 6. Verification

### Graph Structure

```bash
python -c "
from pipeline.graph import build_graph
g = build_graph()
print('Graf derlendi.')
# 10 node beklenir:
# codegen, static_analysis, sandbox, strip_comments, judgement,
# log_iteration, log_severity_stop,
# write_memory_final, stop_max, next_model
"
```

### strip_comments_node Testi

```bash
python -c "
from nodes.cleaner import strip_comments_node

# Yorumlu kod
state = {
    'generated_code': '''
# Add two numbers
def add(a, b):
    # a ve b parametrelerini topla
    result = a + b  # sonucu hesapla
    return result   # return
'''.strip()
}

out = strip_comments_node(state)
print('--- stripped_code ---')
print(out['stripped_code'])
print('--- generated_code (değişmedi) ---')
print(out['generated_code'] == state['generated_code'])
"
```

**Expected output:**

```
--- stripped_code ---
def add(a, b):
    result = a + b
    return result

--- generated_code (değişmedi) ---
True
```

### Gate Testleri

```bash
python -c "
from pipeline.gates import severity_gate, iteration_gate

assert severity_gate({'node1_severity': 'HIGH'}) == 'stop_high'
assert severity_gate({'node1_severity': 'LOW'})  == 'to_node3'
assert severity_gate({'node1_severity': 'OK'})   == 'to_node2'
print('severity_gate: OK')

base = {'difficulty': 'easy', 'iteration': 3, 'max_iterations': 5}
assert iteration_gate({**base, 'node3_total_score': 85, 'node3_pass': True})              == 'write_memory_final'
assert iteration_gate({**base, 'node3_total_score': 60, 'node3_pass': False})             == 'loop_to_codegen'
assert iteration_gate({**base, 'node3_total_score': 60, 'node3_pass': False,
                        'iteration': 5})                                                   == 'stop_max'
print('iteration_gate: OK')
"
```

### Verify All Statuses Are Written

```bash
python -c "
import os, json
os.makedirs('outputs', exist_ok=True)
if os.path.exists('outputs/memory.md'): os.remove('outputs/memory.md')

from memory.logger import write_run_info, log_iteration_node, log_severity_stop_node, read_memory, read_iteration

run_id = write_run_info()

def s(q='q001', **kw):
    base = {
        'run_id': run_id, 'active_model': 'M1', 'iteration': 1,
        'max_iterations': 5, 'question_id': q, 'difficulty': 'easy',
        'problem': '', 'test_inputs': [], 'test_outputs': [],
        'generated_code': 'def solve(): pass',
        'node1_cc': 3.0, 'node1_mi': 72.0, 'node1_halstead': 150.0,
        'node1_issues': [], 'node1_bandit_high': False, 'node1_severity': 'OK',
        'node2_pass_rate': 0.65, 'node2_cpu_ms': 500.0,
        'node2_ram_mb': 120.0, 'node2_test_diversity_score': 0.6,
        'node2_failed_cases': [{'input':'[]','expected':'0','got':'None'}],
        'node3_total_score': 62.0, 'node3_pass': False,
        'node3_score_breakdown': {'accuracy':16.25,'efficiency':12.0,'quality':11.0,
                                   'security':15.0,'algorithm_choice':5,'error_handling':3},
        'node3_failure_reasons': ['pass_rate düşük'],
        'node3_improved_prompt': 'Boş liste kontrolü ekle.',
        'timing': {'codegen_ms':3200,'node1_ms':45,'node2_ms':1200,'node3_ms':2800,'total_ms':7245},
        'tokens': {'codegen_prompt':450,'codegen_completion':320,
                   'judgement_prompt':1200,'judgement_completion':580,'total':2550},
        'memory_log': [], 'final_status': 'PENDING',
    }
    base.update(kw)
    return base

# LOOP
log_iteration_node(s('q001', iteration=1, node3_total_score=62, node3_pass=False))
# PASS
log_iteration_node(s('q001', iteration=2, node3_total_score=85, node3_pass=True))
# STOP_MAX
log_iteration_node(s('q001', iteration=5, node3_total_score=55, node3_pass=False, max_iterations=5))
# STOP_HIGH
log_severity_stop_node(s('q002', node1_severity='HIGH',
                          timing={'codegen_ms':3100,'node1_ms':42,'node2_ms':0,'node3_ms':0,'total_ms':3142},
                          tokens={'codegen_prompt':420,'codegen_completion':310,
                                  'judgement_prompt':0,'judgement_completion':0,'total':730}))

rows = read_memory()
print('Satır sayısı:', len(rows))
print('Statuslar:   ', [r['status'] for r in rows])
print('total_ms[0]: ', rows[0]['total_ms'])
print('total_tok[0]:', rows[0]['total_tok'])

# JSON detail check
detail = read_iteration('q001', 'M1', 1)
print('improved_prompt:', detail['node3']['improved_prompt'])
print('node1_issues:   ', detail['node1']['issues'])
print('failed_cases:   ', detail['node2']['failed_cases'])

# run_info check
info = json.load(open('outputs/run_info.json'))
print('GPU:    ', info['hardware']['gpu_name'])
print('Models: ', info['models'])
"
```

### Expected Output

```
Satır sayısı: 4
Statuslar:    ['LOOP', 'PASS', 'STOP_MAX', 'STOP_HIGH']
total_ms[0]:  7245
total_tok[0]: 2550
improved_prompt: Boş liste kontrolü ekle.
node1_issues:    ['CC yüksek']
failed_cases:    [{'input': '[]', 'expected': '0', 'got': 'None'}]
GPU:     unknown          (Mac mini / OpenRouter; GPU name only on CUDA machines)
Models:  {'M1': 'qwen3.5-9b', 'QWEN': 'qwen3.5-9b'}   (ENV=local)
```

---

## 7. nodes/cleaner.py

**2026-09-30 düzeltme — sadece `tokenize.TokenError` yakalamak yetersizdi:**
Bu commit'in başında `_decide_severity()`, "syntax_error" sentinel'ini
HIGH yerine LOW'a yönlendirecek şekilde değiştirildi — bu da parse
edilemeyen kodun artık `strip_comments_node`'a (ve dolayısıyla
`tokenize.generate_tokens()`'a) ulaşabileceği anlamına geliyordu; önceden
bu kod hiçbir zaman buraya gelmiyordu (doğrudan `log_severity_stop`'a
gidiyordu). `tokenize.generate_tokens()` bozuk girinti seviyelerinde
`tokenize.TokenError` değil `IndentationError`/`SyntaxError` fırlatıyor —
bu ikisi yakalanmadığı için canlı bir 150-problemlik koşuyu 13.6 saatin
sonunda (`IndentationError: unindent does not match any outer indentation
level`) çökertti. Artık üçü de yakalanıyor.

```python
"""
Judgement öncesi yorum temizleyici.

Neden burada:
  - Sandbox generated_code'u çalıştırdı — test sonuçları gerçek koda ait
  - Judgement LLM yorumlardan bias almasın:
      # This is a very efficient algorithm → misleads scoring
      # TODO: can be improved → unnecessary noise
  - stripped_code sadece judgement için; log ve memory.md generated_code'u saklar

Yöntem:
  Python'un tokenize modülü kullanılır.
  AST parse etmek yerine tokenize tercih edilir çünkü:
    - Syntax hatası içeren kodu da işler (LLM bazen hatalı üretir)
    - Docstring'leri korur (bunlar API sözleşmesidir, yorum değildir)
    - Satır sonu boşluklarını temizler
"""
import io
import tokenize
from pipeline.state import PipelineState


def strip_comments(code: str) -> str:
    """
    Python kodundan yorum satırlarını ve inline yorumları kaldırır.
    Docstring'ler (triple-quote) korunur.
    Syntax hatası varsa orijinal kodu döner.

    Örnekler:
        "x = 1  # değer"  →  "x = 1"
        "# başlık\ndef f(): ..."  →  "def f(): ..."
        '\"\"\"docstring\"\"\"'  →  korunur
    """
    try:
        tokens      = tokenize.generate_tokens(io.StringIO(code).readline)
        result      = []
        prev_end    = (1, 0)

        for tok_type, tok_str, tok_start, tok_end, _ in tokens:
            # Skip COMMENT tokens
            if tok_type == tokenize.COMMENT:
                continue
            # Skip ENCODING and ENDMARKER tokens
            if tok_type in (tokenize.ENCODING, tokenize.ENDMARKER):
                continue

            # Preserve line/column spacing (keep indentation intact)
            start_row, start_col = tok_start
            end_row,   _         = prev_end

            if start_row > end_row:
                # New line(s) — blank lines may remain after comment removal
                result.append("\n" * (start_row - end_row))
                result.append(" " * start_col)
            elif start_col > prev_end[1]:
                result.append(" " * (start_col - prev_end[1]))

            result.append(tok_str)
            prev_end = tok_end

        cleaned = "".join(result)

        # Collapse 3+ consecutive blank lines to 2
        import re
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

        return cleaned.strip()

    except (tokenize.TokenError, SyntaxError, IndentationError):
        # Syntax error: return original code, do not stop pipeline
        return code


def strip_comments_node(state: PipelineState) -> PipelineState:
    """
    LangGraph node: sandbox → judgement arasında çalışır.

    generated_code  → değişmez (sandbox çalıştırdı, log bu kodu saklar)
    stripped_code   → judgement LLM'e bu gönderilir
    """
    cleaned = strip_comments(state["generated_code"])
    return {**state, "stripped_code": cleaned}
```

### Behavior Examples

```python
from nodes.cleaner import strip_comments

# 1. Line-start comment
strip_comments("# header\ndef f():\n    return 1")
# → "def f():\n    return 1"

# 2. Inline yorum
strip_comments("x = a + b  # topla")
# → "x = a + b"

# 3. Docstring korunur
strip_comments('def f():\n    """Açıklama."""\n    return 1')
# → 'def f():\n    """Docstring."""\n    return 1'

# 4. Syntax error — original returned, pipeline keeps running
strip_comments("def f(\n    # missing closing paren")
# → "def f(\n    # missing closing paren"
```

### Connection to node3.md

`judgement_node` now uses `state["stripped_code"]`
instead of `state["generated_code"]`:

```python
# nodes/judgement.py — relevant line
llm_scores = _get_llm_judgement(
    code=state["stripped_code"],    # ← not generated_code
    ...
)
```

`generated_code` is always preserved in its original form:
- `memory/logger.py` → the `generated_code` field in JSON
- `nodes/sandbox.py` → the code used during test execution
- `nodes/static_analysis.py` → the code used for metric computation

---

## Sharding (`main.py`)

`main.py` can split the sampled problem list across processes:

```python
shard_count = int(os.getenv("SHARD_COUNT", "1"))
shard_index = int(os.getenv("SHARD_INDEX", "0"))
rows = rows[shard_index::shard_count]
```

Each shard is an independent process writing to the same `outputs/` tree under its own `run_id`.
On the local M4 setup sharding gives no speed-up (the single llama.cpp slot is the bottleneck), so the
official runs use `SHARD_COUNT=1`.

### Resuming a run (`SKIP_QIDS`)

`SKIP_QIDS` (comma-separated question ids) removes already finished problems from the seeded
sample while keeping its order. It was used on 2026-10-01 when the PASS gate changed from
`accuracy > 0` to "all public tests pass": the 46 problems of `run_20260930_222527` whose
trajectory never hit a partial-pass PASS were kept, and the 5 affected problems plus the remaining
ones were run as `run_20261001_070529`. `benchmark/paper_results.py` lets a later run supersede an
earlier one for the same problem.

### PASS decision and `MAX_PROBLEMS` (2026-10-01)

- `iteration_gate` routes on `state["node3_pass"]` only. The score threshold is applied (if at all)
  inside `judgement_node` (`docs/node3.md` §4.2); re-checking it in the gate would override a judge
  `ACCEPT`. The threshold is still passed to the trend extension.
- New state fields: `node3_verdict`, `node3_verdict_reason`, `node3_verdict_source`.
- Filters in `main.py` apply in the order `MAX_PROBLEMS` → `SKIP_QIDS` → `SHARD_*` (shards of a capped run partition exactly that set).
- `MAX_PROBLEMS=N` keeps the first N problems of the seeded sample (the 50-problem runs use the
  same first 50 as the 150-problem run; list in `benchmark/sample50_qids.txt`).
