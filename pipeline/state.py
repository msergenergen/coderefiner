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
