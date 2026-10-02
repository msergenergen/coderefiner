"""
Pipeline-wide runtime settings.
All os.getenv calls are here — no other module calls getenv directly.
"""
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class PipelineSettings:

    # ── Environment selection ──────────────────────────────────────────────────
    env: str = os.getenv("ENV", "cloud").lower()
    # "local"  → Mac Mini M4 16GB debug environment (small models)
    # "cloud"  → self-hosted llama.cpp servers on a rented GPU (Vast.ai)
    # "openrouter" → hosted APIs via OpenRouter (codegen and judge on
    #                different providers' models; paid, budget-capped)
    #
    # Usage:
    #   ENV=local python main.py   ← Mac Mini
    #   ENV=cloud python main.py   ← Vast.ai (default)

    # ── Iteration ────────────────────────────────────────────────────────
    # At most 5 generations per problem (main experiment protocol,
    # docs/ablation.md §7); trend extension below is off by default.
    max_iterations: int = int(os.getenv("MAX_ITERATIONS", 5))

    # ── Sandbox ──────────────────────────────────────────────────────────
    sandbox_timeout_sec:      int = int(os.getenv("SANDBOX_TIMEOUT_SEC",      10))
    sandbox_memory_limit_mb:  int = int(os.getenv("SANDBOX_MEMORY_LIMIT_MB", 512))
    sandbox_flakiness_runs:   int = int(os.getenv("SANDBOX_FLAKINESS_RUNS",    5))
    sandbox_max_failed_cases: int = int(os.getenv("SANDBOX_MAX_FAILED_CASES",  5))

    # ── LiveCodeBench ─────────────────────────────────────────────────────
    lcb_split:       str = os.getenv("LCB_SPLIT",       "test")
    lcb_difficulty:  str = os.getenv("LCB_DIFFICULTY",  "all")
    lcb_sample_size: int = int(os.getenv("LCB_SAMPLE_SIZE", 50))
    # Optional contest-date window "YYYY-MM" (inclusive), applied before the
    # seeded sample. Empty = whole release. Main experiment: 2025-01..2025-04.
    lcb_date_from:   str = os.getenv("LCB_DATE_FROM", "")
    lcb_date_to:     str = os.getenv("LCB_DATE_TO", "")
    # "local"      → LCB-compatible evaluator in sandbox/lcb_eval.py (all reported results)
    # "lcb_runner" → the official lcb_runner, if installed
    # Pinned so a cloud machine with lcb_runner installed grades exactly like the Mac.
    lcb_evaluator:   str = os.getenv("LCB_EVALUATOR", "local").lower()
    # Offline grading on all (hidden) tests: one per-test limit for V0 and the
    # pipeline alike (10 s = the value the ELECO runs were graded with).
    hidden_timeout_sec: int = int(os.getenv("HIDDEN_TIMEOUT_SEC", 10))
    # Repair iterations show the generator its previous solution next to the
    # judge's feedback (LCB self-repair format). false = feedback only, the
    # setting of every run before 2026-10-02 10:10.
    feedback_include_code: bool = os.getenv("FEEDBACK_INCLUDE_CODE", "true").lower() == "true"

    # ── Output ─────────────────────────────────────────────────────────────
    output_dir:        str = os.getenv("OUTPUT_DIR",        "outputs/results")
    memory_log_path:   str = os.getenv("MEMORY_LOG_PATH",   "outputs/memory.md")
    pipeline_log_path: str = os.getenv("PIPELINE_LOG_PATH", "outputs/pipeline.log")

    # ── Model sequence ────────────────────────────────────────────────────
    # "M1"     → single generator (default; every reported experiment)
    # "M1,M2"  → also run the small-scale model afterwards (cloud registry:
    #            Qwen2.5-Coder-1.5B; model-scale study, not run so far)
    pipeline_models: str = os.getenv("PIPELINE_MODELS", "M1")

    # ── Judge model ───────────────────────────────────────────────────────
    # Registry key of the judgement model. Must match a key in ModelRegistry.
    # Default: "QWEN" — change to any registered key (e.g. "M1" = self-judging) to swap.
    judge_model: str = os.getenv("JUDGE_MODEL", "QWEN")

    # ── Judge decision mechanism ──────────────────────────────────────────
    # "verdict" → the judge gives an explicit, justified ACCEPT/REJECT; the
    #             deterministic metrics and test results are evidence in its
    #             prompt, and passing every public test is a precondition.
    # "rubric"  → PASS when the 100-point rubric total clears the difficulty
    #             threshold (with the same public-test precondition).
    judge_decision: str = os.getenv("JUDGE_DECISION", "verdict").lower()

    # ── Token budgets (per call) ──────────────────────────────────────────
    # Local default 6144 is ~7.5% of the 81,920 output tokens the Qwen3.5
    # model card recommends for competition programming; raise on cloud GPUs.
    codegen_max_tokens: int = int(os.getenv("CODEGEN_MAX_TOKENS", 6144))
    judge_max_tokens:   int = int(os.getenv("JUDGE_MAX_TOKENS", 4096))

    # ── Paid API backend (ENV=openrouter) ─────────────────────────────────
    # Hard cap on cumulative spend, summed over every run in the ledger.
    api_budget_usd:   float = float(os.getenv("API_BUDGET_USD", "5.0"))
    api_ledger_path:  str   = os.getenv("API_LEDGER_PATH", "outputs/api_usage.jsonl")
    # Read timeout between streamed chunks, and retries for 429/5xx/drops.
    api_timeout_sec:  int   = int(os.getenv("API_TIMEOUT_SEC", "300"))
    api_max_retries:  int   = int(os.getenv("API_MAX_RETRIES", "5"))

    # ── Trend-aware extension ─────────────────────────────────────────────
    # When enabled, iteration_gate allows up to max_extension_iters additional
    # iterations beyond max_iterations if the score shows a genuine upward trend.
    # Disabled by default — enable to add trend-aware extension to any variant.
    #
    # Activation requires ALL of:
    #   1. accuracy > 0 (at least one test passing)
    #   2. pass_rate_slope > 0 (pass rate increasing in last 3 iters)
    #   3. EMA-smoothed score slope >= trend_min_slope (points/iter)
    #   4. expected_gain >= gap to threshold (slope * trend_projection_window)
    #   5. extension_count < max_extension_iters
    #
    # NOTE: trend_projection_window and max_extension_iters are intentionally
    # separate. The projection window is a fixed look-ahead for the gain estimate;
    # max_extension_iters is the hard safety cap.
    # Increasing max_extension_iters does NOT inflate the gain projection.
    trend_extension_enabled:  bool  = os.getenv("TREND_EXTENSION_ENABLED", "false").lower() == "true"
    max_extension_iters:      int   = int(os.getenv("MAX_EXTENSION_ITERS",       "3"))
    trend_projection_window:  int   = int(os.getenv("TREND_PROJECTION_WINDOW",   "3"))
    trend_min_slope:          float = float(os.getenv("TREND_MIN_SLOPE",         "1.0"))
    trend_ema_alpha:          float = float(os.getenv("TREND_EMA_ALPHA",         "0.5"))

    @property
    def model_sequence(self) -> list:
        return [m.strip() for m in self.pipeline_models.split(",") if m.strip()]

    def __post_init__(self):
        if self.env not in ("local", "cloud", "openrouter"):
            raise ValueError(
                f"Invalid ENV value: '{self.env}'. "
                f"Must be 'local', 'cloud' or 'openrouter'."
            )


settings = PipelineSettings()
