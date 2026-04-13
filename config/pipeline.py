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
    # "cloud"  → Vast.ai A100 80GB production environment (selected models)
    #
    # Usage:
    #   ENV=local python main.py   ← Mac Mini
    #   ENV=cloud python main.py   ← Vast.ai (default)

    # ── Iteration ────────────────────────────────────────────────────────
    max_iterations: int = int(os.getenv("MAX_ITERATIONS", 10))

    # ── Sandbox ──────────────────────────────────────────────────────────
    sandbox_timeout_sec:      int = int(os.getenv("SANDBOX_TIMEOUT_SEC",      10))
    sandbox_memory_limit_mb:  int = int(os.getenv("SANDBOX_MEMORY_LIMIT_MB", 512))
    sandbox_flakiness_runs:   int = int(os.getenv("SANDBOX_FLAKINESS_RUNS",    5))
    sandbox_max_failed_cases: int = int(os.getenv("SANDBOX_MAX_FAILED_CASES",  5))

    # ── LiveCodeBench ─────────────────────────────────────────────────────
    lcb_split:       str = os.getenv("LCB_SPLIT",       "test")
    lcb_difficulty:  str = os.getenv("LCB_DIFFICULTY",  "all")
    lcb_sample_size: int = int(os.getenv("LCB_SAMPLE_SIZE", 50))

    # ── Output ─────────────────────────────────────────────────────────────
    output_dir:        str = os.getenv("OUTPUT_DIR",        "outputs/results")
    memory_log_path:   str = os.getenv("MEMORY_LOG_PATH",   "outputs/memory.md")
    pipeline_log_path: str = os.getenv("PIPELINE_LOG_PATH", "outputs/pipeline.log")

    # ── Model sequence ────────────────────────────────────────────────────
    # "M1"         → single model (local debug)
    # "M1,M2,M3"  → full sequence (cloud benchmark, default)
    pipeline_models: str = os.getenv("PIPELINE_MODELS", "M1,M2,M3")

    # ── Judge model ───────────────────────────────────────────────────────
    # Registry key of the judgement model. Must match a key in ModelRegistry.
    # Default: "QWEN" — change to any registered model (e.g. "M3") to swap.
    judge_model: str = os.getenv("JUDGE_MODEL", "QWEN")

    # ── Trend-aware extension ─────────────────────────────────────────────
    # When enabled, iteration_gate allows up to max_extension_iters additional
    # iterations beyond max_iterations if the score shows a genuine upward trend.
    # Disabled by default — enable for V5 ablation variant.
    #
    # Activation requires ALL of:
    #   1. accuracy > 0 (at least one test passing)
    #   2. pass_rate_slope > 0 (pass rate increasing in last 3 iters)
    #   3. EMA-smoothed score slope >= trend_min_slope (points/iter)
    #   4. expected_gain >= gap to threshold (slope * remaining extensions)
    #   5. extension_count < max_extension_iters
    trend_extension_enabled: bool  = os.getenv("TREND_EXTENSION_ENABLED", "false").lower() == "true"
    max_extension_iters:     int   = int(os.getenv("MAX_EXTENSION_ITERS",  "3"))
    trend_min_slope:         float = float(os.getenv("TREND_MIN_SLOPE",    "1.0"))
    trend_ema_alpha:         float = float(os.getenv("TREND_EMA_ALPHA",    "0.5"))

    @property
    def model_sequence(self) -> list:
        return [m.strip() for m in self.pipeline_models.split(",") if m.strip()]

    def __post_init__(self):
        if self.env not in ("local", "cloud"):
            raise ValueError(
                f"Invalid ENV value: '{self.env}'. "
                f"Must be 'local' or 'cloud'."
            )


settings = PipelineSettings()
