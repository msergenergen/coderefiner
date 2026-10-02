# config.md — Modular Configuration

## 1. Design Decisions

| Decision | Rationale |
|-------|---------|
| `config/` package is the single entry point | `from config import settings, thresholds, model_registry` |
| `.env` is read only inside `config/` | `os.getenv` is not called anywhere else |
| Typed `frozen=True` dataclasses | IDE autocomplete + prevents accidental value mutation |
| **`ENV=local|cloud|openrouter` flag** | Model set is selected automatically per environment — no model config changes needed |
| **`build_registry(env)` factory** | `local` → one llama.cpp server, `cloud` → llama.cpp servers per role, `openrouter` → hosted APIs (main experiment) |
| Threshold values centralized in `thresholds.py` | Node 1/2/3 and judgement prompt all read from the same source |

### Environment Selection

`local` and `cloud` are selected **once** via the `.env` symlink; OpenRouter runs do not touch it:

```bash
ln -sf .env.local .env            # Mac mini M4 — ELECO paper setting
ln -sf .env.cloud .env            # rented GPU (optional, CLOUD_README.md §B)
python main.py

scripts/run_openrouter.sh smoke   # main experiment: sources config/openrouter.env,
                                  # reads only the key from .env.openrouter
```

Values exported in the shell take precedence over `.env` (`load_dotenv` does not override),
which is how the run script layers `config/openrouter.env` over a `.env` that points to `.env.local`.

| ENV | Generator (M1) | Judge (QWEN key) | Memory / cost |
|-----|----|----|----|
| `openrouter` | DeepSeek-V3 (DeepInfra fp4) | Qwen2.5-72B-Instruct (Novita bf16) | API, ≤ $5 cap |
| `local` | qwen3.5-9b | qwen3.5-9b (same server) | ~6 GB |
| `cloud` | Qwen3.5-9B | Qwen3.5-35B-A3B, or `JUDGE_MODEL=M1` | ~6–30 GB |

`M2` (Qwen2.5-Coder-1.5B in the cloud registry; same model as M1 elsewhere) runs only when
`PIPELINE_MODELS=M1,M2`; no reported experiment uses it.

---

## 2. config/\_\_init\_\_.py

```python
"""
Single entry point for configuration.

Usage:
    from config import settings, thresholds, model_registry
"""
from config.pipeline   import settings
from config.thresholds import thresholds
from config.models     import build_registry

# model_registry reads the ENV flag from settings.
# No other module reads ENV directly.
model_registry = build_registry(env=settings.env)

__all__ = ["settings", "thresholds", "model_registry"]
```

---

## 3. config/pipeline.py

```python
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
```

---

### Judge decision and token budgets

| Variable | Default | Meaning |
|---|---|---|
| `JUDGE_DECISION` | `verdict` | `verdict`: judge's ACCEPT/REJECT is final (public tests are a precondition). `rubric`: rubric total ≥ difficulty threshold (earlier design, ablation baseline). See `docs/node3.md` §4.2 |
| `CODEGEN_MAX_TOKENS` | `6144` | Output cap for code generation. The Qwen3.5-9B model card recommends 81,920 for competition programming (32,768 for most queries); 6144 is a local-hardware compromise (~16 tok/s on the M4) |
| `JUDGE_MAX_TOKENS` | `4096` | Output cap for the judge call |

| `LCB_EVALUATOR` | `local` | `local`: project's LCB-compatible evaluator (all reported results). `lcb_runner`: official runner if installed |

Run-shaping variables read in `main.py`: `MAX_PROBLEMS` (first N of the seeded sample),
`SKIP_QIDS` (resume), `SHARD_COUNT` / `SHARD_INDEX` (parallel shards) — applied in this order,
so shards of a capped/resumed run never overlap. Cloud server sizing: `PARALLEL` and `CTX_PER_SLOT`
for `scripts/start_model.sh M1` (context per slot must cover prompt + `CODEGEN_MAX_TOKENS`).
Step-by-step cloud procedure: `CLOUD_README.md`.

### Hosted-API backend (`ENV=openrouter`)

Runs the same pipeline against OpenRouter instead of llama.cpp servers. Experiment settings
live in `config/openrouter.env` (tracked, no secrets); `scripts/run_openrouter.sh` sources it and
reads only `OPENROUTER_API_KEY` from `.env.openrouter` (git-ignored; template
`.env.openrouter.example`). The `.env` symlink is untouched.

| Variable | Default | Meaning |
|---|---|---|
| `OPENROUTER_API_KEY` | — | Required. Only in `.env.openrouter`; never committed |
| `OPENROUTER_CODEGEN_MODEL` / `_PROVIDER` | `deepseek/deepseek-chat` / `deepinfra/fp4` | Generator and the one endpoint it is pinned to |
| `OPENROUTER_JUDGE_MODEL` / `_PROVIDER` | `qwen/qwen-2.5-72b-instruct` / `novita/bf16` | Judge (different model family than the generator) |
| `JUDGE_REASONING_EFFORT` | `medium` | Only for the secondary gpt-oss-120b judge |
| `LCB_DATE_FROM` / `LCB_DATE_TO` | empty | Contest-date window `YYYY-MM` (inclusive), applied before the seeded sample; main experiment `2025-01`..`2025-04` (150 of 182 problems) |
| `API_BUDGET_USD` | `5.0` | Hard cap on cumulative spend over all runs in the ledger |
| `API_LEDGER_PATH` | `outputs/api_usage.jsonl` | One line per call: model, serving provider, tokens, cost, question_id, iteration |
| `API_TIMEOUT_SEC` | `300` | Read timeout between streamed chunks |
| `API_MAX_RETRIES` | `5` (`openrouter.env`: 10) | Retries for 429/5xx, dropped connections and in-stream provider errors. 10 lets a call wait out ~13 min of upstream overload (DeepInfra returned `engine_overloaded` 429s on 2026-10-02) without moving to another endpoint |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | — | Optional Langfuse tracing; only in `.env.openrouter` (git-ignored). Unset → tracing off |
| `LANGFUSE_BASE_URL` | `https://cloud.langfuse.com` | Langfuse host (self-hosted: own URL) |
| `LANGFUSE_ENABLED` | `true` | `false` forces tracing off even with keys |
| `FEEDBACK_INCLUDE_CODE` | `true` | Repair iterations show the previous solution with the judge's feedback (LCB self-repair format); `false` = feedback only (`docs/codegen-models.md`) |
| `HIDDEN_TIMEOUT_SEC` | `10` | Per-test limit for offline grading on all tests (V0 and pipeline outputs alike); grading stops at the first failure as lcb_runner does |

Provider routing per call: `order=[<provider tag>]`, `allow_fallbacks=false`, and a `max_price`
ceiling equal to the pinned endpoint's price, so a run never silently moves to another
quantization or a pricier endpoint. Sampling and prices per (model, provider) pair are in
`_OPENROUTER_MODELS` (`config/models.py`); an unlisted pair is rejected. Qwen2.5-72B uses its
`generation_config.json` (temperature 0.7, top_p 0.8, top_k 20, repetition_penalty 1.05).
DeepSeek-V3 publishes no generation_config.json; its model card's demo command uses temperature
0.7, so that is set and top_p/top_k are left to the provider (`top_p=None` → not sent). Endpoints:
DeepSeek-V3 on DeepInfra fp4 (163,840 context, 16,384 max output), Qwen2.5-72B on Novita bf16
(32,000 context, 8,192 max output); both disclosed in the paper. Both models are open-weight:
DeepSeek-V3 (code MIT, DeepSeek Model License, commercial use allowed), Qwen2.5-72B (Qwen License).
The first pilot pair (Qwen2.5-Coder-32B on Cloudflare + Llama-3.3-70B) stays in the table for
reproducibility only; see `docs/ablation.md` §7.

## 4. config/thresholds.py

```python
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
```

---

## 5. config/models.py

```python
"""
Model definitions and build_registry factory.

ENV=cloud (default, Vast.ai):
    M1   → Qwen3.5-9B                 (~6 GB)    port 8001  ← medium-scale anchor
    M2   → Qwen2.5-Coder-1.5B-Instruct (~1 GB)    port 8002  ← small-scale anchor
    QWEN → Qwen3.5-35B-A3B            (~24 GB)   port 8004

    M1/M2 are deliberately same-family (Qwen) models at two model-size
    tiers, not three cross-vendor models. This isolates model *size* as
    the controlled variable for the pipeline's core claim — that
    structured, multi-stage feedback narrows the gap between a
    medium-scale and a small-scale model — instead of conflating size
    differences with vendor/architecture differences. See
    docs/future-work.md T1 and main.tex §IV-A for the rationale.

ENV=local (Mac Mini M4 16GB):
    M1   → qwen3.5-9b (UD-Q4_K_XL, ~6.0 GB)   port 8001
    M2   → qwen3.5-9b (UD-Q4_K_XL, ~6.0 GB)   port 8001  ← same server
    QWEN → qwen3.5-9b (UD-Q4_K_XL, ~6.0 GB)   port 8001  ← same server

ENV=openrouter (hosted APIs, paid, budget-capped — see models/budget.py):
    M1   → deepseek/deepseek-chat = DeepSeek-V3 (DeepInfra fp4; released 2024-12-26)
    M2   → same as M1 (run with PIPELINE_MODELS=M1)
    QWEN → qwen/qwen-2.5-72b-instruct (Novita bf16; released 2024-09-19)
    Both predate the LCB 2025-01..04 problems they are evaluated on, and the
    judge is a different model family than the generator (self-preference
    bias). Qwen3.5-9B + gpt-oss-120b remain selectable via env.

In local mode all roles route to a single server (8001).
Pipeline logic (ordering, gates, iterations) runs identically.

Sequential execution note:
    M1 and M2 are NEVER in memory simultaneously. Each model runs
    and finishes before the next starts. Memory = single largest codegen
    model + the judge model (judge is called every iteration regardless
    of which codegen model is active, so it stays resident concurrently).
"""
import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class ModelConfig:
    model_id:    str    # "M1" | "M2" | "QWEN"
    base_url:    str    # llama.cpp server URL
    model_name:  str    # Model name sent to llama.cpp
    temperature: float
    port:        int    # Reference for start_model.sh
    role:        str    # "codegen" | "judgement"
    # Extra request-body fields (provider pinning, sampling params that the
    # OpenAI schema lacks, reasoning effort). Empty for llama.cpp servers.
    extra_body:  dict  = field(default_factory=dict)
    # USD per 1M tokens; used for the pre-call worst-case budget reserve
    # and as a fallback when the API does not report the call's cost.
    price_in:    float = 0.0
    price_out:   float = 0.0
    top_p:       float | None = None


@dataclass(frozen=True)
class ModelRegistry:
    M1:   ModelConfig
    M2:   ModelConfig
    QWEN: ModelConfig

    def get(self, model_id: str) -> ModelConfig:
        try:
            return getattr(self, model_id)
        except AttributeError:
            raise ValueError(f"Invalid model_id: {model_id!r}. Must be M1/M2/QWEN.")

    def codegen_models(self, sequence: list[str]) -> list:
        """Returns ModelConfig list for the given model keys in order.

        Args:
            sequence: list of model keys, e.g. settings.model_sequence
        """
        return [self.get(k) for k in sequence]

    def judgement_model(self, key: str) -> ModelConfig:
        """Returns the ModelConfig for the given judgement model key.

        Args:
            key: registry key, e.g. settings.judge_model
        """
        return self.get(key)

    paid: bool = False   # True → calls go through the budget ledger

    def api_key(self) -> str:
        if self.paid:
            key = os.getenv("OPENROUTER_API_KEY", "")
            if not key:
                raise RuntimeError(
                    "OPENROUTER_API_KEY is not set. Put it in .env.openrouter "
                    "(git-ignored) and start runs with scripts/run_openrouter.sh"
                )
            return key
        return os.getenv("LLAMA_API_KEY", "local")

    def is_local(self) -> bool:
        """True when all roles route to the same port (local mode)."""
        ports = {self.M1.port, self.M2.port, self.QWEN.port}
        return len(ports) == 1


def _cloud_registry() -> ModelRegistry:
    """
    Vast.ai production registry.
    Each model runs on its own port and its own server.

    Two same-family Qwen codegen models (medium + small) isolate model
    size as the controlled variable, rather than three cross-vendor
    models whose differences confound size with architecture/training data.
    """
    return ModelRegistry(
        M1=ModelConfig(
            model_id="M1",
            base_url=os.getenv("M1_BASE_URL", "http://localhost:8001/v1"),
            model_name="qwen3.5-9b",
            temperature=0.6,   # Qwen3 family recommendation for thinking mode
            port=8001,
            role="codegen",
        ),
        M2=ModelConfig(
            model_id="M2",
            base_url=os.getenv("M2_BASE_URL", "http://localhost:8002/v1"),
            model_name="qwen2.5-coder-1.5b-instruct",
            temperature=0.7,   # Qwen2.5-Coder recommendation
            port=8002,
            role="codegen",
        ),
        QWEN=ModelConfig(
            model_id="QWEN",
            base_url=os.getenv("QWEN_BASE_URL", "http://localhost:8004/v1"),
            model_name="qwen3.5-35b-a3b",
            temperature=0.3,   # Low temperature for judgement consistency
            port=8004,
            role="judgement",
        ),
    )


def _local_registry() -> ModelRegistry:
    """
    Mac Mini M4 16GB local debug registry.

    All codegen roles (M1/M2) route to the same model:
        qwen3.5-9b (Unsloth UD-Q4_K_XL GGUF)  ~6.0 GB  — fits comfortably in 16 GB

    Judgement also uses the same model on the same port.
    Single llama.cpp server (port 8001) handles all roles.
    Pipeline logic (ordering, gates, iterations) is identical to cloud.
    qwen3.5-9b is a reasoning model: it emits a separate reasoning_content
    field before content (see nodes/codegen.py's fallback) and needs
    --jinja + --cache-ram 0 on the server (the default 8 GiB prompt cache
    caused Metal GPU OOM on this hardware when combined with the model's
    resident weights). Previously ran gpt-oss-20b (~11.9 GB) here, but its
    memory footprint proved too close to this machine's 16 GB ceiling —
    see docs/setup.md.
    """
    local_url = os.getenv("LOCAL_BASE_URL", "http://localhost:8001/v1")

    codegen_slot = ModelConfig(
        model_id="LOCAL_CODEGEN",
        base_url=local_url,
        model_name=os.getenv("LOCAL_CODEGEN_MODEL", "qwen3.5-9b"),
        temperature=0.6,   # Qwen3 family recommendation for thinking mode
        port=8001,
        role="codegen",
    )

    judgement_slot = ModelConfig(
        model_id="LOCAL_JUDGEMENT",
        base_url=local_url,
        model_name=os.getenv("LOCAL_JUDGEMENT_MODEL", "qwen3.5-9b"),
        temperature=0.6,   # Qwen3 family recommendation for thinking mode
        port=8001,
        role="judgement",
    )

    return ModelRegistry(
        M1=codegen_slot,
        M2=codegen_slot,
        QWEN=judgement_slot,
    )


def _pin(provider_tag: str, max_in: float, max_out: float) -> dict:
    """
    OpenRouter provider routing: one exact endpoint, no silent fallback to
    another provider/quantization, and a price ceiling as a second guard.
    """
    return {
        "order":           [provider_tag],
        "allow_fallbacks": False,
        "max_price":       {"prompt": max_in, "completion": max_out},
    }


# Per-model request settings for ENV=openrouter, from each model's card /
# generation_config.json. Prices are the pinned endpoint's USD per 1M tokens
# (also used as the max_price ceiling). Keyed by (model, provider tag).
_OPENROUTER_MODELS = {
    # Main, contamination-free experiment (both released before 2025-01;
    # problems restricted to LCB 2025-01..2025-04 — see docs/dataset.md §10).
    # Generator: DeepSeek-V3 (2024-12-26). No official sampling recommendation;
    # T 0.7 as in the model card's demo command (provider defaults otherwise).
    ("deepseek/deepseek-chat", "deepinfra/fp4"): dict(
        temperature=0.7, top_p=None, price=(0.32, 0.89), extra={},
    ),
    # Judge: Qwen2.5-72B-Instruct (2024-09-19), generation_config.json values.
    ("qwen/qwen-2.5-72b-instruct", "novita/bf16"): dict(
        temperature=0.7, top_p=0.8, price=(0.38, 0.40),
        extra={"top_k": 20, "repetition_penalty": 1.05},
    ),
    # Pilot only (2026-10-02): Cloudflare's Qwen2.5-Coder-32B endpoint dropped
    # "[]" tokens (≈9% of samples had `x =` with nothing after it).
    ("qwen/qwen-2.5-coder-32b-instruct", "cloudflare"): dict(
        temperature=0.7, top_p=0.8, price=(0.66, 1.00),
        extra={"top_k": 20, "repetition_penalty": 1.05},
    ),
    ("meta-llama/llama-3.3-70b-instruct", "novita/bf16"): dict(
        temperature=0.6, top_p=0.9, price=(0.135, 0.40), extra={},
    ),
    # Secondary "modern reasoning model" setting.
    ("qwen/qwen3.5-9b", "deepinfra/bf16"): dict(
        temperature=0.6, top_p=0.95, price=(0.10, 0.15),
        extra={"top_k": 20, "min_p": 0.0},
    ),
    ("openai/gpt-oss-120b", "deepinfra/bf16"): dict(
        temperature=1.0, top_p=1.0, price=(0.037, 0.17),
        extra={"reasoning": {"effort": os.getenv("JUDGE_REASONING_EFFORT", "medium").lower()}},
    ),
}


def _openrouter_config(model_id: str, role: str, model: str, provider: str, port: int) -> ModelConfig:
    try:
        spec = _OPENROUTER_MODELS[(model, provider)]
    except KeyError:
        raise ValueError(
            f"No OpenRouter settings for {model!r} on {provider!r}; "
            f"add sampling and price to _OPENROUTER_MODELS in config/models.py"
        )
    price_in, price_out = spec["price"]
    return ModelConfig(
        model_id=model_id,
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        model_name=model,
        temperature=spec["temperature"],
        top_p=spec["top_p"],
        port=port,
        role=role,
        extra_body={
            "provider": _pin(provider, price_in, price_out),
            **spec["extra"],
            "usage":    {"include": True},
        },
        price_in=price_in,
        price_out=price_out,
    )


def _openrouter_registry() -> ModelRegistry:
    """
    Hosted-API registry. Generator and judge are different model families;
    each is pinned to one provider endpoint so every call in a run hits the
    same weights and precision (logged per call in the API ledger).
    Defaults are the contamination-free main experiment.
    """
    codegen = _openrouter_config(
        "M1", "codegen",
        os.getenv("OPENROUTER_CODEGEN_MODEL", "deepseek/deepseek-chat"),
        os.getenv("OPENROUTER_CODEGEN_PROVIDER", "deepinfra/fp4"), port=0,
    )
    judge = _openrouter_config(
        "QWEN", "judgement",
        os.getenv("OPENROUTER_JUDGE_MODEL", "qwen/qwen-2.5-72b-instruct"),
        os.getenv("OPENROUTER_JUDGE_PROVIDER", "novita/bf16"), port=1,
    )
    return ModelRegistry(M1=codegen, M2=codegen, QWEN=judge, paid=True)


def build_registry(env: str = "cloud") -> ModelRegistry:
    """
    Returns the correct registry based on the ENV flag.

    Args:
        env: "cloud" (default) | "local" | "openrouter"

    Returns:
        ModelRegistry — other modules use this registry and don't need
        to know which environment they're running in.
    """
    if env == "local":
        return _local_registry()
    if env == "openrouter":
        return _openrouter_registry()
    return _cloud_registry()
```

`ModelConfig` also carries `extra_body` (provider pinning and non-OpenAI sampling fields),
`top_p`, and `price_in` / `price_out` (USD per 1M tokens, for the budget reserve and as a
fallback when the API reports no cost). `ModelRegistry.paid` is `True` only for
`_openrouter_registry()`; it switches `models/client.chat()` to the streamed, retried,
ledger-recorded path (see `docs/codegen-models.md`).

---

## 6. .env Files

### config/openrouter.env (main experiment; tracked, no secrets)

The key is **not** here: `scripts/run_openrouter.sh` reads only `OPENROUTER_API_KEY=` from the
git-ignored `.env.openrouter` (template `.env.openrouter.example`).

```bash
# ── OpenRouter experiment settings (tracked; NO secrets here) ──────
# The API key is read only from .env.openrouter (git-ignored) by
# scripts/run_openrouter.sh. See docs/ablation.md §7.
ENV=openrouter

# Main, contamination-free experiment: both models released before 2025-01,
# problems from LCB v6 2025-01..2025-04 (docs/dataset.md §10).
# Generator DeepSeek-V3 (2024-12-26), judge Qwen2.5-72B-Instruct (2024-09-19).
# The pilot pair (Qwen2.5-Coder-32B on Cloudflare + Llama-3.3-70B) was dropped:
# floor effect and a Cloudflare endpoint that lost "[]" tokens.
OPENROUTER_CODEGEN_MODEL=deepseek/deepseek-chat
OPENROUTER_CODEGEN_PROVIDER=deepinfra/fp4
OPENROUTER_JUDGE_MODEL=qwen/qwen-2.5-72b-instruct
OPENROUTER_JUDGE_PROVIDER=novita/bf16

# ── Spending guard ─────────────────────────────────────────────────
API_BUDGET_USD=5.0
API_LEDGER_PATH=outputs/api_usage.jsonl
API_TIMEOUT_SEC=300
API_MAX_RETRIES=10

# ── Token budgets (non-reasoning models; endpoint limits: generator
#    163,840 context / 16,384 output, judge 32,000 / 8,192) ──────────
CODEGEN_MAX_TOKENS=8192
JUDGE_MAX_TOKENS=4096

# ── Problems ───────────────────────────────────────────────────────
LCB_SPLIT=test
LCB_DIFFICULTY=all
LCB_DATE_FROM=2025-01
LCB_DATE_TO=2025-04
LCB_SAMPLE_SIZE=150
LCB_EVALUATOR=local

# ── Pipeline: at most 5 iterations, no trend extension ─────────────
PIPELINE_MODELS=M1
JUDGE_MODEL=QWEN
JUDGE_DECISION=verdict
MAX_ITERATIONS=5
TREND_EXTENSION_ENABLED=false
SANDBOX_TIMEOUT_SEC=15
SANDBOX_MEMORY_LIMIT_MB=256
SANDBOX_FLAKINESS_RUNS=2
SANDBOX_MAX_FAILED_CASES=3

# ── Output (kept apart from the local runs) ────────────────────────
OUTPUT_DIR=outputs/openrouter/results
MEMORY_LOG_PATH=outputs/openrouter/memory.md
PIPELINE_LOG_PATH=outputs/openrouter/pipeline.log

# Repair iterations show the generator its previous solution with the judge
# feedback (LCB self-repair format). false = feedback only (runs before 10:10).
FEEDBACK_INCLUDE_CODE=true
```

### .env.local (Mac mini M4 — ELECO paper runs)

```bash
ENV=local
LOCAL_BASE_URL=http://localhost:8001/v1
LOCAL_CODEGEN_MODEL=qwen3.5-9b      # Unsloth UD-Q4_K_XL ~6 GB
LOCAL_JUDGEMENT_MODEL=qwen3.5-9b    # same model, same server
LLAMA_API_KEY=local

PIPELINE_MODELS=M1
JUDGE_MODEL=QWEN                    # → the same local server
MAX_ITERATIONS=5
TREND_EXTENSION_ENABLED=true        # paper runs: 5 + up to 3 (new runs: false)
MAX_EXTENSION_ITERS=3
TREND_PROJECTION_WINDOW=3
TREND_MIN_SLOPE=1.0
TREND_EMA_ALPHA=0.5
SANDBOX_TIMEOUT_SEC=15
SANDBOX_MEMORY_LIMIT_MB=256
SANDBOX_FLAKINESS_RUNS=2
SANDBOX_MAX_FAILED_CASES=3

LCB_SPLIT=test
LCB_DIFFICULTY=all
LCB_SAMPLE_SIZE=150                 # seed 42; the paper uses the first 50

OUTPUT_DIR=outputs/results
MEMORY_LOG_PATH=outputs/memory.md
PIPELINE_LOG_PATH=outputs/pipeline.log
```

Defaults not set here: `JUDGE_DECISION=verdict`, `CODEGEN_MAX_TOKENS=6144`,
`JUDGE_MAX_TOKENS=4096`, `LCB_EVALUATOR=local`.

### .env.cloud (rented GPU, optional)

Same experiment settings as `.env.local`, with `ENV=cloud`, `M1_BASE_URL=http://localhost:8001/v1`,
`JUDGE_MODEL=M1` and a larger budget (`CODEGEN_MAX_TOKENS=32768`, `JUDGE_MAX_TOKENS=16384`).
Full file: `CLOUD_README.md` §B.

### Usage

```bash
# Symlink setup — done once per environment, not repeated each run

# Mac Mini (local):
ln -sf .env.local .env

# Rented GPU (cloud, optional):
ln -sf .env.cloud .env

# OpenRouter (main experiment) does not use the symlink:
#   scripts/run_openrouter.sh smoke

# After symlink is set, just run:
python main.py

# Verify active environment:
python -c "from config import settings, model_registry; \
  print('ENV:', settings.env); \
  print('M1:', model_registry.M1.model_name); \
  print('QWEN:', model_registry.QWEN.model_name)"
```

---

## 7. Installation

Installation steps live in one place to avoid drift: `docs/setup.md` (Mac mini, llama.cpp,
model download), `CLOUD_README.md` §A (OpenRouter) and §B (rented GPU). Server flags are set by
`scripts/start_model.sh` (Qwen3.5-9B: `--jinja --cache-ram 0`; cloud: `PARALLEL`, `CTX_PER_SLOT`).

---

## 9. Updated Modules (Summary)

There is **not a single line** related to ENV or model selection in any other module —
`config/__init__.py` handles this when building the registry.

```python
# models/client.py — nodes call chat(model_id, messages, max_tokens, tag);
# get_client() builds the OpenAI client from the registry entry
# (llama.cpp URL, or OpenRouter with the key from OPENROUTER_API_KEY).

# pipeline/router.py
MODEL_SEQUENCE = [m.model_id for m in model_registry.codegen_models(settings.model_sequence)]
# → ["M1"] (default PIPELINE_MODELS)

# nodes/judgement.py — unchanged
from config import thresholds
JUDGEMENT_SYSTEM_PROMPT = f"""...\n{thresholds.judgement_context_table()}\n..."""
```

---

## 10. Verification

### Cloud environment

```bash
python -c "
from config import settings, model_registry
print('ENV:        ', settings.env)            # cloud
print('M1 model:   ', model_registry.M1.model_name)   # qwen3.5-9b
print('M1 port:    ', model_registry.M1.port)          # 8001
print('M2 model:   ', model_registry.M2.model_name)   # qwen2.5-coder-1.5b-instruct
print('M2 port:    ', model_registry.M2.port)          # 8002
print('QWEN model: ', model_registry.QWEN.model_name)  # qwen3.5-35b-a3b
print('QWEN port:  ', model_registry.QWEN.port)        # 8004
print('Is local:   ', model_registry.is_local())       # False
"
```

### Local environment (Mac Mini)

```bash
ENV=local python -c "
from config import settings, model_registry
print('ENV:        ', settings.env)            # local
print('M1 model:   ', model_registry.M1.model_name)   # qwen3.5-9b
print('M1 port:    ', model_registry.M1.port)          # 8001
print('M2 model:   ', model_registry.M2.model_name)   # qwen3.5-9b
print('M2 port:    ', model_registry.M2.port)          # 8001 (same)
print('QWEN model: ', model_registry.QWEN.model_name)  # qwen3.5-9b
print('QWEN port:  ', model_registry.QWEN.port)        # 8001 (same)
print('Is local:   ', model_registry.is_local())       # True
"
```

### OpenRouter environment

```bash
set -a; source config/openrouter.env; set +a
python -c "
from config import model_registry as r
print(r.paid, r.M1.model_name, r.M1.extra_body['provider'])   # True deepseek/deepseek-chat {...}
print(r.QWEN.model_name)                                       # qwen/qwen-2.5-72b-instruct
"
```

### Invalid ENV test

```bash
ENV=staging python -c "from config import settings" 2>&1
# ValueError: Invalid ENV value: 'staging'. Must be 'local', 'cloud' or 'openrouter'.
```
