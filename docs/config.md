# config.md — Modular Configuration

## 1. Design Decisions

| Decision | Rationale |
|-------|---------|
| `config/` package is the single entry point | `from config import settings, thresholds, model_registry` |
| `.env` is read only inside `config/` | `os.getenv` is not called anywhere else |
| Typed `frozen=True` dataclasses | IDE autocomplete + prevents accidental value mutation |
| **`ENV=local|cloud` flag** | Model set is selected automatically per environment — no model config changes needed |z |
| **`build_registry(env)` factory** | `local` → Mac Mini models, `cloud` → production models |
| Threshold values centralized in `thresholds.py` | Node 1/2/3 and judgement prompt all read from the same source |

### Environment Selection: `.env` Symlink

The environment is selected **once** via a symlink — no flag needed at runtime:

```bash
# Mac Mini M4 16GB — local debug (done once)
ln -sf .env.local .env

# Vast.ai A100 80GB — cloud production (done once)
ln -sf .env.cloud .env

# Then just run:
python main.py
```

The active `.env` file controls the entire model set:

| ENV | M1 | M2 | Judgement | Toplam Bellek |
|-----|----|----|----|----------|
| `local` | qwen3.5-9b | qwen3.5-9b | qwen3.5-9b | ~6 GB |
| `cloud` | Qwen3.5-9B | Qwen2.5-Coder-1.5B-Instruct | Qwen3.5-35B-A3B | ~30 GB (concurrent) |

In local mode all roles (M1/M2 and judgement) route to the same model
on the same server; pipeline flow, gates, and iteration logic run identically.

### Sequential Model Architecture and Memory

```
START → M1 (çalışır, biter) → M2 (çalışır, biter) → END
         ↑ bellekte tek model aynı anda
```

Both M1 (Qwen3.5-9B, ~6 GB) and M2 (Qwen2.5-Coder-1.5B, ~1 GB) are same-family Qwen models at two size tiers, isolating model size as the controlled variable.
Concurrent memory: M1 + QWEN judgement ≈ 30 GB (cloud); qwen3.5-9b (~6 GB) single server (local).

---

## 2. config/\_\_init\_\_.py

```python
"""
Tek giriş noktası.

Kullanım:
    from config import settings, thresholds, model_registry
"""
from config.pipeline   import settings
from config.thresholds import thresholds
from config.models     import build_registry

# model_registry reads ENV flag from settings.
# No other module reads ENV directly.
model_registry = build_registry(env=settings.env)

__all__ = ["settings", "thresholds", "model_registry"]
```

---

## 3. config/pipeline.py

```python
"""
Pipeline geneli çalışma ayarları.
Tüm os.getenv çağrıları burada — başka modülde getenv yok.
"""
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class PipelineSettings:

    # ── Environment selection ──────────────────────────────────────────────────────
    env: str = os.getenv("ENV", "cloud").lower()
    # "local" → Mac Mini M4 16GB debug environment (small models)
    # "cloud" → Vast.ai A100 80GB production environment (selected models)
    #
    # Usage:
    #   ENV=local python main.py   ← Mac Mini
    #   ENV=cloud python main.py   ← Vast.ai (default)

    # ── Iterasyon ────────────────────────────────────────────────────────
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

    # ── Trend-aware extension ─────────────────────────────────────────
    # max_iterations'a ulaşıldığında stop_max yerine EMA trend analizi yapılır.
    # Tüm 5 aktivasyon koşulu sağlanırsa ek iterasyon verilir.
    # Varsayılan: false — V1–V3 ablasyon sonuçlarını etkilemez.
    # Aktive etmek için: TREND_EXTENSION_ENABLED=true
    #
    # Aktivasyon koşulları (hepsi sağlanmalı):
    #   1. accuracy > 0                           (en az bir test geçiyor)
    #   2. pass_rate_slope > 0                    (son 3 iter'de pass rate artıyor)
    #   3. EMA score_slope >= min                 (skor gerçekten iyileşiyor)
    #   4. slope * TREND_PROJECTION_WINDOW >= gap (sabit pencereyle eşiğe ulaşılabilir)
    #   5. extension_count < max      (bütçe tükenmemiş)
    trend_extension_enabled: bool  = os.getenv("TREND_EXTENSION_ENABLED", "false").lower() == "true"
    max_extension_iters:     int   = int(os.getenv("MAX_EXTENSION_ITERS",  "3"))
    trend_min_slope:         float = float(os.getenv("TREND_MIN_SLOPE",    "1.0"))
    trend_ema_alpha:         float = float(os.getenv("TREND_EMA_ALPHA",    "0.5"))

    def __post_init__(self):
        if self.env not in ("local", "cloud"):
            raise ValueError(
                f"Geçersiz ENV değeri: '{self.env}'. "
                f"'local' veya 'cloud' olmalı."
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

Run-shaping variables read in `main.py`: `MAX_PROBLEMS` (first N of the seeded sample),
`SKIP_QIDS` (resume), `SHARD_COUNT` / `SHARD_INDEX` (parallel shards).

## 4. config/thresholds.py

```python
"""
Tüm metrik eşikleri tek yerde.
Node 1, Node 2, Node 3 ve judgement prompt bu modülden okur.
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Node1Thresholds:
    cc_hard_limit:   float = 20.0   # CC > 20  → HIGH severity → STOP
    cc_warn:         float = 10.0   # CC > 10  → issue report
    mi_low_limit:    float = 40.0   # MI < 40  → LOW severity
    mi_warn:         float = 65.0   # MI < 65  → issue report
    halstead_warn:   float = 1000.0 # > 1000   → LOW severity
    cc_ideal:        float = 5.0
    mi_ideal:        float = 85.0
    halstead_ideal:  float = 500.0


@dataclass(frozen=True)
class Node2Thresholds:
    cpu_limit_ms:         float = 5000.0
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
            raise ValueError(f"Geçersiz difficulty: {difficulty!r}")
        return mapping[difficulty]

    def as_dict(self) -> dict[str, int]:
        return {"easy": self.easy, "medium": self.medium, "hard": self.hard}


@dataclass(frozen=True)
class AllThresholds:
    node1: Node1Thresholds = field(default_factory=Node1Thresholds)
    node2: Node2Thresholds = field(default_factory=Node2Thresholds)
    node3: Node3Thresholds = field(default_factory=Node3Thresholds)

    def judgement_context_table(self) -> str:
        """
        Qwen 3.5'e gönderilecek threshold tablosunu config'ten üretir.
        Hardcode markdown string yok — değer değişirse tablo otomatik güncellenir.
        """
        n1 = self.node1
        n2 = self.node2
        return f"""## Metric Threshold Reference

| Metric | Good | Acceptable | Poor |
|--------|------|------------|------|
| Cyclomatic Complexity (CC) | ≤ {n1.cc_ideal:.0f} | {n1.cc_ideal+1:.0f}–{n1.cc_warn:.0f} | > {n1.cc_warn:.0f} |
| Maintainability Index (MI) | ≥ {n1.mi_ideal:.0f} | {n1.mi_warn:.0f}–{n1.mi_ideal:.0f} | < {n1.mi_warn:.0f} |
| Halstead Volume | ≤ {n1.halstead_ideal:.0f} | {n1.halstead_ideal:.0f}–{n1.halstead_warn:.0f} | > {n1.halstead_warn:.0f} |
| Test Pass Rate | ≥ {n2.pass_rate_easy:.0%} | {n2.pass_rate_hard:.0%}–{n2.pass_rate_easy:.0%} | < {n2.pass_rate_hard:.0%} |
| Peak Memory | ≤ {n2.ram_limit_mb/2:.0f} MB | {n2.ram_limit_mb/2:.0f}–{n2.ram_limit_mb:.0f} MB | > {n2.ram_limit_mb:.0f} MB |
| CPU Time | ≤ {n2.cpu_limit_ms/2500:.0f} s | {n2.cpu_limit_ms/2500:.0f}–{n2.cpu_limit_ms/1000:.0f} s | > {n2.cpu_limit_ms/1000:.0f} s |

Bandit HIGH severity: zero tolerance.
MI < {n1.mi_warn:.0f} means the code is hard to maintain.
CC > {n1.cc_warn:.0f} means the function has too many branches."""


thresholds = AllThresholds()
```

---

## 5. config/models.py

```python
"""
Model tanımları ve build_registry factory.

ENV=cloud (varsayılan, Vast.ai):
    M1   → Qwen3.5-9B                 (~6 GB)    port 8001  ← medium-scale anchor
    M2   → Qwen2.5-Coder-1.5B-Instruct (~1 GB)    port 8002  ← small-scale anchor
    QWEN → Qwen3.5-35B-A3B            (~24 GB)   port 8004

    M1/M2 are deliberately same-family (Qwen) models at two model-size
    tiers, not three cross-vendor models. This isolates model *size* as
    the controlled variable for the pipeline's core claim — that
    structured, multi-stage feedback narrows the gap between a
    medium-scale and a small-scale model — instead of conflating size
    differences with vendor/architecture differences.

ENV=local (Mac Mini M4 16GB):
    M1   → qwen3.5-9b (UD-Q4_K_XL, ~6.0 GB)   port 8001
    M2   → qwen3.5-9b (UD-Q4_K_XL, ~6.0 GB)   port 8001  ← same server
    QWEN → qwen3.5-9b (UD-Q4_K_XL, ~6.0 GB)   port 8001  ← same server

In local mode all roles route to a single server (8001).
Pipeline logic (ordering, gates, iterations) runs identically.

Sequential execution note:
    M1 and M2 are NEVER in memory simultaneously. Each model runs
    and finishes before the next starts. Memory = single largest codegen
    model + the judge model (judge is called every iteration regardless
    of which codegen model is active, so it stays resident concurrently).
"""
import os
from dataclasses import dataclass
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
        """Returns ModelConfig list for the given model keys in order."""
        return [self.get(k) for k in sequence]

    def judgement_model(self, key: str) -> ModelConfig:
        return self.get(key)

    def api_key(self) -> str:
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


def build_registry(env: str = "cloud") -> ModelRegistry:
    """
    ENV flag'ine göre doğru registry'yi döner.

    Args:
        env: "cloud" (varsayılan) | "local"

    Returns:
        ModelRegistry — diğer modüller bu registry'yi kullanır,
        hangi ortamda olduklarını bilmek zorunda değiller.
    """
    if env == "local":
        return _local_registry()
    return _cloud_registry()
```

---

## 6. .env Files

### .env (cloud — Vast.ai, default)

```bash
# ── Ortam ──────────────────────────────────────────────────────────
ENV=cloud

# ── Model server URL'leri ──────────────────────────────────────────
M1_BASE_URL=http://localhost:8001/v1
M2_BASE_URL=http://localhost:8002/v1
M3_BASE_URL=http://localhost:8003/v1
QWEN_BASE_URL=http://localhost:8004/v1
LLAMA_API_KEY=local

# ── Pipeline ──────────────────────────────────────────────────────
MAX_ITERATIONS=10
SANDBOX_TIMEOUT_SEC=10
SANDBOX_MEMORY_LIMIT_MB=512
SANDBOX_FLAKINESS_RUNS=5
SANDBOX_MAX_FAILED_CASES=5

# ── LiveCodeBench ─────────────────────────────────────────────────
LCB_SPLIT=test
LCB_DIFFICULTY=all
LCB_SAMPLE_SIZE=50

# ── Output ─────────────────────────────────────────────────────────
OUTPUT_DIR=outputs/results
MEMORY_LOG_PATH=outputs/memory.md
PIPELINE_LOG_PATH=outputs/pipeline.log

# ── Trend-aware extension ─────────────────────────────────────────
# V1–V3 ablasyon çalıştırmalarında false bırak.
TREND_EXTENSION_ENABLED=false
MAX_EXTENSION_ITERS=3
TREND_MIN_SLOPE=1.0
TREND_EMA_ALPHA=0.5
```

### .env.local (Mac Mini M4 16GB)

```bash
# ── Ortam — LOCAL DEBUG ───────────────────────────────────────────
ENV=local

# ── Single server (all roles go here) ──────────────────────────
LOCAL_BASE_URL=http://localhost:8001/v1
LOCAL_CODEGEN_MODEL=qwen3.5-9b   # Unsloth UD-Q4_K_XL ~6.0 GB
LOCAL_JUDGEMENT_MODEL=qwen3.5-9b  # same model, same server
LLAMA_API_KEY=local

# ── Pipeline — keep short ───────────────────────────────────────────
MAX_ITERATIONS=3
SANDBOX_TIMEOUT_SEC=15              # Mac daha yavaş, marj ver
SANDBOX_MEMORY_LIMIT_MB=256

# ── LiveCodeBench — small sample ───────────────────────────────
LCB_SAMPLE_SIZE=5
LCB_DIFFICULTY=easy

# ── Output ─────────────────────────────────────────────────────────
OUTPUT_DIR=outputs/results
MEMORY_LOG_PATH=outputs/memory.md
PIPELINE_LOG_PATH=outputs/pipeline.log
```

### Usage

```bash
# Symlink setup — done once per environment, not repeated each run

# Mac Mini (local):
ln -sf .env.local .env

# Vast.ai (cloud):
ln -sf .env.cloud .env

# After symlink is set, just run:
python main.py

# Verify active environment:
python -c "from config import settings, model_registry; \
  print('ENV:', settings.env); \
  print('M1:', model_registry.M1.model_name); \
  print('QWEN:', model_registry.QWEN.model_name)"
```

---

## 7. Mac Mini M4 Local Kurulumu

### Model Download (one-time)

```bash
# All roles (codegen + judgement) — ~6.0 GB
hf download unsloth/Qwen3.5-9B-GGUF \
    --include "*UD-Q4_K_XL*" \
    --local-dir ./models/qwen3.5-9b
```

### llama.cpp Metal Derleme

```bash
cmake llama.cpp -B llama.cpp/build \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_CUDA=OFF      # Metal otomatik aktif

cmake --build llama.cpp/build \
    --config Release -j$(sysctl -n hw.logicalcpu)
```

### Start Single Server (codegen and judgement share the same server)

```bash
# Codegen and judgement both go to port 8001
./llama.cpp/llama-server \
    --model ./models/qwen3.5-9b/*.gguf \
    --port 8001 --host 0.0.0.0 \
    --ctx-size 8192 --n-gpu-layers 99 \
    --parallel 1 \
    --jinja \
    --cache-ram 0 \
    --temp 0.6 --top-p 0.95 --min-p 0.01 --top-k 40
```

> `--parallel 1` caps llama-server at a single inference slot. Without it, llama-server
> defaults to 3 parallel slots, each allocating its own KV cache — on a 16GB Mac Mini
> this was observed to trigger `kIOGPUCommandBufferCallbackErrorOutOfMemory` (Metal GPU
> OOM) on the very first request. The pipeline only ever issues one request at a time
> per model (sequential execution), so a single slot is sufficient.

> `--jinja` is required for qwen3.5-9b's chat template. `--cache-ram 0` disables the
> default 8 GiB prompt cache, which caused Metal GPU OOM when combined with the model's
> resident weights on this hardware. In local mode, judgement calls also go to port 8001;
> llama.cpp handles both over the same port and the same model.

### Run Pipeline

```bash
# Symlink already set to .env.local (done in setup step A6)
python main.py
# → ENV=local → M1/M2 → qwen3.5-9b (port 8001)
# → QWEN → qwen3.5-9b (port 8001)
# → 5 easy questions, 3 iterations
```

---

## 8. Vast.ai Cloud Kurulumu

### Model Download

```bash
cd /workspace/models && pip install huggingface_hub

# M1 — Qwen3.5-9B (~6 GB)
huggingface-cli download unsloth/Qwen3.5-9B-GGUF \
    --include "*UD-Q4_K_XL*" --local-dir ./qwen3.5-9b

# M2 — Qwen2.5-Coder-1.5B-Instruct (~1 GB)
huggingface-cli download unsloth/Qwen2.5-Coder-1.5B-Instruct-GGUF \
    --include "*Q4_K_M*" --local-dir ./qwen2.5-coder-1.5b

# QWEN — Judgement (~24 GB)
huggingface-cli download unsloth/Qwen3.5-35B-A3B-GGUF \
    --include "*UD-Q4_K_XL*" --local-dir ./qwen3.5-35b
```

### scripts/start_model.sh

```bash
#!/bin/bash
# Usage: bash scripts/start_model.sh M1|M2|QWEN
# Stops previous server, starts new model.
# Unsloth split GGUFs: first_shard() picks the first shard;
# llama.cpp auto-discovers the rest.

set -e
MODEL=${1:?"\nUsage: $0 M1|M2|QWEN"}
MODELS=/workspace/models

first_shard() {
    ls "$1/"*.gguf 2>/dev/null | sort | head -1
}

stop_server() {
    pkill -f "llama-server" 2>/dev/null || true
    sleep 2
}

case $MODEL in
M1)
    MODEL_PATH=$(first_shard "$MODELS/qwen3.5-9b")
    [ -z "$MODEL_PATH" ] && { echo "M1 model files not found in $MODELS/qwen3.5-9b"; exit 1; }
    echo "Starting M1: $MODEL_PATH"
    stop_server
    /workspace/llama.cpp/llama-server \
        --model "$MODEL_PATH" \
        --port 8001 --host 0.0.0.0 \
        --ctx-size 16384 --n-gpu-layers 99 \
        --jinja --cache-ram 0 \
        --temp 0.6 --top-p 0.95 --min-p 0.01 --top-k 40
    ;;
M2)
    MODEL_PATH=$(first_shard "$MODELS/qwen2.5-coder-1.5b")
    [ -z "$MODEL_PATH" ] && { echo "M2 model files not found in $MODELS/qwen2.5-coder-1.5b"; exit 1; }
    echo "Starting M2: $MODEL_PATH"
    stop_server
    /workspace/llama.cpp/llama-server \
        --model "$MODEL_PATH" \
        --port 8002 --host 0.0.0.0 \
        --ctx-size 16384 --n-gpu-layers 99 \
        --temp 0.7 --top-p 0.95 --min-p 0.01 --top-k 40
    ;;
QWEN)
    MODEL_PATH=$(first_shard "$MODELS/qwen3.5-35b")
    [ -z "$MODEL_PATH" ] && { echo "QWEN model files not found in $MODELS/qwen3.5-35b"; exit 1; }
    echo "Starting QWEN: $MODEL_PATH"
    stop_server
    /workspace/llama.cpp/llama-server \
        --model "$MODEL_PATH" \
        --port 8004 --host 0.0.0.0 \
        --ctx-size 8192 --n-gpu-layers 99 \
        --temp 0.3 --top-p 0.95 --min-p 0.01
    ;;
*)
    echo "Invalid model: $MODEL"
    exit 1
    ;;
esac
```

---

## 9. Updated Modules (Summary)

There is **not a single line** related to ENV or model selection in any other module —
`config/__init__.py` handles this when building the registry.

```python
# models/client.py — unchanged
from config import model_registry

def get_client(model_id: str) -> OpenAI:
    cfg = model_registry.get(model_id)
    # local mode: M1/M2 → port 8001 (qwen3.5-9b)
    # cloud: M1 → port 8001 (qwen3.5-9b), M2 → port 8002 (qwen2.5-coder-1.5b)
    return OpenAI(base_url=cfg.base_url, api_key=model_registry.api_key())

# pipeline/router.py — unchanged
from config import model_registry
MODEL_SEQUENCE = [m.model_id for m in model_registry.codegen_models(settings.model_sequence)]
# → ["M1", "M2"] (default from config/pipeline.py)

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

### Invalid ENV test

```bash
ENV=staging python -c "from config import settings" 2>&1
# ValueError: Geçersiz ENV değeri: 'staging'. 'local' veya 'cloud' olmalı.
```
