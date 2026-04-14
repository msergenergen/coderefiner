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

| ENV | M1 | M2 | M3 | Judgement | Toplam Bellek |
|-----|----|----|----|----------|--------------|
| `local` | Qwen2.5-Coder-7B | Qwen2.5-Coder-7B | Qwen2.5-Coder-7B | Qwen2.5-Coder-3B | ~6 GB |
| `cloud` | gpt-oss-20b | Qwen3-Coder-Next | GLM-4.7-Flash | Qwen3.5-35B-A3B | ~99 GB |

In local mode all codegen models (M1/M2/M3) route to the same small model;
pipeline flow, gates, and iteration logic run identically.
A slightly smaller model (3B) is used for judgement — sufficiently consistent.

### Sequential Model Architecture and Memory

```
START → M1 (çalışır, biter) → M2 (çalışır, biter) → M3 (çalışır, biter) → END
         ↑ bellekte tek model aynı anda
```

Cloud dominant model: **Qwen3-Coder-Next (~46 GB)** → minimum A100 80GB.
Local dominant model: **Qwen2.5-Coder-7B (~4 GB)** → Mac Mini M4 16GB is sufficient.

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
    M1 → gpt-oss-20b      (~11 GB)   port 8001
    M2 → Qwen3-Coder-Next (~46 GB)   port 8002  ← baskın model
    M3 → GLM-4.7-Flash    (~18 GB)   port 8003
  QWEN → Qwen3.5-35B-A3B (~24 GB)   port 8004

ENV=local (Mac Mini M4 16GB):
    M1 → Qwen2.5-Coder-7B (~4 GB)   port 8001
    M2 → Qwen2.5-Coder-7B (~4 GB)   port 8001  ← aynı server
    M3 → Qwen2.5-Coder-7B (~4 GB)   port 8001  ← aynı server
  QWEN → Qwen2.5-Coder-7B (~4 GB)   port 8001  ← aynı server

Local modda tüm roller tek server'a (8001) gider.
Pipeline mantığı (sıra, gate, iterasyon) tamamen aynı çalışır.

Sıralı çalışma notu:
    M1, M2, M3 aynı anda bellekte OLMAZ. Her model çalışıp bittikten
    sonra bir sonraki başlar. Bellek = max(tüm modeller) değil,
    en büyük tek modelin gereksinimi.
"""
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class ModelConfig:
    model_id:    str    # "M1" | "M2" | "M3" | "QWEN"
    base_url:    str    # llama.cpp server URL
    model_name:  str    # Model name sent to llama.cpp
    temperature: float
    port:        int    # Reference for start_model.sh
    role:        str    # "codegen" | "judgement"


@dataclass(frozen=True)
class ModelRegistry:
    M1:   ModelConfig
    M2:   ModelConfig
    M3:   ModelConfig
    QWEN: ModelConfig

    def get(self, model_id: str) -> ModelConfig:
        try:
            return getattr(self, model_id)
        except AttributeError:
            raise ValueError(f"Geçersiz model_id: {model_id!r}. M1/M2/M3/QWEN olmalı.")

    def codegen_models(self) -> list[ModelConfig]:
        """Sıralı çalışma sırası: [M1, M2, M3]."""
        return [self.M1, self.M2, self.M3]

    def judgement_model(self) -> ModelConfig:
        return self.QWEN

    def api_key(self) -> str:
        return os.getenv("LLAMA_API_KEY", "local")

    def is_local(self) -> bool:
        """Tüm roller aynı port'a gidiyorsa local mod aktif."""
        ports = {self.M1.port, self.M2.port, self.M3.port, self.QWEN.port}
        return len(ports) == 1


def _cloud_registry() -> ModelRegistry:
    """
    Vast.ai A100 80GB üretim registry'si.
    Her model kendi portunda, kendi sunucusunda çalışır.
    """
    return ModelRegistry(
        M1=ModelConfig(
            model_id="M1",
            base_url=os.getenv("M1_BASE_URL", "http://localhost:8001/v1"),
            model_name="gpt-oss-20b",
            temperature=1.0,   # Harmony format; OpenAI recommendation
            port=8001,
            role="codegen",
        ),
        M2=ModelConfig(
            model_id="M2",
            base_url=os.getenv("M2_BASE_URL", "http://localhost:8002/v1"),
            model_name="qwen3-coder-next",
            temperature=1.0,   # Qwen recommendation; dominant model (~46 GB)
            port=8002,
            role="codegen",
        ),
        M3=ModelConfig(
            model_id="M3",
            base_url=os.getenv("M3_BASE_URL", "http://localhost:8003/v1"),
            model_name="glm-4.7-flash",
            temperature=1.0,   # Z.ai recommendation
            port=8003,
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
    Mac Mini M4 16GB local debug registry'si.

    Tüm codegen rolleri (M1/M2/M3) aynı küçük modele yönlenir:
        Qwen2.5-Coder-7B  — Q4_K_M ~4 GB  — 16 GB'a rahatça sığar

    Judgement için biraz daha küçük:
        Qwen2.5-Coder-3B  — Q4_K_M ~2 GB

    Tek llama.cpp server (port 8001) tüm roller için yeterli.
    Pipeline mantığı (sıra, gate, iterasyon) cloud ile aynı.
    """
    local_url = os.getenv("LOCAL_BASE_URL", "http://localhost:8001/v1")

    codegen_slot = ModelConfig(
        model_id="LOCAL_CODEGEN",
        base_url=local_url,
        model_name=os.getenv("LOCAL_CODEGEN_MODEL", "qwen2.5-coder-7b"),
        temperature=0.7,
        port=8001,
        role="codegen",
    )

    judgement_slot = ModelConfig(
        model_id="LOCAL_JUDGEMENT",
        base_url=local_url,
        model_name=os.getenv("LOCAL_JUDGEMENT_MODEL", "qwen2.5-coder-7b"),
        temperature=0.3,
        port=8001,
        role="judgement",
    )

    return ModelRegistry(
        M1=codegen_slot,
        M2=codegen_slot,
        M3=codegen_slot,
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
LOCAL_CODEGEN_MODEL=qwen2.5-coder-7b   # Q4_K_M ~4 GB
LOCAL_JUDGEMENT_MODEL=qwen2.5-coder-7b  # Q4_K_M ~4 GB (codegen ile aynı model)
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
# Codegen — ~4 GB
huggingface-cli download bartowski/Qwen2.5-Coder-7B-Instruct-GGUF \
    --include "*Q4_K_M*" \
    --local-dir ./models/qwen2.5-coder-7b

# Judgement — ~2 GB
huggingface-cli download bartowski/Qwen2.5-Coder-3B-Instruct-GGUF \
    --include "*Q4_K_M*" \
    --local-dir ./models/qwen2.5-coder-7b
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
# Start codegen model on first run:
./llama.cpp/llama-server \
    --model ./models/qwen2.5-coder-7b/*.gguf \
    --port 8001 --host 0.0.0.0 \
    --ctx-size 8192 --n-gpu-layers 99 \
    --temp 0.7
```

> In local mode, judgement calls also go to port 8001. llama.cpp
> handles both over the same port.
> If you want to run the judgement model (3B) separately, open a different port,
> but overriding the model name via `LOCAL_JUDGEMENT_MODEL` is sufficient.

### Run Pipeline

```bash
# Symlink already set to .env.local (done in setup step A6)
python main.py
# → ENV=local → M1/M2/M3 → qwen2.5-coder-7b (port 8001)
# → QWEN → qwen2.5-coder-7b (port 8001)
# → 5 easy questions, 3 iterations
```

---

## 8. Vast.ai Cloud Kurulumu

### Model Download

```bash
cd /workspace/models && pip install huggingface_hub

# M1 — gpt-oss-20b (~11 GB)
huggingface-cli download unsloth/gpt-oss-20b-GGUF \
    --include "*UD-Q4_K_XL*" --local-dir ./gpt-oss-20b

# M2 — Qwen3-Coder-Next (~46 GB) — dominant model
huggingface-cli download unsloth/Qwen3-Coder-Next-GGUF \
    --include "*UD-Q4_K_XL*" --local-dir ./qwen3-coder-next

# M3 — GLM-4.7-Flash (~18 GB)
huggingface-cli download unsloth/GLM-4.7-Flash-GGUF \
    --include "*UD-Q4_K_XL*" --local-dir ./glm-4.7-flash

# QWEN — Judgement (~24 GB)
huggingface-cli download unsloth/Qwen3.5-35B-A3B-GGUF \
    --include "*UD-Q4_K_XL*" --local-dir ./qwen3.5-35b
```

### scripts/start_model.sh

```bash
#!/bin/bash
# Usage: bash scripts/start_model.sh M1|M2|M3|QWEN
# Stops previous server, starts new model.
# Unsloth split GGUFs: first_shard() picks the first shard;
# llama.cpp auto-discovers the rest.

set -e
MODEL=${1:?"\nKullanım: $0 M1|M2|M3|QWEN"}
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
    MODEL_PATH=$(first_shard "$MODELS/gpt-oss-20b")
    [ -z "$MODEL_PATH" ] && { echo "M1 model files not found in $MODELS/gpt-oss-20b"; exit 1; }
    echo "Starting M1: $MODEL_PATH"
    stop_server
    /workspace/llama.cpp/llama-server \
        --model "$MODEL_PATH" \
        --port 8001 --host 0.0.0.0 \
        --ctx-size 16384 --n-gpu-layers 99 \
        --jinja \
        --temp 1.0 --top-p 1.0 --min-p 0.0 --top-k 0
    ;;
M2)
    MODEL_PATH=$(first_shard "$MODELS/qwen3-coder-next")
    [ -z "$MODEL_PATH" ] && { echo "M2 model files not found in $MODELS/qwen3-coder-next"; exit 1; }
    echo "Starting M2: $MODEL_PATH"
    stop_server
    /workspace/llama.cpp/llama-server \
        --model "$MODEL_PATH" \
        --port 8002 --host 0.0.0.0 \
        --ctx-size 16384 --n-gpu-layers 99 \
        --temp 1.0 --top-p 0.95 --min-p 0.01 --top-k 40
    ;;
M3)
    MODEL_PATH=$(first_shard "$MODELS/glm-4.7-flash")
    [ -z "$MODEL_PATH" ] && { echo "M3 model files not found in $MODELS/glm-4.7-flash"; exit 1; }
    echo "Starting M3: $MODEL_PATH"
    stop_server
    /workspace/llama.cpp/llama-server \
        --model "$MODEL_PATH" \
        --port 8003 --host 0.0.0.0 \
        --ctx-size 16384 --n-gpu-layers 99 \
        --repeat-penalty 1.0 \
        --temp 1.0 --top-p 0.95 --min-p 0.01
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
    echo "Geçersiz model: $MODEL"
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
    # local mode: M1/M2/M3 → port 8001 (qwen2.5-coder-7b)
    # cloud'da M1 → port 8001 (gpt-oss-20b), M2 → 8002, M3 → 8003
    return OpenAI(base_url=cfg.base_url, api_key=model_registry.api_key())

# pipeline/router.py — unchanged
from config import model_registry
MODEL_SEQUENCE = [m.model_id for m in model_registry.codegen_models()]
# → ["M1", "M2", "M3"] her iki ortamda da

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
print('M1 model:   ', model_registry.M1.model_name)   # gpt-oss-20b
print('M1 port:    ', model_registry.M1.port)          # 8001
print('M2 model:   ', model_registry.M2.model_name)   # qwen3-coder-next
print('M2 port:    ', model_registry.M2.port)          # 8002
print('M3 model:   ', model_registry.M3.model_name)   # glm-4.7-flash
print('M3 port:    ', model_registry.M3.port)          # 8003
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
print('M1 model:   ', model_registry.M1.model_name)   # qwen2.5-coder-7b
print('M1 port:    ', model_registry.M1.port)          # 8001
print('M2 model:   ', model_registry.M2.model_name)   # qwen2.5-coder-7b
print('M2 port:    ', model_registry.M2.port)          # 8001 (aynı)
print('M3 model:   ', model_registry.M3.model_name)   # qwen2.5-coder-7b
print('M3 port:    ', model_registry.M3.port)          # 8001 (aynı)
print('QWEN model: ', model_registry.QWEN.model_name)  # qwen2.5-coder-7b
print('QWEN port:  ', model_registry.QWEN.port)        # 8001 (aynı)
print('Is local:   ', model_registry.is_local())       # True
"
```

### Invalid ENV test

```bash
ENV=staging python -c "from config import settings" 2>&1
# ValueError: Invalid ENV value: 'staging'. Must be 'local' or 'cloud'.
```
