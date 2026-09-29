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
    Returns the correct registry based on the ENV flag.

    Args:
        env: "cloud" (default) | "local"

    Returns:
        ModelRegistry — other modules use this registry and don't need
        to know which environment they're running in.
    """
    if env == "local":
        return _local_registry()
    return _cloud_registry()
