"""
Model definitions and build_registry factory.

ENV=cloud (default, Vast.ai):
    M1   → gpt-oss-20b      (~11 GB)   port 8001
    M2   → Qwen3-Coder-Next (~46 GB)   port 8002  ← dominant model
    M3   → GLM-4.7-Flash    (~18 GB)   port 8003
    QWEN → Qwen3.5-35B-A3B  (~24 GB)   port 8004

ENV=local (Mac Mini M4 16GB):
    M1   → Qwen2.5-Coder-7B (~4 GB)   port 8001
    M2   → Qwen2.5-Coder-7B (~4 GB)   port 8001  ← same server
    M3   → Qwen2.5-Coder-7B (~4 GB)   port 8001  ← same server
    QWEN → Qwen2.5-Coder-7B (~4 GB)   port 8001  ← same server

In local mode all roles route to a single server (8001).
Pipeline logic (ordering, gates, iterations) runs identically.

Sequential execution note:
    M1, M2, M3 are NEVER in memory simultaneously. Each model runs
    and finishes before the next starts. Memory = single largest model,
    not sum of all models.
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
            raise ValueError(f"Invalid model_id: {model_id!r}. Must be M1/M2/M3/QWEN.")

    def codegen_models(self) -> list:
        """Sequential execution order: [M1, M2, M3]."""
        return [self.M1, self.M2, self.M3]

    def judgement_model(self) -> ModelConfig:
        return self.QWEN

    def api_key(self) -> str:
        return os.getenv("LLAMA_API_KEY", "local")

    def is_local(self) -> bool:
        """True when all roles route to the same port (local mode)."""
        ports = {self.M1.port, self.M2.port, self.M3.port, self.QWEN.port}
        return len(ports) == 1


def _cloud_registry() -> ModelRegistry:
    """
    Vast.ai A100 80GB production registry.
    Each model runs on its own port and its own server.
    """
    return ModelRegistry(
        M1=ModelConfig(
            model_id="M1",
            base_url=os.getenv("M1_BASE_URL", "http://localhost:8001/v1"),
            model_name="gpt-oss-20b",
            temperature=1.0,   # Harmony format; follows OpenAI recommendation
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
    Mac Mini M4 16GB local debug registry.

    All codegen roles (M1/M2/M3) route to the same small model:
        Qwen2.5-Coder-7B  — Q4_K_M ~4 GB  — fits easily in 16 GB

    Judgement also uses the same model on the same port.
    Single llama.cpp server (port 8001) handles all roles.
    Pipeline logic (ordering, gates, iterations) is identical to cloud.
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
