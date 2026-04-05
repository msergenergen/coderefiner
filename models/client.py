"""
OpenAI-compatible client for models running on llama.cpp.
Model config is managed via config/models.py;
this module only contains client creation logic.
"""
from openai import OpenAI
from config import model_registry


def get_client(model_id: str) -> OpenAI:
    """
    Returns an OpenAI client for the given model ID.

    Args:
        model_id: "M1" | "M2" | "M3" | "QWEN"

    Returns:
        OpenAI client (base_url points to llama.cpp server)
    """
    cfg = model_registry.get(model_id)
    return OpenAI(base_url=cfg.base_url, api_key=model_registry.api_key())


def get_model_name(model_id: str) -> str:
    return model_registry.get(model_id).model_name


def get_temperature(model_id: str) -> float:
    return model_registry.get(model_id).temperature


def generate(
    model_id: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 2048,
) -> str:
    """
    Sends a completion request to the given model.

    Args:
        model_id:      "M1" | "M2" | "M3" | "QWEN"
        system_prompt: System prompt text
        user_prompt:   User message (problem + requirements)
        max_tokens:    Maximum tokens to generate

    Returns:
        Raw text response from the model
    """
    client      = get_client(model_id)
    model_name  = get_model_name(model_id)
    temperature = get_temperature(model_id)

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return response.choices[0].message.content


def health_check(model_id: str) -> bool:
    """
    Checks whether the model server is responding.

    Returns:
        True: server responded
        False: server unreachable
    """
    try:
        client = get_client(model_id)
        client.models.list()
        return True
    except Exception:
        return False
