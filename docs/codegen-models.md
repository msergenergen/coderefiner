# codegen-models.md — Code Generation Models

## 1. Model Overview

| ID | Model | Port | Temperature | Min. Bellek | Notlar |
|----|-------|------|-------------|------------|--------|
| M1 | gpt-oss-20b | 8001 | 1.0 | ~11 GB | Apache 2.0; `--jinja` zorunlu; Harmony format |
| M2 | Qwen3-Coder-Next | 8002 | 1.0 | ~46 GB | **Dominant model** — determines hardware requirements |
| M3 | GLM-4.7-Flash | 8003 | 1.0 | ~18 GB | `--repeat-penalty 1.0` zorunlu |

All models run as OpenAI-compatible APIs on llama.cpp.
The pipeline connects to all three identically via the `openai` Python SDK.

> **Dominant model: Qwen3-Coder-Next (~46 GB GGUF Q4).**
> Hardware selection is based on this model: minimum A100 80GB.

---

## 2. models/client.py

```python
"""
llama.cpp üzerindeki modeller için OpenAI-uyumlu istemci.
Model config artık config/models.py üzerinden yönetilir;
bu modül sadece istemci oluşturma mantığını içerir.
"""
from openai import OpenAI
from config import model_registry


def get_client(model_id: str) -> OpenAI:
    """
    Verilen model ID için OpenAI istemcisi döner.

    Args:
        model_id: "M1" | "M2" | "M3" | "QWEN"

    Returns:
        OpenAI client (base_url llama.cpp server'ına işaret eder)
    """
    cfg = model_registry.get(model_id)
    return OpenAI(base_url=cfg.base_url, api_key=model_registry.api_key())


def get_model_name(model_id: str) -> str:
    return model_registry.get(model_id).model_name


def get_temperature(model_id: str) -> float:
    return model_registry.get(model_id).temperature

    base_url, _, _ = _MODEL_CONFIG[model_id]
    api_key = os.getenv("LLAMA_API_KEY", "local")

    return OpenAI(base_url=base_url, api_key=api_key)


def get_model_name(model_id: str) -> str:
    _, model_name, _ = _MODEL_CONFIG[model_id]
    return model_name


def get_temperature(model_id: str) -> float:
    _, _, temperature = _MODEL_CONFIG[model_id]
    return temperature


def generate(
    model_id: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 2048,
) -> str:
    """
    Verilen model ID ile tamamlama isteği gönderir.

    Args:
        model_id:      "M1" | "M2" | "M3"
        system_prompt: System prompt metni
        user_prompt:   Kullanıcı mesajı (problem + gereksinimler)
        max_tokens:    Üretilecek maksimum token

    Returns:
        Modelin ham metin yanıtı
    """
    client     = get_client(model_id)
    model_name = get_model_name(model_id)
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
    Model server'ının ayakta olup olmadığını kontrol eder.

    Returns:
        True: server yanıt verdi
        False: server ulaşılamaz
    """
    try:
        client = get_client(model_id)
        client.models.list()
        return True
    except Exception:
        return False
```

---

## 3. models/prompts.py

```python
"""
System prompt şablonları — codegen ve judgement modelleri için.

Codegen prompt dört katmandan oluşur:
  1. CODEGEN_IDENTITY        — kim olduğu
  2. CODEGEN_CONTEXT         — pipeline'da nerede olduğu
  3. CODEGEN_CONSTRAINTS     — davranış kuralları
  4. CODEGEN_FEEDBACK_RULES  — feedback geldiğinde nasıl davranacağı (sadece iterasyon > 1)

İlk iterasyon : katman 1 + 2 + 3
Sonraki iterasyonlar: katman 1 + 2 + 3 + 4 + improved_prompt
"""

# ── Katman 1: Kimlik ──────────────────────────────────────────────────────────

CODEGEN_IDENTITY = """\
You are an expert Python programmer specializing in competitive programming.
Your sole responsibility in this pipeline is to produce correct, efficient,
and secure Python solutions to algorithmic problems.\
"""

# ── Katman 2: Pipeline bağlamı ────────────────────────────────────────────────

CODEGEN_CONTEXT = """\
You are operating as the code generation node of an automated iterative \
refinement pipeline. After you produce a solution, it passes through \
three evaluation stages:

  1. Static Analyzer  — checks complexity (CC, MI) and security (Bandit)
  2. Sandbox Executor — runs your code against public and private test cases
  3. Judge LLM        — scores your solution 0–100 on a structured rubric

If your solution scores below the acceptance threshold, you will receive
specific improvement feedback and be asked to produce a new solution.
This process repeats up to 10 times per model.\
"""

# ── Katman 3: Davranış kuralları ──────────────────────────────────────────────

CODEGEN_CONSTRAINTS = """\
Rules you must follow in every solution:

Output format:
- Output ONLY Python code inside ```python ... ``` markers
- Do not include any text, explanation, or comments outside the markers

I/O format:
- Read all input from stdin using input() or sys.stdin
- Write all output to stdout using print()
- Do not wrap your solution in a function unless the problem explicitly requires it
- Do not include any print() calls for debugging or examples

Correctness:
- Handle ALL edge cases: empty input, negative numbers, zero, large N
- Do not assume input is always well-formed

Security (enforced by static analyzer):
- Allowed imports only: os, sys, math, random, time, datetime,
  collections, itertools, functools, heapq, bisect, string,
  re, json, copy, typing, dataclasses
- Do NOT use: subprocess, socket, requests, exec(), eval(), __import__()

Code quality:
- Keep Cyclomatic Complexity below 10 per function
- Prefer clear variable names over single letters (except loop counters)
- Split functions exceeding 30 lines into smaller helpers\
"""

# ── Katman 4: Feedback entegrasyonu (sadece iterasyon > 1) ───────────────────

CODEGEN_FEEDBACK_RULES = """\
You are receiving improvement feedback from the previous iteration.

How to use this feedback:
- Read every point carefully — each one corresponds to a real failure
- Write a COMPLETELY NEW solution — do not patch or modify the old one
- Address EVERY issue mentioned — skipping any point will cause the same failure
- Reference specific failing test cases in your logic
- The feedback already tells you what went wrong — trust it and fix it

What NOT to do:
- Do not copy the previous solution with minor edits
- Do not add print() statements for the failing cases
- Do not explain what you changed — just produce the fixed code\
"""


# ── Prompt inşası ─────────────────────────────────────────────────────────────

def build_system_prompt(improved_prompt: str | None = None) -> str:
    """
    Codegen sistem promptunu dört katmandan inşa eder.

    Args:
        improved_prompt: Node 3'ün ürettiği iyileştirme talimatları.
                         None ise ilk iterasyon — feedback katmanı eklenmez.

    Returns:
        Tam sistem prompt metni.

    Katman yapısı:
        İlk iterasyon      → IDENTITY + CONTEXT + CONSTRAINTS
        Sonraki iterasyonlar → IDENTITY + CONTEXT + CONSTRAINTS
                              + FEEDBACK_RULES + improved_prompt
    """
    base = "\n\n---\n\n".join([
        CODEGEN_IDENTITY,
        CODEGEN_CONTEXT,
        CODEGEN_CONSTRAINTS,
    ])

    if not improved_prompt:
        return base

    return "\n\n---\n\n".join([
        CODEGEN_IDENTITY,
        CODEGEN_CONTEXT,
        CODEGEN_CONSTRAINTS,
        CODEGEN_FEEDBACK_RULES,
        f"## Improvement Notes from Previous Attempt\n\n{improved_prompt}",
    ])


def build_user_prompt(formatted_problem: str) -> str:
    """
    LCB sorusunu user prompt formatına dönüştürür.
    System prompt'u tekrar etmez — sadece problemi verir.
    """
    return f"Solve the following problem:\n\n{formatted_problem}\n\nWrite your complete Python solution:"
```

---

## 3.1 Layer Structure — Visual Overview

```
İlk iterasyon (improved_prompt = None):
┌─────────────────────────┐
│   CODEGEN_IDENTITY      │  "You are an expert Python programmer..."
├─────────────────────────┤
│   CODEGEN_CONTEXT       │  "You operate inside a pipeline..."
├─────────────────────────┤
│   CODEGEN_CONSTRAINTS   │  "Rules: stdin/stdout, whitelist, edge cases..."
└─────────────────────────┘

Sonraki iterasyonlar (improved_prompt doluysa):
┌─────────────────────────┐
│   CODEGEN_IDENTITY      │
├─────────────────────────┤
│   CODEGEN_CONTEXT       │
├─────────────────────────┤
│   CODEGEN_CONSTRAINTS   │
├─────────────────────────┤
│   CODEGEN_FEEDBACK_RULES│  "Write completely new — address every point..."
├─────────────────────────┤
│   improved_prompt       │  Node 3'ün ürettiği spesifik talimatlar
└─────────────────────────┘
```

---

## 4. nodes/codegen.py

```python
"""
LangGraph Node 0: Kod Üretimi

Aktif modele problemi gönderir, üretilen kodu parse eder
ve state'e yazar.
"""
import re
from pipeline.state import PipelineState
from models.client import generate, health_check
from models.prompts import build_system_prompt, build_user_prompt
from dataset.splitter import format_problem


def codegen_node(state: PipelineState) -> PipelineState:
    """
    Aktif modelden kod üretir.

    Giriş  (state'ten okur): active_model, problem, improved_prompt
    Çıktı  (state'e yazar):  generated_code, iteration (artırılır)
    """
    model_id = state["active_model"]

    # Server health check
    if not health_check(model_id):
        raise RuntimeError(
            f"{model_id} server'ına ulaşılamıyor. "
            f"llama.cpp server'ını başlattığından emin ol."
        )

    # Build prompt
    system_prompt = build_system_prompt(
        improved_prompt=state.get("node3_improved_prompt")
    )
    user_prompt = build_user_prompt(state["problem"])

    # Generate code
    raw_response = generate(
        model_id=model_id,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )

    # Extract ```python ... ``` block
    code = _extract_code(raw_response)

    return {
        **state,
        "generated_code": code,
        "iteration": state.get("iteration", 0) + 1,
    }


def _extract_code(response: str) -> str:
    """
    Modelin yanıtından ```python ... ``` bloğunu çıkarır.
    Blok bulunamazsa tüm yanıtı döner.
    """
    pattern = r"```python\s*(.*?)```"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response.strip()
```

---

## 5. pipeline/router.py

```python
"""
Sequential model router.
M1 → M2 → M3 sırasıyla çalıştırır.
Bir model tamamlanınca bir sonrakine geçer.
"""
from pipeline.state import PipelineState

MODEL_SEQUENCE = ["M1", "M2", "M3"]


def get_next_model(current: str) -> str | None:
    """
    Mevcut modelden sonraki modeli döner.
    M3'ten sonra None döner (tüm modeller tamamlandı).
    """
    idx = MODEL_SEQUENCE.index(current)
    if idx + 1 < len(MODEL_SEQUENCE):
        return MODEL_SEQUENCE[idx + 1]
    return None


def advance_model(state: PipelineState) -> PipelineState:
    """
    Bir sonraki modele geçer ve iterasyon sayacını sıfırlar.
    """
    next_model = get_next_model(state["active_model"])

    if next_model is None:
        # All models completed
        return {**state, "final_status": "PASS"}

    return {
        **state,
        "active_model": next_model,
        "iteration": 0,
        "node3_improved_prompt": "",   # Sıfırla
        "final_status": "PENDING",
    }
```

---

## 6. Verification

### Model Server Testi

```bash
# While M1 server is running:
source .venv/bin/activate
python -c "
from models.client import generate, health_check

# Health check
print('M1 sağlık:', health_check('M1'))

# Simple code generation
code = generate(
    model_id='M1',
    system_prompt='You are a Python expert. Output only code.',
    user_prompt='Write a function that returns the sum of a list.',
    max_tokens=200,
)
print('Üretilen kod:')
print(code)
"
```

### Expected Output

```
M1 sağlık: True
Üretilen kod:
def sum_list(numbers):
    return sum(numbers)
```

### Codegen Node Testi

```bash
python -c "
from pipeline.state import PipelineState
from nodes.codegen import codegen_node

state: PipelineState = {
    'active_model': 'M1',
    'iteration': 0,
    'max_iterations': 10,
    'problem': '## Problem: Two Sum\nDifficulty: easy\nReturn indices of two numbers that add up to target.',
    'difficulty': 'easy',
    'generated_code': '',
    'node3_improved_prompt': '',
    'memory_log': [],
    'final_status': 'PENDING',
}

result = codegen_node(state)
print('Üretilen kod:')
print(result['generated_code'])
print('İterasyon:', result['iteration'])
"
```
