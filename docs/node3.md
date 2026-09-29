# node3.md — Judgement LLM (Node 3)

## 1. Purpose

Synthesize Node 1 and Node 2 metrics — including failing test case details —
to produce a 100-point score, make a PASS/FAIL decision, and on failure
write a **fully self-contained** improved system prompt for the next iteration.

### Design Decisions

| Decision | Rationale |
|-------|---------|
| Judgement model = Judge (independent of M1/M2/M3) | Reduces self-enhancement bias risk [Zheng et al., 2023] |
| `improved_prompt` is **completely rewritten** every iteration | Accumulated prompts produce contradictory instructions (semantic drift) |
| Threshold table is **embedded** in the judgement prompt | So the Judge understands why `mi=48` is poor |
| Failing test cases (input/expected/got) are **included verbatim** | Answers "why did it fail?" far better than numeric metrics alone |
| Fallback mechanism | Rule-based prompt is generated on LLM error — pipeline keeps running |

---

## 2. Rubric (100 Puan)

| Dimension | Weight | Source |
|-------|---------|--------|
| Accuracy | 25p | Node 2: `pass_rate` |
| Efficiency | 20p | Node 2: `cpu_ms` + `ram_mb` |
| Quality | 20p | Node 1: MI + CC composite |
| Security | 15p | Node 1: Bandit severity weighted |
| Algorithm Choice | 10p | Judge: appropriate algorithm for the problem? |
| Error Handling | 10p | Judge: exception and edge case handling |

---

## 3. metrics/scoring.py

`compute_efficiency_score` and `compute_quality_score` read their limits from
`config.thresholds` (`node2.cpu_limit_ms`, `node2.ram_limit_mb`, `node1.cc_hard_limit`)
instead of hardcoding them — this keeps the scoring formula from silently
desyncing from the threshold table shown to the Judge (`judgement_context_table()`).

```python
"""
100 puanlık rubric hesaplama — sayısal boyutlar.
Algorithm Choice (10p) ve Error Handling (10p) Judge modeli tarafından doldurulur.
"""
from config import thresholds


def compute_accuracy_score(pass_rate: float) -> float:
    """Doğruluk skoru: 0–25 puan."""
    return round(pass_rate * 25, 2)


def compute_efficiency_score(cpu_ms: float, ram_mb: float) -> float:
    """Verimlilik skoru: 0–20 puan."""
    n2 = thresholds.node2
    cpu_score = max(0.0, 1 - cpu_ms / n2.cpu_limit_ms)
    ram_score = max(0.0, 1 - ram_mb / n2.ram_limit_mb)
    return round((cpu_score + ram_score) / 2 * 20, 2)


def compute_quality_score(mi: float, cc: float) -> float:
    """Kod kalitesi skoru: 0–20 puan."""
    mi_score = min(1.0, mi / 100)
    cc_score = max(0.0, 1 - cc / thresholds.node1.cc_hard_limit)
    return round((mi_score + cc_score) / 2 * 20, 2)


def compute_security_score(high_count: int, medium_count: int) -> float:
    """Güvenlik skoru: 0–15 puan."""
    score = 15 - (high_count * 15) - (medium_count * 3)
    return round(max(0.0, float(score)), 2)


def compute_numeric_scores(
    pass_rate: float,
    cpu_ms: float,
    ram_mb: float,
    mi: float,
    cc: float,
    bandit_high: bool,
    bandit_medium_count: int = 0,
) -> dict:
    """
    Sayısal olarak hesaplanabilen 4 boyutu hesaplar.

    Returns:
        {
            "accuracy":      float,   # 0–25
            "efficiency":    float,   # 0–20
            "quality":       float,   # 0–20
            "security":      float,   # 0–15
            "numeric_total": float    # 0–80
        }
    """
    acc  = compute_accuracy_score(pass_rate)
    eff  = compute_efficiency_score(cpu_ms, ram_mb)
    qual = compute_quality_score(mi, cc)
    sec  = compute_security_score(
        high_count=1 if bandit_high else 0,
        medium_count=bandit_medium_count,
    )
    return {
        "accuracy":      acc,
        "efficiency":    eff,
        "quality":       qual,
        "security":      sec,
        "numeric_total": round(acc + eff + qual + sec, 2),
    }
```

---

## 4. nodes/judgement.py

```python
"""
LangGraph Node 3: Judgement LLM (Judge)

Kritik tasarım kararları:
  1. Her iterasyonda improved_prompt TAMAMEN YENİDEN YAZILIR — birikim yok.
  2. Threshold tablosu system prompt'a gömülü.
  3. Başarısız test case'ler (input/expected/got) prompt'a eklenir.
  4. Judge modeli (QWEN) kod üreten modellerden (M1/M2/M3) bağımsız.
"""
import json
import re
from pipeline.state import PipelineState
from metrics.scoring import compute_numeric_scores
from models.client import generate

JUDGEMENT_MODEL = "QWEN"

# ── Threshold table — embedded context for the Judge model ──────────────────
_THRESHOLD_CONTEXT = """
## Metric Threshold Reference

| Metric | Good | Acceptable | Poor |
|--------|------|------------|------|
| Cyclomatic Complexity (CC) | ≤ 5 | 6–10 | > 10 |
| Maintainability Index (MI) | ≥ 85 | 65–85 | < 65 |
| Halstead Volume | ≤ 500 | 500–1000 | > 1000 |
| Test Pass Rate | ≥ 0.90 | 0.70–0.90 | < 0.70 |
| Peak Memory | ≤ 256 MB | 256–512 MB | > 512 MB |
| CPU Time | ≤ 2 s | 2–5 s | > 5 s |

Bandit HIGH severity: zero tolerance — any finding is critical.
MI < 65 means the code is hard to maintain and likely poorly structured.
CC > 10 means the function has too many branches and is hard to test.
"""

JUDGEMENT_SYSTEM_PROMPT = f"""You are a strict code quality judge for an iterative code refinement pipeline.

You evaluate Python solutions to competitive programming problems using static analysis
results (Node 1) and sandbox execution results (Node 2).

{_THRESHOLD_CONTEXT}

Your task:
1. Assign "algorithm_choice" (0–10): Did the model choose an appropriate algorithm for the problem?
   - 0–3: Wrong approach (e.g. brute force where DP is required, O(n²) where O(n) exists)
   - 4–6: Correct approach but suboptimal
   - 7–10: Appropriate algorithm, well-suited to constraints and input size
2. Assign "error_handling" (0–10): Does the code handle exceptions and edge cases?
3. List specific "failure_reasons" — reference actual metric values and failing test cases
4. Write a NEW, COMPLETE "improved_system_prompt" that replaces the previous one entirely.

Rules for improved_system_prompt:
  - It must be SELF-CONTAINED: assume the model has seen NO previous instructions
  - Reference SPECIFIC failing test cases: "Your code returns X for input Y, expected Z"
  - Reference SPECIFIC metric violations: "CC is 14, reduce branching in the main loop"
  - Include at least 3 concrete, actionable instructions
  - Do NOT reference "previous attempts" or "last iteration"
  - End with: "Output only Python code inside ```python ... ``` markers."

Respond ONLY with valid JSON. No text outside the JSON object.

Schema:
{{
  "algorithm_choice": <int 0-10>,
  "error_handling": <int 0-10>,
  "failure_reasons": ["<specific reason>", ...],
  "improved_system_prompt": "<complete self-contained instructions>"
}}"""


def judgement_node(state: PipelineState) -> PipelineState:
    """
    Node 3: Metrikleri ve başarısız test case'leri değerlendirir.

    Giriş  (state'ten okur): node1_*, node2_*, stripped_code, problem
                              (generated_code değil — yorumlar kaldırılmış versiyon)
    Çıktı  (state'e yazar):  node3_total_score, node3_pass,
                              node3_failure_reasons, node3_improved_prompt
    """
    numeric = compute_numeric_scores(
        pass_rate=state.get("node2_pass_rate", 0.0),
        cpu_ms=state.get("node2_cpu_ms", 0.0),
        ram_mb=state.get("node2_ram_mb", 0.0),
        mi=state.get("node1_mi", 100.0),
        cc=state.get("node1_cc", 0.0),
        bandit_high=state.get("node1_bandit_high", False),
    )

    llm_scores = _get_llm_judgement(
        numeric_scores=numeric,
        code=state["stripped_code"],   # comments stripped — prevents bias
        problem=state["problem"],
        node1_issues=state.get("node1_issues", []),
        failed_cases=state.get("node2_failed_cases", []),
        difficulty=state["difficulty"],
        iteration=state["iteration"],
    )

    total = round(
        numeric["numeric_total"]
        + llm_scores["algorithm_choice"]
        + llm_scores["error_handling"],
        2,
    )

    from pipeline.gates import PASS_THRESHOLDS
    threshold = PASS_THRESHOLDS[state["difficulty"]]
    passed = total >= threshold

    return {
        **state,
        "node3_total_score":      total,
        "node3_pass":             passed,
        "node3_failure_reasons":  llm_scores["failure_reasons"],
        # Completely rewritten — previous prompt is not inherited
        "node3_improved_prompt":  llm_scores["improved_system_prompt"],
    }


# ── Prompt construction ─────────────────────────────────────────────────────

def _get_llm_judgement(
    numeric_scores: dict,
    code: str,
    problem: str,
    node1_issues: list,
    failed_cases: list,
    halstead: float,
    timed_out: bool,
    difficulty: str,
    iteration: int,
) -> dict:
    user_prompt = _build_judgement_prompt(
        numeric_scores=numeric_scores,
        code=code,
        problem=problem,
        node1_issues=node1_issues,
        failed_cases=failed_cases,
        halstead=halstead,
        timed_out=timed_out,
        difficulty=difficulty,
        iteration=iteration,
    )

    try:
        cfg    = model_registry.get(JUDGEMENT_MODEL)
        client = get_client(JUDGEMENT_MODEL)

        response = client.chat.completions.create(
            model=cfg.model_name,
            messages=[
                {"role": "system", "content": JUDGEMENT_SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=cfg.temperature,
            max_tokens=3072,
        )
        message = response.choices[0].message
        # Reasoning models (e.g. qwen3.5-9b) emit chain-of-thought in a
        # separate reasoning_content field and can leave `content` empty
        # if max_tokens is exhausted during "thinking" — fall back to
        # reasoning_content so _parse_judgement has something to work with.
        raw = message.content or getattr(message, "reasoning_content", None) or ""
        result = _parse_judgement(raw)

        if hasattr(response, "usage") and response.usage:
            result["_judgement_prompt_tokens"]     = response.usage.prompt_tokens
            result["_judgement_completion_tokens"] = response.usage.completion_tokens
        else:
            result.setdefault("_judgement_prompt_tokens",     0)
            result.setdefault("_judgement_completion_tokens", 0)

        return result

    except Exception as e:
        # 2026-09-30: this used to swallow the exception silently. A scan of
        # the pilot run found 5/30 iterations (~17%, all with 0 token usage
        # recorded — i.e. the request itself failed, not JSON parsing) hit
        # this fallback, which hardcodes algorithm_choice/error_handling to
        # the scale midpoint. Now logged via loguru so the cause (context
        # overflow, connection error, etc.) is diagnosable instead of only
        # visible as an opaque "judgement_llm_error" in the output JSON.
        logger.warning(
            "Judge LLM call failed — using rule-based fallback | "
            "question_id={} model={} iter={} difficulty={} | {}: {}",
            question_id, JUDGEMENT_MODEL, iteration, difficulty,
            type(e).__name__, str(e)[:200],
        )
        return {
            "algorithm_choice":       5,
            "error_handling":         5,
            "failure_reasons":        ["judgement_llm_error"],
            "improved_system_prompt": _fallback_prompt(node1_issues, failed_cases, timed_out),
        }


def _build_judgement_prompt(
    numeric_scores: dict,
    code: str,
    problem: str,
    node1_issues: list,
    failed_cases: list,
    halstead: float,
    timed_out: bool,
    difficulty: str,
    iteration: int,
) -> str:
    """
    Builds the full user prompt sent to the Judge model.
    Threshold context is in the system prompt — not repeated here.
    """
    pass_rate = numeric_scores["accuracy"] / 25

    _pr_target = {
        "easy":   thresholds.node2.pass_rate_easy,
        "medium": thresholds.node2.pass_rate_medium,
        "hard":   thresholds.node2.pass_rate_hard,
    }.get(difficulty, thresholds.node2.pass_rate_hard)

    lines = [
        f"## Iteration {iteration} | Difficulty: {difficulty.upper()}",
        f"> Pass rate target for {difficulty.upper()}: ≥ {_pr_target:.0%} is Good. "
        f"Current pass rate: {pass_rate:.0%}",
        "",
        "## Problem",
        problem[:800],
        "",
        "## Generated Code",
        "```python",
        code[:8000],
        "```",
        "",
        "## Node 1 — Static Analysis",
        f"- Code quality score: {numeric_scores['quality']:.1f}/20",
        f"- Halstead Volume:    {halstead:.1f}",
    ]

    if node1_issues:
        lines.append("- Issues:")
        for issue in node1_issues:
            lines.append(f"  • {issue}")
    else:
        lines.append("- No issues detected.")

    timed_out_str = "YES — solution exceeded time limit" if timed_out else "No"
    lines += [
        "",
        "## Node 2 — Execution Results",
        f"- Test pass rate:   {pass_rate:.0%}  →  accuracy score: {numeric_scores['accuracy']}/25",
        f"- Efficiency score: {numeric_scores['efficiency']}/20",
        f"- Security score:   {numeric_scores['security']}/15",
        f"- Numeric subtotal: {numeric_scores['numeric_total']}/80",
        f"- Timed out:        {timed_out_str}",
    ]

    if failed_cases:
        lines += ["", "## Failing Test Cases (up to 5)"]
        for i, case in enumerate(failed_cases, 1):
            lines += [
                f"\nCase {i}:",
                f"  Input:    {case['input']}",
                f"  Expected: {case['expected']}",
                f"  Got:      {case['got'] or '(no output)'}",
            ]
            if case.get("error"):
                lines.append(f"  Error:    {case['error'][:200]}")
    else:
        lines += ["", "## Failing Test Cases", "None — all visible tests passed."]

    lines += [
        "",
        "Evaluate algorithm_choice (0–10) and error_handling (0–10).",
        "Write a NEW, COMPLETE improved_system_prompt addressing the specific issues above.",
        "Respond with JSON only.",
    ]

    return "\n".join(lines)


def _fallback_prompt(
    node1_issues: list,
    failed_cases: list,
    timed_out: bool = False,
) -> str:
    """
    LLM hatası durumunda kural tabanlı temel prompt üretir.
    Judge modeli yetersiz kalırsa pipeline yine de devam eder.
    """
    parts = [
        "You are an expert Python programmer solving a competitive programming problem.",
        "Write a correct, efficient, and clean Python solution.",
        "",
        "Requirements:",
        "- Use only standard library imports",
        "- Handle edge cases (empty input, negative numbers, maximum values)",
        "- Output only Python code inside ```python ... ``` markers",
        "",
    ]

    if node1_issues:
        parts.append("Fix the following code quality issues:")
        for issue in node1_issues[:3]:
            parts.append(f"  - {issue}")
        parts.append("")

    if failed_cases:
        # 2026-09-30: bound each field. LCB inputs can be large arrays —
        # unbounded reprs on 2-3 failed cases could push this whole fallback
        # prompt past judgement_node's _MAX_IMPROVED_PROMPT_CHARS (2000),
        # which then truncated it mid-sentence before reaching codegen.
        _trim = lambda v: repr(v)[:150]
        parts.append("Your previous solution failed these test cases:")
        for case in failed_cases[:3]:
            parts.append(
                f"  - Input {_trim(case['input'])} → "
                f"expected {_trim(case['expected'])}, got {_trim(case['got'])}"
            )
        parts.append("")

    parts.append("Write a completely new solution that passes all test cases.")
    return "\n".join(parts)


def _find_json_objects(text: str) -> list:
    """
    Brace-depth taraması ile tam { ... } alt dizilerini bulur; string literal
    içindeki süslü parantezleri saymaz.
    """
    objects = []
    depth = 0
    start = None
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    objects.append(text[start:i + 1])
    return objects


def _parse_judgement(raw: str) -> dict:
    """
    LLM yanıtından JSON çıkarır. Sırasıyla dener:
      1. ``` ``` fence içindeki { ... }
      2. Brace-depth taramasıyla bulunan tam nesneler (son bulunandan başlar)
      3. Ham metnin tamamı
    """
    candidates = []

    match = re.search(r"```\w*\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if match:
        candidates.append(match.group(1).strip())

    candidates.extend(reversed(_find_json_objects(raw)))
    candidates.append(raw.strip())

    for json_str in candidates:
        if not json_str:
            continue
        try:
            data = json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            continue
        return {
            "algorithm_choice":       max(0, min(10, int(data.get("algorithm_choice", 5)))),
            "error_handling":         max(0, min(10, int(data.get("error_handling", 5)))),
            "failure_reasons":        data.get("failure_reasons", []),
            "improved_system_prompt": data.get("improved_system_prompt", ""),
        }

    return {
        "algorithm_choice":       5,
        "error_handling":         5,
        "failure_reasons":        ["json_parse_error"],
        "improved_system_prompt": "",
    }
```

**2026-09-30 düzeltme — greedy regex bug:** Judge prompt'u, değerlendirilen
Python kodunun tamamını (dict/set/f-string süslü parantezleri dahil) JSON
isteminden ÖNCE gömüyor. Eski `re.search(r"\{.*\}", raw, re.DOTALL)` açgözlü
eşleşmesi, metindeki İLK `{`'den SON `}`'ye kadar her şeyi yakalıyordu —
model kodu tekrar/alıntı ettiğinde (reasoning model olduğu için sık
oluyor) bu, gerçek JSON çıktısını da içine alan ama kendisi geçersiz
(parse edilemeyen) dev bir blok üretiyordu. 100-problem pilot koşusunda
iterasyonların ~%26'sında (`outputs/results/2026-09-30`, run
`run_20260930_000556`) bu `json_parse_error`'a düşüyor, her seferinde
`improved_system_prompt` boş dönüyor ve codegen modeli o iterasyonda
hiçbir spesifik geri bildirim almadan tekrar deniyordu. Brace-depth
taraması bu sorunu ortadan kaldırıyor: yalnızca tam, dengeli `{...}`
nesnelerini aday olarak alır ve en son bulunandan (muhtemel nihai cevap)
başlayarak dener.

---

## 5. models/client.py — Judge Model Registration

Add to the `_MODEL_CONFIG` dict in `models/client.py`:

```python
_MODEL_CONFIG: dict[str, tuple[str, str, float]] = {
    "M1":   (os.getenv("M1_BASE_URL",   "http://localhost:8001/v1"), "kimi-k2.5",    0.6),
    "M2":   (os.getenv("M2_BASE_URL",   "http://localhost:8002/v1"), "glm-4.7-flash", 0.7),
    "M3":   (os.getenv("M3_BASE_URL",   "http://localhost:8003/v1"), "minimax-m2.5",  0.7),
    "QWEN": (os.getenv("QWEN_BASE_URL", "http://localhost:8004/v1"), "qwen3.5",       0.3),
    #                                                                              ↑
    #                              Low temperature for judgement → consistent scores
}
```

`.env`'e ekle:

```bash
QWEN_BASE_URL=http://localhost:8004/v1
```

`scripts/start_servers.sh`'a ekle:

```bash
start_qwen() {
    tmux new-window -t main -n "QWEN-server"
    tmux send-keys -t main:QWEN-server \
        "/workspace/llama.cpp/llama-server \
        --model /workspace/models/qwen3.5/*.gguf \
        --port 8004 --host 0.0.0.0 --ctx-size 8192 \
        --n-gpu-layers 99" Enter
}
```

---

## 6. models/prompts.py — Prompt Biriktirme Yok

```python
def build_system_prompt(improved_prompt: str | None = None) -> str:
    """
    Her iterasyonda TAMAMEN YENİ bir system prompt döner.

    improved_prompt varsa doğrudan kullanılır — BASE_SYSTEM_PROMPT ile
    BİRLEŞTİRİLMEZ. Bu sayede:
      - Çelişkili talimat birikimi önlenir (semantic drift)
      - Her iterasyon temiz bir bağlamla başlar
      - Judge, improved_prompt'u zaten self-contained yazmakla yükümlü

    improved_prompt yoksa (ilk iterasyon) BASE_SYSTEM_PROMPT kullanılır.
    """
    if not improved_prompt:
        return BASE_SYSTEM_PROMPT

    # Previous BASE_SYSTEM_PROMPT is NOT inherited
    return improved_prompt
```

---

## 7. Verification

### Node 3 Testi (QWEN server ayaktayken)

```bash
python -c "
from pipeline.state import PipelineState
from nodes.judgement import judgement_node

state: PipelineState = {
    'active_model': 'M1', 'iteration': 2, 'max_iterations': 10,
    'problem': 'Return indices of two numbers that add up to target.',
    'difficulty': 'medium',
    'test_inputs': ['2 7 11 15\n9', '3 2 4\n6'],
    'test_outputs': ['0 1', '1 2'],
    'generated_code': '''
def two_sum(nums, target):
    for i in range(len(nums)):
        for j in range(i+1, len(nums)):
            if nums[i]+nums[j]==target: return [i,j]
''',
    'node1_cc': 4.0, 'node1_mi': 78.0, 'node1_halstead': 210.0,
    'node1_bandit_high': False, 'node1_severity': 'OK',
    'node1_issues': [],
    'node2_pass_rate': 0.5, 'node2_cpu_ms': 1800.0,
    'node2_ram_mb': 90.0, 'node2_test_diversity_score': 0.5,
    'node2_failed_cases': [{
        'input': '3 2 4\n6',
        'expected': '1 2',
        'got': '0 2',
        'error': None,
    }],
    'node3_improved_prompt': '',
    'memory_log': [], 'final_status': 'PENDING',
}

result = judgement_node(state)
print('Total score:    ', result['node3_total_score'])
print('Pass:           ', result['node3_pass'])
print('Failure reasons:', result['node3_failure_reasons'])
print()
print('--- Improved prompt (ilk 400 karakter) ---')
print(result['node3_improved_prompt'][:400])
"
```

### Expected Behavior

```
Total score:     62.4
Pass:            False
Failure reasons: ['Test case failed: input 3 2 4 / 6 → expected 1 2, got 0 2', ...]

--- Improved prompt ---
You are an expert Python programmer solving a competitive programming problem.
...
Your previous solution failed: input [3,2,4] target 6 → expected [1,2], got [0,2].
The issue is likely an off-by-one error. Use a hash map for O(n) solution.
...
```

### Verify No Prompt Accumulation

```bash
python -c "
from models.prompts import build_system_prompt

p1 = build_system_prompt(None)
p2 = build_system_prompt('Fix edge case. Use hash map.')
p3 = build_system_prompt('Handle empty array. O(n) solution required.')

# p2 and p3 must not contain BASE_SYSTEM_PROMPT
assert 'expert Python programmer' not in p2, 'HATA: birikim var'
assert 'expert Python programmer' not in p3, 'HATA: birikim var'
assert p2 != p3, 'HATA: farklı promptlar aynı çıktı veriyor'
print('Birikim yok: OK')
print('p2 == improved_prompt_2:', p2 == 'Fix edge case. Use hash map.')
"
```
