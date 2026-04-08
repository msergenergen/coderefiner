"""
LangGraph Node 3: Judgement LLM (Judge)

Critical design decisions:
  1. improved_prompt is COMPLETELY REWRITTEN every iteration — no accumulation.
  2. Threshold table is embedded in the system prompt.
  3. Failing test cases (input/expected/got) are included verbatim in the prompt.
  4. Judgement model (Judge) is independent of the code generation models (M1/M2/M3).
"""
import json
import re
import time
from pipeline.state import PipelineState
from metrics.scoring import compute_numeric_scores
from models.client import get_client, get_model_name, get_temperature
from config import model_registry, thresholds

# ── Rule-based fix hints for common static analysis issues ───────────────────
# Each entry: (compiled regex, fix string or callable(match) -> str)
_ISSUE_HINTS: list[tuple] = [
    # mypy: undefined name from typing (e.g. List, Dict, Tuple, Optional)
    (re.compile(r'Name "([A-Za-z]+)" is not defined'),
     lambda m: f'Add `from typing import {m.group(1)}` as the first import line'),
    # mypy: operand with None — uninitialized variable
    (re.compile(r'Unsupported operand types.*None'),
     lambda _: 'Initialize the variable to 0 or an empty collection before the loop, not None'),
    # ruff F821 — same root as mypy name-defined
    (re.compile(r'\[F821\].*Undefined name `([A-Za-z]+)`'),
     lambda m: f'Add `from typing import {m.group(1)}` — `{m.group(1)}` is not a builtin in Python 3.9+'),
    # ruff W293 — trailing whitespace on blank lines
    (re.compile(r'\[W293\]'),
     lambda _: 'Remove trailing whitespace from blank lines (run: sed -i "s/[[:space:]]*$//" your_file.py)'),
    # ruff E501 — line too long
    (re.compile(r'\[E501\].*[Ll]ine too long \((\d+)'),
     lambda m: f'Line is {m.group(1)} chars — break it into two lines (max 88)'),
    # ruff E302/E303 — missing blank lines
    (re.compile(r'\[E30[23]\]'),
     lambda _: 'Add the required blank lines between top-level definitions'),
    # mypy return type mismatch
    (re.compile(r'Incompatible return value type'),
     lambda _: 'Return type does not match the annotation — check your return statement'),
]


def _hint_for_issue(issue: str) -> str | None:
    """Returns a concrete fix suggestion for a known issue pattern, or None."""
    for pattern, fix in _ISSUE_HINTS:
        m = pattern.search(issue)
        if m:
            return fix(m) if callable(fix) else fix
    return None

JUDGEMENT_MODEL = "QWEN"

JUDGEMENT_SYSTEM_PROMPT = f"""You are a strict code quality judge for an iterative code refinement pipeline.

You evaluate Python solutions to competitive programming problems using static analysis
results (Node 1) and sandbox execution results (Node 2).

{thresholds.judgement_context_table()}

Your task:
1. Assign "algorithm_choice" (0–10): Did the model choose an appropriate algorithm for the problem?
   - 0–3: Wrong approach (e.g. brute force where DP is required, O(n²) where O(n) exists)
   - 4–6: Correct approach but suboptimal (unnecessary complexity, missed edge cases in design)
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
    Node 3: Evaluates metrics and failing test cases.

    Reads from state: node1_*, node2_*, stripped_code, problem
                      (not generated_code — comment-stripped version)
    Writes to state:  node3_total_score, node3_pass,
                      node3_failure_reasons, node3_improved_prompt
    """
    t0 = time.perf_counter()

    numeric = compute_numeric_scores(
        pass_rate=state.get("node2_pass_rate", 0.0),
        cpu_ms=state.get("node2_cpu_ms", 0.0),
        ram_mb=state.get("node2_ram_mb", 0.0),
        mi=state.get("node1_mi", 100.0),
        cc=state.get("node1_cc", 0.0),
        bandit_high=state.get("node1_bandit_high", False),
        bandit_medium_count=state.get("node1_bandit_medium_count", 0),
    )

    llm_scores = _get_llm_judgement(
        numeric_scores=numeric,
        code=state.get("stripped_code", state.get("generated_code", "")),
        problem=state["problem"],
        node1_issues=state.get("node1_issues", []),
        failed_cases=state.get("node2_failed_cases", []),
        halstead=state.get("node1_halstead", 0.0),
        timed_out=state.get("node2_timed_out", False),
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

    node3_ms = int((time.perf_counter() - t0) * 1000)
    timing = dict(state.get("timing", {}))
    timing["node3_ms"] = node3_ms
    timing["total_ms"] = sum(
        timing.get(k, 0)
        for k in ("codegen_ms", "node1_ms", "node2_ms", "node3_ms")
    )

    # Write judgement token counts to state
    tokens = dict(state.get("tokens", {}))
    tokens["judgement_prompt"]     = llm_scores.pop("_judgement_prompt_tokens",     0)
    tokens["judgement_completion"] = llm_scores.pop("_judgement_completion_tokens", 0)
    tokens["total"] = (
        tokens.get("codegen_prompt", 0)
        + tokens.get("codegen_completion", 0)
        + tokens["judgement_prompt"]
        + tokens["judgement_completion"]
    )

    # Build score breakdown for logging
    score_breakdown = {
        "accuracy":         numeric["accuracy"],
        "efficiency":       numeric["efficiency"],
        "quality":          numeric["quality"],
        "security":         numeric["security"],
        "algorithm_choice": llm_scores["algorithm_choice"],
        "error_handling":   llm_scores["error_handling"],
    }

    return {
        **state,
        "node3_total_score":      total,
        "node3_pass":             passed,
        "node3_score_breakdown":  score_breakdown,
        "node3_failure_reasons":  llm_scores["failure_reasons"],
        # Completely rewritten — previous prompt is not inherited
        "node3_improved_prompt":  llm_scores["improved_system_prompt"],
        "timing":                 timing,
        "tokens":                 tokens,
    }


# ── Prompt construction ───────────────────────────────────────────────────────

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
            max_tokens=1024,
        )
        raw = response.choices[0].message.content
        result = _parse_judgement(raw)

        # Store token counts so judgement_node can write them to state
        if hasattr(response, "usage") and response.usage:
            result["_judgement_prompt_tokens"]     = response.usage.prompt_tokens
            result["_judgement_completion_tokens"] = response.usage.completion_tokens
        else:
            result.setdefault("_judgement_prompt_tokens",     0)
            result.setdefault("_judgement_completion_tokens", 0)

        return result

    except Exception:
        # LLM error → do not stop pipeline, use rule-based fallback
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
        lines.append("- Issues (with required fixes):")
        seen_hints: set[str] = set()
        for issue in node1_issues:
            lines.append(f"  • {issue}")
            hint = _hint_for_issue(issue)
            if hint and hint not in seen_hints:
                lines.append(f"    → Fix: {hint}")
                seen_hints.add(hint)
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
    Generates a rule-based basic prompt on LLM error.
    Keeps the pipeline running even if the Judge model fails.
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
        parts.append("Your previous solution failed these test cases:")
        for case in failed_cases[:3]:
            parts.append(
                f"  - Input {case['input']!r} → "
                f"expected {case['expected']!r}, got {case['got']!r}"
            )
        parts.append("")

    if timed_out:
        parts.append("Your previous solution exceeded the time limit.")
        parts.append("  - Use a more efficient algorithm (avoid nested loops on large inputs)")
        parts.append("")

    parts.append("Write a completely new solution that passes all test cases.")
    return "\n".join(parts)


def _parse_judgement(raw: str) -> dict:
    """
    Extracts JSON from the LLM response.

    Tries in order:
      1. ``` ... ``` fenced block (any language tag: json, python, or none)
      2. First { ... } JSON object in the raw text
      3. Full raw text stripped
    """
    json_str = None

    # 1. Any fenced block — skip optional language identifier (json / python / etc.)
    match = re.search(r"```\w*\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if match:
        json_str = match.group(1).strip()

    # 2. Largest { ... } block in the raw text (handles plain JSON with no fences)
    if not json_str:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            json_str = match.group(0).strip()

    # 3. Fallback: use the full text as-is
    if not json_str:
        json_str = raw.strip()

    try:
        data = json.loads(json_str)
        return {
            "algorithm_choice":       max(0, min(10, int(data.get("algorithm_choice", 5)))),
            "error_handling":         max(0, min(10, int(data.get("error_handling", 5)))),
            "failure_reasons":        data.get("failure_reasons", []),
            "improved_system_prompt": data.get("improved_system_prompt", ""),
        }
    except (json.JSONDecodeError, ValueError):
        return {
            "algorithm_choice":       5,
            "error_handling":         5,
            "failure_reasons":        ["json_parse_error"],
            "improved_system_prompt": "",
        }
