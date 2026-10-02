"""
LangGraph Node 3: Judgement LLM (Judge)

Critical design decisions:
  1. improved_prompt is COMPLETELY REWRITTEN every iteration — no accumulation.
  2. Threshold table is embedded in the system prompt.
  3. Failing test cases (input/expected/got) are included verbatim in the prompt.
  4. Judgement model (Judge) is independent of the code generation models (M1/M2).
"""
import ast
import json
import re
import time
from loguru import logger
from pipeline.state import PipelineState
from metrics.scoring import compute_numeric_scores
from models.budget import BudgetExceeded
from models.client import chat
from config import model_registry, settings, thresholds

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

# Configurable via .env: JUDGE_MODEL=QWEN (default)
# Change to any registry key (M1/M2/QWEN) to swap the judge.
JUDGEMENT_MODEL: str = settings.judge_model

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
5. Decide the "verdict": "ACCEPT" or "REJECT". You are the final authority on whether
   this solution is delivered; the pipeline stops on ACCEPT and asks for a new attempt on REJECT.
   - The test results and metrics above are measured facts. Use them as evidence and never
     contradict them (do not claim a passing test fails or vice versa).
   - If any visible test failed or the code did not run, the verdict MUST be REJECT.
   - If all visible tests passed, ACCEPT unless you can name a concrete defect the visible
     tests do not cover: an input allowed by the stated constraints on which the code returns
     a wrong answer or crashes (empty / minimum / maximum sizes, duplicates, boundary values,
     special cases in the statement), or a time complexity that cannot meet the constraints.
   - Style, readability or metric scores alone are NOT grounds for REJECT.
   - "verdict_reason": one or two sentences. A REJECT must name the specific input or
     constraint; an ACCEPT should state why the visible evidence is sufficient.
   - On REJECT after all visible tests passed, the improved_system_prompt must describe the
     defect and the input that exposes it.

Rules for improved_system_prompt:
  - It must be SELF-CONTAINED: assume the model has seen NO previous instructions
  - Reference SPECIFIC failing test cases: "Your code returns X for input Y, expected Z"
  - Reference SPECIFIC metric violations: "CC is 14, reduce branching in the main loop"
  - Include at least 3 concrete, actionable instructions
  - Do NOT reference "previous attempts" or "last iteration"
  - Do NOT write code or pseudo-code (no code blocks, no function bodies, no
    line-by-line algorithm). Describe defects and required changes in words;
    you may name a technique or a target complexity.
  - End with: "Output only Python code inside ```python ... ``` markers."

Respond ONLY with valid JSON. No text outside the JSON object.

Schema:
{{
  "algorithm_choice": <int 0-10>,
  "error_handling": <int 0-10>,
  "failure_reasons": ["<specific reason>", ...],
  "improved_system_prompt": "<complete self-contained instructions>",
  "verdict": "ACCEPT" | "REJECT",
  "verdict_reason": "<one or two sentences>"
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
        executed=state.get("node2_total_count", 0) > 0,
        parsed=_parses(state.get("generated_code", "")),
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
        question_id=state.get("question_id", "?"),
        run_id=state.get("run_id"),
    )

    total = round(
        numeric["numeric_total"]
        + llm_scores["algorithm_choice"]
        + llm_scores["error_handling"],
        2,
    )

    from pipeline.gates import PASS_THRESHOLDS
    threshold = PASS_THRESHOLDS[state["difficulty"]]
    passed, verdict_source = _decide_pass(
        state, total=total, threshold=threshold, verdict=llm_scores.get("verdict", ""))

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

    # Cap improved_prompt to prevent codegen context window growth over iterations.
    # Judge max_tokens=1024 can produce ~800 tokens (~3200 chars) per iteration;
    # without a cap this compounds and inflates codegen prompt tokens on every loop.
    _MAX_IMPROVED_PROMPT_CHARS = 2000
    improved_prompt = llm_scores["improved_system_prompt"]
    if len(improved_prompt) > _MAX_IMPROVED_PROMPT_CHARS:
        improved_prompt = improved_prompt[:_MAX_IMPROVED_PROMPT_CHARS]

    return {
        **state,
        "node3_total_score":      total,
        "node3_pass":             passed,
        "node3_score_breakdown":  score_breakdown,
        "node3_failure_reasons":  llm_scores["failure_reasons"],
        # Completely rewritten — previous prompt is not inherited
        "node3_improved_prompt":  improved_prompt,
        "node3_verdict":          llm_scores.get("verdict", ""),
        "node3_verdict_reason":   llm_scores.get("verdict_reason", ""),
        "node3_verdict_source":   verdict_source,
        "timing":                 timing,
        "tokens":                 tokens,
    }


def _public_precondition(state: PipelineState) -> bool:
    """Deterministic guard against hallucinated acceptance: every public test
    must have passed. Variant V2 (no execution) has no test evidence, so the
    guard does not apply there. Partial pass rates used to clear the rubric
    threshold on quality/security/judge points alone, stopping the loop on
    code already known to be wrong."""
    if state.get("variant", "V3") == "V2":
        return True
    return (state.get("node2_total_count", 0) > 0
            and state.get("node2_pass_rate", 0.0) >= 1.0)


def _decide_pass(state: PipelineState, total: float, threshold: float,
                 verdict: str) -> tuple[bool, str]:
    """Returns (passed, verdict_source).

    verdict mode: public-test precondition AND the judge's ACCEPT; if the judge
                  gave no usable verdict (LLM error / unparseable JSON), fall
                  back to the rubric threshold so the run never stalls.
    rubric mode:  public-test precondition AND rubric total >= threshold.
    """
    precondition = _public_precondition(state)
    if settings.judge_decision == "rubric":
        return precondition and total >= threshold, "rubric_mode"
    if verdict in ("ACCEPT", "REJECT"):
        return precondition and verdict == "ACCEPT", "judge"
    return precondition and total >= threshold, "fallback_rubric"


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
    question_id: str = "?",
    run_id: str | None = None,
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
        res = chat(
            JUDGEMENT_MODEL,
            [
                {"role": "system", "content": JUDGEMENT_SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=settings.judge_max_tokens,
            tag={"question_id": question_id, "iteration": iteration, "run_id": run_id},
        )
        # Reasoning models can leave `content` empty if max_tokens is
        # exhausted during "thinking" — chat() then returns the reasoning
        # so _parse_judgement has something to work with.
        result = _parse_judgement(res.text)

        # Store token counts so judgement_node can write them to state
        result["_judgement_prompt_tokens"]     = res.prompt_tokens
        result["_judgement_completion_tokens"] = res.completion_tokens

        return result

    except BudgetExceeded:
        raise   # spending cap reached: stop the run, never fake a verdict
    except Exception as e:
        # LLM error → do not stop pipeline, use rule-based fallback.
        # Logged (not silently swallowed) because this fallback hardcodes
        # algorithm_choice/error_handling to the scale midpoint (5/10) — if it
        # fires often for a specific problem, that problem's score is a rule-
        # based approximation, not a real judge verdict. A prior scan found
        # this fired on 5/30 pilot iterations (~17%), all with 0 token usage
        # recorded — i.e. the request itself failed (e.g. context overflow
        # or connection error), not a JSON-parsing failure — and the previous
        # bare `except Exception:` gave no way to tell which.
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
        f"> Visible tests passed: {pass_rate:.0%}. ACCEPT requires 100% "
        f"(rubric 'Good' band for {difficulty.upper()}: ≥ {_pr_target:.0%}, scoring only).",
        "",
        "## Problem",
        _problem_excerpt(problem),
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
        lines.append("- Quality findings (evidence; not grounds for REJECT on their own):")
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
            # LCB stress-test inputs can be thousands of characters; untrimmed,
            # a single case pushed the judge prompt past the 8192-token server
            # context (observed: 9935 tokens on problem 3620 → HTTP 400 →
            # rule-based fallback instead of a real judgement).
            lines += [
                f"\nCase {i}:",
                f"  Input:    {_clip(case['input'])}",
                f"  Expected: {_clip(case['expected'])}",
                f"  Got:      {_clip(case['got']) or '(no output)'}",
            ]
            if case.get("error"):
                lines.append(f"  Error:    {case['error'][:200]}")
    else:
        lines += ["", "## Failing Test Cases", "None — all visible tests passed."]

    lines += [
        "",
        "Evaluate algorithm_choice (0–10) and error_handling (0–10).",
        "Write a NEW, COMPLETE improved_system_prompt. Correctness comes first: failing tests,",
        "and any input on which the code would be wrong, crash or too slow. Quality findings",
        "come after that, briefly; do not ask for type annotations or renaming.",
    ]
    if failed_cases or pass_rate < 1.0:
        lines.append("Visible tests failed, so the verdict must be REJECT.")
    else:
        lines.append(
            "All visible tests passed. REJECT only if you can name a concrete input allowed by "
            "the constraints on which this code is wrong or crashes, or a time complexity that "
            "cannot meet the constraints; quality findings alone mean ACCEPT."
        )
    lines += [
        "Give your verdict (ACCEPT / REJECT) with verdict_reason, following the rules.",
        "Respond with JSON only.",
    ]

    return "\n".join(lines)


def _problem_excerpt(problem: str, head: int = 800, tail: int = 700) -> str:
    """The judge needs the constraints to assess complexity and edge cases, but
    LCB statements put them at the end. Keep the head and the constraints
    section instead of a plain 800-char cut (8192-token context)."""
    if len(problem) <= head + tail:
        return problem
    idx = problem.lower().rfind("constraints")
    if idx == -1 or idx < head:
        return problem[:head + tail]
    return problem[:head] + "\n[...]\n" + problem[idx:idx + tail]


def _parses(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except (SyntaxError, ValueError):
        return False


def _clip(value, limit: int = 200) -> str:
    text = str(value) if value is not None else ""
    if len(text) <= limit:
        return text
    return f"{text[:limit]}… [{len(text) - limit} more chars truncated]"


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
        # Bound each field — LCB inputs can be large arrays; without this,
        # 2-3 failed cases can push the whole fallback prompt past
        # _MAX_IMPROVED_PROMPT_CHARS, which then truncates it mid-sentence.
        _trim = lambda v: repr(v)[:150]
        parts.append("Your previous solution failed these test cases:")
        for case in failed_cases[:3]:
            parts.append(
                f"  - Input {_trim(case['input'])} → "
                f"expected {_trim(case['expected'])}, got {_trim(case['got'])}"
            )
        parts.append("")

    if timed_out:
        parts.append("Your previous solution exceeded the time limit.")
        parts.append("  - Use a more efficient algorithm (avoid nested loops on large inputs)")
        parts.append("")

    parts.append("Write a completely new solution that passes all test cases.")
    return "\n".join(parts)


def _find_json_objects(text: str) -> list:
    """
    Scans for complete top-level {...} substrings by brace-depth, honoring
    string literals so braces inside quoted strings don't affect nesting.

    Needed because the judge prompt embeds the submitted Python source (which
    itself contains {}/dict/set/f-string braces) ahead of the verdict. A naive
    greedy `\\{.*\\}` regex spans from the first brace anywhere in the text to
    the last one, swallowing the quoted code and producing unparseable JSON
    even when the model's actual verdict object was well-formed on its own.
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
    Extracts JSON from the LLM response.

    Tries in order:
      1. ``` ... ``` fenced block (any language tag: json, python, or none)
      2. Complete top-level {...} objects found by brace-depth scanning,
         tried last-to-first (the verdict is expected to follow any
         reasoning/code the model echoes ahead of it)
      3. Full raw text stripped
    """
    # The judge often writes improved_system_prompt as a Python triple-quoted
    # string ("""...""") instead of a JSON string; observed on every
    # non-truncated json_parse_error in a 141-call sample. Re-encode those
    # spans as JSON strings and retry before giving up.
    repaired = re.sub(r'"""(.*?)"""', lambda m: json.dumps(m.group(1)), raw, flags=re.DOTALL)

    candidates = []
    for text in ([raw, repaired] if repaired != raw else [raw]):
        # 1. Any fenced block — skip optional language identifier (json / python / etc.)
        match = re.search(r"```\w*\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            candidates.append(match.group(1).strip())
        # 2. Balanced {...} objects, most-likely-final-answer first
        candidates.extend(reversed(_find_json_objects(text)))
        # 3. Fallback: use the full text as-is
        candidates.append(text.strip())

    for json_str in candidates:
        if not json_str:
            continue
        try:
            # strict=False tolerates raw newlines/tabs inside strings
            data = json.loads(json_str, strict=False)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        verdict = str(data.get("verdict", "")).strip().upper()
        return {
            "algorithm_choice":       max(0, min(10, int(data.get("algorithm_choice", 5)))),
            "error_handling":         max(0, min(10, int(data.get("error_handling", 5)))),
            "failure_reasons":        data.get("failure_reasons", []),
            "improved_system_prompt": data.get("improved_system_prompt", ""),
            "verdict":                verdict if verdict in ("ACCEPT", "REJECT") else "",
            "verdict_reason":         str(data.get("verdict_reason", ""))[:500],
        }

    return {
        "algorithm_choice":       5,
        "error_handling":         5,
        "failure_reasons":        ["json_parse_error"],
        "improved_system_prompt": "",
    }
