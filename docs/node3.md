# node3.md — Judgement LLM (Node 3)

## 1. Purpose

Synthesize Node 1 and Node 2 metrics — including failing test case details —
to produce a 100-point score, make a PASS/FAIL decision, and on failure
write a **fully self-contained** improved system prompt for the next iteration.

### Design Decisions

| Decision | Rationale |
|-------|---------|
| Judge independent of the generator (main experiment: different model family) | Reduces self-enhancement bias risk [Zheng et al., 2023]; the local runs used the same model, a stated limitation |
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
100-point rubric computation — numeric dimensions.
Algorithm Choice (10p) and Error Handling (10p) are filled by the Judge model.
"""
from config import thresholds


def compute_accuracy_score(pass_rate: float) -> float:
    """Accuracy score: 0–25 points."""
    return round(pass_rate * 25, 2)


def compute_efficiency_score(cpu_ms: float, ram_mb: float) -> float:
    """Efficiency score: 0–20 points.
    Limits sourced from config.thresholds.node2 (cpu_limit_ms, ram_limit_mb)
    so this formula can't silently desync from the threshold table shown to the judge.
    """
    n2 = thresholds.node2
    cpu_score = max(0.0, 1 - cpu_ms / n2.cpu_limit_ms)
    ram_score = max(0.0, 1 - ram_mb / n2.ram_limit_mb)
    return round((cpu_score + ram_score) / 2 * 20, 2)


def compute_quality_score(mi: float, cc: float) -> float:
    """Code quality score: 0–20 points."""
    mi_score = min(1.0, mi / 100)
    cc_score = max(0.0, 1 - cc / thresholds.node1.cc_hard_limit)
    return round((mi_score + cc_score) / 2 * 20, 2)


def compute_security_score(high_count: int, medium_count: int) -> float:
    """Security score: 0–15 points."""
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
    executed: bool = True,
    parsed: bool = True,
) -> dict:
    """
    Computes the 4 numerically calculable dimensions.

    A dimension that could not be measured scores 0, not full marks:
      executed=False (sandbox skipped) → efficiency 0, since cpu_ms/ram_mb
        are 0 only because nothing ran;
      parsed=False (SyntaxError) → quality and security 0, since radon and
        bandit fail on unparseable code and their fallbacks (MI=100, CC=0,
        no findings) would otherwise award the maximum.

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
    eff  = compute_efficiency_score(cpu_ms, ram_mb) if executed else 0.0
    qual = compute_quality_score(mi, cc) if parsed else 0.0
    sec  = compute_security_score(
        high_count=1 if bandit_high else 0,
        medium_count=bandit_medium_count,
    ) if parsed else 0.0
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

### 4.1 Robustness Updates

- **Triple-quoted fields.** The judge sometimes writes `"improved_system_prompt": """..."""`
  (Python style). `_parse_judgement` first tries every balanced `{...}` candidate on the raw text,
  then on a repaired copy where `re.sub(r'"""(.*?)"""', lambda m: json.dumps(m.group(1)), raw, flags=re.DOTALL)`
  turns those blocks into valid JSON strings. `json.loads(..., strict=False)` also tolerates raw
  newlines inside strings; non-dict results are skipped.
- **Context overflow.** Failing-case `input` / `expected` / `got` fields are passed through
  `_clip(value, limit=200)` in `_build_judgement_prompt`; stress-test inputs of several thousand
  characters previously pushed the judge prompt past the 8192-token window.
- **`max_tokens=4096`** for the judge call (was 3072), because the reasoning model spends part of
  the budget on `reasoning_content`.
- **Error logging.** A failed judge call logs
  `question_id`, model, iteration and difficulty with the exception type before falling back to
  `_fallback_prompt` (whose reprs are trimmed to 150 chars).
- **Unmeasured dimensions score 0.** `judgement_node` calls
  `compute_numeric_scores(..., executed=state["node2_total_count"] > 0, parsed=_parses(code))`:
  code that never reached the sandbox gets 0 efficiency, and code that does not parse (`ast.parse`)
  gets 0 quality and 0 security instead of the defaults derived from empty metrics.

---

### 4.2 Judge Verdict — Final Decision (`JUDGE_DECISION=verdict`, default)

**Why.** Measured on the first 50 problems of the 150-problem sample (rubric-threshold
decision, `run_20260930_222527` + `run_20261001_070529`): of the 45 iterations that passed every
public test, the rubric rejected **none**, yet 7 of them failed the hidden tests. The judge's
own 20 points did not separate correct from wrong code (mean 14.2 vs 13.9). With 80 of 100 points
deterministic, the LLM judge effectively never decided anything. The project's premise is
LLM-as-a-Judge, so the judge must own the final decision.

**Design.** Deterministic evidence stays (it protects against hallucinated acceptance), but it
becomes *evidence in the judge's prompt*, not the decision:

| Layer | Role |
|---|---|
| Public tests (Node 2) | Precondition: all must pass (`_public_precondition`; not applied to variant V2, which has no execution) |
| Metrics + rubric total (Node 1/2/3) | Evidence shown to the judge; still logged and used by the trend extension |
| Judge verdict | Final decision: `ACCEPT` / `REJECT` + `verdict_reason` |

`_decide_pass(state, total, threshold, verdict)` returns `(passed, verdict_source)`:

- `judge` — `passed = precondition and verdict == "ACCEPT"`
- `fallback_rubric` — judge gave no usable verdict (LLM error / unparseable JSON):
  `passed = precondition and total >= threshold`, so the run never stalls
- `rubric_mode` — `JUDGE_DECISION=rubric` (the earlier design, kept for the ablation)

Prompt rules for the verdict (system prompt, item 5): test results and metrics are facts and must
not be contradicted; any failing visible test ⇒ `REJECT`; if all visible tests pass, `ACCEPT`
unless the judge can name a concrete defect the tests miss (an input allowed by the constraints
that breaks the code, or a complexity that cannot meet the constraints); style/metrics alone are
not grounds for rejection; a `REJECT` must cite the input/constraint, and the improved prompt must
describe it. `_problem_excerpt` keeps the statement head **and** its Constraints section (the old
800-char cut usually dropped the constraints the judge needs).

`node3_pass` is now the single PASS decision: `iteration_gate` and `log_iteration_node` no longer
re-check `total_score >= threshold` (that would override a judge `ACCEPT`).

New state / JSON fields: `node3_verdict`, `node3_verdict_reason`, `node3_verdict_source`
(`node3.verdict`, `node3.verdict_reason`, `node3.verdict_source` in the iteration JSON).
`max_tokens` comes from `settings.judge_max_tokens` (`JUDGE_MAX_TOKENS`, default 4096).

### 4.3 Hosted judge and no-code rule (`ENV=openrouter`, 2026-10-02)

The prompt now forbids code or pseudo-code in `improved_system_prompt` (defects and required
changes in words; naming a technique or target complexity is allowed). Reason: in the main
experiment the judge (Qwen2.5-72B) might know a solution the generator misses, and gains must
come from feedback, not from the judge writing the solution. Leakage is still measured (code
fragments in judge prompts) — `docs/ablation.md` §7. Local runs before this date used the
prompt without this rule.


The judge call goes through `models/client.chat()`. With `ENV=openrouter` the judge is
`qwen/qwen-2.5-72b-instruct` pinned to Novita bf16 (`JUDGE_MAX_TOKENS=4096`; the endpoint allows
32,000 context / 8,192 output tokens) — a different model family than the DeepSeek-V3 generator, to
avoid self-preference bias. It is weaker than the generator on LCB (28.7 vs 37.6, DeepSeek-V3
report, arXiv:2412.19437, 2024-08..11 window), which keeps the judge from simply solving the problem
for it. Sampling follows its `generation_config.json` (T 0.7, top_p 0.8, top_k 20, rep. pen. 1.05). The secondary setting (Qwen3.5-9B + gpt-oss-120b) is still selectable. Prompt, schema and decision rule are
unchanged. A `BudgetExceeded` error is re-raised rather than turned into the rule-based fallback,
so a capped run stops instead of recording made-up judge scores.

### 4.4 Evidence framing and correctness-first feedback (2026-10-02)

A probe on two problems with the main pair (DeepSeek-V3 → Qwen2.5-72B judge) showed the judge
rejecting code that passed every visible test for metric reasons only ("CC 14", "missing type
annotation" for `char_counts`), although the system prompt says metrics alone are not grounds for
REJECT. The user prompt pushed it there: Node 1 findings were headed "Issues (with required
fixes)", and the closing line asked for feedback "addressing the specific issues above". The
resulting feedback was mostly style fixes; on problem 3771 the next attempts dropped from 95.3% to
65.1% of hidden tests.

Changes (prompt only; rubric, schema and decision rule unchanged):
- Node 1 findings are headed "Quality findings (evidence; not grounds for REJECT on their own)".
- The closing instruction puts correctness first (failing tests, inputs where the code is wrong,
  crashes or is too slow); quality findings follow briefly; no requests for annotations or renaming.
- The rule that applies to the current evidence is restated at the end: visible failures → REJECT;
  all visible tests passed → REJECT only with a concrete input or an infeasible complexity.
- mypy runs with `--disable-error-code var-annotated` (`docs/node1.md`).

Re-run on 3771, 3778, 3688 and abc395_b: every REJECT of public-passing code named an input
(one of them wrong — the judge's counterexample was itself incorrect), and no attempt was rejected
for metrics alone. The judge also accepted one hard solution that passes 66.7% of hidden tests;
these errors are what the judge-accuracy analysis measures (`docs/analysis.md` §10).
Rationale for focusing feedback on the defect: in self-repair, the quality of the feedback is the
bottleneck (Olausson et al., "Is Self-Repair a Silver Bullet for Code Generation?", ICLR 2024,
arXiv:2306.09896).

## 5. Judge model registration

The judge is the registry entry named by `JUDGE_MODEL` (default `QWEN`) in `config/models.py`;
`nodes/judgement.py` calls it through `models/client.chat()`. Per environment:

| `ENV` | `QWEN` entry | Temperature |
|---|---|---|
| `openrouter` | `qwen/qwen-2.5-72b-instruct`, Novita bf16 | 0.7 (top_p 0.8, top_k 20) |
| `local` | `qwen3.5-9b` on :8001 (same server as the generator) | 0.6 |
| `cloud` | `qwen3.5-35b-a3b` on :8004, or `JUDGE_MODEL=M1` | 0.3 / 0.6 |

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
