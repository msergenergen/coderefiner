"""
System prompt templates for codegen and judgment models.

Codegen prompt consists of four layers:
  1. CODEGEN_IDENTITY        — who it is
  2. CODEGEN_CONTEXT         — where it is in the pipeline
  3. CODEGEN_CONSTRAINTS     — behavior rules
  4. CODEGEN_FEEDBACK_RULES  — how to use feedback (only on iteration > 1)

First iteration:  layers 1 + 2 + 3
Subsequent:       layers 1 + 2 + 3 + 4 + improved_prompt
"""

# ── Layer 1: Identity ─────────────────────────────────────────────────────────

CODEGEN_IDENTITY = """\
You are an expert Python programmer specializing in competitive programming.
Your sole responsibility in this pipeline is to produce correct, efficient,
and secure Python solutions to algorithmic problems.\
"""

# ── Layer 2: Pipeline context ─────────────────────────────────────────────────

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

# ── Layer 3: Behavior rules ───────────────────────────────────────────────────

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

# ── Layer 4: Feedback integration (only on iteration > 1) ────────────────────

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


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_system_prompt(improved_prompt: str = None) -> str:
    """
    Builds the codegen system prompt from four layers.

    Args:
        improved_prompt: Improvement instructions produced by Node 3.
                         None on the first iteration — feedback layer is not added.

    Returns:
        Full system prompt text.

    Layer structure:
        First iteration      → IDENTITY + CONTEXT + CONSTRAINTS
        Subsequent iterations → IDENTITY + CONTEXT + CONSTRAINTS
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
    Converts an LCB question to user prompt format.
    Does not repeat the system prompt — only provides the problem.
    """
    return (
        f"Solve the following problem:\n\n{formatted_problem}\n\n"
        "Write your complete Python solution:"
    )
