"""
System prompt templates for codegen and judgement models.

Codegen prompt consists of four layers:
  1. CODEGEN_IDENTITY        — who it is
  2. CODEGEN_CONTEXT         — where it is in the pipeline
  3. CODEGEN_CONSTRAINTS     — behavior rules
  4. CODEGEN_FEEDBACK_RULES  — how to use feedback (only on iteration > 1)

First iteration:  layers 1 + 2 + 3
Subsequent:       layers 1 + 2 + 3 + 4 + improved_prompt
"""
from config import settings
from sandbox.whitelist import ALLOWED_IMPORTS

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

  1. Static Analyzer  — checks complexity (CC, MI), security (Bandit) and imports
  2. Sandbox Executor — runs your code on the example tests shown in the problem
  3. Judge LLM        — reviews this evidence and accepts or rejects the solution

If the judge rejects your solution, you will receive specific feedback and be \
asked to produce a new one (at most """ + str(settings.max_iterations) + """ attempts in total). \
An accepted solution is finally graded on hidden tests, so it must be correct \
for every input the constraints allow, not only for the examples.\
"""

# ── Layer 3: Behavior rules ───────────────────────────────────────────────────

CODEGEN_CONSTRAINTS = """\
Rules you must follow in every solution:

Output format:
- Output ONLY Python code inside ```python ... ``` markers
- Do not include any text, explanation, or comments outside the markers

I/O format:
- If the problem gives Starter Code (a class with a method signature),
  implement exactly that class and method: it is called directly with the
  test arguments and its return value is checked, so do not read stdin
  and do not print the result
- Otherwise read all input from stdin using input() or sys.stdin and
  write all output to stdout using print()
- Do not include any print() calls for debugging or examples

Correctness:
- Handle every input the constraints allow: minimum and maximum sizes,
  boundary values, duplicates and the special cases named in the statement
- Input always follows the stated format and constraints
- Choose an algorithm whose time complexity fits the largest inputs

Security (enforced by static analyzer):
- Allowed imports only: """ + ", ".join(sorted(m for m in ALLOWED_IMPORTS if m != "unittest")) + """
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


# Used when FEEDBACK_INCLUDE_CODE=true (default since 2026-10-02): the previous
# solution is shown with the feedback, as in LiveCodeBench's self-repair
# scenario (Jain et al. 2024, App. C.3: question + code to be corrected +
# feedback) and Olausson et al. (ICLR 2024). Before that, the feedback said
# "your code returns X for input Y" about code the generator could not see.
CODEGEN_REPAIR_RULES = """\
You are receiving improvement feedback on your previous solution, which is
shown after the problem statement.

How to use this feedback:
- Read every point carefully — each one comes from the judge, who saw your code,
  the static analysis and the execution results on the example tests
- Write a COMPLETE corrected solution: keep what the feedback does not criticise,
  change what it identifies; if it says the approach is wrong, use a different one
- Address EVERY correctness issue mentioned — skipping one will cause the same failure

What NOT to do:
- Do not resubmit the previous solution unchanged
- Do not special-case the example inputs or add print() statements for them
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
        CODEGEN_REPAIR_RULES if settings.feedback_include_code else CODEGEN_FEEDBACK_RULES,
        f"## Improvement Notes from Previous Attempt\n\n{improved_prompt}",
    ])


def build_user_prompt(formatted_problem: str, previous_code: str = None) -> str:
    """
    Converts an LCB question to user prompt format.
    Does not repeat the system prompt — only provides the problem and, on a
    repair iteration with FEEDBACK_INCLUDE_CODE=true, the previous solution.
    """
    if previous_code and settings.feedback_include_code:
        return (
            f"Solve the following problem:\n\n{formatted_problem}\n\n"
            f"## Your Previous Solution (rejected)\n```python\n{previous_code[:8000]}\n```\n\n"
            "Write your complete corrected Python solution:"
        )
    return (
        f"Solve the following problem:\n\n{formatted_problem}\n\n"
        "Write your complete Python solution:"
    )
