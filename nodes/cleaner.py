"""
Comment stripper before judgement.

Why here:
  - Sandbox already ran generated_code — test results belong to the real code
  - Judgement LLM must not be biased by comments:
      # This is a very efficient algorithm → misleads scoring
      # TODO: can be improved → unnecessary noise
  - stripped_code is only for judgement; log and memory.md always store generated_code

Method:
  Python's tokenize module is used instead of AST parsing because:
    - It also processes code with syntax errors (LLM sometimes produces broken code)
    - It preserves docstrings (these are API contracts, not comments)
    - It trims trailing whitespace from lines

  Comments are cut from the source lines at the column tokenize reports, so
  line structure is kept 1:1 (the earlier token-joining version doubled every
  newline and left the comment's indentation behind; fixed 2026-10-02).
"""
import io
import re
import tokenize
from pipeline.state import PipelineState


def strip_comments(code: str) -> str:
    """
    Removes comment lines and inline comments from Python code.
    Docstrings (triple-quote) are preserved.
    Returns the original code if a syntax error is encountered.

    Examples:
        "x = 1  # value"  →  "x = 1"
        "# header\ndef f(): ..."  →  "def f(): ..."
        '\"\"\"docstring\"\"\"'  →  preserved
    """
    try:
        # Column of the first comment token on each line (1-based rows).
        # Only real COMMENT tokens count: a "#" inside a string is a STRING
        # token, and docstrings are STRING tokens, so both are preserved.
        comment_col = {}
        for tok in tokenize.generate_tokens(io.StringIO(code).readline):
            if tok.type == tokenize.COMMENT:
                comment_col.setdefault(tok.start[0], tok.start[1])

        out = []
        for row, line in enumerate(code.splitlines(), 1):
            if row in comment_col:
                kept = line[:comment_col[row]].rstrip()
                if not kept:
                    continue            # comment-only line: drop it entirely
                out.append(kept)
            else:
                out.append(line.rstrip())

        cleaned = "\n".join(out)
        # Collapse 3+ consecutive newlines (2+ blank lines) to one blank line
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    except (tokenize.TokenError, SyntaxError, IndentationError):
        # Malformed code: tokenize.generate_tokens() raises TokenError for
        # things like an unterminated string, but a genuinely broken
        # indentation level (e.g. truncated LLM output cut off mid-block)
        # raises IndentationError/SyntaxError instead — both must be caught
        # here, not just TokenError, or a single malformed generation kills
        # the whole pipeline run instead of just skipping comment-stripping
        # for that one iteration. Confirmed: this exact gap crashed a live
        # 150-problem run (IndentationError: "unindent does not match any
        # outer indentation level") once syntax-error code started reaching
        # this node via the LOW-severity path (see nodes/static_analysis.py).
        return code


def strip_comments_node(state: PipelineState) -> PipelineState:
    """
    LangGraph node: runs between sandbox and judgement.

    generated_code  → unchanged (sandbox ran it; log stores this code)
    stripped_code   → sent to judgement LLM
    """
    cleaned = strip_comments(state["generated_code"])
    return {**state, "stripped_code": cleaned}
