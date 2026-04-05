"""
Comment stripper before judgment.

Why here:
  - Sandbox already ran generated_code — test results belong to the real code
  - Judgment LLM must not be biased by comments:
      # This is a very efficient algorithm → misleads scoring
      # TODO: can be improved → unnecessary noise
  - stripped_code is only for judgment; log and memory.md always store generated_code

Method:
  Python's tokenize module is used instead of AST parsing because:
    - It also processes code with syntax errors (LLM sometimes produces broken code)
    - It preserves docstrings (these are API contracts, not comments)
    - It trims trailing whitespace from lines
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
        tokens   = tokenize.generate_tokens(io.StringIO(code).readline)
        result   = []
        prev_end = (1, 0)

        for tok_type, tok_str, tok_start, tok_end, _ in tokens:
            # Skip COMMENT tokens
            if tok_type == tokenize.COMMENT:
                continue
            # Skip ENCODING and ENDMARKER tokens
            if tok_type in (tokenize.ENCODING, tokenize.ENDMARKER):
                continue

            # Preserve line/column spacing (keep indentation intact)
            start_row, start_col = tok_start
            end_row,   _         = prev_end

            if start_row > end_row:
                # New line(s) — blank lines may remain after comment removal
                result.append("\n" * (start_row - end_row))
                result.append(" " * start_col)
            elif start_col > prev_end[1]:
                result.append(" " * (start_col - prev_end[1]))

            result.append(tok_str)
            prev_end = tok_end

        cleaned = "".join(result)

        # Collapse 3+ consecutive blank lines to 2
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

        return cleaned.strip()

    except tokenize.TokenError:
        # Syntax error: return original code, do not stop pipeline
        return code


def strip_comments_node(state: PipelineState) -> PipelineState:
    """
    LangGraph node: runs between sandbox and judgment.

    generated_code  → unchanged (sandbox ran it; log stores this code)
    stripped_code   → sent to judgment LLM
    """
    cleaned = strip_comments(state["generated_code"])
    return {**state, "stripped_code": cleaned}
