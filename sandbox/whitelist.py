"""
Import whitelist for Node 1 security check.
"""

ALLOWED_IMPORTS = frozenset({
    # Stdlib — core
    "os", "sys", "math", "random", "time", "datetime",
    "collections", "itertools", "functools", "heapq",
    "bisect", "string", "re", "json", "copy",
    "typing", "dataclasses", "enum", "abc",
    # Stdlib — I/O and format
    "io", "textwrap", "struct", "decimal", "fractions",
    "statistics", "cmath", "numbers",
    # Stdlib — pure computation common in competitive programming; no file,
    # network or process access (added 2026-10-02 after an OpenRouter smoke
    # test stopped a problem on `import operator`; none occurred locally).
    "operator", "array", "queue", "graphlib", "__future__",
    # Testing
    "unittest",
})
