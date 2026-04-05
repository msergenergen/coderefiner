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
    # Testing
    "unittest",
})
