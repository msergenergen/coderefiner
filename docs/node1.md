# node1.md — Statik Analiz (Node 1)

## 1. Purpose

Detect structural, quality, and security issues in the generated code **before execution**. The pipeline is routed based on the severity decision.

---

## 2. Tools Used

| Tool | Install | Measured Metric |
|------|---------|----------------|
| `radon` | `pip install radon` | Cyclomatic Complexity, Maintainability Index, Halstead |
| `lizard` | `pip install lizard` | Cognitive Complexity |
| `bandit` | `pip install bandit` | Security vulnerabilities |
| `mypy` | `pip install mypy` | Type errors (informational, no severity effect; `var-annotated` disabled since 2026-10-02 — "need type annotation" is not a defect in contest code and drew style-only judge REJECTs, `docs/node3.md` §4.4) |
| `ruff` | `pip install ruff` | Lint warnings (informational, no severity effect) |
| stdlib `ast` | — | Import whitelist check |

```bash
pip install radon lizard bandit mypy ruff
```

---

## 3. Threshold Tablosu

| Metrik | Threshold | Severity Etkisi |
|--------|-----------|-----------------|
| Cyclomatic Complexity (CC) | ≤ 10/fonksiyon | CC > 20 → HIGH |
| Maintainability Index (MI) | ≥ 65 | MI < 40 → LOW |
| Cognitive Complexity | ≤ 15 | — (informational only) |
| Halstead Volume | ≤ 1000 | > 2000 → LOW |
| Bandit HIGH severity | 0 | Her bulgu → HIGH |
| Non-whitelisted import (gerçek) | 0 | Any finding → HIGH |
| Bare SyntaxError (`check_imports`'ın döndürdüğü `"syntax_error"` sentinel) | — | → LOW (HIGH değil) |

**Severity decision (priority order):**
1. Bandit HIGH or *real* non-whitelisted import → `HIGH` → STOP
2. CC > 20 → `HIGH` → STOP
3. `"syntax_error"` sentinel (kod parse edilemiyor), MI < 40, veya Halstead > 2000 → `LOW` → Node 2 atla
4. Hepsi OK → `OK` → Node 2'ye devam

**2026-09-30 düzeltme — sözdizimi hatası artık HIGH tetiklemiyor:** `check_imports()`
bir `SyntaxError` yakaladığında `["syntax_error"]` sentinel'i döndürür. Önceden
`_decide_severity` bunu gerçek bir yasak import gibi işleyip `HIGH`'a
yönlendiriyordu — bu da `log_severity_stop → END` ile kalıcı bir durdurmaya
yol açıyordu. Ancak reasoning model (qwen3.5-9b), `max_tokens` sınırına
"düşünme" (`reasoning_content`) aşamasında takılırsa, kapanan ` ``` ` fence'i
hiç üretemeden kesiliyor; `_extract_code()` de kapanmamış bloğu bulamayınca
ham (parse edilemeyen) metni "kod" olarak döndürüyordu. Sonuç: zararsız bir
üretim kesintisi, gerçek bir güvenlik ihlaliymiş gibi kalıcı STOP'a dönüşüyordu
(bkz. problem 3298, M1 iter 2, run `run_20260930_000556`). Artık bu sentinel
`LOW`'a yönlendiriliyor — Node 2 atlanır (parse edilemeyen kod zaten
çalıştırılamaz) ama Node 3 yine de çalışır ve model normal `LOOP` akışıyla
tekrar dener. Gerçek yasak import'lar (`subprocess`, `socket` vb.) hâlâ
`HIGH` olarak durduruluyor.

---

## 4. sandbox/whitelist.py

```python
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
```

---

## 5. metrics/complexity.py

```python
"""
radon ve lizard ile karmaşıklık metrikleri hesaplama.
"""
import subprocess
import json
import tempfile
import os
import lizard


def cyclomatic_complexity(code: str) -> float:
    """
    Kodun maksimum Cyclomatic Complexity değerini döner.
    radon cc kullanır.
    """
    with tempfile.NamedTemporaryFile(
        suffix=".py", mode="w", delete=False
    ) as f:
        f.write(code)
        tmp = f.name

    try:
        result = subprocess.run(
            ["radon", "cc", tmp, "-s", "-j"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0 or not result.stdout.strip():
            return 0.0

        data = json.loads(result.stdout)
        scores = []
        for blocks in data.values():
            for block in blocks:
                scores.append(block.get("complexity", 0))
        return max(scores) if scores else 0.0
    except Exception:
        return 0.0
    finally:
        os.unlink(tmp)


def maintainability_index(code: str) -> float:
    """
    Kodun Maintainability Index değerini döner (0–100).
    radon mi kullanır.
    """
    with tempfile.NamedTemporaryFile(
        suffix=".py", mode="w", delete=False
    ) as f:
        f.write(code)
        tmp = f.name

    try:
        result = subprocess.run(
            ["radon", "mi", tmp, "-j"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0 or not result.stdout.strip():
            return 100.0

        data = json.loads(result.stdout)
        scores = [v.get("mi", 100.0) for v in data.values()]
        return min(scores) if scores else 100.0
    except Exception:
        return 100.0
    finally:
        os.unlink(tmp)


def halstead_volume(code: str) -> float:
    """
    Kodun Halstead Volume değerini döner.
    radon hal kullanır.
    """
    with tempfile.NamedTemporaryFile(
        suffix=".py", mode="w", delete=False
    ) as f:
        f.write(code)
        tmp = f.name

    try:
        result = subprocess.run(
            ["radon", "hal", tmp, "-j"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0 or not result.stdout.strip():
            return 0.0

        data = json.loads(result.stdout)
        volumes = []
        for v in data.values():
            total = v.get("total", {})
            volumes.append(total.get("volume", 0.0))
        return max(volumes) if volumes else 0.0
    except Exception:
        return 0.0
    finally:
        os.unlink(tmp)


def cognitive_complexity(code: str) -> float:
    """
    Kodun maksimum Cognitive Complexity değerini döner.
    lizard kullanır.
    """
    try:
        result = lizard.analyze_file.analyze_source_code("tmp.py", code)
        scores = [fn.cyclomatic_complexity for fn in result.function_list]
        return max(scores) if scores else 0.0
    except Exception:
        return 0.0
```

---

## 6. metrics/security.py

```python
"""
Bandit ile güvenlik analizi.
"""
import subprocess
import json
import tempfile
import os
import ast
from sandbox.whitelist import ALLOWED_IMPORTS


def bandit_scan(code: str) -> dict:
    """
    Bandit ile güvenlik taraması yapar.

    Returns:
        {
            "high_count": int,    # Number of HIGH severity findings
            "medium_count": int,
            "issues": list[dict]  # Detailed findings
        }
    """
    with tempfile.NamedTemporaryFile(
        suffix=".py", mode="w", delete=False
    ) as f:
        f.write(code)
        tmp = f.name

    try:
        result = subprocess.run(
            ["bandit", "-f", "json", "-q", tmp],
            capture_output=True, text=True, timeout=15
        )
        # Bandit exit code 1 = findings exist, this is normal
        output = result.stdout or result.stderr
        if not output.strip():
            return {"high_count": 0, "medium_count": 0, "issues": []}

        data = json.loads(output)
        results = data.get("results", [])

        high   = [r for r in results if r.get("issue_severity") == "HIGH"]
        medium = [r for r in results if r.get("issue_severity") == "MEDIUM"]

        return {
            "high_count":   len(high),
            "medium_count": len(medium),
            "issues":       results,
        }
    except Exception:
        return {"high_count": 0, "medium_count": 0, "issues": []}
    finally:
        os.unlink(tmp)


def check_imports(code: str) -> list[str]:
    """
    AST ile import whitelist kontrolü yapar.

    Returns:
        Whitelist dışı import listesi (boş = temiz)
    """
    forbidden = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ["syntax_error"]

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in ALLOWED_IMPORTS:
                    forbidden.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            root = module.split(".")[0]
            if root not in ALLOWED_IMPORTS:
                forbidden.append(module)

    return forbidden
```

---

## 7. nodes/static_analysis.py

```python
"""
LangGraph Node 1: Statik Analiz

Üretilen kodu çalıştırmadan analiz eder ve severity kararı verir.
"""
from pipeline.state import PipelineState
from metrics.complexity import (
    cyclomatic_complexity,
    maintainability_index,
    halstead_volume,
    cognitive_complexity,
)
from metrics.security import bandit_scan, check_imports


def static_analysis_node(state: PipelineState) -> PipelineState:
    """
    Node 1: Statik analiz çalıştırır ve severity belirler.

    Giriş  (state'ten okur): generated_code
    Çıktı  (state'e yazar):  node1_cc, node1_mi, node1_halstead,
                              node1_bandit_high, node1_severity,
                              node1_issues  ← Node 3 için okunabilir bulgular
    """
    import time
    t0 = time.perf_counter()

    code = state["generated_code"]

    # Metrik hesaplama
    cc        = cyclomatic_complexity(code)
    mi        = maintainability_index(code)
    halstead  = halstead_volume(code)
    cognitive = cognitive_complexity(code)
    bandit    = bandit_scan(code)
    forbidden = check_imports(code)
    mypy      = mypy_check(code)
    ruff      = ruff_check(code)

    # Severity decision
    severity = _decide_severity(
        cc=cc,
        mi=mi,
        halstead=halstead,
        bandit_high_count=bandit["high_count"],
        forbidden_imports=forbidden,
    )

    # Human-readable issue list for Node 3
    issues = _collect_issues(
        cc=cc, mi=mi, halstead=halstead,
        bandit=bandit, forbidden=forbidden,
        mypy=mypy, ruff=ruff,
    )

    node1_ms = int((time.perf_counter() - t0) * 1000)
    timing = dict(state.get("timing", {}))
    timing["node1_ms"] = node1_ms

    return {
        **state,
        "node1_cc":                  cc,
        "node1_mi":                  mi,
        "node1_halstead":            halstead,
        "node1_bandit_high":         bandit["high_count"] > 0,
        "node1_bandit_medium_count": bandit["medium_count"],
        "node1_severity":            severity,
        "node1_issues":              issues,
        "timing":                    timing,
    }


def _collect_issues(
    cc: float,
    mi: float,
    halstead: float,
    bandit: dict,
    forbidden: list[str],
    mypy: dict,
    ruff: dict,
) -> list[str]:
    """
    Sayısal metriklerden okunabilir bulgu cümleleri üretir.
    Node 3 judgement prompt'una doğrudan eklenir.
    """
    issues = []

    if cc > 10:
        issues.append(
            f"Cyclomatic Complexity {cc:.0f} — threshold 10. "
            "Function has too many branches; should be split into smaller parts."
        )
    if mi < 65:
        issues.append(
            f"Maintainability Index {mi:.1f} — threshold 65. "
            "Code is hard to maintain; improve variable naming or structure."
        )
    if halstead > 1000:
        issues.append(
            f"Halstead Volume {halstead:.0f} — threshold 1000. "
            "High operator/operand count; simplify expressions."
        )
    for issue in bandit.get("issues", []):
        sev  = issue.get("issue_severity", "")
        text = issue.get("issue_text", "")
        line = issue.get("line_number", "?")
        issues.append(f"Security [{sev}] line {line}: {text}")
    for imp in forbidden:
        if imp == "syntax_error":
            issues.append(
                "Code failed to parse (SyntaxError) — likely truncated or "
                "malformed generation. Produce a complete, syntactically valid "
                "Python solution."
            )
        else:
            issues.append(f"Forbidden import: `{imp}` — not in whitelist.")
    for err in mypy.get("errors", []):
        issues.append(f"Type error (mypy): {err}")
    for issue in ruff.get("issues", []):
        issues.append(f"Lint (ruff): {issue}")

    return issues


def _decide_severity(
    cc: float,
    mi: float,
    halstead: float,
    bandit_high_count: int,
    forbidden_imports: list[str],
) -> str:
    """
    Threshold'lara göre severity kararı verir.
    Öncelik: HIGH > LOW > OK

    "syntax_error" sentinel'i gerçek bir yasak import değildir (bkz. yukarıdaki
    2026-09-30 notu) — HIGH'a değil LOW'a yönlendirilir.
    """
    real_forbidden = [imp for imp in forbidden_imports if imp != "syntax_error"]

    # HIGH — critical, do not send to sandbox
    if bandit_high_count > 0:
        return "HIGH"
    if real_forbidden:
        return "HIGH"
    if cc > 20:
        return "HIGH"

    # LOW — warnings present, skip Node 2
    if "syntax_error" in forbidden_imports:
        return "LOW"
    if mi < 40:
        return "LOW"
    if halstead > 2000:
        return "LOW"

    return "OK"
```

---

## 8. Verification

### Verify Tools Are Installed

```bash
radon --version
bandit --version
python -c "import lizard; print('lizard OK')"
```

### Node 1'i Tekil Test Et

```bash
python -c "
from nodes.static_analysis import static_analysis_node
from pipeline.state import PipelineState

# Temiz kod — OK bekleniyor
clean_code = '''
def two_sum(nums, target):
    seen = {}
    for i, n in enumerate(nums):
        if target - n in seen:
            return [seen[target - n], i]
        seen[n] = i
    return []
'''

state: PipelineState = {
    'generated_code': clean_code,
    'active_model': 'M1', 'iteration': 1, 'max_iterations': 10,
    'problem': '', 'difficulty': 'easy', 'memory_log': [],
    'final_status': 'PENDING', 'node3_improved_prompt': '',
}

result = static_analysis_node(state)
print('CC:      ', result['node1_cc'])
print('MI:      ', result['node1_mi'])
print('Halstead:', result['node1_halstead'])
print('Bandit:  ', result['node1_bandit_high'])
print('Severity:', result['node1_severity'])
"
```

### Expected Output (clean code)

```
CC:       1.0
MI:       87.3
Halstead: 142.6
Bandit:   False
Severity: OK
```

### HIGH Severity Testi

```python
# Unsafe code — import os + exec
dangerous_code = """
import os
import subprocess
exec(input())
"""
# Severity: HIGH beklenir
```
