# graph.md — LangGraph Graph Definition

## 1. Graph Flow

```
START
  ↓
codegen             ← timing["codegen_ms"] + tokens["codegen_*"] yazılır
  ↓
static_analysis     ← timing["node1_ms"] yazılır
  ↓
severity_gate ──── HIGH ──→ log_severity_stop_node → END
  │                              (STOP_HIGH: kısmi timing, sıfır token[judgement])
  │ LOW / OK
  ↓
sandbox             ← timing["node2_ms"] yazılır  (LOW → atlanır, 0 kalır)
  ↓
strip_comments      ← generated_code → stripped_code (yorumlar kaldırılır)
  ↓                   generated_code değişmez (sandbox + log bu kodu kullandı)
judgement            ← stripped_code üzerinden değerlendirir
  ↓                   timing["node3_ms"] + tokens["judgement_*"] yazılır
log_iteration_node  ← Her iterasyon: memory.md + JSON yazılır
  ↓                   timing["total_ms"] hesaplanır
iteration_gate
  ├─ PASS      → write_memory_final → next_model → END
  ├─ STOP_MAX  → stop_max → END
  └─ LOOP      → codegen (sonraki iterasyon)
```

**`strip_comments` neden burada?**
- Sandbox already ran `generated_code` — test results belong to the real code, must not change
- Judgement LLM must not be biased by comments: lines like `# This is very efficient` inflate scores
- `stripped_code` is for judgement only; log, memory.md, and JSON always store `generated_code`

---

## 2. pipeline/state.py

```python
"""
Pipeline'ın tüm node'ları arasında taşınan merkezi state.
"""
from typing import TypedDict, Literal, Optional


class PipelineState(TypedDict):

    # ── Run ve routing ────────────────────────────────────────────────
    run_id:         str          # Set in main.py by write_run_info()
    active_model:   Literal["M1", "M2", "M3"]
    variant:        Literal["V1", "V2", "V3"]   # Ablation variant
    iteration:      int
    max_iterations: int          # Sabit: 10

    # ── Input ─────────────────────────────────────────────────────────
    problem:        str          # LCB problem text (format_problem output)
    problem_row:    dict         # Raw row from load_lcb() — includes all_tests
    question_id:    str
    difficulty:     Literal["easy", "medium", "hard"]
    test_inputs:    list[str]    # public_tests["input"] — used in Phase 2 runtime
    test_outputs:   list[str]    # public_tests["output"]
    edge_cases:     list[dict]   # augment_row() output — output=None crash testleri

    # ── Code generation ───────────────────────────────────────────────────
    generated_code: str   # LLM output — used by sandbox and log
    stripped_code:  str   # Comment-stripped version — only judgement sees this

    # ── Node 1 outputs ──────────────────────────────────────────────
    node1_cc:                   float
    node1_mi:                   float
    node1_halstead:             float
    node1_bandit_high:          bool
    node1_bandit_medium_count:  int
    node1_severity:             Literal["LOW", "OK", "HIGH"]
    node1_issues:               list[str]

    # ── Node 2 outputs ──────────────────────────────────────────────
    # Correctness — Faz 1 (lcb_runner)
    node2_pass_rate:            float   # 0.0–1.0
    node2_passed_count:         int     # Number of passed tests
    node2_total_count:          int     # Total test count (public + private)
    node2_failed_cases:         list    # Max 5, {input, expected, got, error}
    node2_crash_count:          int     # Edge case'lerde runtime error
    node2_test_diversity_score: float   # 1 - crash_count/len(edge_cases)
    node2_lcb_available:        bool    # lcb_runner kurulu mu?
    # Runtime — Faz 2 (subprocess)
    node2_cpu_ms:    float
    node2_ram_mb:    float
    node2_timed_out: bool

    # ── Node 3 outputs ──────────────────────────────────────────────
    node3_total_score:     float
    node3_pass:            bool
    node3_score_breakdown: dict       # {"accuracy": 0, "efficiency": 0, "quality": 0, "security": 0, "algorithm_choice": 0, "error_handling": 0}
    node3_failure_reasons: list[str]
    node3_improved_prompt: str        # Completely rewritten every iteration

    # ── Performance metrics (filled by each node) ─────────
    timing: dict
    # {
    #   "codegen_ms": 0,   ← codegen_node yazar
    #   "node1_ms":   0,   ← static_analysis_node yazar
    #   "node2_ms":   0,   ← written by sandbox_node (stays 0 when LOW)
    #   "node3_ms":   0,   ← judgement_node yazar
    #   "total_ms":   0    ← log_iteration_node hesaplar
    # }

    tokens: dict
    # {
    #   "codegen_prompt":      0,   ← codegen_node yazar
    #   "codegen_completion":  0,   ← codegen_node yazar
    #   "judgement_prompt":     0,   ← judgement_node yazar (STOP_HIGH'da 0)
    #   "judgement_completion": 0,   ← judgement_node yazar (STOP_HIGH'da 0)
    #   "total":               0    ← log_iteration_node hesaplar
    # }

    # ── Log ───────────────────────────────────────────────────────────
    memory_log:   list[dict]
    final_status: Literal["PASS", "STOP_HIGH", "STOP_MAX", "PENDING"]
```

---

## 3. pipeline/gates.py

```python
"""
Conditional edge fonksiyonları.
iteration_gate, log_iteration_node SONRASI çalışır —
loglama zaten tamamlandı, sadece yönlendirme kararı verilir.
"""
from config import thresholds
from pipeline.state import PipelineState

PASS_THRESHOLDS = thresholds.node3.as_dict()


def severity_gate(state: PipelineState) -> str:
    """
    Node 1 sonrası yönlendirme.

    "to_node2"  → OK  — sandbox çalışsın
    "to_node3"  → LOW — sandbox atla, direkt judgement
    "stop_high" → HIGH — log_severity_stop_node'a git
    """
    sev = state["node1_severity"]
    if sev == "HIGH":
        return "stop_high"
    elif sev == "LOW":
        return "to_node3"
    return "to_node2"


def iteration_gate(state: PipelineState) -> str:
    """
    log_iteration_node sonrası yönlendirme.
    Satır zaten memory.md ve JSON'a yazıldı.

    "write_memory_final" → PASS    — final_status güncelle, sonraki model
    "stop_max"           → STOP_MAX — final_status güncelle, END
    "loop_to_codegen"    → LOOP    — tekrar dene
    """
    threshold = PASS_THRESHOLDS[state["difficulty"]]
    passed    = state["node3_total_score"] >= threshold and state["node3_pass"]

    if passed:
        return "write_memory_final"
    elif state["iteration"] >= state["max_iterations"]:
        return "stop_max"
    return "loop_to_codegen"
```

---

## 4. pipeline/graph.py

```python
"""
LangGraph StateGraph tanımı.

Her iterasyon log_iteration_node üzerinden memory.md + JSON'a yazılır:
  LOOP      — judgement sonrası, eşik geçilemedi
  PASS      — judgement sonrası, eşik geçildi
  STOP_MAX  — judgement sonrası, max_iter doldu
  STOP_HIGH — severity HIGH, Node 2/3 çalışmadı

Her node kendi timing'ini state["timing"]'e yazar.
Codegen ve judgement node'ları token sayılarını state["tokens"]'a yazar.
"""
from langgraph.graph import StateGraph, END

from pipeline.state  import PipelineState
from pipeline.gates  import severity_gate, iteration_gate
from pipeline.router import advance_model

from nodes.codegen         import codegen_node
from nodes.static_analysis import static_analysis_node
from nodes.sandbox         import sandbox_node
from nodes.cleaner         import strip_comments_node   # ← yeni
from nodes.judgement        import judgement_node
from memory.logger         import log_iteration_node, log_severity_stop_node


def _write_memory_final_node(state: PipelineState) -> PipelineState:
    """PASS: final_status güncelle. Satır zaten log_iteration_node'da yazıldı."""
    return {**state, "final_status": "PASS"}


def _stop_max_node(state: PipelineState) -> PipelineState:
    """STOP_MAX: final_status güncelle. Satır zaten log_iteration_node'da yazıldı."""
    return {**state, "final_status": "STOP_MAX"}


def _next_model_node(state: PipelineState) -> PipelineState:
    """PASS sonrası bir sonraki modele geç."""
    return advance_model(state)


def build_graph() -> StateGraph:
    g = StateGraph(PipelineState)

    # ── Node'lar ─────────────────────────────────────────────────────
    g.add_node("codegen",             codegen_node)
    g.add_node("static_analysis",     static_analysis_node)
    g.add_node("sandbox",             sandbox_node)
    g.add_node("strip_comments",      strip_comments_node)      # ← yeni
    g.add_node("judgement",            judgement_node)
    g.add_node("log_iteration",       log_iteration_node)
    g.add_node("log_severity_stop",   log_severity_stop_node)
    g.add_node("write_memory_final",  _write_memory_final_node)
    g.add_node("stop_max",            _stop_max_node)
    g.add_node("next_model",          _next_model_node)

    # ── Sabit edge'ler ────────────────────────────────────────────────
    g.set_entry_point("codegen")

    g.add_edge("codegen",            "static_analysis")
    g.add_edge("sandbox",            "strip_comments")    # sandbox → strip → judgement
    g.add_edge("strip_comments",     "judgement")          # ← yeni
    g.add_edge("judgement",           "log_iteration")
    g.add_edge("write_memory_final", "next_model")

    # ── Conditional: Severity Gate ────────────────────────────────────
    g.add_conditional_edges(
        "static_analysis",
        severity_gate,
        {
            "to_node2":  "sandbox",
            "to_node3":  "judgement",
            "stop_high": "log_severity_stop",
        },
    )

    # ── Conditional: Iteration Gate (after log_iteration) ───────────
    g.add_conditional_edges(
        "log_iteration",
        iteration_gate,
        {
            "write_memory_final": "write_memory_final",
            "stop_max":           "stop_max",
            "loop_to_codegen":    "codegen",
        },
    )

    # ── Terminal edge'ler ─────────────────────────────────────────────
    g.add_edge("log_severity_stop", END)
    g.add_edge("stop_max",          END)
    g.add_edge("next_model",        END)

    return g.compile()
```

---

## 5. main.py

```python
"""
Pipeline giriş noktası.
Her LCB sorusu için M1→M2→M3 sırasıyla çalıştırır.
"""
from dotenv import load_dotenv
from loguru import logger

from pipeline.graph  import build_graph
from pipeline.state  import PipelineState
from config          import settings
from dataset.loader  import load_lcb
from dataset.splitter import format_problem
from memory.logger   import write_run_info

load_dotenv()

# Initial timing/token values that persist across iterations
_EMPTY_TIMING = {"codegen_ms": 0, "node1_ms": 0, "node2_ms": 0, "node3_ms": 0, "total_ms": 0}
_EMPTY_TOKENS = {"codegen_prompt": 0, "codegen_completion": 0,
                 "judgement_prompt": 0, "judgement_completion": 0, "total": 0}


def run_pipeline_for_problem(problem_row: dict, run_id: str) -> dict:
    graph = build_graph()

    test_inputs  = problem_row.get("public_tests", {}).get("input",  [])
    test_outputs = problem_row.get("public_tests", {}).get("output", [])

    model_sequence = ["M1", "M2", "M3"]
    all_results    = []

    for model_id in model_sequence:
        logger.info(f"[{model_id}] başlıyor — {problem_row['title']}")

        initial_state: PipelineState = {
            "run_id":         run_id,
            "active_model":   model_id,
            "iteration":      0,
            "max_iterations": settings.max_iterations,
            "problem":      format_problem(problem_row),
            "problem_row":  aug_row,
            "question_id":  problem_row["id"],
            "difficulty":   problem_row["difficulty"],
            "test_inputs":  pub_inputs,
            "test_outputs": pub_outputs,
            "edge_cases":   edge_cases,
            "generated_code": "",
            "stripped_code":  "",
            # Node 1
            "node1_cc":       0.0, "node1_mi": 100.0, "node1_halstead": 0.0,
            "node1_bandit_high": False, "node1_severity": "OK", "node1_issues": [],
            # Node 2
            "node2_pass_rate": 0.0, "node2_passed_count": 0, "node2_total_count": 0,
            "node2_failed_cases": [], "node2_crash_count": 0,
            "node2_lcb_available": False,
            "node2_cpu_ms": 0.0, "node2_ram_mb": 0.0, "node2_timed_out": False,
            # Node 3
            "node3_total_score": 0.0, "node3_pass": False,
            "node3_score_breakdown": {}, "node3_failure_reasons": [],
            "node3_improved_prompt": "",
            # Performans
            "timing": dict(_EMPTY_TIMING),
            "tokens": dict(_EMPTY_TOKENS),
            # Log
            "memory_log":   [],
            "final_status": "PENDING",
        }

        final_state = graph.invoke(initial_state)
        status      = final_state["final_status"]
        score       = final_state.get("node3_total_score", 0.0)

        logger.info(f"[{model_id}] tamamlandı — status={status} score={score:.1f}")
        all_results.append({"model": model_id, "status": status, "score": score})

    return {
        "problem_id": problem_row["id"],
        "difficulty": problem_row["difficulty"],
        "results":    all_results,
    }


def main():
    logger.info("Pipeline başlatılıyor...")

    # Run info — written once, returns run_id
    run_id = write_run_info()
    logger.info(f"run_id: {run_id}")

    rows = load_lcb(
        split=settings.lcb_split,
        difficulty=settings.lcb_difficulty,
        sample_size=settings.lcb_sample_size,
    )
    logger.info(f"{len(rows)} soru yüklendi.")

    for row in rows:
        result = run_pipeline_for_problem(row, run_id)
        logger.info(f"Soru tamamlandı: {result}")

    logger.info("Pipeline tamamlandı.")


if __name__ == "__main__":
    main()
```

---

## 6. Verification

### Graph Structure

```bash
python -c "
from pipeline.graph import build_graph
g = build_graph()
print('Graf derlendi.')
# 10 node beklenir:
# codegen, static_analysis, sandbox, strip_comments, judgement,
# log_iteration, log_severity_stop,
# write_memory_final, stop_max, next_model
"
```

### strip_comments_node Testi

```bash
python -c "
from nodes.cleaner import strip_comments_node

# Yorumlu kod
state = {
    'generated_code': '''
# Add two numbers
def add(a, b):
    # a ve b parametrelerini topla
    result = a + b  # sonucu hesapla
    return result   # return
'''.strip()
}

out = strip_comments_node(state)
print('--- stripped_code ---')
print(out['stripped_code'])
print('--- generated_code (değişmedi) ---')
print(out['generated_code'] == state['generated_code'])
"
```

**Expected output:**

```
--- stripped_code ---
def add(a, b):
    result = a + b
    return result

--- generated_code (değişmedi) ---
True
```

### Gate Testleri

```bash
python -c "
from pipeline.gates import severity_gate, iteration_gate

assert severity_gate({'node1_severity': 'HIGH'}) == 'stop_high'
assert severity_gate({'node1_severity': 'LOW'})  == 'to_node3'
assert severity_gate({'node1_severity': 'OK'})   == 'to_node2'
print('severity_gate: OK')

base = {'difficulty': 'easy', 'iteration': 3, 'max_iterations': 10}
assert iteration_gate({**base, 'node3_total_score': 85, 'node3_pass': True})              == 'write_memory_final'
assert iteration_gate({**base, 'node3_total_score': 60, 'node3_pass': False})             == 'loop_to_codegen'
assert iteration_gate({**base, 'node3_total_score': 60, 'node3_pass': False,
                        'iteration': 10})                                                  == 'stop_max'
print('iteration_gate: OK')
"
```

### Verify All Statuses Are Written

```bash
python -c "
import os, json
os.makedirs('outputs', exist_ok=True)
if os.path.exists('outputs/memory.md'): os.remove('outputs/memory.md')

from memory.logger import write_run_info, log_iteration_node, log_severity_stop_node, read_memory, read_iteration

run_id = write_run_info()

def s(q='q001', **kw):
    base = {
        'run_id': run_id, 'active_model': 'M1', 'iteration': 1,
        'max_iterations': 10, 'question_id': q, 'difficulty': 'easy',
        'problem': '', 'test_inputs': [], 'test_outputs': [],
        'generated_code': 'def solve(): pass',
        'node1_cc': 3.0, 'node1_mi': 72.0, 'node1_halstead': 150.0,
        'node1_issues': [], 'node1_bandit_high': False, 'node1_severity': 'OK',
        'node2_pass_rate': 0.65, 'node2_cpu_ms': 500.0,
        'node2_ram_mb': 120.0, 'node2_test_diversity_score': 0.6,
        'node2_failed_cases': [{'input':'[]','expected':'0','got':'None'}],
        'node3_total_score': 62.0, 'node3_pass': False,
        'node3_score_breakdown': {'accuracy':16.25,'efficiency':12.0,'quality':11.0,
                                   'security':15.0,'algorithm_choice':5,'error_handling':3},
        'node3_failure_reasons': ['pass_rate düşük'],
        'node3_improved_prompt': 'Boş liste kontrolü ekle.',
        'timing': {'codegen_ms':3200,'node1_ms':45,'node2_ms':1200,'node3_ms':2800,'total_ms':7245},
        'tokens': {'codegen_prompt':450,'codegen_completion':320,
                   'judgement_prompt':1200,'judgement_completion':580,'total':2550},
        'memory_log': [], 'final_status': 'PENDING',
    }
    base.update(kw)
    return base

# LOOP
log_iteration_node(s('q001', iteration=1, node3_total_score=62, node3_pass=False))
# PASS
log_iteration_node(s('q001', iteration=2, node3_total_score=85, node3_pass=True))
# STOP_MAX
log_iteration_node(s('q001', iteration=10, node3_total_score=55, node3_pass=False, max_iterations=10))
# STOP_HIGH
log_severity_stop_node(s('q002', node1_severity='HIGH',
                          timing={'codegen_ms':3100,'node1_ms':42,'node2_ms':0,'node3_ms':0,'total_ms':3142},
                          tokens={'codegen_prompt':420,'codegen_completion':310,
                                  'judgement_prompt':0,'judgement_completion':0,'total':730}))

rows = read_memory()
print('Satır sayısı:', len(rows))
print('Statuslar:   ', [r['status'] for r in rows])
print('total_ms[0]: ', rows[0]['total_ms'])
print('total_tok[0]:', rows[0]['total_tok'])

# JSON detail check
detail = read_iteration('q001', 'M1', 1)
print('improved_prompt:', detail['node3']['improved_prompt'])
print('node1_issues:   ', detail['node1']['issues'])
print('failed_cases:   ', detail['node2']['failed_cases'])

# run_info check
info = json.load(open('outputs/run_info.json'))
print('GPU:    ', info['hardware']['gpu_name'])
print('Models: ', info['models'])
"
```

### Expected Output

```
Satır sayısı: 4
Statuslar:    ['LOOP', 'PASS', 'STOP_MAX', 'STOP_HIGH']
total_ms[0]:  7245
total_tok[0]: 2550
improved_prompt: Boş liste kontrolü ekle.
node1_issues:    ['CC yüksek']
failed_cases:    [{'input': '[]', 'expected': '0', 'got': 'None'}]
GPU:     NVIDIA A100-SXM4-80GB   (local'de 'unknown' çıkabilir)
Models:  {'M1': 'gpt-oss-20b', 'M2': 'qwen3-coder-next', ...}
```

---

## 7. nodes/cleaner.py

```python
"""
Judgement öncesi yorum temizleyici.

Neden burada:
  - Sandbox generated_code'u çalıştırdı — test sonuçları gerçek koda ait
  - Judgement LLM yorumlardan bias almasın:
      # This is a very efficient algorithm → misleads scoring
      # TODO: can be improved → unnecessary noise
  - stripped_code sadece judgement için; log ve memory.md generated_code'u saklar

Yöntem:
  Python'un tokenize modülü kullanılır.
  AST parse etmek yerine tokenize tercih edilir çünkü:
    - Syntax hatası içeren kodu da işler (LLM bazen hatalı üretir)
    - Docstring'leri korur (bunlar API sözleşmesidir, yorum değildir)
    - Satır sonu boşluklarını temizler
"""
import io
import tokenize
from pipeline.state import PipelineState


def strip_comments(code: str) -> str:
    """
    Python kodundan yorum satırlarını ve inline yorumları kaldırır.
    Docstring'ler (triple-quote) korunur.
    Syntax hatası varsa orijinal kodu döner.

    Örnekler:
        "x = 1  # değer"  →  "x = 1"
        "# başlık\ndef f(): ..."  →  "def f(): ..."
        '\"\"\"docstring\"\"\"'  →  korunur
    """
    try:
        tokens      = tokenize.generate_tokens(io.StringIO(code).readline)
        result      = []
        prev_end    = (1, 0)

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
        import re
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

        return cleaned.strip()

    except tokenize.TokenError:
        # Syntax error: return original code, do not stop pipeline
        return code


def strip_comments_node(state: PipelineState) -> PipelineState:
    """
    LangGraph node: sandbox → judgement arasında çalışır.

    generated_code  → değişmez (sandbox çalıştırdı, log bu kodu saklar)
    stripped_code   → judgement LLM'e bu gönderilir
    """
    cleaned = strip_comments(state["generated_code"])
    return {**state, "stripped_code": cleaned}
```

### Behavior Examples

```python
from nodes.cleaner import strip_comments

# 1. Line-start comment
strip_comments("# header\ndef f():\n    return 1")
# → "def f():\n    return 1"

# 2. Inline yorum
strip_comments("x = a + b  # topla")
# → "x = a + b"

# 3. Docstring korunur
strip_comments('def f():\n    """Açıklama."""\n    return 1')
# → 'def f():\n    """Docstring."""\n    return 1'

# 4. Syntax error — original returned, pipeline keeps running
strip_comments("def f(\n    # missing closing paren")
# → "def f(\n    # missing closing paren"
```

### Connection to node3.md

`judgement_node` now uses `state["stripped_code"]`
instead of `state["generated_code"]`:

```python
# nodes/judgement.py — relevant line
llm_scores = _get_llm_judgement(
    code=state["stripped_code"],    # ← not generated_code
    ...
)
```

`generated_code` is always preserved in its original form:
- `memory/logger.py` → the `generated_code` field in JSON
- `nodes/sandbox.py` → the code used during test execution
- `nodes/static_analysis.py` → the code used for metric computation
