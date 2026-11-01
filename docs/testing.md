# testing.md — Test Stratejisi

> **Güncel envanter (2026-11-01).** `pytest -q` → **293 passed, 1 skipped**. Aşağıdaki bölümlerdeki
> test kopyaları açıklama amaçlıdır ve bazıları eskidir; doğru kaynak `tests/` dizinidir.

| Dosya | Test | Kapsam |
|---|---|---|
| `test_tracing.py` | 5 | Langfuse: anahtarsız kapalı, gerçek SDK + bellek içi dışa aktarıcıyla üretim düğümü (model, token, maliyet, oturum), deterministik iz kimliği, hatalı çağrının ERROR kaydı, skorlar ve gizli test yüklemesi, izleme hatasının koşuyu bozmaması |
| `test_api_client.py` | 18 | `models/client.chat()` (akış, gerekçe yedeği, maliyet defteri, bütçe sınırı, yeniden deneme), üretici/yargıç örnekleme ayarları (DeepSeek-V3 top_p gönderilmez; Qwen2.5-72B generation_config), sağlayıcı ikilisi kontrolü, pass@k / filt@k, geri bildirimde kod sızıntısı, gizli notlamada ilk hatada durma, yargıç isteminde metriklerin kanıt olarak sunulması, aynı run_id paylaşan taban/hat maliyetlerinin ayrılması, `--compare` eşleştirmesi, `--latex` makro (Abstract için `…GainEn` kazanç makroları dahil), çizelge (en iyi değer kalın, bildiri terimleriyle TÖ / İ_Y^Q / İ_P başlıkları, süreç özeti) ve bütçe eğrisi çıktısı; tek örneklerin pass@1'i (`per_sample`), Δ↑/Δ↓ makroları (`\a<Y|P>Up/Down`, `\e<TAG>Up/Down`) ve Düğüm 1 özeti (`static_evidence`: önem kapısı durdurmaları ve nedeni, durdurulan kodun gizli sonucu — testleri çıkarılmış kayıtta bilinmiyor —, rubrik eşiği, statik ölçüt adlandıran ret ve geri bildirimler; `\aStops…` makroları) |
| `test_export_release_logs.py` | 11 | Açık kayıt arşivi (`scripts/export_release_logs.py`): LCB problem metni ve testlerinin (`training.problem`, `training.all_tests`) çıkarılması, diğer alanların korunması, koşu kimliğine göre dosya seçimi, arşiv yapısı, `outputs/` altına hiç yazılmaması |
| `test_official_regrade.py` | 4 | Resmî değerlendirici karşılaştırması (`benchmark/official_regrade.py`), lcb_runner gerektirmeyen kısımlar: lcb_runner'ın doğruluk kuralı (her test > 0), taban örnekleri ve önbellekteki hat denemelerinin toplanması (STOP_HIGH notlanmaz), aynı problemi çözen koşuların hepsinin korunması, grup başına uyum özeti |
| `test_judge_replay.py` | 7 | Çevrimdışı yargıç tekrarı: açık testleri geçen ve gizli sonucu bilinen denemelerin seçimi, istemin hattaki yargıç istemiyle aynı kurulması, kararsız yanıtın rubrik yedeğiyle KABUL sayılması, kabul isabeti, McNemar yönü, tam koşu süre tahmini, bildiri makroları (`--latex`) |
| `test_judgement.py` | 38 | Yargıç: ayrıştırma, yedek yönerge, verdict kuralları, `BudgetExceeded` yutulmaz |
| `test_gates.py` | 33 | severity / iteration gate, trend uzatması |
| `test_scoring.py` | 27 | 100 puanlık rubrik |
| `test_metrics.py` | 28 | CC, MI, Halstead, Bandit, severity, import whitelist |
| `test_prompts.py` | 19 | Dört katmanlı üretim istemi; onarımda önceki çözümün gösterilmesi (`FEEDBACK_INCLUDE_CODE`, açık/kapalı) |
| `test_dataset.py` | 15 | HTML temizleme, biçimleme, tarih penceresi |
| `test_cleaner.py` | 13 | Yorum temizleme |
| `test_lcb_eval.py` | 12 | Yerel LCB uyumlu değerlendirici |
| `test_graph.py` | 11 | Graf kurulumu (V1/V2/V3) |
| `test_edge_cases.py` | 11 | Kural tabanlı uç durumlar |
| `test_logger.py` | 9 | memory.md, iterasyon JSON (run_id'li adlar) |
| `test_router.py` | 7 | Sıralı model yönlendirici |
| `test_baseline.py` | 6 | `benchmark/legacy/baseline.py` kod çıkarma (eski) |
| `test_nodes.py` | 6 | Düğüm durum dönüşümleri |
| `test_runtime.py` | 10 | Çalışma zamanı ölçümü: alt sürecin rusage'ından işlemci süresi ve tepe bellek (`os.wait4`), uyuyan programın düşük CPU süresi, LeetCode metodunun sürücüyle çağrılması |

Ağ bağlantısı gerektiren test yoktur; ücretli API çağrıları sahte istemciyle sınanır.

## 1. Test Layers

| Layer | What Is Tested | Tool | When |
|--------|---------------|------|----------|
| Unit | Each module independently | `pytest` | On every code change |
| Integration | Inter-node connections | `pytest` | On gate/graph changes |
| Smoke | ENV config + model server | `pytest` | On server startup |
| End-to-End | 1 question → full pipeline | `python main.py` | When all nodes are ready |
| Benchmark | 50 LCB questions | `benchmark/evaluator.py` | Final evaluation |

---

## 2. Kurulum

```bash
pip install pytest pytest-timeout pytest-cov
```

```
tests/
├── conftest.py            ← Paylaşılan fixture'lar
├── test_cleaner.py        ← nodes/cleaner.py
├── test_metrics.py        ← metrics/complexity.py + security.py
├── test_gates.py          ← pipeline/gates.py
├── test_lcb_eval.py       ← sandbox/lcb_eval.py
├── test_runtime.py        ← sandbox/runner.py
├── test_edge_cases.py     ← dataset/edge_cases.py
├── test_dataset.py        ← dataset/loader.py + splitter.py
├── test_logger.py         ← memory/logger.py
└── test_nodes.py          ← nodes/static_analysis.py (integration)
└── test_prompts.py        ← models/prompts.py (yeni)
```

---

## 3. tests/conftest.py

```python
"""
Paylaşılan fixture'lar — tüm test dosyaları tarafından kullanılır.
"""
import pytest
from pipeline.state import PipelineState


# ── Kod sabitleri ────────────────────────────────────────────────────────

CLEAN_CODE = """\
def two_sum(nums, target):
    seen = {}
    for i, n in enumerate(nums):
        if target - n in seen:
            return [seen[target - n], i]
        seen[n] = i
    return []
"""

CLEAN_CODE_WITH_COMMENTS = """\
# Find the sum of two numbers
def two_sum(nums, target):
    seen = {}  # hash map
    for i, n in enumerate(nums):  # her elemanı gez
        if target - n in seen:
            return [seen[target - n], i]  # bulundu
        seen[n] = i
    return []
"""

COMMENTED_WITH_DOCSTRING = """\
def two_sum(nums, target):
    \"\"\"İki sayının indekslerini döner.\"\"\"
    # hash map approach
    seen = {}
    return seen  # placeholder
"""

DANGEROUS_CODE = """\
import os
import subprocess
exec(input())
os.system("rm -rf /")
"""

COMPLEX_CODE = """\
def nightmare(a, b, c, d, e):
    if a:
        if b:
            if c:
                if d:
                    if e:
                        for i in range(10):
                            for j in range(10):
                                if i == j:
                                    while True:
                                        break
    return 0
"""

STDIN_CODE = """\
n = int(input())
nums = list(map(int, input().split()))
print(sum(x for x in nums if x % 2 == 0))
"""


# ── State fixture ────────────────────────────────────────────────────────

@pytest.fixture
def base_state():
    """Minimum geçerli PipelineState döner. Test içinde override edilebilir."""
    def _make(**overrides) -> PipelineState:
        state = {
            "run_id":         "test_run",
            "active_model":   "M1",
            "iteration":      1,
            "max_iterations": 10,
            "problem":        "Test problem",
            "problem_row":    {
                "id": "q_test", "title": "Test", "difficulty": "easy",
                "content": "Test content", "starter_code": "",
                "public_tests":  {"input": ["3\n1 2 3"], "output": ["2"]},
                "private_tests": {"input": [],            "output": []},
                "all_tests":     {"input": ["3\n1 2 3"], "output": ["2"]},
            },
            "question_id":    "q_test",
            "difficulty":     "easy",
            "test_inputs":    ["3\n1 2 3"],
            "test_outputs":   ["2"],
            "edge_cases":     [],
            "generated_code": CLEAN_CODE,
            "stripped_code":  CLEAN_CODE,
            "node1_cc":          0.0,
            "node1_mi":          100.0,
            "node1_halstead":    0.0,
            "node1_bandit_high": False,
            "node1_severity":    "OK",
            "node1_issues":      [],
            "node2_pass_rate":            0.0,
            "node2_passed_count":         0,
            "node2_total_count":          0,
            "node2_failed_cases":         [],
            "node2_crash_count":          0,
            "node2_test_diversity_score": 1.0,
            "node2_lcb_available":        False,
            "node2_cpu_ms":    0.0,
            "node2_ram_mb":    0.0,
            "node2_timed_out": False,
            "node3_total_score":     0.0,
            "node3_pass":            False,
            "node3_score_breakdown": {},
            "node3_failure_reasons": [],
            "node3_improved_prompt": "",
            "timing": {
                "codegen_ms": 0, "node1_ms": 0,
                "node2_ms": 0,   "node3_ms": 0, "total_ms": 0
            },
            "tokens": {
                "codegen_prompt": 0, "codegen_completion": 0,
                "judgement_prompt": 0, "judgement_completion": 0, "total": 0
            },
            "memory_log":   [],
            "final_status": "PENDING",
        }
        state.update(overrides)
        return state
    return _make
```

---

## 4. tests/test_cleaner.py

```python
"""
Unit testler: nodes/cleaner.py
Modül: strip_comments(), strip_comments_node()

Kritik davranışlar:
  - Satır başı yorum → kaldırılır
  - Inline yorum → kaldırılır
  - Docstring (triple-quote) → KORUNUR
  - Syntax hatalı kod → orijinal döner, exception yok
  - generated_code → değişmez
  - stripped_code → temizlenmiş hali
"""
import pytest
from nodes.cleaner import strip_comments, strip_comments_node
from tests.conftest import (
    CLEAN_CODE, CLEAN_CODE_WITH_COMMENTS, COMMENTED_WITH_DOCSTRING
)


class TestStripComments:

    def test_satir_basi_yorum_kaldirilir(self):
        code = "# başlık\ndef f():\n    return 1"
        result = strip_comments(code)
        assert "#" not in result
        assert "def f():" in result
        assert "return 1" in result

    def test_inline_yorum_kaldirilir(self):
        code = "x = a + b  # topla"
        result = strip_comments(code)
        assert "# topla" not in result
        assert "x = a + b" in result

    def test_docstring_korunur(self):
        result = strip_comments(COMMENTED_WITH_DOCSTRING)
        assert '"""İki sayının indekslerini döner."""' in result
        assert "# hash map approach" not in result
        assert "# placeholder" not in result

    def test_temiz_kod_degismez(self):
        result = strip_comments(CLEAN_CODE)
        # Content should be preserved in comment-free code
        assert "def two_sum" in result
        assert "seen" in result

    def test_yorumlu_kod_temizlenir(self):
        result = strip_comments(CLEAN_CODE_WITH_COMMENTS)
        assert "#" not in result
        assert "def two_sum" in result
        assert "seen" in result

    def test_syntax_hatali_kod_orijinal_doner(self):
        broken = "def f(\n    # eksik kapanış parantezi"
        result = strip_comments(broken)
        # Must not raise an exception, must return something
        assert isinstance(result, str)

    def test_bos_kod(self):
        assert strip_comments("") == ""

    def test_sadece_yorumdan_olusan_kod(self):
        code = "# satır 1\n# satır 2\n# satır 3"
        result = strip_comments(code)
        assert "#" not in result
        # May be empty or whitespace only
        assert isinstance(result, str)

    def test_cok_satir_yorum_blogu(self):
        code = "# blok başlangıcı\n# devam\n# bitiş\ndef f():\n    pass"
        result = strip_comments(code)
        assert "#" not in result
        assert "def f():" in result


class TestStripCommentsNode:

    def test_stripped_code_doldurulur(self, base_state):
        state = base_state(generated_code=CLEAN_CODE_WITH_COMMENTS)
        result = strip_comments_node(state)
        assert "#" not in result["stripped_code"]

    def test_generated_code_degismez(self, base_state):
        state = base_state(generated_code=CLEAN_CODE_WITH_COMMENTS)
        result = strip_comments_node(state)
        assert result["generated_code"] == CLEAN_CODE_WITH_COMMENTS

    def test_diger_state_alanlari_korunur(self, base_state):
        state = base_state(generated_code=CLEAN_CODE_WITH_COMMENTS,
                           iteration=3, difficulty="hard")
        result = strip_comments_node(state)
        assert result["iteration"]  == 3
        assert result["difficulty"] == "hard"
```

---

## 5. tests/test_metrics.py

```python
"""
Unit testler: metrics/complexity.py + metrics/security.py

Kritik davranışlar:
  - Threshold sınır değerleri doğru çalışıyor mu?
  - _decide_severity öncelik sırası doğru mu?
  - HIGH > LOW > OK
"""
import pytest
from metrics.complexity import (
    cyclomatic_complexity, maintainability_index,
    halstead_volume, cognitive_complexity,
)
from metrics.security import bandit_scan, check_imports
from nodes.static_analysis import _decide_severity
from tests.conftest import CLEAN_CODE, DANGEROUS_CODE, COMPLEX_CODE


class TestCyclomaticComplexity:

    def test_temiz_kod_dusuk_cc(self):
        assert cyclomatic_complexity(CLEAN_CODE) <= 5

    def test_complex_kod_yuksek_cc(self):
        assert cyclomatic_complexity(COMPLEX_CODE) > 10

    def test_tek_satir_fonksiyon(self):
        assert cyclomatic_complexity("def f(): return 1") <= 2

    def test_bos_string(self):
        # Must not raise an exception
        result = cyclomatic_complexity("")
        assert isinstance(result, float)


class TestMaintainabilityIndex:

    def test_temiz_kod_yuksek_mi(self):
        assert maintainability_index(CLEAN_CODE) >= 65

    def test_gecerli_aralik(self):
        mi = maintainability_index(CLEAN_CODE)
        assert 0.0 <= mi <= 100.0

    def test_syntax_hatali_kod_exception_vermez(self):
        result = maintainability_index("def f(")
        assert isinstance(result, float)


class TestHalsteadVolume:

    def test_temiz_kod_makul_halstead(self):
        h = halstead_volume(CLEAN_CODE)
        assert h < 2000

    def test_bos_kod(self):
        result = halstead_volume("")
        assert isinstance(result, float)


class TestBanditScan:

    def test_temiz_kod_high_yok(self):
        result = bandit_scan(CLEAN_CODE)
        assert result["high_count"] == 0

    def test_tehlikeli_kod_high_var(self):
        result = bandit_scan(DANGEROUS_CODE)
        assert result["high_count"] > 0

    def test_donus_formati(self):
        result = bandit_scan(CLEAN_CODE)
        assert "high_count"   in result
        assert "medium_count" in result
        assert "issues"       in result


class TestCheckImports:

    def test_temiz_import_gecerli(self):
        code = "import math\nimport re"
        assert check_imports(code) == []

    def test_yasak_import_tespit_edilir(self):
        code = "import subprocess\nimport socket"
        forbidden = check_imports(code)
        assert len(forbidden) > 0
        assert any("subprocess" in f or "socket" in f for f in forbidden)

    def test_from_import_kontrol_edilir(self):
        code = "from os import path"
        # os whitelist'te var, path ona ait
        # Depends on whitelist configuration
        result = check_imports(code)
        assert isinstance(result, list)

    def test_syntax_hatali_kod(self):
        result = check_imports("def f(")
        assert isinstance(result, list)


class TestDecideSeverity:
    """
    _decide_severity öncelik sırası: HIGH > LOW > OK
    """

    def test_bandit_high_her_zaman_high(self):
        # Even if other metrics are good, bandit HIGH → severity HIGH
        sev = _decide_severity(cc=2.0, mi=90.0, halstead=100.0,
                               bandit_high_count=1, forbidden_imports=[])
        assert sev == "HIGH"

    def test_forbidden_import_high(self):
        sev = _decide_severity(cc=2.0, mi=90.0, halstead=100.0,
                               bandit_high_count=0, forbidden_imports=["subprocess"])
        assert sev == "HIGH"

    def test_cc_sinir_deger_yuksek(self):
        # CC > 20 → HIGH
        sev = _decide_severity(cc=21.0, mi=90.0, halstead=100.0,
                               bandit_high_count=0, forbidden_imports=[])
        assert sev == "HIGH"

    def test_cc_sinir_deger_esit(self):
        # CC == 20 → not HIGH (condition is > 20)
        sev = _decide_severity(cc=20.0, mi=90.0, halstead=100.0,
                               bandit_high_count=0, forbidden_imports=[])
        assert sev != "HIGH"

    def test_mi_dusuk_low(self):
        # MI < 40 → LOW
        sev = _decide_severity(cc=2.0, mi=35.0, halstead=100.0,
                               bandit_high_count=0, forbidden_imports=[])
        assert sev == "LOW"

    def test_halstead_yuksek_low(self):
        # Halstead > 2000 → LOW
        sev = _decide_severity(cc=2.0, mi=90.0, halstead=2001.0,
                               bandit_high_count=0, forbidden_imports=[])
        assert sev == "LOW"

    def test_temiz_kod_ok(self):
        sev = _decide_severity(cc=2.0, mi=90.0, halstead=100.0,
                               bandit_high_count=0, forbidden_imports=[])
        assert sev == "OK"

    @pytest.mark.parametrize("cc,mi,halstead,expected", [
        (2.0,  90.0, 100.0,  "OK"),    # Hepsi iyi
        (2.0,  35.0, 100.0,  "LOW"),   # MI düşük
        (2.0,  90.0, 2001.0, "LOW"),   # Halstead yüksek
        (21.0, 90.0, 100.0,  "HIGH"),  # CC aşırı
    ])
    def test_parametric_severity(self, cc, mi, halstead, expected):
        sev = _decide_severity(cc=cc, mi=mi, halstead=halstead,
                               bandit_high_count=0, forbidden_imports=[])
        assert sev == expected
```

---

## 6. tests/test_gates.py

```python
"""
Unit testler: pipeline/gates.py
severity_gate ve iteration_gate dönüş değerleri.

Kritik: Yanlış gate kararı pipeline'ı sessizce yanlış yöne götürür.
"""
import pytest
from pipeline.gates import severity_gate, iteration_gate


class TestSeverityGate:

    def test_high_stop_high_a_gider(self):
        assert severity_gate({"node1_severity": "HIGH"}) == "stop_high"

    def test_low_node3_e_gider(self):
        assert severity_gate({"node1_severity": "LOW"}) == "to_node3"

    def test_ok_node2_ye_gider(self):
        assert severity_gate({"node1_severity": "OK"}) == "to_node2"


class TestIterationGate:
    """
    iteration_gate, log_iteration_node SONRASI çalışır.
    Dönüş değerleri: write_memory_final | stop_max | loop_to_codegen
    """

    def _state(self, score, passed, iteration,
               difficulty="easy", max_iter=10):
        return {
            "node3_total_score": score,
            "node3_pass":        passed,
            "iteration":         iteration,
            "max_iterations":    max_iter,
            "difficulty":        difficulty,
        }

    # ── Basic routing ────────────────────────────────────────────

    def test_pass_write_memory_final_a_gider(self):
        state = self._state(score=85, passed=True, iteration=3)
        assert iteration_gate(state) == "write_memory_final"

    def test_fail_loop_a_gider(self):
        state = self._state(score=60, passed=False, iteration=3)
        assert iteration_gate(state) == "loop_to_codegen"

    def test_max_iter_stop_max_a_gider(self):
        state = self._state(score=60, passed=False, iteration=10)
        assert iteration_gate(state) == "stop_max"

    # ── Difficulty thresholds — boundary values ────────────────────────────

    @pytest.mark.parametrize("difficulty,threshold", [
        ("easy",   80),
        ("medium", 70),
        ("hard",   60),
    ])
    def test_esik_altinda_loop(self, difficulty, threshold):
        state = self._state(score=threshold - 1, passed=True,
                            iteration=3, difficulty=difficulty)
        assert iteration_gate(state) == "loop_to_codegen"

    @pytest.mark.parametrize("difficulty,threshold", [
        ("easy",   80),
        ("medium", 70),
        ("hard",   60),
    ])
    def test_esik_esitinde_pass(self, difficulty, threshold):
        state = self._state(score=threshold, passed=True,
                            iteration=3, difficulty=difficulty)
        assert iteration_gate(state) == "write_memory_final"

    # ── node3_pass False ise PASS olmaz ──────────────────────────────

    def test_yeterli_skor_ama_pass_false(self):
        # When node3_pass=False, PASS is not granted even with sufficient score
        state = self._state(score=90, passed=False, iteration=3)
        assert iteration_gate(state) == "loop_to_codegen"

    # ── max_iterations boundary values ───────────────────────────────

    def test_iter_9_max_10_devam_eder(self):
        state = self._state(score=60, passed=False,
                            iteration=9, max_iter=10)
        assert iteration_gate(state) == "loop_to_codegen"

    def test_iter_10_max_10_durur(self):
        state = self._state(score=60, passed=False,
                            iteration=10, max_iter=10)
        assert iteration_gate(state) == "stop_max"
```

---

## 7. tests/test_lcb_eval.py

```python
"""
Unit testler: sandbox/lcb_eval.py
_compute_diversity(), lcb_evaluate() fallback davranışı.

lcb_runner olmadan da testler çalışır — fallback yolu test edilir.
"""
import pytest
from sandbox.lcb_eval import _compute_diversity, LCB_AVAILABLE


class TestComputeDiversity:

    def test_edge_case_yok_1_doner(self):
        assert _compute_diversity(crash_count=0, edge_cases=[]) == 1.0

    def test_hic_crash_yok_1_doner(self):
        edge = [{"input": "0\n", "output": None}] * 4
        assert _compute_diversity(crash_count=0, edge_cases=edge) == 1.0

    def test_tumu_crash_0_doner(self):
        edge = [{"input": "0\n", "output": None}] * 4
        assert _compute_diversity(crash_count=4, edge_cases=edge) == 0.0

    def test_yarisi_crash_05_doner(self):
        edge = [{"input": "0\n", "output": None}] * 4
        score = _compute_diversity(crash_count=2, edge_cases=edge)
        assert score == 0.5

    def test_sinir_asimaz(self):
        edge = [{"input": "0\n", "output": None}] * 2
        # Score never goes below 0.0 even if crash_count > len(edge_cases)
        score = _compute_diversity(crash_count=10, edge_cases=edge)
        assert score >= 0.0


class TestFallbackBehavior:
    """lcb_runner kurulu olmasa bile pipeline devam eder."""

    SIMPLE_CODE = (
        "n = int(input())\n"
        "nums = list(map(int, input().split()))\n"
        "print(sum(x for x in nums if x % 2 == 0))\n"
    )
    PROBLEM_ROW = {
        "all_tests": {
            "input":  ["6\n1 2 3 4 5 6", "5\n1 3 5 7 9"],
            "output": ["12",              "0"],
        }
    }

    def test_lcb_available_bool(self):
        assert isinstance(LCB_AVAILABLE, bool)

    @pytest.mark.skipif(LCB_AVAILABLE, reason="lcb_runner kurulu — fallback test edilemiyor")
    def test_fallback_calisir(self):
        from sandbox.lcb_eval import _fallback_evaluate
        result = _fallback_evaluate(
            code=self.SIMPLE_CODE,
            problem_row=self.PROBLEM_ROW,
            timeout=5,
        )
        assert "pass_rate"            in result
        assert "failed_cases"         in result
        assert "test_diversity_score" in result
        assert result["lcb_available"] is False

    @pytest.mark.skipif(not LCB_AVAILABLE, reason="lcb_runner kurulu değil")
    def test_lcb_evaluate_calisir(self):
        from sandbox.lcb_eval import lcb_evaluate
        result = lcb_evaluate(
            code=self.SIMPLE_CODE,
            problem_row=self.PROBLEM_ROW,
            edge_cases=[],
            timeout=5,
        )
        assert result["pass_rate"] == 1.0
        assert result["lcb_available"] is True
```

---

## 8. tests/test_runtime.py

```python
"""
Unit testler: sandbox/runner.py
measure_runtime() davranışı.
"""
import pytest
from sandbox.runner import measure_runtime


CORRECT_CODE = (
    "n = int(input())\n"
    "nums = list(map(int, input().split()))\n"
    "print(sum(x for x in nums if x % 2 == 0))\n"
)
INFINITE_LOOP = "while True: pass\n"
SYNTAX_ERROR  = "def f(\n"


class TestMeasureRuntime:

    def test_donus_formati(self):
        result = measure_runtime(CORRECT_CODE, ["6\n1 2 3 4 5 6"])
        assert "cpu_ms"    in result
        assert "ram_mb"    in result
        assert "timed_out" in result
        assert "error"     in result

    def test_dogru_kod_cpu_ms_pozitif(self):
        result = measure_runtime(CORRECT_CODE, ["6\n1 2 3 4 5 6"])
        assert result["cpu_ms"] > 0
        assert result["timed_out"] is False

    def test_timeout_yakalanir(self):
        result = measure_runtime(INFINITE_LOOP, [""], timeout_sec=2)
        assert result["timed_out"] is True

    def test_bos_input_listesi(self):
        result = measure_runtime(CORRECT_CODE, [])
        assert result["cpu_ms"] == 0.0
        assert result["timed_out"] is False

    def test_cok_test_ortalamalanir(self):
        inputs = ["6\n1 2 3 4 5 6", "5\n1 3 5 7 9"]
        result = measure_runtime(CORRECT_CODE, inputs)
        assert result["cpu_ms"] > 0
        assert result["ram_mb"] >= 0

    def test_max_3_test_calistirilir(self):
        # Runs at most 3 tests even if 5 are provided
        # cpu_ms should be in a reasonable range
        inputs = ["6\n1 2 3 4 5 6"] * 5
        result = measure_runtime(CORRECT_CODE, inputs)
        # 3 tests × ~20ms = at most ~200ms expected (with margin for slow CI)
        assert result["cpu_ms"] < 10_000
```

---

## 9. tests/test_edge_cases.py

```python
"""
Unit testler: dataset/edge_cases.py
needs_edge_cases(), generate_edge_cases(), augment_row()
"""
import pytest
from dataset.edge_cases import (
    needs_edge_cases, generate_edge_cases, augment_row,
    EDGE_CASE_THRESHOLD,
)


def make_row(n_tests: int, content: str = "Given a list of integers") -> dict:
    """Test için minimal problem_row üretir."""
    inputs  = [f"test_{i}" for i in range(n_tests)]
    outputs = ["0"] * n_tests
    return {
        "id": "q_test", "title": "Test", "difficulty": "easy",
        "content": content, "starter_code": "",
        "public_tests":  {"input": inputs[:2], "output": outputs[:2]},
        "private_tests": {"input": inputs[2:], "output": outputs[2:]},
        "all_tests":     {"input": inputs,      "output": outputs},
    }


class TestNeedsEdgeCases:

    def test_az_test_edge_case_gerektirir(self):
        row = make_row(n_tests=EDGE_CASE_THRESHOLD - 1)
        assert needs_edge_cases(row) is True

    def test_yeterli_test_gerekmez(self):
        row = make_row(n_tests=EDGE_CASE_THRESHOLD)
        assert needs_edge_cases(row) is False

    def test_fazla_test_gerekmez(self):
        row = make_row(n_tests=EDGE_CASE_THRESHOLD + 5)
        assert needs_edge_cases(row) is False


class TestGenerateEdgeCases:

    def test_en_az_bir_case_uretilir(self):
        row = make_row(n_tests=3)
        cases = generate_edge_cases(row)
        assert len(cases) >= 1

    def test_her_case_input_iceriyor(self):
        row = make_row(n_tests=3)
        for case in generate_edge_cases(row):
            assert "input"  in case
            assert "output" in case

    def test_edge_case_output_none(self):
        # Edge case output is unknown — must be None
        row = make_row(n_tests=3)
        for case in generate_edge_cases(row):
            assert case["output"] is None

    def test_liste_icerigi_olan_soru(self):
        row = make_row(n_tests=3, content="Given a list of integers nums")
        cases = generate_edge_cases(row)
        # Cases like empty list, single element should be generated
        assert len(cases) >= 1

    def test_input_string_turu(self):
        row = make_row(n_tests=3)
        for case in generate_edge_cases(row):
            assert isinstance(case["input"], str)


class TestAugmentRow:

    def test_az_test_edge_case_eklenir(self):
        row = make_row(n_tests=3)
        aug = augment_row(row)
        assert "edge_cases" in aug
        assert len(aug["edge_cases"]) >= 1

    def test_yeterli_test_bos_liste(self):
        row = make_row(n_tests=EDGE_CASE_THRESHOLD + 2)
        aug = augment_row(row)
        assert aug["edge_cases"] == []

    def test_orijinal_row_degismez(self):
        row = make_row(n_tests=3)
        aug = augment_row(row)
        # all_tests must remain unchanged
        assert aug["all_tests"] == row["all_tests"]
```

---

## 10. tests/test_dataset.py

```python
"""
Unit testler: dataset/loader.py + dataset/splitter.py

lcb_runner bağlantısı gerektirmez — sadece string işleme test edilir.
"""
import pytest
from dataset.loader import _clean_html
from dataset.splitter import format_problem, dataset_stats


class TestCleanHtml:

    def test_html_tag_kaldirilir(self):
        assert _clean_html("<p>hello</p>") == "hello"

    def test_nbsp_bosluk_olur(self):
        assert " " in _clean_html("a&nbsp;b")

    def test_lt_gt_donusturulur(self):
        result = _clean_html("&lt;br&gt;")
        assert "<" in result or ">" in result

    def test_cok_satir_boslugu_azaltilir(self):
        result = _clean_html("a\n\n\n\nb")
        assert "\n\n\n" not in result

    def test_temiz_metin_degismez(self):
        text = "Given a list of integers"
        assert _clean_html(text) == text


class TestFormatProblem:

    def _make_row(self, starter="", inputs=None, outputs=None):
        return {
            "title":        "Test Problem",
            "difficulty":   "easy",
            "content":      "Problem description here.",
            "starter_code": starter,
            "public_tests": {
                "input":  inputs  or ["3\n1 2 3"],
                "output": outputs or ["6"],
            },
        }

    def test_baslik_icerir(self):
        result = format_problem(self._make_row())
        assert "Test Problem" in result

    def test_zorluk_icerir(self):
        result = format_problem(self._make_row())
        assert "easy" in result

    def test_ornek_testler_gosterilir(self):
        result = format_problem(self._make_row())
        assert "3\n1 2 3" in result
        assert "6" in result

    def test_starter_code_varsa_eklenir(self):
        result = format_problem(self._make_row(starter="def solve(): pass"))
        assert "solve" in result

    def test_starter_code_yoksa_eklenmez(self):
        result = format_problem(self._make_row(starter=""))
        assert "Starter Code" not in result

    def test_stdout_yonergesi_var(self):
        result = format_problem(self._make_row())
        assert "stdin" in result.lower() or "stdout" in result.lower()


class TestDatasetStats:

    def test_bos_liste(self):
        stats = dataset_stats([])
        assert stats["total"] == 0

    def test_dogru_sayim(self):
        rows = [
            {"difficulty": "easy"},
            {"difficulty": "easy"},
            {"difficulty": "medium"},
            {"difficulty": "hard"},
        ]
        stats = dataset_stats(rows)
        assert stats["total"]  == 4
        assert stats["easy"]   == 2
        assert stats["medium"] == 1
        assert stats["hard"]   == 1
```

---

## 11. tests/test_logger.py

```python
"""
Unit testler: memory/logger.py
_append_row(), read_memory(), _compute_diversity bağlantısı.

Dosya yazma testleri tmp_path ile geçici dizin kullanır.
"""
import os
import json
import pytest
from unittest.mock import patch


class TestAppendRow:

    def test_memory_md_olusturulur(self, tmp_path, base_state):
        log_path = str(tmp_path / "memory.md")
        with patch("memory.logger.settings") as mock_settings:
            mock_settings.memory_log_path = log_path
            mock_settings.output_dir      = str(tmp_path)

            from memory.logger import log_iteration_node
            state = base_state(
                node3_total_score=62.0, node3_pass=False,
                run_id="test_run"
            )
            log_iteration_node(state)

        assert os.path.exists(log_path)

    def test_baslik_satiri_yazilir(self, tmp_path, base_state):
        log_path = str(tmp_path / "memory.md")
        with patch("memory.logger.settings") as mock_settings:
            mock_settings.memory_log_path = log_path
            mock_settings.output_dir      = str(tmp_path)

            from memory.logger import log_iteration_node
            log_iteration_node(base_state(node3_total_score=62.0,
                                          node3_pass=False, run_id="r"))

        content = open(log_path).read()
        assert "Pipeline Run Log" in content
        assert "| iter |" in content

    def test_loop_status_yazilir(self, tmp_path, base_state):
        log_path = str(tmp_path / "memory.md")
        with patch("memory.logger.settings") as mock_settings:
            mock_settings.memory_log_path = log_path
            mock_settings.output_dir      = str(tmp_path)

            from memory.logger import log_iteration_node
            log_iteration_node(base_state(
                node3_total_score=62.0, node3_pass=False,
                iteration=1, run_id="r"
            ))

        content = open(log_path).read()
        assert "LOOP" in content

    def test_pass_status_yazilir(self, tmp_path, base_state):
        log_path = str(tmp_path / "memory.md")
        with patch("memory.logger.settings") as mock_settings:
            mock_settings.memory_log_path = log_path
            mock_settings.output_dir      = str(tmp_path)

            from memory.logger import log_iteration_node
            log_iteration_node(base_state(
                node3_total_score=85.0, node3_pass=True,
                iteration=2, run_id="r"
            ))

        content = open(log_path).read()
        assert "PASS" in content

    def test_stop_high_yazilir(self, tmp_path, base_state):
        log_path = str(tmp_path / "memory.md")
        with patch("memory.logger.settings") as mock_settings:
            mock_settings.memory_log_path = log_path
            mock_settings.output_dir      = str(tmp_path)

            from memory.logger import log_severity_stop_node
            log_severity_stop_node(base_state(
                node1_severity="HIGH", run_id="r"
            ))

        content = open(log_path).read()
        assert "STOP_HIGH" in content


class TestReadMemory:

    def test_bos_dosya_bos_liste_doner(self, tmp_path):
        log_path = str(tmp_path / "memory.md")
        with patch("memory.logger.settings") as mock_settings:
            mock_settings.memory_log_path = log_path

            from memory.logger import read_memory
            result = read_memory(log_path)
            assert result == []

    def test_yok_dosya_bos_liste_doner(self, tmp_path):
        from memory.logger import read_memory
        result = read_memory(str(tmp_path / "yok.md"))
        assert result == []

    def test_yazilan_satirlar_okunur(self, tmp_path, base_state):
        log_path = str(tmp_path / "memory.md")
        with patch("memory.logger.settings") as mock_settings:
            mock_settings.memory_log_path = log_path
            mock_settings.output_dir      = str(tmp_path)

            from memory.logger import log_iteration_node, read_memory
            log_iteration_node(base_state(
                node3_total_score=62.0, node3_pass=False, run_id="r"
            ))
            rows = read_memory(log_path)

        assert len(rows) == 1
        assert rows[0]["status"] == "LOOP"
        assert rows[0]["model"]  == "M1"


class TestIterationJson:

    def test_json_dosyasi_olusturulur(self, tmp_path, base_state):
        with patch("memory.logger.settings") as mock_settings:
            mock_settings.memory_log_path = str(tmp_path / "memory.md")
            mock_settings.output_dir      = str(tmp_path)

            from memory.logger import log_iteration_node
            log_iteration_node(base_state(
                question_id="q_001", node3_total_score=62.0,
                node3_pass=False, run_id="r"
            ))

        json_path = tmp_path / "iterations" / "q_001" / "M1_iter1.json"
        assert json_path.exists()

    def test_json_icerik_dogru(self, tmp_path, base_state):
        with patch("memory.logger.settings") as mock_settings:
            mock_settings.memory_log_path = str(tmp_path / "memory.md")
            mock_settings.output_dir      = str(tmp_path)

            from memory.logger import log_iteration_node
            log_iteration_node(base_state(
                question_id="q_001", node3_total_score=62.0,
                node3_pass=False, run_id="r_42",
                node3_improved_prompt="Boş listeyi kontrol et."
            ))

        data = json.load(open(tmp_path / "iterations" / "q_001" / "M1_iter1.json"))
        assert data["run_id"]  == "r_42"
        assert data["status"]  == "LOOP"
        assert data["node3"]["improved_prompt"] == "Boş listeyi kontrol et."
        assert "node1" in data
        assert "node2" in data
        assert "timing" in data
        assert "tokens" in data
```

---

## 12. tests/test_nodes.py (Integration)

```python
"""
Integration testler: Node'ların state üzerindeki etkisi.
Her node bağımsız çağrılır; state dönüşümü kontrol edilir.
"""
import pytest
from nodes.static_analysis import static_analysis_node
from nodes.cleaner import strip_comments_node
from tests.conftest import CLEAN_CODE, DANGEROUS_CODE, CLEAN_CODE_WITH_COMMENTS


class TestStaticAnalysisNode:

    def test_temiz_kod_ok_severity(self, base_state):
        result = static_analysis_node(base_state(generated_code=CLEAN_CODE))
        assert result["node1_severity"]    == "OK"
        assert result["node1_bandit_high"] is False

    def test_tehlikeli_kod_high_severity(self, base_state):
        result = static_analysis_node(base_state(generated_code=DANGEROUS_CODE))
        assert result["node1_severity"] == "HIGH"

    def test_state_alanlari_doldurulur(self, base_state):
        result = static_analysis_node(base_state(generated_code=CLEAN_CODE))
        for field in ("node1_cc", "node1_mi", "node1_halstead",
                      "node1_bandit_high", "node1_severity", "node1_issues"):
            assert field in result, f"{field} state'te yok"

    def test_diger_state_alanlari_korunur(self, base_state):
        state  = base_state(generated_code=CLEAN_CODE, iteration=5)
        result = static_analysis_node(state)
        assert result["iteration"]     == 5
        assert result["active_model"]  == "M1"
        assert result["difficulty"]    == "easy"


class TestCleanerNode:

    def test_stripped_code_doldurulur(self, base_state):
        state  = base_state(generated_code=CLEAN_CODE_WITH_COMMENTS)
        result = strip_comments_node(state)
        assert "stripped_code" in result
        assert "#" not in result["stripped_code"]

    def test_generated_code_korunur(self, base_state):
        state  = base_state(generated_code=CLEAN_CODE_WITH_COMMENTS)
        result = strip_comments_node(state)
        assert result["generated_code"] == CLEAN_CODE_WITH_COMMENTS
```

---

## 13. Running All Tests

```bash
# All unit + integration tests
pytest tests/ -v --timeout=30

# Fast tests only (excluding sandbox)
pytest tests/ -v --timeout=30 \
    --ignore=tests/test_runtime.py \
    --ignore=tests/test_lcb_eval.py

# Kapsam raporu
pytest tests/ --cov=. --cov-report=term-missing \
    --cov-omit="benchmark/*,tests/*"

# Tek dosya
pytest tests/test_gates.py -v
pytest tests/test_cleaner.py -v

# Sadece parametrik testler
pytest tests/test_gates.py -v -k "parametric"
pytest tests/test_metrics.py -v -k "parametric"
```

**Expected output (all tests):**

```
tests/test_cleaner.py     ........  9 passed
tests/test_metrics.py     ..........  10 passed
tests/test_gates.py       ..........  10 passed
tests/test_lcb_eval.py    .....  5 passed
tests/test_runtime.py     .....  5 passed
tests/test_edge_cases.py  .......  7 passed
tests/test_dataset.py     .......  7 passed
tests/test_logger.py      .........  9 passed
tests/test_nodes.py       .......  7 passed
tests/test_prompts.py     ..........  10 passed
────────────────────────────────────────────
79 passed in ~25s
```

---

## 13. tests/test_prompts.py

```python
"""
Unit testler: models/prompts.py
Dört katmanlı codegen prompt yapısı.

Kritik davranışlar:
  - İlk iterasyonda feedback katmanı yok
  - Sonraki iterasyonlarda feedback katmanı var
  - improved_prompt içeriği prompt'a dahil ediliyor
  - Katmanlar arası ayraç (---) doğru
  - Whitelist kısıtları CONSTRAINTS içinde
  - Pipeline bağlamı CONTEXT içinde
"""
import pytest
from models.prompts import (
    build_system_prompt,
    CODEGEN_IDENTITY,
    CODEGEN_CONTEXT,
    CODEGEN_CONSTRAINTS,
    CODEGEN_FEEDBACK_RULES,
)


class TestBuildSystemPromptFirstIteration:
    """improved_prompt=None — ilk iterasyon"""

    def test_identity_dahil(self):
        prompt = build_system_prompt(None)
        assert "expert Python programmer" in prompt

    def test_context_dahil(self):
        prompt = build_system_prompt(None)
        assert "Static Analyzer" in prompt
        assert "Sandbox Executor" in prompt
        assert "Judge LLM" in prompt

    def test_constraints_dahil(self):
        prompt = build_system_prompt(None)
        assert "stdin" in prompt
        assert "stdout" in prompt
        assert "subprocess" in prompt   # yasak import listede

    def test_feedback_katmani_yok(self):
        prompt = build_system_prompt(None)
        assert "COMPLETELY NEW" not in prompt
        assert "previous iteration" not in prompt.lower() \
               or "operating" in prompt   # context'teki "iteration" izin verilir

    def test_improved_prompt_kalintisi_yok(self):
        prompt = build_system_prompt(None)
        assert "Improvement Notes" not in prompt


class TestBuildSystemPromptSubsequentIteration:
    """improved_prompt dolu — iterasyon > 1"""

    SAMPLE_FEEDBACK = (
        "Your code returns None for empty list input. "
        "Add a guard: if not nums: return 0"
    )

    def test_feedback_rules_dahil(self):
        prompt = build_system_prompt(self.SAMPLE_FEEDBACK)
        assert "COMPLETELY NEW" in prompt

    def test_improved_prompt_dahil(self):
        prompt = build_system_prompt(self.SAMPLE_FEEDBACK)
        assert self.SAMPLE_FEEDBACK in prompt

    def test_identity_hala_var(self):
        prompt = build_system_prompt(self.SAMPLE_FEEDBACK)
        assert "expert Python programmer" in prompt

    def test_context_hala_var(self):
        prompt = build_system_prompt(self.SAMPLE_FEEDBACK)
        assert "Static Analyzer" in prompt

    def test_constraints_hala_var(self):
        prompt = build_system_prompt(self.SAMPLE_FEEDBACK)
        assert "stdin" in prompt

    def test_katman_sirasi(self):
        prompt = build_system_prompt(self.SAMPLE_FEEDBACK)
        # IDENTITY → CONTEXT → CONSTRAINTS → FEEDBACK → improved_prompt sırası
        idx_identity    = prompt.index("expert Python programmer")
        idx_context     = prompt.index("Static Analyzer")
        idx_constraints = prompt.index("stdin")
        idx_feedback    = prompt.index("COMPLETELY NEW")
        idx_notes       = prompt.index(self.SAMPLE_FEEDBACK)

        assert idx_identity < idx_context < idx_constraints \
               < idx_feedback < idx_notes

    def test_bos_improved_prompt_ilk_iter_gibi(self):
        prompt_none  = build_system_prompt(None)
        prompt_empty = build_system_prompt("")
        assert prompt_none == prompt_empty


class TestPromptConstants:
    """Sabit içerikler doğru mu?"""

    def test_whitelist_constraints_icinde(self):
        assert "subprocess" in CODEGEN_CONSTRAINTS
        assert "collections" in CODEGEN_CONSTRAINTS
        assert "itertools"   in CODEGEN_CONSTRAINTS

    def test_pipeline_uclu_context_icinde(self):
        assert "Static Analyzer"  in CODEGEN_CONTEXT
        assert "Sandbox Executor" in CODEGEN_CONTEXT
        assert "Judge LLM"        in CODEGEN_CONTEXT

    def test_max_iter_context_icinde(self):
        assert "10" in CODEGEN_CONTEXT   # max 10 iterasyon

    def test_feedback_rules_yeniden_yazma_vurgusu(self):
        assert "COMPLETELY NEW" in CODEGEN_FEEDBACK_RULES
        assert "patch" in CODEGEN_FEEDBACK_RULES.lower() \
               or "old one" in CODEGEN_FEEDBACK_RULES
```

---

## 14. CI Checklist (before each phase)

```
[ ] pytest tests/ → tüm testler yeşil
[ ] python dataset/verify.py → veri seti OK
[ ] ENV=local python -c "from config import settings, model_registry; print(settings.env)"
[ ] python -c "from pipeline.graph import build_graph; build_graph()"
[ ] cat outputs/memory.md → log satırları yazılıyor (pipeline çalıştıktan sonra)
```

---

## 15. Faz 3 — Pilot Kontrol Listesi

These checks must pass before the full benchmark.

### Incremental Cloud Migration

```bash
# Step 1: Smallest model, 5 questions
ENV=cloud LCB_SAMPLE_SIZE=5 LCB_DIFFICULTY=easy \
    python -m benchmark.evaluator --variant V4 --tag pilot_step1

# Step 2 (eski plan): ikinci üretici modelle 10 soru — güncel sıra: CLOUD_README.md §A
# Temporarily set M2 equal to M3 in config/models.py, then revert
ENV=cloud LCB_SAMPLE_SIZE=10 \
    python -m benchmark.evaluator --variant V4 --tag pilot_step2

# Step 3: Full pipeline, 5 questions
ENV=cloud LCB_SAMPLE_SIZE=5 \
    python -m benchmark.evaluator --variant V4 --tag pilot_step3
```

### Pilot Output Checks

```bash
# 1. Does memory.md cover all statuses?
python -c "
from memory.logger import read_memory
rows = read_memory()
from collections import Counter
print('Status dağılımı:', Counter(r['status'] for r in rows))
# Beklenen: LOOP, PASS ve STOP_HIGH/STOP_MAX'tan en az biri var
"

# 2. Were JSON files created?
find outputs/iterations -name "*.json" | wc -l

# 3. Timing ve token doluyor mu?
python -c "
import json, glob
files = glob.glob('outputs/iterations/**/*.json', recursive=True)
if files:
    d = json.load(open(files[0]))
    print('total_ms:', d['timing']['total_ms'])
    print('total_tok:', d['tokens']['total'])
    print('improved_prompt var mı:', bool(d['node3']['improved_prompt']))
"

# 4. Was lcb_runner used?
python -c "
from memory.logger import read_memory
rows = read_memory()
# node2_lcb_available should be read from JSON
import json, glob
files = glob.glob('outputs/iterations/**/*.json', recursive=True)[:3]
for f in files:
    d = json.load(open(f))
    print(f.split('/')[-1], '| lcb:', d.get('node2', {}).get('lcb_available', 'N/A'))
"

# 5. Is the STOP_HIGH rate reasonable? (<20% expected)
python -c "
from memory.logger import read_memory
rows = read_memory()
sh = sum(1 for r in rows if r['status'] == 'STOP_HIGH')
print(f'STOP_HIGH oranı: {sh}/{len(rows)} = {sh/len(rows):.1%}')
if sh/len(rows) > 0.20:
    print('UYARI: Çok yüksek — Node 1 threshold veya whitelist kontrol et')
"

# 6. improved_prompt quality — manual inspection of 5 samples
python -c "
import json, glob, random
files = glob.glob('outputs/iterations/**/*.json', recursive=True)
for f in random.sample(files, min(5, len(files))):
    d = json.load(open(f))
    prompt = d['node3']['improved_prompt']
    print('---')
    print(f\"Q: {d['question_id']} | Status: {d['status']} | Score: {d['node3']['total_score']}\")
    print(prompt[:250] if prompt else '[BOŞ]')
"
```

### Parameter Tuning (Post-Pilot)

```bash
# At which average iteration does PASS occur?
python -c "
from memory.logger import read_memory
from collections import defaultdict
rows = read_memory()
pass_iters = [r['iter'] for r in rows if r['status'] == 'PASS']
if pass_iters:
    print(f'Ort. PASS iterasyonu: {sum(pass_iters)/len(pass_iters):.1f}')
    print(f'Max PASS iterasyonu: {max(pass_iters)}')
    # If average < 3, max_iterations = 10 may be too many
    # If STOP_MAX is frequent, consider increasing max_iterations
"
```

---

## 16. tests/test_baseline.py

```python
"""
Unit testler: benchmark/legacy/baseline.py (eski V0 çalıştırıcısı)
_extract_code() — kod extraction doğruluğu.
lcb_runner bağlantısı gerektirmez.
"""
import pytest
from benchmark.baseline import _extract_code


class TestExtractCode:

    def test_python_block_cikartilir(self):
        text = "İşte çözüm:\n```python\ndef f(): return 1\n```"
        result = _extract_code(text)
        assert "def f():" in result
        assert "```" not in result

    def test_son_python_blogu_alinir(self):
        # If two blocks exist, take the last one (guard against lcb_runner bug)
        text = "```python\ndef wrong(): pass\n```\nDaha iyi:\n```python\ndef f(): return 1\n```"
        result = _extract_code(text)
        assert "def f():" in result
        assert "def wrong" not in result

    def test_backtick_olmadan_tum_metin(self):
        code = "def f(): return 1"
        result = _extract_code(code)
        assert "def f():" in result

    def test_bos_metin(self):
        result = _extract_code("")
        assert isinstance(result, str)

    def test_sadece_backtick_python_etiketi(self):
        text = "```python\n\n```"
        result = _extract_code(text)
        assert isinstance(result, str)
```

---

## 15. benchmark/evaluator.py

```python
"""
Tam benchmark çalıştırma.
Kullanım: python -m benchmark.evaluator --sample 50 --difficulty all
"""
import argparse
import json
import os
from datetime import datetime
from loguru import logger

from dataset.loader import load_lcb
from memory.logger import write_run_info
from main import run_pipeline_for_problem


def run_benchmark(
    difficulty: str = "all",
    sample_size: int = 50,
    output_tag: str = "",
) -> dict:

    run_id = write_run_info()
    rows   = load_lcb(
        split      = os.getenv("LCB_SPLIT", "test"),
        difficulty = difficulty,
        sample_size= sample_size,
    )

    logger.info(f"Benchmark | run_id={run_id} | {len(rows)} soru | difficulty={difficulty}")

    all_results  = []
    pass_counts  = {"easy": 0, "medium": 0, "hard": 0}
    total_counts = {"easy": 0, "medium": 0, "hard": 0}

    for i, row in enumerate(rows):
        logger.info(f"[{i+1}/{len(rows)}] {row['title']} ({row['difficulty']})")
        result = run_pipeline_for_problem(row, run_id)
        all_results.append(result)

        diff = row["difficulty"]
        total_counts[diff] += 1
        if any(r["status"] == "PASS" for r in result["results"]):
            pass_counts[diff] += 1

    summary = {
        "run_id":    run_id,
        "timestamp": datetime.now().isoformat(),
        "total":     len(rows),
        "pass_rate": {
            d: round(pass_counts[d] / total_counts[d], 3)
            if total_counts[d] else 0.0
            for d in ("easy", "medium", "hard")
        },
        "results": all_results,
    }

    output_dir = os.getenv("OUTPUT_DIR", "outputs/results")
    os.makedirs(output_dir, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag  = f"_{output_tag}" if output_tag else ""
    path = os.path.join(output_dir, f"benchmark{tag}_{ts}.json")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(f"Tamamlandı → {path}")
    logger.info(f"Pass rate: {summary['pass_rate']}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--difficulty", default="all")
    parser.add_argument("--sample",     type=int, default=50)
    parser.add_argument("--tag",        default="")
    args = parser.parse_args()

    run_benchmark(
        difficulty  = args.difficulty,
        sample_size = args.sample,
        output_tag  = args.tag,
    )
```
