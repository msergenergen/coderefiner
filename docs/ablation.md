# ablation.md — Academic Framework and Ablation Experiments

> **Durum (2026-10-02).** Geçerli deney protokolü **§7**'dedir (sızıntısız ana deney). §1–§6
> özgün planı kayıt için korur; farklar: (i) metrik adı artık pass@1 (aşağıdaki not), (ii) yineleme
> sınırı 10 değil 5, (iii) V0 artık `benchmark/sample_baseline.py` ile problem başına 5 örnek ve
> hattın ilk iterasyon istemiyle üretilir — §4'teki `baseline.py` `benchmark/legacy/` altına
> taşındı (farklı istem, 2048 token; karşılaştırılamaz), (iv) V1/V2 koşuları henüz yapılmadı.

## 1. Research Question and Terminology

### Research Question

> Can a multi-stage iterative refinement pipeline that integrates static code analysis,
> sandboxed execution, and LLM-based judgement mechanisms
> meaningfully improve the LiveCodeBench task success rate of
> selected open-weight models compared to a single-shot baseline?

### Terminology Decisions

> **Güncelleme (2026-10-02):** Raporlarda birincil metrik **pass@1** adıyla verilir: problem
> başına teslim edilen tek kodun tüm gizli testleri geçme oranı. Bu, aşağıdaki
> `TSR_correctness` ile aynı tanımdır; ad, literatürle uyum için değişti (Chen vd. 2021; LCB'nin
> self-repair senaryosu onarım sonrası Pass@1 raporlar, Jain vd. arXiv:2403.07974 Tablo 4).
> Aşağıdaki "neden pass@1 değil" kaygısı — hattın fazladan hesap harcaması — eşit bütçeli
> karşılaştırmalarla (filt@k̄, pass@k̄; Olausson vd., ICLR 2024) ele alınır (§7).

| Term Used | Why | Term Avoided |
|-----------------|-------|-----------------|
| Task Success Rate (TSR) | Pipeline runs iteratively — not a single-shot metric | pass@1 |
| Iterative guided repair | Node feedback is used — not blind retry | self-repair |
| Baseline (single-shot) | Model output without pipeline, single attempt | vanilla |

**Why not pass@1:**
LCB pass@1 measures the rate at which a model gives the correct answer in one attempt.
The pipeline allows up to 10 iterations per model and generates an improved_prompt
with Node 1/2/3 feedback at each iteration. This is test-time compute
with guided repair and is not directly comparable to pass@1.

**İki Katmanlı TSR:**

| Metrik | Kriter | Kullanım Yeri |
|--------|--------|---------------|
| **TSR_correctness** | `final_pass_rate == 1.0` (lcb execution) | V0 ile birincil karşılaştırma; McNemar; H1–H4 |
| **TSR_quality** | `node3_pass == True` (LLM-judge skoru ≥ eşik) | Pipeline optimizasyon hedefi; ikincil analiz |

```
Pipeline TSR_correctness = (final_pass_rate==1.0 olan sorular) / (Toplam sorular)
Pipeline TSR_quality     = (node3_pass==True olan sorular) / (Toplam sorular)
Baseline TSR             = (lcb pass_rate==1.0 olan sorular) / (Toplam sorular)  ← = TSR_correctness
```

`Quality − Correctness gap` pozitifse: judge, lcb'nin geçmediği koda puan veriyor.
Bu fark hem metodolojik şeffaflık hem de tez bulgusu olarak raporlanır.

---

## 2. Ablation Variants

Each variant runs independently. Separate results are obtained for each model per variant.

| Variant | Node 1 | Node 2 | Node 3 | Trend Ext. | Description |
|---------|--------|--------|--------|------------|-------------|
| **V0** — Single-shot baseline | ✗ | ✗ | ✗ | ✗ | Pipeline yok, tek deneme |
| **V1** — Execution + Judgement | ✗ | ✓ | ✓ | ✗ | Test sonuçları + Judgement LLM, statik analiz yok |
| **V2** — Static Analysis + Judgement | ✓ | ✗ | ✓ | ✗ | Statik analiz + Judgement LLM, execution yok |
| **V3** — Full Pipeline | ✓ | ✓ | ✓ | env flag | Tam sistem; `TREND_EXTENSION_ENABLED=true` ile adaptif iterasyon aktif |

### What Each Variant Answers

```
V0 → Modelin doğal yeteneği nedir? (referans taban)
V1 → Execution feedback + Judgement yeterli mi? (Node 1 olmadan)
V2 → Statik analiz + Judgement yeterli mi? (Node 2 olmadan)
V3 → Tüm bileşenler birlikte ne kadar katkı sağlıyor?

V3 - V1 farkı → Node 1'in (statik analiz) marjinal katkısı
V3 - V2 farkı → Node 2'nin (execution) marjinal katkısı
V1 - V0 farkı → Execution + Judgement'ın baseline'a katkısı
V2 - V0 farkı → Static + Judgement'ın baseline'a katkısı
```

### Trend-Aware Extension (V3 özelliği)

`TREND_EXTENSION_ENABLED=true` ile etkinleştirilir — variant değil, V3'ün isteğe bağlı özelliği.
`iteration_gate` max_iterations'a ulaşıldığında stop_max yerine EMA trend analizini çalıştırır;
**tüm 5 koşul sağlanırsa** ek iterasyon verir:

1. `accuracy > 0` — en az bir test geçiyor (hard gate)
2. `pass_rate_slope > 0` — pass rate son 3 iterasyonda artıyor
3. EMA-smoothed `score_slope >= TREND_MIN_SLOPE` — skor gerçekten iyileşiyor
4. `expected_gain = slope × TREND_PROJECTION_WINDOW >= gap` — sabit projeksiyon penceresiyle eşiğe ulaşılabilir
5. `extension_count < MAX_EXTENSION_ITERS` — bütçe tükenmemiş

```bash
# Trend extension aktif (V3 üzerinde)
TREND_EXTENSION_ENABLED=true python -m benchmark.evaluator --variant V3 --sample 50 --tag v3_trend
```

---

## 3. Hipotezler

```
H1: V3 easy TSR ≥ V0 easy TSR + 15 puan
    Gerekçe: Execution + statik analiz + Judgement feedback
             en az 2 iterasyonda yanlış kodu düzeltir

H2: V1 TSR > V0 TSR  (Execution + Judgement baseline'ı geçiyor)
    Gerekçe: failed_cases + improved_prompt, kör yeniden denemeden
             daha fazla bilgi taşır

H3: V2 TSR > V0 TSR  (Static Analysis + Judgement baseline'ı geçiyor)
    Gerekçe: Statik sorunlar düzeltildiğinde kod kalitesi artar
             ve Judgement daha iyi improved_prompt üretir

H4: V3 TSR > V1 TSR  ve  V3 TSR > V2 TSR  (tam pipeline ablasyonları geçiyor)
    Gerekçe: Her iki node'un birlikte çalışması daha zengin feedback sağlar
```

### Statistical Test

**McNemar test** — for paired binary outcomes.
A pair is formed per question: (pipeline PASS?, baseline PASS?)

```python
from statsmodels.stats.contingency_tables import mcnemar

# Tablo:
#         Baseline PASS  Baseline FAIL
# Pipeline PASS    a           b
# Pipeline FAIL    c           d

# McNemar: based on b and c cells
# p < 0.05 → difference is statistically significant

result = mcnemar([[a, b], [c, d]], exact=True)
print(f"p-value: {result.pvalue:.4f}")
print(f"Odds Ratio: {b/c:.2f}" if c > 0 else "OR: inf")
```

**95% Confidence Interval** — Wilson score interval for TSR:

```python
from statsmodels.stats.proportion import proportion_confint

lo, hi = proportion_confint(
    count=passed, nobs=total, alpha=0.05, method="wilson"
)
print(f"TSR: {passed/total:.3f} [{lo:.3f}, {hi:.3f}]")
```

---

## 4. benchmark/baseline.py (legacy → `benchmark/legacy/baseline.py`)

```python
"""
Single-shot baseline çalıştırma — V0 varyantı.

Her model için pipeline olmadan, tek denemeli kod üretimi yapar.
Sonuçlar ablasyon karşılaştırması için kullanılır.

Kullanım:
    python benchmark/baseline.py --model M1 --sample 50 --tag v0_M1
    python benchmark/baseline.py --model M2 --sample 50 --tag v0_M2
    python benchmark/baseline.py --model M3 --sample 50 --tag v0_M3
"""
import argparse
import json
import os
import time
from datetime import datetime
from loguru import logger

from config import settings, model_registry
from dataset.loader import load_lcb
from dataset.splitter import format_problem
from dataset.edge_cases import augment_row
from models.client import get_client, get_model_name, get_temperature
from sandbox.lcb_eval import lcb_evaluate
from memory.logger import write_run_info


def run_single_shot(
    problem_row: dict,
    model_id: str,
    run_id: str,
) -> dict:
    """
    Tek model için pipeline olmadan tek denemeli kod üretir.
    """
    aug_row    = augment_row(problem_row)
    edge_cases = aug_row.get("edge_cases", [])
    prompt     = format_problem(problem_row)

    cfg    = model_registry.get(model_id)
    client = get_client(model_id)

    t0 = time.perf_counter()

    try:
        response = client.chat.completions.create(
            model=cfg.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=cfg.temperature,
            max_tokens=2048,
        )
        code = _extract_code(response.choices[0].message.content)
        tokens = {
            "prompt":     response.usage.prompt_tokens     if response.usage else 0,
            "completion": response.usage.completion_tokens if response.usage else 0,
        }
    except Exception as e:
        logger.warning(f"[{model_id}] Codegen hatası: {e}")
        code   = ""
        tokens = {"prompt": 0, "completion": 0}

    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    # Correctness evaluation — lcb_runner or fallback
    eval_result = lcb_evaluate(
        code=code,
        problem_row=aug_row,
        edge_cases=edge_cases,
        timeout=settings.sandbox_timeout_sec,
    )

    passed = eval_result["pass_rate"] == 1.0 and eval_result["total_count"] > 0

    return {
        "run_id":      run_id,
        "model":       model_id,
        "model_name":  cfg.model_name,
        "question_id": problem_row["id"],
        "difficulty":  problem_row["difficulty"],
        "variant":     "V0",
        "passed":      passed,
        "pass_rate":   eval_result["pass_rate"],
        "passed_count": eval_result["passed_count"],
        "total_count":  eval_result["total_count"],
        "elapsed_ms":  elapsed_ms,
        "tokens":      tokens,
        "code":        code,
        "failed_cases": eval_result["failed_cases"],
    }


def _extract_code(text: str) -> str:
    """
    LLM çıktısından ```python ... ``` bloğunu çıkarır.
    Bulunamazsa tüm metni döner.
    """
    import re
    # Take the last ```python...``` block (guard against lcb_runner bug)
    matches = re.findall(r"```python\s*(.*?)```", text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    # Backtick block only (no python tag)
    matches = re.findall(r"```\s*(.*?)```", text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return text.strip()


def run_baseline(
    model_id: str,
    difficulty: str = "all",
    sample_size: int = 50,
    output_tag: str = "",
) -> dict:

    run_id = write_run_info()
    rows   = load_lcb(
        split       = settings.lcb_split,
        difficulty  = difficulty,
        sample_size = sample_size,
    )

    logger.info(
        f"Baseline | run_id={run_id} | model={model_id} | "
        f"{len(rows)} soru | difficulty={difficulty}"
    )

    results      = []
    pass_counts  = {"easy": 0, "medium": 0, "hard": 0}
    total_counts = {"easy": 0, "medium": 0, "hard": 0}

    for i, row in enumerate(rows):
        logger.info(f"[{i+1}/{len(rows)}] {row['title']} ({row['difficulty']})")
        result = run_single_shot(row, model_id, run_id)
        results.append(result)

        diff = row["difficulty"]
        total_counts[diff] += 1
        if result["passed"]:
            pass_counts[diff] += 1

    summary = {
        "run_id":    run_id,
        "variant":   "V0",
        "model":     model_id,
        "timestamp": datetime.now().isoformat(),
        "total":     len(rows),
        "tsr": {
            d: round(pass_counts[d] / total_counts[d], 3)
            if total_counts[d] else 0.0
            for d in ("easy", "medium", "hard")
        },
        "results": results,
    }

    output_dir = settings.output_dir
    os.makedirs(output_dir, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag  = f"_{output_tag}" if output_tag else ""
    path = os.path.join(output_dir, f"baseline{tag}_{ts}.json")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(f"Tamamlandı → {path}")
    logger.info(f"TSR: {summary['tsr']}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM-as-a-Judge — V0 Single-Shot Baseline")
    parser.add_argument("--model",      required=True, choices=["M1", "M2", "M3"])
    parser.add_argument("--difficulty", default="all")
    parser.add_argument("--sample",     type=int, default=50)
    parser.add_argument("--tag",        default="")
    args = parser.parse_args()

    run_baseline(
        model_id   = args.model,
        difficulty = args.difficulty,
        sample_size= args.sample,
        output_tag = args.tag,
    )
```

---

## 5. benchmark/evaluator.py — Variant Support

The difference between V1–V3 is managed via the `build_graph(variant=...)` call.

```python
# benchmark/evaluator.py — CLI argument

parser.add_argument(
    "--variant",
    default="V3",
    choices=["V1", "V2", "V3"],
    help=(
        "V1: Execution + Judgement (Node 2 + Node 3)  |  "
        "V2: Static Analysis + Judgement (Node 1 + Node 3)  |  "
        "V3: Full pipeline (Node 1 + Node 2 + Node 3)"
    ),
)
```

```python
# pipeline/graph.py — build_graph() variant support

def build_graph(variant: str = "V3") -> StateGraph:
    """
    variant:
        V1 — Node 2 + Node 3  (Execution + Judgement, no static analysis)
        V2 — Node 1 + Node 3  (Static Analysis + Judgement, no execution)
        V3 — Full pipeline     (Node 1 + Node 2 + Node 3)
    """
    use_node1 = variant in ("V2", "V3")
    use_node2 = variant in ("V1", "V3")

    # Core nodes always present: codegen, judgement, log_iteration, etc.
    # Node 1 (static_analysis, severity_gate) added if use_node1
    # Node 2 (sandbox, strip_comments)        added if use_node2
    # V2 routing: codegen → static_analysis → strip_before_judge → judgement
    # V1 routing: codegen → sandbox → strip_comments → judgement
    # V3 routing: codegen → static_analysis → sandbox/strip_before_judge → judgement
```

PASS criterion is identical across all variants: `node3_total_score ≥ threshold`
(Judgement always runs — V0 is the only variant without it, and V0 uses `baseline.py`).

### benchmark_V*.json — Output Format

Each evaluator run writes a JSON file with two TSR layers:

```json
{
  "run_id":    "run_20250413_143000",
  "variant":   "V3",
  "timestamp": "2025-04-13T14:30:00",
  "total":     50,
  "tsr_quality": {
    "easy":   0.880,
    "medium": 0.720,
    "hard":   0.600
  },
  "tsr_correctness": {
    "easy":   0.800,
    "medium": 0.640,
    "hard":   0.520
  },
  "results": [
    {
      "problem_id": "q_001",
      "difficulty": "easy",
      "results": [
        { "model": "M1", "status": "PASS", "score": 84.0, "final_pass_rate": 0.95 },
        { "model": "M2", "status": "PASS", "score": 91.0, "final_pass_rate": 1.00 }
      ]
    }
  ]
}
```

- **`tsr_quality`** — `node3_pass == True` (LLM-judge criterion). Pipeline optimization target.
- **`tsr_correctness`** — `final_pass_rate == 1.0` (lcb execution criterion). Primary comparison metric; apples-to-apples with V0 `"tsr"`.
- V0 baseline JSON uses `"tsr"` (correctness-only, single model, no judge).

---

## 6. Experiment Run Commands

Ana deneyin komutları §7'dedir (`scripts/run_openrouter.sh`). Özgün plandaki model başına
`baseline.py` + V1/V2/V3 komutları (M1/M2/M3, 50 soru) artık kullanılmıyor.

V1/V2 ablasyonu isteğe bağlıdır ve bütçe kalırsa ana deneyin problem kümesinde koşulur
(`benchmark/evaluator.py --variant V1|V2`; aynı `config/openrouter.env` ayarlarıyla).
Bunun için çalıştırma betiğine bir alt komut eklenmesi gerekir; henüz eklenmedi.

---

## 7. Sızıntısız ana deney planı (2026-10-02)

Gerekçe: `docs/dataset.md` §10. Bu bölüm OpenRouter üzerinden yapılacak ana deneyin
protokolüdür; yerel Qwen3.5-9B koşuları ek bulgu olarak kalır.

### Modeller

| Rol | Model | Yayın | LCB referansı |
|---|---|---|---|
| Üretici | DeepSeek-V3 (`deepseek/deepseek-chat`) | 2024-12-26 | Pass@1 37,6 (LCB 2024-08..11; DeepSeek-V3 raporu, arXiv:2412.19437) |
| Yargıç | Qwen2.5-72B-Instruct (farklı aile) | 2024-09-19 | Pass@1 28,7 (aynı tablo) |

İlk plan Qwen2.5-Coder-32B (LCB 2407–2409'da 31,4; arXiv:2409.12186) + Llama-3.3-70B idi. 20
problemlik pilotta pass@1 %12,2 çıktı (kolay %44, orta ve zor %0) ve Cloudflare uç noktası
örneklerin ~%9'unda `[]` belirteçlerini düşürüyordu (`x =` sonrası boş). Taban etkisi ve sağlayıcı
hatası nedeniyle, ana koşulardan **önce** üretici 2025 öncesi yayımlanmış en güçlü erişilebilir
modelle değiştirildi; bu değişiklik ve pilot sonucu yöntem bölümünde açıkça yazılır. Yargıç
üreticiden farklı ailedendir ve LCB'de üreticiden zayıftır; çözümü kendisi yazarak kazanç
yaratması daha az olasıdır. Örnekleme: DeepSeek-V3 için resmî öneri yok, model kartındaki
demo komutunun T = 0,7'si kullanıldı; Qwen2.5-72B `generation_config.json` değerleriyle.
Uç noktalar: DeepSeek-V3 DeepInfra **fp4**, Qwen2.5-72B Novita bf16; yöntem bölümünde belirtilir.

### Problem kümesi
LCB v6, `contest_date` 2025-01 – 2025-04 (182 problem; kolay 45, orta 55, zor 82); tez
planındaki 150 soruluk ölçek için bunlardan seed 42 ile 150 problem (kolay 40, orta 38, zor 72).
Tümü her iki modelin yayın tarihinden sonra.

### Notlama
Tek değerlendirici (`sandbox/lcb_eval.py`, yerel). Açık testler: test başına 10 s (Düğüm 2 ve
V0'ın filt@k'sı aynı değerle). Gizli testler: test başına 10 s (`HIDDEN_TIMEOUT_SEC`) ve ilk
başarısız testte durma; lcb_runner'ın `grade_stdio`/`grade_call_based` işlevleri de böyle yapar
(LiveCodeBench deposu, `lcb_runner/evaluation/testing_util.py`). Bu, problemin geçti/kaldı
sonucunu değiştirmez; yalnızca kısmi gizli geçme oranı alt sınır olur. V0 ve V3 birebir aynı
kurallarla notlanır. İlk V0 koşusu açık testleri 15 s ile notlayarak başladı; kayıtlar
`sample_baseline --regrade` ile (API çağrısı olmadan) aynı kurallarla yeniden notlanır.

### Model lisansları
Modellerin hepsi açık ağırlıklıdır: DeepSeek-V3 (kod MIT, model DeepSeek Model License, ticari
kullanım serbest; HF deepseek-ai/DeepSeek-V3), Qwen2.5-72B-Instruct (Qwen License, HF
Qwen/Qwen2.5-72B-Instruct), ek düzendeki Qwen3.5-9B (Apache-2.0). Çıkarım OpenRouter üzerinden barındırılan
sağlayıcılarla yapılır; aynı ağırlıklar yerelde/kiralık GPU'da çalıştırılarak tekrarlanabilir.

### Ölçüm
- **Birincil:** pass@1 = problem başına teslim edilen tek nihai kodun gizli testlerin tamamını
  geçme oranı (Chen vd. 2021; LCB self-repair senaryosu da onarım sonrası Pass@1 raporlar).
- **Taban (V0):** hattın 1. iterasyonuyla aynı istemle problem başına n=5 bağımsız örnek;
  pass@1 = c/n ortalaması (yansız kestirici).
- **Hat (V3):** en fazla 5 iterasyon, eğilim uzatması kapalı (`MAX_ITERATIONS=5`,
  `TREND_EXTENSION_ENABLED=false`); Y (`verdict`) ve P (`rubric`) tasarımları.
- **Eşit bütçe karşılaştırmaları** (Olausson vd., ICLR 2024): aynı 5 örnekten (i) açık testlere
  göre seçilen ilk kod (kâhinsiz, gerçekçi rakip), (ii) pass@k̄ (k̄ = hattın ortalama üretim
  sayısı; gizli testle seçtiği için üst sınır).
- **Maliyet:** çözülen problem başına token ve $ (`outputs/api_usage.jsonl`).
- **Kontroller:** yargıç modelinin tek başına pass@1'i; yargıç yönergelerinde kod parçası
  sayımı (çözüm sızıntısı denetimi); zorluk düzeyine göre ayrı raporlama.

### Komutlar
```bash
scripts/run_openrouter.sh smoke          # hat, 2 problem
scripts/run_openrouter.sh pilot 20       # taban çizgisi, 20 problem × 5 örnek (floor denetimi)
scripts/run_openrouter.sh baseline       # taban çizgisi, 150 problem × 5 örnek
scripts/run_openrouter.sh run            # hat, 150 problem × {verdict, rubric}, tasarım başına 4 süreç
scripts/run_openrouter.sh status         # ilerleme + harcama
```
Taban çizgisi `benchmark/sample_baseline.py` ile üretilir (Langfuse açıksa her örnek ayrı bir iz,
`public_pass_rate` ve `hidden_pass` skorlarıyla; `--regrade <jsonl>`: kayıtlı
örnekleri API çağrısı olmadan güncel notlama kurallarıyla yeniden notlar): istem, hattın 1. iterasyonuyla
birebir aynıdır (`build_system_prompt(None)` + `build_user_prompt(format_problem(row))`); her
örnek açık ve tüm testlerle yerel LCB uyumlu değerlendiriciyle notlanır. Çıktı:
`outputs/openrouter/results/baseline_samples/<run_id>.jsonl`; özet pass@1, pass@k ve filt@k
(k=1..5) değerlerini zorluk düzeyine göre verir. Problem sırası hat ve taban çizgisinde aynıdır
(pencere + seed 42).

### Koşu kayıtları (2026-10-02)
| Koşu | run_id | Onarım istemi |
|---|---|---|
| V0 taban çizgisi | `run_20261002_091834` (`.regraded.jsonl`) | — |
| Y, yalnız geri bildirim (A) | `run_20261002_091834/36/38/40` | `FEEDBACK_INCLUDE_CODE=false` davranışı |
| P, yalnız geri bildirim (A) | `run_20261002_091842/44/46/48` | aynı |
| Y, geri bildirim + önceki çözüm (B, ana) | `run_20261002_100638/40/42/44` | `FEEDBACK_INCLUDE_CODE=true` |
| P, geri bildirim + önceki çözüm (B, ana) | `run_20261002_100646/48/50/52` | aynı |

A koşularının yarısında yalnızca 2 problem 1. iterasyonda yanlıştan sonda doğruya geçti; yargıcın
"kodun X girdisinde Y döndürüyor" geri bildirimi üreticinin göremediği koda atıf yapıyordu.
LCB'nin self-repair istemi soruyu, düzeltilecek kodu ve hata geri bildirimini birlikte verir
(Jain vd., arXiv:2403.07974, Ek C.3). B koşuları bu biçimi izler (`docs/codegen-models.md`).
A koşuları ablasyon olarak raporlanır (önceki çözümün katkısı); 1. iterasyon istemi her iki
düzende aynıdır. Not: `run_20261002_091834` hem V0 hem Y-A'nın bir parçasıdır (aynı saniyede
başladılar); maliyet `phase` etiketiyle ayrılır.

### Taban etkisi (floor) denetimi
Pencerenin %45'i zor problem. İlk pilot (Coder-32B) genel pass@1 %12,2 verdi → üretici
DeepSeek-V3'e geçildi (yukarıda). Yeni çiftle ayrı pilot yapılmadı; 6 problemlik prob koşusu
istem/karar denetimi için kullanıldı (`docs/node3.md` §4.4).
