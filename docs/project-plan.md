# LLM-as-a-Judge for Iterative Code Refinement
## Master's Thesis Work Plan

---

## Status Indicators

| Indicator | Meaning |
|--------|-------|
| ✅ | Documentation complete |
| 🔲 | Not yet documented |
| `[x]` | Implementation step complete |
| `[ ]` | Implementation step pending |

> Documentation ✅ = the relevant `docs/*.md` file exists and is up to date.
> Implementation `[ ]` = real code not yet written; will be checked off when the project begins.

---

## Research Question

> Can a multi-stage iterative refinement pipeline integrating static code analysis,
> sandboxed execution, and LLM-based judgement mechanisms
> meaningfully improve the LiveCodeBench task success rate of
> selected open-weight models compared to a single-shot baseline?

---

## ✅ Phase 0 — Academic Framework

> Document: `docs/ablation.md`

### Tasks

**0.1 Literature positioning**
- Document the difference from the LCB self-repair scenario
  (LCB: only error message → ours: static analysis + runtime + rubric)
- Clarify the difference from iterative repair approaches in CodeRL, AlphaCode, self-debugging
- "pass@1" yerine "task success rate with iterative repair" kullan —
  define this in the terminology section of the thesis

**0.2 Ablation design — core of the thesis**

Run the following 4 variants separately for each model:

| Variant | Node 1 | Node 2 | Node 3 | Description |
|---------|--------|--------|--------|----------|
| V0 — Single-shot | ✗ | ✗ | ✗ | Vanilla baseline, tek deneme |
| V1 — Exec+Judge | ✗ | ✓ | ✓ | Test execution + Judgement LLM |
| V2 — Static+Judge | ✓ | ✗ | ✓ | Static analysis + Judgement LLM |
| V3 — Full Pipeline | ✓ | ✓ | ✓ | Tam sistem |

**0.3 Hipotezler**

- H1: V3 task success rate ≥ V0 + %15 puan (easy sorularda)
- H2: V1 TSR > V0 TSR (Execution+Judgement improves single-shot)
- H3: V3 TSR > V1 TSR and V3 TSR > V2 TSR (full pipeline is best)
- H4: Due to ceiling effect, gain on easy questions < gain on hard questions
- H5: Pipeline contribution is lower in models with high STOP_HIGH rate

---

## ✅ Phase 1 — Environment Setup

> Document: `docs/setup.md`
> Estimated duration: 1–2 days

### 1.1 Mac Mini M4 (Local Debug)

```
[ ] Xcode CLI Tools kurulumu
[ ] Homebrew + Python 3.11 + cmake
[ ] llama.cpp Metal derleme
[ ] Proje repo clone + venv + pip install
[ ] .env.local oluşturma (ENV=local)
[ ] Qwen2.5-Coder-7B indirme (~4 GB)
[ ] llama-server port 8001 başlatma
[ ] ENV=local python -c "from config import settings; print(settings.env)"
[ ] python dataset/verify.py
```

### 1.2 Vast.ai A100 80GB (Cloud)

```
[ ] Reliability > 0.95, 80GB VRAM, 300GB SSD instance seç
[ ] SSH bağlantısı + tmux kurulumu
[ ] Sistem paketleri (build-essential, cmake, python3.11-dev)
[ ] llama.cpp CUDA derleme
[ ] Proje clone + venv + pip install
[ ] lcb_runner kurulumu (pip install -e LiveCodeBench/)
[ ] .env.cloud oluşturma (ENV=cloud)
[ ] Model indirme (4 model, ~99 GB toplam, tmux paralel)
[ ] scripts/start_model.sh M1 → curl test
[ ] ENV=cloud python -c "from config import settings; print(settings.env)"
```

### 1.3 Verification

```
[ ] pytest tests/ -v --timeout=30  → 69 test yeşil
[ ] python dataset/verify.py       → LCB yükleniyor
[ ] python -c "from pipeline.graph import build_graph; build_graph()"
```

---

## ✅ Phase 2 — Implementation

> Documents: all `docs/*.md`
> Estimated duration: 3–4 weeks
> Verify locally (ENV=local) first, then move to cloud

### 2.1 Core Infrastructure ✅

> `docs/config.md` · `docs/dataset.md`

```
[ ] config/pipeline.py    — ENV flag, settings
[ ] config/models.py      — _cloud_registry(), _local_registry()
[ ] config/thresholds.py  — CC, MI, pass threshold'ları
[ ] config/__init__.py    — build_registry(env)
[ ] pipeline/state.py     — PipelineState TypedDict (tüm alanlar)
[ ] dataset/loader.py     — load_lcb(), _clean_html(), all_tests birleştirme
[ ] dataset/splitter.py   — format_problem(), dataset_stats()
[ ] dataset/edge_cases.py — needs_edge_cases(), generate_edge_cases()
[ ] dataset/verify.py     — kurulum doğrulama scripti
```

Checkpoint:
```bash
ENV=local python dataset/verify.py
# Output: public=2, private=8, all=10, edge_needed=True/False
```

### 2.2 Node 1 — Statik Analiz ✅

> `docs/node1.md`

```
[ ] sandbox/whitelist.py          — import whitelist
[ ] metrics/complexity.py         — CC, MI, Halstead, Cognitive
[ ] metrics/security.py           — bandit_scan(), check_imports()
[ ] nodes/static_analysis.py      — static_analysis_node()
[ ] nodes/cleaner.py              — strip_comments(), strip_comments_node()
```

Checkpoint:
```bash
pytest tests/test_metrics.py tests/test_cleaner.py -v
```

### 2.3 Node 2 — Sandbox ✅

> `docs/node2.md`

```
[ ] sandbox/lcb_eval.py   — _compute_diversity(), lcb_evaluate(), fallback
[ ] sandbox/runner.py     — measure_runtime() (cpu_ms, ram_mb)
[ ] nodes/sandbox.py      — sandbox_node() — iki fazı birleştir
```

Checkpoint:
```bash
pytest tests/test_lcb_eval.py tests/test_runtime.py -v
```

### 2.4 Node 3 — Judgement ✅

> `docs/node3.md`

```
[ ] metrics/scoring.py    — compute_numeric_scores(), rubric
[ ] nodes/judgement.py     — judgement_node(), JUDGEMENT_SYSTEM_PROMPT
                            _get_llm_judgement(), fallback
```

Checkpoint:
```bash
ENV=local python -c "from nodes.judgement import judgement_node; print('OK')"
```

### 2.5 Pipeline Orkestrasyonu ✅

> `docs/graph.md` · `docs/codegen-models.md` · `docs/memory-logging.md`

```
[ ] pipeline/gates.py     — severity_gate(), iteration_gate()
[ ] pipeline/router.py    — advance_model(), MODEL_SEQUENCE
[ ] pipeline/graph.py     — build_graph(variant=) — 10 node, tüm edge'ler
[ ] models/client.py      — get_client(), generate()
[ ] models/prompts.py     — CODEGEN_IDENTITY, CODEGEN_CONTEXT,
                            CODEGEN_CONSTRAINTS, CODEGEN_FEEDBACK_RULES
                            build_system_prompt(improved_prompt)
[ ] nodes/codegen.py      — codegen_node() (timing + token)
[ ] memory/logger.py      — write_run_info(), log_iteration_node(),
                            log_severity_stop_node(), read_memory()
[ ] main.py               — run_pipeline_for_problem(), main()
```

Checkpoint — Local end-to-end test:
```bash
ENV=local LCB_SAMPLE_SIZE=1 LCB_DIFFICULTY=easy python main.py
# outputs/memory.md      → en az 1 satır
# outputs/run_info.json  → oluştu
# outputs/iterations/q_xxx/M1_iter1.json → oluştu
```

### 2.6 Benchmark Infrastructure ✅

> `docs/ablation.md` · `docs/analysis.md`

```
[ ] benchmark/baseline.py   — single-shot baseline çalıştırma (V0)
[ ] benchmark/evaluator.py  — run_benchmark(), --variant flag desteği
[ ] benchmark/analysis.py   — compute_tsr(), mcnemar_test(), raporlama
```

### 2.7 Testler ✅

> `docs/testing.md`

```
[ ] tests/conftest.py          — paylaşılan fixture'lar
[ ] tests/test_cleaner.py      — strip_comments()
[ ] tests/test_metrics.py      — CC, MI, severity kararı
[ ] tests/test_gates.py        — severity_gate(), iteration_gate()
[ ] tests/test_lcb_eval.py     — _compute_diversity(), fallback
[ ] tests/test_runtime.py      — measure_runtime()
[ ] tests/test_edge_cases.py   — needs_edge_cases(), augment_row()
[ ] tests/test_dataset.py      — _clean_html(), format_problem()
[ ] tests/test_logger.py       — memory.md yazma/okuma, JSON
[ ] tests/test_nodes.py        — static_analysis_node(), cleaner
[ ] tests/test_baseline.py     — _extract_code()
```

---

## ✅ Phase 3 — Validation (Pilot)

> Document: `docs/testing.md` (Section 15 — Pilot Checklist)
> Estimated duration: 3–5 days

### 3.1 Incremental Cloud Migration

```
[ ] Adım 1: ENV=cloud, sadece M3 (GLM-4.7-Flash), 5 easy soru
[ ] Adım 2: ENV=cloud, M1 + M3, 10 soru
[ ] Adım 3: ENV=cloud, tam pipeline (M1+M2+M3+QWEN), 5 soru
```

### 3.2 Pilot Kontrolleri

```
[ ] memory.md tüm durumları içeriyor mu? (LOOP/PASS/STOP_HIGH/STOP_MAX)
[ ] Her iterasyonda JSON oluşuyor mu?
[ ] timing ve tokens alanları doluyor mu?
[ ] improved_prompt mantıklı mı? (manuel inceleme — 5 örnek)
[ ] lcb_runner çalışıyor mu? (node2_lcb_available=True)
[ ] STOP_HIGH oranı makul mü? (< %20 olmalı)
```

### 3.3 Parameter Tuning

```
[ ] max_iterations ayarı — 10 mu fazla?
[ ] PASS threshold'ları — 80/70/60 doğru mu?
[ ] Edge case threshold — 8 mi yoksa 5 mi?
[ ] Judgement prompt kalitesi — improved_prompt actionable mı?
```

---

## ✅ Phase 4 — Main Experiments

> Document: `docs/ablation.md` (Section 6 — Run Commands)
> Estimated duration: 1–2 weeks

### 4.1 Baseline — V0 (Separately per model)

```
[ ] ENV=cloud python benchmark/baseline.py --model M1 --sample 50 --tag v0_M1
[ ] ENV=cloud python benchmark/baseline.py --model M2 --sample 50 --tag v0_M2
[ ] ENV=cloud python benchmark/baseline.py --model M3 --sample 50 --tag v0_M3
[ ] Sonuçları yerel makineye kopyala (scp)
```

### 4.2 Ablation Variants

```
[ ] V1: ENV=cloud python -m benchmark.evaluator --variant V1 --sample 50 --tag v1
[ ] V2: ENV=cloud python -m benchmark.evaluator --variant V2 --sample 50 --tag v2
[ ] Sonuçları yerel makineye kopyala (scp)
```

### 4.3 Tam Pipeline — V3

```
[ ] ENV=cloud python -m benchmark.evaluator --variant V3 --sample 50 --tag v3_full
[ ] Sonuçları yerel makineye kopyala (scp)
```

### 4.4 Difficulty-Based Analysis

```
[ ] V3 easy:   --difficulty easy   --sample 25 --tag v3_easy
[ ] V3 medium: --difficulty medium --sample 15 --tag v3_medium
[ ] V3 hard:   --difficulty hard   --sample 10 --tag v3_hard
```

---

## ✅ Phase 5 — Analysis and Findings

> Document: `docs/analysis.md`
> Estimated duration: 1 week

### 5.1 Metric Computation

```
[ ] python benchmark/analysis.py --results-dir outputs/results/
[ ] TSR tablosu (zorluk × varyant)
[ ] Model bazlı karşılaştırma tablosu
[ ] %95 Wilson güven aralıkları
```

### 5.2 Statistical Tests

```
[ ] H1: V3 easy TSR ≥ V0 + 15 puan?
[ ] H2: McNemar testi — V1 vs V0 (p < 0.05?)
[ ] H3: V3 TSR > V1 TSR ve V3 TSR > V2 TSR?
[ ] H4: Easy/Hard kazanım grafiği — tavan etkisi var mı?
[ ] H5: STOP_HIGH oranı yüksek modelde pipeline katkısı düşük mü?
```

### 5.3 Error Analysis

```
[ ] STOP_MAX sorular — neden geçemedi?
[ ] improved_prompt kalite kontrolü — 20 örnek manuel inceleme
[ ] Node 1 vs Node 2 katkı analizi
[ ] Ortalama skor artışı (iter 1 → son iter)
```

---

## ✅ Phase 6 — Academic Paper

> **Document:** `docs/paper.md` (IEEE iki sütun, Türkçe, 4 sayfa — taslak hazır)
> **GitHub:** `llm-code-refinement`
> **Estimated duration:** 2–3 weeks after Phase 5 is complete

### Paper Identity

**Turkish title:**
> Yinelemeli Kod Arıtımında Yargıç Olarak BDM: Çok Aşamalı Bir Yaklaşım

**English title:**
> LLM-as-a-Judge for Iterative Code Refinement: A Multi-Stage Approach

**Keywords (Turkish):** büyük dil modeli, kod arıtımı, yargıç olarak BDM, iteratif geri bildirim, statik analiz

**Keywords (English):** large language model, code refinement, LLM-as-a-judge, iterative feedback, static analysis

---

### Paper Status

| Bölüm | Durum |
|-------|-------|
| Özetçe + Abstract | ✅ Yazıldı |
| I. Giriş | ✅ Yazıldı |
| II. İlgili Çalışmalar | ✅ Yazıldı |
| III. Önerilen Sistem | ✅ Yazıldı |
| IV. Deneysel Kurulum | ✅ Yazıldı |
| V. Bulgular (Tablo III, IV, Şekil 2) | 🔲 Cloud sonrası doldurulacak |
| VI. Sonuç | ✅ Yazıldı |
| Kaynaklar | ✅ 18 referans |

---

### Post-Cloud Writing Checklist

```
[ ] Cloud sonuçlarını docs/paper.md Bölüm V'e ekle
[ ] Tablo III (ablation TSR): V0–V3 × easy/medium/hard
[ ] Tablo IV (model karşılaştırma): M1/M2/M3 V0 vs V3
[ ] Şekil 2 (ablation bar chart): benchmark/plot.py ile üret
[ ] H1–H5 hipotez sonuçlarını hipotez tablosuna yaz
[ ] Özetçe'deki "dört varyantlı ablasyon" rakamını doğrula
[ ] benchmark/plot.py çalıştır → figures/ dizinine kaydet
[ ] Paper için LaTeX veya Word şablonuna aktar
```

---

## Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|----------|-------|
| Judgement model miscalibrated | Medium | Manual quality check: 20 samples |
| Ceiling effect on easy questions | High | Focus on Medium/Hard results |
| lcb_runner API change | Low | Pin version, fallback ready |
| Vast.ai instance instability | Medium | Filter Reliability > 0.95 |
| 50-question statistical power | High | Use McNemar test + report 95% CI |
| M1→M2→M3 causality issue | High | Run V0 ablation separately per model |

---

## Cost Estimate

| Phase | Duration | Vast.ai (~$0.80/hr A100) |
|-------|------|--------------------------|
| Pilot (Faz 3) | ~3 sa | ~$2.40 |
| V0 baseline × 3 model | ~6 sa | ~$4.80 |
| V1–V3 ablasyon | ~18 sa | ~$14.40 |
| V4 tam pipeline | ~13 sa | ~$10.40 |
| Difficulty-based runs | ~8 hrs | ~$6.40 |
| **Toplam** | **~48 sa** | **~$38–45** |

---

## Critical Success Factors

1. **Ablation comes first.** The thesis cannot be defended without V0 baseline.

2. **improved_prompt kalitesini manuel kontrol et.**
   Judge 20 samples — is it producing actionable feedback?

3. **Save checkpoints.** After every major run,
   copy `outputs/` to your local machine.

4. **Prepare for ceiling effect.** Small gain on easy questions is not failure —
   present it as a "ceiling effect" section in the thesis.

5. **Negative results are also valuable.** The question "why did it not contribute?"
   may become the thesis's most valuable finding.
