# Metrik Sistemi — Tam Referans

Bu doküman pipeline'daki tüm metriklerin değer aralıklarını, ağırlıklarını ve başarı oranına etkilerini açıklar.

---

## Node 1 — Static Analysis

| Metrik | Araç | İdeal | Uyarı Eşiği | Kritik Eşik | Severity Etkisi | Skor Etkisi |
|--------|------|-------|-------------|-------------|-----------------|-------------|
| **Cyclomatic Complexity (CC)** | radon | ≤ 5 | > 10 → issue | > 20 → HIGH | HIGH: pipeline durur | quality skora girer |
| **Maintainability Index (MI)** | radon | ≥ 85 | < 65 → issue | < 40 → LOW | LOW: Node 2 atlanır | quality skora girer |
| **Halstead Volume** | radon | ≤ 500 | > 1000 → issue | > 2000 → LOW | LOW: Node 2 atlanır | skora girmez |
| **Cognitive Complexity** | lizard | — | — | — | etkisiz | skora girmez |
| **Bandit HIGH** | bandit | 0 | — | ≥ 1 → HIGH | HIGH: pipeline durur | security skora girer |
| **Bandit MEDIUM** | bandit | 0 | ≥ 1 → issue | — | etkisiz | security skora girer (−3p/bulgu) |
| **Forbidden import** | AST | — | — | ≥ 1 → HIGH | HIGH: pipeline durur | skora girmez |
| **mypy hatası** | mypy | 0 | ≥ 1 → issue | — | etkisiz | skora girmez |
| **ruff hatası** | ruff | 0 | ≥ 1 → issue | — | etkisiz | skora girmez |

### Severity Kararı (öncelik sırası: HIGH > LOW > OK)

| Severity | Koşul | Sonuç |
|----------|-------|-------|
| **HIGH** | bandit_high ≥ 1 veya forbidden_import veya CC > 20 | Pipeline durur, Node 2/3 çalışmaz |
| **LOW** | MI < 40 veya Halstead > 2000 | Node 2 atlanır, doğrudan Node 3'e gider |
| **OK** | Yukarıdakilerin hiçbiri | Normal akış: Node 2 → Node 3 |

### Akademik Referanslar

| Metrik | Referans |
|--------|----------|
| CC eşiği (≤10) | McCabe, T.J. (1976). *A Complexity Measure.* IEEE TSE |
| MI eşiği (65/85) | Coleman et al. (1994). *Using Metrics to Evaluate Software System Maintainability.* IEEE Computer |
| Halstead Volume | Halstead, M.H. (1977). *Elements of Software Science.* Elsevier |
| Bandit | OWASP / CWE standartları |

---

## Node 2 — Sandbox & Execution

| Metrik | Araç | İdeal | Uyarı Eşiği | Skor Etkisi | Başarıya Etkisi |
|--------|------|-------|-------------|-------------|-----------------|
| **pass_rate** | yerel LCB uyumlu değerlendirici (açık testler) | 1.0 | easy ≥ 0.90 / med ≥ 0.75 / hard ≥ 0.55 | accuracy 0–25p (`pass_rate × 25`) | **En kritik** — 1.0 olmadan PASS olamaz (ön koşul) |
| **cpu_ms** | subprocess | ≤ 2000ms | > 5000ms | efficiency 0–20p | yüksek CPU → skor düşer |
| **ram_mb** | psutil | ≤ 256MB | > 512MB | efficiency 0–20p | yüksek RAM → skor düşer |
| **timed_out** | subprocess | False | True | skora girmez | Node 3 prompt'a bilgi olarak gider |
| **failed_cases** | yerel değerlendirici | [] | — | skora girmez | Node 3 prompt'a gider (max 5 case) |
| **crash_count** | yerel değerlendirici | 0 | — | skora girmez | pass_rate içinde sayılır |
| **test_diversity_score** | yerel değerlendirici | 1.0 | — | skora girmez | izleme amaçlı |

### İki Fazlı Çalışma

**Faz 1 — Doğruluk (yerel LCB uyumlu değerlendirici):**
Döngüde **yalnızca açık (public) testleri** çalıştırır; pass_rate, failed_cases, crash_count üretir.
Gizli testler döngüde hiç kullanılmaz; teslim edilen kod çevrimdışı notlanır.

**Faz 2 — Runtime (subprocess):**
Sadece public test'lerle cpu_ms ve ram_mb ölçer.

---

## Node 3 — Judgement LLM (Qwen3.5-35B)

### Puanlama Rubric'i (toplam 100p)

| Boyut | Kaynak | Formül | Ağırlık | Tür |
|-------|--------|--------|---------|-----|
| **accuracy** | node2_pass_rate | `pass_rate × 25` | **25p** | Deterministik |
| **efficiency** | cpu_ms + ram_mb | `(1 − cpu/2000 + 1 − ram/512) / 2 × 20` | **20p** | Deterministik |
| **quality** | CC + MI | `(mi/100 + 1 − cc/20) / 2 × 20` | **20p** | Deterministik |
| **security** | bandit | `15 − high×15 − medium×3` | **15p** | Deterministik |
| **algorithm_choice** | Qwen judgement | LLM değerlendirmesi (0–10) | **10p** | LLM subjektif |
| **error_handling** | Qwen judgement | LLM değerlendirmesi (0–10) | **10p** | LLM subjektif |
| **TOPLAM** | | | **100p** | %80 deterministik |

### Puan Dağılımı

```
accuracy     ████████████████████████░░  25p  deterministik — test sonucu
efficiency   ████████████████████░░░░░░  20p  deterministik — runtime
quality      ████████████████████░░░░░░  20p  deterministik — CC + MI
security     ███████████████░░░░░░░░░░░  15p  deterministik — bandit
algorithm_choice  ██████████░░░░░░░░░░░░░░░░  10p  LLM — subjektif
error_handl. ██████████░░░░░░░░░░░░░░░░  10p  LLM — subjektif
```

**80p deterministik, 20p LLM subjektif.**

### PASS Eşikleri

> **Güncel karar kuralı (2026-10-02):** Varsayılan tasarımda (`JUDGE_DECISION=verdict`) PASS'i
> yargıcın gerekçeli ACCEPT/RET kararı verir; ön koşul açık testlerin tamamının geçmesidir.
> Aşağıdaki eşikler yalnızca (i) puan kararı tasarımında (P, `rubric`) ve (ii) yargıç geçerli
> karar üretemediğinde yedek olarak kullanılır. Bkz. `docs/node3.md` §4.2.

| Difficulty | PASS Eşiği | Gerekli Minimum |
|------------|-----------|-----------------|
| **easy** | ≥ 80p | pass_rate ~%60 + tam efficiency + quality |
| **medium** | ≥ 70p | pass_rate ~%40 + iyi metrikler |
| **hard** | ≥ 60p | pass_rate ~%20 + iyi metrikler |

---

## Metriklerin Node 3 Prompt'una Etkisi

Aşağıdaki metrikler **skora girmez** ama Node 3'ün ürettiği `improved_prompt`'a bağlam olarak aktarılır:

| Metrik | Prompt'a Katkısı |
|--------|-----------------|
| Halstead Volume | Prompt'ta sayısal değer olarak gösterilir |
| timed_out | "Solution exceeded time limit" uyarısı |
| failed_cases | Input/expected/got olarak listelenir (max 5) |
| mypy hataları | `node1_issues` içinde tip hataları |
| ruff hataları | `node1_issues` içinde lint uyarıları |
| node1_issues | Tüm uyarılar maddeler halinde |

---

## Ablasyon Variant'larında Metrik Kullanımı

| Metrik | V1 (Exec+Judge) | V2 (Static+Judge) | V3 (Full) |
|--------|:---------------:|:-----------------:|:---------:|
| CC, MI, Halstead | ✗ | ✓ | ✓ |
| Bandit, mypy, ruff | ✗ | ✓ | ✓ |
| pass_rate | ✓ | ✗ | ✓ |
| cpu_ms, ram_mb | ✓ | ✗ | ✓ |
| failed_cases | ✓ | ✗ | ✓ |
| algorithm_choice, error_handling | ✓ | ✓ | ✓ |
| **PASS kriteri** | yargıç ACCEPT + açık testler | yargıç ACCEPT (açık test ön koşulu yok) | yargıç ACCEPT + açık testler |
