# Sensitivity Analysis Plan — `benchmark/sensitivity.py`

> **Durum (2026-10-02): plan.** Henüz uygulanmadı. Dosya adları artık `<model>_iter<n>_<run_id>.json`, ana deney çıktıları `outputs/openrouter/results/` altında; yineleme sınırı 5 (eğilim uzatması kapalı).

Cloud sonuçları geldikten sonra uygulanacak analiz modülü.
Yeni çalıştırma gerektirmez — mevcut iteration JSON'lardan türetilir.

---

## Veri Kaynağı

Her iterasyon için `outputs/results/YYYY-MM-DD/iterations/<q_id>/<MODEL>_iter<N>.json`
içinde aşağıdaki alanlar mevcuttur:

```
node1: cc, mi, halstead, severity, bandit_high, bandit_medium_count
node2: pass_rate, cpu_ms, ram_mb, timed_out
node3: total_score, pass, breakdown {
           accuracy, efficiency, quality, security,
           algorithm_choice, error_handling
       }
status: PASS / LOOP / STOP_MAX / STOP_HIGH
difficulty: easy / medium / hard
```

---

## Analiz 1 — Metrik-TSR Spearman Korelasyonu

**Soru:** İlk iterasyondaki ham metrik değerleri, problemin sonunda PASS alıp almayacağını öngörüyor mu?

```python
# Her (q_id, model) çifti için:
#   - iter=1 metrik değeri (iyileştirme öncesi başlangıç durumu)
#   - nihai sonuç: PASS=1, STOP_MAX/STOP_HIGH=0

from scipy.stats import spearmanr

correlations = {
    "cc":        spearmanr(first_iter_cc,       final_pass_binary),
    "mi":        spearmanr(first_iter_mi,        final_pass_binary),
    "halstead":  spearmanr(first_iter_halstead,  final_pass_binary),
    "pass_rate": spearmanr(first_iter_pass_rate, final_pass_binary),
    "cpu_ms":    spearmanr(first_iter_cpu_ms,    final_pass_binary),
}
```

**Beklenti:** `pass_rate` en güçlü pozitif korelasyon. `cc` negatif korelasyon (yüksek CC → düşük PASS olasılığı).

**Paper katkısı:** "CC, TSR'ı ρ=−0.42 (p<0.01) ile tahmin ediyor — eşik değeri literatürle uyumlu" → Bölüm III metodoloji gerekçesi.

---

## Analiz 2 — Rubric Boyut Önem Analizi

**Soru:** 6 rubric boyutundan hangisi PASS/FAIL'i en iyi ayırt ediyor? Mevcut ağırlıklar (25/20/20/15/10/10) savunulabilir mi?

```python
from scipy.stats import spearmanr
import numpy as np

for dim in ["accuracy", "efficiency", "quality", "security",
            "algorithm_choice", "error_handling"]:

    dim_scores   = [r["breakdown"][dim] for r in all_records]
    pass_binary  = [1 if r["passed"] else 0 for r in all_records]

    # Spearman ρ: boyut skoru vs PASS sonucu
    rho, p = spearmanr(dim_scores, pass_binary)

    # Cohen's d: PASS ve FAIL grupları arasındaki etki büyüklüğü
    pass_vals = [r["breakdown"][dim] for r in all_records if r["passed"]]
    fail_vals = [r["breakdown"][dim] for r in all_records if not r["passed"]]
    pooled_std = np.sqrt((np.std(pass_vals)**2 + np.std(fail_vals)**2) / 2)
    cohens_d   = (np.mean(pass_vals) - np.mean(fail_vals)) / pooled_std
```

**Mevcut ağırlıklar:**

| Boyut           | Puan | Oran  |
|-----------------|------|-------|
| accuracy        | 0–25 | %25   |
| efficiency      | 0–20 | %20   |
| quality         | 0–20 | %20   |
| security        | 0–15 | %15   |
| algorithm_choice| 0–10 | %10   |
| error_handling  | 0–10 | %10   |

**Beklenti:** `accuracy` en güçlü predictor; `efficiency` en zayıf (competitive programming'de süre nadiren belirleyici).

**Paper katkısı:** Ağırlık seçiminin veriyle desteklenmesi → Bölüm IV rubric tasarımı savunması.

---

## Analiz 3 — Node 3 Eşik Duyarlılığı

**Soru:** PASS eşiğini değiştirseydik TSR nasıl değişirdi? Mevcut 80/70/60 seçimi optimal bölgede mi?

```python
threshold_grid = {
    "easy":   range(70, 91, 5),   # 70, 75, 80, 85, 90
    "medium": range(60, 81, 5),   # 60, 65, 70, 75, 80
    "hard":   range(50, 71, 5),   # 50, 55, 60, 65, 70
}

# Mevcut skorları yeniden sınıflandır — yeni çalıştırma gerekmez
for easy_t in threshold_grid["easy"]:
    for medium_t in threshold_grid["medium"]:
        for hard_t in threshold_grid["hard"]:
            counterfactual_tsr = recount_pass(
                scores, easy=easy_t, medium=medium_t, hard=hard_t
            )
```

**Çıktı:** Easy eşiği × TSR tablosu. Mevcut 80/70/60'ın TSR eğrisinin "dirsek noktasında" olup olmadığı görselleştirilir.

**Paper katkısı:** Bölüm V tablo veya dipnot — "eşik seçimi makul bir bölgede."

---

## Analiz 4 — Rubric Ağırlık Karşı-Olgusal Analizi

**Soru:** Ağırlıkları değiştirseydik, PASS/FAIL kararları değişir miydi? Sonuçlar robustness gösteriyor mu?

```python
weight_variants = {
    "current": {
        "accuracy": 25, "efficiency": 20, "quality": 20,
        "security": 15, "algorithm_choice": 10, "error_handling": 10
    },
    "accuracy_heavy": {
        "accuracy": 35, "efficiency": 15, "quality": 15,
        "security": 15, "algorithm_choice": 10, "error_handling": 10
    },
    "quality_heavy": {
        "accuracy": 20, "efficiency": 15, "quality": 30,
        "security": 15, "algorithm_choice": 10, "error_handling": 10
    },
    "security_heavy": {
        "accuracy": 20, "efficiency": 15, "quality": 15,
        "security": 30, "algorithm_choice": 10, "error_handling": 10
    },
}

# Her variant için toplam skoru yeniden normalize ederek hesapla
# Kaç problem PASS→FAIL veya FAIL→PASS değişti? (flip count)
```

**Çıktı:** Her ağırlık seti için TSR ve "flip count" (kaç karar değişti).

**Paper katkısı:** "Ağırlık setinden bağımsız olarak V3, V0'ı geçiyor" → robustness kanıtı.

---

## Analiz 5 — İterasyon Verimliliği

**Soru:** Kaçıncı iterasyonda en fazla iyileşme oluyor? `max_iterations=10` seçimi yerinde mi?

```python
from statistics import mean

for difficulty in ["easy", "medium", "hard"]:
    score_by_iter = {}
    for i in range(1, max_iter + 1):
        scores_at_i = [
            r["score"] for r in records
            if r["iter"] == i and r["difficulty"] == difficulty
        ]
        if scores_at_i:
            score_by_iter[i] = mean(scores_at_i)

    # Marjinal kazanım: iter i → i+1 arası ortalama skor farkı
    marginal_gain = {
        i: score_by_iter[i + 1] - score_by_iter[i]
        for i in range(1, max_iter)
        if i in score_by_iter and (i + 1) in score_by_iter
    }

    # Medyan PASS iterasyonu (PASS durumuna ilk ulaşılan iter numarası)
    first_pass_iters = [r["iter"] for r in records
                        if r["status"] == "PASS" and r["difficulty"] == difficulty]
```

**Beklenti:** İlk 3 iterasyonda kazanım yüksek, sonra azalan getiri (diminishing returns). `max_iterations=10` kararını retroaktif olarak doğrular ya da revize önerisi üretir.

**Paper katkısı:** Bölüm V Figure (skor ilerleyiş eğrisi) — `benchmark/plot.py`'daki `fig_score_progression()` ile birleşir.

---

## Analiz 6 — Severity Gate Kalibrasyonu

**Soru:** STOP_HIGH verilen problemler gerçekten kötü kodlar mıydı? CC>20 eşiği doğru calibrate mı?

```python
stop_high_records = [r for r in all_records if r["status"] == "STOP_HIGH"]

# STOP_HIGH'daki CC dağılımı — eşiğin ne kadar "sıkı" olduğunu gösterir
cc_values = [r["node1"]["cc"] for r in stop_high_records]

# False positive: STOP_HIGH dedik ama pass_rate yüksek olabilirdi?
# (pass_rate, severity=HIGH olduğu için Node 2 çalışmadı — dolaylı tahmin)
# Proxy: bandit_high=False ama CC>20 olanlar → sadece CC tetikledi
cc_only_stops = [
    r for r in stop_high_records
    if not r["node1"]["bandit_high"] and r["node1"]["cc"] > 20
]

# Bu problemlerin zorluk dağılımı — hard problemlerde CC>20 daha kabul edilebilir mi?
```

**Paper katkısı:** "Severity gate, toplam çalışmaların X%'ini erken durdurdu. Bu stopların Y%'inde tek tetikleyici CC>20 idi" → Bölüm V ablation tartışması.

---

## CLI Tasarımı

```bash
python benchmark/sensitivity.py \
    --results-dir outputs/results/2026-XX-XX \
    [--analysis all | metric_corr | dim_importance | threshold | weight | iter_eff | severity_gate] \
    [--output-md docs/sensitivity_report.md]
```

Varsayılan: `--analysis all` tüm analizleri sırayla çalıştırır.

---

## Paper ile Eşleştirme

| Analiz                  | Paper Bölümü                                      |
|-------------------------|---------------------------------------------------|
| Metrik-TSR korelasyon   | Bölüm III — metodoloji gerekçesi                 |
| Boyut önem analizi      | Bölüm IV — rubric tasarımı savunması             |
| Eşik duyarlılığı        | Bölüm V — tablo veya dipnot                      |
| Ağırlık robustness      | Bölüm V — "sonuçlar ağırlık seçiminden bağımsız" |
| İterasyon verimliliği   | Bölüm V — skor ilerleyiş figürü                  |
| Severity gate kalibrasyon| Bölüm V — ablation tartışması                   |

---

## Uygulama Notu

- Bu modül **cloud çalışması tamamlandıktan sonra** yazılacak.
- Ön koşul: `outputs/results/YYYY-MM-DD/iterations/` altında yeterli sayıda JSON bulunması
  (50 problem × 3 model × ortalama 5 iter ≈ 750 kayıt bekleniyor).
- `benchmark/analysis.py` ile paralel, bağımsız modül — mevcut kodu değiştirmez.
- `benchmark/plot.py` ile entegrasyon: Analiz 5 çıktısı doğrudan `fig_score_progression()`'a beslenir.
