# LLM-as-a-Judge for Iterative Code Refinement
## Yüksek Lisans Tezi — Çalışma Planı ve Durum

> Son güncelleme: 2026-10-02. Durum işaretleri: `[x]` tamamlandı · `[~]` sürüyor · `[ ]` bekliyor.

---

## Araştırma Sorusu

> Statik kod analizini, korumalı yürütmeyi ve son kararı veren bir BDM yargıcını birleştiren
> çok aşamalı, yinelemeli bir arıtım hattı, aynı modelin tek-atışlı üretimine kıyasla
> LiveCodeBench problemlerinde teslim edilen kodun doğruluğunu (pass@1, gizli testler)
> artırır mı; artırıyorsa bu kazanç eşit hesaplama bütçesinde de korunur mu?

Terminoloji: pass@1 = problem başına teslim edilen **tek** kodun tüm gizli testleri geçme oranı
(Chen vd. 2021; LCB'nin self-repair senaryosu da onarım sonrası Pass@1 raporlar). Döngü yalnızca
açık testleri görür. Hat fazladan hesap harcadığı için eşit bütçeli karşılaştırmalar da
raporlanır (Olausson vd., ICLR 2024).

---

## Deney Düzenleri

| Düzen | Üretici | Yargıç | Problemler | Durum |
|---|---|---|---|---|
| **Ana (sızıntısız)** | DeepSeek-V3 | Qwen2.5-72B-Instruct | LCB v6 2025-01..04, 150/182 | `[x]` sonuçlar: `docs/main-results.md` |
| Ek (güncel akıl yürütme modeli) | Qwen3.5-9B | aynı model | seed 42 örneklemin ilk 50'si | `[x]` (ELECO bildirisi) |

Gerekçe ve ezber riski: `docs/dataset.md` §10. Protokol: `docs/ablation.md` §7.

---

## Hipotezler

| | Hipotez | Ölçüm | Durum |
|---|---|---|---|
| H1 | V3 pass@1 > V0 pass@1 | V3 − V0, bootstrap GA, Wilcoxon; ilk iterasyon → nihai McNemar | Ana: Y +2,5 (p = 0,175), P +5,9 (p = 0,015); arıtım her koşuda 5–6 düzeltme, 0 bozulma (`main-results.md`). Ek düzen: Y +10,0, P +22,0 |
| H2 | Kazanç eşit bütçede de korunur | V3 vs filt@k̄ (açık testle seçim) ve pass@k̄ (üst sınır) | Ana: desteklenmedi — filt@2/3 %38,7; Y %36,7, P %40,0 |
| H3 | Yargıç kararı (Y), puan eşiğine (P) göre daha isabetli kabul verir | Kabul isabeti (gizli testlere göre), Y vs P McNemar | Ana: desteklenmedi — kabul isabeti %57,9 vs %61,9; Y vs P 3/8 (p = 0,23). Ek düzen: %92,9 vs %84,4 (anlamlı değil) |
| H4 | Kazanç zorluk düzeyine göre değişir (kolayda tavan, zorda taban etkisi) | Zorluk kırılımı | Ana: kazanç kolay/ortada, zor %7,5 → %8–11. Ek düzende zor problemde kazanç yok |
| H5 | Yargıç kod yazmadan, sözlü geri bildirimle iyileştirir | Geri bildirimde kod parçası sayımı | Ana: 0/376 (Y), 0/346 (P). Ek düzende Y 7/97, P 9/109 |

Özgün plandaki ablasyon hipotezleri (V1/V2'nin katkısı, STOP_HIGH etkisi) `[ ]` — V1/V2 koşuları
yapılmadı; bütçe kalırsa ana düzende koşulabilir (`benchmark/evaluator.py`).

---

## Faz 1 — Kurulum `[x]`

- `[x]` Mac mini M4: llama.cpp (Metal), Qwen3.5-9B UD-Q4_K_XL, `.env.local` (`docs/setup.md`)
- `[x]` OpenRouter katmanı: `models/client.py` (akış, yeniden deneme, sağlayıcı sabitleme),
  `models/budget.py` (harcama defteri, $5 sınırı), `config/openrouter.env`, `scripts/run_openrouter.sh`
- `[ ]` Vast.ai: isteğe bağlı, ana deney için gerekmiyor (`CLOUD_README.md` §B)

## Faz 2 — Uygulama `[x]`

- `[x]` Düğüm 1–3, kapılar, kayıt (`docs/node1–3.md`, `docs/graph.md`, `docs/memory-logging.md`)
- `[x]` Yargıç son kararı verir (`JUDGE_DECISION=verdict`), açık testlerin tamamı ön koşul
- `[x]` Gizli test sızıntısız değerlendirme; yerel LCB uyumlu değerlendirici sabitlendi
- `[x]` Tarih penceresi (`LCB_DATE_FROM/TO`), n örnekli taban çizgisi (`benchmark/sample_baseline.py`)
- `[x]` Yargıca "kod yazma" kuralı; sızıntı sayacı (`benchmark/openrouter_results.py`)
- `[x]` Testler: 253 geçti, 1 atlandı (`docs/testing.md`)

## Faz 3 — Ek düzen (yerel) `[x]`

- `[x]` 50 problem, iki tasarım (Y: `run_20261001_102445`; P: `run_20260930_222527` + `run_20261001_070529`)
- `[x]` Analiz: `benchmark/paper_results.py` → bildiri tabloları, makrolar, Şekil 2
- Bulgu özeti: Y'de V0 %68,0 → V3 %78,0 (b/c 5/0, p = 0,062); P'de %54,0 → %76,0 (11/0, p < 0,001);
  Y vs P teslim edilen kodda 4/3, p = 1. Çıktı bütçesi önerilenin ~%7,5'i (6144 token) — sınırlılık.

## Faz 4 — Ana deney (OpenRouter) `[x]`

```
[x] Kredi yükleme + anahtar (.env.openrouter)
[x] İlk smoke + pilot (Qwen2.5-Coder-32B + Llama-3.3-70B): pilot pass@1 %12,2, orta/zor %0;
    Cloudflare uç noktası "[]" belirteçlerini düşürüyordu → model çifti değişti, pilot karşılaştırılamaz
[x] İstem/metrik denetimi: LeetCode stdin çelişkisi, eski sistem istemi, yorum silmede çift satır,
    ruff boşluk gürültüsü, yargıç isteminde "%75 iyi" satırı düzeltildi (yerel ELECO koşularını da etkilemişti)
[x] Prob koşusu (6 problem, DeepSeek-V3 + Qwen2.5-72B): yargıç açık testleri geçen kodu yalnız metrik
    gerekçesiyle reddediyordu → istem düzeltildi (docs/node3.md §4.4); gizli test notlama LCB gibi ilk
    hatada duruyor, V0 ve V3 aynı süre sınırıyla (docs/node2.md)
[x] baseline + run A (Y, P; yalnız geri bildirim) 09:18–10:35
[x] Onarım istemine önceki çözüm eklendi (LCB self-repair biçimi) → run B (Y, P) 10:06–11:06
[x] Analiz: docs/main-results.md (harcama 3,35 $)
[x] scripts/run_openrouter.sh baseline     → V0, 150 × 5 (pass@1 %34,1)
[ ] scripts/run_openrouter.sh run          → V3, Y ve P (bütçe yetmezse önce Y)
[ ] python -m benchmark.openrouter_results → rapor
```

Tahmini maliyet: taban çizgisi ~1,6 $, tasarım başına ~1,5–2,5 $; sınır 5 $ (aşılırsa P için onay alınır).

## Faz 5 — Analiz `[~]`

- `[x]` Ek düzen analizi (`paper_results.py`)
- `[x]` Ana düzen: pass@1/pass@k/filt@k, V3 − V0, eşit bütçe, Y vs P, yargıç isabeti, sızıntı, maliyet (`docs/main-results.md`)
- `[ ]` Hata analizi: STOP_MAX problemleri, yargıcın yanlış retleri (ek düzende 3'ü de biçim temelliydi)

## Faz 6 — Yayın `[~]`

- `[x]` ELECO 2026 bildirisi (Türkçe, 5 sayfa, 32 doğrulanmış kaynak): `~/Desktop/workspace/eleco2026-bildiri/`;
  ezber riski ve LCB'nin kesim sonrası önerisi §4.2 ve §6'da
- `[ ]` Ana düzen sonuçlarının bildiriye/teze eklenip eklenmeyeceği — sonuçlar geldikten sonra karar
- `[ ]` Tez metni

---

## Riskler

| Risk | Olasılık | Önlem |
|---|---|---|
| Ana düzende taban etkisi (zor problemler %45) | Orta | İlk pilot (Coder-32B) taban etkisi gösterdi → üretici DeepSeek-V3'e geçildi; yeni pilot |
| Bütçe aşımı | Orta | Defter + çağrı öncesi kontrol; öncelik sırası V0 → Y → P |
| Yargıç çözümü sızdırır | Orta | Kod yazma yasağı + sızıntı sayımı + raporlama |
| Üretici uç noktası fp4 nicemlemeli (DeepInfra) | Kesin | Yöntem bölümünde açıkça belirtilir |
| Küçük örneklem gücü | Yüksek | 150 problem (seed 42); eşleştirilmiş testler, GA raporu |
| Ezber (güncel modeller) | Kesin (ek düzen) | Ana düzen sızıntısız; ek düzende sınırlılık olarak yazılır |

## Kritik İlkeler

1. Yargıcın kararı merkezdedir; metrikler kanıt ve halüsinasyon korumasıdır.
2. Gizli testler yalnızca çevrimdışı ölçümde kullanılır.
3. Her kaynak birincil kaynaktan doğrulanır; hayalet referans yok.
4. Olumsuz sonuçlar da raporlanır; abartılı iddia yok.
