# main-results.md — Ana deney sonuçları (sızıntısız, 2026-10-02)

> Protokol: `docs/ablation.md` §7. Araç: `benchmark/openrouter_results.py` (`docs/analysis.md` §10).
> Ham çıktılar: `outputs/openrouter/analysis_B_main/` (ana), `outputs/openrouter/analysis_A_feedback_only/`
> (ablasyon), `outputs/openrouter/judge_replay/` (yargıç tekrarı, §10), `analysis_C_gptoss_judge/` ve `analysis_D_llama8b_gptoss/` (gpt-oss yargıçlı tam
> koşular, §11). Harcama: 3,35 $ / 5 $ (deneme ve problar dahil); yargıç tekrarıyla 3,61 $; §11 koşularıyla 4,41 $.

## 1. Düzen

| | |
|---|---|
| Üretici | DeepSeek-V3 (`deepseek/deepseek-chat`, DeepInfra fp4), 2024-12-26, açık ağırlıklı (DeepSeek Model License) |
| Yargıç | Qwen2.5-72B-Instruct (Novita bf16), 2024-09-19, açık ağırlıklı (Qwen License) |
| Problemler | LCB v6, 2025-01..2025-04 penceresinden seed 42 ile 150 problem (kolay 40, orta 38, zor 72) |
| V0 | Hattın 1. iterasyon istemiyle problem başına 5 bağımsız örnek |
| V3 | En fazla 5 üretim; Y = yargıç kararı (`verdict`), P = rubrik eşiği (`rubric`) |
| Ölçüt | pass@1 = teslim edilen tek kodun tüm gizli testleri geçmesi; gizli testler döngüye girmez |

İki model de problemlerden önce yayımlandı (LCB'nin kesim sonrası değerlendirme önerisi, Jain vd.
2024). Puanlar: DeepSeek-V3 raporu (arXiv:2412.19437) LCB 2024-08..11'de DeepSeek-V3 37,6, Qwen2.5-72B
28,7 Pass@1 verir; yargıç üreticiden zayıftır.

## 2. Ana sonuç (B: geri bildirim + önceki çözüm)

| Zorluk | n | V0 pass@1 | V3 Y | V3 P |
|---|---|---|---|---|
| Toplam | 150 | %34,1 | %36,7 | %40,0 |
| Kolay | 40 | %87,0 | %92,5 | %97,5 |
| Orta | 38 | %28,9 | %31,6 | %34,2 |
| Zor | 72 | %7,5 | %8,3 | %11,1 |

- Y: V3 − V0 = +2,5 puan (%95 GA −0,7 … +6,0; Wilcoxon p = 0,175).
- P: V3 − V0 = +5,9 puan (%95 GA +1,3 … +10,7; Wilcoxon p = 0,015).
- Y vs P teslim edilen kodda: yalnız Y doğru 3, yalnız P doğru 8 (McNemar p = 0,23).

## 3. Arıtımın kendi katkısı (aynı koşuda 1. iterasyon → nihai)

V3 − V0 farkı iki şeyi karıştırır: 1. iterasyondaki tek örneğin şansı ve döngünün katkısı. Döngünün
katkısı aynı koşudaki 1. iterasyon koduyla eşleştirilerek ölçülür:

| Koşu | 1. iterasyon | Nihai | Düzelen | Bozulan | Döngüye giren | STOP_MAX |
|---|---|---|---|---|---|---|
| Y, B (ana) | %33,3 | %36,7 | 5 | 0 | 73 | 47 |
| P, B (ana) | %36,7 | %40,0 | 5 | 0 | 62 | 42 |
| Y, A (yalnız geri bildirim) | %32,7 | %36,7 | 6 | 0 | 83 | 42 |
| P, A (yalnız geri bildirim) | %31,3 | %35,3 | 6 | 0 | 67 | 38 |

- Dört koşunun hepsinde arıtım 5–6 problemi düzeltti, hiçbirini bozmadı (+3,3 … +4,0 puan; tek koşuda
  McNemar p = 0,031–0,063).
- P'nin Y'ye üstünlüğü 1. iterasyondan gelir (%36,7 vs %33,3); arıtım katkısı iki tasarımda aynıdır
  (5/0). Yani bu düzende Y ile P arasında arıtım açısından fark gözlenmedi.
- 1. iterasyon oranlarının ortalaması (%33,5) V0'a (%34,1) yakındır; ölçüm tutarlıdır.

## 4. Eşit bütçe (Olausson vd., ICLR 2024)

| | Ortalama üretim k̄ | V3 | filt@2 = filt@3 | pass@2 | pass@3 |
|---|---|---|---|---|---|
| Y | 2,56 | %36,7 | %38,7 | %36,5 | %37,9 |
| P | 2,38 | %40,0 | %38,7 | %36,5 | %37,9 |

filt@k: aynı k örnekten açık testleri geçen ilkini teslim etmek (kâhinsiz rakip). Aynı üretim
bütçesinde hat, açık testle süzmeden açıkça iyi değildir: Y altında, P biraz üstünde kalır.

## 5. Yargıç isabeti (Y)

| | B (ana) | A |
|---|---|---|
| Açık testleri geçen deneme | 110 | 122 |
| Yargıcın reddettiği | 15 | 29 |
| … gizli testte gerçekten yanlış | 11 | 21 |
| … gizli testte doğruydu (yanlış ret) | 4 | 8 |
| Kabul edilen kodun gizli testte doğru oranı | %57,9 | %59,1 |
| Kararı yargıç verdi | %99,2 | %100 |
| Geri bildirimde kod parçası | 0 / 376 | 0 / 374 |

Yargıcın retleri çoğunlukla haklıdır (B'de 11/15), ama kabul ettiği kodun %42'si gizli testlerde
kalır: açık testleri geçen yanlış kodu yakalamakta zayıftır. Haklı retler de çoğunlukla düzeltmeye
dönüşmez; üretici, yargıcın gösterdiği hatayı çözemez (özellikle zor problemlerde).

## 6. Ablasyon: önceki çözümün gösterilmesi

| Tasarım | A (yalnız geri bildirim) | B (+ önceki çözüm) | Yalnız B doğru / yalnız A doğru | McNemar p |
|---|---|---|---|---|
| Y | %36,7 | %36,7 | 2 / 2 | 1,0 |
| P | %35,3 | %40,0 | 11 / 4 | 0,12 |

Önceki çözümü göstermek ölçülebilir bir fark yaratmadı; P'deki fark yine 1. iterasyon örneğinden
kaynaklanır (A %31,3, B %36,7 başlangıç).

## 7. Maliyet

| | Çağrı | $ |
|---|---|---|
| V0 (150 × 5) | 750 | 0,51 |
| Y (B) | 760 | 0,68 |
| P (B) | 703 | 0,65 |

## 8. Değerlendirme

1. Hat, aynı modelin tek atışına göre teslim edilen kodun doğruluğunu artırır (Y +2,5, P +5,9 puan);
   artış küçüktür ve yalnızca P'de istatistiksel olarak anlamlıdır. Arıtımın kendi katkısı her koşuda
   5–6 problemdir ve hiçbir doğru çözüm bozulmamıştır.
2. Kazanç kolay ve orta problemlerde toplanır; zor problemlerde (%7,5 → %8–11) üretici kapasitesi
   sınırdır.
3. Eşit hesaplama bütçesinde açık testle süzme (filt@k) hatla aynı düzeydedir; bu, Olausson vd.
   (2024) bulgusuyla uyumludur: arıtım kazancı bütçe hesaba katıldığında mütevazıdır.
4. Bu düzende yargıç kararı (Y) rubrik eşiğinden (P) iyi değildir. Darboğaz yargıcın kabul
   isabetidir (%58): Qwen2.5-72B açık testleri geçen yanlış kodun çoğunu kabul eder.
5. Tez savı (yargıç son kararı verir) bu sonuçla çürütülmez ama desteklenmez de; daha güçlü bir
   yargıçla (ör. üreticiden güçlü) yeniden sınanması gerekir.

## 9. Sınırlılıklar

- Her tasarım bir kez koşuldu; tek koşunun 1. iterasyon örneği sonucu ±3 puan oynatabiliyor (§3).
- Üretici fp4 nicemlemeli uç noktadan sunuldu (DeepInfra); tam hassasiyetli ağırlıklarla sonuç
  farklı olabilir.
- Yargıç LCB'de üreticiden zayıf (28,7 vs 37,6); güçlü yargıç denenmedi.
- Değerlendirici yerel, LCB uyumlu (`sandbox/lcb_eval.py`); lcb_runner ile notlanmadı.
- Yargıç istemi ve onarım istemi ana koşulardan önce prob koşularıyla düzeltildi (`docs/node3.md` §4.4,
  `docs/codegen-models.md`); A koşuları önceki çözümün gösterilmesinden önce başladı.

---

## 10. Çevrimdışı yargıç tekrarı: daha güçlü yargıçlar (2026-10-02)

`benchmark/judge_replay.py` (`docs/analysis.md` §11), Y koşularında (A: 122, B: 110) açık testlerin
tamamını geçen 232 denemeyi hattın kendi yargıç istemiyle başka yargıçlara yeniden değerlendirtti ve
her kararı gizli testlerle karşılaştırdı. Kod üretilmedi. Harcama 0,27 $ (defterde
`phase=judge_replay`); tekrar sonrası toplam 3,61 $ / 5,00 $.

| Yargıç (B, ana tasarım, n = 110; 59'u gizli testlerde doğru) | Kabul | Kabul isabeti | Reddedilen yanlış kod | Reddedilen doğru kod | Karar doğruluğu |
|---|---|---|---|---|---|
| Qwen2.5-72B-Instruct (hattaki yargıç) | 95 | %57,9 | 11 / 51 (%21,6) | 4 | %60,0 |
| DeepSeek-V3 (üreticiyle aynı model) | 100 | %59,0 | 10 / 51 (%19,6) | 0 | %62,7 |
| gpt-oss-120b (akıl yürütmeli, effort medium) | 55 | %90,9 | 46 / 51 (%90,2) | 9 | %87,3 |

- **gpt-oss-120b ile hattaki yargıç, aynı denemeler:** yalnız gpt-oss'un doğru karar verdiği 37,
  yalnız Qwen'in doğru karar verdiği 7 deneme (kesin McNemar p = 5,3e-6). 232 denemenin tamamında
  79'a 15 (p = 1,2e-11); kabul isabeti %90,0'a karşı %58,5. 6/232 denemede kullanılabilir karar
  vermedi (rubrik yedeğiyle KABUL sayıldı).
- **DeepSeek-V3** Qwen2.5-72B'den iyi değil (B: 11'e 8, p = 0,65; tümü: 25'e 22, p = 0,77); neredeyse
  her şeyi kabul ediyor (207/232). Kodda daha güçlü bir model kendiliğinden daha iyi bir yargıç
  olmuyor; fark açık akıl yürütmeyle geliyor.
- **Yargı çağrısı başına süre / maliyet:** Qwen2.5-72B 16–19 s (hat kaydı); DeepSeek-V3 9,6 s,
  0,00084 $; gpt-oss-120b 26,7 s (p90 45 s), 1373 çıktı belirteci, 0,00031 $.

**Sınırlılıklar.**
- gpt-oss-120b 2025-08-05'te, yani LCB 2025-01..04 problemlerinden sonra yayımlandı. Ön eğitim bilgi
  kesimi Haziran 2024'tür (model kartı, arXiv:2508.10925), ancak son eğitim verisi açıklanmadığı için
  bu problemleri görmüş olabilir. Yargıç kod yazmaz, ama ezberlenmiş bir çözüm yanlış kodu fark
  etmeyi kolaylaştırır. Kullanılırsa bu açıkça yazılmalıdır.
- Tekrar yalnızca kararı ölçer. Daha çok yanlış kodu reddetmek ancak üretici sonra onarabilirse
  doğruluğu artırır; bunu tam hat koşusu gösterir.
- Her deneme bir kez yargılandı (sıcaklık gpt-oss için 1,0, diğerleri 0,7).

**gpt-oss-120b yargıçlı tam Y koşusu (tahmin).** B koşusunun iterasyonlarıyla (381 yargı çağrısı,
4 süreç) yaklaşık 66 dk duvar süresi (B 60 dk sürdü), 0,12 $ yargı + 0,31 $ üretim. gpt-oss yaklaşık
45 açık-testi-geçen denemeyi daha reddediyor ve her biri en çok 4 iterasyon daha getirebilir; gerçekçi
aralık 70–100 dk ve 0,45–0,65 $. Gizli test notlaması ve analizle (~10 dk) sonuç yaklaşık 1,5–2 saatte
hazır olur. Kalan bütçe (1,39 $) bir koşuya yeter.

## 11. Tam koşular: gpt-oss-120b yargıç (2026-10-03)

**Düzen.** Yalnızca Y tasarımı (`JUDGE_DECISION=verdict`), aynı 150 problem, en çok 5 üretim. Yargıç
`openai/gpt-oss-120b` (DeepInfra; T 1,0, akıl yürütme `medium`). Çalışma zamanı ölçümü düzeltilmiş
sürümle (`sandbox/runner.py`, `docs/node2.md` §3): bellek ve işlemci süresi bu koşularda doğru,
Qwen yargıçlı B koşularında değildi.

| Koşu | Üretici | TA (taban) | Run id'leri | Analiz |
|---|---|---|---|---|
| C | DeepSeek-V3 (DeepInfra fp4) | `run_20261002_091834.regraded` (B ile aynı) | `run_20261003_093242/44/46/48` | `outputs/openrouter/analysis_C_gptoss_judge/` |
| D | Llama-3.1-8B-Instruct (DeepInfra fp8; 2024-07-23) | `run_20261003_093410` (150 × 5) | `run_20261003_100935/37/39/41` | `outputs/openrouter/analysis_D_llama8b_gptoss/` |

**Sonuçlar** (gizli testlerle katı doğruluk, %):

| | TA | A_Y | Fark (%95 GA) | Wilcoxon p | D/B (1. iter → son) | Kabul isabeti | Ort. üretim | Maliyet |
|---|---|---|---|---|---|---|---|---|
| B: DeepSeek-V3 + Qwen2.5-72B | 34,1 | 36,7 | +2,5 (−0,7…6,0) | 0,175 | 5/0 | 57,9 | 2,56 | 0,68 $ |
| C: DeepSeek-V3 + gpt-oss-120b | 34,1 | 50,0 | +15,9 (10,3…21,9) | 3,5e-6 | 27/1 | 88,1 | 2,83 | 0,50 $ |
| D: Llama-3.1-8B + gpt-oss-120b | 13,2 | 30,0 | +16,8 (11,6…22,3) | 9,5e-8 | 23/1 | 95,7 | 4,04 | 0,19 $ (+0,02 $ taban) |

- C zorluğa göre: kolay 87,0 → 97,5; orta 28,9 → 57,9; zor 7,5 → 19,4. Bütçe eğrisi (k = 1..5):
  32,7 / 40,7 / 45,3 / 50,0 / 50,0; filt@3 38,7, pass@5 39,3. C, B'nin Y koşusuna göre 22'ye 2
  (McNemar p = 3,6e-5).
- D zorluğa göre: kolay 43,5 → 85,0; orta 5,8 → 26,3; zor 0,3 → 1,4. Taban pass@5 20,7, filt@4 18,7.
- Açık testleri geçen deneme / ret / retlerin gizli testte gerçekten hatalı olanı: C 157 / 73 / 55,
  D 73 / 27 / 22.

**Geri bildirim içeriği.**
- `has_code` C'de 2/404, D'de 4/601 geri bildirimi işaretledi. Elle inceleme: 5'i örnek girdi-çıktı
  ya da boş (`...your code...`) kod bloğu; 1'i (C, `abc391_e` iter 2) kod içermiyor ama formülleriyle
  adım adım bir DP tarifi veriyor (sözde koda yakın).
- Basit anahtar sözcük taraması (`dp[`, `= min(`, segment tree, Fenwick, Dijkstra, binary search,
  two pointer, prefix sum, `O(n log n)` …): somut algoritma/formül adlandıran geri bildirim oranı
  C %45 (181/398), D %39 (236/600), B'nin Y koşusu %10 (36/373). gpt-oss daha yönlendirici; kazancın
  bir kısmı yargıcın kendi çözüm bilgisinden gelebilir.

**Sınırlılıklar.** gpt-oss-120b problem penceresinden sonra yayımlandı (bkz. §10); ezber
dışlanamaz. Bildiride ek koşular ana deneyin yerine değil, tamamlayıcı bulgu olarak verildi.

**Komut** (bildiri çizelgeleri/şekil):

```bash
B=outputs/openrouter/results/baseline_samples/run_20261002_091834.regraded.jsonl
.venv/bin/python -m benchmark.openrouter_results --baseline $B \
  --y-run-ids run_20261002_100638 run_20261002_100640 run_20261002_100642 run_20261002_100644 \
  --p-run-ids run_20261002_100646 run_20261002_100648 run_20261002_100650 run_20261002_100652 \
  --out outputs/openrouter/analysis_B_main --compare outputs/openrouter/analysis_A_feedback_only/results.json \
  --latex ~/Desktop/eleco2026-bildiri \
  --extra DeepSeek-V3 gpt-oss-120b G outputs/openrouter/analysis_C_gptoss_judge/results.json \
  --extra Llama-3.1-8B gpt-oss-120b L outputs/openrouter/analysis_D_llama8b_gptoss/results.json
```

**Harcama.** Bu iki koşu + Llama tabanı ≈ 0,72 $; yanlış yargıçla başlayıp durdurulan koşu
(betik `config/openrouter.env` ile ortamı eziyordu, düzeltildi) ≈ 0,07 $. Toplam 4,41 $ / 5 $.
