# Specialized Code Judge Model — Research Roadmap

> **Durum (2026-10-02): plan.** Gelecek çalışma. Mevcut yargıçlar: yerelde Qwen3.5-9B, ana deneyde Qwen2.5-72B-Instruct (OpenRouter); "QWEN 35B" karşılaştırmaları eski plana aittir. Eğitim verisi ana deneyin iterasyon JSON'larından üretilebilir.

Bu döküman, mevcut genel amaçlı QWEN judge modelinin yerine
domain-specific (Python competitive programming) küçük bir judge modeli
eğitmenin fizibilitesini ve adım adım planını sunar.

---

## Motivasyon

| | Mevcut (QWEN 35B) | Hedef (Özel Judge 7B) |
|---|---|---|
| Boyut | ~24 GB | ~5 GB |
| Donanım | A100 zorunlu | Mac Mini M4'te çalışır |
| Domain | Genel amaçlı | Python kodu + LCB rubric |
| Şeffaflık | Kara kutu | Fine-tune verisi görünür |
| Inference hızı | ~10-50 sn/iter | ~2-8 sn/iter |
| Rubric esnekliği | Prompt bağımlı | Fine-tune ile içselleştirilmiş |

**Kilit avantaj:** Pipeline zaten etiketli veri üretiyor.
Her `iteration JSON` dosyası, potansiyel bir eğitim örneğidir —
dışarıdan veri toplama gerekmez.

---

## Literatür Özeti

### Temel Referanslar

| Çalışma | Model | Yöntem | Sonuç |
|---|---|---|---|
| **Prometheus** (Kim et al., 2023) | 13B | SFT, rubric-based, GPT-4 feedback | Pearson 0.897 (≈ GPT-4) |
| **Prometheus 2** (Kim et al., 2024, EMNLP) | 7B / 8×7B | SFT + pairwise merge | Open evaluator SOTA |
| **Auto-J** (Li et al., 2023) | 13B | SFT, 58 scenario | Pairwise + pointwise |
| **JudgeLM** (Zhu et al., 2023) | 7B–33B | SFT, scalable judge | GPT-4 verisi distilasyon |
| **MT-Bench** (Zheng et al., 2023) | GPT-4 | Bias analizi | Position / verbosity bias tanımlandı |

### Bu Proje İçin Kritik Bulgular

1. **13B → 7B indirgenebilir:** Prometheus 2, 7B ile 13B'ye yakın sonuç alıyor.
   Domain daralınca (genel NLP → Python kodu) daha küçük model yeterli olabilir.

2. **Rubric + reference answer şart:** Prometheus'un başarısı rubric ve
   reference answer kombinasyonundan geliyor. LCB'nin `all_tests` ve
   problem statement'ı bunu doğrudan sağlıyor.

3. **GPT-4 feedback → SFT:** Prometheus, 100K GPT-4 çıktısıyla eğitildi.
   Bu pipeline'da cloud run çıktıları (M2/M3 + QWEN judge) aynı işlevi görür.

4. **Validation ground truth önemli:** Code domain'inde sandbox execution
   gerçek doğrulama kriteri — NLP domain'indeki insan değerlendirmesinden
   daha güvenilir ve otomatik.

---

## Roadmap

### Faz 0 — Ön Koşul: Cloud Run Verisi (1–2 hafta)

**Hedef:** Eğitim için yeterli yüksek kaliteli örnek toplamak.

```
Cloud run (V3, M1+M2+M3, 150–300 problem, easy+medium+hard)
    → outputs/results/*/iterations/**/*.json
    → Toplam örnek hedefi: 3K–8K iteration JSON
```

**Kalite filtresi (hangi örnekler kullanılır):**
- `node3.failure_reasons` içinde `json_parse_error` YOK
- `improved_prompt` boş değil (LOOP durumundaki örnekler)
- `node3.total_score` mevcut
- `training.stripped_code` boş değil (kod logu eksikse örnek atılır)

**Neden bu kadar önemli:** Eğitim verisi yok = model eğitimi mümkün değil.
Bu faz olmadan diğer fazlar başlayamaz.

---

### Faz 1 — Dataset İnşası (3–5 gün)

**Kaynak:** `outputs/results/*/iterations/**/*.json`

Her JSON dosyasının `training` bloğu eğitim örneğini doğrudan içerir:
- `training.problem` → `[PROBLEM]`
- `training.stripped_code` → `[CODE]`
- `training.all_tests` → `[REFERENCE_TESTS]`
- `node3.*` → hedef OUTPUT

**Eğitim formatı (Prometheus tarzı):**

```
INPUT:
  [PROBLEM]: <training.problem>
  [CODE]: <training.stripped_code>
  [RUBRIC]:
    - accuracy (0–30): Does code pass all test cases?
    - efficiency (0–20): Time/space complexity appropriate?
    - quality (0–20): Readable, maintainable?
    - security (0–15): No dangerous imports/patterns?
    - algorithm_choice (0–10): Right algorithm for constraints?
    - error_handling (0–5): Edge cases handled?
  [REFERENCE_TESTS]: <all_tests from LCB>

OUTPUT (JSON):
  {
    "breakdown": {accuracy, efficiency, quality, security,
                  algorithm_choice, error_handling},
    "total_score": <float 0–100>,
    "pass": <bool>,
    "failure_reasons": [...],
    "improved_prompt": "<next iteration instruction>"
  }
```

**Veri bölünmesi:**
- Train: 80% (LCB problem ID bazında split — problem sızıntısı olmamalı)
- Val:   10%
- Test:  10% (cloud run'dan ayrılmış, hiç görülmemiş problemler)

**Veri zenginleştirme:**
- Aynı problem için farklı iterasyon çiftleri → contrastive pairs
- Yüksek skor (≥80) vs düşük skor (<50) çiftleri → DPO için
- Score oscillation örnekleri → "bu kötü judge çıktısı" negatif label

---

### Faz 2 — Base Model Seçimi ve SFT (1–2 hafta)

**Birincil seçenek: Qwen2.5-Coder-7B**
- Zaten pipeline'da, Metal backend'de çalışıyor
- Coding pre-training → fine-tune transferi verimli
- GGUF formatına kolayca dönüştürülür

**Alternatifler:**
- `Qwen2.5-7B-Instruct` — instruction-tuned, rubric formatını daha iyi anlayabilir
- `Llama-3.1-8B-Instruct` — güçlü genel taban, farklı failure mode'lar
- `Mistral-7B-Instruct-v0.3` — küçük ve hızlı, minimum baseline

**Eğitim setup (Mac Mini M4, 16GB):**

```python
# LoRA konfigürasyonu (QLoRA ile 16GB'a sığar)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

# Eğitim parametreleri
training_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,   # effective batch = 16
    num_train_epochs=3,
    learning_rate=2e-4,
    bf16=True,                        # M4 destekler
    optim="adamw_torch",
)
```

**Kayıp fonksiyonu:** Cross-entropy, sadece response token'ları üzerinde
(input/rubric token'larına loss uygulanmaz — Prometheus ile aynı yaklaşım).

**Süre tahmini:** ~6–12 saat (Mac Mini M4, 3K örnek, 3 epoch)

---

### Faz 3 — Tercihen: Preference Optimization (1 hafta)

Bu faz opsiyoneldir; Faz 2 sonuçlarına göre karar verilmeli.

**DPO için pair oluşturma:**

```
winner = iteration[i] where score[i+1] > score[i]   # judge sonrası kod iyileşti
loser  = iteration[j] where score[j+1] <= score[j]  # judge sonrası değişim yok
```

**Mantık:** İyi bir judge çıktısı, bir sonraki iterasyonda kodu iyileştiren
`improved_prompt` üretir. Sandbox pass_rate artışı → iyi judge sinyali.

**Dikkat:** DPO, SFT'den sonra uygulanır. Önce SFT şart.

---

### Faz 4 — Değerlendirme (3–5 gün)

**Metrik hiyerarşisi:**

| # | Metrik | Yöntem | İyi değer |
|---|---|---|---|
| 1 | Sandbox korelasyonu | Pearson(judge_score, pass_rate) | ≥ 0.70 |
| 2 | Quality-Correctness gap | TSR_quality − TSR_correctness | < 0.10 |
| 3 | Tutarlılık | Aynı kod, 3 run → std(score) | < 5 puan |
| 4 | Downstream etkisi | Küçük judge ile V3 TSR_correctness | ≥ QWEN judge |
| 5 | Inference hızı | ms/iteration (node3_ms) | < 5 sn |

**Ablasyon karşılaştırması:**

```
V3 + QWEN 35B (mevcut)   vs   V3 + Özel Judge 7B (yeni)
    TSR_correctness              TSR_correctness
    node3_ms                     node3_ms
    quality-correctness gap      quality-correctness gap
```

---

### Faz 5 — Entegrasyon (2–3 gün)

Eğitilen model GGUF'a dönüştürülür ve mevcut stack'e eklenir:

```bash
# GGUF dönüşümü
python llama.cpp/convert_hf_to_gguf.py ./judge-7b-finetuned \
    --outfile ./models/judge-7b-code.gguf \
    --outtype q4_k_m

# .env.local'de port değişikliği
LOCAL_JUDGEMENT_MODEL=judge-7b-code
```

`models/client.py` ve `config/models.py` değişiklik gerektirmez —
OpenAI-compatible API standardı korunur.

---

## Riskler ve Azaltma Stratejileri

| Risk | Olasılık | Etki | Azaltma |
|---|---|---|---|
| Veri sızıntısı (train/test aynı problem) | Yüksek | Kritik | LCB problem ID bazlı split |
| Reward hacking (yüksek skor, düşük pass_rate) | Orta | Yüksek | Sandbox korelasyonu primary metrik |
| Improved_prompt kalitesi düşer | Orta | Orta | SFT'yi scoring ve prompting için ayrı başlıklara böl |
| Cold start (veri yok) | Yüksek (şimdi) | Yüksek | Cloud run önce şart |
| Catastrophic forgetting | Düşük | Orta | LoRA ile base model ağırlıkları korunur |

---

## Tez İçin Çerçeveleme

**Mevcut tez katkısı (V0 vs V3):**
> "LLM-as-a-Judge ile iteratif refinement, single-shot baseline'a kıyasla
> TSR_correctness'ı anlamlı ölçüde artırıyor mu?"

**Ek katkı (bu roadmap ile):**
> "Pipeline kendi execution feedback'ini kullanarak domain-specific bir judge
> modeli eğitebilir mi? Ve bu küçük judge, genel amaçlı büyük modele
> kıyasla daha iyi pipeline davranışı sağlar mı?"

Bu, Prometheus'un "domain-specific rubric + reference" yaklaşımını
competitive programming kod değerlendirmesine uyarlayan özgün bir katkıdır.

---

## Tahmini Zaman Çizelgesi

```
Cloud run (veri toplama)      ── 1–2 hafta
Dataset inşası                ──── 3–5 gün
SFT eğitimi                   ──── 1–2 hafta
Değerlendirme                 ──── 3–5 gün
DPO (opsiyonel)               ──── 1 hafta
Entegrasyon                   ──── 2–3 gün
────────────────────────────────────────────
Toplam (SFT'ye kadar)         ── ~5–6 hafta
Toplam (DPO dahil)            ── ~7–8 hafta
```

---

## Referanslar

- Prometheus: Kim et al. (2023). arXiv:2310.08491
- Prometheus 2: Kim et al. (2024, EMNLP). arXiv:2405.01535
- Auto-J: Li et al. (2023). arXiv:2310.05470
- JudgeLM: Zhu et al. (2023). arXiv:2310.17631
- MT-Bench / LLM-as-a-Judge: Zheng et al. (2023). arXiv:2306.05685
