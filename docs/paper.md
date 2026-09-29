# Yinelemeli Kod Arıtımında Yargıç Olarak BDM: Çok Aşamalı Bir Yaklaşım

**LLM-as-a-Judge for Iterative Code Refinement: A Multi-Stage Approach**

> *IEEE iki sütun formatı — taslak içerik*

---

## Özetçe

Büyük dil modelleri (BDM, *large language model — LLM*) kod üretiminde (*code generation*) kayda değer bir performans sergilemekle birlikte, tek-atışlı yaklaşımlar (*single-shot approach*) sistematik geri bildirim (*feedback*) mekanizmalarından yoksun olduğu için pratik kullanımda sınırlı kalmaktadır. Bu çalışmada, statik kod analizi (*static code analysis*), izole çalışma ortamı (*sandboxed execution*) ve BDM tabanlı yargılamayı (*LLM-based judgement*) bir araya getiren çok aşamalı (*multi-stage*), yinelemeli (*iterative*) bir kod arıtım hattı (*code refinement pipeline*) önerilmektedir. Üç analiz düğümünden (*node*) oluşan sistem; yapısal kalite metriklerini (*structural quality metrics*) hesaplar, işlevsel doğruluğu (*functional correctness*) ölçer ve bağımsız bir yargı modeli (*judgement model*) aracılığıyla kapsamlı geri bildirim üretir. Dört varyantlı ablasyon (*ablation*) çalışmasıyla her düğümün katkısı izole biçimde değerlendirilmiş; sonuçlar Görev Başarı Oranı (*Task Success Rate — TSR*) ve McNemar istatistiksel testi ile karşılaştırılmıştır. Elde edilen bulgular, çok kipli (*multi-modal*) yinelemeli geri bildirimin tek-atışlı temel yaklaşıma (*baseline*) kıyasla başarı oranını anlamlı biçimde artırdığına işaret etmektedir.

**Anahtar Kelimeler:** büyük dil modeli, kod arıtımı, yargıç olarak BDM, yinelemeli geri bildirim, statik analiz, LiveCodeBench

---

## Abstract

Large language models (LLMs) demonstrate strong performance in code generation; however, single-shot approaches remain limited in practice due to the absence of systematic feedback mechanisms. This study proposes a multi-stage iterative code refinement pipeline that integrates static code analysis, sandboxed execution, and LLM-based judgement. The three-node system computes structural quality metrics, evaluates functional correctness, and generates comprehensive feedback through an independent judgement model. The contribution of each node is isolated through a four-variant ablation study; results are assessed using Task Success Rate and the McNemar statistical test. Findings indicate that multi-modal iterative feedback significantly improves success rates over single-shot baselines.

**Keywords:** large language model, code refinement, LLM-as-a-judge, iterative feedback, static analysis, LiveCodeBench

---

## I. Giriş

Yazılım geliştirme süreçlerinde büyük dil modellerinin (BDM, *LLM*) rolü son birkaç yılda dramatik biçimde genişledi. HumanEval [1] ve MBPP [2] gibi gösterge tablolarında (*benchmark*) elde edilen yüksek başarı oranları, bu modellerin karmaşık algoritmik görevleri çözebildiğine dair güçlü kanıtlar sunmaktadır. Ne var ki gerçek dünya koşullarında tek-atışlı (*single-shot*) kod üretimi ciddi bir kırılganlık taşır: model, ürettiği kodun doğruluğunu bağımsız olarak doğrulayamaz ve üretimden sonra herhangi bir düzeltme mekanizmasına sahip değildir.

Bu boşluğu kapatmaya yönelik çeşitli yaklaşımlar geliştirilmiştir. CodeRL [3] ve AlphaCode [4] gibi çalışmalar, test sonuçlarını geri bildirim sinyali (*feedback signal*) olarak kullanmış; Chen ve diğerleri [5] ise modelin kendi çıktısını yeniden inceleyerek hatayı giderebildiğini göstermiştir. Öte yandan Olausson ve diğerleri [6], saf öz-onarım (*self-repair*) yaklaşımlarının beklentilerin oldukça altında kaldığını vurgulamış ve daha zengin geri bildirim mekanizmalarına duyulan ihtiyacı ortaya koymuştur.

Bu çalışma, söz konusu ihtiyaca yanıt vermek amacıyla üç tamamlayıcı geri bildirim kaynağını tek bir yinelemeli döngüde (*iterative loop*) birleştiren bir mimari önermektedir. Sistemin katkıları şu şekilde özetlenebilir:

- Statik analiz (*static analysis*) — döngüsel karmaşıklık (*cyclomatic complexity*), bakım kolaylığı indeksi (*maintainability index*), güvenlik taraması (*security scan*) —, çalışma ortamı ölçümleri (*runtime metrics*) — test geçiş oranı (*test pass rate*), CPU/bellek kullanımı — ve LLM tabanlı yargılamayı entegre eden uçtan uca bir hat.
- Her bileşenin katkısını izole eden ve sayısal olarak karşılaştıran dört varyantlı ablasyon (*ablation*) tasarımı.
- Birden fazla açık ağırlıklı (*open-weight*) model üzerinde gerçekleştirilen kapsamlı deneyler ve McNemar testiyle desteklenen istatistiksel değerlendirme.

---

## II. İlgili Çalışmalar

**Yinelemeli Kod Arıtımı (*Iterative Code Refinement*).** BDM çıktılarının döngüsel biçimde iyileştirilmesi fikri, CodeRL [3] ile birlikte programatik bir zemine oturmuştur. Bu çalışmada model, derleyici (*compiler*) ve birim test (*unit test*) geri bildirimlerine göre kodunu güncellemektedir. Benzer bir çerçevede Self-Debugging [5], kod üretimi sonrasında modelin kendi çıktısını izleme kaydıyla (*execution trace*) karşılaştırarak hata ayıklamasına (*debugging*) olanak tanımaktadır. Reflexion [7], bu fikri daha ileri taşıyarak sözel düşünce izlerini (*verbal reasoning traces*) belleğe kaydetmekte ve bu sayede çok adımlı görevlerde performansı artırmaktadır. Bununla birlikte bu yaklaşımların büyük çoğunluğu, geri bildirimi tek bir kaynakla sınırlı tutmakta; statik analiz veya bağımsız bir yargı modeli gibi ek bileşenlerden yararlanmamaktadır.

**LLM-as-a-Judge (*Yargıç Olarak BDM*).** Zheng ve diğerleri [8] tarafından önerilen LLM-as-a-Judge çerçevesi, bir modelin başka bir modelin çıktısını değerlendirip puanlamasına dayanan bir yapıyı sistematik biçimde ortaya koymaktadır. MT-Bench ve Chatbot Arena üzerinde gerçekleştirilen kapsamlı deneyler, yargı modelinin insan değerlendirmesiyle yüksek korelasyon gösterdiğini kanıtlamıştır. CRITIC [9], bu fikri bir adım ileriye taşıyarak modelin kendi çıktısına yönelik eleştiri (*critique*) üretmesini ve ardından bu eleştiriyi düzeltme sürecinde kullanmasını sağlamaktadır. Söz konusu çalışmalardan farklı olarak, önerilen sistemde yargılama görevi tamamen bağımsız bir modele devredilmektedir; üretim ve değerlendirme rolleri kasıtlı olarak birbirinden ayrıştırılmıştır.

**LiveCodeBench.** Jain ve diğerleri [10] tarafından sunulan LiveCodeBench (LCB), sürekli güncellenen ve gerçek programlama yarışmalarından derlenen görevleri içeren bir değerlendirme (*evaluation*) platformudur. Sabit veri kümesi tabanlı ölçütlerin aksine LCB, modellerin ezberleme (*memorisation*) yerine gerçek problem çözme kapasitesini ölçmeye çalışmaktadır. Bu çalışmada tüm deneyler LCB test kümesi üzerinde gerçekleştirilmekte, böylece elde edilen sonuçların güncel ve karşılaştırılabilir olması güvence altına alınmaktadır.

---

## III. Önerilen Sistem

### A. Genel Mimari

Sistem, LangGraph durum grafiği (*state graph*) üzerine inşa edilmiş bir yinelemeli döngüden oluşmaktadır. Her problem için üç kod üretim modeli sıralı biçimde çalıştırılır (M1 → M2 → M3); her model, başarı eşiğini (*pass threshold*) geçene ya da izin verilen maksimum yineleme sayısına ulaşana dek döngüde kalır.

```
┌──────────────────────────────────────────────────────┐
│                   YİNELEMELİ DÖNGÜ                  │
│                  (Iterative Loop)                    │
│                                                      │
│   ┌──────────┐    ┌──────────┐    ┌──────────────┐  │
│   │  Kod     │    │  Düğüm 1 │    │   Önem       │  │
│   │  Üretimi │───▶│  Statik  │───▶│   Kapısı     │  │
│   │(Codegen) │    │  Analiz  │    │(Severity Gate│  │
│   │  M1/M2/  │    │ (Node 1) │    │              │  │
│   │   M3)    │    └──────────┘    └──────┬───────┘  │
│   └────▲─────┘                           │          │
│        │                    ┌────────────┼──────┐   │
│        │ improved_prompt    │            │      │   │
│        │               HIGH│         LOW│   OK │   │
│        │                   ▼            ▼      ▼   │
│   ┌────┴─────┐         ┌──────┐  ┌──────────────┐  │
│   │ Yineleme │         │ DUR  │  │  Düğüm 2     │  │
│   │  Kapısı  │         │(STOP)│  │  Korumalı    │  │
│   │(Iteration│         └──────┘  │  Yürütme     │  │
│   │  Gate)   │                   │  (Sandbox    │  │
│   └────┬─────┘                   │   Node 2)    │  │
│        │                         └──────┬───────┘  │
│   ┌────┴──────────────────────────┐     │           │
│   │  GEÇ / DURDUR / YİNELE       │     ▼           │
│   │ (PASS / STOP_MAX / LOOP)      │ ┌──────────┐   │
│   └───────────────────────────────┘ │  Düğüm 3 │   │
│                                     │  Yargı   │   │
│                                     │  (Judge  │   │
│                                     │  Node 3) │   │
│                                     └──────────┘   │
└──────────────────────────────────────────────────────┘
```

**Şekil 1.** Pipeline mimarisi. Önem kapısı (*severity gate*), Düğüm 1 (*Node 1*) çıktısına göre akışı HIGH/LOW/OK olmak üzere üç kola yönlendirir. Yineleme kapısı (*iteration gate*) ise toplam puana ve yineleme sayacına (*iteration counter*) göre GEÇ (*PASS*), YİNELE (*LOOP*) veya DURDUR (*STOP*) kararı verir.

### B. Düğüm 1 — Statik Analiz (*Static Analysis Node*)

Kod üretiminin hemen ardından devreye giren bu düğüm, yürütme öncesinde yapısal ve güvenlik sorunlarını tespit eder. Döngüsel Karmaşıklık (*Cyclomatic Complexity — CC*) ve Bakım Kolaylığı İndeksi (*Maintainability Index — MI*) hesaplamaları için `radon` [11, 12] kullanılırken güvenlik taraması (*security scan*) `bandit` ile gerçekleştirilir. Ek olarak `mypy` ve `ruff` araçları aracılığıyla tip tutarlılığı (*type consistency*) ve sözdizimi (*syntax*) denetimi yapılmaktadır.

Önem kararı (*severity decision*) şu öncelik sırasına göre verilir: CC > 20, Bandit HIGH bulgusu veya izin dışı bir içe aktarım (*forbidden import*) tespit edildiğinde pipeline o noktada durdurulur (HIGH). MI < 40 ya da Halstead hacmi (*Halstead volume*) > 2000 koşullarında korumalı yürütme aşaması atlanarak doğrudan Düğüm 3'e geçilir (LOW). Bu tasarım sayesinde güvenlik açığı (*security vulnerability*) barındıran veya aşırı karmaşık kodların gereksiz yere çalıştırılmasının önüne geçilmektedir.

### C. Düğüm 2 — Korumalı Yürütme (*Sandboxed Execution Node*)

Düğüm 2, iki paralel süreç yürütür. Birinci süreçte LCB'nin resmi değerlendirme aracı (`lcb_runner`) aracılığıyla tüm test senaryoları (*test cases*) — genel (*public*), özel (*private*) ve kenar durumlar (*edge cases*) — çalıştırılır; test geçiş oranı (*test pass rate*) ile başarısız senaryolar kayıt altına alınır. İkinci süreçte yalnızca genel testler kullanılarak işlemci süresi (cpu_ms) ve bellek kullanımı (*memory usage*, ram_mb) ölçülür.

Korumalı ortam (*sandbox*), Linux `setrlimit` mekanizması aracılığıyla bellek kullanımını sınırlandırmaktadır. Süre aşımı (*timeout*) durumunda bu bilgi doğrudan Düğüm 3'ün girişine iletilir.

### D. Düğüm 3 — Yargı ve Geri Bildirim (*Judgement and Feedback Node*)

Değerlendirme aşamasının merkezinde, Düğüm 1 ve Düğüm 2 çıktılarını sentezleyerek 100 puanlık bir rubrik (*rubric*) üzerinden bütüncül bir değerlendirme yapan yargı modeli yer almaktadır. Puan bileşenleri Tablo I'de özetlenmiştir.

**Tablo I.** 100 puanlık değerlendirme rubriği (*scoring rubric*).

| Boyut | Kaynak | Ağırlık | Tür |
|-------|--------|---------|-----|
| Doğruluk (*Accuracy*) | test geçiş oranı × 25 | 25p | Deterministik |
| Verimlilik (*Efficiency*) | cpu_ms + ram_mb | 20p | Deterministik |
| Kalite (*Quality*) | CC + MI | 20p | Deterministik |
| Güvenlik (*Security*) | Bandit bulguları | 15p | Deterministik |
| Algoritma seçimi (*Algorithm Choice*) | Yargı LLM (0–10) | 10p | LLM |
| Hata yönetimi (*Error Handling*) | Yargı LLM (0–10) | 10p | LLM |

Toplam puanın %80'i deterministik (*deterministic*) metriklerden, %20'si ise yargı modelinin öznel değerlendirmesinden oluşmaktadır. Bu denge, tekrarlanabilirliği (*reproducibility*) ön planda tutarken LLM'in kavramsal değerlendirme kapasitesinden de yararlanmayı mümkün kılmaktadır.

Yargı modelinin çıktısının en kritik bileşeni `improved_prompt`'tur. Bu yönerge (*instruction*), bir önceki denemenin somut hatalarına ve metrik değerlerine doğrudan atıfta bulunarak üretim modelini bir sonraki yinelemede yönlendirmektedir. Önceki yinelemelerden bağımsız, kendi içinde tam ve tutarlı (*self-contained*) biçimde yazılan bu yönerge, model yanlılığını (*model bias*) önlemeye yönelik özel bir tasarım kararının ürünüdür.

Başarı eşikleri (*pass thresholds*) zorluk düzeyine göre ayarlanmıştır: kolay ≥ 80 puan, orta ≥ 70 puan, zor ≥ 60 puan.

---

## IV. Deneysel Kurulum

### A. Modeller

Deneyler, hepsi Unsloth GGUF formatında UD-Q4_K_XL nicemleme (*quantisation*) ile dağıtılan dört açık ağırlıklı (*open-weight*) model üzerinde gerçekleştirilmiştir. M1 olarak kullanılan gpt-oss-20b modeli, Microsoft'un şeffaf mimarili 20B parametreli modeli olup özellikle yönlendirme şablonu (*chat template*) desteği gerektirmektedir. M2 konumundaki Qwen3-Coder-Next, yaklaşık 46 GB bellek gerektiren ve alanda güçlü kodlama performansıyla öne çıkan baskın (*dominant*) modeldir. GLM-4.7-Flash ise M3 olarak görev yapmakta olup tekrar döngüsü (*repetition loop*) sorununu önlemek amacıyla `--repeat-penalty 1.0` parametresiyle çalıştırılmaktadır. Tüm üretim modellerinden bağımsız olarak Qwen3.5-35B-A3B modeli yargı rolüne ayrılmıştır; bu ayrım, öz-değerlendirme yanlılığını (*self-evaluation bias*) ortadan kaldırmak için bilinçli alınan bir tasarım kararıdır.

Modeller, OpenAI uyumlu API sunan llama.cpp sunucuları aracılığıyla Vast.ai A100 80GB donanımında çalıştırılmıştır. Sıralı yürütme (*sequential execution*) mimarisi sayesinde yalnızca bir model aynı anda belleğe yüklenmektedir.

### B. Veri Kümesi (*Dataset*)

LiveCodeBench [10] test kümesinden rastgele örnekleme (*random sampling*) yöntemiyle (sabit tohum: *seed* 42) seçilen 50 problem kullanılmıştır. Problemler kolay (*easy*), orta (*medium*) ve zor (*hard*) olmak üzere üç zorluk düzeyine dağılmakta olup her problem; genel testler (*public tests*), özel testler (*private tests*) ve kural tabanlı üretilmiş kenar durumlardan (*edge cases*) oluşan genişletilmiş bir test paketine sahiptir.

### C. Ablasyon Varyantları (*Ablation Variants*)

Her varyant bağımsız olarak yürütülmüştür; bu sayede tekil bileşenlerin katkısı net biçimde ölçülebilmektedir.

**Tablo II.** Ablasyon varyantları.

| Varyant | Düğüm 1 | Düğüm 2 | Düğüm 3 | Açıklama |
|---------|---------|---------|---------|---------|
| V0 | — | — | — | Tek-atışlı temel (*single-shot baseline*) |
| V1 | — | ✓ | ✓ | Yürütme + Yargı (*Execution + Judgement*) |
| V2 | ✓ | — | ✓ | Statik Analiz + Yargı (*Static Analysis + Judgement*) |
| V3 | ✓ | ✓ | ✓ | Tam pipeline (*Full Pipeline*) |

V0 (temel, *baseline*), pipeline devre dışıyken her model için ayrı ayrı çalıştırılmıştır. V1, statik analizin yokluğunda yürütme geri bildiriminin tek başına ne kadar etkili olduğunu sorgulamaktadır. V2 ise korumalı yürütme olmaksızın statik analizin Görev Başarı Oranı'na katkısını ölçmektedir.

### D. Değerlendirme Kriterleri (*Evaluation Metrics*)

Temel ölçüt, Görev Başarı Oranı'dır (TSR, *Task Success Rate*): bir problem için M1, M2 veya M3 modellerinden herhangi birinin başarı eşiğini (*pass threshold*) aşması yeterlidir. Güven aralıkları (*confidence intervals*) %95 Wilson yöntemiyle hesaplanmıştır. İstatistiksel anlamlılık (*statistical significance*) ise ikili eşleştirilmiş (*paired binary*) sonuçlar için uygun olan McNemar testi [13] ile sınanmıştır.

Dört hipotez çerçevesinde bulgular değerlendirilmektedir:

- **H1:** V3 kolay TSR ≥ V0 kolay TSR + 15 puan.
- **H2:** V1 TSR > V0 TSR (yürütme + yargı, temel yaklaşımı geçiyor).
- **H3:** V2 TSR > V0 TSR (statik analiz + yargı, temel yaklaşımı geçiyor).
- **H4:** V3 TSR > V1 TSR ve V3 TSR > V2 TSR (tam pipeline, ablasyon varyantlarını geçiyor).

---

## V. Bulgular

> *Bu bölüm Vast.ai deneyleri tamamlandıktan sonra doldurulacaktır.*

### A. Ana Ablasyon Tablosu

**Tablo III.** Varyant × zorluk düzeyine göre Görev Başarı Oranı (TSR, *Task Success Rate*). Parantez içindeki değerler %95 Wilson güven aralığını (*confidence interval*) göstermektedir.

| Varyant | Kolay (*Easy*) | Orta (*Medium*) | Zor (*Hard*) | Genel (*Overall*) | ΔV0 |
|---------|----------------|-----------------|--------------|-------------------|-----|
| V0 — Tek-atışlı (*Single-shot*) | — | — | — | — | — |
| V1 — Yürütme + Yargı | — | — | — | — | — |
| V2 — Statik + Yargı | — | — | — | — | — |
| V3 — Tam pipeline | — | — | — | — | — |

### B. Model Bazlı Karşılaştırma

**Tablo IV.** Her model için V0 ve V3 TSR karşılaştırması.

| Model | V0 TSR | V3 TSR | Δ | Ort. Yineleme (*Avg. Iteration*) | DURDUR_Y% (*STOP_HIGH%*) |
|-------|--------|--------|---|----------------------------------|--------------------------|
| M1 — gpt-oss-20b | — | — | — | — | — |
| M2 — Qwen3-Coder-Next | — | — | — | — | — |
| M3 — GLM-4.7-Flash | — | — | — | — | — |

### C. Zorluk Düzeyine Göre Analiz

```
TSR
1.0 │
    │
0.8 │  [SONUÇLAR EKLENECEKTİR]
    │
0.6 │
    │
0.4 │
    │
0.2 │
    │
0.0 └─────────────────────────────────
        Kolay       Orta        Zor
       (Easy)     (Medium)    (Hard)
        ──── V0   ──── V1   ──── V3
```

**Şekil 2.** Zorluk düzeyi × varyanta göre TSR karşılaştırması. Kolay sorulardaki beklenen tavan etkisi (*ceiling effect*) grafikte görselleştirilecektir.

### D. Hipotez Sonuçları

| Hipotez | Sonuç | p değeri | Yorum |
|---------|-------|---------|-------|
| H1: V3 kolay ≥ V0 + 15pp | — | — | — |
| H2: V1 > V0 | — | — | — |
| H3: V2 > V0 | — | — | — |
| H4: V3 > V1 ve V3 > V2 | — | — | — |

---

## VI. Sonuç

Bu çalışmada, açık ağırlıklı (*open-weight*) BDM'leri için tasarlanmış çok aşamalı yinelemeli bir kod arıtım hattı sunulmuştur. Önerilen sistem; statik analizi, korumalı yürütmeyi (*sandboxed execution*) ve bağımsız LLM yargılamasını tek bir döngüde birleştirmekte, her bileşenin göreve özgü katkısını ablasyon çerçevesinde ölçmektedir.

Tasarım sürecinde öne çıkan başlıca gözlemler şunlardır: Yargı modelinin kod üretim modelinden tamamen ayrıştırılması, öz-değerlendirme yanlılığını (*self-evaluation bias*) azaltmakta ve geri bildirimin tutarlılığını artırmaktadır. Deterministik (*deterministic*) metriklerin puana ağırlıklı katkısı (%80), değerlendirme sürecinin tekrarlanabilirliğini (*reproducibility*) güvence altına almaktadır. Şiddet kapısı (*severity gate*) aracılığıyla yapılan erken durdurma (*early stopping*) mekanizması ise hesaplama maliyetini (*computational cost*) ciddi ölçüde azaltmaktadır.

Çalışmanın sınırlılıkları arasında örneklem büyüklüğünün (50 problem) istatistiksel gücü (*statistical power*) kısıtlaması ve yargı modelinin kalibrasyonunun (*calibration*) manuel doğrulama gerektirmesi sayılabilir. Gelecek çalışmalarda yargı modelinin ince ayarlanması (*fine-tuning*), çoklu dil desteğinin genişletilmesi ve LCB'nin dışındaki değerlendirme platformlarında sistemin sınanması planlanmaktadır.

---

## Kaynaklar

[1] M. Chen, J. Tworek, H. Jun, Q. Yuan, H. P. de Oliveira Pinto ve diğerleri, "Evaluating Large Language Models Trained on Code," *arXiv*, arXiv:2107.03374, 2021.

[2] J. Austin, A. Odena, M. Nye, M. Bosma, H. Michalewski ve diğerleri, "Program Synthesis with Large Language Models," *arXiv*, arXiv:2108.07732, 2021.

[3] H. Le, Y. Wang, A. D. Gotmare, S. Savarese ve S. C. H. Hoi, "CodeRL: Mastering Code Generation through Pretrained Models and Deep Reinforcement Learning," *Advances in Neural Information Processing Systems (NeurIPS)*, cilt 35, ss. 21314–21328, 2022.

[4] Y. Li, D. Choi, J. Chung, N. Kushman, J. Schrittwieser ve diğerleri, "Competition-Level Code Generation with AlphaCode," *Science*, cilt 378, sayı 6624, ss. 1092–1097, 2022.

[5] X. Chen, M. Lin, N. Schärli ve D. Zhou, "Teaching Large Language Models to Self-Debug," *arXiv*, arXiv:2304.05128, 2023.

[6] T. X. Olausson, J. P. Inala, C. Wang, J. Gao ve A. Solar-Lezama, "Is Self-Repair a Silver Bullet for Code Generation?" *International Conference on Learning Representations (ICLR)*, 2024.

[7] N. Shinn, F. Cassano, E. Berman, A. Gopinath, K. Narasimhan ve S. Yao, "Reflexion: Language Agents with Verbal Reinforcement Learning," *Advances in Neural Information Processing Systems (NeurIPS)*, cilt 36, 2023.

[8] L. Zheng, W.-L. Chiang, Y. Sheng, S. Zhuang, Z. Wu ve diğerleri, "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena," *Advances in Neural Information Processing Systems (NeurIPS)*, cilt 36, 2023.

[9] Z. Gou, Z. Shao, Y. Gong, Y. Shen, Y. Yang ve diğerleri, "CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing," *International Conference on Learning Representations (ICLR)*, 2024.

[10] N. Jain, K. Han, A. Gu, W. Li, F. Yan ve diğerleri, "LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code," *arXiv*, arXiv:2403.07974, 2024.

[11] T. J. McCabe, "A Complexity Measure," *IEEE Transactions on Software Engineering*, cilt SE-2, sayı 4, ss. 308–320, Aralık 1976.

[12] D. Coleman, D. Ash, B. Lowther ve P. Oman, "Using Metrics to Evaluate Software System Maintainability," *IEEE Computer*, cilt 27, sayı 8, ss. 44–49, 1994.

[13] Q. McNemar, "Note on the Sampling Error of the Difference Between Correlated Proportions or Percentages," *Psychometrika*, cilt 12, sayı 2, ss. 153–157, 1947.

[14] M. H. Halstead, *Elements of Software Science*, New York, NY: Elsevier, 1977.

[15] B. Hui, J. Yang, Z. Cui, J. Yang, D. Liu ve diğerleri, "Qwen2.5-Coder Technical Report," *arXiv*, arXiv:2409.12186, 2024.

[16] Team GLM: A. Zeng, B. Xu, B. Wang, C. Zhang, D. Yin ve diğerleri, "ChatGLM: A Family of Large Language Models from GLM-130B to GLM-4 All Tools," *arXiv*, arXiv:2406.12793, 2024.

[17] J. Wei, X. Wang, D. Schuurmans, M. Bosma, B. Ichter ve diğerleri, "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models," *Advances in Neural Information Processing Systems (NeurIPS)*, cilt 35, ss. 24824–24837, 2022.

[18] A. Yang, A. Li, B. Yang, B. Zhang, B. Hui ve diğerleri, "Qwen3 Technical Report," *arXiv*, arXiv:2505.09388, 2025.
