# Koşu Rehberi

İki yol var:

- **§A — OpenRouter (ana deney).** GPU kiralanmaz; sızıntısız ana deney (DeepSeek-V3 üretici,
  Qwen2.5-72B-Instruct yargıç, LCB 2025-01 – 2025-04) Mac'ten API ile koşulur.
- **§B — Vast.ai (isteğe bağlı).** Mac'teki Qwen3.5-9B deneyinin **aynısını**, model kartının
  önerdiğine yakın bir token bütçesiyle kiralık GPU'da tekrarlamak için. Model, istemler,
  örnekleme ayarları, problem sırası ve değerlendirici yereldekiyle aynıdır; değişen yalnızca
  çıktı bütçesi ve hızdır.

Komutlar kopyala-yapıştır için hazırdır; `<...>` ile gösterilen yerleri kendi bilgilerinizle
değiştirin.

---

## A. OpenRouter ile koşu (GPU kiralamadan, ana deney)

Ana deney sızıntısız tasarımdır (gerekçe: `docs/dataset.md` §10, protokol: `docs/ablation.md` §7):
üretici `deepseek/deepseek-chat` = DeepSeek-V3 (DeepInfra fp4, 2024-12-26), yargıç
`qwen/qwen-2.5-72b-instruct` (Novita bf16, 2024-09-19); ikisi de 2025'ten önce yayımlandı. Problemler
yalnızca LCB'nin 2025-01 – 2025-04 penceresinden: 182 problemden seed 42 ile seçilen 150 problem. Koşu Mac'te çalışır; sandbox ve
değerlendirme yerelde kalır. Deney ayarları `config/openrouter.env` dosyasındadır; ortamda önceden tanımlanmış bir değişken dosyadakinin önüne geçer (ör. `OPENROUTER_JUDGE_MODEL=openai/gpt-oss-120b OPENROUTER_JUDGE_PROVIDER=deepinfra/bf16 scripts/run_openrouter.sh run` yargıcı yalnızca o koşu için değiştirir).

1. **Kredi ve anahtar.** https://openrouter.ai → Credits ($5) → Keys → yeni anahtar (anahtara
   $5 kredi sınırı koymak önerilir). Anahtarı **sohbete yapıştırmayın**; yalnızca şu dosyaya:
   ```bash
   cd ~/Desktop/llm-code-refinement
   open -e .env.openrouter        # OPENROUTER_API_KEY=<anahtar>
   ```
   Bu dosyadan yalnızca anahtar satırı okunur; dosya git tarafından yok sayılır.
2. **Duman testi** (hat, 2 problem, birkaç sent): `scripts/run_openrouter.sh smoke`
3. **Pilot taban çizgisi** (20 problem × 5 örnek; taban etkisi denetimi):
   `scripts/run_openrouter.sh pilot 20`
4. **Tam taban çizgisi** (150 × 5): `scripts/run_openrouter.sh baseline`
5. **Hat** (150 problem × Y ve P tasarımı, tasarım başına 4 süreç):
   ```bash
   scripts/run_openrouter.sh run
   scripts/run_openrouter.sh status      # ilerleme + harcama
   ```
   **İsteğe bağlı Langfuse izleme:** `.env.openrouter` dosyasına `LANGFUSE_PUBLIC_KEY`,
   `LANGFUSE_SECRET_KEY` (ve kendi sunucunuz için `LANGFUSE_BASE_URL`) satırlarını ekleyin;
   `scripts/run_openrouter.sh langfuse-check` anahtarları sınar. Analizde `--langfuse` gizli test
   sonuçlarını izlere skor olarak ekler. Anahtar yoksa izleme kapalıdır.
   Önceki koşular sürerken yeni bir koşu başlatılacaksa günlük adları çakışmasın diye
   `LOG_TAG=_x scripts/run_openrouter.sh run` kullanılır. `FEEDBACK_INCLUDE_CODE=false`
   onarım isteminden önceki çözümü çıkarır (yalnız geri bildirim; ablasyon için).
   **Yargıç tekrarı:** `scripts/run_openrouter.sh judge-replay MODEL SAĞLAYICI AD --run-ids …`
   kod üretmeden, açık testlerin tamamını geçen kayıtlı denemeleri başka bir yargıca yeniden
   değerlendirtir ve kararları gizli testlerle karşılaştırır (`docs/analysis.md` §11).
6. **Analiz.** Taban çizgisi dosyası ve koşu kimlikleri günlüklerde yazar
   (`outputs/openrouter/logs/`, satır `run_id:`):
   ```bash
   python -m benchmark.openrouter_results \
       --baseline outputs/openrouter/results/baseline_samples/<run_id>.jsonl \
       --y-run-ids <verdict run_id'leri> --p-run-ids <rubric run_id'leri>
   ```
   Rapor: `outputs/openrouter/report/report.md` ve `results.json`.
7. **Harcama sınırı.** Her çağrı `outputs/api_usage.jsonl` defterine yazılır. Toplam harcama +
   bir sonraki çağrının en kötü maliyeti `API_BUDGET_USD`'yi ($5) aşacaksa çağrı gönderilmez ve
   süreç durur. Yarıda kalan hat koşusu `SKIP_QIDS`, taban çizgisi `RESUME=<jsonl>` ile sürdürülür.

---

## B. Vast.ai ile koşu (isteğe bağlı)

Aşağıdaki 0–12. adımlar yalnızca bu yol içindir. Sonuçlar bildirideki yerel deneyle aynı
ayarlarla (5 + en fazla 3 uzatma) karşılaştırılabilir olsun diye ortam dosyasında eğilim
uzatması açıktır; yeni protokol için `TREND_EXTENSION_ENABLED=false` yapın.

## 0. Başlamadan önce (Mac'te, bir kez)

1. **Kodu GitHub'a gönderin.** Bulut makinesi kodu GitHub'dan çeker; yereldeki son değişiklikler
   henüz orada değil.
   ```bash
   cd ~/Desktop/llm-code-refinement
   git status          # "ahead of origin" yazıyorsa:
   git push origin main
   ```
2. **Vast.ai hesabı ve kredi.** https://cloud.vast.ai → hesap açın, kredi yükleyin.
3. **SSH anahtarı.** Mac'te anahtarınız yoksa oluşturun ve Vast.ai'ye ekleyin
   (Account → Keys → SSH Keys):
   ```bash
   ls ~/.ssh/id_ed25519.pub || ssh-keygen -t ed25519 -C "you@example.com"
   cat ~/.ssh/id_ed25519.pub     # çıktıyı Vast.ai'ye yapıştırın
   ```
4. **GitHub erişimi.** Depo özelse bulut makinesinin kodu çekebilmesi gerekir. En kolayı
   GitHub'da bir *fine-grained personal access token* (yalnızca bu depo, yalnızca okuma)
   oluşturup klonlarken kullanmaktır (adım 4).

---

## 1. Makine seçimi

Bellek hesabı (yereldeki llama.cpp günlüğünden ölçüldü): model ~5,6 GB; Qwen3.5-9B'nin 32
katmanından yalnızca 8'i tam dikkat kullandığı için önbellek token başına ~32 KB tutar.

| Kurulum | Slot başına bağlam | 16 slot önbellek | Toplam VRAM (yaklaşık) | Uygun GPU |
|---|---|---|---|---|
| Çıktı bütçesi 32.768 (önerilen başlangıç) | 40.960 | ~20 GB | ~27 GB | A100 80GB, H100 80GB, L40S 48GB, RTX A6000 48GB |
| Çıktı bütçesi 81.920 (model kartının yarışma önerisi) | 90.112 | ~44 GB | ~51 GB | A100 80GB, H100 80GB |

Vast.ai arama filtreleri:

```
GPU:          1x A100 80GB / H100 80GB  (48 GB kart da 32.768 bütçe için yeter)
Disk:         ≥ 60 GB
Reliability:  ≥ 0.98
CUDA:         ≥ 12.4
Image:        nvidia/cuda:12.4.1-devel-ubuntu22.04   ← "devel" şart (llama.cpp derlenecek)
```

Fiyatı Vast.ai ekranında görürsünüz; toplam maliyet ≈ saatlik ücret × koşu süresi (adım 9'da
ölçeceğiz). **İş bitince makineyi silmeyi unutmayın** (adım 12) — durdurulmuş makine de disk
ücreti yazar.

---

## 2. Bağlanma

Vast.ai konsolunda makinenin **Connect** düğmesi SSH komutunu verir:

```bash
ssh -p <PORT> root@<HOST>          # örnek: ssh -p 41234 root@ssh5.vast.ai
tmux new -s main                    # bağlantı kopsa da işler sürer
# Bağlantı koparsa: tekrar ssh ile girin ve  tmux attach -t main
```

tmux kısayolları: yeni pencere `Ctrl-b c`, pencereler arası `Ctrl-b n`, ayrılmak `Ctrl-b d`.

---

## 3. Sistem paketleri ve llama.cpp (CUDA)

```bash
apt-get update && apt-get install -y git cmake build-essential curl rsync tmux
nvidia-smi                           # GPU görünüyor mu?

cd /workspace
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DLLAMA_CURL=OFF
cmake --build build --config Release -j --target llama-server
ln -sf /workspace/llama.cpp/build/bin/llama-server /workspace/llama.cpp/llama-server
./llama-server --version
```

Yerelde kullandığımız sürüm `8589 (08f21453a)`. Bulutta bundan **yeni ya da aynı** bir sürüm
olmalı (Qwen3.5 mimarisi yeni sürümlerde destekleniyor). Model yüklenirken
`unknown model architecture: 'qwen35'` hatası alırsanız llama.cpp'yi güncelleyip yeniden derleyin.

---

## 4. Proje kurulumu

```bash
cd /workspace
git clone https://github.com/msergenergen/llm-code-refinement.git
#   (depo özelse)  git clone https://<GITHUB_KULLANICI>:<TOKEN>@github.com/msergenergen/llm-code-refinement.git
cd llm-code-refinement
git log --oneline -1                 # Mac'teki son commit ile aynı mı?

# Python 3.11 (Mac'te kullanılan sürüm) — uv, Python'u kendisi indirir
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env
uv venv -p 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

> **`lcb_runner` kurmayın.** Yereldeki tüm sonuçlar projedeki LCB uyumlu değerlendiriciyle
> alındı. `.env.cloud` içindeki `LCB_EVALUATOR=local` ayarı bunu zaten sabitliyor; yine de
> `docs/setup.md`'deki lcb_runner adımını atlayın.

---

## 5. Ortam dosyası

```bash
cat > .env.cloud << 'EOF'
ENV=cloud
M1_BASE_URL=http://localhost:8001/v1
LLAMA_API_KEY=local

# Yereldekiyle aynı deney ayarları
PIPELINE_MODELS=M1
JUDGE_MODEL=M1               # yargıç = aynı Qwen3.5-9B (yerelde de böyleydi)
MAX_ITERATIONS=5
TREND_EXTENSION_ENABLED=true
MAX_EXTENSION_ITERS=3
TREND_PROJECTION_WINDOW=3
TREND_MIN_SLOPE=1.0
TREND_EMA_ALPHA=0.5
SANDBOX_TIMEOUT_SEC=15
SANDBOX_MEMORY_LIMIT_MB=256
SANDBOX_FLAKINESS_RUNS=2
SANDBOX_MAX_FAILED_CASES=3
LCB_SPLIT=test
LCB_DIFFICULTY=all
LCB_SAMPLE_SIZE=150
LCB_EVALUATOR=local

# Bulutta değişen tek şey: token bütçesi
CODEGEN_MAX_TOKENS=32768
JUDGE_MAX_TOKENS=16384

OUTPUT_DIR=outputs/results
MEMORY_LOG_PATH=outputs/memory.md
PIPELINE_LOG_PATH=outputs/pipeline.log
EOF
ln -sf .env.cloud .env
mkdir -p outputs/results
```

`CODEGEN_MAX_TOKENS=81920` kullanacaksanız adım 7'de `CTX_PER_SLOT=90112` yapın ve 80 GB'lık
kart seçin.

---

## 6. Modeli indirme (yereldeki dosyanın aynısı)

```bash
source /workspace/llm-code-refinement/.venv/bin/activate
mkdir -p /workspace/models/qwen3.5-9b
hf download unsloth/Qwen3.5-9B-GGUF \
    --include "Qwen3.5-9B-UD-Q4_K_XL.gguf" \
    --local-dir /workspace/models/qwen3.5-9b
ls -lh /workspace/models/qwen3.5-9b/      # ~5,6 GB tek dosya
```

`hf` komutu yoksa: `uv pip install -U huggingface_hub`. İndirme yarıda kalırsa aynı komutu
tekrar çalıştırın, kaldığı yerden devam eder.

---

## 7. Model sunucusunu başlatma

Ayrı bir tmux penceresinde (`Ctrl-b c`):

```bash
cd /workspace/llm-code-refinement
PARALLEL=16 CTX_PER_SLOT=40960 bash scripts/start_model.sh M1
```

Günlükte şunları görmelisiniz: `offloaded 33/33 layers to GPU` (tüm katmanlar GPU'da) ve
`server is listening on http://0.0.0.0:8001`. Başka bir pencereden kontrol:

```bash
curl -s localhost:8001/health          # {"status":"ok"}
nvidia-smi                             # bellek kullanımı ~25–30 GB civarı
```

`PARALLEL`, aynı anda işlenebilecek istek sayısıdır; adım 9'daki toplam süreç sayısına eşit
olmalı.

---

## 8. Duman testi (2 problem, ~5–10 dk)

```bash
cd /workspace/llm-code-refinement && source .venv/bin/activate
MAX_PROBLEMS=2 JUDGE_DECISION=verdict python main.py 2>&1 | tee outputs/smoke.log
grep -c "Problem done" outputs/smoke.log            # 2 olmalı
python - << 'EOF'
import json, glob
for p in sorted(glob.glob("outputs/results/*/iterations/*/*.json"))[-3:]:
    d = json.load(open(p)); n3 = d["node3"]
    print(p.split("/")[-2], d["status"], n3.get("verdict"), n3.get("verdict_source"), d["tokens"])
EOF
```

Beklenen: `verdict` alanı ACCEPT/REJECT, `verdict_source` çoğunlukla `judge`,
`codegen_completion` değerleri 6144'ü aşabiliyor (bütçe artışı çalışıyor demektir).

---

## 9. Asıl koşular

İki karar tasarımını, yereldeki 50 problemle aynı problemler üzerinde, aynı anda koşturuyoruz:

- **Y — yargıç kararı** (`JUDGE_DECISION=verdict`): önerdiğimiz tasarım
- **P — puan kararı** (`JUDGE_DECISION=rubric`): karşılaştırma tasarımı

Her tasarım 8 parçaya bölünür (toplam 16 süreç = `PARALLEL=16`):

```bash
cd /workspace/llm-code-refinement && source .venv/bin/activate
mkdir -p outputs/logs
for DESIGN in verdict rubric; do
  for i in $(seq 0 7); do
    MAX_PROBLEMS=50 JUDGE_DECISION=$DESIGN SHARD_COUNT=8 SHARD_INDEX=$i \
      nohup python main.py > outputs/logs/${DESIGN}_shard$i.log 2>&1 &
    sleep 2        # her parçaya ayrı run_id (saniye damgalı) düşsün
  done
done
```

**Süreyi ölçün:** 20–30 dakika sonra

```bash
cat outputs/logs/*.log | grep -c "Problem done"     # biten problem sayısı (iki tasarım toplamı)
```

Toplam 100 problem-koşusu var (50 × 2 tasarım). Bitmiş sayı / geçen süre oranından kalan süreyi
hesaplayın. Süre uygunsa 50 bittikten sonra aynı komutu `MAX_PROBLEMS=150` ve
`SKIP_QIDS=$(paste -sd, benchmark/sample50_qids.txt)` ile çalıştırıp kalan 100 problemi de
koşturabilirsiniz (liste depoda: `benchmark/sample50_qids.txt`).

---

## 10. İzleme

```bash
tail -f outputs/logs/verdict_shard0.log                     # tek parça
grep -h "Problem done" outputs/logs/*.log | wc -l           # toplam ilerleme
grep -l "Traceback" outputs/logs/*.log                      # çöken parça var mı?
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
pgrep -fc "python main.py"                                  # çalışan süreç sayısı (16)
```

Bir parça çökerse yalnızca o parçayı yeniden başlatın; biten problemleri atlamak için
`SKIP_QIDS` kullanın (çöken günlükteki `Problem done` satırlarındaki kimlikler).

---

## 11. Sonuçları Mac'e alma ve analiz

Mac'te (bulut makinesinde değil):

```bash
cd ~/Desktop/llm-code-refinement
mkdir -p outputs/results/cloud
rsync -avz -e "ssh -p <PORT>" \
    root@<HOST>:/workspace/llm-code-refinement/outputs/results/ outputs/results/cloud/
rsync -avz -e "ssh -p <PORT>" \
    root@<HOST>:/workspace/llm-code-refinement/outputs/logs/ outputs/results/cloud/logs/

# Hangi run_id hangi tasarıma ait?
grep -h "run_id:" outputs/results/cloud/logs/verdict_shard*.log | awk '{print $NF}'
grep -h "run_id:" outputs/results/cloud/logs/rubric_shard*.log  | awk '{print $NF}'
```

Gizli testlerle değerlendirme Mac'te yapılır (yereldeki sonuçlarla aynı değerlendirici):

```bash
python -m benchmark.paper_results \
    --results-root outputs/results/cloud \
    --run-ids <verdict run_id'leri, boşlukla> \
    --baseline-run-ids <rubric run_id'leri> \
    --qid-file benchmark/sample50_qids.txt \
    --out /tmp/cloud_sonuclar
```

Bu komutu Claude'a bırakabilirsiniz; bildiriye ikinci bütçe koşulu olarak eklenecek.

---

## 12. Makineyi kapatma

Sonuçları Mac'e aldığınızı doğruladıktan sonra Vast.ai konsolunda makineyi **Destroy** edin.
Yalnızca *Stop* demek disk ücretini sürdürür.

---

## Sorun giderme

| Belirti | Neden / çözüm |
|---|---|
| `cudaMalloc failed` / OOM | `PARALLEL` ya da `CTX_PER_SLOT` fazla. Önce `PARALLEL=8` deneyin (adım 9'da parça sayısını 4'e indirin) |
| Çok yavaş, GPU kullanımı düşük | Günlükte `offloaded 33/33 layers` yoksa model CPU'da kalmış; `--n-gpu-layers 99` ve CUDA derlemesini kontrol edin |
| `unknown model architecture` | llama.cpp eski; `git pull` + yeniden derleme (adım 3) |
| `Cannot reach` / bağlantı hatası | Sunucu kapanmış ya da port farklı; `curl localhost:8001/health` |
| `datasets` yükleme hatası | `datasets<3.0.0` gerekli (LiveCodeBench yükleme betiği); `uv pip install "datasets<3.0.0"` |
| `radon` / `bandit` bulunamadı | venv etkin değil; `source .venv/bin/activate` |
| `resource` / sandbox hataları | Linux'ta çalışır; Mac'e özgü bir ayar gerekmez |
| SSH koptu | Tekrar bağlanıp `tmux attach -t main`; `nohup` ile başlatılan süreçler etkilenmez |
| Aynı problem iki kez | Parçalar örtüşmez; `SKIP_QIDS` / `MAX_PROBLEMS` / `SHARD_*` önce bu sırayla uygulanır |

Ayrıntılı ayar açıklamaları: `docs/config.md`; yargıç kararı tasarımı: `docs/node3.md` §4.2;
analiz betiği: `docs/analysis.md` §9.
