# cloud-comparison.md — Cloud Platform Comparison

> **Durum (2026-10-02): tarihsel belge.** Ana deney artık GPU kiralamadan OpenRouter API üzerinden koşuluyor (`CLOUD_README.md` §A). Vast.ai yalnızca isteğe bağlı ek bulgu için (§B). Aşağıdaki karşılaştırma eski model setine göredir.

> Goal: Select the best platform to run the iterative code refinement pipeline (gpt-oss-20b + Qwen3-Coder-Next + GLM-4.7-Flash + Qwen3.5-35B-A3B)
> using an on-premise GGUF / llama.cpp architecture.

> **⚠️ OUT OF DATE:** This document's hardware/cost analysis assumes the removed M2 model (Qwen3-Coder-Next, ~46 GB "dominant model"). The current registry (M1=Qwen3.5-9B ~6GB, M2=Qwen2.5-Coder-1.5B ~1GB, QWEN=~24GB) requires much less memory; A100 80GB is likely no longer necessary, but the numbers below should not be trusted without re-verification against current Vast.ai/RunPod pricing. Existing tables are preserved as historical record.

---

## 1. Core Requirement: Sequential Models, Single Memory

M1, M2, M3 do not run simultaneously. After each model finishes, its server is stopped
and the next one starts. Memory requirement is determined by the **largest single model**.

| Model | GGUF Format | Tahmini Boyut | Gereksinim |
|-------|-------------|---------------|------------|
| gpt-oss-20b (M1) | UD-Q4_K_XL | ~11 GB | Small; runs even on RTX 4090 |
| Qwen3-Coder-Next (M2) | UD-Q4_K_XL | ~46 GB | **Dominant model** |
| GLM-4.7-Flash (M3) | UD-Q4_K_XL | ~18 GB | Smaller than Qwen |
| Qwen3.5-35B-A3B (Judgement) | UD-Q4_K_XL | ~24 GB | Runs after each model |

**Practical requirement:** A100 80GB VRAM alone is sufficient — 256 GB RAM is **not required**.

---

## 2. Platform Comparison

### 2.1 Vast.ai

**Structure:** Peer-to-peer GPU marketplace — individuals and data centers list their hardware.

| Criterion | Assessment |
|--------|--------------|
| Price (A100 80GB) | ~$0.67/hr — cheapest option |
| High RAM instance | ✅ Nodes with 256GB+ RAM available |
| SSH + full control | ✅ Bare metal access, install anything |
| llama.cpp compat. | ✅ Direct compilation and execution |
| Reliability | ⚠️ Varies by host (filter reliability score: > 0.95) |
| Persistent storage | ✅ Paid volume, sufficient for model downloads |
| Learning curve | Medium — marketplace UI takes some getting used to |

**Strength:** Lowest hourly cost, SSH access, high-RAM node option.

**Weakness:** Host reliability is inconsistent; a poorly chosen node may shut down mid-run.
Check Reliability > 0.95 and uptime history.

---

### 2.2 RunPod

**Structure:** Managed GPU cloud — both Community Cloud (cheap, variable) and Secure Cloud (stable).

| Criterion | Assessment |
|--------|--------------|
| Fiyat (A100 80GB) | ~$1.19/sa Community, ~$1.79/sa Secure |
| High RAM instance | ⚠️ 256GB+ RAM option is limited and more expensive |
| SSH + tam kontrol | ✅ Pod sistemi, SSH ve JupyterLab |
| llama.cpp compat. | ✅ Bring your own container |
| Reliability | ✅ More consistent than Vast.ai |
| Persistent storage | ✅ Network Volume $0.07/GB/month |
| Learning curve | Low — template system starts quickly |

**Strength:** More predictable experience, good documentation, fast setup templates.

**Weakness:** Price is 40–80% higher than Vast.ai. Finding a 256GB RAM node is difficult
and expensive when available.

---

### 2.3 Modal

**Structure:** Serverless GPU platform — Python code is deployed, container auto-scales.

| Criterion | Assessment |
|--------|--------------|
| Fiyat (A100 40GB) | ~$0.001036/sn = ~$3.73/sa |
| High RAM instance | ❌ No 256GB RAM node in serverless architecture |
| SSH + persistent server | ❌ New container per request, model reloads every time |
| llama.cpp compat. | ❌ Serverless paradigm does not support persistent inference server |
| Reliability | ✅ Managed, scales automatically |
| Learning curve | High — requires transition to serverless paradigm |

**Verdict: Modal is not suitable for this use case.**

Qwen3-Coder-Next (~46 GB) would need to reload on every API request —
this takes minutes, which the pipeline cannot afford each iteration. Modal's strength
is in stateless, short-lived inference; it is incompatible with the llama.cpp persistent server model.

---

## 3. Side-by-Side Summary

| | Vast.ai | RunPod | Modal |
|--|---------|--------|-------|
| **A100 80GB price** | ~$0.67/hr | ~$1.19/hr | ~$3.73/hr |
| **256GB+ RAM node** | ✅ Available | ⚠️ Limited | ❌ None |
| **Persistent llama.cpp server** | ✅ | ✅ | ❌ |
| **SSH / tam kontrol** | ✅ | ✅ | ❌ |
| **Reliability** | ⚠️ Variable | ✅ Good | ✅ Good |
| **Suitability for this pipeline** | ✅ **Best** | ✅ Alternative | ❌ Not suitable |

---

## 4. Recommendation: Vast.ai

Rationale:

1. **Memory flexibility.** Nodes with 256GB+ RAM are listed on the marketplace. Reaching this
   configuration on RunPod is harder and more expensive.
2. **Cost.** Vast.ai ~$0.67/hr vs RunPod ~$1.19/hr for A100 80GB. Over a 40-hour
   benchmark run, the difference is ~$20.
3. **Full control.** Direct SSH connection; compile llama.cpp, download models,
   run with any parameters. No managed layer.
4. **Pipeline validated on Mac Mini transfers to Vast.ai** — same environment (Ubuntu + llama.cpp),
   only different hardware.

**When to prefer RunPod:** If you cannot find a reliable 256GB+ node on Vast.ai,
switch to RunPod Secure Cloud. The price difference is acceptable for research reliability.

---

## 5. Vast.ai Instance Selection Criteria

### Filters (in the Vast.ai interface)

```
Disk Space:  ≥ 300 GB
RAM:         ≥ 64 GB
GPU RAM:     ≥ 80 GB
Reliability: ≥ 0.95
CUDA:        ≥ 12.1
Image:       nvidia/cuda:12.1.0-base-ubuntu22.04
```

### Checks (before selecting an instance)

- [ ] Host uptime history ≥ 30 days
- [ ] Disk type SSD (HDD will make model loading very slow)
- [ ] Internet bandwidth ≥ 500 Mbps (for model downloads)
- [ ] No sudden drops in the reliability score graph

---

## 6. Maliyet Tahmini

Due to sequential execution, only one model and one server run at a time.
Dominant model Qwen3-Coder-Next (~46 GB) → minimum A100 80GB (~$0.80/hr).

| Scenario | Duration | A100 80GB (~$0.80/hr) |
|---------|------|----------------------|
| Mac Mini validation (5 easy questions, dummy mode) | Free | $0 |
| Pilot benchmark: 5 soru × 3 model | ~1.5 sa | ~$1.20 |
| Ana benchmark: 50 soru × 3 model | ~13 sa | **~$11** |
| Tam tez benchmark: 200 soru × 3 model | ~50 sa | ~$40 |
| Ablation (tek model, 50 soru) | ~4.5 sa | ~$3.60 |

> Qwen3-Coder-Next ve gpt-oss-20b'nin MoE mimarisinde A100 80GB'da
> MoE CPU offload is not needed; all layers fit on GPU.
> GLM-4.7-Flash has been reported at 110 tok/s on RTX 4090 —
> this is the fastest component of the pipeline.

---

## 7. Development → Production Migration Plan

```
1. Mac Mini M4 (Ortam 1 — Local Debug)
   ENV=local, Qwen2.5-Coder-7B (codegen) + Qwen2.5-Coder-3B (judgement), 5 easy soru
   → Pipeline akışı, gate'ler, memory.md doğrulandı

2. Vast.ai (Ortam 2 — GLM-4.7-Flash ile başla)
   ENV=cloud, sadece M3 (GLM) aktif, 5 soru
   → Gerçek model + Vast.ai kurulumu doğrulandı; en hızlı kurulan model

3. Vast.ai (Ortam 2 — gpt-oss-20b ekle)
   ENV=cloud, M1 + M3, 10 soru
   → gpt-oss-20b Harmony format, --jinja doğrulandı

4. Vast.ai (Ortam 2 — Tam pipeline)
   ENV=cloud, M1 + M2 + M3 + QWEN, 50 soru
   → Tez için ana benchmark çalışması

5. Vast.ai (Ortam 2 — Ablation)
   Sadece M1, sadece Node 1+2 (Node 3 yok) vb.
   → Her node'un katkısı ölçülür
```

> Even though Qwen3-Coder-Next (M2) is the largest model, 46 GB fits
> comfortably in an A100 80GB. Validate the Vast.ai environment first with
> the smaller GLM-4.7-Flash, then add Qwen3-Coder-Next.
