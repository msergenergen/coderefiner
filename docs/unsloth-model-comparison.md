# unsloth-model-comparison.md — Unsloth Model Comparison

> **Durum (2026-10-02): tarihsel belge.** Model seçimi bu tablodan sonra değişti: yerel ek bulgu Qwen3.5-9B, ana deney sızıntısız tasarımla OpenRouter üzerinde DeepSeek-V3 (üretici) + Qwen2.5-72B-Instruct (yargıç).3-70B (yargıç). Gerekçe: `docs/dataset.md` §10, `docs/ablation.md` §7. Tablolar kayıt amacıyla korunuyor.

> Criteria: LiveCodeBench (LCB) pass@1 performance · Unsloth GGUF memory requirement · Vast.ai cost estimate  
> Kaynak: Unsloth Model Catalog · Artificial Analysis · Onyx Coding Leaderboard · BenchLM.ai (Mart 2026)  
> Note: LCB pass@1 scores come from different test windows (v5/v6 mixed); use for relative ranking, not absolute comparison.

---

## 1. Pipeline Models (Current Selection)

| Model | Unsloth GGUF | Total Params | Active Params | LCB pass@1 | Min. Memory (Q4) | Vast.ai Hardware | Est. Cost/hr |
|-------|-------------|-------------|------------|------------|-----------------|-----------------|-------------------|
| **Kimi K2.5** (M1) | UD-IQ1_S 1-bit | ~1T (MoE) | 32B | **~85%** | 240 GB (1-bit) | 256 GB RAM + 1×A100 | ~$2.00/sa |
| **GLM-4.7-Flash** (M2) | GGUF Q4 | ~14B (Flash) | ~14B | ~65–70%¹ | ~9 GB | 1×RTX 4090 | ~$0.50/sa |
| **MiniMax M2.5** (M3) | GGUF Q4 | 230B (MoE) | 46B | **65%** | ~130 GB | 2×A100 80GB | ~$1.40/sa |
| **Qwen3.5-35B-A3B** (Judgement) | GGUF Q4 | 35B (MoE) | 3B | ~62%² | ~24 GB | 1×RTX 4090 | ~$0.50/sa |

> ¹ GLM-4.7-Flash LCB score not separately reported; GLM-4.7 (358B) scores 84.9%, Flash version is much smaller.  
> ² Qwen3.5-35B-A3B general coding benchmark; no official LCB-specific data.

---

## 2. High Performance Tier (LCB > 75%)

Models that could be alternatives to or outperform the current pipeline models.

| Model | Total Params | Active Params | LCB pass@1 | Min. Memory (Q4) | Vast.ai Hardware | Est. Cost/hr | Notes |
|-------|-------------|------------|------------|-----------------|-----------------|-------------------|--------|
| **GLM-4.7** | 358B (MoE) | ~36B | **84.9%** | ~200 GB | 3×A100 80GB | ~$2.00/hr | Equal to Kimi K2.5 on LCB |
| **Kimi K2.5** | ~1T (MoE) | 32B | **85.0%** | 240 GB (1-bit) | 256 GB RAM + GPU | ~$2.00/sa | HumanEval 99%, en iyi kodlama modeli |
| **Qwen3-Coder-Next** | 80B (MoE) | 3B | ~78%³ | **46 GB** | 2×RTX 4090 or 1×A100 | ~$0.80/hr | SWE-bench Pro 44.3%; most efficient large model |
| **DeepSeek-V3.1** | ~370B (MoE) | ~37B | ~80%⁴ | ~200 GB | 3×A100 80GB | ~$2.00/hr | MIT license; wide support ecosystem |

> ³ Qwen3-Coder-Next official LCB not published; estimated from SWE-bench Pro leadership (44.3%) and active training.  
> ⁴ DeepSeek-V3.2 at 74.1%, V3.1 in a similar range.

---

## 3. Mid Tier (LCB 50–75%, good efficiency/cost balance)

| Model | Total Params | Active Params | LCB pass@1 | Min. Memory (Q4) | Vast.ai Hardware | Est. Cost/hr | Notes |
|-------|-------------|------------|------------|-----------------|-----------------|-------------------|--------|
| **Qwen3-Coder-30B-A3B** | 30B (MoE) | 3B | ~68%⁵ | ~18 GB | 1×RTX 4090 | ~$0.50/hr | Small active params, fast inference |
| **gpt-oss-120B** | 120B | ~120B | ~60%⁶ | ~66 GB | 1×A100 80GB | ~$0.80/hr | Transparent architecture; MIT license |
| **Qwen3.5-27B** | 27B | 27B | ~60%² | ~18 GB | 1×RTX 4090 | ~$0.50/sa | Dense model, tahmin edilebilir latency |
| **Llama 4 Maverick** | ~400B (MoE) | 17B | ~58%⁷ | ~220 GB | 256 GB RAM + GPU | ~$2.00/hr | 1M token context; for large context tasks |
| **MiniMax M2.5** | 230B (MoE) | 46B | **65%** | ~130 GB | 2×A100 80GB | ~$1.40/hr | SWE-bench 80.2% — real-world code repair champion |
| **GLM-4.7-Flash** | ~14B | ~14B | ~65–70%¹ | ~9 GB | 1×RTX 4090 | ~$0.50/hr | Fastest; ideal as pipeline M2 |

> ⁵ Qwen3-Coder-30B-A3B runs with 3B active params; small but has MoE advantage.  
> ⁶ gpt-oss-120B LCB score not reported; mid-tier performance observed on SWE-rebench.  
> ⁷ Llama 4 Maverick general LCB; approximate from Meta's report.

---

## 4. Efficient / Small Models (for Judgement or Dummy Testing)

| Model | Params | LCB pass@1 | Min. Memory (Q4) | Vast.ai Hardware | Est. Cost/hr | Notes |
|-------|-------|------------|-----------------|-----------------|-------------------|--------|
| **gpt-oss-20B** | 20B | ~45%⁶ | ~11 GB | 1×RTX 4090 | ~$0.50/hr | Sufficient for judgement; small and fast |
| **Qwen3-32B** | 32B | ~55% | ~18 GB | 1×RTX 4090 | ~$0.50/hr | Dense; strong reasoning |
| **Llama 3.3-70B** | 70B | ~50% | ~39 GB | 1×A6000 48GB | ~$1.10/hr | Mature ecosystem; easy to fine-tune |
| **Llama 4 Scout** (17B-16E) | ~280B (MoE) | ~52% | ~18 GB (4-bit) | 1×RTX 4090 | ~$0.50/hr | Active 17B; 1M context; fast |
| **Qwen3.5-9B** | 9B | ~40%² | ~6 GB | 1×RTX 3090 | ~$0.35/hr | Ideal for Mac Mini dummy testing |
| **Qwen2.5-Coder-7B** | 7B | ~38% | ~4 GB | 1×RTX 3090 | ~$0.35/hr | Recommended dummy model for Mac Mini M4 16GB |

---

## 5. Memory Requirements and Vast.ai Hardware Mapping

Due to sequential execution, only one model resides in memory at a time. The largest model's requirement determines the hardware.

| Memory Range | Required Hardware | Vast.ai Est. Price | Examples |
|---------------|-----------------|----------------------|---------|
| < 10 GB | 1× RTX 3090/4090 (24 GB VRAM) | ~$0.35–$0.50/sa | GLM-4.7-Flash, Qwen3.5-9B, Qwen2.5-Coder-7B |
| 10–25 GB | 1× RTX 4090 (24 GB VRAM) | ~$0.50/sa | gpt-oss-20B, Qwen3-30B-A3B, Qwen3.5-35B-A3B |
| 25–50 GB | 1× A6000 (48 GB) veya 1× A100 (40 GB) | ~$0.80–$1.10/sa | Qwen3-Coder-Next, gpt-oss-120B, Llama 3.3-70B |
| 50–90 GB | 1× A100 (80 GB) | ~$0.80/hr | gpt-oss-120B (borderline) |
| 90–170 GB | 2× A100 80 GB | ~$1.40–$1.60/hr | MiniMax M2.5, GLM-4.7-Flash dual setup |
| > 200 GB | 3× A100 or high-RAM node | ~$2.00–$2.50/hr | GLM-4.7, DeepSeek-V3.1, Kimi K2.5 |

---

## 6. 50-Question Benchmark Cost Estimate

Assumptions: 3 models sequential × avg 5 iterations × 45 sec/iteration × 50 questions = ~94 hours total.  
This is active GPU time; model switching durations (~5–20 min/model) are included.

| Pipeline Combination | Total Memory | Hardware | Hourly Cost | 50-Question Estimate |
|----------------------|--------------|---------|-----------------|-----------------|
| **Current: Kimi K2.5 + GLM-4.7-Flash + MiniMax M2.5** | 240 GB (Kimi dominant) | 256 GB RAM node | ~$2.00/hr | **~$190–220** |
| **Alt A: GLM-4.7-Flash + Qwen3-Coder-Next + gpt-oss-20B** | 46 GB (Qwen dominant) | 2×RTX 4090 | ~$0.80/hr | **~$75–90** |
| **Alternatif B: Qwen3-Coder-Next × 3 (tek model)** | 46 GB | 1×A100 40GB | ~$0.80/sa | **~$75–85** |
| **Alternatif C: GLM-4.7-Flash × 3 (en ekonomik)** | 9 GB | 1×RTX 4090 | ~$0.50/sa | **~$47–55** |
| **Mac Mini Dummy (Qwen2.5-Coder-7B × 4)** | 4 GB | Apple M4 Metal | $0 | **$0** |

---

## 7. Optimal Model Recommendations by Pipeline Role

### Performance-First
| Role | Recommendation | LCB | Memory | Rationale |
|-----|-------|-----|--------|---------|
| M1 (ilk deneme) | **Kimi K2.5** | 85% | 240 GB | LCB ve HumanEval'de zirve |
| M2 (second attempt) | **GLM-4.7-Flash** | ~68% | 9 GB | Fast and small; where Kimi doesn't fit |
| M3 (third attempt) | **Qwen3-Coder-Next** | ~78% | 46 GB | Best efficiency/memory after Kimi K2.5 |
| Judgement | **Qwen3.5-35B-A3B** | — | 24 GB | Independent judge; low self-enhancement risk |

### Cost-First (budget-constrained thesis work)
| Role | Recommendation | LCB | Memory | Rationale |
|-----|-------|-----|--------|---------|
| M1 | **Qwen3-Coder-Next** | ~78% | 46 GB | Near Kimi K2.5 performance at ~30% of the cost |
| M2 | **GLM-4.7-Flash** | ~68% | 9 GB | Cheapest strong option |
| M3 | **gpt-oss-20B** | ~45% | 11 GB | Sufficient for ablation testing |
| Judgement | **gpt-oss-20B** | — | 11 GB | Fast and cheap; runs on the same hardware |

---

## 8. Important Notes

- **LCB version gap**: Some scores are from LCB v5 (Jan 2025), others from v6 (Apr 2025). The same model's v5/v6 scores may differ; rankings are relatively correct.
- **MoE memory**: All parameters are held in memory; only a fraction is computed during activation. Therefore total parameter count determines memory requirement, not active parameter count.
- **Unsloth Dynamic GGUF advantage**: Unsloth's Dynamic 2.0 quants offer better quality than standard GGUF at the same bit width. Higher precision is applied to critical layers. Score degradation compared to standard GGUF is lower.
- **CPU Offload**: In MoE models, llama.cpp `-ot ".ffn_.*_exps.=CPU"` can offload MoE layers to CPU. This reduces memory by 40–60% but lowers speed (~3–10 tok/s).
- **Vast.ai prices**: Actual prices vary by host and real-time supply/demand. March 2026 averages. Filter before renting: Reliability > 0.95, SSD disk.
