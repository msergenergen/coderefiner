# future-work.md — Future Work Directions

## Overview

This document outlines future research directions that extend the current thesis work.
Items are organized by scope: **thesis-scope additions** that can strengthen the
current submission, and **post-thesis directions** for follow-up research.

---

## Thesis-Scope Additions

These can be incorporated into the current work with modest additional effort.

### T1. Small Model Framing — Efficiency vs. Scale

**Status (2026-09-30): promoted from future-work to the primary experimental
design.** This item originally proposed extracting a size-vs-gain analysis as
a *secondary* by-product of an evaluation built around three cross-vendor
models (M1=gpt-oss-20b, M2=Qwen3-Coder-Next, M3=GLM-4.7-Flash). That design
was replaced: the cloud registry (`config/models.py`) now has exactly two
codegen models, both same-family Qwen models at two size tiers —
M1=Qwen3.5-9B (medium anchor, ~6 GB) and M2=Qwen2.5-Coder-1.5B-Instruct
(small anchor, ~1 GB) — with the judge (QWEN=Qwen3.5-35B-A3B) unchanged and
independent. The reasoning: a three-cross-vendor-model comparison dilutes the
paper's actual claim (the pipeline improves outcomes) into a "which model is
better" question, and confounds model size with architecture/vendor
differences. A same-family, two-size design isolates size as the controlled
variable and makes T1's core claim — that structured feedback narrows the
gap between a medium and a small model — the paper's primary, not
secondary, empirical result. Full rationale: main.tex §IV-A, CLAUDE.md
Models table, `config/models.py` module docstring.

**Core claim (unchanged):**
> Systematic multi-stage feedback can meaningfully improve task success rate
> without increasing model size — challenging the assumption that better code
> generation requires larger models.

**What to extract post-cloud (updated for the 2-model design):**

```
M1 (Qwen3.5-9B,            medium, ~9B)   V0 TSR → V3 TSR   ΔTSR = ?
M2 (Qwen2.5-Coder-1.5B,    small,  ~1.5B) V0 TSR → V3 TSR   ΔTSR = ?
```

**Key visualization:**
Plot model size (B parameters) on the x-axis against pipeline contribution
(ΔTSR = V3 − V0) on the y-axis. Expected pattern: larger gain for the smaller
model (pipeline matters more), smaller gain for the medium model (ceiling
effect). With only two points this is a single-line comparison, not a curve
— frame it as "does the gap narrow" rather than implying a fitted trend.

**Academic positioning:**
Connects to the test-time compute vs. model scale debate
(Snell et al., 2024; Jones, 2021). The framing shifts from
"which model is best?" to "how much can structured feedback compensate for model size?"

---

### T2. Learned Metric Gating — Logistic Regression Before Node 3

**Relationship to sensitivity.md:**
`docs/sensitivity.md` Analiz 2 (Boyut Önem Analizi) and Analiz 3 (Eşik Duyarlılığı)
provide the statistical groundwork for T2. Those analyses identify which metrics
are actually predictive; T2 replaces the fixed thresholds with a learned model.
Run sensitivity analyses first; implement T2 only if predictive signal is strong.

**Core idea:**
Replace fixed, hand-tuned thresholds (CC ≤ 10, MI ≥ 65, pass_rate ≥ 0.90)
with a learned gating model that predicts whether the current iteration will
pass judgement — before calling the expensive judgement LLM.

**Model formulation:**

```
Input features:
  - node1_cc, node1_mi, node1_halstead       (static analysis)
  - node2_pass_rate, node2_cpu_ms, node2_ram_mb  (execution)
  - difficulty (one-hot: easy/medium/hard)
  - iteration_number (normalized)

Output:
  P(this iteration will receive PASS from Node 3)

Training signal:
  Binary label — did this iteration actually receive PASS?
```

**Two usage modes:**

*Early termination:* If P(PASS) is high, skip Node 3 entirely.
Reduces judgement LLM calls, lowers cost per benchmark run.

*Threshold adaptation:* Replace fixed pass thresholds with
difficulty-aware and iteration-aware dynamic thresholds.
An easy problem on iteration 3 warrants a different acceptance
criterion than a hard problem on iteration 9.

**Data availability:**
Pipeline runs produce ~750 labeled iterations (50 questions × 3 models × avg 5 iterations).
Sufficient for logistic regression with 8 features; cross-validation required
to guard against overfitting.

**Academic novelty:**
Existing LLM-as-a-Judge work delegates all evaluation to the LLM.
A learned intermediate layer between numeric metrics and LLM judgement —
"learned metric gating for LLM judgement" — has no direct precedent in the literature.

**Ablation opportunity:**
Compare three configurations:
(a) fixed thresholds only (current V3),
(b) logistic gate only,
(c) logistic gate + Node 3.
Isolates the contribution of each evaluation layer.

---

### T3. Cross-Model Feedback Transfer

**Core idea:**
When M1 fails and the pipeline advances to M2, M2 currently starts from scratch.
The `improved_prompt` values generated during M1's iterations contain specific,
actionable information about what went wrong — this information is discarded.

**Proposed extension:**
Carry forward a condensed version of M1's failure history when initializing M2:

```python
# Instead of:
initial_state["node3_improved_prompt"] = ""

# Use:
initial_state["node3_improved_prompt"] = summarize_failures(m1_iterations)
```

The summarization can be rule-based (top-N failure reasons by frequency)
or LLM-based (judgement model summarizes M1's trajectory).

**Academic framing:**
"Cross-model knowledge transfer via structured feedback" —
an application of curriculum learning principles to iterative code refinement.
The downstream model benefits from the upstream model's trial-and-error.

**Measurable impact:**
Compare M2 V3 with and without M1 feedback transfer.
If M2 reaches PASS in fewer iterations when initialized with M1's failures,
transfer is effective.

---

## Post-Thesis Directions

These require additional data, infrastructure, or scope beyond the current thesis.

### P1. Domain-Specific Judge Fine-Tuning

**Motivation:**
The current pipeline uses Qwen3.5-35B-A3B as a general-purpose judge.
A judge fine-tuned specifically for iterative code refinement could produce
more consistent scores and more actionable `improved_prompt` outputs.

**Recommended approach — DPO (Direct Preference Optimization):**

Collect preference pairs from pipeline runs:

```
Preferred:  improved_prompt from an iteration that led to PASS
Rejected:   improved_prompt from an iteration that led to continued FAIL
```

Fine-tune the judge on these pairs. This directly aligns the judge's
feedback generation with downstream success — a clear, measurable reward signal.

**Why DPO over SFT:**
SFT on (input, improved_prompt) pairs teaches the judge to imitate
past outputs. DPO teaches it to produce outputs that cause success.
The optimization target is downstream pass rate, not output similarity.

**Data requirement:**
Approximately 500–1000 preference pairs for meaningful fine-tuning.
This requires running the full pipeline on a larger question set than
the 50-question thesis benchmark.

**Longer-term direction — Reward Model as Judge:**
Frame the judge as a learned reward function for code quality.
Test-time compute + learned reward signal is a novel combination;
its code-specific instantiation does not exist in the literature.

**Concrete phase roadmap (extends the existing `judge_training/` Faz 0–2 naming):**

The current `judge_training/` subsystem (`build_dataset.py` → `train.py`)
implements Faz 0 (cloud pipeline logging) through Faz 2 (QLoRA SFT). Faz 2 is
pure teacher-imitation distillation: the training labels are the existing
QWEN judge's own outputs, so the resulting model is capped at teacher
quality — it cannot learn anything the teacher didn't already know, including
whether its own `improved_prompt` actually worked downstream.

- **Faz 3 — Preference pair construction (near-zero additional cost).**
  Extend `build_dataset.py` to pair iterations sharing the same
  `(problem, node1_*, node2_*, stripped_code)` state: label the
  `improved_prompt` whose *next* iteration's `total_score` increased as
  `chosen`, the one that didn't as `rejected`. Fully derivable from existing
  `outputs/results/` iteration JSONs — no new pipeline runs required.
- **Faz 4 — DPO training.** Swap `train.py`'s `SFTTrainer` for `trl`'s
  `DPOTrainer`, initialized from the Faz 2 checkpoint (standard
  SFT-then-preference-tuning pattern, as in RLHF/InstructGPT). Unlike Faz 2,
  the optimization target is downstream success, not output similarity.
- **Faz 5 (post-thesis) — GRPO online refinement.** Group Relative Policy
  Optimization is a strong fit here because the pipeline already *is* the
  RL environment: for a given state, sample K `improved_prompt` candidates
  from the Faz 4 policy, roll each forward through one real
  `codegen_node` → `static_analysis_node` → `sandbox_node` call, and use
  the within-group relative ranking of the resulting deterministic score
  (`metrics/scoring.py`) as the advantage signal — no separate critic/reward
  model needed, unlike PPO. Scope to a small problem subset (~10–15, not the
  full 50) given the per-step rollout cost; this is a follow-up publication
  scope, not a thesis-timeline item.
- **Evaluation.** Add a new ablation arm — V3+zero-shot-judge (current) vs.
  V3+SFT-judge (Faz 2) vs. V3+DPO-judge (Faz 4) — using the same TSR/McNemar
  methodology already established in `benchmark/analysis.py`.

**Literature gap — update (2026-09-30):** the initial gap noted above (no
references covering RL/preference-trained *judge* models) has been partially
closed by a dedicated search, verified paper-by-paper against the live arXiv
API (title + full author list matched, not taken from a search snippet).
CodeRL [3] remains the closest code-specific RL precedent but applies RL with
execution-based reward to the *generator*, not a judge/critic; Reflexion [7]
and CRITIC [9] are training-free self-correction. None of the newly found
references train a *code-specific* judge/critic via RL or preference
learning either — this remains the genuine gap and the paper's claimed
contribution — but the following now ground the *method*:

- **DPO** — Rafailov et al., "Direct Preference Optimization: Your Language
  Model is Secretly a Reward Model," NeurIPS 2023 (arXiv:2305.18290).
- **RLHF pipeline template** — Dong et al., "RLHF Workflow: From Reward
  Modeling to Online RLHF," TMLR 2024 (arXiv:2405.07863) — the reward-model
  → online-RLHF pipeline this project's Faz 3/4 roadmap adapts.
- **GRPO** — Deng et al., "Latent-GRPO" (arXiv:2604.27998, preprint) and
  Zhang et al., "Scaf-GRPO," ICLR 2026 (arXiv:2510.19807) — GRPO variants
  for the proposed Faz 5 online refinement.
- **Verifiable-reward RL infrastructure** — Stojanovski et al., "Reasoning
  Gym: Reasoning Environments for Reinforcement Learning with Verifiable
  Rewards," NeurIPS 2025 Spotlight (arXiv:2505.24760).
- **Process reward models (closest to "judge as reward function")** — Chen
  et al., "From Mathematical Reasoning to Code: Generalization of Process
  Reward Models in Test-Time Scaling" (arXiv:2506.00027, preprint); Zhao et
  al., "GenPRM" (arXiv:2504.00891, preprint); Shao et al., "DeepSeekMath-V2:
  Towards Self-Verifiable Mathematical Reasoning" (arXiv:2511.22570,
  preprint) — a self-verifying judge component, conceptually close to P1's
  "Reward Model as Judge" direction, though not code-specific.
- **Judge fine-tuning generalization (general-purpose, not code)** — Singh
  et al., "On the Shelf Life of Fine-Tuned LLM-Judges," ICLR 2026
  (arXiv:2509.23542) — studies SFT vs. DPO-trained judges under distribution
  shift; the closest existing work to this project's judge-training
  question, but for general LLM judges, not code.
- **Iterative refinement with deterministic feedback** — Bi et al.,
  "Iterative Refinement of Project-Level Code Context for Precise Code
  Generation with Compiler Feedback" (CoCoGen), arXiv:2403.16792, 2024 —
  applies a similar deterministic-feedback loop at project scope; a claimed
  ICSE 2024 venue could not be independently confirmed via arXiv metadata,
  so cite as an arXiv preprint only.

Four of these (GenPRM, DeepSeekMath-V2, Math→Code PRM, Latent-GRPO) are
arXiv preprints without confirmed peer review as of this writing — cite
accordingly, not as accepted publications.

---

### P2. Difficulty-Aware Prompt Adaptation

**Motivation:**
The current `improved_prompt` is generated with the same template regardless
of problem difficulty. Easy problems may benefit from short, direct feedback
("add a bounds check"). Hard problems may require algorithm-level suggestions
("replace the O(n²) nested loop with a hash map approach").

**Proposed extension:**
Add a difficulty conditioning signal to the judgement system prompt.
Provide the judge with explicit instructions on feedback granularity
per difficulty tier:

```
Easy:   Identify the single most impactful fix. Be concise.
Medium: Address correctness first, then efficiency.
Hard:   Suggest an alternative algorithmic approach if pass_rate < 0.4.
```

**Measurable impact:**
Compare average iterations-to-PASS across difficulty tiers,
with and without difficulty-aware prompting.

---

### P3. Iteration Budget Optimization

**Motivation:**
The current pipeline uses a fixed `max_iterations = 10` for all problems.
Some problems reach PASS in 2 iterations; others never converge in 10.
A fixed budget wastes compute on non-convergent problems and potentially
exits too early on problems that would converge at iteration 11–12.

**Relationship to sensitivity.md:**
`docs/sensitivity.md` Analiz 5 (İterasyon Verimliliği) quantifies marginal gain
per iteration and identifies the diminishing-returns threshold.
That analysis informs the cutoff value; P3 extends it to per-problem adaptive decisions.

**Proposed extension:**
Use the logistic regression model from T2 to predict convergence probability
at each iteration. If P(eventual PASS) falls below a threshold, stop early.
If P(eventual PASS) is high but the budget is exhausted, extend selectively.

**Academic framing:**
"Adaptive test-time compute allocation for iterative code refinement."
Connects to recent work on dynamic inference budgets (Adaptive Computation,
Graves 2016; Banino et al., 2021).

---

### P4. Static Analysis Feedback Loop

**Motivation:**
Node 1 metrics (CC, MI, Halstead) are currently forwarded to Node 3 as
informational context. However, the `improved_prompt` rarely contains
specific structural refactoring instructions derived from these metrics.
A model told "CC is 14" rarely knows which function to split or how.

**Proposed extension:**
Add a rule-based translation layer between Node 1 and Node 3:

```python
# Instead of: "Cyclomatic Complexity 14 — threshold 10"
# Generate:   "Function solve() has 14 branches. Split the input
#              parsing logic (lines 3–12) into a separate helper."
```

This requires mapping radon output to actionable code-level suggestions —
achievable with AST analysis and a small rule set.

**Academic framing:**
"Grounding static analysis findings in actionable code edits" —
a form of program synthesis guided by quality metrics.

---

## Priority Summary

### Recommended for current thesis

| Item | Effort | Academic Impact | Prerequisite |
|------|--------|-----------------|--------------|
| T1. Per-model ΔTSR analysis | None — data already produced | High — reframes core claim | Cloud benchmark done |
| T2. Logistic regression gating | Medium | High — novel technical contribution | sensitivity.md Analiz 2-3 done |
| T3. Cross-model feedback transfer | Low | Medium — clean ablation opportunity | Cloud benchmark done |

### Recommended for future publications

| Item | Effort | Academic Impact | Notes |
|------|--------|-----------------|-------|
| P1. Domain-specific judge (DPO) | High | Very High — standalone paper potential | Needs 500–1000 preference pairs |
| P2. Difficulty-aware prompting | Low | Medium — strong empirical addition | Quick to implement and test |
| P3. Iteration budget optimization | Medium | High — connects to active literature | Depends on T2 |
| P4. Static analysis feedback loop | Medium | Medium — systems contribution | AST analysis required |

---

## Connection to Thesis Claims

Each future direction maps directly to a current limitation:

```
Current limitation                    Future direction
─────────────────────────────────────────────────────────────────
Fixed thresholds, no learning     →   T2: Learned metric gating
General-purpose judge             →   P1: Domain-specific judge (DPO)
Cross-model feedback discarded    →   T3: Cross-model transfer
Fixed iteration budget            →   P3: Adaptive budget
Difficulty-agnostic prompting     →   P2: Difficulty-aware prompting
Static analysis not grounded      →   P4: Actionable static feedback
Model size not isolated           →   T1: Small model framing (free from existing data)
```

When writing the thesis discussion section (paper.md Bölüm VI — Sonuç),
present each limitation alongside its corresponding future direction.
This transforms weaknesses into a structured research agenda —
methodologically stronger than listing limitations alone.

---

## Relationship to Other Documents

| Document | Connection |
|----------|-----------|
| `docs/sensitivity.md` | Statistical precursor to T2; Analiz 5 informs P3 |
| `docs/ablation.md` | V0–V3 variant design; T1 and T3 are natural ablation extensions |
| `docs/paper.md` | Bölüm VI (Sonuç) draws from this document's limitation-direction mapping |
| `benchmark/analysis.py` | T1's per-model ΔTSR uses `compare_variants()` and `print_model_table()` |
