# future-work.md — Future Work Directions

## Overview

This document outlines future research directions that extend the current thesis work.
Items are organized by scope: **thesis-scope additions** that can strengthen the
current submission, and **post-thesis directions** for follow-up research.

---

## Thesis-Scope Additions

These can be incorporated into the current work with modest additional effort.

### T1. Small Model Framing — Efficiency vs. Scale

**Core claim:**
> Systematic multi-stage feedback can meaningfully improve task success rate
> without increasing model size — challenging the assumption that better code
> generation requires larger models.

**What to add:**

Run each model (M1, M2, M3) independently as both V0 baseline and V4 full pipeline.
This isolates pipeline contribution from model capability:

```
gpt-oss-20b  V0  →  TSR = ?
gpt-oss-20b  V4  →  TSR = ?    ← pipeline delta for a small model
```

**Key visualization:**
Plot model size (B parameters) on the x-axis against pipeline contribution (ΔTSR = V4 − V0)
on the y-axis. Expected pattern: large gain for small models (pipeline matters more),
small gain for large models (ceiling effect). This graph alone is a strong standalone
contribution.

**Academic positioning:**
Connects to the test-time compute vs. model scale debate
(Snell et al., 2024; Jones, 2021). The framing shifts from
"which model is best?" to "how much can structured feedback compensate for model size?"

---

### T2. Learned Metric Gating — Logistic Regression Before Node 3

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
Compare three configurations: (a) fixed thresholds only, (b) logistic gate only,
(c) logistic gate + Node 3. Isolates the contribution of each evaluation layer.

---

### T3. Cross-Model Feedback Transfer

**Core idea:**
When M1 fails and the pipeline advances to M2, M2 currently starts from scratch.
The improved_prompts generated during M1's iterations contain specific, actionable
information about what went wrong — this information is discarded.

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
Compare M2 V4 with and without M1 feedback transfer.
If M2 reaches PASS in fewer iterations when initialized with M1's failures,
transfer is effective.

---

## Post-Thesis Directions

These require additional data, infrastructure, or scope beyond the current thesis.

### P1. Domain-Specific Judge Fine-Tuning

**Motivation:**
The current pipeline uses Qwen3.5-35B-A3B as a general-purpose judge.
A judge fine-tuned specifically for iterative code refinement could produce
more consistent scores and more actionable improved_prompts.

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

---

### P2. Difficulty-Aware Prompt Adaptation

**Motivation:**
The current improved_prompt is generated with the same template regardless
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
A fixed budget wastes compute on non-convergent problems and
potentially exits too early on problems that would converge at iteration 11–12.

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
informational context. However, the improved_prompt rarely contains
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

| Item | Effort | Academic Impact |
|------|--------|-----------------|
| T1. Small model framing + ΔTSR graph | Low | High — reframes core claim |
| T2. Logistic regression gating | Medium | High — novel technical contribution |
| T3. Cross-model feedback transfer | Low | Medium — clean ablation opportunity |

### Recommended for future publications

| Item | Effort | Academic Impact |
|------|--------|-----------------|
| P1. Domain-specific judge (DPO) | High | Very High — standalone paper potential |
| P2. Difficulty-aware prompting | Low | Medium — strong empirical addition |
| P3. Iteration budget optimization | Medium | High — connects to active literature |
| P4. Static analysis feedback loop | Medium | Medium — systems contribution |

---

## Connection to Thesis Claims

Each future direction maps directly to a limitation of the current work:

```
Current limitation                    Future direction
─────────────────────────────────────────────────────────────────
Fixed thresholds, no learning     →   T2: Learned metric gating
General-purpose judge             →   P1: Domain-specific judge (DPO)
Cross-model feedback discarded    →   T3: Cross-model transfer
Fixed iteration budget            →   P3: Adaptive budget
Difficulty-agnostic prompting     →   P2: Difficulty-aware prompting
Static analysis not grounded      →   P4: Actionable static feedback
Model size not isolated           →   T1: Small model framing
```

When writing the thesis discussion section, present each limitation
alongside its corresponding future direction. This transforms weaknesses
into a structured research agenda — methodologically stronger than
listing limitations alone.
