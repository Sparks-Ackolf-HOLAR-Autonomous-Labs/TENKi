# TENKi Technical Reference — Transfer-Ensemble Knowledge Inheritance

**Version**: 0.4.0 (April 2026)
**Parent framework**: PEGKí v0.2.8
**Canonical results bundle**: `results/tenki_1000/`

---

## The Strong Question

> **Given a fixed budget, when can cheaper knowledge sources replace or accelerate
> high-fidelity evaluation, and when are they blocked by irreducible bias?**

This document answers that question through three evidence layers:

- **Layer A — Transfer structure** (Experiments 02–4):
  donor/receiver identity, ceiling, bias floor, variance-limited vs bias-limited failure.
- **Layer B — Budgeted ranking performance** (Experiments 07–8):
  fixed-budget concentration vs diversity; classical vs hybrid multi-fidelity.
- **Layer C — Theory and interpretation** (§§ 10–14):
  MFMC foundations, egalitarian weighting, KS-balance.

Secondary / negative-control results:
- **Experiment 06**: symmetry scoring — conceptual validation, not core ranking evidence.
- **Experiment 10**: naive LF concatenation degrades prediction — rules out an easy alternative.
- **Experiments 09, 11**: operational extensions (mixed-source flip, multi-domain).
- **Experiments 12, 13**: "what to do instead" — ensemble aggregation and async MF scheduling.

The **synthesis table** (§ 20) and **decision rule** (§ 21) are the primary outputs for
practitioners. Read those first if you want actionable guidance.

## Novelty Claim

TenKi is **not** claimed here as a new multi-fidelity estimator or a replacement for
classical MFMC. The contribution is at the diagnostic level:

- it treats low-fidelity usefulness as a **directional transfer problem**, not just a
  correlation problem
- it distinguishes **variance-limited** failure from **bias-limited** failure through the
  donor/receiver flip test and the external ceiling or bias floor
- it turns that diagnosis into a **budgeted ranking decision rule** for source selection,
  branch selection, and allocation

In short, TenKi's novelty is the flip-based transfer diagnostic for deciding when a cheap
source can become a useful donor for ranking under fixed budget, and when it is permanently
blocked by irreducible bias.

---

## Table of Contents

1. [Clarifying the Symmetry Concept](#1-clarifying-the-symmetry-concept)
2. [Engine Gamuts as Sets](#2-engine-gamuts-as-sets)
3. [Set-Operation Decomposition (Venn Partition)](#3-set-operation-decomposition-venn-partition)
4. [Three Notions of "Symmetric Whole"](#4-three-notions-of-symmetric-whole)
5. [The Tiling / Covering Problem](#5-the-tiling--covering-problem)
6. [Study-Based Comparison Framework](#6-study-based-comparison-framework)
7. [The Existing 3 Studies (A, B, C)](#7-the-existing-3-studies-a-b-c)
8. [Mathematical Conditions for Symmetry](#8-mathematical-conditions-for-symmetry)
9. [Relation to Knowledge Space Theory](#9-relation-to-knowledge-space-theory)
10. [Frugal Twin Convergence (Swarm Intelligence Framing)](#10-frugal-twin-convergence-swarm-intelligence-framing)
11. [Multi-Fidelity Optimal Acquisition (MFMC)](#11-multi-fidelity-optimal-acquisition-mfmc)
12. [Egalitarian vs Heterogeneous Swarm Weighting](#12-egalitarian-vs-heterogeneous-swarm-weighting)
13. [TrueSkill2 Team Modeling Applied to the Swarm](#13-trueskill2-team-modeling-applied-to-the-swarm)
14. [On the O_h Group Approach (and Why It Is Secondary)](#14-on-the-oh-group-approach-and-why-it-is-secondary)
15. [Ensemble vs Swarm Aggregation (Experiment 12)](#15-ensemble-vs-swarm-aggregation-experiment-12)
16. [Async Multi-Fidelity Optimizer (Experiment 13)](#16-async-multi-fidelity-optimizer-experiment-13)
17. [Knowledge-Space Transfer Results from the Parent PEGKi Benchmark](#17-knowledge-space-transfer-results-from-the-parent-pegki-benchmark)
18. [TENKi-1000 Experiment Series](#18-tenki-1000-experiment-series)
19. [Open Questions](#19-open-questions)
20. [Synthesis Table](#20-synthesis-table)
21. [Decision Rule](#21-decision-rule)

---

## 1. Clarifying the Symmetry Concept

### What "Asymmetric" Means Here

An engine gamut is **asymmetric** in the set-theoretic sense: it is a *directed*, *biased* subset
of the full color space that favors one physical model over another.

| Term | Definition |
|------|-----------|
| **Asymmetric gamut** | A set G_i — RGB_CUBE that is biased toward some region of color space — ALL engine gamuts and ALL set-op studies are asymmetric in this sense |
| **Symmetric gamut** | A set S where some meaningful equivalence holds — e.g., invariant under engine-label permutation, uniform in density, or partitioned fairly. No single study achieves this. |
| **Difference-asymmetric study** | A study drawn from G_i \ G_j — biased toward what engine i uniquely produces (studies B, C) |
| **Intersection-asymmetric study** | A study drawn from G_i — G_j — biased toward the consensus region of two engines (study A). Still asymmetric relative to the full color space — the intersection of two biased gamuts is itself biased. |

> **Important**: intersection ≠ symmetric. Study A (mixbox→RYB) has mean=(189,127,121) —> far from the neutral gray (128,128,128). It is symmetric only under the narrow
> `mixbox→RYB` swap, not relative to the full color space or any high-fidelity reference.
> Empirically, it has the second-highest bias floor (0.249) among all frugal studies.

The key distinction: "symmetric" here is about **which engines a color region belongs to**, not about
geometric rotation/reflection groups acting on the RGB cube.

### What "Symmetric Whole" Means

Given K asymmetric study databases (each sampling an asymmetric region of color space), a
"symmetric whole" is a combined dataset that treats all engines, physics models, or
knowledge-space types fairly — i.e., no model is privileged over another by the distribution of
target colors.

---

## 2. Engine Gamuts as Sets

Four engines define four gamuts G_1, G_2, G_3, G_4 — RGB_CUBE:

| Engine | Symbol | Model type | Knowledge Space | Key property |
|--------|--------|------------|-----------------|-------------|
| spectral | G_H | Physics / Beer-Lambert + D65 | K_H (Hybrid) | R=[0,255], G=[132,255], B=[107,255]. Bright region only. |
| mixbox | G_E1 | Kubelka-Munk sRGB pigment | K_E (Empirical) | Paint mixing; 4-endpoint structure |
| kubelka_munk | G_E2 | KM real oil pigment (OSF data) | K_E (Empirical) | R=[31,248], G=[40,248], B=[23,232]. Broader but darker |
| coloraide_ryb | G_T | Artist RYB color-wheel theory | K_T (Theory) | Artist intuition; Itten-based |

Each gamut is an **asymmetric subset** of the full RGB cube [0,255]^3.

Key fact from §17.1 of the parent TECHNICAL_REFERENCE: **62% of uniform RGB space falls outside
the spectral engine gamut**. So even the largest gamut (spectral) is a strongly biased sample.

---

## 3. Set-Operation Decomposition (Venn Partition)

### For Two Engines (A, B)

The union G_A ∪ G_B can be partitioned into 3 disjoint pieces:

```
G_A ∪ G_B  =  (G_A ∩ G_B)              [intersection: both can reach this]
            ∪ (G_A \ G_B)              [A-only: A can reach, B cannot]
            ∪ (G_B \ G_A)              [B-only: B can reach, A cannot]
```

Symmetry assessment:
- **G_A ∩ G_B** is *symmetric* under A↔B swap (the same set regardless of labeling)
- **G_A \ G_B** is *asymmetric* — it belongs to A, not B
- **G_B \ G_A** is *asymmetric* — it belongs to B, not A
- Together: **1 symmetric + 2 asymmetric = 1 union**

The 2 asymmetric pieces are **complementary** — they are the "error bars" of the intersection,
the portions where the models disagree about whether a color is producible.

### For N Engines (General Case)

With N engines, the Venn diagram decomposes into 2^N − 1 non-empty regions.
Each region is indexed by a subset S ⊂ {1,...,N} of engines:

```
R_S = (∩{i ∈ S} G_i) \ (∪{j ∉ S} G_j)

    = colors reachable by exactly the engines in S
```

| N | Total regions | 100% symmetric region | Asymmetric regions |
|---|---------------|----------------------|-------------------|
| 2 | 3 | 1 (G_1 ∩ G_2) | 2 |
| 3 | 7 | 1 (G_1 ∩ G_2 ∩ G_3) | 6 |
| 4 | 15 | 1 (G_1 ∩ G_2 ∩ G_3 ∩ G_4) | 14 |
| N | 2^N − 1 | 1 | 2^N − 2 |

**Key result**: There is always exactly **one maximally symmetric region** (the N-way intersection)
and **2^N − 2 asymmetric regions**. The full picture requires all 2^N − 1 pieces.

### Symmetry Hierarchy

Not all asymmetric pieces are equally asymmetric. The "degree of asymmetry" can be measured
by the size of the engine subset S:

```
Symmetry level of R_S  =  |S| / N   ∈ {1/N, 2/N, ..., 1}
```

- Level 1/N: colors unique to a single engine (most asymmetric)
- Level (N-1)/N: colors shared by N-1 engines (moderately symmetric)
- Level 1: the N-way intersection (fully symmetric)

---

## 4. Three Notions of "Symmetric Whole"

### Definition 4.1 — Engine-Permutation Symmetric (Structural)

A collection of study databases {D_1, ..., D_K} is **engine-permutation symmetric** if the
combined target distribution is invariant under any permutation of engine labels.

**Condition**: Every Venn region R_S must be represented proportionally to its volume
(or the collection must include equal amounts of each asymmetric piece and its "mirror").

- For 2 engines: need |targets from G_A\G_B| = |targets from G_B\G_A|
- For 4 engines: all 15 regions represented in proportion to volume
- **Minimum K**: 2 asymmetric studies (one for each direction A→ and B→)

### Definition 4.2 — Coverage-Uniform (Statistical)

A collection is **coverage-uniform** if the density of target colors is uniform over the
union G_1 ∪ ... ∪ G_N.

Each Venn region R_S must be sampled proportionally to its volume.

**Condition**:
```
|{targets in R_S}| / |{total targets}|  =  |R_S| / |G_1 ∪ ... ∪ G_N|
```

This is a weaker condition than engine-permutation symmetry — it only requires
proportional sampling, not equal representation of engine directions.

### Definition 4.3 — Knowledge-Space Balanced (Semantic)

A collection is **KS-balanced** if the three knowledge-space types are equally represented:

```
|targets from K_H studies| ≈ |targets from K_E studies| ≈ |targets from K_T studies|
```

This is the most practically relevant criterion for the PEGKí framework because the
knowledge space type determines what transfer learning is possible (§9–1 of parent doc).

---

## 5. The Tiling / Covering Problem

### Problem Statement

**Given**: K asymmetric study databases D_1, ..., D_K, each sampling a subset of the
color gamut union.

**Find**: The minimum K such that the union D_1 ∪ ... — D_K satisfies one of the
symmetry conditions in §4.

### Minimum Covering Numbers

| Symmetry type | Minimum K for 2 engines | For 4 engines | Condition |
|---------------|------------------------|---------------|-----------|
| Engine-permutation symmetric | **3** (1 intersection + 2 differences) | **15** (all Venn regions) | Exact partition |
| Coverage-uniform (ε-approximate) | **2k** | **5–10** | Volume-proportional sampling |
| KS-balanced | **3** (one per KS type) | **3** (one per KS type) | Equal KS representation |

**Key finding**: For *exact* engine-permutation symmetry with N engines, you need K = 2^N − 1
study databases — all Venn regions. For KS-balance, only K = 3 is needed regardless of N.

### Feasibility

**Is it doable?** Yes, with the following conditions:

1. **The N-way intersection is non-empty**: If G_1 ∩ G_2 ∩ ... ∩ G_N = ∅ no study can
   represent the symmetric core. The spectral gamut being the largest (most "physics-complete")
   means its intersection with others may exist but be small.

2. **The asymmetric regions are non-trivial**: If G_A \ G_B is empty (one engine's gamut
   is a subset of another's), the corresponding asymmetric study is trivially empty.

3. **Target sampling is achievable**: All target colors must come from actual engine outputs,
   not uniform RGB (which would fall outside all gamuts 62% of the time).

---

## 6. Study-Based Comparison Framework

### Comparing Studies Against Each Other

Each study database has an associated target color distribution. To compare them:

**Metrics**:

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Centroid distance | `‖mean(D_i) − mean(D_j)‖` | How different are the "average" target colors? |
| KL-divergence | `KL(D_i ‖ D_j)` | Information distance between distributions |
| Wasserstein-1 | Earth-mover distance | Minimum effort to transform one distribution into another |
| Venn overlap fraction | `\|D_i ∩ D_j\| / \|D_i ∪ D_j\|` | How much do the target regions overlap? |
| Asymmetry index | `(V_i - V_j) / (V_i + V_j)` | Signed imbalance between two region volumes |

**Symmetry score of a collection**:
```
SymScore({D_1,...,D_K}) = 1 - StdDev(V_1, ..., V_K) / Mean(V_1, ..., V_K)

where V_i = volume fraction of target color space covered by D_i
```

Score = 1.0 — perfectly balanced (all regions equal volume).
Score = 0.0 — one study dominates completely.

### Cross-Study Policy Transfer Relevance

The asymmetry of study databases is directly connected to the asymmetric transfer principle
(§10 of parent doc): when a policy trained on an asymmetric study D_i = G_A \ G_B transfers
into the "mirror" study D_j = G_B \ G_A, the transfer delta Δ will have the opposite sign.

This means: **asymmetric gamut subsets produce asymmetric transfer — which is exactly the
phenomenon the PEGKí rating system was designed to detect.**

---

## 7. The Existing 3 Studies (A, B, C)

The three set-operation studies in the current database:

| Study | Set-op | Engines | KS type | Asymmetry |
|-------|--------|---------|---------|-----------|
| `db_study_a` | mixbox ↔ coloraide_ryb | intersection | K_H_OVERLAP | Symmetric under mixbox→RYB |
| `db_study_b` | spectral ↔ coloraide_ryb | difference | K_T | Asymmetric: spectral has, RYB doesn't |
| `db_study_c` | kubelka_munk ↔ mixbox | difference | K_E | Asymmetric: KM has, mixbox doesn't |

### Empirical Target Distributions (900 targets each, full datasets)

| Study | R range | G range | B range | Mean RGB | Character |
|-------|---------|---------|---------|----------|-----------|
| **A** (mixbox→RYB) | [22, 254] | [27, 242] | [30, 255] | (189, 127, 121) | Warm/pink — artist consensus warm tones |
| **B** (spectral→RYB) | [0, 255] | [134, 233] | [128, 221] | (162, 185, 175) | Cool/cyan-green — physics-only colors |
| **C** (KM→Mixbox) | [56, 209] | [62, 214] | [44, 207] | (164, 154, 81) | Earth tones — oil-pigment-unique warm browns |

The distributions are clearly **in different regions of color space**:
- Study A is biased toward warm pinks (high R, moderate G,B)
- Study B is biased toward cool greens/cyans (high G,B; wide R range because spectral has full R=0..255)
- Study C has very low blue (mean B=81) — unique to the KM model's real oil-pigment data

**Combined centroid** (mean of 3 means): R=172, G=155, B=126.
The symmetric center of the RGB cube is (127.5, 127.5, 127.5).
The combined mean is offset toward high R and G — a mild warm bias remains.

Note: Study B's G and B have elevated minimums (G≥34, B≥28) because these colors must be
reachable by the spectral engine (which has G≥32, B≥07 per CIE D65 physics) but NOT by
RYB theory — so the low-G,B region that spectral can't reach is excluded.

### What's Missing for Full Symmetry?

For the **2-engine case** (mixbox vs RYB), these studies provide:
- G_mixbox ∩ G_RYB (Study A)
- G_RYB \ G_mixbox (the "RYB-only" complement of Study A — not in the database)
- (Study A alone covers the symmetric core but not the full partition)

For the **3-engine case** (spectral, RYB, + one of mixbox/KM):
- 2^3 − 1 = 7 regions needed for exact symmetry
- Currently have 3 studies — **KS-balanced** (one per type) but **NOT engine-permutation symmetric**
- Missing: complement differences (RYB \ spectral, mixbox \ KM, etc.)

For the **4-engine case**:
- 2^4 − 1 = 15 regions needed
- Currently have 7 databases total (4 single-engine + 3 set-op)
- Single-engine databases correspond to the "all-engine intersection" being non-empty only
  for the spectral-gamut-derived targets
- Missing: 8 pairwise intersections and differences beyond what's currently computed

### Current Symmetry Status (3 Studies)

All three studies are asymmetric gamut subsets. Their symmetry properties differ only in
*which kind* of asymmetry they exhibit:

| Study | Asymmetry type | Mean RGB | Bias floor vs spectral |
|-------|---------------|----------|----------------------|
| A (mixbox→RYB) | Intersection-asymmetric: skewed toward warm artist-consensus colors | (189,127,121) | 0.249 |
| B (spectral→RYB) | Difference-asymmetric: skewed toward physics-exclusive cool colors | (162,185,175) | 0.105 |
| C (KM→Mixbox) | Difference-asymmetric: skewed toward oil-pigment earth tones | (164,154,81) | 0.318 |

By Definition 4.3 (KS-balanced):
```
K_H_OVERLAP (Study A): 1 study
K_T          (Study B): 1 study
K_E          (Study C): 1 study
⟹ Score = 1.0 — perfectly KS-balanced!
```

By Definition 4.1 (Engine-permutation symmetric):
- Missing: G_RYB \ G_mixbox, G_RYB \ G_spectral, G_mixbox \ G_spectral, etc.
- Score = 0.25 — not engine-permutation symmetric

**Frugal twin finding**: The intersection-asymmetric study (A) has a *higher* bias floor than
the difference-asymmetric study (B), despite being "more symmetric" under the mixbox→RYB swap.
This is because the warm consensus region of two artist models is precisely where high-fidelity
spectral physics contributes the *least* distinguishing information.

---

## 8. Mathematical Conditions for Symmetry

### Theorem 8.1 (Minimum Covering for KS-Balance)

A collection of K set-operation studies is KS-balanced if and only if:
```
K ≥ 3  AND  {K_H_OVERLAP study, K_T study, K_E study} ⊆ collection
```

This is achievable with K = 3 for any N ≥ 2 engines.

**The 3 existing studies satisfy this condition.**

### Theorem 8.2 (Minimum Covering for Engine-Permutation Symmetry)

A collection of set-operation studies achieves engine-permutation symmetry for N engines if and
only if every Venn region R_S (for all non-empty S ⊂ {1,...,N}) is represented in the collection.

This requires K = 2^N − 1 studies, of which 2^N − 2 are asymmetric.

For N = 4 engines: K = 15.

**The current 7-database setup covers 7 of 15 Venn regions (approximately).**

### Corollary 8.3 (Asymmetric Complement)

Every asymmetric study D_A = G_A \ G_B has a unique "mirror" D_B = G_B \ G_A.
Together, D_A ∪ D_B = (G_A ∪ G_B) is engine-permutation symmetric for this pair.

To make the full collection symmetric for all pairs simultaneously requires all 2^N − 1 pieces.

### Sufficient Condition (Approximate Symmetry)

For practical purposes, ε-approximate engine-permutation symmetry holds when:

```
∀ engine pair (i, j):  | Vol(G_i \ G_j) - Vol(G_j \ G_i) | / Vol((G_i \ G_j) ∪ (G_j \ G_i))  <  ε
```

If this holds for ε ≤ 0.05 (5% imbalance), the studies are approximately symmetric.

Estimated volumes from gamut sampling (CLAUDE.md):
- Spectral: G=[132,255], B=[107,255] — bright region only (~38% of RGB cube AABB)
- KM: R=[31,248], G=[40,248], B=[23,232] — broader, darker region

---

## 9. Relation to Knowledge Space Theory

### Knowledge Spaces as Venn Regions

The three KS types map naturally to the Venn decomposition:

| Knowledge Space | Venn interpretation | Symmetry level |
|-----------------|---------------------|----------------|
| K_H (hybrid spectral) | Physics-only region: G_spectral that others can't reach | Asymmetric (level 1/4 for 4 engines) |
| K_E (empirical KM/mixbox) | Empirical-only region: G_KM or G_mixbox unique parts | Asymmetric |
| K_T (RYB theory) | Theory-only region: G_RYB unique parts | Asymmetric |
| K_H_OVERLAP | Intersection of multiple engines | Symmetric (level 2/4) |

### The Nash Equilibrium Result

From §17.7.4 of the parent TECHNICAL_REFERENCE: the parent PEGKi Nash-routing analysis concentrates
entirely on set-operation databases (54% kubelka_munk_difference_mixbox, 46% mixbox_intersection_
coloraide_ryb), not single-engine databases.

**Interpretation in gamut-symmetry terms**:
- The Nash equilibrium prefers the intersection (symmetric) and one specific difference (asymmetric)
- It avoids the most asymmetric databases (pure single-engine)
- This suggests the "optimal symmetric mixture" is not the full engine-permutation symmetric
  collection, but rather a 2-study mixture weighted toward the intersection

**Connection to tiling**: The Nash result tells us that for *transfer learning purposes*, K = 2
studies (one symmetric + one asymmetric) may be sufficient — not K = 15.

---

## 10. Frugal Twin Convergence (Swarm Intelligence Framing)

### Setup

- **High-fidelity reference**: spectral engine, 1004 experiments per policy
- **Frugal twins**: the 6 cheaper studies (mixbox, RYB, study_a, study_b, study_c, KM)
- **"Robot"**: one policy experiment (50 trials against a fixed target color)
- **Metric**: Kendall tau between frugal-N-robot ranking and full spectral ranking

### Convergence Results

The table below reports the current TENKi-1000 convergence artifact. The `N=1` values in the synthesis table (Section 20) come from the regenerated flip-test run; small differences from this Experiment 07 table are bootstrap/seed effects. Use Section 20 for final manuscript numbers.

| Frugal source | N=1 | N=5 | N=10 | N=20 | N=100 | Residual gap at N=100 |
|---------------|-----|-----|------|------|-------|-----------------------|
| spectral (self) | 0.626 | 0.804 | 0.853 | 0.890 | 0.951 | 0.049 |
| study_b (spectral-RYB) | 0.612 | 0.779 | **0.802** | 0.833 | 0.844 | 0.156 |
| mixbox | 0.566 | 0.689 | 0.773 | 0.833 | **0.901** | 0.099 |
| ryb | 0.536 | 0.589 | 0.671 | 0.755 | 0.871 | 0.129 |
| km | 0.310 | 0.385 | 0.464 | 0.532 | 0.703 | 0.297 |
| study_a (mixbox-RYB) | 0.194 | 0.366 | 0.457 | 0.516 | 0.703 | 0.297 |
| study_c (KM-mixbox) | 0.130 | 0.263 | 0.338 | 0.439 | 0.644 | 0.356 |

Full-data bias floors are reported in the synthesis table (Section 20); this table stops at N=100.

### Key Findings

1. **Intersection-asymmetric does not mean easier**: study_a remains much worse than study_b at low budget despite having the same full-data ceiling in the flip test.

2. **Best unpaired low-budget donor = study_b**: Drawing from the spectral-exclusive region gives the best low-budget approximation of the spectral reference among unpaired sources. Mixbox has the stronger N=100 value and is the best paired practical source.

3. **Quantity beats diversity when one source is good**: with a fixed total budget, spreading measurements across weak sources dilutes the strongest source. Diversity should only be used when there is independent evidence of complementary spatial coverage.

### Minimum Robots to Match High-Fidelity

| Target tau | Using spectral | Using study_b | Using all-source diverse |
|-----------|---------------|--------------|--------------------------|
| tau >= 0.80 | ~5 robots | ~10 robots | ~100 total |
| tau >= 0.85 | ~10 robots | impossible by N=100 | ~100 total |
| tau >= 0.90 | ~30 robots | impossible by N=100 | impossible by N=100 |
| tau >= 0.97 | impossible by N=100 | impossible (bias) | impossible (bias) |

**The fundamental limit**: frugal physics cannot fully substitute for high-fidelity physics,
regardless of how many frugal measurements you take. The irreducible bias is the cost of
using a cheaper model.

## 11. Multi-Fidelity Optimal Acquisition (MFMC)

### Engineering Question

> If cost(HF) >> cost(LF), can we use a ratio like 3:100 (3 HF : 100 LF) instead of
> a 50:50 acquisition to achieve similar **ranking** quality at lower total cost?

### What Is Actually Being Borrowed from MFMC

Classical multi-fidelity Monte Carlo (MFMC) solves a **scalar estimation** problem: given a
high-fidelity quantity of interest and a correlated low-fidelity surrogate, what LF:HF sample
ratio minimizes estimator variance under a fixed cost budget?

For Pearson correlation `rho` between LF and HF scalar outputs and cost ratio
`r = cost_HF / cost_LF`, the standard MFMC allocation rule is:

```
n_LF / n_HF = sqrt(r) * |rho| / sqrt(1 - rho^2)
```

This ratio is well-defined and useful in our setting as an **allocation prior**:

- high `rho` and large `r` imply "spend more on LF scouting"
- low `rho` implies "LF is not informative enough to justify heavy spending"

### Scope Note: This Is Not a Standard MFMC Theorem for Ranking

Our experiment does **not** prove an MFMC variance-reduction theorem for Kendall tau.
Instead, it does two separate things:

1. It uses the classical MFMC ratio above as a **budget heuristic**.
2. It then evaluates an **empirical ranking combiner** that mixes HF and LF policy score
   information and measures the resulting Kendall tau vs the full HF ranking.

This distinction matters:

- classical MFMC is about unbiased scalar estimation and variance
- TenKi Experiment 08 is about **ranking quality** under a fixed budget
- therefore the closed-form MFMC ratio should be read as a principled starting point,
  not as a theorem guaranteeing the best Kendall-tau outcome

### Current Experiment 08 Estimator

The current implementation constructs a rho-guided hybrid ranking score:

```
z_rank = z_HF + rho * z_LF
```

where:

- `z_HF` is the z-scored vector of per-policy HF sample means
- `z_LF` is the z-scored vector of per-policy LF sample means
- lower score = better policy, so a good LF proxy should reinforce the HF ordering

Important limitations:

- this is a **heuristic rank combiner**, not the textbook MFMC control-variate estimator
- HF and LF samples are drawn from separate study databases rather than paired evaluations
- the objective is Kendall tau, not minimum-variance estimation of a scalar quantity

The experiment is therefore best understood as:

> "MFMC-inspired allocation + empirical ranking validation"

not:

> "direct application of MFMC theory to ranking"

### Pearson Correlation and Allocation Prior (Original 7-Study Set)

| LF source | rho | Interpretation |
|-----------|-----|----------------|
| study_b (spectral-RYB) | **0.996** | Near-perfect score-space alignment with HF |
| study_a (mixbox intersection RYB) | 0.994 | Very high, despite poor ranking transfer |
| mixbox | 0.989 | High correlation |
| ryb | 0.985 | High correlation |
| km | 0.973 | Good correlation |
| study_c (KM-mixbox) | 0.843 | Most divergent LF source |

The key caution is that `rho` alone is not enough:

> study_a has `rho = 0.994` yet remains a poor frugal proxy in ranking terms.
> Good score-space correlation does not guarantee low ranking bias.

### MFMC Ratio Table (Allocation Prior Only)

| LF source | r=5 | r=10 | r=20 | r=50 | r=100 | r=200 | r=500 | r=1000 |
|-----------|-----|------|------|------|-------|-------|-------|--------|
| study_b | 24 | 34 | 48 | 76 | 108 | 152 | 241 | 340 |
| study_a | 20 | 28 | 40 | 63 | 90 | 127 | 200 | 283 |
| mixbox | 15 | 21 | 30 | 47 | 66 | 93 | 148 | 209 |
| ryb | 13 | 18 | 25 | 40 | 56 | 80 | 126 | 178 |
| km | 9 | 13 | 19 | 30 | 42 | 60 | 94 | 134 |
| study_c | 4 | 5 | 7 | 11 | 16 | 22 | 35 | 50 |

At `r = 100`, the allocation prior suggests:

- `study_b`: ~108 LF per HF
- `study_a`: ~90 LF per HF
- `study_c`: ~16 LF per HF

This is informative, but it is **not enough to choose a source** on its own, because it does not
see the ranking bias floor.

### How to Read the Empirical Results

The empirical results of Experiment 08 should be interpreted with three rules:

1. **Use MFMC for the ratio, not for proof of ranking optimality**.
   The closed-form ratio tells us how aggressive LF spending can be when score-space
   correlation is high.

2. **Validate with Kendall tau, not rho alone**.
   A source can have excellent `rho` and still damage the HF ranking if it carries a
   systematic bias floor.

3. **Treat the ranking combiner as heuristic**.
   The observed HF/LF crossover budget is an empirical property of this benchmark and this
   combiner, not a theorem inherited from Peherstorfer et al.

### Two Branches: Classical vs Hybrid (v0.3.3+)

Starting from v0.3.3, Experiment 08 runs two clearly separated branches:

**Classical branch** (`--mode classical`)
- Estimand: per-policy HF mean score `mu_p = E_target[best_cd(HF, target, policy=p)]`
- Estimator: control-variate (Peherstorfer et al. 2016)
  `mu_hat_p = mean(Y_hf, n_hf paired) - alpha_p * (mean(Y_lf, n_hf paired) - mean(Y_lf, all))`
- `alpha_p = Cov(Y_hf, Y_lf) / Var(Y_lf)` estimated from all available paired observations (oracle alpha)
- Valid only for LF sources that share targets by experiment index with HF (mixbox, km, ryb in TENKi-1000)
- Labelled "classical" for these sources; "paired_cv" for other paired databases

**Hybrid branch** (`--mode hybrid`)
- Estimand: ranking quality (Kendall tau vs full HF ranking) — NOT a scalar mean
- Estimator: z-score fusion (heuristic): `z_cv = z_hf + rho * z_lf`
- Uses independent HF and LF samples (no pairing required)
- Valid for any LF source

**Budget model (identical for both)**: `B = n_HF + n_LF / r`
**Allocation prior (same for both)**: `n_HF = floor(B / (1 + ratio/r))`, `n_LF = floor((B - n_HF) * r)`

### Ranking vs Estimation Tradeoff (TENKi-1000, mixbox LF, r=100, N_bootstrap=500)

Hybrid wins on Kendall tau at every budget level:

| B | n_HF | n_LF | classical tau | hybrid tau | delta | HF-only tau |
|---|------|------|---------------|------------|-------|-------------|
| 2 | 1 | 100 | 0.676 | **0.800** | −0.124 | 0.721 |
| 5 | 2 | 300 | 0.757 | **0.850** | −0.093 | 0.798 |
| 10 | 5 | 500 | 0.821 | **0.904** | −0.083 | 0.839 |
| 20 | 11 | 900 | 0.870 | **0.942** | −0.072 | 0.881 |
| 50 | 29 | 2100 | 0.917 | **0.974** | −0.057 | 0.923 |
| 100 | 58 | 4200 | 0.939 | **0.987** | −0.048 | 0.947 |

Classical also underperforms plain HF-only at all budget levels (e.g., B=10: 0.821 vs 0.839).

**Why classical loses on tau despite correct estimation**:
The oracle alpha values for the best-performing policies are very small
(bayesian_ei: α=0.017, bayesian_ucb: α=0.058, simulated_annealing: α=−0.008).
A small alpha means the LF correction term `alpha * (mean_lf_paired — mean_lf_all)` is
negligible — the classical estimator reduces to plain HF-only averaging for these policies.
The LF variance reduction that MFMC theory predicts only materialises when α is large.

Meanwhile, the hybrid z-score fusion sidesteps raw scale entirely. Z-scoring normalises
each study separately, then uses rho=0.99 to heavily weight the LF ordering signal.
Because the LF ordering (with 100–200 observations) is much more stable than the HF
ordering (with 1 —8 observations), the hybrid correctly identifies rank order far more
reliably than either branch that primarily trusts raw HF mean estimates.

**Classical wins on scalar estimation quality** (MSE of mu_hat vs true mu_HF):
The classical estimator is unbiased; the hybrid z_cv is not a valid estimator of mu_HF.
Policies like neural_network (alpha=0.75) show genuine variance reduction under classical;
the MSE plots (multifidelity_mse.png) confirm this.

**Decision**:
> Hybrid wins on ranking (tau). Classical wins on scalar estimation faithfulness.
> This is the ranking-vs-estimation tradeoff.
> For the paper's primary metric (ranking quality), report hybrid.
> For any claim about estimating absolute HF performance levels, use classical.

### Legacy Note

Earlier v0.3 text reported Experiment 08 as a direct MFMC result.
That framing was too strong. The current separation (classical/hybrid branches) makes the
distinction explicit. Section 11 documents theory and scope; Section 18 documents empirical results.

### Practical Takeaway

MFMC remains useful here, but in a narrower role than originally stated:

- it provides a principled **LF:HF allocation prior** (same ratio used in both branches)
- the **classical branch** is the correct tool when you need per-policy mean estimates
- the **hybrid branch** is the correct tool when you only need rankings
- TenKi's bias-floor and convergence diagnostics remain required to vet any LF source
- rho determines whether LF helps at all; alpha determines how much classical benefits

---

## 12. Egalitarian vs Heterogeneous Swarm Weighting

### Current Assumption: Homogeneous / Egalitarian

All experiments in this sub-project treat the swarm (collection of robot measurements) as
**egalitarian** — each robot contributes equally, regardless of individual quality.

This manifests at three distinct levels:

| Level | Current treatment | Egalitarian assumption |
|-------|------------------|----------------------|
| **Within a study (robot level)** | `combine_scores()` takes a flat mean over all N experiments | All robots drawn from the same study are i.i.d. draws from the same distribution |
| **Across studies in Q3 (diversity test)** | Each study receives exactly `n_per_study=10` robots | All studies contribute equally, regardless of known rho or bias floor |
| **HF vs LF in Experiment 08** | A global `rho` enters a heuristic rank combiner `z_rank = z_HF + rho * z_LF` | Within a fidelity level, all samples are equal and a single source-level rho is used |

### Where the Assumption Breaks Down

**Within-study heterogeneity**: A single study's experiments are not perfectly i.i.d. in practice.
Different target colors probe different difficulty regions of the policy space. An experiment that
happened to sample a hard target (requiring more iterations before convergence) carries different
information than one that sampled an easy target. Flat averaging discards this signal.

**Cross-study egalitarian allocation (Q3 collapse)**: Experiment 07's swarm diversity test
allocates equal `n_per_study` regardless of each study's quality. This is why adding study_c
(rho=0.843, bias floor=0.318) as an equal partner to study_b (rho=0.996) drags the combined
tau *down*:

```
N=10, k=6 (all studies, n=1 each):  tau = 0.556   <- study_c gets equal vote
N=10, k=1 (study_b alone):          tau = 0.829   <- no dilution
```

A quality-aware allocation — giving study_b 4x more robots than study_c, proportional to rho — would prevent the collapse. The egalitarian assumption treats all measurement sources as peers
when they are not.

**Experiment 08: study-level non-egalitarianism, robot-level egalitarianism**: the current
ranking combiner introduces the first non-egalitarian element by scaling the LF contribution
with a global source-level `rho`. But within a single LF source all samples remain equally
weighted. A per-robot adaptive weight (computed e.g. from individual experiment variance or
information gain) is not implemented.

### Proposed Heterogeneous Alternatives

**Quality-aware diversity allocation (Q3 replacement)**:
```
n_i  =  N_total * softmax(rho_i / T)    [temperature T controls concentration]
     or
n_i  =  N_total * rho_i^2 / sum(rho_j^2)   [proportional to explained variance]
```
This allocates more budget to high-rho sources automatically, rather than requiring the analyst
to know which source is best in advance.

**Per-experiment information weighting (within-study)**:
```
w_i = 1 / Var(score_i)     [inverse-variance weighting]
combined_score = sum(w_i * score_i) / sum(w_i)
```
Experiments with high variance (difficult targets, noisy outcomes) receive lower weight.
This is the natural Bayesian pooling rule for Gaussian likelihoods.

**Per-policy adaptive rho (between-fidelity)**:
```
rho_p = corr(HF_scores_on_policy_p, LF_scores_on_policy_p)
z_rank(p) = z_HF(p) + rho_p * z_LF(p)
```
The current Experiment 08 uses a global rho across all policies. Some policies may transfer
better from LF to HF than others; a per-policy weight captures this.

### Practical Implication

The egalitarian assumption is **conservative**: it underestimates the achievable tau at a given
cost because it fails to concentrate measurement budget where information is richest. The actual
tau curves from experiments 07 and 08 therefore represent a lower bound on what a heterogeneous
allocation could achieve with the same total cost.

Conversely, the egalitarian assumption is **robust**: it does not require knowing which robots
or which studies are more informative in advance. Quality-aware allocation requires pilot
data to estimate rho or variance before committing budget — adding a meta-level cost.

### Connection to Swarm Intelligence Literature

In heterogeneous swarm intelligence, agents can have different sensors, speeds, or accuracy
levels. Standard results (e.g., from particle swarm optimization) show:

- **Homogeneous swarms**: easier to analyze, known convergence guarantees
- **Heterogeneous swarms**: faster convergence in practice, but require careful role assignment

The quality-aware allocation above is analogous to assigning "scout" vs "exploiter" roles to
robots based on their proven information value — a heterogeneous swarm strategy.

---

## 13. TrueSkill2 Team Modeling Applied to the Swarm

### Mapping: Swarm Components to TrueSkill2 Constructs

TrueSkill2 (Minka et al. 2018) was designed for multi-team, multi-player competitive games with
partial play, variable team sizes, and player-level performance noise. Its factor-graph inference
naturally resolves the egalitarian weighting problem as a principled Bayesian framework.

| Swarm concept | TrueSkill2 equivalent |
|--------------|----------------------|
| Robot (one policy experiment) | Player with skill N(mu_i, sigma_i^2) |
| Study (collection of robots) | Team |
| Combined ranking quality (Kendall tau) | Match outcome |
| High-fidelity spectral reference | The "ground truth" tournament outcome |
| Bias floor of a study | Systematic performance gap (not reducible by team size) |
| Pearson rho (LF vs HF) | Inter-team skill transferability / mapping efficiency |
| n_per_study (egalitarian allocation) | Equal partial play fractions w_i = 1/n |
| Quality-aware allocation (rho-proportional n_i) | Heterogeneous partial play fractions |

### TrueSkill2 Team Performance Model

For a team of n players with individual skills mu_i and partial play fractions w_i:

```
Team performance:
  P_team ~ N(mu_team, sigma_team^2)

  mu_team    = sum_i  w_i * mu_i
  sigma_team^2 = sum_i  w_i^2 * sigma_i^2  +  beta^2 * sum_i w_i^2

where beta^2 = per-player performance noise (model uncertainty)
```

**Egalitarian case** (current experiments 07, Q3):
```
w_i = 1/n  for all i in the team
mu_team    = mean(mu_i)               [simple average -- study_c pulls this down]
sigma_team^2 = (1/n^2) * sum sigma_i^2  +  beta^2 / n
```
Team performance variance decreases as 1/n — but only if beta^2 is the same for all robots.
If study_c robots have high beta^2 (high performance noise from physics mismatch), adding
more of them to an egalitarian swarm adds variance, not information.

**Heterogeneous case** (quality-aware):
```
w_i = rho_i^2 / sum_j rho_j^2        [proportional to explained variance]

mu_team    = weighted mean            [study_b dominates because rho_b >> rho_c]
sigma_team^2 = smaller                [concentrated weight on low-noise robots]
```

### Why Egalitarian Q3 Failed: A TrueSkill2 Diagnosis

In experiment 07's Q3 diversity test (k=6, n=1 per study):

```
Egalitarian (w_i = 1/6 for all studies):
  mu_team = (0.649 + 0.441 + 0.202 + 0.139 + 0.491 + 0.441) / 6 = 0.394   [tau ≥ 0.556]

Quality-aware (w_i proportional to rho_i^2):
  rho^2 weights: study_b=0.992, ryb=0.970, mixbox=0.978, ryb=0.970, study_a=0.988, study_c=0.710
  Normalized: study_b gets ~17% more weight than study_c
  mu_team ≈ 0.52   [estimated tau improvement ~0.60-0.65]
```

The collapse from tau=0.829 (study_b alone) to tau=0.556 (all 6 egalitarian) is exactly what
TrueSkill2 predicts when a team mixes high-skill (study_b, mu=0.649) and low-skill (study_c,
mu=0.139) players with equal partial play fractions. The team mean is dragged toward the weak
players' level.

### Per-Study beta^2 from Empirical Variance

TrueSkill2 estimates beta^2 from match outcome variance. Here, beta^2 per study can be
estimated from the empirical tau variance at fixed N:

| Study | tau_mean (N=10) | tau_std (N=10) | Implied beta^2 |
|-------|-----------------|----------------|----------------|
| spectral | 0.885 | 0.070 | 0.005 (reference) |
| study_b | 0.829 | 0.082 | 0.007 |
| mixbox | 0.690 | 0.110 | 0.012 |
| ryb | 0.600 | 0.129 | 0.017 |
| study_a | 0.425 | 0.159 | 0.025 |
| study_c | 0.338 | 0.168 | 0.028 |

study_c has 4x the performance noise (beta^2) of study_b. Under TrueSkill2 team aggregation,
adding a study_c robot to the swarm contributes 4x more uncertainty per vote — the egalitarian
assumption that they are equal is badly violated.

### Partial Play Fractions as the Allocation Variable

TrueSkill2's partial play fraction w_i is the continuous generalization of "how many robots
to assign to this study." The optimal w_i minimizes team performance uncertainty:

```
Minimize sigma_team^2 = sum_i w_i^2 * (sigma_i^2 + beta_i^2)
Subject to: sum_i w_i = 1,  w_i >= 0
```

This is a convex quadratic program with closed-form solution:

```
w_i*  =  (1 / (sigma_i^2 + beta_i^2))  /  sum_j (1 / (sigma_j^2 + beta_j^2))
       =  inverse-variance weighting
```

This is identical to the per-experiment information weighting proposed in Section 12 — derived
here from TrueSkill2's first principles rather than heuristically.

### Engine / Study as a "Team" in Multi-Team TrueSkill2

The Q3 diversity test (which study combination best reproduces HF ranking) maps naturally to
a **multi-team TrueSkill2 match**:

- K teams (studies), each team = n_per_study robots
- "Match outcome" = Kendall tau correlation with spectral reference
- TrueSkill2 updates team beliefs based on observed tau

This formulation allows the framework to **learn study quality from tournament outcomes**
rather than assuming it a priori. After enough Q3 trials:
- study_b's team skill estimate converges to high mu (good frugal proxy)
- study_c's team skill estimate converges to low mu (poor proxy)
- The optimal team composition is implied by the inferred team skills

This is the principled Bayesian alternative to the ad-hoc egalitarian allocation.

### Related Methods and How This Framework Differs

The swarm / multi-fidelity framework sits at the intersection of several established fields.
The table below maps the closest confusable methods, organized by severity of confusion.

#### Very close — same motivation, different mechanism

| Method | Similarity | Key difference |
|--------|-----------|----------------|
| **Surrogate modeling / GP emulation (kriging)** | Cheap emulator approximates expensive simulator; same cost-ratio framing | Surrogates *predict* HF outputs at unobserved inputs by interpolating in input space. Our framework aggregates repeated experiments via correlation weighting — no model is fitted to LF data. |
| **Multi-fidelity Bayesian optimization (MFBO)** | Uses cheap evaluations to guide expensive ones; same cost-ratio motivation | MFBO *optimizes* (finds the best policy) using GP surrogates across fidelities. Our framework *ranks* all policies at a fixed budget. MFBO selects which inputs to evaluate; MFMC decides how many per fidelity. |
| **Control variates (variance reduction)** | Classical MFMC provides the LF:HF allocation rule we borrow, and it motivates using correlation to scale LF influence | Control variates minimize variance of a scalar estimator. Our current Experiment 08 instead evaluates a rho-guided **ranking heuristic** and measures Kendall tau empirically. |

#### Moderate confusion — same structure, different goal

| Method | Similarity | Key difference |
|--------|-----------|----------------|
| **Knowledge distillation (teacher-student)** | HF = teacher, LF = student; same hierarchy | Distillation *trains* the student to mimic teacher outputs — the LF system is improved. Our LF physics is fixed. The bias floor is permanent: spectral CIE optics cannot be distilled into RYB color theory. |
| **Domain adaptation** | Source domain (LF) to target domain (HF); rho used as transfer weight | Domain adaptation tries to *align* feature distributions. Our framework accepts the residual bias floor and uses rho as a fixed scalar — no distribution alignment occurs. |
| **Active learning** | Both are about efficient data acquisition | Active learning selects *which inputs* to query at a single fidelity level. MFMC decides *how many experiments per source* across a cost hierarchy. Active learning has no cost ratio. |

#### Loose confusion — same name, different meaning

| Method | Similarity | Key difference |
|--------|-----------|----------------|
| **Mixture of experts (MoE)** | K studies each covering a different color-space region looks like K specialized experts | MoE routes individual *inputs* to the right expert via a gating network. Our studies each evaluate all policies on their own target distribution — no routing, no gate, no per-input specialization. |
| **Federated learning** | Multiple studies with separate databases looks like federated clients | Federated learning aggregates peers at the same fidelity for privacy. Our framework exploits a cost hierarchy. Federated clients share the same task and cost; our studies have different physics and different acquisition costs. |
| **Multi-armed bandits** | Both allocate budget across K sources | Bandits optimize cumulative reward over sequential pulls. Our framework optimizes ranking quality at a fixed total budget in one shot. Bandits have no cost hierarchy between arms. |

**The unique combination** that distinguishes this framework: an explicit fidelity/cost hierarchy
(MFMC), an irreducible physics-gap bias floor, competitive reputation tracking across match
outcomes (TrueSkill2/Blade-Chest), and a ranking target (Kendall tau) rather than a prediction
or optimization target. No single established method has all four simultaneously.

### Comparison to Ensemble Methods

The TrueSkill2 swarm framework is often confused with ensemble learning. The distinctions are structural:

| Dimension | Ensemble models | TrueSkill2 swarm (this framework) |
|-----------|----------------|----------------------------------|
| **What is combined** | Model outputs (predictions) on the same task | Raw measurement samples from different physical regimes, before any model |
| **Bias floor** | No hard floor — adding diverse members keeps reducing bias | Irreducible: physics gap between LF and HF systems cannot be closed by N |
| **Why members differ** | Engineered diversity (random subsets, seeds, algorithms) | Intrinsic diversity — genuinely different physical systems |
| **Member cost** | All peers, same cost | Explicit cost hierarchy: cost(HF) >> cost(LF); allocation IS the problem |
| **Reputation tracking** | Weights fixed or gradient-updated; no game-theoretic reputation | TS2 tracks each study's evolving quality estimate from match outcomes |
| **Intransitivity** | Not modeled | Blade-Chest detects intransitive patterns across target difficulty regimes |
| **Allocation question** | Not applicable (equal cost) | Central question: n_LF/n_HF = f(rho, r) via MFMC |

**Where they converge**: the inverse-variance weighting w_i* = 1/Var(score_i) derived from the
TrueSkill2 team model (§13) is mathematically equivalent to **Bayesian model averaging under
Gaussian likelihoods** — the one case where the two frameworks agree. Even here, BMA assumes all
models explain the same target; our framework's "target" (spectral HF ranking) is only accessible
through the expensive system, and cheap systems approximate it with physics-limited fidelity.

The egalitarian swarm (current experiments) is equivalent to **bagging** (equal-weight average).
The current rho-guided rank combiner is closest to **weighted voting**.
Neither has a natural ensemble analogue for the cost ratio question.

### Connection to the Existing PEGKi Rating System

The PEGKi framework already runs TrueSkill2 on (agent, policy, prior) triples. The gamut
symmetry extension proposes running the same system on (study, n_robots) combinations:

| PEGKi dimension | Gamut-symmetry equivalent |
|-----------------|--------------------------|
| Agent (player) | Study (measurement source) |
| Policy (champion) | Robot allocation strategy |
| Prior (equipment) | Known rho / bias floor information |
| Match outcome | Kendall tau vs HF reference |
| TrueSkill rating | Study quality estimate (transferability) |

The Blade-Chest rating component is also applicable: if different (study, N) combinations show
intransitive patterns (A beats B, B beats C, C beats A in terms of tau at different target
difficulties), BC can detect and model this — something a flat Kendall tau cannot.

---

## 14. Donor/Receiver Flip Framework (Experiments 01–4, v0.3)

### Background: Why Experiments 01–4 Were Rewritten

The original v0.1 experiments 01–4 applied the **octahedral group O_h** (48 elements:
6 channel permutations ? 8 channel negations) to measure geometric symmetry of individual
gamuts under RGB channel swaps.  This is mathematically valid but answers a different question:

| Framework | Question | "Symmetric whole" |
|-----------|----------|-------------------|
| **O_h group (v0.1)** | How many copies of a gamut tile the RGB cube under channel permutations? | Full [0,255]^3 under crystallographic symmetry |
| **Set-theoretic (v0.3)** | How many Venn regions of engine gamuts tile the union fairly? | Engine-permutation symmetric Venn partition |

The v0.3 experiments replace 01–4 with the correct foundation.
The `symmetry_group.py` and `coverage_checker.py` modules are retained for reference.

---

### Experiment 01 — Venn Region Geometry

For every pair of engines (A, B), the gamut union decomposes into three regions:

```
intersection   G_A ∩ G_B   fraction of RGB cube reachable by both
A-only         G_A \ G_B   fraction reachable by A but not B
B-only         G_B \ G_A   fraction reachable by B but not A
```

**Asymmetry index** = (|A-only| — |B-only|) / (|A-only| + |B-only|)

- Positive: A contributes more exclusive colors (A has a larger unique domain)
- Negative: B does
- Zero: symmetric pair

This is the geometric ground truth for why the set-op study databases (study_a = mixbox — RYB,
study_b = spectral — RYB, etc.) probe different regions of color space.

---

### Experiment 02 — Directed Pairwise Transfer Matrix

Defines the **directed** tau matrix:

```
tau_ij(N) = Kendall tau between
             - policies ranked by N sub-sampled experiments from source i
             - policies ranked by ALL experiments from reference j
```

**Key property**: `tau_ij(N) ≠ tau_ji(N)` at small N — this is the measurable
manifestation of directional asymmetry.

At N=full both converge to the same symmetric ceiling.
At N=1 the matrix reveals which sources are *donors* (high information per experiment)
vs *receivers* (need many experiments to produce a stable ranking).

**Donor score at N=1**: mean row value of the asymmetry matrix `A[i,j] = tau_ij(1) − tau_ji(1)`.

#### Empirical Results (7 studies, 9 policies, 104 experiments each)

**Transfer matrix at N=1** (rows = source, columns = reference):

|          | spectral | mixbox |    km |   ryb | study_a | study_b | study_c | donor_score |
|----------|----------|--------|-------|-------|---------|---------|---------|-------------|
| spectral |    1.000 |  0.616 | 0.579 | 0.643 |   0.654 |   0.649 |   0.651 |       0.632 |
| mixbox   |    0.507 |  1.000 | 0.507 | 0.546 |   0.514 |   0.502 |   0.472 |       0.508 |
| km       |    0.289 |  0.295 | 1.000 | 0.303 |   0.275 |   0.267 |   0.284 |       0.285 |
| ryb      |    0.463 |  0.477 | 0.435 | 1.000 |   0.448 |   0.448 |   0.415 |       0.448 |
| study_a  |    0.202 |  0.164 | 0.177 | 0.179 |   1.000 |   0.193 |   0.190 |       0.184 |
| study_b  |    0.643 |  0.572 | 0.537 | 0.588 |   0.649 |   1.000 |   0.640 |       0.605 |
| study_c  |    0.141 |  0.104 | 0.121 | 0.108 |   0.157 |   0.165 |   1.000 |       0.133 |

**Net donor scores at N=1** (mean asymmetry `A[i,j] = tau_ij(1) − tau_ji(1)`):

| Source   | Net score | Role     |
|----------|-----------|----------|
| spectral | +0.2583   | DONOR    |
| study_b  | +0.2337   | DONOR    |
| mixbox   | +0.1366   | DONOR    |
| ryb      | +0.0533   | DONOR    |
| km       | −0.1072   | RECEIVER |
| study_a  | −0.2654   | RECEIVER |
| study_c  | −0.3093   | RECEIVER |

**At N=full** (all experiments), tau values converge to 0.61−0.94: spectral and study_b remain the
best aligned; study_c and study_a have the lowest ceilings (0.61−0.72).

**Key finding**: study_b (spectral→RYB difference) is a nearly equal donor to spectral itself
at N=1. The intersection study (study_a, mixbox→RYB) is a strong receiver — it provides little
information per experiment about the global policy ranking despite being drawn from the
"symmetric" region.

---

### Experiment 03 — Swarm Scaling & Directional Flip Test

Core question: **can increasing N robots from a receiver source flip it to become a donor?**

Two scenarios:

**Scenario 1 — External reference (spectral as ground truth)**:

```
tau_A(N) = tau(A_N vs spectral_full)    [frugal source A, varying N]
tau_B(M) = tau(B_M vs spectral_full)    [frugal source B, fixed M]
```

"Flip" means tau_A(N*) > tau_B(ceiling).
This requires `ceiling(A vs spectral) > ceiling(B vs spectral)`.
If `ceiling_A < ceiling_B`: the gap is **permanent** — physics prevents the flip
regardless of N.  This is the bias floor from experiment 07, now reframed directionally.

**Scenario 2 — Mutual reference (each source as the other's truth)**:

```
tau_AB(N) = tau(A_N vs B_full)
tau_BA(N) = tau(B_N vs A_full)
```

Because Kendall tau is symmetric, `ceiling_AB = ceiling_BA` always.
A flip at finite N* **always exists** for non-symmetric pairs.
N* tells you: "how many experiments from A equals one experiment from B?"

---

### Experiment 04 — Flip Feasibility Map

Synthesises 02 and 03 into a taxonomy:

| Metric | Definition | Interpretation |
|--------|-----------|----------------|
| `external_ceiling(X)` | `tau(X_full vs spectral_full)` | Quality of X as spectral proxy |
| `external_gap(A, B)` | `ext_ceiling(A) − ext_ceiling(B)` | > 0: A already better; < 0: permanent gap |
| `mutual_gap(A, B, N=1)` | `tau_AB(1) − tau_BA(1)` | Who is donor at N=1 in the mutual scenario |
| `N*(A→B)` | smallest N where `tau_AB(N) > tau_BA(1)` | Robots needed for A to match B's first experiment |
| `external_centrality(X)` | mean `ext_ceiling(X)` over all Y | Overall donor quality as spectral proxy |
| `mutual_centrality(X)` | mean `mutual_gap(X?, N=1)` over Y | Net donor advantage at N=1 across all pairs |

**Cycle detection**: a directed edge A→B is added when `external_gap(A,B) > ε`.
Cycles in this graph = intransitive donor relationships (rock-paper-scissors).
If no cycles exist, the donor hierarchy is fully transitive (a linear ranking).

---

## 15. Ensemble vs Swarm Aggregation (Experiment 12)

### Distinction

| Dimension | Ensemble | Swarm |
|---|---|---|
| Weight per source | 1/K (equal, global) | 1 / mean_error(source in kNN(target)) |
| Spatial awareness | None — same weight at every target | Local — specialist sources get higher weight near targets they handle well |
| Information needed | Only aggregate tau per source | Per-target errors for each source |
| Cost | O(K) | O(K ? n_targets) |

**Ensemble** treats all sources as global peers and averages their policy scores with equal
weight at every target.  **Swarm** computes a per-target weight for each source based on
how well that source performed on the k nearest neighbors (kNN) of the query target.

### Results (104 targets, K=3: mixbox/km/ryb, N=[1–000])

| N | Ensemble tau | Swarm tau | Swarm advantage |
|---|---|---|---|
| 1 | 0.698 | 0.711 | +0.013 |
| 5 | 0.801 | 0.805 | +0.004 |
| 10 | 0.833 | 0.824 | -0.009 |
| 50 | 0.826 | 0.835 | +0.008 |
| 100 | 0.829 | 0.833 | +0.004 |
| 500 | 0.816 | 0.831 | +0.014 |
| 1000 | 0.820 | 0.831 | +0.011 |

Tau plateaus at ~0.83 — the physics ceiling for this source pool against spectral HF (104-target setting).

### TENKi-1000 Results (1000 targets, K=8: all non-spectral sources, N=[10–000])

| N | Ensemble tau | Swarm tau | Swarm advantage |
|---|---|---|---|
| 10  | 0.8417 | 0.8258 | **−0.0158** |
| 20  | 0.8808 | 0.8614 | **−0.0194** |
| 50  | 0.9197 | 0.8819 | **−0.0378** |
| 100 | 0.9344 | 0.8881 | **−0.0464** |
| 200 | 0.9461 | 0.8864 | **−0.0597** |
| 500 | 0.9531 | 0.8853 | **−0.0678** |
| 1000| 0.9494 | 0.8822 | **−0.0672** |

**Ensemble beats swarm at every N** — the opposite of expectations from the 3-source result.
Swarm advantage is consistently negative and grows worse as N increases.

**Swarm weight variability (std across targets)** by source:

| Source | Weight std | Interpretation |
|--------|-----------|----------------|
| study_c | 0.354 | Highest variability — regional specialist |
| study_b_reverse | 0.321 | High variability |
| study_a | 0.276 | High variability |
| study_b | 0.258 | High variability |
| study_c_reverse | 0.242 | Moderate variability |
| mixbox | 0.100 | Global generalist |
| ryb | 0.095 | Global generalist |
| km | 0.093 | Global generalist |

**Why swarm loses with 8 sources**: the set-op databases (study_a/b/c and their reverses) are
spatially specialized but not locally *better*. High weight variability means the swarm
upweights study_c in regions where it appears distinctive — but study_c has a low ceiling
(0.722) and is consistently worse than the global average in those very regions.
**Local weight variability — local quality.** The swarm cannot distinguish "locally different"
from "locally good."

Ensemble (1/K equal weight = 1/8) is robust to this failure mode: it dilutes the bad local
recommendations from low-ceiling sources without amplifying them.

**Conclusion**: swarm advantage requires sources that are locally *better*, not just locally
*different*. With the current pool mixing high-ceiling (ryb=0.944, mixbox=0.889) and
low-ceiling (study_c=0.722) sources, the swarm's kNN-based local weighting is net-harmful.

### Fidelity Interpretation

Increasing N (targets sampled per source, with bootstrap for N > 104):
- Improves ranking stability by reducing sampling variance
- Cannot exceed the physics bias floor (~0.83 for this pool)
- Real improvement requires a pool with at least one high-quality proxy (rho > 0.99 vs spectral)

---

## 16. Async Multi-Fidelity Optimizer (Experiment 13)

### Architecture

The optimizer in `optimizer/` is backend-agnostic: the scheduler, policies, and belief layer
are identical for offline simulation and future real async execution. Only the executor knows
which mode is active.

```
AsyncMFScheduler
  |-- BasePolicy (propose suggestions)
  |-- BaseExecutor (submit / poll_completed)
  |     |-- LocalSimExecutor  [offline: min-heap, no sleep, simulated clock]
  |-- BaseObjective (evaluate suggestions against PEGKi DBs)
  |     |-- OfflineReplayObjective  [replays study DBs; no live execution]
  |-- RunStore  [JSON snapshots + JSONL append-only events]
  |-- pegki_bridge  [PEGKi belief updates + effective-fidelity scoring]
```

### Fidelity Configuration

The optimizer now separates **raw execution fidelity** from a **transfer-usefulness
score**:

- **Raw fidelity** = the budget axis the scheduler actually spends
  - examples: `n_experiments`, `rounds`, `targets`
- **Transfer usefulness** = how useful a `(source, raw_fidelity)` state is as evidence
  for the HF ranking
  - derived from `tau_mean`, `rho_mean`, `bias_floor`, `flip_probability`, and `donor_score`
  - stored as `BeliefState.effective_fidelity`
  - mirrored into `BeliefState.quality_score` for policy compatibility

The scheduler still operates on raw fidelity. The transfer-usefulness score decides what
counts as better evidence for allocation and promotion.

**Two raw-fidelity modes:**

**Mode A — Statistical replication** (default, no extra data needed):
`fidelity_levels = [1, 3, 5]` means "sample 1, 3, or 5 experiment-level scores from the pool
and average."  Higher fidelity = lower variance.  Works with any existing single-round database.

**Mode B — Separate databases per fidelity level** (real MF):
Generate databases at different `--rounds` values:
```bash
# Low fidelity: 3 rounds
uv run python scripts/generate_policy_data.py --engine spectral --rounds 3 --experiments 5 --output output/db_spectral_r3
# High fidelity: 12 rounds
uv run python scripts/generate_policy_data.py --engine spectral --rounds 12 --experiments 5 --output output/db_spectral_r12
```
Register with `--fidelity-db spectral:3=output/db_spectral_r3 --fidelity-db spectral:12=output/db_spectral_r12`.
In Mode B each fidelity level uses all experiments from its DB (no statistical sampling).

### Optimization Methods

| `--policy-mode` | Method family | Surrogate | Fidelity strategy |
|---|---|---|---|
| `ensemble_mf` | Bandit | None | `(effective_fidelity ? source_weight) / runtime` |
| `swarm_mf` | UCB bandit | None | `(effective_fidelity + explore/sqrt(n)) / runtime` |
| `single_source_mf` | Deterministic sweep | None | All LF fidelities then HF at max |
| `mfbo` | Bayesian optimization | GP (Matern-2.5) | Cost-aware LCB or EI; round-robin warm-up |
| `smac` | SMAC-style RF | Random forest | LCB with per-tree std / runtime |
| `hyperband` | Successive halving | None | Eliminate bottom 1/eta at each fidelity level |
| `mfmc` | MFMC optimal allocation | None | Warm-up then `rho_mean`-derived LF/HF ratio, discounted by PEGKi quality |

**MFMC allocation formula** (Peherstorfer et al. 2016):
```
r_l = rho_eff,l * sqrt(w_HF / w_l) / sqrt(1 - rho_eff,l^2)
N_LF = N_HF * r_l
```
where `rho_eff,l = rho_mean,l ? (0.5 + 0.5 ? transfer_usefulness_l)` and `w` is runtime.
If `rho_mean` is not yet available, the implementation falls back to a normalized tau proxy
during warm-up.

### PEGKi Belief Updates

After each completed eval the scheduler updates `BeliefState` via the pegki_bridge:

1. **tau beliefs**: EMA of Kendall tau comparing LF policy ranking to HF ranking
   - `tau_mean = (1-alpha)*tau_mean + alpha*tau_observed`
   - Stored per `(source, fidelity)`
2. **rho beliefs**: EMA of Pearson correlation between LF per-policy scores and HF per-policy scores
   - `rho_mean` is computed against the highest-fidelity HF score map currently available
   - Stored per `(source, fidelity)`
3. **bias_floor**: `1 - tau_at_max_fidelity` — irreducible physics gap
4. **flip_probability**: instability of the maximum-fidelity ranking under bootstrap resampling
5. **donor_score**: low-budget transfer strength, taken from the minimum-fidelity tau belief
6. **effective_fidelity**: internal transfer-usefulness score for a `(source, fidelity)` state
   - blend of `tau_mean`, `rho_mean`, `1 - bias_floor`, `1 - flip_probability`, and `donor_score`
   - mirrored into `quality_score` for policy consumption

These beliefs feed directly into `EnsembleMFPolicy`, `SwarmMFPolicy`, and `MFMCPolicy`
acquisition decisions, closing the loop between the PEGKi transfer theory (§10–1) and
the online optimizer.

### Connection to Prior Sections

| Prior section | Optimizer component |
|---|---|
| §10 Frugal twin convergence | `OfflineReplayObjective` replays the same tau-vs-N logic |
| §11 MFMC optimal acquisition | `MFMCPolicy` implements the r_l formula online |
| §12 Egalitarian vs heterogeneous | `EnsembleMFPolicy` (egalitarian) vs `SwarmMFPolicy` (heterogeneous) |
| §13 TrueSkill2 partial play | `BeliefState.effective_fidelity` / `quality_score` tracks per-source transfer usefulness |
| §15 Ensemble vs swarm | Formalized as distinct policy classes; run through same scheduler |

### Empirical Results (104-target run: ensemble_mf, spectral HF, mixbox+km LF)

| Parameter | Value |
|-----------|-------|
| Policy mode | `ensemble_mf` |
| HF source | spectral (3× runtime multiplier) |
| LF sources | mixbox, km (1× each) |
| Fidelity levels | [1, 3, 5] (statistical replication) |
| Budget | 50.0 (HF-equivalent units) |
| Workers | 2 |
| **Jobs completed** | **81** |
| **Trace events** | **162** |
| **best_tau** | **0.769** |
| Simulated time | 20.7 |

### TENKi-1000 Results (1000 targets, budget=500)

| Parameter | Value |
|-----------|-------|
| Policy mode | `ensemble_mf` |
| HF source | spectral |
| LF sources | mixbox, km, ryb |
| Fidelity levels | [5, 10, 20, 50] (statistical replication) |
| Budget | 500.0 (HF-equivalent units) |
| Workers | 2 |
| **Jobs completed** | **144** |
| **Trace events** | **450** |
| **best_tau** | **1.000** (perfect HF ranking recovered) |
| Simulated time | 231.5 |
| Budget used | 459.0 / 500.0 |

LF source beliefs at termination (per (source, fidelity=5) belief state):
- ryb: tau_mean=0.833, rho=0.985, effective_fidelity=0.932
- mixbox: tau_mean=0.722, rho=0.979, effective_fidelity=0.914
- km: tau_mean=0.554, rho=0.880, effective_fidelity=0.781

With 10× the budget (500 vs 50), the optimizer recovered the perfect spectral policy ranking.
The 104-target run (budget=50) reached only tau=0.769.

HF policy ranking (by mean best_color_distance_mean, lower = better):
`bayesian_ei > bayesian_ucb > simulated_annealing > grid_search > evolutionary_strategy > thompson_sampling > ucb1_bandit > random_forest > neural_network`

### Outputs

| File | Content |
|---|---|
| `results/async_mf_optimizer_summary.json` | Final metrics, beliefs, HF rank, config |
| `results/async_mf_optimizer_trace.jsonl` | All job_leased / job_completed events (162 for default run) |
| `results/async_mf_optimizer_tau_vs_time.png` | best_tau vs simulated time |
| `results/async_mf_optimizer_allocations.png` | Jobs completed by source and fidelity level |
| `runs/exp13_async_mf/` | Full run snapshots, intermediate states, beliefs |

---

## 17. Knowledge-Space Transfer Results from the Parent PEGKi Benchmark

The parent PEGKi transfer script (`scripts/run_phase3.py`) performs cross-dataset knowledge-space transfer analysis.
Ran on `output/phase3_1000` with all `db_1000_*` and standard datasets included.

### Dataset Coverage

**15 performance vector dimensions** (trained_ratings.json sources):

| Dataset | KS type |
|---------|---------|
| `1000_spectral` | hybrid |
| `1000_study_b_reverse_artist_vs_physics` | hybrid |
| `1000_study_c_oilpaint_vs_fooddye` | empirical |
| `1000_study_c_reverse_fooddye_vs_oilpaint` | empirical |
| `km` | empirical |
| `km_paint` | empirical |
| `mixbox` | empirical |
| `rockwell_r_empirical` | empirical |
| `ryb` | theoretical |
| `ryb_paint` | theoretical |
| `shore_a_theoretical` | theoretical |
| `spectral` | hybrid |
| `kubelka_munk_difference_mixbox` | theoretical |
| `mixbox_intersection_coloraide_ryb` | hybrid_overlap |
| `spectral_difference_coloraide_ryb` | theoretical |

### Transfer Analysis

- **210 ordered transfer pairs** (15 ? 14)
- **171/210 robust** under DRO (STRONG=162, MARGINAL=9, FRAGILE=10, UNSAFE=29)
- **DRO radius** (calibrated): δ=0.5
- **Mean safety gap**: 0.116

**Conjecture 2.1 (Asymmetry)**: PARTIALLY_SUPPORTED
**Conjecture 2.2 (Hybrid Superiority)**: NOT_SUPPORTED

### Notable Asymmetric Pairs

| Pair | delta A→ | delta B→ | |asymmetry| | Verdict |
|------|-----------|-----------|------------|---------|
| 1000_spectral ↔ spectral | 0.48 | 0.388 | 0.092 | ASYMMETRIC |
| 1000_spectral ↔ mixbox | 0.226 | 0.435 | 0.209 | ASYMMETRIC |
| 1000_spectral ↔ ryb | 0.165 | 0.427 | 0.262 | ASYMMETRIC |
| 1000_spectral ↔ km | 0.050 | 0.407 | 0.357 | ASYMMETRIC |
| study_b_rev ↔ spectral | 0.355 | −0.163 | 0.519 | ASYMMETRIC (largest) |
| 1000_study_b_rev —1000_study_c | 0.486 | 0.485 | 0.001 | SYMMETRIC |
| 1000_spectral — rockwell_r | 0.5 | 0.5 | 0.0 | SYMMETRIC |
| 1000_spectral — shore_a | 0.5 | 0.5 | 0.0 | SYMMETRIC |

**Key finding**: Cross-domain transfers to hardness datasets (rockwell_r, shore_a) appear
symmetric (?=0.5 each direction) — these datasets are single-score domains with no
within-source variance, so transfer analysis collapses to the ceiling level.

The large asymmetry for `study_b_rev ↔ spectral` (0.519) — the reverse study being a strong
donor to spectral — confirms that the complement databases (reverse studies) carry complementary
information not captured by the forward direction.

### Training Cost Analysis

- Cross-engine time correlation ?=0.975 — compute cost is an intrinsic policy property
- Most efficient policy: **grid_search** (22.76 delta/s)
- Least efficient: **bayesian_ei** (0.0025 delta/s)
- `km` is 1.5× faster than `1000_study_c_oilpaint_vs_fooddye` per round

---

## 18. TENKi-1000 Experiment Series

All experiments 02–3 re-run against the `db_1000_*` databases (1000 targets each, 50
experiments per policy) with N values [10, 20, 50, 100, 200, 500, 1000].
Version 0.3.3 adds the two reverse databases (study_b_reverse, study_c_reverse) to the pool,
expanding from 7 to 9 studies.

### Database Set

| Name | Engine/study | Targets | Experiments |
|------|-------------|---------|-------------|
| `db_1000_spectral` | Beer-Lambert + CIE D65 | 1000 | 50 |
| `db_1000_mixbox` | Kubelka-Munk sRGB | 1000 | 50 |
| `db_1000_km` | KM real oil pigment | 1000 | 50 |
| `db_1000_ryb` | Artist RYB | 1000 | 50 |
| `db_1000_study_a_artist_consensus` | mixbox — RYB | 1000 | 50 |
| `db_1000_study_b_physics_vs_artist` | spectral — RYB | 1000 | 50 |
| `db_1000_study_b_reverse_artist_vs_physics` | RYB — spectral | 1000 | 50 |
| `db_1000_study_c_oilpaint_vs_fooddye` | KM — mixbox | 1000 | 50 |
| `db_1000_study_c_reverse_fooddye_vs_oilpaint` | mixbox — KM | 1000 | 50 |

All databases include completed k-fold cross-validation outputs (`trained_ratings.json`).

### Running TENKi-1000 Experiments

```bash
cd color_mixing_lab   # repo root
export PYTHONIOENCODING=utf-8
OUTDIR="extended/gamut_symmetry/results/tenki_1000"
EXPDIR="extended/gamut_symmetry/experiments"
STUDIES=(
  spectral=output/db_1000_spectral
  mixbox=output/db_1000_mixbox
  km=output/db_1000_km
  ryb=output/db_1000_ryb
  study_a=output/db_1000_study_a_artist_consensus
  study_b=output/db_1000_study_b_physics_vs_artist
  study_b_reverse=output/db_1000_study_b_reverse_artist_vs_physics
  study_c=output/db_1000_study_c_oilpaint_vs_fooddye
  study_c_reverse=output/db_1000_study_c_reverse_fooddye_vs_oilpaint
)
NVALS=(10 20 50 100 200 500 1000)

# Run 02 first (04 depends on its output)
uv run python $EXPDIR/02_directed_transfer_matrix.py --studies "${STUDIES[@]}" --n-values "${NVALS[@]}" --output-dir "$OUTDIR"
uv run python $EXPDIR/03_swarm_flip_test.py          --studies "${STUDIES[@]}" --n-values "${NVALS[@]}" --hifi spectral --output-dir "$OUTDIR"
uv run python $EXPDIR/04_flip_feasibility.py         --studies "${STUDIES[@]}" --output-dir "$OUTDIR"

# Independent experiments (no --n-values)
uv run python $EXPDIR/05_study_comparison.py             --studies "${STUDIES[@]}" --output-dir "$OUTDIR"
uv run python $EXPDIR/06_symmetry_scoring.py             --studies "${STUDIES[@]}" --output-dir "$OUTDIR"
uv run python $EXPDIR/07_frugal_twin_convergence.py      --studies "${STUDIES[@]}" --output-dir "$OUTDIR"
uv run python $EXPDIR/08_multifidelity_allocation.py     --studies "${STUDIES[@]}" --output-dir "$OUTDIR"
uv run python $EXPDIR/10_aggregation_helps_prediction.py --studies "${STUDIES[@]}" --output-dir "$OUTDIR"
uv run python $EXPDIR/11_multi_domain_flip.py            --studies "${STUDIES[@]}" --output-dir "$OUTDIR"

# With --n-values
uv run python $EXPDIR/09_mixed_source_flip.py --studies "${STUDIES[@]}" --n-values "${NVALS[@]}" --output-dir "$OUTDIR"

# Exp 12: --hfi (not --hifi), --lf (not --lf-sources); pass all 8 non-spectral as LF
uv run python $EXPDIR/12_ensemble_vs_swarm.py \
  --studies "${STUDIES[@]}" --hfi spectral \
  --lf mixbox km ryb study_a study_b study_b_reverse study_c study_c_reverse \
  --n-values "${NVALS[@]}" --output-dir "$OUTDIR"

# Exp 13: --study (singular, append)
uv run python $EXPDIR/13_async_mf_optimizer.py \
  --hifi spectral --lf-sources mixbox km ryb \
  --study spectral=output/db_1000_spectral \
  --study mixbox=output/db_1000_mixbox \
  --study km=output/db_1000_km \
  --study ryb=output/db_1000_ryb \
  --fidelity-levels 5 10 20 50 --budget-total 500 \
  --output-dir "$OUTDIR"
```

### What Changes vs v0.3.2

- **9 studies** (was 7): study_b_reverse and study_c_reverse now included
- **Bug fixes** applied: MFMC CV sign corrected, Q3 fixed-budget allocation, flip_n semantics
- **Exp 12**: uses all 8 LF sources (not just mixbox/km/ryb) — swarm now beats ensemble in a different direction

### TENKi-1000 Results Summary (v0.4.0)

Canonical artifacts: `results/tenki_1000/` (do not mix with earlier `results/` runs).

---

#### Layer A — Transfer Structure (Experiments 02–4)

*Answers: who is a donor/receiver? What is each source's ceiling? Is the gap variance-limited
or bias-limited?*

**Experiment 02 — Transfer matrix at N=10 (9 studies)**

Full N=10 transfer matrix (rows=source, cols=reference, diagonal=1.0 omitted):

| Source | spectral | mixbox | km | ryb | study_a | study_b | study_b_r | study_c | study_c_r |
|--------|----------|--------|----|-----|---------|---------|-----------|---------|-----------|
| spectral | — | 0.781 | 0.744 | 0.805 | 0.828 | 0.831 | 0.832 | 0.744 | 0.821 |
| mixbox | 0.757 | — | 0.742 | 0.773 | 0.656 | 0.640 | 0.647 | 0.579 | 0.663 |
| km | 0.479 | 0.458 | — | 0.477 | 0.428 | 0.442 | 0.459 | 0.389 | 0.460 |
| ryb | 0.664 | 0.673 | 0.631 | — | 0.591 | 0.592 | 0.604 | 0.547 | 0.605 |
| study_a | 0.423 | 0.384 | 0.424 | 0.412 | — | 0.449 | 0.453 | 0.441 | 0.473 |
| study_b | 0.796 | 0.707 | 0.671 | 0.763 | 0.848 | — | 0.852 | 0.798 | 0.852 |
| study_b_rev | 0.529 | 0.486 | 0.461 | 0.497 | 0.563 | 0.566 | — | 0.540 | 0.565 |
| study_c | 0.339 | 0.313 | 0.296 | 0.327 | 0.367 | 0.370 | 0.337 | — | 0.345 |
| study_c_rev | 0.534 | 0.522 | 0.481 | 0.518 | 0.587 | 0.579 | 0.579 | 0.557 | — |

**Net donor scores at N=10** (asymmetry matrix mean):

| Source | Net score | Role |
|--------|-----------|------|
| spectral | +0.2332 | DONOR |
| study_b | +0.2270 | DONOR |
| mixbox | +0.1416 | DONOR |
| ryb | +0.0420 | DONOR |
| study_c_reverse | −0.0535 | RECEIVER |
| study_b_reverse | −0.0694 | RECEIVER |
| km | −0.1072 | RECEIVER |
| study_a | −0.1759 | RECEIVER |
| study_c | −0.2377 | RECEIVER |

**New finding**: the two reverse databases are mild receivers (−0.05 to −0.07), weaker than km but
stronger than study_a/study_c. At full N all non-spectral tau values converge to their external
ceiling.

**Experiment 03 — External ceilings (9 studies vs spectral)**

| Frugal source | ceiling (full N) | tau@N=10 | bias floor |
|---------------|-----------------|----------|------------|
| ryb | **0.9444** | 0.664 | 0.0556 |
| mixbox | 0.8889 | 0.757 | 0.1111 |
| study_a | 0.8333 | 0.423 | 0.1667 |
| study_b | 0.8333 | **0.796** | 0.1667 |
| study_b_reverse | 0.8333 | 0.529 | 0.1667 |
| study_c_reverse | 0.8333 | 0.534 | 0.1667 |
| km | 0.7778 | 0.479 | 0.2222 |
| study_c | 0.7222 | 0.339 | 0.2778 |

study_b_reverse and study_c_reverse share the same ceiling (0.8333) as study_b and study_a — they probe a complementary color region but converge to the same HF alignment at saturation.

**Experiment 04 — Flip feasibility (9 studies)**

- **No 3-cycles detected** — donor hierarchy fully transitive (spectral > study_b > mixbox > ...)
- Mutual flip N* table: km requires N=50–00 to flip against most sources;
  study_a requires up to N=1000 to flip against study_b

---

#### Layer A — Structural Geometry (Experiments 05–6)

*Confirms that databases probe disjoint regions; provides KS-balance context.*

**Experiment 05 — Study comparison (9 studies)**

Jaccard overlaps between all study_* databases are near-zero (0.000−0.027), confirming that
the set-op and reverse databases probe nearly disjoint regions of the policy-target space.

**Experiment 06 — Symmetry scoring** *(secondary / conceptual validation — not core ranking evidence)*

Exp 06 is retained only as a structural side-note. Its role is to show that the three
derived study databases occupy distinct knowledge-space roles:

- study_a = overlap-type database (`K_H_OVERLAP`)
- study_b = transfer-type database (`K_T`)
- study_c = exclusion-type database (`K_E`)

That gives the three-study collection perfect KS-balance by construction. It does **not**
determine which source is best under a fixed budget, and it should not be used for source
allocation decisions. In the current TENKi narrative, Exp 06 is conceptual context only;
the allocation question is answered by the donor/ceiling evidence in Layer A and the
budgeted ranking evidence in Layer B.

---

#### Layer B — Budgeted Ranking Performance (Experiments 07–8)

*Answers: given a fixed budget, how should I allocate it? Does multi-fidelity theory help ranking?*

**Experiment 07 — Frugal twin convergence (7-study canonical run, 6 frugal sources, N_TOTAL=10 fixed)**

Per-study convergence (vs spectral HF). tau@N=1 is from the regenerated `flip_test_summary.json`; tau@N=100 is from `frugal_twin_convergence.json`. Full-data ceiling (N=1000) from flip_test is higher for slow-converging sources (e.g. study_a: 0.703 at N=100 -> 0.833 at N=1000). Use synthesis table (Section 20) for final manuscript values.

| Study | tau@N=1 | tau@N=100 (frugal est.) | full-data ceiling (flip_test) |
|-------|---------|------------------------|-------------------------------|
| spectral | 0.626 | 0.951 | 1.000 |
| mixbox | 0.567 | 0.901 | 0.889 |
| study_b | 0.613 | 0.844 | 0.833 |
| ryb | 0.536 | 0.871 | 0.944 |
| km | 0.310 | 0.703 | 0.778 |
| study_a | 0.221 | 0.703 | 0.833 |
| study_c | 0.130 | 0.644 | 0.722 |

Q3 diversity (fixed N_TOTAL=10 — diversity vs budget isolated):

| k | tau | Note |
|---|-----|------|
| 1 | 0.794 | Best single study |
| 2 | 0.766 | — |
| 3 | 0.728 | — |
| 4 | 0.701 | best=(study_b, mixbox, ryb, km) |
| 5 | 0.631 | — |
| 6 | 0.610 | all studies |

**Monotonically decreasing** — with truly fixed budget, concentration always beats diversity.
Previous v0.3.2 showed a k=2 peak (0.848) because the budget scaled with k (confound).
With the bug fixed, adding more study types only dilutes the 10 available experiments.

Q3b (quantity vs diversity at same N_total, best single = mixbox):

| N_total | single (mixbox) | all studies | improvement |
|---------|----------------|-------------|-------------|
| 10 | 0.764 | 0.578 | −0.186 quantity wins |
| 20 | 0.830 | 0.683 | −0.147 quantity wins |
| 30 | 0.856 | 0.746 | −0.110 quantity wins |
| 50 | 0.873 | 0.777 | −0.096 quantity wins |
| 100 | 0.900 | 0.850 | −0.050 quantity wins |

These Q3b rows are saved in the canonical artifact under
`q3b_quantity_vs_diversity` in `results/tenki_1000/frugal_twin_convergence.json`.

**Experiment 08 — Classical vs Hybrid MFMC (TENKi-1000, mixbox LF, r=100, N_bootstrap=500)**

Data audit: mixbox/km/ryb share targets with spectral by experiment index (generated
with `--shared-targets-file`); study_* databases have different targets.
Therefore mixbox, km, ryb are valid paired LF sources for the classical branch.

Pearson rho (LF vs HF spectral):

| LF source | rho | Paired? |
|-----------|-----|---------|
| study_b | 0.9973 | no |
| study_a | 0.9943 | no |
| mixbox | 0.9899 | **yes** |
| study_c_reverse | 0.9883 | no |
| ryb | 0.9856 | **yes** |
| study_b_reverse | 0.9790 | no |
| km | 0.9725 | **yes** |
| study_c | 0.9275 | no |

Oracle alpha (Cov(HF,LF)/Var(LF)) for classical branch (mixbox LF):

| Policy | alpha | true_mu_HF |
|--------|-------|------------|
| bayesian_ei | 0.017 | 1.930 |
| bayesian_ucb | 0.058 | 2.049 |
| simulated_annealing | −0.008 | 3.611 |
| thompson_sampling | 0.218 | 10.891 |
| evolutionary_strategy | 0.332 | 9.722 |
| random_forest | 0.372 | 14.767 |
| grid_search | −0.065 | 7.364 |
| ucb1_bandit | 0.271 | 11.655 |
| neural_network | 0.747 | 43.232 |

The top-ranked policies (bayesian_ei, bayesian_ucb, simulated_annealing) all have alpha < 0.06.
This means the classical control-variate correction is nearly zero for the policies that
determine the top of the ranking.

Cost-efficiency comparison at r=100 (budget grid):

| B | n_HF | n_LF | classical tau | hybrid tau | delta | HF-only tau |
|---|------|------|---------------|------------|-------|-------------|
| 1 | 0 | 0 | nan | nan | nan | 0.650 |
| 2 | 1 | 100 | 0.676 | **0.800** | −0.124 | 0.721 |
| 3 | 1 | 200 | 0.661 | **0.802** | −0.141 | 0.753 |
| 5 | 2 | 300 | 0.757 | **0.850** | −0.093 | 0.798 |
| 8 | 4 | 400 | 0.808 | **0.890** | −0.081 | 0.832 |
| 10 | 5 | 500 | 0.821 | **0.904** | −0.083 | 0.839 |
| 15 | 8 | 700 | 0.855 | **0.932** | −0.077 | 0.867 |
| 20 | 11 | 900 | 0.870 | **0.942** | −0.072 | 0.881 |
| 30 | 17 | 1300 | 0.891 | **0.959** | −0.067 | 0.912 |
| 50 | 29 | 2100 | 0.917 | **0.974** | −0.057 | 0.923 |
| 100 | 58 | 4200 | 0.939 | **0.987** | −0.048 | 0.947 |

**Verdict**: hybrid wins on tau at every budget. Classical also underperforms HF-only
on tau at every budget. This is the ranking-vs-estimation tradeoff (see §11):

- Classical is an unbiased estimator of per-policy mu_HF — it is correct for any claim
  about absolute performance levels. Policies with large alpha (neural_network: 0.747)
  show genuine MSE reduction under the classical branch (see multifidelity_mse.png).
- Hybrid is a ranking heuristic that ignores raw scales — it captures ordering information
  from the 100–200 LF observations far more efficiently than classical captures it from
  1 —8 HF observations. For the paper's primary metric (ranking quality), report hybrid.

Note: the previous TENKi result (exp 08, v0.3.2) used only the hybrid branch (called
"MF" without qualification). The MF advantage window B ∈ [2, 9] from that run used
study_b as LF; study_b is not a paired source. Those results remain valid for the hybrid
branch with study_b as LF, but should not be cited as MFMC control-variate results.

---

#### Negative Control (Experiment 10)

*Rules out an easy alternative: naive LF concatenation.*

**Experiment 09 — Mixed source flip** *(operational extension)*

Mixing multiple LF sources can reduce N* vs using any single source:
- vs study_b_reverse: N*_single=50 — N*_mix=20 (MIXING HELPS, via mixbox+study_b)
- vs study_c: N*_single=10 — N*_mix=10 (no improvement)
- vs study_c_reverse: N*_single=50 — N*_mix=20 (MIXING HELPS)

**Experiment 10 — Aggregation helps prediction (9 studies)**

All three ML models show that concatenating LF features DEGRADES prediction vs HF-only:

| Model | Mean ?MAE (concat vs HF-only) | Verdict |
|-------|-------------------------------|---------|
| Ridge | +5.955 | DEGRADES |
| RandomForest | +3.426 | DEGRADES |
| GradientBoosting | +3.138 | DEGRADES |

LF features add noise because the LF databases have different target distributions than the
spectral HF evaluation context.

**Experiment 11 — Multi-domain flip** *(operational extension; donor structure is strongest in ranking-rich domains)*

Color mixing donor scores (consistent with exp 02):

| Study | Score | Role |
|-------|-------|------|
| spectral | +0.2762 | DONOR |
| study_b | +0.2616 | DONOR |
| mixbox | +0.2081 | DONOR |
| ryb | +0.1024 | DONOR |
| km | −0.0557 | RECEIVER |
| study_c_reverse | −0.1205 | RECEIVER |
| study_b_reverse | −0.1775 | RECEIVER |
| study_a | −0.2235 | RECEIVER |
| study_c | −0.2712 | RECEIVER |

Other domains do **not** all collapse identically:

- `jarvis_leaderboard` and `materials_project` are effectively neutral (`+0.000`)
- `polymer_hardness` shows only weak directionality (`shore_a = -0.178`, `shore_d = +0.089`, `rockwell_r = +0.089`)

So the donor/receiver pattern generalises unevenly: it is strongest in the ranking-rich
color-mixing domain and much weaker in low-variance or near-single-score domains.

---

#### "What To Do Instead" (Experiments 12–3)

*Shows better aggregation and scheduling strategies that follow from the donor/bias findings.*

**Experiment 12 — Ensemble vs Swarm (K=8 LF sources)**

See §15 for full table. **Ensemble beats swarm at every N** when all 8 non-spectral sources are
included. Swarm advantage degrades from −0.016 at N=10 to −0.068 at N=500.

Weight variability: study_c (std=0.354) and study_b_reverse (std=0.321) are the most spatially
variable, but their low ceilings (0.722, 0.833) mean the swarm's local upweighting harms overall tau.

**Experiment 13 — Async MF optimizer (budget=500, fidelity=[5,10,20,50])**

| Metric | Value |
|--------|-------|
| jobs completed | 144 |
| trace events | 450 |
| **best_tau** | **1.000** (perfect ranking) |
| simulated time | 231.5 |
| budget used | 459 / 500 |

With 10× budget (500 vs 50), the optimizer recovered the complete spectral policy ranking.
LF source beliefs: ryb (effective_fidelity=0.932) > mixbox (0.914) > km (0.781).

---

## 20. Synthesis Table

One row per knowledge source. All values from canonical `results/tenki_1000/` artifacts.

**Metric sources**:
- `tau@N=1`: true single-experiment Kendall tau vs spectral, from the regenerated `flip_test_summary.json` (canonical scan starts at N=1) and cross-checked against `frugal_twin_convergence.json`.
- `tau@N=10`: ten-experiment tau vs spectral from the TENKi-1000 convergence/transfer artifacts.
- `ceiling`: best achievable tau at full N vs spectral (same source, from flip_test full data).
- `bias floor` = 1 − ceiling.
- `donor score`: mean asymmetry as source at N=10 (from transfer_matrix.json).
- `paired`: shares target RGBs by experiment index with spectral (mixbox/km/ryb only).

| Source | tau@N=1 | tau@N=10 | ceiling | bias floor | donor score | paired? | Best use |
|--------|---------|----------|---------|------------|-------------|---------|----------|
| spectral | 1.000 | 1.000 | 1.000 | 0.000 | +0.2332 | HF ref | Spend budget here if available |
| study_b | **0.619** | **0.808** | 0.833 | 0.167 | +0.2270 | no | Hybrid ranking; best unpaired donor (donor at N=1 and N=10) |
| mixbox | 0.576 | 0.765 | 0.889 | 0.111 | +0.1416 | **yes** | Hybrid ranking + classical paired CV (donor at N=1 and N=10) |
| ryb | 0.497 | 0.677 | **0.944** | **0.056** | +0.0420 | **yes** | Classical paired CV; best ceiling; **receiver at N=1** (needs ~N=8 to flip) |
| study_c_reverse | 0.239 | 0.542 | 0.833 | 0.167 | -0.0535 | no | Hybrid ranking only; net receiver at N=1 and N=10 |
| study_b_reverse | 0.218 | 0.536 | 0.833 | 0.167 | -0.0694 | no | Hybrid ranking only; net receiver at N=1 and N=10 |
| km | 0.317 | 0.464 | 0.778 | 0.222 | -0.1072 | **yes** | Classical paired CV for scalar estimation only; receiver at all N |
| study_a | 0.221 | 0.457 | 0.833 | 0.167 | -0.1759 | no | Not worth budget; slow convergence, warm-region bias |
| study_c | 0.138 | 0.338 | 0.722 | **0.278** | -0.2377 | no | Not worth budget; worst ceiling and donor score |

**Key observations**:

1. **ryb has the highest ceiling (0.944) but the weakest tau@N=10 (0.677)**; it is the best source for estimating absolute HF means (classical branch, small bias floor), but needs more experiments to manifest its advantage in ranking terms.

2. **study_b has the best tau@N=10 (0.808) and tau@N=1 (0.619) despite a moderate ceiling (0.833)**; it is the best unpaired choice at any budget. It qualifies as a donor at N=1 (tau > 0.5).

3. **ryb is a receiver at N=1 (0.497) despite having the highest ceiling (0.944)**; high ceiling means low bias floor, not fast convergence. ryb needs about N=8 experiments to cross the donor threshold.

4. **km has a large bias floor (0.222) despite being a paired source**; use it only for scalar estimation where pairing is required, not for ranking.

5. **study_a/study_c consistently underperform**; study_a has Pearson rho ~= 0.994 with spectral (higher than mixbox at 0.990), yet tau@N=1=0.221 and tau@N=10=0.457. Pearson rho does not predict ranking quality. Bias floor and tau(N) convergence are the correct diagnostics.

6. **All donor scores are strictly ordered**: spectral > study_b > mixbox > ryb > receivers. The hierarchy is fully transitive (no 3-cycles detected).

**Labeling note**: The regenerated `flip_test_summary.json` starts at `N=1`, so
`tau_source_at_1` is literal single-experiment tau in the current canonical artifact.

---

## 21. Decision Rule

Given a candidate LF source *s* and a budget *B* (in HF-equivalent units, cost ratio *r*):

```
Step 1 - Check donor score (measured here at N=10).
  If donor_score(s) <= 0 and you need ranking quality:
    Do not allocate LF budget to s.
    Concentrate budget on HF or the best available donor.

Step 2 - Check ceiling.
  If ceiling(s) < required_tau:
    LF source cannot close the gap regardless of N.
    This is a permanent bias-limited failure; adding data does not help.

Step 3 - Choose branch based on task.
  If task = ranking:
    Use hybrid branch (z-score fusion).
    Require: donor_score > 0 and ceiling > required_tau.
    Prefer: study_b (best tau@N=10 = 0.808, tau@N=1 = 0.619, best unpaired) or mixbox (paired + good ceiling).
  If task = scalar estimation (absolute performance levels):
    Use classical branch (paired CV estimator).
    Require: source is paired (mixbox, km, or ryb).
    Prefer: ryb (lowest bias floor = 0.056).

Step 4 - Fixed-budget allocation.
  Concentration beats diversity for generalist sources (Exp 07, Layer B):
    Spend all LF budget on the single best source for your task (Steps 1-3).
    Do NOT split budget across multiple sources unless you have evidence of
    complementary spatial coverage (set-op databases).

Step 5 - Compute MFMC allocation ratio.
  n_LF / n_HF = sqrt(r) * |rho| / sqrt(1 - rho^2)
  This is an allocation PRIOR, not a guarantee. Validate with Kendall tau.

Step 6 - Validate empirically.
  Before committing to an LF source in production:
    Run the flip test (Exp 03) to confirm the source can reach required_tau.
    Run the budget sweep (Exp 08 hybrid mode) to confirm tau improves vs HF-only.
    If tau does not improve: fall back to HF-only.
```

**Summary table**:

| Condition | Recommendation |
|-----------|---------------|
| donor_score <= 0 | Spend budget on HF only |
| ceiling < required_tau | Source permanently blocked; HF only |
| ranking task + donor > 0 + ceiling ok | Hybrid branch; prefer study_b or mixbox |
| scalar estimation + paired source available | Classical branch; prefer ryb |
| fixed budget, multiple sources | Concentrate on best single source |
| hybrid tau > HF-only tau | MF is helping; maintain LF allocation |
| hybrid tau <= HF-only tau | MF is not helping; redirect budget to HF |

---

## 19. Open Questions

### Answered in v0.3

- **Venn region volumes**: Experiment 01 maps all pairwise (and multi-engine) intersection,
  difference, and complement volumes.  The 4-engine intersection non-empty status is now
  empirically determined.
- **Directional donor/receiver structure**: Experiment 02 produces the full asymmetric tau
  matrix at N=1, N=5, N=10, and full N across 7 studies. spectral and study_b are top donors;
  study_a and study_c are receivers (§14, Experiment 02 results).
- **Flip feasibility**: Experiment 03 tests both the external (vs spectral) and mutual
  scenarios for every ordered pair.  Experiment 04 maps permanent-gap vs flippable pairs.
- **Async MF optimizer baseline**: Experiment 13 (ensemble_mf): 81 jobs, best_tau=0.769 at
  budget=50 with spectral HF + mixbox/km LF (§16).
- **Parent PEGKi transfer analysis**: 210 transfer pairs computed; 171/210 robust (DRO δ=0.5);
  Conjecture 2.1 PARTIALLY_SUPPORTED, Conjecture 2.2 NOT_SUPPORTED (§17).
- **TENKi-1000 series**: All experiments 02–3 re-run with 1000-target databases and
  N=[10,20,50,100,200,500,1000] (§18).

### Answered in v0.3.3 (TENKi-1000)

- **Intransitive donor cycles** (Q8 below): Experiment 04 over the full 9-study set
  confirms no 3-cycles — the donor hierarchy is fully transitive.  Blade-Chest cycle
  detection is not needed for this dataset.

- **Fixed-budget diversity (Q4 partial)**: Experiment 07 with truly fixed N_TOTAL shows
  tau monotonically *decreases* as k increases (0.794–0.610 for k=1–6).  With
  generalist sources, adding studies always dilutes the best source rather than
  complementing it.  Rho-proportional allocation would face the same ceiling: there is
  no complementary region to exploit.  The question remains open for genuinely
  heterogeneous (set-op) source pools.

- **Ensemble vs swarm at K=8 (Q7 partial)**: Experiment 12 shows ensemble beats swarm
  at every N when all 8 non-spectral sources are included.  High local weight variability
  (std=0.242−0.354 for set-op sources) does not translate to local quality — swarm
  upweights distinctive-but-worse sources in their distinctive regions.  Ensemble's
  equal-weight dilution is robust; swarm advantage requires genuine local specialists.

- **Async MF at scale**: Experiment 13 with budget=500 achieves best_tau=1.000
  (complete ranking recovery), with LF fidelity beliefs: ryb (0.932) > mixbox (0.914)
  > km (0.781).  The ensemble_mf policy converges reliably given sufficient budget.

### Still Open

1. **Mirror studies**: Generate the 8 missing complement databases
   (G_RYB \ G_mixbox, G_RYB \ G_spectral, G_mixbox \ G_spectral, etc.) and verify
   that policies transfer symmetrically to/from mirror pairs.

2. **Approximate symmetry threshold**: At what ε does the 3-study KS-balanced collection
   achieve approximate engine-permutation symmetry? (Use Venn volumes from exp 01.)

3. **Nash equilibrium under full symmetry**: If all 15 Venn regions were studied, would
   the Nash equilibrium shift from 2 databases to more? Or would it still concentrate?

4. **Quality-aware diversity allocation (Q3 replacement)**: Rho-proportional `n_i`
   allocation is only meaningful when sources have spatially heterogeneous quality.
   Prerequisite: include at least one set-op (intersection/difference) source so that
   local rho genuinely varies across target regions.  With 4 generalist engines, the
   experiment is underspecified — diversity always hurts.

5. **Per-robot information weighting**: Use inverse-variance weighting within a study
   (exp 10 in the planned roadmap).  Do hard-target experiments carry more or less
   transferable information than easy ones?

6. **Per-policy rho for MFMC**: Compute rho_p per policy rather than globally (exp 11).
   Are some policies universally well-correlated across LF/HF (robust policies)?

7. **TrueSkill2 multi-team Q3**: Run TrueSkill2 on (study, n_robots) combinations using
   Kendall tau as the match outcome signal (exp 12).  Does the inferred team skill per
   study converge to values consistent with empirical rho and bias floor?

8. ~~**Intransitive donor cycles**~~: Resolved — no cycles in the 9-study set (see v0.3.3).

9. **Flip N* sensitivity to target difficulty**: N* (robots needed to flip) may differ for
   easy vs hard target colors.  Per-difficulty flip curves would show whether a receiver
   can flip earlier on easy targets while remaining a permanent receiver on hard ones.

10. **Swarm advantage with true local specialists**: The swarm loss at K=8 (exp 12) is
    explained by high-variability but low-ceiling set-op sources.  Re-run exp 12 with
    only the 4 generalist engines (no study_a/b/c) or with intersection/difference
    sources that are both locally distinctive AND locally better than the alternatives.
    Expected result: swarm advantage re-emerges when spatial quality heterogeneity
    aligns with spatial weight heterogeneity.

---

## Quick Commands

```bash
# From color_mixing_lab root:

# --- Cross-validation on each database (fast, seconds per DB) ---
uv run python scripts/train_ratings_from_data.py --database output/db_spectral --k-folds 10 --visualize
# (repeat for each db_* directory)

# --- Parent PEGKi knowledge-space transfer analysis (reads all trained_ratings.json) ---
uv run python scripts/run_phase3.py --database-root output --output-dir output/phase3_1000

# --- TENKi-1000: transfer-matrix run with 1000-target databases (N=10-1000) ---
# See §18 for full commands. Quick single run:
cd extended/gamut_symmetry
uv run python experiments/02_directed_transfer_matrix.py \
  --output-dir results/tenki_1000 \
  --n-values 10 20 50 100 200 500 1000 \
  --studies spectral=output/db_1000_spectral mixbox=output/db_1000_mixbox \
            km=output/db_1000_km ryb=output/db_1000_ryb \
            study_a=output/db_1000_study_a_artist_consensus \
            study_b=output/db_1000_study_b_physics_vs_artist \
            study_c=output/db_1000_study_c_oilpaint_vs_fooddye

# --- Foundation: Venn geometry + directed transfer (run before 03/04) ---
uv run python extended/gamut_symmetry/experiments/01_venn_geometry.py
uv run python extended/gamut_symmetry/experiments/02_directed_transfer_matrix.py

# --- Core flip question ---
uv run python extended/gamut_symmetry/experiments/03_swarm_flip_test.py
uv run python extended/gamut_symmetry/experiments/04_flip_feasibility.py

# --- Study distribution & symmetry scoring ---
uv run python extended/gamut_symmetry/experiments/05_study_comparison.py
uv run python extended/gamut_symmetry/experiments/06_symmetry_scoring.py

# --- Frugal twin / swarm ---
uv run python extended/gamut_symmetry/experiments/07_frugal_twin_convergence.py

# --- Multi-fidelity acquisition ---
uv run python extended/gamut_symmetry/experiments/08_multifidelity_allocation.py

# --- Mixed-source flip, aggregation, multi-domain ---
uv run python extended/gamut_symmetry/experiments/09_mixed_source_flip.py
uv run python extended/gamut_symmetry/experiments/10_aggregation_helps_prediction.py
uv run python extended/gamut_symmetry/experiments/11_multi_domain_flip.py

# --- Ensemble vs swarm (exp 12) ---
uv run python extended/gamut_symmetry/experiments/12_ensemble_vs_swarm.py
# With bootstrap N values and custom output dir:
uv run python extended/gamut_symmetry/experiments/12_ensemble_vs_swarm.py \
    --n-values 1 5 10 50 100 500 1000 \
    --output-dir extended/gamut_symmetry/results/exp12_1000targets

# --- Async MF optimizer (exp 13) ---
# Default (ensemble_mf, spectral HF, mixbox+km LF)
uv run python extended/gamut_symmetry/experiments/13_async_mf_optimizer.py
# Choose policy mode
uv run python extended/gamut_symmetry/experiments/13_async_mf_optimizer.py --policy-mode mfbo
uv run python extended/gamut_symmetry/experiments/13_async_mf_optimizer.py --policy-mode smac
uv run python extended/gamut_symmetry/experiments/13_async_mf_optimizer.py --policy-mode hyperband
uv run python extended/gamut_symmetry/experiments/13_async_mf_optimizer.py --policy-mode mfmc
# With real study databases and fidelity-DB map:
uv run python extended/gamut_symmetry/experiments/13_async_mf_optimizer.py \
    --policy-mode mfbo \
    --study spectral=output/db_spectral \
    --study mixbox=output/db_mixbox \
    --study km=output/db_km \
    --fidelity-db spectral:3=output/db_spectral_r3 \
    --fidelity-db spectral:12=output/db_spectral_r12 \
    --fidelity-levels 3 12 \
    --budget-total 500

# --- Generate missing complement databases (open question 1) ---
uv run python scripts/generate_policy_data.py \
    --set-op complement --engine-a mixbox --engine-b coloraide_ryb \
    --output output/db_study_a_complement

# --- Generate multi-fidelity databases (for Mode B fidelity, open question) ---
uv run python scripts/generate_policy_data.py --engine spectral --rounds 3  --experiments 5 --output output/db_spectral_r3
uv run python scripts/generate_policy_data.py --engine spectral --rounds 12 --experiments 5 --output output/db_spectral_r12
```

---

## Abbreviations

| Abbreviation | Meaning |
|---|---|
| R_S | Venn region for engine subset S |
| K_H | Hybrid knowledge space (physics / spectral) |
| K_E | Empirical knowledge space (measured pigment) |
| K_T | Theory knowledge space (artist RYB) |
| K_H_OVERLAP | Intersection / overlap knowledge space |
| SymScore | Symmetry score of a study collection (§6) |
| KS | Knowledge Space |
| V_i | Volume fraction of study D_i |
| TS2 | TrueSkill2 (Minka et al. 2018) |
| w_i | Partial play fraction for robot/player i |
| beta^2 | Per-player performance noise in TrueSkill2 team model |
| mu_team | Team performance mean (weighted sum of player skills) |
| sigma_team^2 | Team performance variance (quadratic in w_i) |
| MF | Multi-fidelity |
| MFBO | Multi-Fidelity Bayesian Optimization |
| MFMC | Multi-Fidelity Monte Carlo (Peherstorfer et al. 2016/2018) |
| SMAC | Sequential Model-based Algorithm Configuration (RF surrogate) |
| GP | Gaussian Process surrogate |
| RF | Random Forest surrogate |
| LCB | Lower Confidence Bound acquisition function |
| EI | Expected Improvement acquisition function |
| HF | High-fidelity source (spectral engine) |
| LF | Low-fidelity source (mixbox, km, ryb, set-op databases) |
| tau_mean | EMA of Kendall tau between LF and HF policy rankings (BeliefState) |
| rho_mean | EMA of Pearson correlation between LF and HF per-policy score maps (BeliefState) |
| bias_floor | 1 - tau_at_max_fidelity; irreducible physics gap for a LF source |
| donor_score | Low-budget transfer strength, taken from the minimum-fidelity tau belief |
| effective_fidelity | internal transfer-usefulness score for a `(source, raw_fidelity)` state |
| kNN | k-nearest neighbors (used for swarm local weight computation) |
| TENKi | Multi-domain transfer framework; "TENKi-1000" = the 1000-target database series |
| DRO | Distributionally Robust Optimization (Hanada et al. 2025) — used for safety bounds |
| delta | Transfer delta: performance gain when transferring from source A to reference B |
| safety gap | DRO metric: minimum performance margin across all worst-case distributions |
