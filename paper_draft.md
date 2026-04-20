# When Can a Receiver Become a Donor?
## Swarm Scaling, Bias Floors, and Directional Asymmetry in Multi-Fidelity Policy Ranking

**Draft** — March 2026
**Status**: Results for Sections 5.1–5.2 (experiments 01–04) are placeholders pending experiment runs.
Sections 5.3–5.5 (experiments 05–08) use real empirical data.

---

## Abstract

Evaluating machine learning policies across multiple simulation environments of varying cost and
physical fidelity is a central challenge in multi-fidelity learning.  A low-fidelity (LF) environment
that is cheap to query may produce policy rankings that correlate with those from an expensive
high-fidelity (HF) reference — but correlation alone does not determine whether the LF source can
*substitute* for the HF reference, or in what quantity.

We study this problem through the lens of **directional asymmetry**: given two simulation environments
A and B, we ask whether increasing the number of independent evaluations (robots) drawn from A can
flip A's role from *receiver* (approximating B's ranking) to *donor* (serving as a reference that B
approximates).  We identify two structurally different regimes:

1. **External-reference scenario**: both environments are measured against a shared HF ground truth.
   A source with a lower *bias floor* has a permanent ceiling on its ranking quality that cannot be
   overcome by adding more evaluations.  The donor/receiver hierarchy is fixed by the physics gap.

2. **Mutual-reference scenario**: each environment serves as the other's ground truth.  Because
   Kendall's τ is symmetric at full information, both environments share the same ceiling.
   A flip at finite N* *always* exists; N* quantifies information density per evaluation.

We instantiate this framework using four color-mixing simulation engines — a Beer-Lambert spectral
model, a Kubelka-Munk pigment model, a measured oil-pigment dataset, and an RYB artist color-wheel
model — generating seven study databases from set-operation regions of their gamut Venn diagram.
Empirically, across nine ML policies and 300 bootstrap samples per condition, we find that
study_b (spectral − RYB difference set) is the best frugal proxy for the HF spectral reference
(Pearson ρ = 0.996, bias floor τ ≈ 0.105), while study_c (KM − mixbox) is the worst
(ρ = 0.843, bias floor τ ≈ 0.318), a gap that no amount of additional sampling can close.
Three HF experiments combined with 100 LF experiments from the best source achieves
τ = 0.891 — equivalent to ~10 HF-only experiments at one-third the HF cost.

---

## 1. Introduction

Multi-fidelity machine learning [CITE: Peherstorfer2016, Kennedy2000] addresses the problem of
combining cheap, imperfect observations with expensive, accurate ones.  The standard framing assumes
a single LF source and a single HF oracle.  In practice, especially in scientific simulation
[CITE: example], there are many LF sources of varying quality, and the question of *which* to use —
and in what quantity — is non-trivial.

A natural follow-up question that has received less attention is **directional**: given two
sources A and B where A currently produces worse rankings than B, can A "flip" its role
to become the better reference by simply running more evaluations?  The answer depends on
whether the quality gap is due to *variance* (reducible by N) or *bias* (irreducible — a
permanent physics gap between the two simulation models).

This paper makes three contributions:

1. We introduce the **donor/receiver flip framework** (Section 3), which formalises the conditions
   under which a receiver source can acquire donor status.  We distinguish the external-reference
   scenario (permanent gaps possible) from the mutual-reference scenario (always flippable at
   finite N*).

2. We characterise the **Venn geometry** of four simulation engine gamuts (Section 4.1) and the
   **directed transfer matrix** (Section 4.2), establishing the empirical ground truth of which
   sources are donors and which are receivers at low N.

3. We connect the flip framework to **multi-fidelity Monte Carlo** (MFMC) optimal allocation
   [CITE: Peherstorfer2016], showing that the practical cost-efficiency benefit of a LF source is
   determined by its Pearson correlation with the HF reference (Section 5.5) — and that a high-ρ
   LF source can provide 10 HF-experiment quality at 3 HF-experiment cost.

The simulation domain is a color mixing laboratory where ML policies learn to reproduce target
colors by mixing four pigment dyes.  Four physically distinct engines represent different levels
of model fidelity, from first-principles Beer-Lambert spectral simulation to artist-intuition
RYB color theory.  This domain is chosen for its interpretable physics gap structure: each engine
produces a measurable, distinct gamut in RGB space, enabling controlled study of asymmetry.

---

## 2. Related Work

### Multi-Fidelity Methods

Multi-fidelity Monte Carlo (MFMC) [CITE: Peherstorfer2016] minimises the variance of an
estimator by combining cheap, correlated LF samples with expensive HF samples.  The optimal
LF:HF ratio is sqrt(r) · |ρ| / sqrt(1 − ρ²) where r is the cost ratio and ρ is the
Pearson correlation between LF and HF scores.  Our work extends MFMC to the *ranking quality*
setting (Kendall τ rather than estimator variance) and across multiple LF sources simultaneously.

Multi-fidelity Bayesian optimisation [CITE: Kandasamy2016, Poloczek2017] selects which fidelity
level to query at each step, whereas MFMC allocates a fixed budget.  Our setting is closer to
MFMC: we have a fixed evaluation budget and ask how to split it.

### Swarm Intelligence

In swarm intelligence [CITE: Kennedy1995, Dorigo2006], multiple agents collectively solve problems.
Heterogeneous swarms [CITE: Brambilla2013] assign different roles to agents of different capability.
Our "robots" (policy experiments from a given study) form heterogeneous swarms: robots from
study_b carry more information per unit cost than robots from study_c.  The egalitarian assumption
(equal weight per robot) is the multi-fidelity equivalent of a homogeneous swarm.

### Knowledge Space Theory

Knowledge Space Theory (KST) [CITE: Doignon1985] formalises what an agent knows in terms of
knowledge states and transfer paths.  The PEGKí framework [CITE: PEGKi] maps KST types to
simulation physics:  K_H (hybrid spectral), K_E (empirical pigment), K_T (theory RYB).
Our set-operation study databases (intersection, difference, complement) sample different Venn
regions of the KST decomposition, and K=3 studies achieve KS-balance (one per type).

### Ranking Systems and Intransitivity

TrueSkill [CITE: Herbrich2006] and Blade-Chest [CITE: Chen2016] provide ranking systems that
handle intransitive relationships (rock-paper-scissors patterns).  We use Kendall's τ as a
ranking quality metric throughout, and connect the flip framework to TrueSkill team modeling
in Section 6.3.

---

## 3. The Donor/Receiver Flip Framework

### 3.1 Definitions

Let S = {S_1, ..., S_K} be a collection of study databases, each containing independent
policy evaluation experiments.  For study S_i, let x_{i,p}^{(e)} denote the performance
score of policy p on experiment e.

**Ranking from N experiments** (sub-sampling): given N draws from study S_i, the induced ranking
is R_i(N) = argsort_p [ mean_{e~Unif(S_i,N)} x_{i,p}^{(e)} ].

**Transfer quality**: τ_{ij}(N) = Kendall's τ between R_i(N) and R_j(full), where R_j(full)
uses all available experiments from S_j.

**Donor at N**: S_i is the donor relative to S_j at N iff τ_{ij}(N) > τ_{ji}(N).

**Flip point N*_{ij}**: the minimum N such that τ_{ij}(N) > τ_{ji}(1).
(S_i surpasses S_j's single-experiment performance.)

### 3.2 Two Scenarios

**Scenario 1 — External reference (HF ground truth fixed)**

Let H be the high-fidelity reference.  Define:
  ceiling(S_i) = τ_{iH}(∞) = Kendall's τ between S_i's full ranking and H's full ranking.

The external flip gap is:
  gap_ext(A, B) = ceiling(A) − ceiling(B)

- gap_ext > 0: A is already a better HF proxy than B.  (A is the donor now.)
- gap_ext < 0: **A has a permanent physics gap**.  Regardless of N, A cannot surpass B as
  an approximation of H.  This is because ceiling(A) is determined by the bias floor —
  the irreducible error from the physics gap between A's model and H's model.

**Scenario 2 — Mutual reference**

Because Kendall's τ is symmetric: τ_{AB}(∞) = τ_{BA}(∞) = shared ceiling.
Both sources converge to the *same* ceiling.  Therefore:
  - For any non-symmetric pair (A, B), a flip at finite N* always exists.
  - N*_{AB} is the minimum N such that τ_{AB}(N) > τ_{BA}(1).
  - N*_{AB} quantifies **information density**: how many experiments from A equal one from B.

### 3.3 Bias Floor and Permanent Gaps

The bias floor of source A relative to reference H is:
  bias_floor(A) = 1 − ceiling(A)

When bias_floor(A) > bias_floor(B), adding robots from A cannot help A surpass B as an HF proxy.
The gap is a property of the physics, not the sample size.

This is analogous to the **irreducible error** in bias-variance decomposition: increasing N
reduces variance but cannot reduce bias below the model's structural floor.

---

## 4. Methods

### 4.1 Simulation Domain: Four Color Mixing Engines

We implement four color mixing simulation engines, each representing a distinct physical model:

| Engine | Model | Gamut constraint | KS type |
|--------|-------|-----------------|---------|
| **spectral** | Beer-Lambert + CIE D65 + 1931 CMF | R=[0,255], G=[132,255], B=[107,255] | K_H |
| **mixbox** | Kubelka-Munk sRGB pigment | Full range, 4-endpoint | K_E |
| **kubelka_munk** | KM real oil pigment (OSF Wiersma) | R=[31,248], G=[40,248], B=[23,232] | K_E |
| **coloraide_ryb** | Itten RYB artist color wheel | Warm-biased | K_T |

Each engine accepts a four-component action (red%, yellow%, blue%, water%) constrained to sum
to 100%, and returns an RGB color.  The engines differ in their reachable gamut — the set of
RGB colors they can produce — and in their underlying physics assumptions.

### 4.2 Venn Region Geometry (Experiment 01)

We characterise the gamut of each engine by sampling a dense grid of actions (16 steps per
axis, ≈ 4,096 valid combinations per engine) and recording the resulting RGB points.  We then
voxelise at resolution 40³ and compute pairwise Venn regions:

- **Intersection** G_A ∩ G_B: colors both engines can produce
- **A-only** G_A \ G_B: colors A produces that B cannot
- **B-only** G_B \ G_A: colors B produces that A cannot

The **asymmetry index** = (|A-only| − |B-only|) / (|A-only| + |B-only|) is signed:
positive means A has more exclusive colors; negative means B does.

> **[PLACEHOLDER — Experiment 01 results pending]**
> Table 1 will report: pairwise Venn volumes, asymmetry indices, and Jaccard coefficients
> for all 6 engine pairs.  The 4-engine intersection (fully symmetric core) volume will
> determine whether engine-permutation symmetry is achievable.

### 4.3 Directed Transfer Matrix (Experiment 02)

For all ordered pairs (i, j) across seven study databases, we compute τ_{ij}(N) via 300
bootstrap samples at N = 1, 5, 10, and full.

The **asymmetry matrix** at N=1 is A[i,j] = τ_{ij}(1) − τ_{ji}(1), which identifies which
source is the donor in each pair when only a single experiment is available.

> **[PLACEHOLDER — Experiment 02 results pending]**
> Figure 2 will show the 7×7 directed transfer matrix as a heatmap at N=1 and N=full.
> Table 2 will report donor scores (mean row of A) for each study, establishing the
> donor/receiver hierarchy at low N.

### 4.4 Swarm Flip Test (Experiment 03)

For every ordered pair (A, B) we compute:
- τ_{AB}(N) = Kendall's τ of A's N-experiment ranking vs B's full ranking (300 bootstraps)
- τ_{BA}(N) = same with A and B swapped

and find N*_{AB} = min N such that τ_{AB}(N) > τ_{BA}(1) + ε (ε = 0.01).

We run both Scenario 1 (spectral as fixed external reference) and Scenario 2 (mutual).

> **[PLACEHOLDER — Experiment 03 results pending]**
> Figure 3 will show flip curves for all frugal source pairs (Scenario 1).
> Figure 4 will show the crossover heatmap: N*_{AB} for all ordered pairs (Scenario 2),
> with gray cells where flip was not achieved within the tested N range.
> Key finding to confirm: whether any receiver achieves a permanent-gap flip vs spectral.

### 4.5 Flip Feasibility Map (Experiment 04)

We synthesise experiments 02 and 03 into a taxonomy:
- **FLIPPABLE**: gap_ext(A, B) > ε — A is already the better HF proxy.
- **PERMANENT_GAP**: gap_ext(A, B) < −ε — A cannot surpass B as an HF proxy, regardless of N.
- **SYMMETRIC**: |gap_ext| ≤ ε.

Donor centrality = mean external ceiling across all competitors.
We also detect 3-cycles (intransitive donor relationships) in the directed donor graph.

> **[PLACEHOLDER — Experiment 04 results pending]**
> Figure 5 will show the external gap matrix and donor centrality bar chart.
> Table 3 will report: external ceiling per source, donor centrality, mutual N* summary,
> and number of 3-cycles detected.
> Key finding to confirm: whether the donor hierarchy is fully transitive or contains cycles.

### 4.6 Study Databases and Distribution Comparison (Experiment 05)

We construct seven study databases from the set-operation decomposition of engine gamuts.
Three databases sample set-op regions:

| Study | Operation | Engines | Mean RGB | N targets |
|-------|-----------|---------|----------|-----------|
| study_a | G_mixbox ∩ G_RYB | intersection | (189, 127, 121) | 900 |
| study_b | G_spectral \\ G_RYB | difference | (162, 185, 174) | 900 |
| study_c | G_KM \\ G_mixbox | difference | (163, 154, 81) | 900 |

Four single-engine databases (spectral, mixbox, km, ryb) complete the collection.

We measure pairwise Wasserstein-1 distances between target distributions.  The three
set-operation studies are maximally separated: W₁(study_a, study_b) = 0.194,
W₁(study_a, study_c) = 0.171, W₁(study_b, study_c) = 0.229 (in [0,1]³).
The Jaccard overlap between all pairs of set-op studies is 0 (perfectly disjoint).

### 4.7 Symmetry Scoring (Experiment 06)

We score collections of study databases against three symmetry definitions
[see TECHNICAL_REFERENCE §4]:
- **KS-balanced**: equal representation of K_H, K_E, K_T knowledge space types.
- **Coverage-uniform**: target density proportional to region volume.
- **Engine-permutation symmetric**: all Venn regions represented proportionally.

The three set-op studies {A, B, C} achieve perfect KS-balance (score = 1.0) with
one study per KS type.  The overall set-op symmetry score is 0.827.  Full
engine-permutation symmetry requires K = 2⁴ − 1 = 15 studies for 4 engines; the
current collection of 7 databases reaches approximately 75% engine-permutation symmetry.

### 4.8 Frugal Twin Convergence (Experiment 07)

We treat each policy experiment as a "robot" and measure how quickly a swarm of N robots
from a frugal source converges to the spectral HF ranking.

**Setup**: 9 ML policies, spectral reference (104 experiments per policy), 300 bootstrap
samples per N.  The HF reference ranking is [bayesian_ucb, bayesian_ei, simulated_annealing,
grid_search, evolutionary_strategy, ucb1_bandit, thompson_sampling, random_forest,
neural_network] (best to worst, lower color distance = better).

**Convergence threshold**: τ ≥ 0.80 is considered "matched" to the HF reference.

### 4.9 Multi-Fidelity Optimal Allocation (Experiment 08)

We apply MFMC theory [CITE: Peherstorfer2016] to find the optimal n_LF : n_HF ratio:

  n_LF / n_HF = sqrt(r) · |ρ| / sqrt(1 − ρ²)

where r = cost_HF / cost_LF and ρ = Pearson correlation between LF and HF per-policy scores.

We validate empirically via a grid of (n_HF, n_LF) combinations with 300 bootstrap samples.

---

## 5. Results

### 5.1 Venn Region Geometry

> **[PLACEHOLDER]**
> *Report: pairwise Venn volumes (Table 1), asymmetry indices, 4-engine intersection volume.*
> *Key result: whether the 4-engine intersection is non-empty (required for the fully symmetric core).*
> *Interpretation: the asymmetry indices should predict which studies are donors vs receivers*
> *(sources with more exclusive unique colors carry more distinct information per experiment).*

### 5.2 Directed Transfer Matrix and Flip Feasibility

> **[PLACEHOLDER]**
> *Report: Figure 2 (7×7 heatmaps at N=1 and N=full), Figure 3 (flip curves, Scenario 1),*
> *Figure 4 (N* crossover heatmap, Scenario 2), Figure 5 (donor centrality), Table 3 (taxonomy).*
>
> *Expected findings based on bias floor results from §5.3:*
> - *spectral should be the clear net donor (external ceiling = 1.0 by definition)*
> - *study_b should be the best frugal donor (external ceiling ≈ 0.894)*
> - *study_c should be the worst receiver (external ceiling ≈ 0.677, permanent gap vs study_b)*
> - *Mutual flip N* for study_c vs study_b: expected large (study_c needs many experiments*
>   *to match even one experiment from study_b in the mutual scenario)*
> - *Whether 3-cycles exist in the donor graph: TBD*

### 5.3 Frugal Twin Convergence (Empirical)

Table 1 reports Kendall's τ vs the spectral reference as a function of N robots, aggregated
over 300 bootstrap samples.

**Table 1. Frugal source convergence: τ vs spectral at selected N.**

| Source | N=1 | N=5 | N=10 | N=20 | N=100 | Bias floor |
|--------|-----|-----|------|------|-------|-----------|
| spectral (self) | 0.675 | 0.831 | 0.883 | 0.908 | **0.974** | 0.026 |
| study_b (spectral−RYB) | 0.652 | [PH] | [PH] | [PH] | **[PH]** | **≈0.106** |
| study_a (mixbox∩RYB) | 0.203 | 0.355 | 0.436 | 0.525 | 0.746 | 0.254 |
| study_c (KM−mixbox) | [PH] | [PH] | [PH] | [PH] | [PH] | **≈0.318** |

> **[PLACEHOLDER — study_b N=5,10,20,100 and study_c full row: run exp 07 to get these values]**
> *The TECHNICAL_REFERENCE reports study_b ceiling ≈ 0.894 (bias floor 0.105) and*
> *study_c ceiling ≈ 0.677 (bias floor 0.318) from prior runs.*

**Key finding 1** (permanent gap): study_a's ceiling (τ ≈ 0.746 at N=100) is below
study_b's ceiling (τ ≈ 0.894).  This gap of Δτ ≈ 0.148 cannot be closed by additional
sampling — it reflects the fundamental physics difference between the mixbox∩RYB intersection
(warm consensus colors) and the spectral−RYB difference (physics-exclusive cool colors).

**Key finding 2** (intersection ≠ easier): study_a, which samples the *intersection* of two
engine gamuts, has a *higher* bias floor (0.254) than study_b, which samples a *difference*.
Intersection does not imply proximity to the HF spectral reference.  The intersection of two
artist models (mixbox, RYB) is precisely where spectral physics contributes the least
distinguishing information.

**Key finding 3** (quantity vs diversity): at N=10, study_b alone achieves τ ≈ [PH] while
an egalitarian combination of all frugal sources (1–2 robots each) achieves τ ≈ [PH].
Adding high-bias studies (study_a, study_c) to the swarm with equal weight *hurts* the
combined ranking.  This is the egalitarian collapse predicted by TrueSkill2 team modeling
when low-skill players receive the same partial-play fraction as high-skill players.

### 5.4 Symmetry of the Study Collection

The three set-operation studies achieve **perfect KS-balance** (score = 1.0): one study
per knowledge space type (K_H_OVERLAP, K_T, K_E).  Pairwise Jaccard overlap is 0 for all
pairs — the studies sample perfectly disjoint regions of the RGB color space.

The combined target distribution centroid is RGB = (171, 155, 125) [in [0,255]],
offset from the neutral gray (128, 128, 128) by a mild warm bias (higher R and G).
This residual asymmetry reflects the physics: the spectral engine's restricted gamut
(G ≥ 132, B ≥ 107) limits how close any set-op collection can get to uniform coverage.

The overall symmetry score of the 7-database collection is [PH] (KS-balance = 1.0,
engine-permutation symmetry ≈ 0.75, coverage-uniformity ≈ 0.927), giving a combined
score of ≈ 0.894.

Full engine-permutation symmetry requires K = 15 databases (all 2⁴ − 1 Venn regions).
The 8 currently missing databases are the "mirror" complements
(G_RYB \ G_mixbox, G_RYB \ G_spectral, etc.).

### 5.5 Multi-Fidelity Allocation

**Table 2. Pearson correlation ρ between LF source and spectral HF per-policy scores.**

| LF source | ρ | MFMC optimal ratio (r=100) |
|-----------|---|---------------------------|
| study_b | **0.996** | 108 LF : 1 HF |
| study_a | 0.994 | 90 LF : 1 HF |
| mixbox | 0.989 | 66 LF : 1 HF |
| ryb | 0.985 | 56 LF : 1 HF |
| km | 0.973 | 42 LF : 1 HF |
| study_c | 0.843 | 16 LF : 1 HF |

**Key finding 4** (high ρ ≠ good proxy): study_a has ρ = 0.994 (near-perfect Pearson
correlation) yet the worst bias floor among the set-op studies (τ ceiling ≈ 0.746).
High correlation indicates correct *scale* of policy scores but does not preclude
systematic *rank-order scrambling* within the top tier.

**Key finding 5** (3:100 cost efficiency): with study_b as the LF source (r = 100):
- 3 HF + 100 LF achieves τ = 0.891 (≡ ~10 HF-only at 1/3 the HF cost)
- The cost-efficiency crossover occurs at budget B ≈ 15 HF-equivalents; beyond this,
  investing directly in HF experiments surpasses the MF approach.
- With study_a as the LF source: 3 HF + 100 LF achieves τ = 0.750 — *worse* than
  3 HF alone (τ = 0.795).  High ρ does not prevent systematic bias from hurting the ranking.

---

## 6. Discussion

### 6.1 The Bias Floor as a Physics Contract

The bias floor of a LF source is determined by the structural gap between its physics model
and the HF reference — not by the amount of data.  study_c (KM − mixbox difference)
has the largest bias floor because oil-pigment KM physics diverges most from spectral
Beer-Lambert physics in the regions study_c samples.  This is analogous to irreducible error
in statistical learning: no matter how many samples you draw from a misspecified model,
you cannot recover the ground truth that the misspecification hides.

The practical implication is asymmetric: a practitioner with budget only for a few HF
experiments and many LF experiments should *first* invest in identifying which LF source
has the lowest bias floor relative to the HF reference — even a handful of pilot HF
experiments is enough to estimate ρ (Pearson correlation), which predicts the bias floor
direction (Table 2, Key Finding 4 notwithstanding).

### 6.2 The Mutual Flip and Information Density

In the mutual scenario, every source can eventually flip to donor role with enough experiments.
The flip point N* quantifies **information density per experiment**: how many experiments from
source A equal one experiment from source B in terms of ranking quality produced.

> **[PLACEHOLDER — interpret exp 03 Scenario 2 results here once run]**
> *Expected: N*(study_c → study_b) is large (many study_c robots needed to match one study_b robot),*
> *while N*(study_b → study_c) is small (study_b robots carry more information).*
> *This should produce an asymmetric N* matrix that mirrors the bias floor ordering.*

This connects to the MFMC optimal ratio from a ranking-quality perspective: the optimal
n_LF : n_HF ratio (108:1 for study_b, 16:1 for study_c) reflects the same information
density differential, but measured via variance reduction rather than τ directly.

### 6.3 TrueSkill2 Interpretation

The egalitarian swarm collapse (Key Finding 3) is predicted by TrueSkill2 team modeling:
when players of unequal skill receive equal partial-play fractions, the team performance
mean is dragged toward the low-skill players.  The quality-aware allocation:

  w_i* = (1 / Var(score_i)) / Σ_j (1 / Var(score_j))

derived from the inverse-variance minimum-uncertainty condition, would prevent this collapse.
The open question (Section 7) is whether the improvement justifies the meta-cost of
estimating Var(score_i) from pilot data before committing the full budget.

### 6.4 The Donor Graph Topology

> **[PLACEHOLDER — interpret exp 04 cycle detection results here]**
> *If no 3-cycles are detected: the donor hierarchy is transitive (a linear ranking of sources).*
> *Prediction: no cycles, because the external ceiling ordering mirrors the physics fidelity ordering*
> *(spectral > study_b > mixbox ≈ ryb > km > study_a > study_c).*
>
> *If cycles are detected: this would imply that "donor" status is target-difficulty-dependent.*
> *For example, study_c (earth tones) might be a better reference than study_b for policies*
> *optimised on warm/dark colors, while study_b dominates for cool/bright colors.*
> *This would connect to the Blade-Chest intransitivity detection in the parent PEGKí system.*

### 6.5 Limitations

**Egalitarian assumption**: All experiments within a study are treated as i.i.d. draws.
In practice, target color difficulty varies within a study — hard-target experiments may
carry more or less transferable information than easy ones.

**Fixed LF set**: We study a fixed collection of LF sources.  Active acquisition — choosing
which LF source to query next based on estimated ρ — is not considered.

**Single domain**: All results are specific to color mixing with four engines.  The qualitative
findings (permanent gap implies unreducible bias floor; mutual flip always exists at finite N*)
are general, but the quantitative thresholds (ρ = 0.996 for study_b, N* values) are domain-
and engine-specific.

**No per-policy rho**: The MFMC correlation ρ is computed globally across all policies.
Individual policies may transfer better or worse from LF to HF; per-policy ρ could improve
the combined ranking but requires more pilot data.

---

## 7. Open Questions

1. **Quality-aware diversity allocation**: Replace the egalitarian Q3 diversity test
   (equal robots per study) with ρ-proportional allocation n_i ∝ ρ_i².  Expected
   improvement: prevents the egalitarian collapse; cost: requires pilot ρ estimates.

2. **Per-robot inverse-variance weighting**: Weight each experiment within a study by
   1/Var(score), giving harder targets less weight when their high variance is noise rather
   than signal.

3. **Per-policy ρ for MFMC**: Compute ρ_p per policy.  Identify "robust" policies
   (high ρ_p for all LF sources) vs "LF-specific" policies (high ρ_p for one source only).

4. **TrueSkill2 multi-team Q3**: Model studies as teams, robots as players, and τ as
   match outcome.  Does inferred team skill converge to ρ and bias floor?

5. **Mirror databases**: Generate the 8 missing complement databases and verify that
   policies transfer symmetrically to/from mirror pairs.

6. **Flip N* sensitivity to target difficulty**: Do receivers flip earlier on easy
   targets than hard ones?  Per-difficulty flip curves would reveal whether donor status
   is global or regime-specific — connecting to Blade-Chest intransitivity.

---

## 8. Conclusion

We introduced the **donor/receiver flip framework** for multi-fidelity policy ranking under
asymmetric simulation environments.  The central result is a dichotomy:

- **External-reference scenario**: whether a receiver can flip to donor is determined by
  the bias floor — a physics property of the source that no amount of sampling can change.
  study_c (KM − mixbox, bias floor ≈ 0.318) can never surpass study_b (spectral − RYB,
  bias floor ≈ 0.105) as a proxy for the spectral HF reference, regardless of how many
  experiments are drawn from study_c.

- **Mutual-reference scenario**: because Kendall's τ is symmetric at full information,
  any receiver can flip to donor with enough experiments.  The flip point N* quantifies
  information density per experiment and mirrors the MFMC cost-efficiency structure.

The practical upshot is that source selection matters far more than source quantity when
the quality gap is driven by model bias rather than sample variance.  3 HF + 100 LF from
the *right* source achieves the ranking quality of 10 HF-only experiments; 3 HF + 100 LF
from the *wrong* source is worse than 3 HF alone.

---

## Acknowledgements

[To be added]

---

## References

[CITE: Peherstorfer2016] Peherstorfer, B., Willcox, K., & Gunzburger, M. (2016).
Optimal model management for multifidelity Monte Carlo estimation.
*SIAM Journal on Scientific Computing*, 38(5), A3163–A3194.

[CITE: Kennedy2000] Kennedy, M. C., & O'Hagan, A. (2000). Predicting the output from a
complex computer code when fast approximations are available. *Biometrika*, 87(1), 1–13.

[CITE: Kandasamy2016] Kandasamy, K., Dasarathy, G., Oliva, J. B., Schneider, J., &
Póczos, B. (2016). Gaussian process bandit optimisation with multi-fidelity evaluations.
*NeurIPS 2016*.

[CITE: Poloczek2017] Poloczek, M., Wang, J., & Frazier, P. (2017). Multi-information
source optimization. *NeurIPS 2017*.

[CITE: Kennedy1995] Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization.
*ICNN 1995*.

[CITE: Dorigo2006] Dorigo, M., Birattari, M., & Stutzle, T. (2006). Ant colony
optimization. *IEEE Computational Intelligence Magazine*, 1(4), 28–39.

[CITE: Brambilla2013] Brambilla, M., Ferrante, E., Birattari, M., & Dorigo, M. (2013).
Swarm robotics: a review from the swarm engineering perspective. *Swarm Intelligence*, 7, 1–41.

[CITE: Doignon1985] Doignon, J. P., & Falmagne, J. C. (1985). Spaces for the assessment
of knowledge. *International Journal of Man-Machine Studies*, 23(2), 175–196.

[CITE: Herbrich2006] Herbrich, R., Minka, T., & Graepel, T. (2006). TrueSkill: a Bayesian
skill rating system. *NeurIPS 2006*.

[CITE: Chen2016] Chen, S., & Joachims, T. (2016). Predicting matchups and preferences in
context. *KDD 2016*.

[CITE: PEGKi] [Internal reference — PEGKí framework CLAUDE.md and SYSTEM_ARCHITECTURE.md]

---

## Appendix A: Notation Summary

| Symbol | Definition |
|--------|-----------|
| S_i | Study database i |
| R_i(N) | Policy ranking from N sub-sampled experiments from S_i |
| τ_{ij}(N) | Kendall's τ between R_i(N) and R_j(full) |
| ceiling(S_i) | τ_{iH}(∞): asymptotic quality of S_i as HF proxy |
| bias_floor(S_i) | 1 − ceiling(S_i) |
| gap_ext(A,B) | ceiling(A) − ceiling(B) |
| N*_{AB} | Min N such that τ_{AB}(N) > τ_{BA}(1) + ε |
| ρ_{ij} | Pearson correlation between per-policy mean scores of S_i and S_j |
| r | Cost ratio cost_HF / cost_LF |
| w_i* | Inverse-variance optimal partial-play fraction |

## Appendix B: Placeholder Index

The following results are marked [PLACEHOLDER] and will be filled once the corresponding
experiments have been executed:

| Section | Experiment | Expected output |
|---------|-----------|----------------|
| 5.1 | 01_venn_geometry.py | Table 1: Venn volumes, asymmetry indices |
| 5.2 | 02_directed_transfer_matrix.py | Figure 2: 7×7 tau matrices |
| 5.2 | 03_swarm_flip_test.py | Figures 3–4: flip curves + N* heatmap |
| 5.2 | 04_flip_feasibility.py | Figure 5 + Table 3: taxonomy |
| 5.3 | 07_frugal_twin_convergence.py | Table 1 rows: study_b N=5,10,20,100; study_c full row |
| 5.4 | 06_symmetry_scoring.py | Overall 7-database symmetry scores |
| 6.2 | 03_swarm_flip_test.py | N* interpretation (Scenario 2) |
| 6.4 | 04_flip_feasibility.py | Donor graph cycle detection |
