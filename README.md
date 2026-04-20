# TENKi: Can More Low-Fidelity Sources Guide Budget-Limited Acquisition in Autonomous Materials Workflows?

**Version**: 0.4.0 (April 2026)  
**Parent framework**: PEGKi v0.2.0

---

## The Research Question

TENKi is a diagnostic for any workflow where the same decision problem can be
evaluated through multiple knowledge sources.

Imagine a team that can evaluate a design problem through:

- a trusted but expensive reference
- faster approximate models
- historical or simulated sources with partial coverage
- derived subsets that only cover part of the design space

The practical question is not simply which source correlates best. The real question is:

> **Which source is trustworthy enough to guide the next decision when budget is limited?**

TENKi answers that question through three linked sub-questions.

### Q1: Structural

How many asymmetric pieces make a symmetric whole?

Given study databases drawn from overlapping, source-exclusive, or missing regions
of the design space, what is the minimum collection that treats the knowledge-space
types fairly?

**Current answer**:

- A small number of sources achieves Knowledge-Space balance when each covers a distinct region type: overlap, transfer, and exclusion
- Full engine-permutation symmetry requires a larger collection
- A KS-balanced set is achievable without full permutation symmetry

### Q2: Empirical

How many cheap measurements are needed to match the ranking quality of the expensive reference?

**Current answer**:

- The best frugal proxy reaches strong ranking agreement with relatively few measurements
- All frugal sources hit a bias floor: beyond that point, more measurements do not remove the underlying physics gap
- An apparently favorable overlap region can still be a poor ranking donor

### Q3: Engineering

How should high-fidelity and low-fidelity budget be split?

**Current answer**:

- For ranking, a good low-fidelity source can help substantially
- For absolute scalar estimation, the classical paired control-variate branch is the cleaner choice
- At very low budget (N=10), concentrate on the single strongest donor
- At moderate budget (N=20), add one independent strong donor (e.g. mixbox+study_b)
- Quality-aware allocation consistently beats equal allocation; oracle single- or two-source selection sets the practical upper bound at low budgets

---

## How This Project Relates to PEGKi

PEGKi provides the broader source-transfer framework, benchmark databases, and
matched policy evaluations. TENKi asks a narrower question on top of those outputs:

> If a cheap source is available, should a materials scientist trust it to guide the next ranked
> decision, or is the source blocked by a physics gap that more data will not fix?

In this repository, that question is answered with three linked pieces:

- set-based geometry, which explains why different sources cover different parts of design space
- donor/receiver transfer tests, which show who helps whom and where the ceiling sits
- budgeted allocation tests, which show whether more cheap data actually improves decisions

TENKi is therefore best read as a diagnostic extension built on PEGKi data, not as a separate
parent framework.

---

## What TENKi Needs

To use TENKi on another autonomous workflow, the inputs are:

| Input | General meaning |
|------|------|
| Reference source | The source whose ranking or scalar estimate is trusted most |
| Candidate sources | Cheaper, faster, simulated, historical, or partial-coverage sources |
| Shared decision policies | The optimizers, agents, heuristics, or acquisition rules being compared |
| Replicate observations | Enough repeated evaluations to estimate ranking quality as budget grows |
| Source metadata | Whether sources are paired with the reference, unpaired, overlapping, source-exclusive, or coverage-limited |

TENKi then estimates:

- which sources are donors, receivers, or permanent-gap sources
- how ranking quality changes with measurement budget
- when a cheap source can substitute for the reference
- when paired classical MFMC is appropriate for scalar estimation
- whether a fixed budget should be concentrated or split across donors

The color-mixing benchmark in this repository is the worked example, not the
only intended use case. TENKi-1000 instantiates the general setup with nine
sources: four paired pigment-mixing modalities and five set-operation source
regions. Those sources are useful because they expose directional coverage
asymmetry under controlled conditions, but the diagnostic is meant to transfer
to materials workflows with simulations, experiments, historical archives, or
remote autonomous-lab data.

---

## Experiments

| Experiment | Question |
|------|------|
| `01_venn_geometry.py` | How do the source coverage regions overlap and differ geometrically? |
| `02_directed_transfer_matrix.py` | Who donates ranking signal to whom at low budget? |
| `03_swarm_flip_test.py` | Can a receiver flip into a donor if given more budget? |
| `04_flip_feasibility.py` | Which gaps are permanent, flippable, or already donor-favoring? |
| `05_study_comparison.py` | How different are the study databases as sampled distributions? |
| `06_symmetry_scoring.py` | Which study combinations satisfy structural balance criteria? |
| `07_frugal_twin_convergence.py` | Does quantity beat diversity under fixed budget? |
| `08_multifidelity_allocation.py` | How do classical and hybrid multi-fidelity allocation compare? |
| `15_q1_mirror_pair_test.py` | Are mirror-pair (forward/reverse set-op) databases direction-asymmetric? |
| `16_q2_epsilon_symmetry.py` | What epsilon-volume and epsilon-quality balance is achievable by source subset selection? |
| `17_q3_nash_equilibrium.py` | What is the Nash-optimal fixed-budget source allocation at varying N? |
| `18_q4_diversity_allocation.py` | Does quality-aware allocation beat equal weighting across sources? |
| `19_q5_robot_difficulty.py` | Does source usefulness vary with target difficulty (easy/hard)? |
| `20_q6_per_policy_rho.py` | Which policies have robust vs fragile LF/HF correlation for MFMC? |
| `21_q7_trueskill2.py` | Does TrueSkill2 multi-team rating reproduce the TENKi donor taxonomy? |
| `22_q9_flip_difficulty.py` | Is flip N* sensitive to target difficulty tertile? |
| `23_q10_swarm_specialists.py` | Does swarm (kNN-local) weighting beat equal ensemble without spatial specialists? |

---

## Quick Start

```bash
git clone https://github.com/Sparks-Ackolf-HOLAR-Autonomous-Labs/TENKi.git
cd TENKi
uv sync

# For the included color-mixing example, point to a PEGKi output directory
# that contains the TENKi-1000 source databases
# (see https://github.com/Sparks-Ackolf-HOLAR-Autonomous-Labs/PEGKi)
export PEGKI_OUTPUT=/path/to/PEGKi/output

uv run python experiments/01_venn_geometry.py
uv run python experiments/02_directed_transfer_matrix.py
uv run python experiments/03_swarm_flip_test.py
uv run python experiments/04_flip_feasibility.py
uv run python experiments/05_study_comparison.py
uv run python experiments/06_symmetry_scoring.py
uv run python experiments/07_frugal_twin_convergence.py
uv run python experiments/08_multifidelity_allocation.py

# TENKi-1000 follow-up diagnostics (Q1-Q10, Experiments 15-23)
bash run_open_questions.sh
# or individually, e.g.:
uv run python experiments/15_q1_mirror_pair_test.py
uv run python experiments/17_q3_nash_equilibrium.py
```

---

## Documentation

- [`docs/TECHNICAL_REFERENCE.md`](docs/TECHNICAL_REFERENCE.md): technical derivations, saved-result interpretation, and decision rule

