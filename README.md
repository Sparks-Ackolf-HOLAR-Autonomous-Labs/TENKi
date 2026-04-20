# TENKi: Can More Low-Fidelity Sources Guide Budget-Limited Acquisition in Autonomous Materials Workflows?

**Version**: 0.3.0 (March 2026)  
**Parent framework**: PEGKi v0.2.0

---

## The Research Question

This project is easiest to read as one materials-workflow story.

Imagine a team that can evaluate the same design problem through several knowledge sources:

- a trusted but expensive reference
- faster approximate models
- derived subsets that only cover part of the design space

The practical question is not simply which source correlates best. The real question is:

> **Which source is trustworthy enough to guide the next decision when budget is limited?**

TenKi answers that question through three linked sub-questions.

### Q1: Structural

How many asymmetric pieces make a symmetric whole?

Given study databases drawn from intersections, differences, and complements of the engine
gamuts, what is the minimum collection that treats the knowledge-space types fairly?

**Current answer**:

- `K = 3` achieves Knowledge-Space balance: one overlap source, one transfer source, one
  exclusion source
- `K = 15` is required for full engine-permutation symmetry with four engines
- the current studies A, B, and C are KS-balanced but not engine-permutation symmetric

### Q2: Empirical

How many cheap measurements are needed to match the ranking quality of the expensive reference?

**Current answer**:

- `study_b` is the best unpaired frugal proxy and reaches `tau >= 0.80` in about 10 measurements
- all frugal sources hit a bias floor: beyond that point, more measurements do not remove the
  underlying physics gap
- `study_a` is the clearest warning case: an apparently favorable overlap region can still be a
  poor ranking donor

### Q3: Engineering

How should high-fidelity and low-fidelity budget be split?

**Current answer**:

- for ranking, a good low-fidelity source can help substantially
- for absolute scalar estimation, the classical paired control-variate branch is the cleaner choice
- concentration on the best source beats naive diversity under fixed budget

---

## How This Project Relates to PEGKi

PEGKi provides the broader source-transfer framework, the benchmark databases, and the matched
policy evaluations. TenKi asks a narrower question on top of those outputs:

> If a cheap source is available, should a materials scientist trust it to guide the next ranked
> decision, or is the source blocked by a physics gap that more data will not fix?

In this repository, that question is answered with three linked pieces:

- set-based geometry, which explains why different sources cover different parts of design space
- donor/receiver transfer tests, which show who helps whom and where the ceiling sits
- budgeted allocation tests, which show whether more cheap data actually improves decisions

TenKi is therefore best read as a diagnostic extension built on PEGKi data, not as a separate
parent framework.

---

## Three Studies in the Current Database

| Study | Set-op | KS type | Bias floor vs spectral |
|------|------|------|------|
| **A** | `mixbox ∩ RYB` | `K_H_OVERLAP` | 0.249 |
| **B** | `spectral \ RYB` | `K_T` | 0.105 |
| **C** | `KM \ mixbox` | `K_E` | 0.318 |

Interpretation:

- Study A is an overlap source, but overlap does not mean neutral or trustworthy
- Study B is the strongest frugal donor in the current benchmark
- Study C is the most divergent and carries the largest persistent gap

---

## Experiments

| Experiment | Question |
|------|------|
| `01_venn_geometry.py` | How do the engine gamuts overlap and differ geometrically? |
| `02_directed_transfer_matrix.py` | Who donates ranking signal to whom at low budget? |
| `03_swarm_flip_test.py` | Can a receiver flip into a donor if given more budget? |
| `04_flip_feasibility.py` | Which gaps are permanent, flippable, or already donor-favoring? |
| `05_study_comparison.py` | How different are the study databases as sampled distributions? |
| `06_symmetry_scoring.py` | Which study combinations satisfy structural balance criteria? |
| `07_frugal_twin_convergence.py` | Does quantity beat diversity under fixed budget? |
| `08_multifidelity_allocation.py` | How do classical and hybrid multi-fidelity allocation compare? |

---

## Quick Start

```bash
git clone https://github.com/Sparks-Ackolf-HOLAR-Autonomous-Labs/TENKi.git
cd TENKi
uv sync

# Point to a PEGKi output directory that contains the seven benchmark databases
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
```

---

## Documentation

- [`docs/TECHNICAL_REFERENCE.md`](docs/TECHNICAL_REFERENCE.md): technical derivations, saved-result interpretation, and decision rule

