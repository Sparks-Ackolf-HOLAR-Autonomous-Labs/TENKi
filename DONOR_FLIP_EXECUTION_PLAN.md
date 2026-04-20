# Donor-Flip Execution Plan

## Goal

Answer the project question in a form that is executable in code:

> Given two asymmetric data sources, if source `A` is currently the receiver and source `B` is currently the donor, can increasing the amount of lower-fidelity data from `A` flip the sign of the asymmetry so that `A` becomes the donor? If yes, how many `A` experiments are required?

This plan treats that as a ranking-transfer problem, not a raw regression problem. The output object is a directional donor/receiver map plus a flip-count `N*`.

## PEGKi grounding

The extension should stay aligned with the parent PEGKi framework, not invent a new vocabulary.

- PEGKi Conjecture 4.2.1 says transfer asymmetry is directional and should be explained by coverage mismatch, calibration basis, and rank-geometry mismatch.
- PEGKi Conjecture 4.2.2 says some material knowledge is irreducible across fidelities, which is exactly the reason some external flips are impossible.
- PEGKi Conjecture 4.2.4 says mixed-source solutions can be optimal under shift, which justifies the swarm/ensemble framing once single-source flips are characterized.
- PEGKi transfer analysis already uses ranking-based directional signals, not just point prediction.
- PEGKi set-operation diagnostics and the policy-proximity metric explain whether a difference/intersection study behaves like one parent source, the other parent source, or an incoherent extrapolative domain.

Implication for this project:

1. `donor` means "higher information per experiment for reproducing another source's ranking".
2. `receiver` means "needs more experiments to achieve the same ranking fidelity".
3. `impossible flip` is not a sampling failure; it is a PEGKi-style irreducibility result.

## Decision target

The code should report two different flip notions. They answer different questions and must not be conflated.

### 1. External-reference flip

Reference is fixed to the higher-fidelity source `H` (currently `spectral`).

- `tau_AH(N) = KendallTau(rank(A from N experiments), rank(H from full data))`
- `tau_BH(full) = KendallTau(rank(B from full data), rank(H from full data))`

`A` flips over `B` at:

```text
N*_ext(A over B | H) = min N such that tau_AH(N) > tau_BH(full) + eps
```

Interpretation:

- If `ceiling_AH <= ceiling_BH + eps`, the flip is impossible.
- If `ceiling_AH > ceiling_BH + eps`, the flip is possible and `N*_ext` is meaningful.

This is the main answer to the user's question when "higher fidelity" is the ground truth.

### 2. Mutual-reference flip

Each source is treated as the other's reference.

- `tau_AB(N) = KendallTau(rank(A from N experiments), rank(B from full data))`
- `tau_BA(1) = KendallTau(rank(B from 1 experiment), rank(A from full data))`

`A` flips at:

```text
N*_mut(A over B) = min N such that tau_AB(N) > tau_BA(1) + eps
```

Interpretation:

- This measures how many `A` experiments are needed to beat `B`'s one-shot donor power.
- Because the full-data ceiling is shared, this flip should exist for non-degenerate pairs if the scan range is large enough.

## What the current code already does

The current project already contains the right skeleton:

- `experiments/02_directed_transfer_matrix.py` computes the directional transfer matrix at small `N`.
- `experiments/03_swarm_flip_test.py` computes external and mutual flip curves.
- `experiments/04_flip_feasibility.py` turns those into donor/receiver taxonomy and centrality summaries.
- `experiments/07_frugal_twin_convergence.py` estimates the bias floor and swarm convergence against the high-fidelity source.

The gap is not the basic idea. The gap is that the implementation is still script-first and needs to be turned into a clearer execution pipeline with explicit contracts, reusable helpers, and outputs that answer the donor-flip question directly.

## Execution scope

The implementation should be split into four layers.

### Layer A. Data loading and score normalization

Create a reusable module:

- `analysis/flip_data.py`

Responsibilities:

- Load per-policy per-experiment scores from each database.
- Intersect common policy sets across compared sources.
- Build deterministic full-data rankings.
- Expose metadata such as `max_experiments_per_policy`.

Proposed API:

```python
from dataclasses import dataclass

@dataclass
class StudyScores:
    name: str
    db_path: str
    policy_scores: dict[str, list[float]]
    full_rank: list[str]
    n_policies: int
    max_n: int

def load_study_scores(db_path: str, score_key: str = "best_color_distance_mean") -> StudyScores: ...
def load_many_studies(study_map: dict[str, str]) -> dict[str, StudyScores]: ...
def common_policy_subset(studies: dict[str, StudyScores], selected: list[str] | None = None) -> list[str]: ...
def restrict_to_common_policies(study: StudyScores, common: list[str]) -> StudyScores: ...
```

Notes:

- Keep the score direction explicit. Current code assumes lower `best_color_distance_mean` is better.
- Make `score_key` configurable so future PEGKi-style composite scores can be substituted.

### Layer B. Ranking and bootstrap core

Create:

- `analysis/flip_metrics.py`

Responsibilities:

- Ranking from `N` sampled experiments.
- Kendall tau and optional Spearman rho.
- Bootstrap confidence intervals.
- Ceiling estimation.

Proposed API:

```python
def rank_from_sample(
    study: StudyScores,
    policies: list[str],
    n: int,
    rng: np.random.Generator,
    replace: bool = True,
) -> list[str]: ...

def kendall_tau(rank_a: list[str], rank_b: list[str]) -> float: ...

def bootstrap_tau_curve(
    source: StudyScores,
    reference_rank: list[str],
    policies: list[str],
    n_values: list[int],
    n_bootstrap: int,
    rng_seed: int,
) -> dict[int, dict[str, float]]: ...

def full_data_ceiling(
    source: StudyScores,
    reference: StudyScores,
    policies: list[str],
) -> float: ...
```

Each `bootstrap_tau_curve` entry should include:

```json
{
  "mean_tau": 0.829,
  "std_tau": 0.082,
  "p05_tau": 0.690,
  "p50_tau": 0.835,
  "p95_tau": 0.931,
  "n_bootstrap": 300
}
```

### Layer C. Donor-flip logic

Create:

- `analysis/flip_models.py`

Responsibilities:

- Compute directional asymmetry at `N=1`.
- Compute external flip feasibility and `N*`.
- Compute mutual flip feasibility and `N*`.
- Classify `PERMANENT_GAP`, `FLIPPABLE`, `ALREADY_DONOR`, `UNRESOLVED_IN_RANGE`.

Proposed API:

```python
from dataclasses import dataclass

@dataclass
class FlipResult:
    source: str
    competitor: str
    reference: str
    mode: str  # "external" | "mutual"
    tau_source_at_1: float
    tau_competitor_at_1: float | None
    ceiling_source: float
    ceiling_competitor: float
    gap_now: float
    gap_ceiling: float
    flip_possible: bool
    flip_n: int | None
    verdict: str

def asymmetry_at_n1(
    studies: dict[str, StudyScores],
    n_bootstrap: int,
    rng_seed: int,
) -> dict[tuple[str, str], float]: ...

def external_flip_result(
    source: StudyScores,
    competitor: StudyScores,
    hifi: StudyScores,
    policies: list[str],
    n_values: list[int],
    n_bootstrap: int,
    eps: float,
    rng_seed: int,
) -> FlipResult: ...

def mutual_flip_result(
    source: StudyScores,
    competitor: StudyScores,
    policies: list[str],
    n_values: list[int],
    n_bootstrap: int,
    eps: float,
    rng_seed: int,
) -> FlipResult: ...
```

Required verdict semantics:

- `ALREADY_DONOR`: source is already ahead at `N=1`.
- `FLIPPABLE`: source starts behind but crosses within tested `N`.
- `PERMANENT_GAP`: source ceiling is below competitor ceiling in external mode.
- `UNRESOLVED_IN_RANGE`: source ceiling allows a flip but tested `N_values` did not reach it.
- `NEAR_SYMMETRIC`: absolute gap below `eps`.

### Layer D. Reports and plots

Create:

- `analysis/flip_reports.py`

Responsibilities:

- Heatmaps for `tau_ij(N)`.
- Heatmap for `A[i,j] = tau_ij(1) - tau_ji(1)`.
- External flip curves vs the HF reference.
- Mutual `N*` heatmap.
- Summary tables in markdown and JSON.

Proposed outputs:

- `results/transfer_matrix.json`
- `results/transfer_asymmetry_N1.png`
- `results/flip_test_summary.json`
- `results/flip_feasibility.json`
- `results/flip_external_all.png`
- `results/flip_crossover_heatmap.png`
- `results/flip_summary.md`

## Script-level execution plan

Keep the current experiment numbering, but make each script a thin CLI wrapper over the reusable modules.

### Step 1. Refactor Experiment 02

Target file:

- `experiments/02_directed_transfer_matrix.py`

Changes needed:

- Replace local helper duplication with imports from `analysis/flip_data.py` and `analysis/flip_metrics.py`.
- Add CLI flags:
  - `--studies`
  - `--n-values`
  - `--n-bootstrap`
  - `--score-key`
  - `--seed`
- Save both mean tau and bootstrap spread, not just the mean matrix.

JSON contract:

```json
{
  "studies": ["spectral", "mixbox", "km", "ryb", "study_a", "study_b", "study_c"],
  "n_values": [1, 5, 10, "full"],
  "matrices": {
    "1": [[...]],
    "5": [[...]],
    "10": [[...]],
    "full": [[...]]
  },
  "matrix_std": {
    "1": [[...]],
    "5": [[...]],
    "10": [[...]]
  },
  "asymmetry_at_n1": [[...]],
  "donor_score_n1": {
    "study_b": 0.18
  }
}
```

### Step 2. Refactor Experiment 03 into the main answer script

Target file:

- `experiments/03_swarm_flip_test.py`

This becomes the primary execution script for the donor-flip question.

Changes needed:

- Use `FlipResult` objects from `analysis/flip_models.py`.
- Separate external and mutual results cleanly.
- Write a compact markdown summary in addition to JSON.
- Add scan extension logic:
  - if `ceiling_source > ceiling_competitor + eps` but no flip was found in current range, extend `N_values` up to the source's max available experiments before declaring `UNRESOLVED_IN_RANGE`.

CLI flags:

- `--hifi spectral`
- `--frugal study_a study_b study_c mixbox ryb km`
- `--n-values 1 2 3 5 8 10 15 20 30 50 100`
- `--n-bootstrap 300`
- `--flip-eps 0.01`
- `--score-key best_color_distance_mean`
- `--seed 42`

Required markdown summary sections:

- donor/receiver ordering at `N=1`
- external permanent-gap pairs
- external flippable pairs with `N*`
- mutual flip matrix
- top candidates for cheap-source replacement of a stronger donor

### Step 3. Refactor Experiment 04 into taxonomy only

Target file:

- `experiments/04_flip_feasibility.py`

Role after refactor:

- consume `transfer_matrix.json` and `flip_test_summary.json`
- compute centrality and donor topology
- detect cycles
- produce concise taxonomy artifacts

Add PEGKi-aligned fields:

- `external_ceiling`
- `mutual_gap_at_n1`
- `policy_proximity_class` if available from parent outputs
- `physics_class` and `ks_type` metadata if available

This is where the PEGKi interpretation layer belongs:

- directionality due to fidelity / calibration basis
- impossible flips as irreducibility
- cycles as Blade-Chest-relevant intransitivity candidates

### Step 4. Upgrade Experiment 07 for swarm allocation

Target file:

- `experiments/07_frugal_twin_convergence.py`

Changes needed:

- Keep current single-study convergence curve.
- Add quality-aware swarm mode:
  - `equal`
  - `rho_squared`
  - `inverse_variance`
- Make the multi-study swarm output comparable to single-source flip output.

New helper:

```python
def combine_scores_weighted(
    score_dicts: list[StudyScores],
    weights: dict[str, float],
    n_allocations: dict[str, int],
    rng: np.random.Generator,
) -> list[str]: ...
```

This is needed because the user question explicitly allows a swarm or ensemble, not just one low-fidelity source.

## PEGKi integration points

These should be treated as optional enrichments in phase 2, not blockers for the core donor-flip pipeline.

### Integration A. Parent transfer score as a second metric

Current gamut-symmetry scripts use Kendall tau on policy rankings. That should remain the primary metric.

Add an optional secondary metric:

- PEGKi transfer score `alpha(K_s -> K_t)`

Reason:

- Tau answers ranking agreement.
- PEGKi `alpha` adds structural compatibility, source reliability, mapping efficiency, and transfer overhead.

Implementation:

- Add `--metric tau|pegki_alpha|both`.
- If `pegki_alpha` is selected, load parent project outputs or reuse the parent transfer-equation code path.
- Store both metrics side by side, never merge them silently.

### Integration B. Policy-proximity annotation for set-operation studies

For any pair involving a set-operation study, annotate the flip result with:

- `mean_t`
- `std_t`
- coherence class
- whether the study behaves like parent A, parent B, or extrapolates beyond both

Reason:

- This explains why a source is a poor donor.
- It also stops the project from over-interpreting high `rho` or high overlap as useful transfer.

### Integration C. Phase-3 mixed-source routing

Once single-source flips are stable, add a second script:

- `experiments/09_mixed_source_flip.py`

Question:

- Can a weighted mixture of low-fidelity sources flip against a stronger donor when no single low-fidelity source can?

This is where PEGKi Conjecture 4.2.4 becomes operational.

Inputs:

- source pool
- weights or robot allocations
- HF reference

Outputs:

- `N*_mix`
- best source mixture under a total LF budget
- comparison against best single LF source

## File plan

Recommended files to add or refactor:

- `analysis/__init__.py`
- `analysis/flip_data.py`
- `analysis/flip_metrics.py`
- `analysis/flip_models.py`
- `analysis/flip_reports.py`
- `experiments/02_directed_transfer_matrix.py`
- `experiments/03_swarm_flip_test.py`
- `experiments/04_flip_feasibility.py`
- `experiments/07_frugal_twin_convergence.py`
- `experiments/09_mixed_source_flip.py`

## Data and config contract

Create one shared config object instead of hard-coded constants spread across scripts.

Example:

```python
from dataclasses import dataclass

@dataclass
class FlipConfig:
    studies: dict[str, str]
    hifi: str = "spectral"
    n_values: tuple[int, ...] = (1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 100)
    n_bootstrap: int = 300
    flip_eps: float = 0.01
    score_key: str = "best_color_distance_mean"
    seed: int = 42
```

Store resolved run metadata in every JSON artifact:

- timestamp
- config
- common policy count
- max available `N` per study
- missing studies

## Acceptance criteria

The code is complete when the following are true.

1. For every ordered pair `(A, B)` and chosen HF reference `H`, the pipeline reports whether `A` can externally flip over `B`, and if so gives `N*_ext`.
2. For every ordered pair `(A, B)`, the pipeline reports `N*_mut(A over B)`.
3. Results are reproducible from a fixed random seed.
4. Every output has both machine-readable JSON and a human-readable markdown summary.
5. The scripts no longer duplicate score loading, ranking, or bootstrap logic.
6. The summary explicitly distinguishes `sampling-limited` from `physics-limited` failures.
7. Optional PEGKi annotations can be added without changing the core tau-based logic.

## Recommended implementation order

1. Build `analysis/flip_data.py` and `analysis/flip_metrics.py`.
2. Refactor `02_directed_transfer_matrix.py` to use them.
3. Build `analysis/flip_models.py`.
4. Refactor `03_swarm_flip_test.py` so it becomes the main donor-flip runner.
5. Refactor `04_flip_feasibility.py` into a post-processing taxonomy step.
6. Upgrade `07_frugal_twin_convergence.py` for weighted swarm allocation.
7. Add PEGKi transfer-score and policy-proximity annotations.
8. Add `09_mixed_source_flip.py` for mixed-source donor flips.

## Bottom line

The correct code objective is not just "plot more tau curves". It is:

- estimate the directional asymmetry at low sample count
- estimate the full-data ceiling for each source relative to the HF target
- separate impossible flips from finite-sample flips
- report the minimum low-fidelity swarm size needed to reverse donor status
- explain the result using PEGKi's directional-transfer and irreducibility concepts

That is the implementation that directly answers the project question.
