# CLAUDE.md — extended/gamut_symmetry

This directory implements the donor-flip transfer analysis pipeline and the async
multi-fidelity optimizer built on top of the PEGKi framework.

---

## Directory Layout

```
gamut_symmetry/
  analysis/           Core data loading and flip metrics
  experiments/        Numbered experiment scripts (02–13)
  optimizer/          Async MF optimizer package (7 allocation methods)
  results/            Output JSON, PNG, JSONL from experiments
  domains/            Domain definitions for multi-domain experiments
  writing/            Paper drafts and notes
  REAL_ASYNC_PEGKI_MF_OPTIMIZER_PLAN.md   Full design spec for the optimizer
  DONOR_FLIP_EXECUTION_PLAN.md            Experiment roadmap
```

---

## Analysis Package (`analysis/`)

| Module | Purpose |
|---|---|
| `flip_data.py` | Load per-policy per-experiment scores from PEGKi DBs; `StudyScores`, `load_study_scores`, `load_many_studies`, `load_trial_records` |
| `flip_metrics.py` | Kendall tau, donor-flip verdicts (PERMANENT_GAP / FLIPPABLE), tau-vs-N curves |
| `flip_models.py` | Statistical models for flip probability estimation |
| `flip_reports.py` | Text/markdown report generation |
| `adapters.py` | Bridge between analysis types and PEGKi database schema |

**Key data structure — `StudyScores`**:
```python
StudyScores(
    name: str,
    db_path: str,
    policy_scores: dict[str, list[float]],  # policy -> per-experiment scores
    full_rank: list[str],                    # policies sorted best->worst
    n_policies: int,
    max_n: int,                              # min experiments across all policies
)
```

`policy_scores[policy]` contains one `best_color_distance_mean` value per experiment,
read from `experiment_*/summary.json` → `policy_stats.best_color_distance_mean`.

---

## Experiments (`experiments/`)

All scripts run from the repo root with `uv run python experiments/NN_name.py`.
Unicode-safe (ASCII-only print statements; set `PYTHONIOENCODING=utf-8` on Windows if needed).

| Script | Topic | Key output |
|---|---|---|
| `02_directed_transfer_matrix.py` | Pairwise tau(A→B) transfer matrix at N=1,3,5,full | `results/transfer_matrix.json` |
| `03_` ... `11_` | See `DONOR_FLIP_EXECUTION_PLAN.md` for full list | `results/*.json` / `*.png` |
| `12_ensemble_vs_swarm.py` | Ensemble (equal weights) vs swarm (local kNN weights) aggregation | `results/ensemble_vs_swarm.json` |
| `13_async_mf_optimizer.py` | Async MF optimizer entry point (all 7 policy modes) | `results/async_mf_optimizer_*.json/png` |

### Running experiment 12

```bash
# Default (104 targets, N=[1,5,10,50,100], K=3 sources)
uv run python experiments/12_ensemble_vs_swarm.py

# With custom output dir and N values including bootstrap
uv run python experiments/12_ensemble_vs_swarm.py \
    --n-values 1 5 10 50 100 500 1000 \
    --output-dir results/exp12_1000targets

# Provide explicit study databases
uv run python experiments/12_ensemble_vs_swarm.py \
    --studies mixbox=output/db_mixbox km=output/db_km ryb=output/db_ryb
```

**Ensemble vs Swarm distinction**:
- **Ensemble**: weight = 1/K for every source at every target (global, no spatial awareness)
- **Swarm**: weight = 1 / mean_error(source in kNN(target)) — local specialists get higher weight near targets they handle well

### Running experiment 13

```bash
# Default (ensemble_mf, spectral HF, mixbox+km LF, budget=50)
uv run python experiments/13_async_mf_optimizer.py

# Choose policy mode
uv run python experiments/13_async_mf_optimizer.py --policy-mode mfbo
uv run python experiments/13_async_mf_optimizer.py --policy-mode smac
uv run python experiments/13_async_mf_optimizer.py --policy-mode hyperband
uv run python experiments/13_async_mf_optimizer.py --policy-mode mfmc
uv run python experiments/13_async_mf_optimizer.py --policy-mode swarm_mf
uv run python experiments/13_async_mf_optimizer.py --policy-mode single_source_mf

# With real study databases
uv run python experiments/13_async_mf_optimizer.py \
    --hifi spectral \
    --lf-sources mixbox km \
    --study spectral=output/db_spectral \
    --study mixbox=output/db_mixbox \
    --study km=output/db_km \
    --budget-total 200

# With fidelity-DB map (separate databases per fidelity level)
# Requires generating DBs at different --rounds values first (see below)
uv run python experiments/13_async_mf_optimizer.py \
    --policy-mode mfbo \
    --fidelity-db spectral:3=output/db_spectral_r3 \
    --fidelity-db spectral:12=output/db_spectral_r12 \
    --fidelity-levels 3 12 \
    --budget-total 500
```

---

## Optimizer Package (`optimizer/`)

A backend-agnostic async multi-fidelity optimizer. The scheduler and policies are
identical whether running offline (simulated time) or real async (future work).
Only the executor knows which mode is active.

### Modules

| Module | Class(es) | Purpose |
|---|---|---|
| `types.py` | `Config`, `RunState`, `WorkerState`, `Suggestion`, `CompletedEval`, `IntermediateState`, `BeliefState` | Core dataclasses |
| `runtime_model.py` | `ConstantRuntimeModel`, `SourceFidelityRuntimeModel`, `EmpiricalRuntimeModel` | Runtime estimators for simulated clock |
| `objective.py` | `OfflineReplayObjective` | Replays PEGKi study DBs; no live execution |
| `executor.py` | `LocalSimExecutor` | Min-heap offline simulator; never sleeps |
| `store.py` | `RunStore` | JSON snapshots + JSONL append-only persistence |
| `pegki_bridge.py` | `compute_implied_rank`, `update_tau_rho_beliefs`, `update_bias_floor_estimates` | PEGKi tau/rho belief updates |
| `policies.py` | 7 policy classes (see table below) | Job suggestion / allocation logic |
| `scheduler.py` | `AsyncMFScheduler` | Event loop: dispatch → advance time → poll → update beliefs → snapshot |

### Policy Modes

| `--policy-mode` | Class | Method | Surrogate | Fidelity strategy |
|---|---|---|---|---|
| `ensemble_mf` | `EnsembleMFPolicy` | Bandit | None | (tau × source_weight) / runtime; global weights |
| `swarm_mf` | `SwarmMFPolicy` | UCB bandit | None | (tau + explore/sqrt(n)) / runtime; per-source UCB |
| `single_source_mf` | `SingleSourceMFPolicy` | Deterministic sweep | None | All LF fidelities first, then HF at max fidelity |
| `mfbo` | `MFBOPolicy` | Bayesian optimization | GP (Matern-2.5, sklearn) | LCB or EI / runtime; round-robin warm-up then GP |
| `smac` | `SMACPolicy` | SMAC-style | Random forest (sklearn) | LCB with per-tree std / runtime |
| `hyperband` | `HyperbandPolicy` | Successive halving | None | Evaluate all at low fidelity; keep top 1/eta; promote |
| `mfmc` | `MFMCPolicy` | MFMC optimal allocation | None | Warm-up → tau-derived LF/HF ratio (Peherstorfer 2016) |

`mfbo` and `smac` require sklearn (already in dependencies). `mfbo` optionally uses
scipy for EI acquisition (`--acquisition ei`).

### Fidelity Configuration

**Two modes for fidelity:**

**Mode A — Statistical replication (default, no extra data needed)**:
`--fidelity-levels 1 3 5` means "sample 1, 3, or 5 experiment-level scores from the
pool and average them." Higher fidelity = lower variance estimate. Works with any
existing single-round database.

**Mode B — Separate databases per fidelity level (real MF)**:
Generate databases with different `--rounds` settings, then register them:
```bash
# Step 1: generate low-fidelity DB (3 rounds)
uv run python ../../scripts/generate_policy_data.py \
    --engine spectral --rounds 3 --experiments 5 \
    --output output/db_spectral_r3

# Step 2: generate high-fidelity DB (12 rounds)
uv run python ../../scripts/generate_policy_data.py \
    --engine spectral --rounds 12 --experiments 5 \
    --output output/db_spectral_r12

# Step 3: run optimizer with fidelity-DB map
uv run python experiments/13_async_mf_optimizer.py \
    --policy-mode mfbo \
    --fidelity-db spectral:3=output/db_spectral_r3 \
    --fidelity-db spectral:12=output/db_spectral_r12 \
    --fidelity-levels 3 12
```

In Mode B, each fidelity level uses ALL experiments from its DB (not statistical sampling).
The `fidelity` value in `Suggestion` selects which DB to load, not how many items to draw.

### Offline Simulation Contract

- `LocalSimExecutor` advances a simulated clock via a min-heap of completion events
- `submit(suggestion, current_time)` pushes `(current_time + expected_runtime, job_id, ...)`
- `poll_completed(current_time)` pops all entries with completion_time <= current_time
- No sleeping, no threads, no real processes
- Swap executor for `ThreadExecutor` / `ProcessExecutor` later without changing scheduler or policies

### RunStore Layout

```
runs/<run_name>/
  config.json
  latest_snapshot.json
  beliefs.json
  completed.jsonl          # append-only; one CompletedEval per line
  snapshots/
    state_00001.json
    state_00002.json
    ...
  intermediate/
    <resume_token>.json    # resumable fidelity promotions
  coordination/
    events.jsonl           # append-only; job_leased / job_completed events
```

### Belief Updates (PEGKi bridge)

After each completed eval the scheduler calls:

1. `update_tau_rho_beliefs`: EMA of Kendall tau comparing LF ranking to HF ranking at each fidelity
   - `tau_mean = (1 - alpha) * tau_mean + alpha * tau_observed`
   - Stored per `(source_name, fidelity)` in `BeliefState`
2. `update_bias_floor_estimates`: `bias_floor = 1 - tau_at_max_fidelity`
   - How wrong the LF source is even at its best fidelity

### Quick API Example

```python
from optimizer.types import Config, RunState, WorkerState, BeliefState
from optimizer.runtime_model import SourceFidelityRuntimeModel
from optimizer.objective import OfflineReplayObjective
from optimizer.executor import LocalSimExecutor
from optimizer.store import RunStore
from optimizer.policies import MFBOPolicy
from optimizer.scheduler import AsyncMFScheduler

cfg = Config(
    run_name="my_run",
    hifi_source="spectral",
    lf_sources=["mixbox", "km"],
    policy_mode="mfbo",
    fidelity_levels=[1, 3, 5],
    budget_total=100.0,
    max_workers=4,
)

objective = OfflineReplayObjective(
    study_map={"spectral": "output/db_spectral", "mixbox": "output/db_mixbox", "km": "output/db_km"},
    hifi_source="spectral",
)
policies = objective.all_policies()
hf_rank  = objective.hf_rank()

rt_model = SourceFidelityRuntimeModel(
    source_multipliers={"spectral": 3.0, "mixbox": 1.0, "km": 1.0},
    fidelity_scale=0.1,
)
policy   = MFBOPolicy(sources=["spectral","mixbox","km"], policies=policies,
                       fidelity_levels=[1,3,5], runtime_model=rt_model)
executor = LocalSimExecutor(objective=objective)
store    = RunStore(save_dir="runs", run_name="my_run")
run_state = RunState(
    workers=[WorkerState(worker_id=f"w{i}") for i in range(4)],
    beliefs=[BeliefState(source_name=s, tau_mean=0.5, quality_score=0.5)
             for s in ["spectral","mixbox","km"]],
)

scheduler = AsyncMFScheduler(cfg, run_state, executor, objective, policy, store, hf_rank)
scheduler.run_until_budget(snapshot_interval=10)
store.close()
```

---

## Data Generation for Cross-Engine / Set-Op Studies

These commands are run from the **repo root** (`color_mixing_lab/`), not from this directory.

```bash
# Shared targets (run once)
uv run python scripts/generate_shared_targets.py --n 600 --output output/shared_targets.json

# Per-engine databases (can run in parallel)
uv run python scripts/generate_policy_data.py --engine spectral      --shared-targets-file output/shared_targets.json --output output/db_spectral
uv run python scripts/generate_policy_data.py --engine mixbox        --shared-targets-file output/shared_targets.json --output output/db_mixbox
uv run python scripts/generate_policy_data.py --engine kubelka_munk  --shared-targets-file output/shared_targets.json --output output/db_km
uv run python scripts/generate_policy_data.py --engine coloraide_ryb --shared-targets-file output/shared_targets.json --output output/db_ryb

# Set-operation databases (after engine DBs exist)
uv run python scripts/generate_policy_data.py --set-op intersection --engine-a spectral --engine-b kubelka_munk --output output/db_intersection
uv run python scripts/generate_policy_data.py --set-op difference   --engine-a spectral --engine-b kubelka_munk --output output/db_difference
uv run python scripts/generate_policy_data.py --set-op complement   --engine-a spectral --engine-b kubelka_munk --output output/db_complement

# Multi-fidelity databases at different round counts (for Mode B fidelity)
uv run python scripts/generate_policy_data.py --engine spectral --rounds 3  --experiments 5 --output output/db_spectral_r3
uv run python scripts/generate_policy_data.py --engine spectral --rounds 12 --experiments 5 --output output/db_spectral_r12
```

Databases live in `color_mixing_lab/output/` (three levels up from this directory).
`load_study_scores` resolves paths relative to the repo root automatically.

---

## Key Implementation Notes

**`best_color_distance` vs `color_distance`**:
- `round['best_color_distance']` in round JSON = minimum over all 50 trials (correct metric)
- `trial['color_distance']` = per-trial distance including exploration attempts (includes bad guesses)
- Always use `best_color_distance_mean` from `summary.json` for policy ranking

**N = targets sampled, not evaluations per target**:
- In experiment 12, N is the number of targets sampled from the pool
- Each target has exactly one `best_color_distance` per experiment
- N > available targets uses bootstrap with replacement (tau plateaus at physics ceiling)

**Partial target overlap (set-op sources)**:
- Set-op databases cover subsets of targets; `_build_score_cube` uses union of targets
- Missing entries are handled as `None`; coverage % is reported per source
- Swarm local weights degrade gracefully when a source has no data near a query target

**Swarm advantage with current data**:
- Sources mixbox/km/ryb are global generalists (weight std = 0.03-0.05 across targets)
- Swarm advantage is small (~0.01 tau) because there are no regional specialists
- Set-op databases (intersection/difference) are spatially specialized by construction
- Including set-op sources in the pool should produce meaningful swarm advantage

**MFBO/SMAC warm-up**:
- Both policies require `min_obs_for_gp` (default 5) or `min_obs_for_fit` (default 5) observations
- Before threshold: round-robin at lowest fidelity
- After threshold: surrogate-driven acquisition

**Hyperband eta**:
- Default eta=3: keep top 1/3 at each fidelity level
- With 2 sources × 3 policies = 6 candidates and 2 fidelity levels: 6 evals → keep 2 → 2 evals at high fidelity
