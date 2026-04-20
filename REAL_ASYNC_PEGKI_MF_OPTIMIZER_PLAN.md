# Real Async PEGKi MF Optimizer Plan

## Goal

Build a PEGKi-aligned multi-fidelity optimizer that works fully offline at first by
using runtime as a proxy for process duration, while keeping a clean path open to
real asynchronous execution later.

The design should support both:

- ensemble-style allocation
- swarm-style allocation

The key constraint is architectural:

- the optimizer core must be shared between offline simulated async and future real async
- upgrading to real async should be mostly a backend swap, not a rewrite

## High-Level Strategy

Treat this as a two-layer system:

1. offline async simulator now
2. real async executor later

Both layers should share:

- the same optimizer state
- the same event model
- the same promotion / resume semantics
- the same PEGKi update logic

## Phase 0: Reuse Existing Analysis Foundations

Anchor the optimizer in the existing analysis stack rather than reimplementing the
current project logic.

Relevant existing components:

- `analysis/flip_data.py`
- `experiments/07_frugal_twin_convergence.py`
- `experiments/08_multifidelity_allocation.py`
- `experiments/09_mixed_source_flip.py`
- `experiments/12_ensemble_vs_swarm.py`

These already provide:

- study loading
- tau-based comparison against HF
- MFMC-style cost allocation ideas
- mixed-source flip analysis
- ensemble vs local-aware swarm comparison

The optimizer should call into these ideas, not fork the framework into a separate vocabulary.

## Phase 1: Create a Dedicated Optimizer Package

Add a new package, for example `optimizer/`, with the following modules:

- `optimizer/types.py`
- `optimizer/objective.py`
- `optimizer/store.py`
- `optimizer/executor.py`
- `optimizer/scheduler.py`
- `optimizer/policies.py`
- `optimizer/pegki_bridge.py`
- `optimizer/runtime_model.py`

Keep this separate from `analysis/` so the optimizer can evolve independently from
the experiment scripts.

## Phase 2: Define Core Data Types

Create explicit types for the async MF system.

Suggested objects:

- `Config`
- `FidelityLevel`
- `Source`
- `WorkerState`
- `Suggestion`
- `Observation`
- `CompletedEval`
- `IntermediateState`
- `BeliefState`

These should carry:

- source identity
- policy identity
- target or context identity
- fidelity
- score
- runtime estimate
- resume eligibility
- resume token / intermediate state id
- uncertainty metadata

## Phase 3: Build the Offline Async Simulation Contract

The first executor should be a pure offline simulator.

Each evaluation should return:

- `score`
- `runtime_estimate`
- `source`
- `policy`
- `fidelity`
- `can_resume`
- `resume_token` or intermediate-state id

Important rule:

- do not sleep
- do not launch background processes yet
- advance a simulated clock instead

The simulated clock should be driven by returned runtime estimates, similar in spirit
to compressed-runtime async wrappers from the literature, but implemented entirely
offline and deterministically.

## Phase 4: Make Fidelity Explicit

Support a discrete fidelity ladder from the start.

Natural first fidelity for this project:

- number of experiments / robot measurements

Later fidelity axes can include:

- target count
- round budget
- trials per target
- epochs

Each higher-fidelity request should optionally resume from a lower-fidelity state
instead of restarting from scratch.

## Phase 5: Implement a Backend-Agnostic Scheduler

The scheduler should not know whether execution is simulated or real.

Core scheduler methods:

- `ask(worker_id) -> Suggestion`
- `start(worker_id, suggestion)`
- `complete(worker_id, result)`
- `promote(resume_token, new_fidelity)`
- `step_until_next_completion()`
- `run_until_budget()`

Offline behavior:

- maintain a priority queue ordered by simulated completion time
- when a worker is free, assign the next suggestion
- pop the next completion event and update the optimizer state

Future real async behavior:

- keep the same scheduler API
- replace only the executor backend

## Phase 6: Implement the Objective Layer

Create an objective adapter that can work in two modes:

- offline replay mode over PEGKi databases
- future live execution mode against real workers

The offline objective should use existing PEGKi data and return:

- a score
- a runtime proxy
- optional resumable state

This keeps the optimizer grounded in current data while preserving a path to real jobs later.

## Phase 7: Add Runtime Models

Runtime is the core approximation for offline async.

Start with a configurable runtime model:

- fixed runtime per source
- fixed runtime per fidelity
- additive or multiplicative source x fidelity cost

Then support empirical runtime models:

- mean runtime by source
- mean runtime by source and fidelity
- sampled runtime distributions if variance matters

The runtime model should be pluggable so the simulator can test:

- count-based allocation
- cost-aware allocation
- runtime-aware async scheduling

## Phase 8: Add Three Initial Optimization Policies

Implement the policy layer in increasing complexity:

### 1. `single_source_mf`

One LF source plus one HF source.

Purpose:

- baseline multi-fidelity optimizer
- simplest testbed for promotions and runtime-aware scheduling

### 2. `ensemble_mf`

Global source weights shared across all targets / contexts.

Purpose:

- fixed or slowly updated source weighting
- direct extension of current mixture and MFMC ideas

### 3. `swarm_mf`

Local routing based on target-space or context-space competence.

Purpose:

- extend current local-awareness logic from `experiments/12_ensemble_vs_swarm.py`
- choose source and fidelity based on local information value

## Phase 9: Keep Ensemble and Swarm Distinct

Do not collapse ensemble and swarm into one generic averaging routine.

Ensemble decisions:

- how much budget to allocate per source
- how much budget to allocate per fidelity
- one global source-weight vector

Swarm decisions:

- which source to query for this specific target or context
- which fidelity to use for that source
- whether to exploit a strong local specialist or explore a weaker uncertain source

This distinction should exist in the optimizer itself, not only in post hoc reports.

## Phase 10: Add PEGKi Belief Updates

PEGKi should become the belief/update layer of the optimizer.

Maintain beliefs over:

- source quality
- fidelity usefulness
- transferability to HF
- uncertainty
- bias floor
- donor/receiver potential

Initial implementation:

- empirical updates from observed `tau`, `rho`, and variance

Later implementation:

- PEGKi / TrueSkill-style updates for `(study, fidelity, allocation mode)` combinations

This aligns with the existing technical reference direction for swarm/team modeling.

## Phase 11: Define Acquisition Functions

Each candidate job should be scored by expected information value relative to cost.

Good first acquisition score:

`value = expected_delta_tau / expected_runtime`

Then extend to:

`value = (expected_delta_tau + lambda * flip_probability + gamma * uncertainty_reduction) / expected_runtime`

Possible ingredients:

- expected ranking gain vs HF
- expected uncertainty reduction
- probability of donor flip
- local competence in swarm mode
- promotion benefit over restart

This keeps runtime central now and naturally carries over to real async later.

## Phase 12: Support Resumable Promotion

Promotion must be treated as a first-class operation.

A promotion is:

- same source
- same policy / target / context
- higher fidelity
- resumed from prior state when valid

It is not merely a new independent evaluation.

The state store must record enough metadata to validate:

- whether a prior state exists
- whether the requested higher fidelity is compatible
- whether resume is cheaper than restart

This seam is critical for future real async execution.

## Phase 13: Add Persistent State Storage

Persist optimizer state from day one.

Suggested stored objects:

- workers
- queued jobs
- in-flight jobs
- completed observations
- intermediate states
- optimizer beliefs
- simulated clock
- random seed metadata

Store these in a run directory using JSON files.

Benefits:

- deterministic offline replay
- interruption recovery
- transparent debugging
- straightforward path to file-based real async coordination later

## Phase 14: Offline Benchmarking Before Real Async

Benchmark the optimizer entirely in offline replay mode before touching live execution.

Recommended comparisons:

- egalitarian MF baseline
- quality-aware ensemble MF
- local-aware swarm MF
- promotion enabled vs disabled
- resume vs restart
- runtime-aware scheduling vs count-only scheduling

Primary evaluation metrics:

- best achieved HF Kendall tau over simulated time
- tau at fixed total budget
- tau at fixed simulated runtime
- donor-flip time `N*`
- regret vs oracle allocation
- source / fidelity allocation trace

## Detailed Implementation Expansion

Add a concrete implementation spec for the core async optimizer modules so the
plan can be translated into code with minimal redesign.

### 1. `optimizer/types.py`

Define the core dataclasses first. These should be stable enough that the rest of
the optimizer can build around them.

Suggested dataclasses:

- `Config`
- `FidelityLevel`
- `SourceSpec`
- `TargetContext`
- `WorkerState`
- `Suggestion`
- `CompletedEval`
- `IntermediateState`
- `BeliefState`
- `RunState`

#### `Config`

Suggested fields:

- `run_name: str`
- `seed: int`
- `hifi_source: str`
- `lf_sources: list[str]`
- `policy_mode: str`
- `executor_mode: str`
- `runtime_mode: str`
- `max_workers: int`
- `budget_total: float`
- `budget_units: str`
- `fidelity_axis: str`
- `fidelity_levels: list[int | float]`
- `n_bootstrap: int`
- `allow_resume: bool`
- `save_dir: str`

#### `FidelityLevel`

Suggested fields:

- `name: str`
- `axis: str`
- `value: int | float`
- `cost_estimate: float`
- `runtime_estimate: float`
- `resume_from: str | None`

#### `SourceSpec`

Suggested fields:

- `name: str`
- `kind: str`
- `base_cost: float`
- `base_runtime: float`
- `supports_resume: bool`
- `supports_local_routing: bool`
- `metadata: dict`

#### `TargetContext`

Suggested fields:

- `target_id: str`
- `target_rgb: tuple[float, float, float] | None`
- `domain: str`
- `features: dict`

#### `WorkerState`

Suggested fields:

- `worker_id: str`
- `status: str`
- `current_job_id: str | None`
- `busy_until: float | None`
- `completed_jobs: int`
- `metadata: dict`

Expected statuses:

- `idle`
- `running`
- `paused`
- `failed`

#### `Suggestion`

Suggested fields:

- `job_id: str`
- `source: str`
- `policy_name: str`
- `target_id: str | None`
- `fidelity: int | float`
- `priority: float`
- `expected_value: float`
- `expected_runtime: float`
- `resume_token: str | None`
- `reason: str`

#### `CompletedEval`

Suggested fields:

- `job_id: str`
- `source: str`
- `policy_name: str`
- `target_id: str | None`
- `fidelity: int | float`
- `score: float`
- `runtime_observed: float`
- `runtime_simulated_end: float`
- `resume_token_out: str | None`
- `can_resume: bool`
- `metadata: dict`

#### `IntermediateState`

Suggested fields:

- `resume_token: str`
- `source: str`
- `policy_name: str`
- `target_id: str | None`
- `fidelity_reached: int | float`
- `state_path: str | None`
- `estimated_resume_gain: float | None`
- `created_at_time: float`

#### `BeliefState`

Suggested fields:

- `source_name: str`
- `fidelity: int | float | None`
- `tau_mean: float | None`
- `tau_std: float | None`
- `rho_mean: float | None`
- `bias_floor: float | None`
- `flip_probability: float | None`
- `quality_score: float | None`
- `last_updated_time: float`
- `metadata: dict`

#### `RunState`

Suggested fields:

- `sim_time: float`
- `config: dict`
- `workers: list[WorkerState]`
- `pending_jobs: list[Suggestion]`
- `running_jobs: list[Suggestion]`
- `completed_jobs: list[CompletedEval]`
- `intermediate_states: list[IntermediateState]`
- `beliefs: list[BeliefState]`
- `metrics: dict`

### 2. `optimizer/executor.py`

The executor should be responsible only for job execution semantics.

Suggested classes:

- `BaseExecutor`
- `LocalSimExecutor`
- later `ThreadExecutor`
- later `ProcessExecutor`

#### `BaseExecutor` interface

Suggested methods:

- `submit(suggestion: Suggestion) -> None`
- `poll_completed(current_time: float) -> list[CompletedEval]`
- `cancel(job_id: str) -> None`
- `resume(resume_token: str, new_suggestion: Suggestion) -> None`
- `shutdown() -> None`

#### `LocalSimExecutor`

Responsibilities:

- compute simulated end time from runtime model
- store in-flight jobs in a min-heap keyed by completion time
- return completed jobs when scheduler advances simulated time

Implementation note:

- this executor should never sleep or spawn real workers
- it should only transform runtime estimates into completion events

### 3. `optimizer/scheduler.py`

The scheduler should own the event loop and decision flow.

Suggested class:

- `AsyncMFScheduler`

Suggested methods:

- `initialize()`
- `ask(worker_id: str) -> Suggestion | None`
- `dispatch_idle_workers()`
- `advance_to_next_event()`
- `ingest_completed(results: list[CompletedEval])`
- `update_beliefs(results: list[CompletedEval])`
- `schedule_promotions()`
- `run_until_budget()`
- `snapshot()`

Suggested event-loop order:

1. find idle workers
2. generate suggestions
3. submit to executor
4. advance simulated time to next completion
5. collect finished jobs
6. update score / belief layer
7. decide on promotions and new jobs
8. persist snapshot
9. stop when budget or stopping rule is met

Suggested stopping rules:

- budget exhausted
- target tau threshold reached
- no valid jobs remain
- max simulated time reached

### 4. `optimizer/store.py`

The store should make runs resumable and inspectable.

Suggested class:

- `RunStore`

Suggested methods:

- `save_config(config: Config)`
- `save_snapshot(run_state: RunState)`
- `load_latest_snapshot() -> RunState | None`
- `append_completed(results: list[CompletedEval])`
- `save_intermediate_state(state: IntermediateState)`
- `load_intermediate_state(resume_token: str) -> IntermediateState | None`

Suggested directory layout:

- `runs/<run_name>/config.json`
- `runs/<run_name>/snapshots/state_00001.json`
- `runs/<run_name>/completed.jsonl`
- `runs/<run_name>/beliefs.json`
- `runs/<run_name>/intermediate/<resume_token>.json`
- `runs/<run_name>/artifacts/`

Design rule:

- snapshots should be append-only or versioned
- completed results should also be written as JSONL for easy audit

### 5. `optimizer/runtime_model.py`

The runtime model should be explicit and swappable.

Suggested classes:

- `BaseRuntimeModel`
- `ConstantRuntimeModel`
- `SourceFidelityRuntimeModel`
- `EmpiricalRuntimeModel`

Suggested method:

- `estimate(source: str, fidelity: int | float, context: dict | None = None) -> float`

Recommended first implementation:

- runtime scales linearly with fidelity
- source-specific multiplier
- optional random noise term for simulation realism

### 6. `optimizer/objective.py`

The objective layer should translate a suggestion into a score.

Suggested classes:

- `BaseObjective`
- `OfflineReplayObjective`

Suggested methods:

- `evaluate(suggestion: Suggestion) -> CompletedEval`
- `resume(intermediate: IntermediateState, suggestion: Suggestion) -> CompletedEval`
- `supports_resume(source: str) -> bool`

Offline replay behavior:

- pull data from PEGKi study databases
- sample according to source / target / fidelity
- compute score and any resumable output

### 7. `optimizer/policies.py`

The policy layer should own job proposal logic.

Suggested classes:

- `BasePolicy`
- `SingleSourceMFPolicy`
- `EnsembleMFPolicy`
- `SwarmMFPolicy`

Suggested method:

- `propose(run_state: RunState, worker_id: str) -> Suggestion | None`

#### `SingleSourceMFPolicy`

Should optimize:

- HF vs one LF source
- promotion timing
- runtime-aware acquisition

#### `EnsembleMFPolicy`

Should optimize:

- global source weights
- fidelity allocation per source
- group value vs best-member value

#### `SwarmMFPolicy`

Should optimize:

- local source routing
- local fidelity routing
- explore vs exploit balance
- memory-aware reuse of local competence information

### 8. `optimizer/pegki_bridge.py`

This module should translate completed evaluations into PEGKi-style belief updates.

Suggested functions:

- `update_tau_rho_beliefs(...)`
- `update_bias_floor_estimates(...)`
- `update_flip_probability(...)`
- `build_team_rating_inputs(...)`

Suggested future adapters:

- `TrueSkillBridge`
- `BladeChestBridge`

Design rule:

- the optimizer should depend only on bridge outputs, not on the internals of every rating model

### 9. JSON schema for saved run state

The top-level snapshot schema should be explicit.

Suggested structure:

```json
{
  "meta": {
    "run_name": "exp13_swarm_runtime",
    "timestamp": "2026-03-31T12:00:00Z",
    "version": "0.1"
  },
  "config": {},
  "sim_time": 123.4,
  "budget_used": 87.5,
  "workers": [],
  "pending_jobs": [],
  "running_jobs": [],
  "completed_jobs": [],
  "intermediate_states": [],
  "beliefs": [],
  "metrics": {
    "best_tau": 0.84,
    "best_source": "study_b",
    "best_group_mode": "swarm_mf"
  }
}
```

Minimum `metrics` keys:

- `best_tau`
- `best_rho`
- `best_group_mode`
- `budget_used`
- `runtime_elapsed`
- `n_completed_jobs`
- `n_promotions`

### 10. Experiment entry point details

The first async optimizer experiment should have a stable CLI.

Suggested script:

- `experiments/13_async_mf_optimizer.py`

Suggested CLI arguments:

- `--hifi`
- `--lf-sources`
- `--policy-mode`
- `--executor-mode`
- `--runtime-mode`
- `--fidelity-axis`
- `--fidelity-levels`
- `--budget-total`
- `--max-workers`
- `--allow-resume`
- `--target-tau`
- `--n-bootstrap`
- `--seed`
- `--save-dir`

Required outputs:

- `results/async_mf_optimizer_summary.json`
- `results/async_mf_optimizer_trace.jsonl`
- `results/async_mf_optimizer_tau_vs_time.png`
- `results/async_mf_optimizer_allocations.png`

### 11. Validation checklist for first implementation

The first implementation should be considered acceptable when all of the following are true:

- a full run can be started from config only
- a stopped run can be resumed from saved snapshot
- the offline executor advances simulated time without sleeping
- promotions produce valid resumed jobs when enabled
- ensemble and swarm policies both run through the same scheduler
- results include tau vs simulated time
- results include budget and allocation trace
- saved outputs are sufficient to reproduce plots without rerunning the optimizer

## TODO: Add a Lightweight Distributed Coordination Layer

Make the distributed-systems aspect of the async optimizer explicit rather than leaving
it implied by the scheduler and store.

Target framing:

- lightweight distributed coordination layer
- PEGKi / TENKi score analysis on top
- file-based offline-first coordination now
- clean path to real remote or multi-process workers later

This layer should coordinate workers, jobs, resumable states, and score updates while
keeping the actual evaluation logic separate.

### Scope

This is not meant to be a large service platform.

It should instead provide a minimal distributed-systems substrate for:

- worker registration
- job leasing
- completion reporting
- heartbeat / liveness tracking
- stale-job recovery
- resume-token ownership
- auditability of score-affecting events

### Suggested modules

- `optimizer/coordination.py`
- `optimizer/leases.py`
- `optimizer/heartbeat.py`
- `optimizer/recovery.py`

These can be folded into `store.py` and `scheduler.py` at first if needed, but the
responsibilities should still be kept conceptually distinct.

### Core coordination objects

Suggested records:

- `WorkerLease`
- `JobLease`
- `HeartbeatRecord`
- `RecoveryAction`
- `CoordinationEvent`

#### `WorkerLease`

Suggested fields:

- `worker_id: str`
- `lease_id: str`
- `status: str`
- `job_id: str | None`
- `leased_at: float`
- `expires_at: float`
- `last_heartbeat: float | None`

#### `JobLease`

Suggested fields:

- `job_id: str`
- `lease_id: str`
- `worker_id: str`
- `claimed_at: float`
- `expires_at: float`
- `resume_token: str | None`
- `state: str`

Expected states:

- `queued`
- `leased`
- `running`
- `completed`
- `expired`
- `recovered`

#### `HeartbeatRecord`

Suggested fields:

- `worker_id: str`
- `timestamp: float`
- `job_id: str | None`
- `status: str`
- `progress: float | None`
- `metadata: dict`

#### `RecoveryAction`

Suggested fields:

- `job_id: str`
- `action: str`
- `reason: str`
- `old_worker_id: str | None`
- `new_worker_id: str | None`
- `timestamp: float`

#### `CoordinationEvent`

Suggested fields:

- `event_id: str`
- `event_type: str`
- `timestamp: float`
- `worker_id: str | None`
- `job_id: str | None`
- `payload: dict`

### Phase A: Worker registration and liveness

Add worker lifecycle support.

Minimum operations:

- register worker
- mark worker idle
- mark worker busy
- heartbeat worker
- detect worker timeout
- retire worker

Validation questions:

- Can the system distinguish an idle worker from a dead worker?
- Can score-affecting jobs be traced back to the worker that produced them?

### Phase B: Job leasing

Use leases rather than simple assignment so stale jobs can be recovered safely.

Minimum operations:

- claim next job
- attach job lease to worker
- renew lease on heartbeat
- expire lease if heartbeat stops
- return expired job to queue or recovery path

Design rule:

- the scheduler should never assume a leased job will finish

### Phase C: Recovery and retry policy

Add minimal fault-recovery behavior even in offline-first mode.

Recovery cases:

- worker disappears while running
- job lease expires
- result arrives after recovery already happened
- resume token becomes stale or invalid

Suggested recovery actions:

- requeue job
- restart job from scratch
- resume from latest valid intermediate state
- discard stale duplicate completion

Important score-layer rule:

- recovered or duplicated jobs must not double-count into PEGKi / TENKi scoring

### Phase D: Resume-token ownership and state consistency

Promotion and resume introduce distributed consistency concerns even offline.

The coordination layer should answer:

- Which worker currently owns a resume token?
- Has that token already been promoted?
- Is a completion result for that token still valid?
- Can two workers accidentally resume the same state?

Required safeguards:

- single-owner resume-token lease
- versioned intermediate-state records
- stale-result rejection rules

### Phase E: Event log as source of truth

Keep an append-only event log of coordination events.

Recommended event types:

- `worker_registered`
- `worker_heartbeat`
- `job_queued`
- `job_leased`
- `job_completed`
- `job_expired`
- `job_recovered`
- `resume_token_created`
- `resume_token_promoted`
- `resume_token_invalidated`

Benefits:

- full auditability
- replayable debugging
- score reproducibility
- later compatibility with remote workers

### Phase F: File-based coordination protocol

For the offline-first version, coordination can be file-backed rather than networked.

Suggested files:

- `coordination/workers.json`
- `coordination/job_queue.json`
- `coordination/job_leases.json`
- `coordination/heartbeats.jsonl`
- `coordination/events.jsonl`
- `coordination/recovery.jsonl`

Design rule:

- every state mutation should correspond to a coordination event

### Phase G: PEGKi / TENKi scoring integration

The distributed layer should not score anything itself, but it must expose clean,
trustworthy inputs to the score layer.

It must guarantee:

- each completed evaluation is counted at most once
- recovered jobs are clearly labeled
- duplicate completions are detectable
- worker provenance is preserved
- promotion lineage is preserved

This is what lets PEGKi / TENKi scoring sit safely on top of distributed execution.

### Phase H: Future real-async path

This layer should be enough to support later:

- multi-process workers on one machine
- multiple local executors
- remote workers on shared storage
- eventual service-backed coordination if needed

The first goal is not scale.

The first goal is correctness of:

- job ownership
- resume semantics
- recovery semantics
- score accounting

### Recommended outputs

- `runs/<run_name>/coordination/events.jsonl`
- `runs/<run_name>/coordination/heartbeats.jsonl`
- `runs/<run_name>/coordination/recovery.jsonl`
- `results/distributed_coordination_summary.md`

### Validation checklist

This layer is good enough for first use when:

- workers can be registered and retired cleanly
- a leased job can expire and be recovered
- stale completions do not corrupt scoring
- resume tokens cannot be consumed twice
- event logs are sufficient to replay coordination history
- PEGKi / TENKi score summaries remain stable under retry and recovery

## TODO: Add a Local Information vs Memory Value Experiment

Add a dedicated experiment to measure not just whether swarm routing helps, but how
much of that benefit comes from:

- local information
- persistent memory / state

Suggested new experiment:

- `experiments/14_local_info_vs_memory.py`

This should extend the existing ensemble-vs-swarm comparison and turn it into an
ablation study.

Recommended conditions:

- `ensemble_global`
  No local information, no memory.
- `swarm_local_stateless`
  Uses local competence at decision time, but does not cache or carry state forward.
- `swarm_local_cached`
  Uses local competence and caches neighbourhood/source competence estimates.
- `swarm_with_promotion_memory`
  Uses local competence plus resumable promotion history.
- `swarm_with_belief_memory`
  Uses local competence plus persistent PEGKi-style per-source or per-target beliefs.

Recommended evaluation questions:

- How much tau improvement comes from local routing alone?
- How much additional improvement comes from cached memory?
- Does promotion / resume memory reduce simulated runtime to reach the same tau?
- Does belief-state memory improve budget allocation compared with stateless routing?

Recommended metrics:

- Kendall tau vs HF reference
- tau vs simulated runtime
- tau vs total budget
- donor-flip time `N*`
- number of resumed promotions
- improvement over stateless local swarm

Recommended outputs:

- `results/local_info_vs_memory.json`
- `results/local_info_vs_memory.md`
- `results/local_info_vs_memory_tau.png`
- `results/local_info_vs_memory_runtime.png`

Design note:

- reuse the current local-information idea from `experiments/12_ensemble_vs_swarm.py`
- treat that as the baseline "local but stateless" swarm
- then layer memory on top as a controlled extension

## TODO: Add a Concrete Score Analysis Layer for Individuals and Groups

Add a dedicated scoring layer so the optimizer and analysis pipeline can evaluate:

- individual sources
- individual robots / experiments
- ensemble groups
- swarm groups
- mixed-source teams across fidelities

This layer should make scoring explicit rather than leaving it distributed across
several experiment scripts.

Suggested new package or module set:

- `analysis/score_models.py`
- `analysis/group_scores.py`
- `analysis/score_reports.py`

Core design requirement:

- separate raw performance metrics from rating / structure models
- support both individual-level and group-level scoring with the same data backbone

### Scope of the score layer

The score layer should support at least four families of outputs:

1. raw performance scores
2. ranking-agreement scores
3. rating / reputation scores
4. group aggregation scores

### 1. Raw performance scores

For individuals:

- mean score per policy / study / robot
- variance and confidence interval
- per-target and per-fidelity summaries

For groups:

- mean aggregated score for an ensemble or swarm
- per-source contribution statistics
- per-fidelity contribution statistics

These should remain close to the existing `StudyScores` and score-cube structures.

### 2. Ranking-agreement scores

For individuals:

- Kendall tau vs HF reference
- Spearman rho vs HF reference
- donor score / asymmetry score
- external ceiling / bias floor

For groups:

- tau and rho for ensemble rankings
- tau and rho for swarm rankings
- donor-flip time `N*` for a group vs a competitor
- stability of group ranking over bootstrap or simulated time

This should unify the logic currently split across experiments 07, 08, 11, and 12.

### 3. Rating / reputation scores

Add a TODO path for parent-PEGKi style scoring models beyond simple tau/rho.

Individual-level models to support:

- TrueSkill / TrueSkill2
- Blade-Chest
- FFG / Hodge decomposition outputs
- ICBT
- ICBT+BC
- BNN-derived rating or descriptor-aware score
- Spinning Tops / cyclic-subgame support diagnostics

Group-level models to support:

- team-level TrueSkill2 performance
- team-level Blade-Chest matchup analysis
- intransitivity diagnostics for ensemble vs swarm vs mixed-source teams
- partial-play / weighted team skill estimates

Important implementation note:

- this project extension does not yet need to implement every parent PEGKi model immediately
- but the score layer should reserve interfaces for them now

### 4. Group aggregation scores

Define explicit group score objects for:

- `EnsembleScore`
- `SwarmScore`
- `TeamScore`

Each should report:

- aggregate score
- aggregate rank
- uncertainty
- source weights
- fidelity weights
- local-routing statistics
- contribution breakdown by member

### Suggested APIs

Potential functions:

- `score_individual_raw(...)`
- `score_individual_rank(...)`
- `score_group_raw(...)`
- `score_group_rank(...)`
- `score_team_trueskill(...)`
- `score_team_blade_chest(...)`
- `compute_donor_score(...)`
- `compute_bias_floor(...)`
- `compute_group_flip_n(...)`
- `compare_group_vs_individual(...)`

Potential result objects:

- `IndividualScoreResult`
- `GroupScoreResult`
- `RatingModelResult`
- `ScoreComparisonResult`

### Concrete implementation plan

#### Phase A: Build a shared score data backbone

Create canonical score records that can represent both individuals and groups.

Suggested internal records:

- `IndividualObservation`
- `GroupObservation`
- `ScoreSlice`
- `ContributionSlice`

Each record should support:

- source name
- policy name
- target or context id
- fidelity
- raw score
- runtime or cost
- bootstrap replicate id
- group id if part of an ensemble or swarm
- source weight and fidelity weight if aggregated

Design rule:

- one normalized schema should feed both simple tau/rho analyses and future TS2/BC hooks

#### Phase B: Implement individual score summaries

Build the individual layer first because every group score depends on it.

Minimum outputs per individual source / robot:

- raw mean score
- raw median score
- standard deviation
- confidence interval
- per-target breakdown
- per-fidelity breakdown
- tau vs HF
- rho vs HF
- donor score
- external ceiling
- bias floor

Recommended helper functions:

- `summarize_individual_scores(...)`
- `summarize_individual_targets(...)`
- `summarize_individual_fidelities(...)`
- `rank_individuals_vs_reference(...)`

#### Phase C: Implement group score summaries

Treat a group as a first-class scored object rather than a temporary average.

Supported group types:

- equal-weight ensemble
- weighted ensemble
- local swarm
- cached-memory swarm
- mixed-source team across fidelities

Minimum outputs per group:

- aggregate raw score
- aggregate rank
- tau vs HF
- rho vs HF
- group donor score
- group flip `N*`
- group uncertainty
- mean source weights
- fidelity allocation breakdown
- best-member comparison

Recommended helper functions:

- `summarize_group_scores(...)`
- `rank_group_vs_reference(...)`
- `compute_group_uncertainty(...)`
- `compute_group_vs_best_member_gap(...)`

#### Phase D: Add contribution analysis

A major goal of the score layer is to explain why a group performed the way it did.

Per-group contribution outputs should include:

- per-source contribution to final score
- per-source contribution to final rank
- per-fidelity contribution
- local-routing frequency
- effective weight after missing-data normalization
- contribution volatility across bootstrap runs

Recommended helper functions:

- `compute_source_contributions(...)`
- `compute_fidelity_contributions(...)`
- `compute_routing_statistics(...)`
- `compute_contribution_stability(...)`

#### Phase E: Add rating / reputation model hooks

Do not implement every model at once, but define stable interfaces now.

Suggested adapter pattern:

- `BaseScoreModel`
- `TauRhoModel`
- `TrueSkillAdapter`
- `BladeChestAdapter`
- `HodgeAdapter`
- `ICBTAdapter`
- `BNNAdapter`

Each adapter should accept the same normalized comparison data and return:

- model name
- fitted parameters or summary statistics
- individual-level outputs
- group-level outputs
- uncertainty or fit diagnostics

Design rule:

- the score layer owns the interface
- parent-PEGKi model specifics stay inside adapters

#### Phase F: Add comparison reports

The score layer should answer comparison questions directly, not just dump tables.

Required comparison views:

- individual vs individual
- group vs group
- group vs best individual
- ensemble vs swarm
- equal-weight vs quality-aware
- local-stateless vs local-with-memory
- low-fidelity group vs HF baseline

Recommended report functions:

- `build_individual_score_table(...)`
- `build_group_score_table(...)`
- `build_group_vs_individual_report(...)`
- `build_score_ablation_report(...)`

#### Phase G: Add experiment entry points

Create a dedicated experiment runner so the score layer is directly testable.

Suggested scripts:

- `experiments/15_score_layer_baseline.py`
- `experiments/16_group_vs_individual_scores.py`
- `experiments/17_rating_model_hooks.py`

Recommended CLI controls:

- `--hifi`
- `--sources`
- `--fidelity-mode`
- `--group-mode`
- `--score-model`
- `--n-bootstrap`
- `--seed`

#### Phase H: Validation questions

The score layer should be considered complete enough for first use when it can answer:

- Which individual source is strongest by raw score?
- Which individual source is strongest by tau/rho agreement?
- Which source is the best donor?
- Is the best group better than the best individual?
- When does a swarm outperform an ensemble?
- When does weighting outperform equal allocation?
- When does local routing change the winner?
- When does memory change the winner?

#### Phase I: JSON output schema expectations

At minimum, saved JSON should include:

- metadata
- analysis configuration
- individual summaries
- group summaries
- comparison summaries
- contribution summaries
- optional model-specific sections

Suggested top-level keys:

- `meta`
- `config`
- `individual_scores`
- `group_scores`
- `comparisons`
- `contributions`
- `model_outputs`

### Recommended first milestone

Implement a minimal version that covers:

- individual raw mean / variance
- individual tau / rho / donor score
- ensemble group tau / rho
- swarm group tau / rho
- group vs individual comparison tables

This should be enough to answer:

- when is a group better than its best member?
- when is a swarm better than an ensemble?
- when does weighting help more than quantity?

### Recommended second milestone

Add explicit parent-PEGKi score-model hooks for:

- TrueSkill2
- Blade-Chest
- Hodge / cyclic diagnostics
- clustered context models

### Recommended outputs

- `results/score_layer_individuals.json`
- `results/score_layer_groups.json`
- `results/score_layer_comparison.md`
- `results/score_layer_group_vs_individual.png`

### Design note

This score layer should become the shared analysis backend for future:

- async optimizer evaluation
- local-info vs memory ablations
- heterogeneous swarm weighting studies
- team-level PEGKi / TENKi reputation tracking

## Phase 15: Keep the Real Async Seam Narrow

Define an executor interface with multiple backends:

- `LocalSimExecutor`
- `ThreadExecutor`
- `ProcessExecutor`
- later `RemoteExecutor`

Only the executor should know whether time is:

- simulated
- real

The scheduler, PEGKi updater, and policy logic should remain backend-agnostic.

## Suggested File Layout

- `optimizer/types.py`
- `optimizer/store.py`
- `optimizer/objective.py`
- `optimizer/executor.py`
- `optimizer/scheduler.py`
- `optimizer/policies.py`
- `optimizer/pegki_bridge.py`
- `optimizer/runtime_model.py`
- `experiments/13_async_mf_optimizer.py`
- `results/async_mf_optimizer_*`

## Milestone 1

Get a minimal offline async optimizer working with:

- HF source: `spectral`
- LF sources: `study_b`, `mixbox`
- fidelity: number of experiments
- runtime proxy: configurable source/fidelity cost model
- optimizer modes: `ensemble_mf` and `swarm_mf`

Outputs:

- tau vs simulated time
- allocation trace
- promotion trace
- source usage summary

## Milestone 2

Generalize to all studies and add:

- quality-aware weighting
- local swarm routing
- resumable promotion
- PEGKi belief updates

## Milestone 3

Swap `LocalSimExecutor` for a real async backend without changing:

- scheduler logic
- PEGKi update logic
- acquisition functions

At that point, real async becomes an execution problem, not a framework redesign.

## Main Risks

- mixing offline replay logic with live execution too early
- letting ensemble/swarm logic leak into scheduler or storage layers
- using a single global `rho` where local competence matters
- modeling promotion as restart
- optimizing only by sample count rather than simulated runtime

## Recommendation

Build the offline simulator first and treat it as the reference backend.

If the offline backend is architecturally clean, the real async version becomes a
backend swap plus worker coordination, rather than a second research project.
