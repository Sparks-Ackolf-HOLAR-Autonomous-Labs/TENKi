"""
optimizer/types.py -- Core dataclasses for the async MF optimizer.

All data structures are intentionally simple (no methods) so that:
- they serialise cleanly to JSON via dataclasses.asdict()
- the rest of the optimizer can build around stable, explicit contracts
- upgrading to real async requires only executor changes, not type changes
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Config:
    run_name: str
    seed: int = 42
    hifi_source: str = "spectral"
    lf_sources: list[str] = field(default_factory=list)
    policy_mode: str = "ensemble_mf"        # single_source_mf | ensemble_mf | swarm_mf
    executor_mode: str = "local_sim"        # local_sim | thread | process
    runtime_mode: str = "source_fidelity"   # constant | source_fidelity | empirical
    max_workers: int = 4
    budget_total: float = 1000.0
    budget_units: str = "runtime"           # runtime | count | cost
    fidelity_axis: str = "n_experiments"
    fidelity_levels: list[int | float] = field(default_factory=lambda: [1, 5, 10, 50])
    fidelity_db_map: dict = field(default_factory=dict)
    # {source_name: {str(fidelity_value): db_path}}
    # When non-empty: each fidelity level maps to a pre-generated PEGKi database path
    # When empty: legacy mode — fidelity = number of experiment scores to sample
    n_bootstrap: int = 100
    allow_resume: bool = True
    save_dir: str = "runs"


@dataclass
class FidelityLevel:
    name: str
    axis: str
    value: int | float
    cost_estimate: float = 1.0
    runtime_estimate: float = 1.0
    resume_from: str | None = None


@dataclass
class SourceSpec:
    name: str
    kind: str = "lf"                        # lf | hf
    base_cost: float = 1.0
    base_runtime: float = 1.0
    supports_resume: bool = True
    supports_local_routing: bool = False
    metadata: dict = field(default_factory=dict)


@dataclass
class TargetContext:
    target_id: str
    target_rgb: tuple[float, float, float] | None = None
    domain: str = "color_mixing"
    features: dict = field(default_factory=dict)


@dataclass
class WorkerState:
    worker_id: str
    status: str = "idle"                    # idle | running | paused | failed
    current_job_id: str | None = None
    busy_until: float | None = None
    completed_jobs: int = 0
    metadata: dict = field(default_factory=dict)


@dataclass
class Suggestion:
    job_id: str
    source: str
    policy_name: str
    target_id: str | None = None
    fidelity: int | float = 1
    priority: float = 0.0
    expected_value: float = 0.0
    expected_runtime: float = 1.0
    resume_token: str | None = None
    reason: str = ""


@dataclass
class CompletedEval:
    job_id: str
    source: str
    policy_name: str
    target_id: str | None
    fidelity: int | float
    score: float
    runtime_observed: float
    runtime_simulated_end: float
    resume_token_out: str | None = None
    can_resume: bool = False
    metadata: dict = field(default_factory=dict)


@dataclass
class IntermediateState:
    resume_token: str
    source: str
    policy_name: str
    target_id: str | None
    fidelity_reached: int | float
    state_path: str | None = None
    estimated_resume_gain: float | None = None
    created_at_time: float = 0.0
    # Offline replay: sampled values + seed for deterministic resume
    sampled_scores: list[float] = field(default_factory=list)
    sampled_seed: int = 0


@dataclass
class BeliefState:
    source_name: str
    fidelity: int | float | None = None
    tau_mean: float | None = None
    tau_std: float | None = None
    rho_mean: float | None = None
    bias_floor: float | None = None
    flip_probability: float | None = None
    donor_score: float | None = None
    effective_fidelity: float | None = None
    quality_score: float | None = None
    last_updated_time: float = 0.0
    n_observations: int = 0
    metadata: dict = field(default_factory=dict)


@dataclass
class RunState:
    sim_time: float = 0.0
    config: dict = field(default_factory=dict)
    workers: list[WorkerState] = field(default_factory=list)
    pending_jobs: list[Suggestion] = field(default_factory=list)
    running_jobs: list[Suggestion] = field(default_factory=list)
    completed_jobs: list[CompletedEval] = field(default_factory=list)
    intermediate_states: list[IntermediateState] = field(default_factory=list)
    beliefs: list[BeliefState] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
