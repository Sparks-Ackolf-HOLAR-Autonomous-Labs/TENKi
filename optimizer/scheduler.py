"""
optimizer/scheduler.py -- Async MF event loop and worker dispatch.

The scheduler owns the event loop and decision flow.
It does not know whether execution is simulated or real --
that distinction lives entirely in the executor backend.

Event loop order (run_until_budget):
1. find idle workers
2. ask policy for suggestions
3. submit suggestions to executor
4. advance simulated time to next completion
5. collect finished jobs
6. update beliefs (tau, bias floor)
7. decide on promotions (stub in Milestone 1)
8. persist snapshot
9. stop when budget or stopping condition is met
"""

from __future__ import annotations

import numpy as np

from .types import Config, RunState, WorkerState, CompletedEval
from .executor import BaseExecutor
from .objective import BaseObjective
from .store import RunStore
from .pegki_bridge import (
    update_tau_rho_beliefs,
    update_bias_floor_estimates,
    update_flip_probability,
)
from .policies import BasePolicy


class AsyncMFScheduler:
    def __init__(
        self,
        config: Config,
        run_state: RunState,
        executor: BaseExecutor,
        objective: BaseObjective,
        policy: BasePolicy,
        store: RunStore,
        hf_rank: list[str],
        lower_is_better: bool = True,
    ):
        self.config = config
        self.state = run_state
        self.executor = executor
        self.objective = objective
        self.policy = policy
        self.store = store
        self.hf_rank = hf_rank
        self.lower_is_better = lower_is_better
        self._all_sources = [config.hifi_source] + list(config.lf_sources)

    # ------------------------------------------------------------------
    # Initialisation

    def initialize(self) -> None:
        if not self.state.workers:
            for i in range(self.config.max_workers):
                self.state.workers.append(WorkerState(worker_id=f"worker_{i:02d}"))
        self.state.metrics.setdefault("best_tau", 0.0)
        self.state.metrics.setdefault("best_quality", 0.0)
        self.state.metrics.setdefault("budget_used", 0.0)
        self.state.metrics.setdefault("n_completed_jobs", 0)
        self.state.metrics.setdefault("n_promotions", 0)
        self.state.metrics.setdefault("runtime_elapsed", 0.0)
        self.state.config = {
            "run_name": self.config.run_name,
            "policy_mode": self.config.policy_mode,
            "hifi_source": self.config.hifi_source,
            "lf_sources": self.config.lf_sources,
        }
        self.store.save_config(self.config)

    # ------------------------------------------------------------------
    # Step methods

    def dispatch_idle_workers(self) -> int:
        dispatched = 0
        for worker in self.state.workers:
            if worker.status != "idle":
                continue
            suggestion = self.policy.propose(self.state, worker.worker_id)
            if suggestion is None:
                continue
            self.executor.submit(suggestion, self.state.sim_time)
            worker.status = "running"
            worker.current_job_id = suggestion.job_id
            worker.busy_until = self.state.sim_time + suggestion.expected_runtime
            self.state.running_jobs.append(suggestion)
            self.store.log_event({
                "event_type": "job_leased",
                "timestamp": self.state.sim_time,
                "worker_id": worker.worker_id,
                "job_id": suggestion.job_id,
                "source": suggestion.source,
                "policy": suggestion.policy_name,
                "fidelity": suggestion.fidelity,
            })
            dispatched += 1
        return dispatched

    def advance_to_next_event(self) -> float:
        next_t = self.executor.next_completion_time()
        if next_t is not None and next_t > self.state.sim_time:
            self.state.sim_time = next_t
        self.state.metrics["runtime_elapsed"] = self.state.sim_time
        return self.state.sim_time

    def ingest_completed(self, results: list[CompletedEval]) -> None:
        for r in results:
            self.state.completed_jobs.append(r)
            self.state.metrics["n_completed_jobs"] += 1
            self.state.metrics["budget_used"] += r.runtime_observed
            for w in self.state.workers:
                if w.current_job_id == r.job_id:
                    w.status = "idle"
                    w.current_job_id = None
                    w.busy_until = None
                    w.completed_jobs += 1
            self.state.running_jobs = [
                j for j in self.state.running_jobs if j.job_id != r.job_id
            ]
            self.store.log_event({
                "event_type": "job_completed",
                "timestamp": self.state.sim_time,
                "job_id": r.job_id,
                "source": r.source,
                "policy": r.policy_name,
                "fidelity": r.fidelity,
                "score": r.score,
            })
        if results:
            self.store.append_completed(results)

    def update_beliefs(self, results: list[CompletedEval]) -> None:
        if not results:
            return
        self.state.beliefs = update_tau_rho_beliefs(
            self.state.beliefs,
            self.state.completed_jobs,
            self.hf_rank,
            self.config.hifi_source,
            self._all_sources,
            self.config.fidelity_levels,
            self.lower_is_better,
            self.state.sim_time,
        )
        self.state.beliefs = update_bias_floor_estimates(
            self.state.beliefs,
            self.state.completed_jobs,
            self.hf_rank,
            self._all_sources,
            self.lower_is_better,
            self.state.sim_time,
        )
        self.state.beliefs = update_flip_probability(
            self.state.beliefs,
            self.state.completed_jobs,
            self.hf_rank,
            self._all_sources,
            self.lower_is_better,
            current_time=self.state.sim_time,
        )
        best_tau = max(
            (b.tau_mean for b in self.state.beliefs if b.tau_mean is not None),
            default=0.0,
        )
        best_quality = max(
            (b.quality_score for b in self.state.beliefs if b.quality_score is not None),
            default=0.0,
        )
        self.state.metrics["best_tau"] = best_tau
        self.state.metrics["best_quality"] = best_quality
        self.store.save_beliefs(self.state.beliefs)

    def schedule_promotions(self) -> None:
        """
        Promote low-fidelity results to higher fidelity when beneficial.
        Stub in Milestone 1; full promotion logic in Milestone 2.
        """
        pass

    def snapshot(self) -> None:
        self.store.save_snapshot(self.state)

    # ------------------------------------------------------------------
    # Stopping conditions

    def _within_budget(self) -> bool:
        if self.config.budget_units == "runtime":
            return self.state.sim_time < self.config.budget_total
        return self.state.metrics.get("budget_used", 0.0) < self.config.budget_total

    def _has_work(self) -> bool:
        if self.executor.n_inflight() > 0:
            return True
        test = self.policy.propose(self.state, "__probe__")
        return test is not None

    # ------------------------------------------------------------------
    # Main loop

    def run_until_budget(self, snapshot_interval: int = 10) -> None:
        self.initialize()
        step = 0
        while self._within_budget():
            n_dispatched = self.dispatch_idle_workers()
            if n_dispatched == 0 and self.executor.n_inflight() == 0:
                break  # No more work available
            self.advance_to_next_event()
            results = self.executor.poll_completed(self.state.sim_time)
            self.ingest_completed(results)
            self.update_beliefs(results)
            self.schedule_promotions()
            step += 1
            if step % snapshot_interval == 0:
                self.snapshot()
        self.snapshot()
